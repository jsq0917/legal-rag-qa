"""编排入口：一次查询的完整决策（意图 → 路由 → 结构化判定）。

数据流::

    query
      → recognize_intent()                     # LLM JSON 模式，失败降级规则
      → route(intent, structured_hit=False)    # 静态路由（§6.3 矩阵）
      → [仅结构化候选] qa_retriever.lookup()    # Redis → BM25 → 余弦门
      → route(intent, structured_hit=命中?)    # 终裁：pure_structured / hybrid
      → QueryPlan                              # 交给 Phase 5 生成链路

设计约束（§6.3）：纯结构化问题**不产生 Milvus 调用**——路由层就不置
``use_milvus``；纯法条问题**不查 qa_pairs**——``use_mysql_qa=False`` 时
直接跳过结构化链路。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from app.core.access import AccessScope
from app.core.logger import get_logger
from app.orchestration.intent import recognize_intent
from app.orchestration.models import IntentResult, RouteDecision
from app.orchestration.router import ALL_KB_IDS
from app.orchestration.qa_cache import QALookupResult, QARetriever, get_qa_retriever
from app.orchestration.router import route
from config import Settings, get_settings

logger = get_logger("lawanswer.orchestration")


@dataclass(frozen=True)
class QueryPlan:
    """一次查询的完整决策结果，生成链路（Phase 5）的唯一输入。"""

    query: str
    intent: IntentResult
    route: RouteDecision
    qa: QALookupResult
    sub_queries: list[str] = field(default_factory=list)  # 实际用于检索的查询（改写后）
    elapsed_ms: int = 0

    @property
    def degraded(self) -> bool:
        """LLM 是否降级到规则兜底（SSE meta 的 degraded 标记用）。"""
        return self.intent.source == "rules"

    def to_dict(self) -> dict:
        """调试/预览接口用（不依赖生成层类型）。"""
        return {
            "query": self.query,
            "intent": self.intent.to_dict(),
            "route": {
                "route": self.route.route,
                "kb_ids": list(self.route.kb_ids),
                "use_mysql_qa": self.route.use_mysql_qa,
                "use_milvus": self.route.use_milvus,
                "reason": self.route.reason,
            },
            "qa": {
                "hit": self.qa.hit.to_dict() if self.qa.hit else None,
                "references": [r.to_dict() for r in self.qa.references],
                "source": self.qa.source,
                "similarity": self.qa.similarity,
                "error": self.qa.error,
            },
            "sub_queries": list(self.sub_queries),
            "degraded": self.degraded,
            "elapsed_ms": self.elapsed_ms,
        }


def analyze_query(
    query: str,
    *,
    history: list[dict] | None = None,
    selected_kb_ids: list[str] | None = None,
    qa_retriever: QARetriever | None = None,
    settings: Settings | None = None,
    scope: AccessScope | None = None,
) -> QueryPlan:
    """意图识别 + 路由 + 结构化判定，返回 :class:`QueryPlan`。

    :param query: 用户当前问题
    :param history: 最近对话 ``[{"role": ..., "content": ...}]``
    :param selected_kb_ids: 前端勾选的知识库；空 = 全部勾选
    :param qa_retriever: 注入问答检索器（测试用）
    :param scope: 服务端授权范围；给定时任何路由结果都会被收敛到该范围
    """
    s = settings or get_settings()
    query = (query or "").strip()
    started = time.perf_counter()
    requested_kb_ids = list(selected_kb_ids or [])
    if scope is not None:
        # 客户端勾选只能缩小范围，不能扩大权限
        selected_kb_ids = scope.restrict(selected_kb_ids)
        if requested_kb_ids and not selected_kb_ids:
            # 明确勾选了库但全部不存在/不在授权范围：不回落全库，直接引导
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return QueryPlan(
                query=query,
                intent=IntentResult(
                    intent="other",
                    rewrite_type="direct",
                    sub_queries=[query] if query else [],
                    complexity="simple",
                    source="selection",
                ),
                route=RouteDecision(
                    route="no_data",
                    kb_ids=[],
                    reason="所选知识库不存在或不在授权范围",
                ),
                qa=QALookupResult(),
                sub_queries=[query] if query else [],
                elapsed_ms=elapsed_ms,
            )

    # 0) 缓存前置：先查 Redis 精确缓存（0.1ms 级），命中就直接走 pure_structured，
    #    跳过 LLM 意图识别（~1s）。高频问题 TTFT 从 ~2s 砍到 ~0.1s。
    qa_retriever = qa_retriever or get_qa_retriever()
    exact_hit = qa_retriever.lookup_exact(query, scope=scope)
    if exact_hit is not None and requested_kb_ids and exact_hit.kb_id and exact_hit.kb_id not in requested_kb_ids:
        # 客户端勾选的库不含该问答所属库：不能跨库回放标准答案
        exact_hit = None
    if exact_hit is not None:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        # 构造伪意图结果：标记 source="cache"，不调 LLM
        intent_result = IntentResult(
            intent=exact_hit.intent,
            rewrite_type="direct",
            sub_queries=[query],
            complexity="simple",
            hit_structured=True,
            structured_topic=exact_hit.category or None,
            law_hints=[],
            source="cache",
        )
        if exact_hit.kb_id:
            kb_ids = [exact_hit.kb_id]
        elif scope is not None and scope.has_grants:
            kb_ids = list(scope.kb_ids)
        else:
            kb_ids = list(ALL_KB_IDS)
        cached_route = RouteDecision(
            route="pure_structured",
            kb_ids=kb_ids,
            use_mysql_qa=True,
            use_milvus=False,
            reason="精确缓存命中：标准答案直接回放，不调 LLM 意图识别",
        )
        qa_result = QALookupResult(
            hit=exact_hit,
            source="redis_exact",
            similarity=1.0,
            elapsed_ms=elapsed_ms,
        )
        logger.info(
            "缓存前置命中 q=%.40s qa_id=%s 跳过意图识别 耗时=%dms",
            query, exact_hit.id, elapsed_ms,
        )
        return QueryPlan(
            query=query,
            intent=intent_result,
            route=cached_route,
            qa=qa_result,
            sub_queries=[query],
            elapsed_ms=elapsed_ms,
        )

    # 1) 意图识别（LLM，失败自动降级规则）
    intent_result = recognize_intent(query, history=history, settings=s)

    # 2) 静态路由：先按"未精确命中"裁决，结构化候选再经问答库确认
    pre_route = route(intent_result, selected_kb_ids=selected_kb_ids, structured_hit=False)

    # 3) 结构化链路（仅路由要求时；纯法条/问候/无数据直接跳过，省一次 ngram 检索）
    qa_result = QALookupResult()
    final_route = pre_route
    if pre_route.use_mysql_qa:
        retriever = qa_retriever or get_qa_retriever()
        qa_result = retriever.lookup(
            query,
            structured_topic=intent_result.structured_topic,
            kb_ids=pre_route.kb_ids,
            scope=scope,
        )
        if qa_result.hit is not None:
            # 精确命中 → 纯结构化：不碰 Milvus
            final_route = route(intent_result, selected_kb_ids=selected_kb_ids, structured_hit=True)
            logger.info("路由终裁：纯结构化（问答精确命中 id=%s）", qa_result.hit.id)

    # 路由兜底收敛：cross_law 等分支会返回 ALL_KB_IDS，必须再按授权范围裁剪。
    if scope is not None and scope.has_grants:
        restricted = scope.restrict(final_route.kb_ids)
        if requested_kb_ids and not restricted:
            # 明确勾选的库都不可用时，不能借兜底扩回全库
            final_route = replace(final_route, kb_ids=[])
        else:
            final_route = replace(final_route, kb_ids=restricted or list(scope.kb_ids))

    sub_queries = list(intent_result.sub_queries) or ([query] if query else [])
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "查询规划完成 q=%.40s intent=%s route=%s(%s→%s) 子查询=%d 条 degraded=%s 耗时=%dms",
        query,
        intent_result.intent,
        final_route.route,
        pre_route.route,
        final_route.route,
        len(sub_queries),
        intent_result.source == "rules",
        elapsed_ms,
    )
    return QueryPlan(
        query=query,
        intent=intent_result,
        route=final_route,
        qa=qa_result,
        sub_queries=sub_queries,
        elapsed_ms=elapsed_ms,
    )
