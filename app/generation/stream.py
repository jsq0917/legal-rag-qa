"""生成链路编排：SSE 事件流 + 7 层容错（设计方案 §6.5）。

一次问答的事件序列::

    meta → stage(intent) → [stage(retrieval) → stage(rerank)] → sources
         → stage(generate) → delta×N → done
    异常路径：…… → delta×N（已产出的保留）→ error{code, message, retriable}

**7 层容错**（任一层的失败都不让整条链路断）：
1. 意图 LLM 失败 → 规则兜底（阶段 4 已实现，``meta.degraded=true`` 标记）；
2. 生成 LLM 首次连接失败 → 指数退避重试（``llm_max_retries``）；
3. 生成 LLM 上游断流 → 保留已产出 partial，发 ``error(retriable=true)`` 并落库；
4. 客户端断开（生成器被 ``close()``）→ 停止读上游、照常落库、不发 error；
5. Milvus 检索异常 → 追加 ``meta(degraded="vector_unavailable")``，
   有问答参考则降级回答，否则走无数据引导话术（§1.3 双链路独立降级）；
6. Redis 问答缓存异常 → qa_retriever 内部降级（阶段 4 已实现）；
7. 无任何资料 → 引导话术，**不编造**。

结构化命中（pure_structured）不走 LLM 全文生成：标准答案分片回放，
SSE 语义与真实流式一致（打字机效果），``done.cached`` 标记是否来自 Redis 精确缓存。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterator, Sequence

from app.core.access import AccessScope, answer_identity
from app.core.logger import get_logger
from app.core.metrics import record_guardrail_hit
from app.core.sse import sse_frame
from app.core.trace import Trace
from app.security.guardrails import redact_sensitive
from app.domain_profiles import profile_for_kbs
from app.generation.context import (
    aggregate_sources,
    build_context_sources,
    qa_conclusion_block,
)
from app.generation.citations import (
    append_citation_note,
    check_citations,
    format_citation_note,
    strip_citation_note,
)
from app.generation.history import trim_history
from app.generation.llm_client import LLMClient, LLMError, LLMStreamInterrupted
from app.generation.persistence import persist_exchange
from app.generation.prompts import (
    GREETING_REPLY,
    NO_DATA_REPLY,
    PROMPT_VERSION,
    build_context_block,
    build_messages,
)
from app.orchestration.answer_cache import AnswerCache, CachedAnswer, get_answer_cache
from app.orchestration.qa_cache import QARetriever
from app.orchestration.service import QueryPlan, analyze_query
from config import ConfigError, Settings, get_settings

logger = get_logger("lawanswer.generation")


@dataclass
class StreamResult:
    """一次流式问答的累积状态（done 事件与落库的数据源）。"""

    content: str = ""
    sources: list[dict] = field(default_factory=list)
    ttft_ms: int | None = None
    latency_ms: int = 0
    token_usage: dict = field(default_factory=dict)
    cached: bool = False
    error: str | None = None
    persisted: bool = False
    #: 生成后引用校验结果（写进 SSE meta.citation_check，见 citations.py）
    citation_check: dict = field(default_factory=dict)
    citation_check_ms: int = 0
    #: 降级原因（如 ``llm_fallback:<model>``）；正常为 None
    degraded: str | None = None
    #: 生成是否真正跑完（含引用校验）；客户端断流时保持 False，禁止写缓存
    completed: bool = False
    #: 输出护栏命中的敏感信息类别（如 api_key/internal_path/system_prompt）
    guardrail_hits: list[str] = field(default_factory=list)


def _meta_frame(
    message_id: str,
    *,
    intent: str,
    route: str,
    rewrite_type: str,
    complexity: str,
    cached: bool,
    degraded: bool | str,
    citation_check: dict | None = None,
    trace_id: str = "",
) -> str:
    """meta 帧统一构造：前端收到 meta 会整体覆盖，字段必须一次给全。"""
    payload = {
        "message_id": message_id,
        "trace_id": trace_id,
        "intent": intent,
        "route": route,
        "rewrite_type": rewrite_type,
        "complexity": complexity,
        "cached": cached,
        "degraded": degraded,
    }
    if citation_check is not None:
        payload["citation_check"] = citation_check
    return sse_frame("meta", payload)


class AnswerEngine:
    """把一次查询变成 SSE 帧序列的**同步生成器**。

    API 层用 ``starlette.concurrency.iterate_in_threadpool`` 包装；
    同步实现让 TestClient 与单测可以直接逐帧消费，也避免在异步端点里
    混用阻塞调用（embedder/Milvus/LLM 全是同步 SDK）。
    """

    def __init__(
        self,
        *,
        qa_retriever: QARetriever | None = None,
        llm_client: LLMClient | None = None,
        retrieve_fn: Callable[..., Any] | None = None,
        persist_fn: Callable[..., bool] | None = None,
        answer_cache: AnswerCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._qa_retriever = qa_retriever
        self._llm_client = llm_client
        self._retrieve_fn = retrieve_fn
        self._persist_fn = persist_fn or persist_exchange
        self._answer_cache = answer_cache
        self._settings = settings

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    @property
    def answer_cache(self) -> AnswerCache:
        if self._answer_cache is None:
            self._answer_cache = get_answer_cache()
        return self._answer_cache

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def stream(
        self,
        query: str,
        *,
        history: Sequence[dict] | None = None,
        selected_kb_ids: list[str] | None = None,
        session_id: str | None = None,
        scope: AccessScope | None = None,
        trace: Trace | None = None,
    ) -> Iterator[str]:
        """产出 SSE 帧（``event: X\\ndata: {...}\\n\\n``）。"""
        s = self.settings
        gen_cfg = s.generation
        message_id = uuid.uuid4().hex
        started = time.perf_counter()
        result = StreamResult()
        plan: QueryPlan | None = None
        query = (query or "").strip()
        scope = scope or AccessScope.legacy_all()
        trace = trace or Trace()
        trace.user_id = scope.user_id
        trace.record(
            "request",
            kb_ids=list(scope.kb_ids),
            requested=list(scope.requested_kb_ids),
            denied=list(scope.denied_kb_ids),
            tenant_id=scope.tenant_id,
            role=scope.role,
        )
        identity = self._answer_identity(scope)
        # 客户端勾选只能缩小范围；最终检索范围一律由 scope 决定。
        effective_kb_ids = scope.restrict(selected_kb_ids)

        try:
            # ---- 0) 授权范围为空：不检索、不回缓存、不调 LLM ----
            if scope.has_grants and not effective_kb_ids:
                logger.info(
                    "授权范围为空，走引导话术 requested=%d denied=%d",
                    len(scope.requested_kb_ids),
                    len(scope.denied_kb_ids),
                )
                yield _meta_frame(
                    message_id,
                    intent="general",
                    route="no_data",
                    rewrite_type="direct",
                    complexity="simple",
                    cached=False,
                    degraded=False,
                    trace_id=trace.trace_id,
                )
                yield from self._replay_fixed(NO_DATA_REPLY, gen_cfg.replay_chunk_chars, result, started)
                yield self._done_frame(result, started, trace.trace_id)
                return

            # ---- 意图 + 路由 + 结构化判定（阶段 4 编排入口，含 LLM 降级）----
            yield sse_frame("stage", {"stage": "intent"})
            intent_started = time.perf_counter()
            plan = analyze_query(
                query,
                history=list(history) if history else None,
                selected_kb_ids=effective_kb_ids,
                qa_retriever=self._qa_retriever,
                settings=s,
                scope=scope,
            )
            route = plan.route.route
            trace.record(
                "intent",
                latency_ms=int((time.perf_counter() - intent_started) * 1000),
                source=plan.intent.source,
                intent=plan.intent.intent,
                rewrite_type=plan.intent.rewrite_type,
                complexity=plan.intent.complexity,
                model=s.llm_default_model,
                prompt_version=self._intent_prompt_version(),
                tokens=plan.intent.token_usage or {},
                degraded=bool(plan.degraded),
                validation_errors=list(plan.intent.validation_errors),
            )
            trace.record(
                "route",
                latency_ms=0,
                measured_with="intent",
                route=route,
                kb_ids=list(plan.route.kb_ids),
                reason=plan.route.reason,
            )

            # ---- 0.5) 答案级缓存：必须在独立化改写之后查 ----
            # key 由「改写后问题 + 租户/知识库/版本/模型/prompt/as-of」共同决定。
            # 多轮追问原样做键会跨上下文误命中（O1-1/O1-2）；plan.cache_query
            # 为空表示多子查询且有历史，直接跳过缓存。
            cache_started = time.perf_counter()
            cache_query = plan.cache_query
            cache_eligible = route in self._ANSWER_CACHE_ROUTES and bool(cache_query)
            cached_answer = (
                self.answer_cache.lookup(
                    cache_query,
                    history=list(history) if history else None,
                    cache_identity=identity,
                )
                if cache_eligible
                else None
            )
            trace.record(
                "cache_lookup",
                latency_ms=int((time.perf_counter() - cache_started) * 1000),
                layer="answer",
                hit=cached_answer is not None,
                eligible=cache_eligible,
                query=cache_query,
                identity=identity,
            )
            if cached_answer is not None:
                yield from self._answer_replay(
                    cached_answer, result, gen_cfg, started, message_id, scope, trace
                )
                yield self._done_frame(result, started, trace.trace_id)
                return

            yield _meta_frame(
                message_id,
                intent=plan.intent.intent,
                route=route,
                rewrite_type=plan.intent.rewrite_type,
                complexity=plan.intent.complexity,
                cached=False,
                degraded=plan.degraded,
                trace_id=trace.trace_id,
            )

            # ---- 分支 A：问候 → 固定话术，不调 LLM ----
            if route == "greeting":
                yield from self._replay_fixed(GREETING_REPLY, gen_cfg.replay_chunk_chars, result, started)
                yield self._done_frame(result, started, trace.trace_id)
                return

            # ---- 分支 B：纯结构化 → 标准答案回放，不碰 Milvus / 不调 LLM ----
            if route == "pure_structured":
                yield from self._structured_replay(plan, result, gen_cfg, started)
                yield self._done_frame(result, started, trace.trace_id)
                return

            # ---- 分支 B2：无数据（知识库不匹配/与法律无关）→ 引导话术，
            #      不调检索、不调 LLM（§6.3：no_data 不调用任何数据源）----
            if route == "no_data":
                logger.info("路由 no_data：引导话术 q=%.40s reason=%s", query, plan.route.reason)
                yield from self._replay_fixed(NO_DATA_REPLY, gen_cfg.replay_chunk_chars, result, started)
                yield self._done_frame(result, started, trace.trace_id)
                return

            # ---- 分支 C：检索（pure_vector / hybrid / cross_law）----
            yield sse_frame("stage", {"stage": "retrieval"})
            retrieve_started = time.perf_counter()
            hits, degraded, retrieve_debug = self._retrieve(plan, s, scope)
            retrieve_ms = int((time.perf_counter() - retrieve_started) * 1000)
            hits = self._filter_hits(hits, scope)
            trace.record(
                "retrieve",
                latency_ms=retrieve_ms,
                hits=len(hits),
                mode=retrieve_debug.get("mode", ""),
                kb_ids=retrieve_debug.get("kb_ids", []),
                kb_versions=retrieve_debug.get("kb_versions", {}),
                reranked=bool(retrieve_debug.get("reranked", False)),
                degraded=degraded or "",
            )
            trace.record(
                "rerank",
                latency_ms=int(retrieve_debug.get("rerank_ms", 0) or 0),
                reranked=bool(retrieve_debug.get("reranked", False)),
                candidates=int(retrieve_debug.get("candidates", 0) or 0),
            )
            if degraded:
                # 第 5 层容错：Milvus 故障 → 追加一条 meta 告知前端降级状态
                logger.warning("Milvus 检索不可用，按 vector_unavailable 降级回答")
                yield _meta_frame(
                    message_id,
                    intent=plan.intent.intent,
                    route=route,
                    rewrite_type=plan.intent.rewrite_type,
                    complexity=plan.intent.complexity,
                    cached=False,
                    degraded="vector_unavailable",
                    trace_id=trace.trace_id,
                )

            if hits:
                yield sse_frame("stage", {"stage": "rerank"})

            # 混合链路才带结构化素材：精确命中作结论段，同类参考作补充
            qa_hit = plan.qa.hit if route == "hybrid" else None
            similarity = plan.qa.similarity if route == "hybrid" else None
            references = plan.qa.references if route == "hybrid" else []

            result.sources = [
                r.to_dict()
                for r in aggregate_sources(
                    qa_hit=qa_hit,
                    similarity=similarity,
                    references=references,
                    hits=hits,
                    limit=gen_cfg.sources_limit,
                )
            ]

            # ---- 分支 D：无任何资料 → 引导话术，不编造（第 7 层）----
            if not result.sources:
                logger.info("无资料可用，走引导话术 q=%.40s route=%s", query, route)
                yield from self._replay_fixed(NO_DATA_REPLY, gen_cfg.replay_chunk_chars, result, started)
                yield self._done_frame(result, started, trace.trace_id)
                return

            context_sources = build_context_sources(
                qa_hit=qa_hit,
                similarity=similarity,
                references=references,
                hits=hits,
            )
            context_block = self._build_context(context_sources, qa_hit, gen_cfg)
            trimmed_history = trim_history(
                history,
                max_rounds=gen_cfg.history_max_rounds,
                token_budget=gen_cfg.history_token_budget,
            )
            # 领域口径：法律库用六段式法条 Prompt；产品库用资料/章节口径
            profile = profile_for_kbs(
                effective_kb_ids or [getattr(h, "kb_id", "") for h in hits],
                settings=s,
            )
            messages = build_messages(
                query,
                context_block=context_block,
                history=trimmed_history,
                system_prompt=profile.system_prompt,
            )

            yield sse_frame("sources", {"sources": result.sources})
            yield sse_frame("stage", {"stage": "generate"})

            # ---- 流式生成（第 2/3 层容错）----
            generate_started = time.perf_counter()
            yield from self._generate(messages, result, gen_cfg, started)
            # 只有生成器正常耗尽（含引用校验）才算完成；客户端 close() 会跳过
            # 这一行，result.error 仍为 None，必须靠该标记拦住半截回答写缓存。
            result.completed = True
            trace.record(
                "generate",
                latency_ms=int((time.perf_counter() - generate_started) * 1000),
                ttft_ms=result.ttft_ms or 0,
                model=s.llm_default_model,
                prompt_version=PROMPT_VERSION,
                tokens=result.token_usage or {},
                cached=False,
                degraded=result.degraded or "",
                error=result.error or "",
            )
            if result.guardrail_hits:
                trace.record(
                    "guardrail",
                    latency_ms=0,
                    hits=sorted(set(result.guardrail_hits)),
                    count=len(result.guardrail_hits),
                )
            trace.record(
                "citation_check",
                latency_ms=result.citation_check_ms,
                enabled=bool(gen_cfg.citation_check),
                unverified=len(result.citation_check.get("unverified", []) if result.citation_check else []),
                error=result.error or "",
            )
            if result.citation_check:
                # 生成后才拿得到完整答案，此时再发一条「带全字段」的 meta：
                # 前端收 meta 会整体覆盖，缺字段会清空 UI 元信息。
                yield _meta_frame(
                    message_id,
                    intent=plan.intent.intent,
                    route=route,
                    rewrite_type=plan.intent.rewrite_type,
                    complexity=plan.intent.complexity,
                    cached=False,
                    degraded=degraded or plan.degraded,
                    citation_check=result.citation_check,
                    trace_id=trace.trace_id,
                )
            yield self._done_frame(result, started, trace.trace_id)

        except LLMStreamInterrupted as exc:
            # 第 3 层：断流保留 partial（delta 已在 _generate 内产出）
            result.error = f"stream_interrupted: {exc}"
            logger.warning("生成断流，保留已产出 %d 字", len(exc.partial))
            yield sse_frame(
                "error",
                {
                    "code": "stream_interrupted",
                    "message": redact_sensitive(str(exc)[:200])[0],
                    "retriable": True,
                },
            )
            yield self._done_frame(result, started, trace.trace_id)
        except ConfigError as exc:
            # 缺 DASHSCOPE_API_KEY：明确的错误码，前端可引导用户/管理员去配 key
            result.error = f"llm_not_configured: {exc}"
            logger.warning("LLM 未配置密钥：%s", str(exc)[:200])
            yield sse_frame(
                "error", {"code": "llm_not_configured", "message": "生成服务未配置 API Key，请联系管理员", "retriable": False}
            )
            yield self._done_frame(result, started, trace.trace_id)
        except LLMError as exc:
            # 第 2 层：重试后仍失败
            result.error = f"llm_error: {exc}"
            logger.warning("生成失败：%s", str(exc)[:200])
            yield sse_frame(
                "error",
                {
                    "code": "llm_error",
                    "message": redact_sensitive(str(exc)[:200])[0],
                    "retriable": True,
                },
            )
            yield self._done_frame(result, started, trace.trace_id)
        except Exception as exc:  # noqa: BLE001 - 顶层兜底：链路绝不能裸奔
            result.error = f"internal_error: {exc}"
            logger.exception("问答链路未捕获异常")
            yield sse_frame("error", {"code": "internal_error", "message": "服务内部错误", "retriable": False})
            yield self._done_frame(result, started, trace.trace_id)
        finally:
            # 第 4 层：客户端断开（GeneratorExit）或任何路径，finally 保证落库一次
            cache_write_started = time.perf_counter()
            self._cache_answer(result, plan, identity=identity, scope=scope)
            trace.record(
                "answer_cache_write",
                latency_ms=int((time.perf_counter() - cache_write_started) * 1000),
                hit=result.cached,
                error=result.error or "",
            )
            persist_started = time.perf_counter()
            self._persist(session_id, query, result, plan)
            trace.record(
                "persistence",
                latency_ms=int((time.perf_counter() - persist_started) * 1000),
                persisted=bool(result.persisted and session_id),
                error=result.error or "",
            )
            # 失败路径（意图/生成抛错）没有 generate 事件，这里补一条便于还原
            if not any(e.get("stage") == "generate" for e in trace.events):
                trace.record(
                    "generate",
                    latency_ms=0,
                    model=s.llm_default_model,
                    prompt_version=PROMPT_VERSION,
                    cached=False,
                    error=result.error or "aborted",
                )
            trace.record(
                "done",
                latency_ms=0,
                ok=not bool(result.error),
                ttft_ms=result.ttft_ms or 0,
                total_ms=int((time.perf_counter() - started) * 1000),
                cached=result.cached,
            )
            trace.log()
            trace.save()

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------
    def _retrieve(
        self,
        plan: QueryPlan,
        settings: Settings,
        scope: AccessScope | None = None,
    ) -> tuple[list, str | None, dict]:
        """检索：按意图改写类型选路。

        - ``rewrite_type == "decompose"``（多跳拆解）→ ``retrieve_multi`` 并行
          分检 + 名次交错融合（评测证实：多跳 +8.3pt，P6 组）；
        - 其余（direct/hyde/simplify 的 sub_queries 只是**同义改写**，不是拆解）
          → 原句单查询直检（评测证实：改写句作第二投票人反而稀释锚点排序，
          单跳 −13.3pt，故不用）。

        Milvus/检索异常 → 第 5 层容错：空列表 + degraded 标记，
        由调用方决定降级回答还是引导话术。
        """
        from app.retrieval.service import retrieve, retrieve_multi

        # 路由给出的 kb_ids 再收敛一次：即使路由层被注入/越权，也不会扩大范围。
        route_kb_ids = list(plan.route.kb_ids) if scope is None else scope.restrict(plan.route.kb_ids)
        if scope is not None and scope.has_grants and not route_kb_ids:
            return [], None, {}
        if scope is not None and scope.has_grants:
            route_kb_ids = route_kb_ids or list(scope.kb_ids)

        try:
            use_multi = plan.intent.rewrite_type == "decompose" and len(plan.sub_queries or []) > 1
            if use_multi:
                # 多跳必须混合检索：拆解子查询各自是单跳，但子查询之间靠稀疏路
                # 命中跨法术语；这里固定 hybrid，不交给单查询的 auto 分流。
                res = retrieve_multi(
                    plan.sub_queries,
                    kb_ids=list(route_kb_ids) or None,
                    complexity=plan.intent.complexity,
                    law_hints=list(plan.intent.law_hints) or None,
                    anchor_query=plan.query,
                    mode="hybrid",
                    settings=settings,
                    retrieve_fn=self._retrieve_fn or retrieve,
                )
            else:
                res = (self._retrieve_fn or retrieve)(
                    plan.query,
                    kb_ids=list(route_kb_ids) or None,
                    complexity=plan.intent.complexity,
                    law_hints=list(plan.intent.law_hints) or None,
                    mode="auto",
                    rewrite_type=plan.intent.rewrite_type,
                    route=plan.route.route,
                    settings=settings,
                )
            return res.hits, None, dict(getattr(res, "debug", None) or {})
        except Exception as exc:  # noqa: BLE001 - 检索故障降级，不打断链路
            logger.warning("检索失败，降级处理（%s: %s）", type(exc).__name__, str(exc)[:200])
            return [], "vector_unavailable", {}

    @staticmethod
    def _build_context(context_sources: Sequence, qa_hit, gen_cfg) -> str:
        """混合链路：标准答案作结论段置顶，法规/产品资料作依据段（§6.3）。"""
        parts: list[str] = []
        if qa_hit is not None:
            parts.append(qa_conclusion_block(qa_hit))
        # 法条 kind="law"，非法律库章节 kind="section"：两者都必须进 Prompt，
        # 否则 generic 知识库的检索结果会被静默丢弃（N3 验收缺陷）。
        knowledge_sources = [c for c in context_sources if c.kind in ("law", "section")]
        if knowledge_sources:
            label = (
                "【法条资料】"
                if all(c.kind == "law" for c in knowledge_sources)
                else "【知识库资料】"
            )
            block, _ = build_context_block(
                knowledge_sources, max_chars=gen_cfg.context_max_chars
            )
            if block:
                parts.append(f"{label}\n{block}")
        # 无精确命中时的同类参考也进上下文（问+答成对给出）
        refs = [c for c in context_sources if c.kind == "qa" and c.law != "高频问答库"]
        for ref in refs[:3]:
            parts.append(f"【同类问题参考】{ref.content}")
        return "\n\n".join(parts)

    def _generate(self, messages: list[dict], result: StreamResult, gen_cfg, started: float) -> Iterator[str]:
        """流式生成：逐 chunk 产出 delta 帧，记录 TTFT（首 token）与用量。

        ``llm_fallback_model`` 配置且可用时，首个 token 前失败会换备用模型
        重试一次；已产出内容后的失败不换模型（避免两段回答拼接错乱）。
        """
        client = self._llm_client or self._default_llm_client()
        fallback = (self.settings.llm_fallback_model or "").strip()
        attempt_models: list[str | None] = [None]
        if fallback:
            attempt_models.append(fallback)

        last_exc: LLMError | None = None
        for idx, model in enumerate(attempt_models):
            produced_before = len(result.content)
            try:
                for chunk in client.chat_stream(
                    messages,
                    model=model,
                    temperature=gen_cfg.temperature,
                    max_tokens=gen_cfg.max_tokens,
                ):
                    if chunk.usage:
                        result.token_usage = dict(chunk.usage)
                    if not chunk.text:
                        continue
                    # 输出护栏：逐 chunk 脱敏，密钥/内部路径绝不流到前端
                    text, leaks = redact_sensitive(chunk.text)
                    if leaks:
                        kinds = [hit["kind"] for hit in leaks]
                        result.guardrail_hits.extend(kinds)
                        for kind in kinds:
                            record_guardrail_hit(kind)
                        logger.warning("输出护栏命中：%s", sorted(set(kinds)))
                    if not text:
                        continue
                    if result.ttft_ms is None:
                        result.ttft_ms = int((time.perf_counter() - started) * 1000)
                    result.content += text
                    yield sse_frame("delta", {"text": text})
            except LLMStreamInterrupted:
                raise
            except LLMError as exc:
                last_exc = exc
                can_fallback = (
                    idx == 0
                    and fallback
                    and len(result.content) == produced_before
                    and exc.kind != "circuit_open"
                )
                if not can_fallback:
                    raise
                result.degraded = f"llm_fallback:{fallback}"
                logger.warning(
                    "生成失败（%s），改用备用模型 %s 重试一次", exc.kind, fallback
                )
                continue
            break
        else:
            # 所有候选模型都失败（fallback 未配置时只会有一轮）
            raise last_exc or LLMError("生成失败：无可用模型")

        # ---- 引用在线校验：答案里出现来源外的条号时补一行提示 ----
        if gen_cfg.citation_check:
            check_started = time.perf_counter()
            check = check_citations(result.content, result.sources)
            result.citation_check = check
            result.citation_check_ms = int((time.perf_counter() - check_started) * 1000)
            if check["unverified"]:
                note = format_citation_note(check["unverified"])
                result.content += note
                yield sse_frame("delta", {"text": note})

    def _answer_replay(
        self,
        cached: CachedAnswer,
        result: StreamResult,
        gen_cfg,
        started: float,
        message_id: str,
        scope: AccessScope | None = None,
        trace: Trace | None = None,
    ) -> Iterator[str]:
        """答案缓存命中：整段回放（保留打字机效果），不调意图/检索/LLM。"""
        trace = trace or Trace()
        result.cached = True
        result.sources = self._filter_sources(cached.sources, scope)
        # 缓存里存的是「未补注」的原答案（写缓存时剔除），回放时按同一来源
        # 重新校验，保证实时校验与缓存命中给出一致的标记。
        answer = strip_citation_note(cached.answer)
        citation_check: dict | None = None
        if gen_cfg.citation_check:
            check = check_citations(answer, result.sources)
            result.citation_check = check
            answer = append_citation_note(answer, check)
            citation_check = check
        yield _meta_frame(
            message_id,
            intent=cached.intent,
            route=cached.route,
            rewrite_type=cached.rewrite_type,
            complexity=cached.complexity,
            cached=True,
            degraded=False,
            citation_check=citation_check,
            trace_id=trace.trace_id,
        )
        trace.record(
            "generate",
            latency_ms=0,
            cached=True,
            model=self.settings.llm_default_model,
            prompt_version=PROMPT_VERSION,
            token_usage={},
        )
        if result.sources:
            yield sse_frame("sources", {"sources": result.sources})
        yield sse_frame("stage", {"stage": "generate"})
        yield from self._replay_fixed(answer, gen_cfg.replay_chunk_chars, result, started)

    #: 只有"检索 + 生成"链路才落答案缓存：greeting / no_data 是固定话术，
    #: pure_structured 已由 qa_cache 写 qa:exact，重复写只是浪费内存。
    _ANSWER_CACHE_ROUTES = frozenset({"pure_vector", "hybrid", "cross_law"})

    def _cache_answer(
        self,
        result: StreamResult,
        plan: QueryPlan | None,
        *,
        identity: str = "",
        scope: AccessScope | None = None,
    ) -> None:
        """把本次成功生成的回答写入答案缓存。

        写入条件很严：缓存直出等于跳过 LLM，一条错答案会被固化整个 TTL。
        断流 / 报错 / 未完成 / 无来源 / 缓存回放自身 / 固定话术分支一律不写。
        """
        if result.cached or plan is None or result.error or not result.completed:
            return
        if plan.route.route not in self._ANSWER_CACHE_ROUTES:
            return
        if not result.content.strip():
            return
        # 与查询侧同一把键：用独立化改写后的问题；空表示多子查询且有历史，不缓存
        store_query = plan.cache_query
        if not store_query:
            return
        try:
            self.answer_cache.store(
                store_query,
                answer=strip_citation_note(result.content),
                sources=result.sources,
                route=plan.route.route,
                intent=plan.intent.intent,
                rewrite_type=plan.intent.rewrite_type,
                complexity=plan.intent.complexity,
                cache_identity=identity,
                cache_meta=self._cache_meta(scope),
            )
        except Exception as exc:  # noqa: BLE001 - 缓存失败永不影响主链路
            logger.warning("答案缓存回写异常（已忽略）: %s", str(exc)[:200])

    @staticmethod
    def _intent_prompt_version() -> str:
        from app.orchestration.intent import INTENT_PROMPT_VERSION

        return INTENT_PROMPT_VERSION

    def _answer_identity(self, scope: AccessScope) -> str:
        s = self.settings
        as_of = s.retrieval_effective_as_of or date.today().isoformat()
        return answer_identity(
            scope,
            model=s.llm_default_model,
            prompt_version=PROMPT_VERSION,
            as_of=as_of,
        )

    def _cache_meta(self, scope: AccessScope | None) -> dict:
        s = self.settings
        scope = scope or AccessScope.legacy_all()
        return {
            "namespace": scope.cache_namespace(),
            "kb_ids": list(scope.kb_ids),
            "corpus_versions": {g.kb_id: int(g.corpus_version) for g in scope.grants},
            "index_versions": {g.kb_id: int(g.index_version) for g in scope.grants},
            "model": s.llm_default_model,
            "prompt_version": PROMPT_VERSION,
            "as_of": s.retrieval_effective_as_of or date.today().isoformat(),
        }

    @staticmethod
    def _filter_hits(hits: Sequence, scope: AccessScope | None) -> list:
        if scope is None or not scope.has_grants:
            return list(hits)
        return [h for h in hits if scope.allows_kb(getattr(h, "kb_id", None))]

    @staticmethod
    def _filter_sources(sources: Sequence[dict], scope: AccessScope | None) -> list[dict]:
        """来源级双保险：没有 kb_id 的旧缓存快照保持原样（key 已隔离）。"""
        if scope is None or not scope.has_grants:
            return list(sources)
        kept: list[dict] = []
        for src in sources:
            kb_id = src.get("kb_id") if isinstance(src, dict) else None
            if kb_id and not scope.allows_kb(str(kb_id)):
                continue
            kept.append(dict(src))
        return kept

    def _structured_replay(self, plan: QueryPlan, result: StreamResult, gen_cfg, started: float) -> Iterator[str]:
        """纯结构化分支：标准答案分片回放（防御：无命中则降级引导话术）。"""
        hit = plan.qa.hit
        if hit is None:
            # 理论不可达（路由已保证命中）；缺失时不碰 Milvus，直接引导话术
            logger.warning("pure_structured 但问答未命中，降级引导话术")
            yield from self._replay_fixed(NO_DATA_REPLY, gen_cfg.replay_chunk_chars, result, started)
            return
        yield sse_frame("stage", {"stage": "generate"})
        result.sources = [
            r.to_dict()
            for r in aggregate_sources(qa_hit=hit, similarity=plan.qa.similarity, limit=gen_cfg.sources_limit)
        ]
        result.cached = plan.qa.source == "redis_exact"
        yield sse_frame("sources", {"sources": result.sources})
        yield from self._replay_fixed(hit.answer, gen_cfg.replay_chunk_chars, result, started)

    def _replay_fixed(self, text: str, chunk_chars: int, result: StreamResult, started: float) -> Iterator[str]:
        """固定话术/标准答案分片回放（打字机效果，SSE 语义与流式一致）。"""
        text = (text or "").strip()
        if not text:
            return
        step = max(5, chunk_chars)
        for i in range(0, len(text), step):
            piece = text[i : i + step]
            if result.ttft_ms is None:
                # 回放路径同样记录 TTFT（缓存命中速度是结构化链路的核心指标）
                result.ttft_ms = int((time.perf_counter() - started) * 1000)
            result.content += piece
            yield sse_frame("delta", {"text": piece})

    @staticmethod
    def _done_frame(result: StreamResult, started: float, trace_id: str = "") -> str:
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return sse_frame(
            "done",
            {
                "trace_id": trace_id,
                "ttft_ms": result.ttft_ms or 0,
                "latency_ms": result.latency_ms,
                "token_usage": result.token_usage or {},
                "sources_count": len(result.sources),
                "cached": result.cached,
            },
        )

    def _default_llm_client(self) -> LLMClient:
        from app.generation.llm_client import get_llm_client

        return get_llm_client(self.settings)

    def _persist(self, session_id: str | None, query: str, result: StreamResult, plan: QueryPlan | None) -> None:
        """幂等落库：正常结束、断流、客户端断开都保证调用且只调一次。"""
        if result.persisted or not session_id:
            return
        result.persisted = True
        try:
            self._persist_fn(
                session_id,
                user_content=query,
                assistant_content=result.content,
                sources=result.sources,
                intent=plan.intent.intent if plan else None,
                rewrite_type=plan.intent.rewrite_type if plan else None,
                route=plan.route.route if plan else None,
                cached=result.cached,
                latency_ms=result.latency_ms,
                ttft_ms=result.ttft_ms,
                token_usage=result.token_usage or None,
                error=result.error,
                settings=self.settings,
            )
        except Exception as exc:  # noqa: BLE001 - 双保险：落库永不影响流
            logger.warning("落库调用失败（已忽略）: %s", str(exc)[:200])
