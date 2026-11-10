"""数据源路由（设计方案 §6.3 路由矩阵）。

::

    场景          intent        hit_structured   MySQL   Milvus
    --------------------------------------------------------------
    问候          greeting       -                 ✘       ✘
    无数据        any(空库/无关)  -                 ✘       ✘
    纯结构化       labor_law      ✔(且问答精确命中)   ✔       ✘
    纯法条        labor/civil     ✘                 ✘       ✔
    混合          labor_law      ✔(未精确命中)      ✔       ✔
    跨法          general        -(忽略)            ✘       ✔(全部勾选库)

**纯结构化 vs 混合**的区分点在问答库是否**精确命中**：命中 → 只走
MySQL（不碰 Milvus，省 60–120ms 检索+重排）；未命中 → 结构化答案降级为
"参考"，与法条合并后走混合。因此 ``structured_hit`` 是显式入参：调用方
先用 qa_retriever 做一次廉价判定，再决定最终路由。
"""

from __future__ import annotations

from app.domain_profiles import GENERIC, profile_name_for_kb
from app.orchestration.models import (
    LAW_CIVIL,
    LAW_LABOR,
    LAW_LABOR_CONTRACT,
    IntentResult,
    RouteDecision,
)

#: 法律名 → 知识库 ID（与 mysql/init/02_seed_kb.sql 对齐）
LAW_TO_KB: dict[str, str] = {
    LAW_LABOR_CONTRACT: "kb_labor",
    LAW_LABOR: "kb_labor",
    LAW_CIVIL: "kb_civil",
}

#: 全部知识库
ALL_KB_IDS: tuple[str, ...] = ("kb_labor", "kb_civil")


def kbs_for_laws(law_hints: list[str]) -> list[str]:
    """法律名列表 → 去重后的知识库 ID 列表（保持出现顺序）。"""
    out: list[str] = []
    for law in law_hints:
        kb = LAW_TO_KB.get(law)
        if kb and kb not in out:
            out.append(kb)
    return out


def route(
    intent_result: IntentResult,
    *,
    selected_kb_ids: list[str] | None = None,
    structured_hit: bool = False,
    settings=None,
) -> RouteDecision:
    """按路由矩阵决策。

    :param intent_result: 意图识别结果
    :param selected_kb_ids: 用户在前端勾选的知识库；``None``/空 = 全部勾选
    :param structured_hit: 问答库是否精确命中（qa_retriever 判定），
        仅当 ``hit_structured=True`` 时有意义
    """
    intent = intent_result.intent
    selected = list(dict.fromkeys(selected_kb_ids or []))
    #: 所选库全部显式配置为 generic（产品 FAQ/手册）时，不套法律路由规则
    generic_only = bool(selected) and all(
        profile_name_for_kb(kb_id, settings) == GENERIC for kb_id in selected
    )

    # ---- 问候 / 与法律无关：不检索，固定话术 ----
    if intent == "greeting":
        return RouteDecision(route="greeting", kb_ids=[], reason="问候/闲聊：固定引导话术，不调用任何数据源")
    if intent == "other":
        if generic_only:
            return RouteDecision(
                route="pure_vector",
                kb_ids=selected,
                use_milvus=True,
                reason="产品资料库：非法律问题直接向量检索，不套法条路由",
            )
        return RouteDecision(route="no_data", kb_ids=[], reason="与法律无关：引导话术，不调用任何数据源")

    # ---- 解析目标知识库 ----
    if selected_kb_ids:
        kb_ids = list(dict.fromkeys(selected_kb_ids))  # 去重保序
        hinted = kbs_for_laws(intent_result.law_hints)
        if hinted and not generic_only and not set(kb_ids) & set(hinted):
            return RouteDecision(
                route="no_data",
                kb_ids=[],
                reason=f"所选知识库 {kb_ids} 不含相关法律（{intent_result.law_hints}），建议切换知识库",
            )
    else:
        kb_ids = kbs_for_laws(intent_result.law_hints) or list(ALL_KB_IDS)

    if not kb_ids:
        return RouteDecision(route="no_data", kb_ids=[], reason="无可用知识库")

    # ---- 跨法/泛法律：全库向量检索，不走结构化 ----
    if intent == "general" and generic_only:
        return RouteDecision(
            route="pure_vector",
            kb_ids=kb_ids,
            use_milvus=True,
            reason="产品资料库：泛问题直接向量检索",
        )
    if intent == "general":
        return RouteDecision(
            route="cross_law",
            kb_ids=kb_ids,
            use_milvus=True,
            reason="跨法/泛法律问题：多库检索，prompt 要求分法条陈述",
        )

    # ---- 劳动/民法典：按 hit_structured 分流 ----
    if intent_result.hit_structured:
        if structured_hit:
            return RouteDecision(
                route="pure_structured",
                kb_ids=kb_ids,
                use_mysql_qa=True,
                use_milvus=False,
                reason="高频问题精确命中：标准答案直接作答，不调 Milvus",
            )
        return RouteDecision(
            route="hybrid",
            kb_ids=kb_ids,
            use_mysql_qa=True,
            use_milvus=True,
            reason="命中高频话题但无精确答案：结构化答案作结论段 + 法条作依据段",
        )

    return RouteDecision(
        route="pure_vector",
        kb_ids=kb_ids,
        use_milvus=True,
        reason="纯法条问题：仅向量检索，不查问答表",
    )
