"""兜底话术与降级标记测试（阶段 5，全离线，§6.5 验收标准第 3 条）。

验收标准："知识库无相关内容时给出引导话术而非编造"——
本文件锁定该行为的每一层：无资料 / 无知识库 / Milvus 故障 / 降级标记 /
LLM 全挂。话术内容本身在 prompts 中断言，此处断言**链路行为**。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.generation import stream as stream_mod
from app.generation.llm_client import LLMError, LLMStreamInterrupted
from app.generation.prompts import GREETING_REPLY, NO_DATA_REPLY
from app.generation.stream import AnswerEngine
from app.orchestration.models import IntentResult, RouteDecision
from app.orchestration.qa_cache import QALookupResult, QAPair
from app.orchestration.service import QueryPlan

from tests.test_sse import (
    FakeLLMStream,
    all_deltas,
    event_names,
    fake_persist,
    fake_retrieve_fn,
    get_event,
    make_hit,
    make_intent,
    make_plan,
    make_qa_pair,
    make_route,
)


def run(monkeypatch, plan, *, llm=None, retrieve=None, session_id=None):
    monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
    persist = fake_persist()
    engine = AnswerEngine(
        llm_client=llm or FakeLLMStream(),
        retrieve_fn=retrieve if retrieve is not None else fake_retrieve_fn([]),
        persist_fn=persist,
    )
    frames_raw = list(engine.stream(plan.query, session_id=session_id))
    from app.core.sse import parse_sse_frames

    frames = parse_sse_frames("".join(frames_raw))
    return frames, persist


class TestNoDataGuidance:
    def test_no_data_route_guidance(self, monkeypatch):
        """路由 no_data（无关问题/库不含相关法）：引导话术，不调 LLM、不调检索。"""
        plan = make_plan("no_data", intent=make_intent(intent="other", sub_queries=[]))
        retrieve = fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)])  # 若被调用即回归
        llm = FakeLLMStream()
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(llm_client=llm, retrieve_fn=retrieve)
        from app.core.sse import parse_sse_frames

        frames = parse_sse_frames("".join(engine.stream(plan.query)))
        assert "暂未在所选知识库中找到" in all_deltas(frames)
        assert get_event(frames, "done")["sources_count"] == 0
        assert retrieve.calls == []  # no_data 不调用任何数据源（§6.3）
        assert llm.calls == []

    def test_empty_retrieval_guidance(self, monkeypatch):
        plan = make_plan("pure_vector")
        frames, _ = run(monkeypatch, plan, retrieve=fake_retrieve_fn([]))
        text = all_deltas(frames)
        assert "暂未在所选知识库中找到" in text
        assert "劳动监察" in text  # 给下一步路径
        assert "仲裁" in text

    def test_guidance_never_calls_llm(self, monkeypatch):
        plan = make_plan("pure_vector")
        llm = FakeLLMStream()
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(llm_client=llm, retrieve_fn=fake_retrieve_fn([]))
        list(engine.stream(plan.query))
        assert llm.calls == []

    def test_no_fabrication_markers_in_reply(self):
        """话术本身不得含具体法条号（无资料时引用即编造）。"""
        import re

        assert not re.search(r"第[一二三四五六七八九十百零〇\d]+条", NO_DATA_REPLY)

    def test_greeting_guidance_no_retrieval(self, monkeypatch):
        plan = make_plan("greeting", intent=make_intent(intent="greeting", sub_queries=[]))
        retrieve = fake_retrieve_fn([])
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        llm = FakeLLMStream()
        engine = AnswerEngine(llm_client=llm, retrieve_fn=retrieve)
        from app.core.sse import parse_sse_frames

        frames = parse_sse_frames("".join(engine.stream(plan.query)))
        assert all_deltas(frames) == GREETING_REPLY
        assert retrieve.calls == [] and llm.calls == []


class TestDegradedMark:
    def test_degraded_true_on_rules_fallback(self, monkeypatch):
        plan = make_plan("pure_vector", intent=make_intent(source="rules"))
        frames, _ = run(monkeypatch, plan, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]))
        assert get_event(frames, "meta")["degraded"] is True

    def test_degraded_vector_unavailable(self, monkeypatch):
        plan = make_plan("hybrid", references=[make_qa_pair(3)])
        frames, _ = run(monkeypatch, plan, retrieve=fake_retrieve_fn([], raises=RuntimeError("milvus timeout")))
        metas = [d for n, d in frames if n == "meta"]
        assert metas[0]["degraded"] is False
        assert metas[-1]["degraded"] == "vector_unavailable"

    def test_degraded_persisted_flag(self, monkeypatch):
        """degraded 语义落到落库：intent.source=rules → 该条消息可被阶段 7 统计。"""
        plan = make_plan("pure_vector", intent=make_intent(source="rules"))
        _, persist = run(monkeypatch, plan, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]), session_id="s1")
        # 落库记录里 route/intent 齐全（degraded 由 intent.source 推导，阶段 7 可 join 统计）
        assert persist.records[0]["route"] == "pure_vector"


class TestLLMTotallyDown:
    def test_llm_error_still_completes_protocol(self, monkeypatch):
        plan = make_plan("pure_vector")
        frames, persist = run(
            monkeypatch, plan,
            retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            llm=FakeLLMStream(raises=LLMError("额度不足")),
            session_id="s2",
        )
        names = event_names(frames)
        assert "error" in names and names[-1] == "done"
        err = get_event(frames, "error")
        assert err["retriable"] is True
        # 没有产出任何 delta（首 token 前就失败）→ 内容为空，不带编造
        assert all_deltas(frames) == ""
        assert persist.records[0]["assistant_content"] == ""
        assert persist.records[0]["error"].startswith("llm_error")

    def test_llm_interrupt_partial_kept(self, monkeypatch):
        plan = make_plan("pure_vector")
        frames, persist = run(
            monkeypatch, plan,
            retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            llm=FakeLLMStream(raises=LLMStreamInterrupted("半截答案", "断流"), partial="半截答案"),
            session_id="s3",
        )
        assert all_deltas(frames) == "半截答案"
        assert persist.records[0]["assistant_content"] == "半截答案"
        assert persist.records[0]["error"].startswith("stream_interrupted")

    def test_structured_path_survives_llm_outage(self, monkeypatch):
        """LLM 全挂时结构化链路照样回答（不经过 LLM）。"""
        plan = make_plan("pure_structured", qa_hit=make_qa_pair(), similarity=0.98)
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(
            llm_client=FakeLLMStream(raises=LLMError("no key")),
            retrieve_fn=fake_retrieve_fn([]),
        )
        from app.core.sse import parse_sse_frames

        frames = parse_sse_frames("".join(engine.stream(plan.query)))
        assert "六个月" in all_deltas(frames)
        assert get_event(frames, "done")["cached"] is False


class TestHybridContext:
    def test_references_only_hybrid(self, monkeypatch):
        """无精确命中、只有同类参考 + 法条：两类资料都进上下文。

        注意让法条命中与参考的法条引用**不同条号**，否则去重会按高分保留法条。
        """
        plan = make_plan("hybrid", references=[make_qa_pair(7)])
        llm = FakeLLMStream()
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(llm_client=llm, retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=2.0, article="第二十条")]))
        from app.core.sse import parse_sse_frames

        frames = parse_sse_frames("".join(engine.stream(plan.query)))
        user_msg = llm.calls[0]["messages"][-1]["content"]
        assert "【同类问题参考】" in user_msg
        assert "【法条资料】" in user_msg
        sources = get_event(frames, "sources")["sources"]
        assert {s["kind"] for s in sources} == {"qa", "law"}

    def test_exact_hit_plus_laws(self, monkeypatch):
        plan = make_plan("hybrid", qa_hit=make_qa_pair(), similarity=0.93)
        llm = FakeLLMStream()
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(llm_client=llm, retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=2.0)]))
        from app.core.sse import parse_sse_frames

        list(engine.stream(plan.query))
        user_msg = llm.calls[0]["messages"][-1]["content"]
        assert user_msg.index("【结论参考】") < user_msg.index("【法条资料】")


class TestRetrievePathPolicy:
    """检索选路策略：仅 decompose 走多路分检，其余原句单查询。

    依据：P6 消融——多路对多跳 +8.3pt，但同义改写句作第二投票人
    会稀释锚点排序（单跳 −13.3pt，q04/q15 从满分掉到 0）。
    """

    def test_direct_rewrite_uses_single_query_even_with_subqueries(self, monkeypatch):
        intent = make_intent(sub_queries=["改写后的子查询一", "改写后的子查询二"], rewrite_type="direct")
        plan = make_plan("pure_vector", intent=intent)
        retrieve = fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)])
        run(monkeypatch, plan, retrieve=retrieve)
        # direct：只调 1 次且用原句，不按 sub_queries 分检
        assert len(retrieve.calls) == 1
        assert retrieve.calls[0]["question"] == plan.query

    def test_decompose_uses_multi_path(self, monkeypatch):
        intent = make_intent(
            sub_queries=["经济补偿计算标准", "代通知金规定"],
            rewrite_type="decompose",
        )
        plan = make_plan("pure_vector", intent=intent)
        retrieve = fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)])
        run(monkeypatch, plan, retrieve=retrieve)
        # decompose：原句锚点 + 2 条子查询，各检索一路
        questions = [c["question"] for c in retrieve.calls]
        assert set(questions) == {plan.query, "经济补偿计算标准", "代通知金规定"}
        assert len(retrieve.calls) == 3

    def test_multi_issue_question_gets_topk_bonus(self, monkeypatch):
        """多诉求单点问句：最终 topK 放大，补回被截断在 6-13 名的第二处规定。"""
        from config import get_settings

        plan = make_plan("pure_vector", query="公司拖欠工资，我能不能直接走人还要补偿？")
        retrieve = fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)])
        run(monkeypatch, plan, retrieve=retrieve)
        settings = get_settings()
        expected = (
            settings.retrieval.topk_for("medium")
            + settings.retrieval_multi_issue_topk_bonus
        )
        assert retrieve.calls[0]["topk"] == expected

    def test_single_issue_question_keeps_default_topk(self, monkeypatch):
        """单一诉求：不放大，避免无谓增加上下文。"""
        plan = make_plan("pure_vector", query="试用期最长可以约定多久？")
        retrieve = fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)])
        run(monkeypatch, plan, retrieve=retrieve)
        assert retrieve.calls[0]["topk"] is None
