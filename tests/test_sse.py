"""AnswerEngine SSE 事件流测试（阶段 5，全离线）。

用假 LLM 流客户端 + 假检索 + 假落库驱动真实 :class:`AnswerEngine`，
断言事件顺序、内容与容错行为；``analyze_query`` 被 monkeypatch 成
预制的 :class:`QueryPlan`，把变量收窄到"生成链路本身"。
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from app.core.sse import parse_sse_frames
from app.generation import stream as stream_mod
from app.generation.llm_client import LLMError, LLMStreamInterrupted, StreamChunk
from app.generation.citations import append_citation_note, check_citations
from app.generation.stream import AnswerEngine, StreamResult
from app.orchestration.answer_cache import CachedAnswer
from app.orchestration.models import IntentResult, RouteDecision
from app.orchestration.qa_cache import QALookupResult, QAPair
from app.orchestration.service import QueryPlan

LAW = "中华人民共和国劳动合同法"


# ---------------------------------------------------------------------------
# 假件工厂
# ---------------------------------------------------------------------------
def make_intent(**kw) -> IntentResult:
    base = dict(intent="labor_law", rewrite_type="direct", sub_queries=["试用期最长多久"], complexity="medium", source="llm")
    base.update(kw)
    return IntentResult(**base)


def make_route(route: str, **kw) -> RouteDecision:
    base = dict(route=route, kb_ids=["kb_labor"])
    base.update(kw)
    return RouteDecision(**base)


def make_plan(route: str, *, qa_hit=None, references=None, similarity=None, intent=None, query="试用期最长多久") -> QueryPlan:
    intent = intent or make_intent()
    return QueryPlan(
        query=query,
        intent=intent,
        route=make_route(route),
        qa=QALookupResult(hit=qa_hit, references=references or [], similarity=similarity, source="mysql_bm25_cosine" if qa_hit else "none"),
        sub_queries=list(intent.sub_queries) or [query],
        cache_query=query,
    )


def make_qa_pair(qid: int = 1) -> QAPair:
    return QAPair(
        id=qid,
        category="试用期",
        question_std="试用期最长可以约定多久？",
        question_variants=["试用期能定多久"],
        answer="结论：试用期最长不得超过六个月。依据：《中华人民共和国劳动合同法》第十九条……",
        law_refs=[{"law": LAW, "article": "第十九条"}],
        kb_id="kb_labor",
    )


def make_hit(chunk_id: str, score: float, rerank: float | None = None, article: str = "第十九条") -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=chunk_id,
        parent_id=chunk_id,
        kb_id="kb_labor",
        doc_id="d1",
        content=f"{article} 试用期最长不得超过六个月……",
        title=f"{LAW}·第四章·{article}",
        article_no=article,
        chapter_path="第四章",
        law_name=LAW,
        source_file="labor_contract.md",
        score=score,
        rerank_score=rerank,
        parent_content=f"{LAW} {article} 劳动合同期限三个月以上不满一年的，试用期不得超过一个月……",
    )


class FakeLLMStream:
    """假流式客户端：按预设 chunk 序列产出。"""

    def __init__(self, chunks: list[StreamChunk] | None = None, *, raises: Exception | None = None, partial: str = ""):
        self.chunks = chunks if chunks is not None else [StreamChunk(text="结论："), StreamChunk(text="六个月。"), StreamChunk(usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})]
        self.raises = raises
        self.partial = partial
        self.calls: list[dict] = []

    def chat_stream(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.raises is not None:
            if self.partial:
                # 真实 SDK 的行为：partial 已作为 delta 产出，之后才断流
                yield StreamChunk(text=self.partial)
                raise LLMStreamInterrupted(self.partial, "上游断流")
            raise self.raises
        yield from self.chunks


class FakeRetrieveResult:
    def __init__(self, hits, reranked=True):
        self.hits = hits
        self.reranked = reranked


def fake_retrieve_fn(hits, *, raises: Exception | None = None):
    calls = []

    def _fn(question, *, kb_ids=None, complexity="medium", settings=None, **kw):
        calls.append({"question": question, "kb_ids": kb_ids, "complexity": complexity})
        if raises is not None:
            raise raises
        return FakeRetrieveResult(hits)

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


def fake_persist():
    records = []

    def _fn(session_id, **kwargs):
        records.append({"session_id": session_id, **kwargs})
        return True

    _fn.records = records  # type: ignore[attr-defined]
    return _fn


def run_engine(monkeypatch, plan, *, llm=None, retrieve=None, persist=None, history=None, session_id=None, kb_ids=None):
    """跑一遍 AnswerEngine，返回 (frames, llm, retrieve_fn, persist_fn)。"""
    monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
    llm = llm or FakeLLMStream()
    retrieve = retrieve or fake_retrieve_fn([])
    persist = persist or fake_persist()
    engine = AnswerEngine(llm_client=llm, retrieve_fn=retrieve, persist_fn=persist)
    frames = parse_sse_frames("".join(engine.stream(plan.query, history=history, selected_kb_ids=kb_ids, session_id=session_id)))
    return frames, llm, retrieve, persist


def event_names(frames):
    return [name for name, _ in frames]


def get_event(frames, name):
    for n, data in frames:
        if n == name:
            return data
    return None


def all_deltas(frames):
    return "".join(d["text"] for n, d in frames if n == "delta")


# ---------------------------------------------------------------------------
# 事件顺序
# ---------------------------------------------------------------------------
class TestEventOrder:
    def test_pure_vector_full_sequence(self, monkeypatch):
        plan = make_plan("pure_vector")
        frames, llm, retrieve, _ = run_engine(
            monkeypatch, plan, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=3.2)])
        )
        names = event_names(frames)
        assert names[0] == "stage" and names[1] == "meta"
        assert "retrieval" in [d["stage"] for n, d in frames if n == "stage"]
        assert "rerank" in [d["stage"] for n, d in frames if n == "stage"]
        assert "generate" in [d["stage"] for n, d in frames if n == "stage"]
        # sources 在 generate stage 之前、第一个 delta 之前
        assert names.index("sources") < names.index("generate" if False else "delta")
        assert names.count("delta") >= 2
        assert names[-1] == "done"
        assert all_deltas(frames) == "结论：六个月。"

    def test_meta_fields(self, monkeypatch):
        plan = make_plan("pure_vector", intent=make_intent(rewrite_type="hyde", complexity="simple"))
        frames, *_ = run_engine(monkeypatch, plan, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]))
        meta = get_event(frames, "meta")
        assert set(meta) >= {"message_id", "intent", "route", "rewrite_type", "complexity", "cached", "degraded"}
        assert meta["route"] == "pure_vector"
        assert meta["intent"] == "labor_law"
        assert meta["degraded"] is False  # LLM 路径未降级

    def test_degraded_meta_when_rules_fallback(self, monkeypatch):
        plan = make_plan("pure_vector", intent=make_intent(source="rules"))
        frames, *_ = run_engine(monkeypatch, plan, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]))
        assert get_event(frames, "meta")["degraded"] is True

    def test_done_fields(self, monkeypatch):
        plan = make_plan("pure_vector")
        frames, *_ = run_engine(monkeypatch, plan, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]))
        done = get_event(frames, "done")
        assert set(done) >= {"ttft_ms", "latency_ms", "token_usage", "sources_count", "cached"}
        assert done["token_usage"]["total_tokens"] == 120
        assert done["sources_count"] == 1
        assert done["ttft_ms"] >= 0 and done["latency_ms"] >= done["ttft_ms"]

    def test_sources_content_and_order(self, monkeypatch):
        plan = make_plan("pure_vector")
        frames, *_ = run_engine(
            monkeypatch, plan,
            retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0), make_hit("c2", 0.6, rerank=5.0, article="第二十条")]),
        )
        sources = get_event(frames, "sources")["sources"]
        assert [s["article_no"] for s in sources] == ["第二十条", "第十九条"]  # 重排分降序
        assert sources[0]["law"] == LAW
        assert sources[0]["chapter_path"] == "第四章"
        assert "试用期" in sources[0]["snippet"]
        assert sources[0]["kind"] == "law"

    def test_retrieval_uses_sub_queries_and_kbs(self, monkeypatch):
        """decompose 意图走多路分检（原句锚点 + 子查询），并带 kb 过滤。

        选路策略：仅 rewrite_type=decompose 走多路（direct 等同义改写
        不拆，见 TestRetrievePathPolicy）；本用例意图默认 direct，
        显式指定 decompose 覆盖多路分支。
        """
        intent = make_intent(
            sub_queries=["试用期最长多久", "劳动合同法 试用期 六个月"],
            rewrite_type="decompose",
        )
        plan = make_plan("pure_vector", intent=intent)
        _, _, retrieve, _ = run_engine(monkeypatch, plan, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]))
        # 原句已在子查询列表里（第 1 条），不会重复加锚点
        assert {c["question"] for c in retrieve.calls} == {
            "试用期最长多久", "劳动合同法 试用期 六个月",
        }
        assert all(c["kb_ids"] == ["kb_labor"] for c in retrieve.calls)


# ---------------------------------------------------------------------------
# 引用在线校验
# ---------------------------------------------------------------------------
class TestCitationCheck:
    def test_unverified_citation_gets_note_and_meta(self, monkeypatch):
        """答案引用了来源外条号 → 回答尾部补注 + meta.citation_check 标记。"""
        plan = make_plan("pure_vector")
        llm = FakeLLMStream(chunks=[StreamChunk(text="依据《中华人民共和国劳动合同法》第三十九条，用人单位可以解除。")])
        frames, *_ = run_engine(
            monkeypatch, plan, llm=llm, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)])
        )
        meta = [d for n, d in frames if n == "meta"][-1]
        # 迟到 meta 必须带齐全部字段，否则前端整体覆盖会清空元信息
        assert set(meta) >= {
            "message_id", "intent", "route", "rewrite_type", "complexity", "cached", "degraded",
            "citation_check",
        }
        check = meta["citation_check"]
        assert check["ok"] is False
        assert check["unverified"] == ["劳动合同法#第三十九条"]
        assert "引用校验：" in all_deltas(frames)
        assert all_deltas(frames).endswith("第三十九条")

    def test_verified_citation_no_note(self, monkeypatch):
        plan = make_plan("pure_vector")
        llm = FakeLLMStream(chunks=[StreamChunk(text="依据《劳动合同法》第十九条，试用期最长六个月。")])
        frames, *_ = run_engine(
            monkeypatch, plan, llm=llm, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)])
        )
        check = [d for n, d in frames if n == "meta"][-1]["citation_check"]
        assert check["ok"] is True
        assert check["verified"] == ["劳动合同法#第十九条"]
        assert "引用校验：" not in all_deltas(frames)

    def test_cached_replay_revalidates(self):
        cached = CachedAnswer(
            question="试用期最长多久",
            answer="依据《劳动合同法》第三十九条。",
            sources=[{"law": LAW, "article_no": "第十九条", "kind": "law"}],
            route="pure_vector",
            intent="labor_law",
        )
        engine = AnswerEngine(llm_client=FakeLLMStream(), retrieve_fn=fake_retrieve_fn([]), persist_fn=fake_persist())
        frames = parse_sse_frames(
            "".join(engine._answer_replay(cached, StreamResult(), engine.settings.generation, time.perf_counter(), "m1"))
        )
        meta = frames[0][1]
        assert meta["cached"] is True
        assert meta["citation_check"]["ok"] is False
        assert "引用校验：" in all_deltas(frames)

    def test_cached_replay_never_double_notes(self):
        base = "依据《劳动合同法》第三十九条。"
        sources = [{"law": LAW, "article_no": "第十九条", "kind": "law"}]
        annotated = append_citation_note(base, check_citations(base, sources))
        cached = CachedAnswer(question="q", answer=annotated, sources=sources, route="pure_vector")
        engine = AnswerEngine(llm_client=FakeLLMStream(), retrieve_fn=fake_retrieve_fn([]), persist_fn=fake_persist())
        frames = parse_sse_frames(
            "".join(engine._answer_replay(cached, StreamResult(), engine.settings.generation, time.perf_counter(), "m1"))
        )
        assert all_deltas(frames).count("引用校验：") == 1

    def test_cache_stores_answer_without_note(self, monkeypatch):
        """补注只用于展示：写缓存前剔除，否则回放会固化提示。"""
        stored = {}

        class FakeCache:
            def store(self, query, **kwargs):
                stored.update(kwargs)

            def lookup(self, *a, **kw):
                return None

        plan = make_plan("pure_vector")
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(
            llm_client=FakeLLMStream(chunks=[StreamChunk(text="依据《劳动合同法》第三十九条。")]),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            persist_fn=fake_persist(),
            answer_cache=FakeCache(),
        )
        list(engine.stream(plan.query, session_id="s1"))
        assert "引用校验：" not in stored["answer"]
        assert stored["answer"] == "依据《劳动合同法》第三十九条。"


class TestPartialCacheWrite:
    """客户端断流（GeneratorExit）时不得把半截回答写进答案缓存。"""

    @staticmethod
    def _spy_cache(stored: list):
        class SpyCache:
            def lookup(self, *a, **kw):
                return None

            def store(self, *a, **kw):
                stored.append({"args": a, "kwargs": kw})

        return SpyCache()

    def test_disconnect_after_first_delta_does_not_cache(self, monkeypatch):
        stored: list = []
        plan = make_plan("pure_vector")
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(
            llm_client=FakeLLMStream(
                chunks=[StreamChunk(text="结论：六个月。"), StreamChunk(text="依据：第十九条")]
            ),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            persist_fn=fake_persist(),
            answer_cache=self._spy_cache(stored),
        )
        gen = engine.stream(plan.query, session_id="s1")
        for frame in gen:
            if frame.startswith("event: delta"):
                break
        gen.close()  # 模拟客户端在首个 delta 后断开
        assert stored == [], f"断流后写入了缓存：{stored}"

    def test_completed_answer_is_cached(self, monkeypatch):
        stored: list = []
        plan = make_plan("pure_vector")
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(
            llm_client=FakeLLMStream(
                chunks=[StreamChunk(text="结论：六个月。"), StreamChunk(text="依据：第十九条")]
            ),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            persist_fn=fake_persist(),
            answer_cache=self._spy_cache(stored),
        )
        list(engine.stream(plan.query, session_id="s1"))
        assert len(stored) == 1
        assert stored[0]["kwargs"]["answer"] == "结论：六个月。依据：第十九条"


# ---------------------------------------------------------------------------
# 结构化回放
# ---------------------------------------------------------------------------
class TestStructuredReplay:
    def test_replay_sequence_and_cached_flag(self, monkeypatch):
        plan = make_plan("pure_structured", qa_hit=make_qa_pair(), similarity=0.97)
        frames, llm, retrieve, _ = run_engine(monkeypatch, plan)
        names = event_names(frames)
        assert "meta" in names and "sources" in names and names[-1] == "done"
        assert llm.calls == []  # 不走 LLM 全文生成
        assert retrieve.calls == []  # 不碰 Milvus（验收硬标准）
        assert all_deltas(frames) == make_qa_pair().answer
        assert get_event(frames, "done")["cached"] is False  # 来源 mysql_bm25_cosine
        assert get_event(frames, "done")["sources_count"] == 1
        assert get_event(frames, "sources")["sources"][0]["kind"] == "qa"

    def test_cached_true_when_redis_exact(self, monkeypatch):
        qa = make_qa_pair()
        plan = QueryPlan(
            query=qa.question_std, intent=make_intent(), route=make_route("pure_structured"),
            qa=QALookupResult(hit=qa, source="redis_exact", similarity=1.0), sub_queries=[qa.question_std],
        )
        frames, *_ = run_engine(monkeypatch, plan)
        assert get_event(frames, "done")["cached"] is True

    def test_replay_is_chunked(self, monkeypatch):
        plan = make_plan("pure_structured", qa_hit=make_qa_pair())
        frames, *_ = run_engine(monkeypatch, plan)
        deltas = [d["text"] for n, d in frames if n == "delta"]
        assert len(deltas) > 1  # 打字机分片，不是一整块
        assert all(len(t) <= 30 for t in deltas)

    def test_replay_records_ttft(self, monkeypatch):
        """回放路径也要记录 TTFT（缓存命中速度是结构化链路核心指标）。"""
        plan = make_plan("pure_structured", qa_hit=make_qa_pair())
        frames, *_ = run_engine(monkeypatch, plan)
        done = get_event(frames, "done")
        assert done["ttft_ms"] is not None and done["ttft_ms"] >= 0


# ---------------------------------------------------------------------------
# 容错
# ---------------------------------------------------------------------------
class TestFaultTolerance:
    def test_llm_error_emits_error_event(self, monkeypatch):
        plan = make_plan("pure_vector")
        frames, *_ = run_engine(
            monkeypatch, plan,
            retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            llm=FakeLLMStream(raises=LLMError("限流")),
        )
        names = event_names(frames)
        assert "error" in names
        err = get_event(frames, "error")
        assert err["code"] == "llm_error" and err["retriable"] is True
        assert names[-1] == "done"  # error 后仍走完协议
        assert get_event(frames, "done")["token_usage"] == {}

    def test_stream_interrupted_keeps_partial(self, monkeypatch):
        plan = make_plan("pure_vector")
        frames, *_ = run_engine(
            monkeypatch, plan,
            retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            llm=FakeLLMStream(raises=LLMStreamInterrupted("已产出半截", "上游断流"), partial="已产出半截"),
        )
        assert all_deltas(frames) == "已产出半截"  # partial 已作为 delta 产出
        err = get_event(frames, "error")
        assert err["code"] == "stream_interrupted" and err["retriable"] is True

    def test_client_disconnect_persists_without_error(self, monkeypatch):
        """第 4 层：客户端断开 → 落库 partial，不发 error 帧。"""
        plan = make_plan("pure_vector")
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        llm = FakeLLMStream()
        retrieve = fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)])
        persist = fake_persist()
        engine = AnswerEngine(llm_client=llm, retrieve_fn=retrieve, persist_fn=persist)
        gen = engine.stream(plan.query, session_id="sess-1")
        frames = []
        for frame in gen:
            frames.append(frame)
            if frame.startswith("event: delta"):
                break  # 客户端在首个 delta 后断开
        gen.close()  # 触发 GeneratorExit → finally 落库
        assert len(persist.records) == 1
        rec = persist.records[0]
        assert rec["session_id"] == "sess-1"
        assert rec["assistant_content"] == "结论："
        assert rec["error"] is None
        assert rec["route"] == "pure_vector"

    def test_milvus_failure_degrades_with_qa_reference(self, monkeypatch):
        """第 5 层：Milvus 故障 + 混合链路有同类参考 → 降级回答（不引导话术）。"""
        plan = make_plan("hybrid", qa_hit=None, references=[make_qa_pair(2)])
        frames, llm, *_ = run_engine(
            monkeypatch, plan,
            retrieve=fake_retrieve_fn([], raises=ConnectionError("milvus down")),
        )
        metas = [d for n, d in frames if n == "meta"]
        assert metas[-1]["degraded"] == "vector_unavailable"
        assert llm.calls  # 仍走生成（同类参考作上下文）
        assert get_event(frames, "done")["sources_count"] >= 1

    def test_milvus_failure_without_any_source_falls_back(self, monkeypatch):
        plan = make_plan("pure_vector")
        frames, llm, *_ = run_engine(
            monkeypatch, plan,
            retrieve=fake_retrieve_fn([], raises=ConnectionError("milvus down")),
        )
        assert llm.calls == []  # 无资料不调 LLM
        assert "暂未在所选知识库中找到" in all_deltas(frames)
        metas = [d for n, d in frames if n == "meta"]
        assert metas[-1]["degraded"] == "vector_unavailable"  # 降级标记在追加的 meta 上

    def test_no_data_no_llm(self, monkeypatch):
        plan = make_plan("pure_vector")
        frames, llm, *_ = run_engine(monkeypatch, plan, retrieve=fake_retrieve_fn([]))
        assert llm.calls == []
        assert "暂未在所选知识库中找到" in all_deltas(frames)
        assert get_event(frames, "done")["sources_count"] == 0

    def test_greeting_no_data_source(self, monkeypatch):
        plan = make_plan("greeting", intent=make_intent(intent="greeting", sub_queries=[]))
        frames, llm, retrieve, _ = run_engine(monkeypatch, plan)
        assert llm.calls == [] and retrieve.calls == []
        assert "法律咨询助手" in all_deltas(frames)
        assert get_event(frames, "done")["sources_count"] == 0


# ---------------------------------------------------------------------------
# 落库
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_persist_once_on_success(self, monkeypatch):
        plan = make_plan("pure_vector")
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        persist = fake_persist()
        engine = AnswerEngine(
            llm_client=FakeLLMStream(),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            persist_fn=persist,
        )
        list(engine.stream(plan.query, session_id="sess-9"))
        assert len(persist.records) == 1
        rec = persist.records[0]
        assert rec["assistant_content"] == "结论：六个月。"
        assert rec["intent"] == "labor_law"
        assert rec["rewrite_type"] == "direct"
        assert rec["route"] == "pure_vector"
        assert rec["ttft_ms"] is not None
        assert rec["token_usage"]["total_tokens"] == 120
        assert rec["error"] is None
        assert rec["cached"] is False

    def test_no_session_no_persist(self, monkeypatch):
        plan = make_plan("pure_vector")
        _, _, _, persist = run_engine(monkeypatch, plan, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]))
        assert persist.records == []

    def test_persist_failure_never_breaks_stream(self, monkeypatch):
        plan = make_plan("pure_vector")

        def bad_persist(session_id, **kwargs):
            raise RuntimeError("db down")

        frames, *_ = run_engine(
            monkeypatch, plan,
            retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            persist=bad_persist,
            session_id="sess-x",
        )
        assert event_names(frames)[-1] == "done"  # 流完整结束

    def test_error_persisted_with_error_field(self, monkeypatch):
        plan = make_plan("pure_vector")
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        persist = fake_persist()
        engine = AnswerEngine(
            llm_client=FakeLLMStream(raises=LLMError("限流")),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            persist_fn=persist,
        )
        list(engine.stream(plan.query, session_id="sess-e"))
        assert len(persist.records) == 1
        assert persist.records[0]["error"] and persist.records[0]["error"].startswith("llm_error")


# ---------------------------------------------------------------------------
# 上下文装配进 LLM
# ---------------------------------------------------------------------------
class TestContextAssembly:
    def test_context_and_history_in_messages(self, monkeypatch):
        plan = make_plan("pure_vector")
        history = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "请讲"}, {"role": "user", "content": "试用期"}]
        _, llm, *_ = run_engine(
            monkeypatch, plan,
            retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=3.0)]),
            history=history,
        )
        messages = llm.calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert "只能依据" in messages[0]["content"]
        user_msg = messages[-1]["content"]
        assert "【参考资料】" in user_msg and "【资料1】" in user_msg
        assert LAW in user_msg  # 法名不丢失
        assert "第十九条" in user_msg
        assert [m["content"] for m in messages[1:-1]] == ["你好", "请讲", "试用期"]

    def test_hybrid_puts_qa_conclusion_first(self, monkeypatch):
        plan = make_plan("hybrid", qa_hit=make_qa_pair(), similarity=0.95)
        _, llm, *_ = run_engine(
            monkeypatch, plan,
            retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=3.0)]),
        )
        user_msg = llm.calls[0]["messages"][-1]["content"]
        assert user_msg.index("【结论参考】") < user_msg.index("【法条资料】")
        assert "试用期最长不得超过六个月" in user_msg

    def test_generation_params(self, monkeypatch):
        plan = make_plan("pure_vector")
        _, llm, *_ = run_engine(monkeypatch, plan, retrieve=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]))
        call = llm.calls[0]
        assert call["temperature"] == 0.3
        assert call["max_tokens"] == 2048
