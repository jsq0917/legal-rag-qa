"""``POST /api/chat/stream`` 接口契约测试（阶段 5，离线）。

用 TestClient 打真实端点：``AnswerEngine`` 的 LLM/检索/落库全部注入假件，
重点验证 HTTP 层——状态码、SSE 响应头、帧可解析、参数校验、客户端断开语义。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.sse import parse_sse_frames
from app.generation import stream as stream_mod
from app.generation.stream import AnswerEngine
from app.main import app
from app.orchestration.models import IntentResult, RouteDecision
from app.orchestration.qa_cache import QALookupResult, QAPair
from app.orchestration.service import QueryPlan

from tests.test_sse import (
    FakeLLMStream,
    fake_persist,
    fake_retrieve_fn,
    make_hit,
    make_intent,
    make_plan,
    make_qa_pair,
    make_route,
)

LAW = "中华人民共和国劳动合同法"


@pytest.fixture(autouse=True)
def _legacy_scope(monkeypatch):
    """离线测试不连 MySQL：授权范围由 tests/test_access_scope.py 覆盖。"""
    from app.core.access import AccessScope

    monkeypatch.setattr(
        "app.api.chat.prepare_scope", lambda user, session_id, kb_ids: AccessScope.legacy_all()
    )


@pytest.fixture
def wired_client(monkeypatch):
    """把 AnswerEngine 的依赖换成假件，返回 (client, fake_llm, persist)。"""
    plan = make_plan("pure_vector")
    monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
    llm = FakeLLMStream()
    retrieve = fake_retrieve_fn([make_hit("c1", 0.5, rerank=2.5)])
    persist = fake_persist()

    def _factory(**kwargs):
        return AnswerEngine(llm_client=llm, retrieve_fn=retrieve, persist_fn=persist)

    monkeypatch.setattr("app.api.chat.AnswerEngine", _factory)
    return TestClient(app), llm, persist


class TestChatStreamAPI:
    def test_sse_headers_and_status(self, wired_client):
        client, _, _ = wired_client
        resp = client.post("/api/chat/stream", json={"query": "试用期最长多久"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"

    def test_full_frame_sequence(self, wired_client, monkeypatch):
        client, llm, persist = wired_client
        # 会话模式需要登录：覆盖鉴权依赖 + 归属校验，把变量收窄到流本身
        from app.core.security import CurrentUser, get_current_user_optional

        monkeypatch.setattr("app.api.chat.get_owned_session", lambda *a, **kw: object())
        app.dependency_overrides[get_current_user_optional] = lambda: CurrentUser(id=1, username="tester")
        try:
            resp = client.post(
                "/api/chat/stream",
                json={"query": "试用期最长多久", "session_id": "sess-api", "history": [{"role": "user", "content": "你好"}]},
            )
        finally:
            app.dependency_overrides.pop(get_current_user_optional, None)
        frames = parse_sse_frames(resp.text)
        names = [n for n, _ in frames]
        assert names[0] == "stage"
        assert names[1] == "meta"
        assert "sources" in names and "delta" in names
        assert names[-1] == "done"
        text = "".join(d["text"] for n, d in frames if n == "delta")
        assert text == "结论：六个月。"
        # 落库被触发（session_id 传入且 persist 假件记录）
        assert len(persist.records) == 1
        assert persist.records[0]["session_id"] == "sess-api"

    def test_meta_and_done_payload(self, wired_client):
        client, _, _ = wired_client
        frames = parse_sse_frames(client.post("/api/chat/stream", json={"query": "试用期最长多久"}).text)
        meta = next(d for n, d in frames if n == "meta")
        done = next(d for n, d in frames if n == "done")
        assert meta["route"] == "pure_vector"
        assert "message_id" in meta
        assert done["sources_count"] == 1
        assert done["token_usage"]["total_tokens"] == 120

    def test_sources_payload_shape(self, wired_client):
        client, _, _ = wired_client
        frames = parse_sse_frames(client.post("/api/chat/stream", json={"query": "试用期最长多久"}).text)
        sources = next(d for n, d in frames if n == "sources")["sources"]
        src = sources[0]
        assert set(src) >= {"law", "chapter_path", "article_no", "score", "snippet", "kind"}
        assert src["law"] == LAW

    def test_validation_empty_query(self, wired_client):
        client, _, _ = wired_client
        assert client.post("/api/chat/stream", json={"query": ""}).status_code == 422

    def test_validation_missing_query(self, wired_client):
        client, _, _ = wired_client
        assert client.post("/api/chat/stream", json={}).status_code == 422

    def test_query_too_long(self, wired_client):
        client, _, _ = wired_client
        assert client.post("/api/chat/stream", json={"query": "法" * 2001}).status_code == 422

    def test_no_session_no_persist(self, wired_client):
        client, _, persist = wired_client
        client.post("/api/chat/stream", json={"query": "试用期最长多久"})
        assert persist.records == []

    def test_greeting_flow(self, monkeypatch):
        plan = make_plan("greeting", intent=make_intent(intent="greeting", sub_queries=[]))
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        monkeypatch.setattr(
            "app.api.chat.AnswerEngine",
            lambda **kw: AnswerEngine(llm_client=FakeLLMStream(), retrieve_fn=fake_retrieve_fn([])),
        )
        from app.main import app

        client = TestClient(app)
        frames = parse_sse_frames(client.post("/api/chat/stream", json={"query": "你好"}).text)
        assert "".join(d["text"] for n, d in frames if n == "delta") == (
            "你好！我是 LawAnswer 法律咨询助手，可以回答劳动法、民法典相关的法律问题。"
            "比如“试用期最长可以约定多久”“加班费怎么算”“被辞退有经济补偿吗”，"
            "点击上方知识库还可以切换检索范围。请问你想咨询什么？"
        )

    def test_error_frame_on_llm_failure(self, monkeypatch):
        from app.generation.llm_client import LLMError

        plan = make_plan("pure_vector")
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        monkeypatch.setattr(
            "app.api.chat.AnswerEngine",
            lambda **kw: AnswerEngine(
                llm_client=FakeLLMStream(raises=LLMError("限流")),
                retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            ),
        )
        from app.main import app

        client = TestClient(app)
        resp = client.post("/api/chat/stream", json={"query": "试用期最长多久"})
        frames = parse_sse_frames(resp.text)
        names = [n for n, _ in frames]
        assert "error" in names and names[-1] == "done"
        err = next(d for n, d in frames if n == "error")
        assert err["code"] == "llm_error" and err["retriable"] is True

    def test_structured_replay_no_milvus(self, monkeypatch):
        """验收硬标准：纯结构化问题不产生检索调用（API 层复验）。"""
        plan = make_plan("pure_structured", qa_hit=make_qa_pair(), similarity=0.99)
        retrieve = fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)])  # 若被调用则会记 calls
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        monkeypatch.setattr(
            "app.api.chat.AnswerEngine",
            lambda **kw: AnswerEngine(llm_client=FakeLLMStream(), retrieve_fn=retrieve),
        )
        # 测试环境绕过限流（避免连续测试触发限流）
        monkeypatch.setattr("app.api.chat.check_rate_limit", lambda *a, **kw: None)
        from app.main import app

        client = TestClient(app)
        frames = parse_sse_frames(client.post("/api/chat/stream", json={"query": "试用期最长可以约定多久？"}).text)
        assert retrieve.calls == []
        done = next(d for n, d in frames if n == "done")
        assert done["sources_count"] == 1
        assert "六个月" in "".join(d["text"] for n, d in frames if n == "delta")
