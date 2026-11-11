"""LA-O4：LLM 治理与全链路 trace（离线）。

覆盖：
1. 配置非法 fail-closed（base_url/model/fallback）；
2. 错误统一分类 + 并发信号量 + 熔断打开后快速失败；
3. trace 记录完整阶段顺序，且不写密钥/问题/回答正文；
4. trace 查询接口只能看自己的请求（他人 404）。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app.api.traces as traces_mod
from app.core.access import AccessScope, KbGrant
from app.core.security import CurrentUser
from app.core.trace import Trace
from app.generation import stream as stream_mod
from app.generation.llm_client import LLMClient, LLMError, StreamChunk, _classify_error
from app.main import app
from tests.fakes_db import FakeDB, override_deps
from tests.test_access_scope import LONG_ANSWER, FakeRedis
from tests.test_sse import (
    FakeLLMStream,
    fake_persist,
    fake_retrieve_fn,
    make_hit,
    make_plan,
)

from config import ConfigError, Settings, validate_llm_config


class _APITimeoutError(Exception):
    pass


class _RateLimitError(Exception):
    pass


def _settings(**kwargs) -> Settings:
    kwargs.setdefault("_env_file", None)
    return Settings(**kwargs)


class TestConfigFailClosed:
    def test_non_http_base_url_rejected(self):
        with pytest.raises(ConfigError, match="LLM_BASE_URL"):
            validate_llm_config(_settings(llm_base_url="ftp://example.com/v1"))

    def test_empty_model_rejected(self):
        with pytest.raises(ConfigError, match="LLM_DEFAULT_MODEL"):
            validate_llm_config(_settings(llm_default_model=""))

    def test_fallback_same_as_default_rejected(self):
        with pytest.raises(ConfigError, match="LLM_FALLBACK_MODEL"):
            validate_llm_config(
                _settings(llm_default_model="deepseek-chat", llm_fallback_model="deepseek-chat")
            )

    def test_defaults_pass(self):
        validate_llm_config(_settings())


class TestErrorClassification:
    def test_timeout_and_rate_limit(self):
        assert _classify_error(_APITimeoutError("slow")) == "timeout"
        assert _classify_error(_RateLimitError("429 too many")) == "rate_limit"

    def test_unknown_is_server(self):
        assert _classify_error(RuntimeError("boom")) == "server"


class TestCircuitBreaker:
    def _client(self, **kwargs) -> LLMClient:
        return LLMClient(
            api_key="test-key",
            base_url="http://localhost:1/v1",
            model="m",
            max_retries=0,
            circuit_fail_threshold=2,
            circuit_cooldown_s=60,
            **kwargs,
        )

    def test_opens_after_threshold_and_skips_upstream(self):
        client = self._client()
        calls = {"n": 0}

        class _Completions:
            def create(self, **_kw):
                calls["n"] += 1
                raise _APITimeoutError("timeout")

        client._client = type(
            "FakeOpenAI", (), {"chat": type("Chat", (), {"completions": _Completions()})()}
        )()

        for _ in range(2):
            with pytest.raises(LLMError) as exc:
                client.chat([{"role": "user", "content": "hi"}])
            assert exc.value.kind == "timeout"
        # 熔断已打开：第三次不再打到上游
        with pytest.raises(LLMError) as exc:
            client.chat([{"role": "user", "content": "hi"}])
        assert exc.value.kind == "circuit_open"
        assert calls["n"] == 2


class TestTraceRedaction:
    def test_events_ordered_and_secrets_redacted(self):
        trace = Trace(trace_id="t1", user_id=7)
        with trace.stage("scope", api_key="sk-secret", kb_ids=["kb_labor"]):
            pass
        trace.record("retrieve", latency_ms=12, hits=3, kb_versions={"kb_labor": {"corpus_version": 2}})
        trace.record("done", ok=True, long="x" * 500)

        data = trace.to_dict()
        assert [e["stage"] for e in data["events"]] == ["scope", "retrieve", "done"]
        assert data["events"][0]["api_key"] == "***"
        assert data["events"][1]["kb_versions"]["kb_labor"]["corpus_version"] == 2
        assert len(data["events"][2]["long"]) <= 161
        assert data["trace_id"] == "t1"

    def test_cost_fields_convert_tokens(self):
        from app.core.trace import cost_fields

        cost = cost_fields(
            {"prompt_tokens": 1000, "completion_tokens": 500},
            prompt_per_1k=0.002,
            completion_per_1k=0.008,
        )["cost"]
        assert cost["prompt"] == 0.002
        assert cost["completion"] == 0.004
        assert cost["total"] == 0.006
        assert cost_fields(None, prompt_per_1k=0.002, completion_per_1k=0.008)["cost"]["total"] == 0.0


class TestEngineTrace:
    def test_full_pipeline_trace_without_content(self, monkeypatch, settings_factory):
        plan = make_plan("hybrid")
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = stream_mod.AnswerEngine(
            llm_client=FakeLLMStream(chunks=[StreamChunk(text="结论：可以申请支付令。")]),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.9, rerank=2.0)]),
            persist_fn=fake_persist(),
            settings=settings_factory(cache_answer_ttl=3600, trace_store_enabled=False),
        )
        scope = AccessScope.for_grants(
            [KbGrant(kb_id="kb_labor", corpus_version=1, index_version=1)],
            tenant_id="default",
            user_id=7,
        )
        trace = Trace(user_id=7)
        frames = list(engine.stream("申请支付令需要什么材料", scope=scope, trace=trace))

        stages = [e["stage"] for e in trace.events]
        for required in (
            "request",
            "cache_lookup",
            "intent",
            "rewrite",
            "route",
            "retrieve",
            "rerank",
            "generate",
            "citation_check",
            "answer_cache_write",
            "persistence",
            "done",
        ):
            assert required in stages, required
        assert (
            stages.index("intent")
            < stages.index("rewrite")
            < stages.index("route")
            < stages.index("retrieve")
            < stages.index("generate")
        )
        # 独立成本字段：intent 与 generate 两个 LLM 阶段都要有
        for event in trace.events:
            if event["stage"] in ("intent", "generate"):
                assert "cost" in event, event["stage"]
                assert event["cost"]["currency"] == "CNY"
                assert set(event["cost"]) >= {
                    "prompt_tokens",
                    "completion_tokens",
                    "prompt",
                    "completion",
                    "total",
                }

        meta = [f for f in frames if f.startswith("event: meta")][0]
        assert trace.trace_id in meta
        dumped = json.dumps(trace.to_dict(), ensure_ascii=False)
        # 问题/回答正文与密钥都不能进 trace
        assert "申请支付令需要什么材料" not in dumped
        assert "结论：可以申请支付令" not in dumped
        assert "api_key" not in dumped

    def test_answer_cache_replay_marks_cached(self, monkeypatch, settings_factory):
        from app.orchestration.answer_cache import AnswerCache

        question = "申请支付令需要准备哪些材料"
        # 缓存键取自改写后的 plan.cache_query，桩 plan 必须与真实问题一致
        monkeypatch.setattr(
            stream_mod, "analyze_query", lambda *a, **kw: make_plan("hybrid", query=question)
        )
        settings = settings_factory(cache_answer_ttl=3600, trace_store_enabled=False)
        cache = AnswerCache(redis=FakeRedis(), settings=settings)
        engine = stream_mod.AnswerEngine(
            llm_client=FakeLLMStream(chunks=[StreamChunk(text=LONG_ANSWER)]),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.9, rerank=2.0)]),
            persist_fn=fake_persist(),
            answer_cache=cache,
            settings=settings,
        )
        scope = AccessScope.for_grants([KbGrant(kb_id="kb_labor")], tenant_id="default", user_id=1)
        list(engine.stream(question, scope=scope))
        trace2 = Trace(user_id=1)
        list(engine.stream(question, scope=scope, trace=trace2))

        cache_events = [e for e in trace2.events if e["stage"] == "cache_lookup"]
        assert cache_events and cache_events[0]["hit"] is True


class TestTraceEndpoint:
    def test_owner_can_read_other_user_gets_404(self, monkeypatch):
        monkeypatch.setattr(
            traces_mod,
            "load_trace",
            lambda trace_id: {"trace_id": trace_id, "user_id": 1, "events": []},
        )
        db = FakeDB()
        restore = override_deps(
            app, db, user=CurrentUser(id=1, username="alice", role="viewer")
        )
        try:
            client = TestClient(app)
            resp = client.get("/api/traces/abc")
            assert resp.status_code == 200 and resp.json()["trace_id"] == "abc"
        finally:
            restore()

        restore = override_deps(
            app, db, user=CurrentUser(id=2, username="bob", role="viewer")
        )
        try:
            client = TestClient(app)
            assert client.get("/api/traces/abc").status_code == 404
        finally:
            restore()
