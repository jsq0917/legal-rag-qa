"""授权范围 + 缓存版本隔离单测（全离线，LA-O1）。

覆盖三类"错误回放"：

1. 同一问题换知识库 → 不能命中同一份答案；
2. 同一问题换语料/索引版本 → 旧缓存必须失效；
3. 对话框里带历史 / 换租户 → 不能把上一轮或别人的答案回放给当前用户。

另外验证"授权范围为空"时既不检索、也不调 LLM、也不回写缓存。
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app.core.access import (
    CACHE_SCHEMA_VERSION,
    AccessScope,
    KbGrant,
    answer_identity,
    resolve_access_scope,
)
from app.core.sse import parse_sse_frames
from app.generation import stream as stream_mod
from app.generation.prompts import NO_DATA_REPLY
from app.generation.stream import AnswerEngine
from app.orchestration.answer_cache import AnswerCache, answer_key
from app.orchestration.intent import _intent_cache_key
from app.orchestration.qa_cache import QARetriever, cache_key, normalize_question

from app.generation.llm_client import StreamChunk

from tests.test_sse import (
    FakeLLMStream,
    all_deltas,
    fake_persist,
    fake_retrieve_fn,
    get_event,
    make_hit,
    make_plan,
    make_qa_pair,
    make_route,
)

QUESTION = "申请支付令需要准备哪些材料"
LONG_ANSWER = (
    "结论：申请支付令需要准备申请书、身份证明、劳动关系证明与欠薪证据。"
    "依据：《中华人民共和国劳动合同法》第三十条，用人单位拖欠劳动报酬的，"
    "劳动者可以依法向当地人民法院申请支付令。实务提示：材料齐备后向有管辖权"
    "的基层人民法院立案窗口提交，法院应在五日内决定是否受理。"
)


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value
        if ex:
            self.ttls[key] = ex
        return True

    def expire(self, key, ttl):
        if key in self.store:
            self.ttls[key] = ttl
            return True
        return False


def _scope(
    kb_ids=("kb_labor",),
    *,
    version=1,
    index_version=1,
    tenant="default",
    role="editor",
    is_public=True,
) -> AccessScope:
    return AccessScope.for_grants(
        [
            KbGrant(
                kb_id=kb,
                corpus_version=version,
                index_version=index_version,
                is_public=is_public,
                tenant_id=tenant,
            )
            for kb in kb_ids
        ],
        tenant_id=tenant,
        user_id=1,
        role=role,
    )


class TestScopeMath:
    def test_restrict_drops_unauthorized(self):
        scope = _scope(["kb_labor"])
        assert scope.restrict(["kb_labor", "kb_civil"]) == ["kb_labor"]
        assert scope.restrict(None) == ["kb_labor"]
        assert scope.restrict(["kb_civil"]) == []

    def test_public_namespace_shared_private_tenant_isolated(self):
        assert _scope(["kb_labor"]).cache_namespace() == "public"
        private = _scope(["kb_labor"], is_public=False)
        assert private.cache_namespace() == "tenant:default"
        other = _scope(["kb_labor"], is_public=False, tenant="other")
        assert private.cache_token() != other.cache_token()

    def test_cache_token_changes_with_versions_and_kb_set(self):
        base = _scope(["kb_labor"]).cache_token()
        assert base == _scope(["kb_labor"]).cache_token()
        assert base != _scope(["kb_labor", "kb_civil"]).cache_token()
        assert base != _scope(["kb_labor"], version=2).cache_token()
        assert base != _scope(["kb_labor"], index_version=2).cache_token()

    def test_answer_identity_changes_with_model_prompt_as_of(self):
        scope = _scope(["kb_labor"])
        base = answer_identity(scope, model="m1", prompt_version="p1", as_of="2026-11-11")
        assert base == answer_identity(scope, model="m1", prompt_version="p1", as_of="2026-11-11")
        assert base != answer_identity(scope, model="m2", prompt_version="p1", as_of="2026-11-11")
        assert base != answer_identity(scope, model="m1", prompt_version="p2", as_of="2026-11-11")
        assert base != answer_identity(scope, model="m1", prompt_version="p1", as_of="2026-11-12")

    def test_write_permission_matrix(self):
        editor = _scope(["kb_labor"], role="editor")
        viewer = _scope(["kb_labor"], role="viewer")
        admin = _scope(["kb_labor"], role="admin")
        # 公共库属于 other 租户：default 租户的编辑可读但不可写
        other_public = AccessScope.for_grants(
            [KbGrant(kb_id="kb_labor", is_public=True, tenant_id="other")],
            tenant_id="default",
            user_id=1,
            role="editor",
        )
        assert editor.can_write("kb_labor")
        assert not viewer.can_write("kb_labor")
        assert admin.can_write("kb_labor")
        assert not other_public.can_write("kb_labor")
        assert not editor.can_write("kb_civil")

    def test_public_kb_is_read_only_for_other_tenants_even_admin(self):
        """公共库所有租户可读、不可写：跨租户 admin 也不能写。"""
        admin_b = AccessScope.for_grants(
            [KbGrant(kb_id="kb_labor", is_public=True, tenant_id="tenant-a")],
            tenant_id="tenant-b",
            user_id=2,
            role="admin",
        )
        assert admin_b.can_read("kb_labor")
        assert not admin_b.can_write("kb_labor")

    def test_editor_cannot_write_others_private_kb(self):
        """私有库归 owner：同租户 editor 只能写本人私有库，管理员例外。"""
        grant = KbGrant(kb_id="kb_priv", is_public=False, tenant_id="default", owner_id=99)
        other_editor = AccessScope.for_grants(
            [grant], tenant_id="default", user_id=1, role="editor"
        )
        owner_editor = AccessScope.for_grants(
            [grant], tenant_id="default", user_id=99, role="editor"
        )
        tenant_admin = AccessScope.for_grants(
            [grant], tenant_id="default", user_id=1, role="admin"
        )
        assert not other_editor.can_write("kb_priv")
        assert owner_editor.can_write("kb_priv")
        assert tenant_admin.can_write("kb_priv")

    def test_scope_resolution_fails_closed_when_columns_missing(self):
        """缺列/查询失败时不能默认按公共库放开，应返回空范围。"""

        class _Boom:
            def execute(self, *a, **kw):
                raise RuntimeError("Unknown column 'is_public'")

        class _User:
            id = 1
            tenant_id = "default"
            role = "admin"

        scope = resolve_access_scope(_Boom(), _User(), ["kb_labor"])
        assert scope.kb_ids == ()
        assert scope.restrict(None) == []


class TestAnswerCacheIsolation:
    def test_same_query_different_scope_never_replays(self, settings_factory):
        redis = FakeRedis()
        cache = AnswerCache(redis=redis, settings=settings_factory(cache_answer_ttl=3600))
        scope_a = _scope(["kb_labor"])
        scope_b = _scope(["kb_labor"], version=2)
        id_a = answer_identity(scope_a, model="m", prompt_version="p", as_of="2026-11-11")
        id_b = answer_identity(scope_b, model="m", prompt_version="p", as_of="2026-11-11")

        assert cache.store(QUESTION, answer=LONG_ANSWER, route="hybrid", cache_identity=id_a)
        assert cache.lookup(QUESTION, cache_identity=id_a) is not None
        assert cache.lookup(QUESTION, cache_identity=id_b) is None
        assert answer_key(normalize_question(QUESTION), id_a) != answer_key(normalize_question(QUESTION), id_b)

    def test_legacy_value_without_meta_is_ignored(self, settings_factory):
        redis = FakeRedis()
        cache = AnswerCache(redis=redis, settings=settings_factory(cache_answer_ttl=3600))
        norm = normalize_question(QUESTION)
        redis.set(
            answer_key(norm),
            json.dumps({"question": QUESTION, "answer": LONG_ANSWER, "sources": [], "route": "hybrid"}),
        )
        assert cache.lookup(QUESTION) is None, "旧格式缓存必须在上线后失效"


class TestEngineScope:
    def test_cross_version_recomputes_and_same_scope_replays(self, monkeypatch, settings_factory):
        redis = FakeRedis()
        cache = AnswerCache(redis=redis, settings=settings_factory(cache_answer_ttl=3600))
        # 缓存键取改写后的 plan.cache_query，桩 plan 必须与真实问题一致
        plan = make_plan("hybrid", query=QUESTION)
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        llm = FakeLLMStream(chunks=[StreamChunk(text=LONG_ANSWER)])
        engine = AnswerEngine(
            llm_client=llm,
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.9, rerank=2.0)]),
            persist_fn=fake_persist(),
            answer_cache=cache,
            settings=settings_factory(),
        )
        scope_v1 = _scope(["kb_labor"], version=1)
        scope_v2 = _scope(["kb_labor"], version=2)

        frames = parse_sse_frames("".join(engine.stream(QUESTION, scope=scope_v1)))
        assert get_event(frames, "meta")["cached"] is False
        assert len(llm.calls) == 1

        frames = parse_sse_frames("".join(engine.stream(QUESTION, scope=scope_v2)))
        assert get_event(frames, "meta")["cached"] is False
        assert len(llm.calls) == 2, "语料版本变化必须重新生成"

        frames = parse_sse_frames("".join(engine.stream(QUESTION, scope=scope_v1)))
        assert get_event(frames, "meta")["cached"] is True
        assert len(llm.calls) == 2, "同范围同版本应直接回放"

    def test_denied_scope_skips_everything(self, monkeypatch, settings_factory):
        calls = {"analyze": 0}
        plan = make_plan("hybrid")

        def _analyze(*a, **kw):
            calls["analyze"] += 1
            return plan

        monkeypatch.setattr(stream_mod, "analyze_query", _analyze)
        llm = FakeLLMStream()
        retrieve = fake_retrieve_fn([make_hit("c1", 0.9, rerank=2.0)])
        engine = AnswerEngine(
            llm_client=llm,
            retrieve_fn=retrieve,
            persist_fn=fake_persist(),
            answer_cache=AnswerCache(redis=FakeRedis(), settings=settings_factory(cache_answer_ttl=3600)),
            settings=settings_factory(),
        )
        denied = AccessScope.for_grants([], tenant_id="t1", user_id=1, role="viewer")
        frames = parse_sse_frames("".join(engine.stream(QUESTION, scope=denied)))
        assert calls["analyze"] == 0
        assert llm.calls == []
        assert retrieve.calls == []
        assert get_event(frames, "meta")["route"] == "no_data"
        assert all_deltas(frames) == NO_DATA_REPLY

    def test_retrieval_routes_are_clipped_to_scope(self, monkeypatch, settings_factory):
        plan = replace(make_plan("hybrid"), route=make_route("hybrid", kb_ids=["kb_labor", "kb_civil"]))
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        retrieve = fake_retrieve_fn([make_hit("c1", 0.9, rerank=2.0)])
        engine = AnswerEngine(
            llm_client=FakeLLMStream(),
            retrieve_fn=retrieve,
            persist_fn=fake_persist(),
            answer_cache=AnswerCache(redis=FakeRedis(), settings=settings_factory(cache_answer_ttl=3600)),
            settings=settings_factory(),
        )
        list(engine.stream(QUESTION, scope=_scope(["kb_labor"])))
        assert retrieve.calls[0]["kb_ids"] == ["kb_labor"]

    def test_cached_sources_outside_scope_are_dropped(self, monkeypatch, settings_factory):
        plan = make_plan("hybrid")
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(
            llm_client=FakeLLMStream(),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.9, rerank=2.0)]),
            persist_fn=fake_persist(),
            answer_cache=AnswerCache(redis=FakeRedis(), settings=settings_factory(cache_answer_ttl=3600)),
            settings=settings_factory(),
        )
        cached_sources = [
            {"law": "劳动合同法", "kb_id": "kb_labor", "article_no": "第十九条"},
            {"law": "民法典", "kb_id": "kb_civil", "article_no": "第五百条"},
        ]
        filtered = engine._filter_sources(cached_sources, _scope(["kb_labor"]))
        assert [s["kb_id"] for s in filtered] == ["kb_labor"]


class TestQACacheIsolation:
    def test_exact_cache_key_is_scope_bound(self, settings_factory):
        redis = FakeRedis()
        retriever = QARetriever(redis=redis, settings=settings_factory(cache_qa_ttl=3600))
        scope_a = _scope(["kb_labor"])
        scope_b = _scope(["kb_civil"])
        norm = normalize_question(QUESTION)
        redis.set(cache_key(norm, scope_a.cache_token()), json.dumps(make_qa_pair().to_dict()))

        assert retriever.lookup_exact(QUESTION, scope=scope_a) is not None
        assert retriever.lookup_exact(QUESTION, scope=scope_b) is None

    def test_exact_cache_hit_outside_scope_is_rejected(self, settings_factory):
        redis = FakeRedis()
        retriever = QARetriever(redis=redis, settings=settings_factory(cache_qa_ttl=3600))
        scope = _scope(["kb_civil"])
        norm = normalize_question(QUESTION)
        redis.set(cache_key(norm, scope.cache_token()), json.dumps(make_qa_pair().to_dict()))
        assert retriever.lookup_exact(QUESTION, scope=scope) is None


class TestIntentCacheVersioning:
    def test_key_changes_with_model(self, settings_factory):
        s1 = settings_factory(llm_default_model="model-a")
        s2 = settings_factory(llm_default_model="model-b")
        assert _intent_cache_key(QUESTION, s1) != _intent_cache_key(QUESTION, s2)

    def test_schema_version_is_bumped_for_scope_keys(self):
        assert CACHE_SCHEMA_VERSION >= 2


class TestResolveScope:
    def test_unauthorized_requested_ids_are_reported(self):
        class _Rows:
            def mappings(self):
                return self

            def all(self):
                return [
                    {
                        "id": "kb_labor",
                        "corpus_version": 3,
                        "index_version": 4,
                        "is_public": 1,
                        "tenant_id": "default",
                        "owner_id": None,
                    }
                ]

        class _DB:
            def execute(self, *a, **kw):
                return _Rows()

        scope = resolve_access_scope(_DB(), None, ["kb_labor", "kb_other"])
        assert scope.kb_ids == ("kb_labor",)
        assert scope.denied_kb_ids == ("kb_other",)
        assert scope.grants[0].corpus_version == 3
        assert scope.grants[0].index_version == 4

    def test_scope_token_includes_versions_and_tenant(self):
        a = _scope(["kb_labor"])
        b = _scope(["kb_labor"])
        assert a.cache_token() == b.cache_token()
        assert a.cache_token() != _scope(["kb_labor"], tenant="other", is_public=False).cache_token()
