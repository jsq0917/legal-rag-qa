"""答案级缓存单测（全离线）。

覆盖三层：
1. 判据（自包含 / 重复提问）—— 决定"能不能命中"，错一次就是答非所问；
2. 读写（TTL、续期、损坏内容、Redis 故障降级）；
3. 与 ``AnswerEngine`` 的集成 —— 第一次生成后写缓存，第二次问同一句直接回放，
   且**不再调用意图识别与 LLM**（这才是"秒答"的实质）。
"""

from __future__ import annotations

import json

import pytest

from app.core.sse import parse_sse_frames
from app.generation import stream as stream_mod
from app.generation.citations import strip_citation_note
from app.generation.llm_client import LLMError, StreamChunk
from app.generation.stream import AnswerEngine
from app.orchestration.answer_cache import (
    REDIS_ANSWER_PREFIX,
    AnswerCache,
    answer_key,
    is_self_contained,
)
from app.orchestration.qa_cache import normalize_question

from tests.test_sse import (
    FakeLLMStream,
    all_deltas,
    fake_persist,
    fake_retrieve_fn,
    get_event,
    make_hit,
    make_plan,
)

QUESTION = "申请支付令需要准备哪些材料"
OTHER_QUESTION = "劳动仲裁的申请时效是多长时间"

LONG_ANSWER = (
    "结论：申请支付令需要准备申请书、身份证明、劳动关系证明与欠薪证据。"
    "依据：《中华人民共和国劳动合同法》第三十条，用人单位拖欠劳动报酬的，"
    "劳动者可以依法向当地人民法院申请支付令。实务提示：材料齐备后向有管辖权"
    "的基层人民法院立案窗口提交，法院应在五日内决定是否受理。"
)


class FakeRedis:
    """最小 Redis 假件：只实现答案缓存用到的 get/set/expire。"""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail = False

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.fail:
            raise ConnectionError("redis down")
        self.store[key] = value
        if ex:
            self.ttls[key] = ex
        return True

    def expire(self, key, ttl):
        if self.fail:
            raise ConnectionError("redis down")
        if key in self.store:
            self.ttls[key] = ttl
            return True
        return False


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def cache(fake_redis, settings_factory):
    return AnswerCache(redis=fake_redis, settings=settings_factory(cache_answer_ttl=3600))


# ---------------------------------------------------------------------------
# 1. 自包含判据
# ---------------------------------------------------------------------------
class TestSelfContained:
    def test_full_question_is_self_contained(self):
        assert is_self_contained(QUESTION)
        assert is_self_contained(OTHER_QUESTION)

    @pytest.mark.parametrize(
        "query",
        ["那这个呢", "上面说的怎么办", "它有效吗", "然后呢，还有哪些材料", "继续"],
    )
    def test_pronoun_question_is_not(self, query):
        assert not is_self_contained(query)

    @pytest.mark.parametrize(
        "query",
        [
            "那如果是我主动辞职呢？",  # 多轮实测漏判：旧指代词表不含"那如果"
            "工资按8000算，N+1是多少？",  # 条件覆盖型追问，答案依赖上文
        ],
    )
    def test_followup_found_in_multiturn_eval_is_not(self, query):
        assert not is_self_contained(query)

    def test_too_short_is_not(self):
        assert not is_self_contained("试用期多久")


# ---------------------------------------------------------------------------
# 2. 读写
# ---------------------------------------------------------------------------
class TestStoreAndLookup:
    def test_roundtrip(self, cache, fake_redis):
        assert cache.store(QUESTION, answer=LONG_ANSWER, sources=[{"title": "劳动合同法"}], route="hybrid", intent="labor_law")
        key = answer_key(normalize_question(QUESTION))
        assert key.startswith(REDIS_ANSWER_PREFIX)
        assert key in fake_redis.store
        hit = cache.lookup(QUESTION)
        assert hit is not None
        assert hit.answer == LONG_ANSWER
        assert hit.route == "hybrid"
        assert hit.sources == [{"title": "劳动合同法"}]

    def test_short_answer_not_stored(self, cache):
        assert not cache.store(QUESTION, answer="六个月。", route="hybrid")

    def test_error_not_stored(self, cache):
        assert not cache.store(QUESTION, answer=LONG_ANSWER, route="hybrid", error="stream_interrupted: x")
        assert cache.lookup(QUESTION) is None

    def test_pronoun_question_not_stored(self, cache):
        assert not cache.store("那这个呢怎么办", answer=LONG_ANSWER, route="hybrid")
        assert cache.lookup("那这个呢怎么办") is None

    def test_disabled_by_settings(self, fake_redis, settings_factory):
        cache = AnswerCache(redis=fake_redis, settings=settings_factory(cache_answer_enabled=False))
        assert not cache.store(QUESTION, answer=LONG_ANSWER, route="hybrid")
        assert cache.lookup(QUESTION) is None

    def test_ttl_applied_and_refreshed(self, cache, fake_redis):
        cache.store(QUESTION, answer=LONG_ANSWER, route="hybrid")
        key = answer_key(normalize_question(QUESTION))
        assert fake_redis.ttls[key] == 3600
        fake_redis.ttls[key] = 10  # 模拟已过了大半
        cache.lookup(QUESTION)
        assert fake_redis.ttls[key] == 3600, "命中必须续期，否则热缓存 7 天到点必失"

    def test_corrupted_value_ignored(self, cache, fake_redis):
        key = answer_key(normalize_question(QUESTION))
        fake_redis.store[key] = "{not json"
        assert cache.lookup(QUESTION) is None

    def test_empty_answer_ignored(self, cache, fake_redis):
        key = answer_key(normalize_question(QUESTION))
        fake_redis.store[key] = json.dumps({"question": QUESTION, "answer": "   "})
        assert cache.lookup(QUESTION) is None

    def test_redis_failure_degrades(self, fake_redis, settings_factory):
        cache = AnswerCache(redis=fake_redis, settings=settings_factory())
        fake_redis.fail = True
        assert cache.store(QUESTION, answer=LONG_ANSWER, route="hybrid") is False
        assert cache.lookup(QUESTION) is None


# ---------------------------------------------------------------------------
# 3. 多轮对话门槛
# ---------------------------------------------------------------------------
class TestHistoryGate:
    def test_empty_history_hits(self, cache):
        cache.store(QUESTION, answer=LONG_ANSWER, route="hybrid")
        assert cache.lookup(QUESTION, history=None) is not None
        assert cache.lookup(QUESTION, history=[]) is not None

    def test_repeated_question_in_history_hits(self, cache):
        """同一对话框里问第二次：历史中已有这句 → 正是"重复提问"的定义。"""
        cache.store(QUESTION, answer=LONG_ANSWER, route="hybrid")
        history = [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": "（上次的回答）"},
        ]
        assert cache.lookup(QUESTION, history=history) is not None

    def test_variant_spacing_still_hits(self, cache):
        cache.store(QUESTION, answer=LONG_ANSWER, route="hybrid")
        history = [
            {"role": "user", "content": "申请支付令需要准备哪些材料？"},
            {"role": "assistant", "content": "（上次的回答）"},
        ]
        assert cache.lookup(QUESTION, history=history) is not None

    def test_first_turn_with_history_misses(self, cache):
        """历史里没问过这句 → 可能是"同一句话但上半句给了新事实"，不能回放。"""
        cache.store(QUESTION, answer=LONG_ANSWER, route="hybrid")
        history = [
            {"role": "user", "content": "我在深圳工作，月薪八千"},
            {"role": "assistant", "content": "好的"},
        ]
        assert cache.lookup(QUESTION, history=history) is None

    def test_rewritten_key_matches_raw_question_in_history(self, cache):
        """改写后的键与历史原话不完全一致时，用原话判断"确实问过"。"""
        rewritten = "高温津贴发放标准"
        raw = "高温津贴的发放标准是什么？"
        cache.store(rewritten, answer=LONG_ANSWER, route="hybrid")
        history = [
            {"role": "user", "content": raw},
            {"role": "assistant", "content": "（上次的回答）"},
        ]
        # 只拿改写键比历史：归一化后不同 → 漏命中
        assert cache.lookup(rewritten, history=history) is None
        # 带上原话：命中
        assert cache.lookup(rewritten, history=history, raw_query=raw) is not None


# ---------------------------------------------------------------------------
# 4. 与 AnswerEngine 的集成：第二次提问必须秒答
# ---------------------------------------------------------------------------
def _long_llm():
    return FakeLLMStream(chunks=[StreamChunk(text=LONG_ANSWER)])


def _run(engine, query, history=None):
    return parse_sse_frames("".join(engine.stream(query, history=history)))


class TestEngineIntegration:
    def test_second_identical_question_skips_llm(self, monkeypatch, cache):
        plan = make_plan("pure_vector", query=QUESTION)
        analyze_calls = []

        def _fake_analyze(*a, **kw):
            analyze_calls.append(1)
            return plan

        monkeypatch.setattr(stream_mod, "analyze_query", _fake_analyze)
        llm = _long_llm()
        engine = AnswerEngine(
            llm_client=llm,
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=3.0)]),
            persist_fn=fake_persist(),
            answer_cache=cache,
        )

        first = _run(engine, QUESTION)
        assert get_event(first, "done")["cached"] is False
        # 引用校验可能追加一行提示；缓存断言只看正文
        assert strip_citation_note(all_deltas(first)) == LONG_ANSWER
        assert len(llm.calls) == 1
        assert len(analyze_calls) == 1

        second = _run(engine, QUESTION)
        assert get_event(second, "done")["cached"] is True
        assert get_event(second, "meta")["cached"] is True
        assert strip_citation_note(all_deltas(second)) == LONG_ANSWER
        assert len(llm.calls) == 1, "命中缓存后不得再调 LLM"
        # O1-2：缓存键依赖独立化改写，命中时仍会先做一次意图识别（不生成）
        assert len(analyze_calls) == 2

    def test_replay_keeps_sources(self, monkeypatch, cache):
        plan = make_plan("pure_vector", query=QUESTION)
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(
            llm_client=_long_llm(),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=3.0)]),
            persist_fn=fake_persist(),
            answer_cache=cache,
        )
        first = _run(engine, QUESTION)
        sources_first = get_event(first, "sources")["sources"]
        assert sources_first

        second = _run(engine, QUESTION)
        assert get_event(second, "sources")["sources"] == sources_first

    def test_failed_generation_not_cached(self, monkeypatch, cache):
        plan = make_plan("pure_vector", query=QUESTION)
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(
            llm_client=FakeLLMStream(chunks=[], raises=LLMError("限流")),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=3.0)]),
            persist_fn=fake_persist(),
            answer_cache=cache,
        )
        frames = _run(engine, QUESTION)
        assert get_event(frames, "error") is not None
        assert cache.lookup(QUESTION) is None, "报错的残缺回答绝不能固化"

    def test_greeting_not_cached(self, monkeypatch, cache):
        plan = make_plan("greeting", query=QUESTION)
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(
            llm_client=_long_llm(), retrieve_fn=fake_retrieve_fn([]), persist_fn=fake_persist(), answer_cache=cache
        )
        _run(engine, QUESTION)
        assert cache.lookup(QUESTION) is None

    def test_pronoun_followup_not_cached(self, monkeypatch, cache):
        plan = make_plan("pure_vector", query="那这个呢怎么办")
        monkeypatch.setattr(stream_mod, "analyze_query", lambda *a, **kw: plan)
        engine = AnswerEngine(
            llm_client=_long_llm(),
            retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=3.0)]),
            persist_fn=fake_persist(),
            answer_cache=cache,
        )
        _run(engine, "那这个呢怎么办")
        assert cache.lookup("那这个呢怎么办") is None


def test_probe_verdict_covers_uncacheable_multi_subquery():
    """无缓存键的情形探针必须判红；拆解条数只作观察项，不参与 ok。

    这条分支不依赖线上 LLM 的随机拆解结果：这里用合成轮次把"无缓存键"的情形
    固定覆盖住，避免它只在某几次真实运行里偶然出现。
    """
    from scripts.verify_multiturn_cache import _variant_verdict

    def turn(query: str, *, cache_query: str = "", hit: bool = False, sub_queries: int = 1) -> dict:
        return {
            "query": query,
            "rewrite_type": "decompose" if sub_queries > 1 else "direct",
            "sub_queries": sub_queries,
            "cache_query": cache_query,
            "cache": {"eligible": bool(cache_query), "hit": hit, "query": cache_query},
            "cached": hit,
        }

    spec = {
        "id": "original_multi_subquery",
        "text": "那如果是我主动辞职呢？",
        "expect": "uncacheable_multi_subquery",
    }
    follow_up = spec["text"]
    results = {
        "A_multi_turn": {"turns": [turn("高温津贴"), turn(follow_up, sub_queries=2)]},
        "B_new_session": {"turns": [turn(follow_up)]},
        "C_other_context": {"turns": [turn("年假"), turn(follow_up, sub_queries=2)]},
    }

    verdict = _variant_verdict(spec, results, positive_control=True)
    assert verdict["observations"]["mode"] == "uncacheable_multi_subquery"
    assert verdict["observations"]["mode_matches_expect"] is True
    assert verdict["observations"]["a_sub_queries"] == 2
    assert verdict["ok"] is True

    # 没有缓存键却出现命中 = 探针必须判红（无键就不该读缓存）
    results["B_new_session"]["turns"][0]["cache"]["hit"] = True
    bad = _variant_verdict(spec, results, positive_control=True)
    assert bad["ok"] is False
    assert bad["criteria"]["no_hit_without_cache_key"] is False

    # 子查询条数属 LLM 观察项：拆解与否不改变判定，只如实记录
    results["B_new_session"]["turns"][0]["cache"]["hit"] = False
    results["A_multi_turn"]["turns"][1]["sub_queries"] = 1
    undecomposed = _variant_verdict(spec, results, positive_control=True)
    assert undecomposed["observations"]["a_sub_queries"] == 1
    assert undecomposed["ok"] is True

    # 原句被替换属确定性缺陷：落到检索的文本必须就是用户原话
    results["A_multi_turn"]["turns"][1]["query"] = "主动辞职能否主张经济补偿"
    replaced = _variant_verdict(spec, results, positive_control=True)
    assert replaced["criteria"]["followup_text_not_replaced"] is False
    assert replaced["ok"] is False
