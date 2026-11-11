"""意图识别 LLM 路径单测：正常解析、字段校正、非法 JSON 降级、超时降级。

全部用假 LLM 客户端，不调真实 DashScope。
"""

from __future__ import annotations

import json

import pytest

from app.orchestration.intent import (
    INTENT_SYSTEM_PROMPT,
    _build_messages,
    _coerce_sub_queries,
    _strip_json_fences,
    _validate,
    recognize_intent,
)
from app.orchestration.models import IntentResult

FENCE = "`" * 3


def _llm_response(content: str):
    """假 LLM 响应（与 app.generation.llm_client.LLMResponse 同构）。"""
    from app.generation.llm_client import LLMResponse

    return LLMResponse(content=content, model="deepseek-v4.1-flash", latency_ms=100, usage={})


class FakeClient:
    """返回固定内容的假客户端；可模拟异常。"""

    def __init__(self, content: str | None = None, exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.exc is not None:
            raise self.exc
        return _llm_response(self.content)


VALID_JSON = json.dumps(
    {
        "intent": "labor_law",
        "rewrite_type": "direct",
        "sub_queries": ["试用期最长期限"],
        "complexity": "simple",
        "hit_structured": True,
        "structured_topic": "试用期",
        "law_hints": ["中华人民共和国劳动合同法"],
        "need_history": False,
    },
    ensure_ascii=False,
)


class TestStripJsonFences:
    def test_plain_json(self):
        assert _strip_json_fences('{"a": 1}') == '{"a": 1}'

    def test_markdown_fenced(self):
        text = FENCE + "json\n{\"a\": 1}\n" + FENCE
        assert _strip_json_fences(text) == '{"a": 1}'

    def test_with_surrounding_prose(self):
        text = "好的，结果如下：\n{\"a\": 1}\n希望有帮助"
        assert _strip_json_fences(text) == '{"a": 1}'

    def test_no_braces_returns_stripped(self):
        assert _strip_json_fences("  not json  ") == "not json"


class TestCoerceSubQueries:
    def test_string_becomes_single(self):
        assert _coerce_sub_queries("一个问题", "原问题") == ["一个问题"]

    def test_dedupe_and_cap_at_three(self):
        assert _coerce_sub_queries(["a", "a", "b", "c", "d"], "原问题") == ["a", "b", "c"]

    def test_empty_falls_back_to_query(self):
        assert _coerce_sub_queries([], "原问题") == ["原问题"]
        assert _coerce_sub_queries(None, "原问题") == ["原问题"]

    def test_blank_items_dropped(self):
        assert _coerce_sub_queries(["  ", "b"], "原问题") == ["b"]


class TestValidate:
    def test_valid_passthrough(self):
        result = _validate(json.loads(VALID_JSON), "原问题")
        assert result.intent == "labor_law"
        assert result.rewrite_type == "direct"
        assert result.sub_queries == ["试用期最长期限"]
        assert result.complexity == "simple"
        assert result.hit_structured is True
        assert result.structured_topic == "试用期"
        assert result.law_hints == ["中华人民共和国劳动合同法"]
        assert result.source == "llm"

    def test_invalid_enum_falls_back(self):
        data = {
            "intent": "martian",
            "rewrite_type": "translate",
            "complexity": "very_hard",
            "sub_queries": ["ok"],
            "law_hints": ["中华人民共和国劳动合同法"],
        }
        result = _validate(data, "原问题")
        assert result.intent == "general"
        assert result.rewrite_type == "direct"
        assert result.complexity == "medium"

    def test_unknown_topic_nulled(self):
        data = {
            "intent": "labor_law",
            "hit_structured": True,
            "structured_topic": "不存在的话题",
            "law_hints": ["中华人民共和国劳动合同法"],
        }
        result = _validate(data, "原问题")
        assert result.structured_topic is None
        # topic 不可信但 hit 标记保留（调用方仍会做一次廉价问答查询）
        assert result.hit_structured is True

    def test_missing_law_hints_backfilled_by_intent(self):
        data = {"intent": "labor_law", "sub_queries": ["q"]}
        result = _validate(data, "原问题")
        assert result.law_hints == ["中华人民共和国劳动合同法", "中华人民共和国劳动法"]

    def test_fake_law_filtered(self):
        data = {
            "intent": "civil_code",
            "law_hints": ["中华人民共和国劳动合同法", "假法律"],
        }
        result = _validate(data, "原问题")
        assert "假法律" not in result.law_hints


class TestRecognizeIntentLLM:
    def test_success(self):
        client = FakeClient(VALID_JSON)
        result = recognize_intent("试用期最长多久", client=client)
        assert result.source == "llm"
        assert result.intent == "labor_law"
        assert result.sub_queries == ["试用期最长期限"]
        # JSON 模式必须开启
        assert client.calls[0]["json_mode"] is True

    def test_history_included_in_messages(self):
        client = FakeClient(VALID_JSON)
        recognize_intent(
            "那个多久",
            history=[
                {"role": "user", "content": "试用期最长多久"},
                {"role": "assistant", "content": "六个月"},
            ],
            client=client,
        )
        messages = client.calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert "试用期最长多久" in messages[1]["content"]
        assert "那个多久" in messages[1]["content"]

    def test_invalid_json_degrades_to_rules(self):
        client = FakeClient(FENCE + "json\n{not valid")
        result = recognize_intent("试用期最长多久", client=client)
        assert result.source == "rules"
        assert result.intent == "labor_law"
        assert result.structured_topic == "试用期"

    def test_json_not_object_degrades(self):
        client = FakeClient("[1, 2, 3]")
        result = recognize_intent("试用期最长多久", client=client)
        assert result.source == "rules"

    def test_timeout_degrades(self):
        client = FakeClient(exc=TimeoutError("read timeout"))
        result = recognize_intent("试用期最长多久", client=client)
        assert result.source == "rules"

    def test_network_error_degrades(self):
        client = FakeClient(exc=ConnectionError("connection refused"))
        result = recognize_intent("试用期最长多久", client=client)
        assert result.source == "rules"

    def test_empty_content_degrades(self):
        client = FakeClient("")
        result = recognize_intent("试用期最长多久", client=client)
        assert result.source == "rules"

    def test_fenced_valid_json_parsed(self):
        payload = json.dumps(
            {
                "intent": "civil_code",
                "rewrite_type": "direct",
                "sub_queries": ["离婚财产分割"],
                "complexity": "medium",
                "hit_structured": False,
                "structured_topic": None,
                "law_hints": ["中华人民共和国民法典"],
                "need_history": False,
            },
            ensure_ascii=False,
        )
        client = FakeClient(FENCE + "json\n" + payload + "\n" + FENCE)
        result = recognize_intent("离婚财产怎么分", client=client)
        assert result.source == "llm"
        assert result.intent == "civil_code"

    def test_empty_query_uses_rules(self):
        client = FakeClient(VALID_JSON)
        result = recognize_intent("   ", client=client)
        assert result.source == "rules"
        assert client.calls == []


class TestBuildMessages:
    def test_no_history(self):
        messages = _build_messages("问题", None, max_turns=2)
        assert len(messages) == 2
        assert messages[0]["content"] == INTENT_SYSTEM_PROMPT
        assert "问题" in messages[1]["content"]
        assert "最近对话" not in messages[1]["content"]

    def test_history_window_truncated(self):
        history = [{"role": "user", "content": f"第{i}轮问题"} for i in range(10)]
        messages = _build_messages("当前问题", history, max_turns=2)
        # 只带最近 2 轮（4 条消息）：第 6-9 轮
        assert "第9轮问题" in messages[1]["content"]
        assert "第6轮问题" in messages[1]["content"]
        assert "第0轮问题" not in messages[1]["content"]
        assert "第5轮问题" not in messages[1]["content"]


class TestIntentClient:
    """意图客户端必须与主链路一致地下发 thinking（DeepSeek 默认为 enabled）。"""

    @staticmethod
    def _settings(**kwargs):
        from types import SimpleNamespace

        base = {
            "llm_base_url": "https://api.deepseek.com",
            "llm_default_model": "deepseek-flash",
            "intent_timeout_s": 5,
            "intent_max_retries": 1,
            "llm_thinking": "disabled",
        }
        base.update(kwargs)
        return SimpleNamespace(**base)

    def _client(self, monkeypatch, settings):
        import app.orchestration.intent as intent_mod

        monkeypatch.setattr(intent_mod, "get_llm_api_key", lambda: "sk-test")
        monkeypatch.setattr(intent_mod, "get_settings", lambda: settings)
        intent_mod.reset_intent_client()
        try:
            return intent_mod.get_intent_client()
        finally:
            intent_mod.reset_intent_client()

    def test_thinking_propagated(self, monkeypatch):
        client = self._client(monkeypatch, self._settings())
        assert client._thinking == "disabled"

    def test_thinking_blank_when_unset(self, monkeypatch):
        """DashScope/OpenAI 无该参数，留空时不能凭空发送。"""
        client = self._client(monkeypatch, self._settings(llm_thinking=""))
        assert client._thinking == ""


class TestFastRulePrecheck:
    """快速规则前置：只有**整句纯问候语**才短路 greeting。

    回归背景：旧实现用 startswith 匹配且把"请问"放进问候列表，
    "请问试用期被辞退有补偿吗"会被误路由成固定话术（不检索、不回答）。
    """

    def test_pure_greeting_short_circuits(self):
        from app.orchestration.intent import _fast_rule_precheck

        for q in ["你好", "您好", "在吗", "hi", "hello", "嗨", "谢谢", "多谢", "你好！", "在吗？", "hi~~"]:
            pre = _fast_rule_precheck(q)
            assert pre is not None, q
            assert pre.intent == "greeting"
            assert pre.source == "fast_rules"

    def test_greeting_with_particles_matches(self):
        from app.orchestration.intent import _fast_rule_precheck

        assert _fast_rule_precheck("你好呀").intent == "greeting"
        assert _fast_rule_precheck("hello! ").intent == "greeting"

    def test_polite_prefix_real_question_not_greeting(self):
        """礼貌前缀 + 真问题绝不能被当成问候语（本次回归 bug 的核心场景）。"""
        from app.orchestration.intent import _fast_rule_precheck

        for q in [
            "请问试用期被辞退有补偿吗",
            "请问，试用期最长可以约定多久？",
            "谢谢解答，那加班费怎么算？",
            "你好，我想问问离职补偿的事",
            "在吗？我有个劳动法问题",
        ]:
            assert _fast_rule_precheck(q) is None, q

    def test_real_question_with_client_none_goes_llm_path(self):
        """client=None（生产路径）时，真问题不能被 fast_rules 短路成 greeting。

        离线环境无密钥 → LLM 抛错 → 规则兜底，source 应为 rules 而非 fast_rules，
        且 intent 由规则判为 labor_law。
        """
        result = recognize_intent("请问试用期被辞退有补偿吗")
        assert result.source != "fast_rules"
        assert result.intent != "greeting"

    def test_pure_greeting_via_recognize_intent_skips_llm(self):
        result = recognize_intent("你好")
        assert result.source == "fast_rules"
        assert result.intent == "greeting"


class FakeIntentRedis:
    """意图缓存测试用的最小 Redis 桩：get/set，可注入异常。"""

    def __init__(self, exc: Exception | None = None) -> None:
        self.store: dict[str, str] = {}
        self.exc = exc

    def get(self, key):
        if self.exc:
            raise self.exc
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.exc:
            raise self.exc
        self.store[key] = value


class TestIntentCache:
    """意图结果缓存：无历史重复问题第二次不再调 LLM；带历史不走缓存。"""

    @pytest.fixture()
    def wired(self, monkeypatch):
        """把 get_intent_client 换成假 LLM、get_redis 换成桩，走生产路径（client=None）。"""
        import app.orchestration.intent as intent_mod
        from app.db import redis_client as redis_mod

        fake_redis = FakeIntentRedis()
        fake_llm = FakeClient(VALID_JSON)

        def fake_get_intent_client(settings=None):
            return fake_llm

        monkeypatch.setattr(intent_mod, "get_intent_client", fake_get_intent_client)
        # 两个缓存辅助函数内部都 from app.db.redis_client import get_redis，
        # 打补丁到源模块属性即可生效
        monkeypatch.setattr(redis_mod, "get_redis", lambda s=None: fake_redis)
        return fake_redis, fake_llm

    def test_second_call_hits_cache_and_skips_llm(self, wired):
        fake_redis, fake_llm = wired
        first = recognize_intent("加班费怎么算")
        assert first.source == "llm"
        assert len(fake_llm.calls) == 1
        assert len(fake_redis.store) == 1

        second = recognize_intent("加班费怎么算")
        assert second.source == "intent_cache"
        assert second.intent == first.intent
        assert second.sub_queries == first.sub_queries
        # LLM 没有被再次调用
        assert len(fake_llm.calls) == 1

    def test_normalized_variants_share_cache(self, wired):
        """标点/停用词差异不影响命中（与 qa 精确缓存同一归一化口径）。"""
        fake_redis, fake_llm = wired
        recognize_intent("加班费怎么算")
        recognize_intent("加班费，怎么算？")
        assert len(fake_llm.calls) == 1

    def test_history_bypasses_cache(self, wired):
        fake_redis, fake_llm = wired
        history = [{"role": "user", "content": "我上个月入职"}]
        recognize_intent("那试用期多久", history=history)
        recognize_intent("那试用期多久", history=history)
        # 带历史：不读不写缓存，每次都调 LLM
        assert len(fake_redis.store) == 0
        assert len(fake_llm.calls) == 2

    def test_rules_fallback_not_cached(self, wired, monkeypatch):
        """LLM 失败降级 rules 的结果不缓存（兜底结果不该被复用）。"""
        import app.orchestration.intent as intent_mod

        fake_redis, _ = wired
        monkeypatch.setattr(
            intent_mod, "get_intent_client", lambda s=None: FakeClient(exc=TimeoutError("t"))
        )
        result = recognize_intent("工伤认定标准是什么")
        assert result.source == "rules"
        assert len(fake_redis.store) == 0

    def test_redis_failure_fails_open(self, monkeypatch):
        """Redis 挂了不阻断链路：读失败当未命中，写失败不影响返回。"""
        import app.orchestration.intent as intent_mod
        from app.db import redis_client as redis_mod

        monkeypatch.setattr(redis_mod, "get_redis", lambda s=None: FakeIntentRedis(exc=ConnectionError("down")))
        monkeypatch.setattr(intent_mod, "get_intent_client", lambda s=None: FakeClient(VALID_JSON))
        result = recognize_intent("竞业限制补偿金标准")
        assert result.source == "llm"

    def test_cache_disabled_by_ttl_zero(self, wired, monkeypatch):
        fake_redis, fake_llm = wired
        monkeypatch.setattr(
            "config.get_settings",
            lambda: __import__("types").SimpleNamespace(intent_cache_ttl=0),
        )
        # settings 显式传入 ttl=0
        from types import SimpleNamespace

        s = SimpleNamespace(intent_cache_ttl=0, intent_history_turns=2)
        recognize_intent("劳务派遣合同怎么签", settings=s)
        recognize_intent("劳务派遣合同怎么签", settings=s)
        assert len(fake_redis.store) == 0
        assert len(fake_llm.calls) == 2


class SequencedClient:
    """按调用顺序返回不同内容，用于测「二次拆解」这第二次调用。"""

    def __init__(self, contents: list[str], exc: Exception | None = None, fail_at: int = 99) -> None:
        self.contents = contents
        self.exc = exc
        self.fail_at = fail_at
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.exc is not None and len(self.calls) >= self.fail_at:
            raise self.exc
        return _llm_response(self.contents[min(len(self.calls) - 1, len(self.contents) - 1)])


class TestSecondDecompose:
    """首拆没拆、但问句带第二处诉求信号时，补一次拆解调用。"""

    QUERY = "公司拖欠工资，我能不能直接走人还要补偿？"

    def _settings(self, enabled: bool = True):
        from types import SimpleNamespace

        return SimpleNamespace(
            intent_decompose_second_opinion=enabled,
            intent_history_turns=2,
        )

    @staticmethod
    def _first(complexity: str = "medium", rewrite_type: str = "direct") -> str:
        payload = json.loads(VALID_JSON)
        payload["complexity"] = complexity
        payload["rewrite_type"] = rewrite_type
        return json.dumps(payload, ensure_ascii=False)

    def test_two_subs_promote_to_decompose(self):
        second = json.dumps(
            {
                "sub_queries": [
                    "用人单位未及时足额支付劳动报酬，劳动者能否解除劳动合同并要求经济补偿",
                    "用人单位拖欠工资的法律责任与赔偿标准",
                ],
                "complexity": "medium",
            },
            ensure_ascii=False,
        )
        client = SequencedClient([self._first(), second])
        result = recognize_intent(self.QUERY, client=client, settings=self._settings())
        assert len(client.calls) == 2
        assert result.rewrite_type == "decompose"
        assert len(result.sub_queries) == 2
        # 第二次调用用的是拆解专用 prompt，不是主 prompt
        assert client.calls[1]["messages"][0]["content"] != INTENT_SYSTEM_PROMPT

    def test_empty_subs_keeps_first_result(self):
        client = SequencedClient([self._first(), json.dumps({"sub_queries": []}, ensure_ascii=False)])
        result = recognize_intent(self.QUERY, client=client, settings=self._settings())
        assert len(client.calls) == 2
        assert result.rewrite_type == "direct"
        assert result.sub_queries == ["试用期最长期限"]

    def test_second_call_failure_keeps_first_result(self):
        client = SequencedClient([self._first()], exc=TimeoutError("read timeout"), fail_at=2)
        result = recognize_intent(self.QUERY, client=client, settings=self._settings())
        assert result.source == "llm"
        assert result.rewrite_type == "direct"

    def test_lower_complexity_ignored(self):
        """二次拆解只能抬高复杂度：压低会缩小最终 topK，直接吃掉召回。"""
        second = json.dumps(
            {"sub_queries": ["a 诉求", "b 诉求"], "complexity": "simple"},
            ensure_ascii=False,
        )
        client = SequencedClient([self._first("medium"), second])
        result = recognize_intent(self.QUERY, client=client, settings=self._settings())
        assert result.rewrite_type == "decompose"
        assert result.complexity == "medium"

    def test_no_signal_skips_second_call(self):
        client = SequencedClient([self._first()])
        result = recognize_intent("试用期最长可以约定多久", client=client, settings=self._settings())
        assert len(client.calls) == 1
        assert result.rewrite_type == "direct"

    def test_already_decompose_skips_second_call(self):
        client = SequencedClient([self._first("complex", "decompose")])
        result = recognize_intent(self.QUERY, client=client, settings=self._settings())
        assert len(client.calls) == 1
        assert result.rewrite_type == "decompose"

    def test_disabled_by_config(self):
        client = SequencedClient([self._first()])
        result = recognize_intent(self.QUERY, client=client, settings=self._settings(False))
        assert len(client.calls) == 1
        assert result.rewrite_type == "direct"
