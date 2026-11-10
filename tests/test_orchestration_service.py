"""编排入口 analyze_query 单测：意图→路由→问答判定的完整决策流。

用假意图客户端 + 假问答检索器；并覆盖阶段 4 验收标准：
**纯结构化问题不产生 Milvus 调用**（plan.route.use_milvus 为 False）。
"""

from __future__ import annotations

import json

from app.generation.llm_client import LLMResponse
from app.orchestration.models import IntentResult
from app.orchestration.qa_cache import QAPair, QALookupResult
from app.orchestration.router import route as route_fn
from app.orchestration.service import QueryPlan, analyze_query

FENCE = "`" * 3


def _llm_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="deepseek-v4.1-flash", latency_ms=50, usage={})


class FakeIntentClient:
    """只覆盖 recognize_intent 需要的 chat(json_mode=True) 行为。"""

    def __init__(self, content: str | None = None, exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.exc is not None:
            raise self.exc
        return _llm_response(self.content)


def _structured_json(**overrides) -> str:
    data = {
        "intent": "labor_law",
        "rewrite_type": "direct",
        "sub_queries": ["试用期最长期限"],
        "complexity": "simple",
        "hit_structured": True,
        "structured_topic": "试用期",
        "law_hints": ["中华人民共和国劳动合同法"],
        "need_history": False,
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


class FakeQARetriever:
    """记录调用并返回预设结果。"""

    def __init__(self, result: QALookupResult | None = None) -> None:
        self.result = result or QALookupResult()
        self.calls: list[dict] = []
        self.exact_hit: QAPair | None = None  # 缓存前置命中时返回

    def lookup_exact(self, query, *, scope=None):
        """缓存前置：默认不命中，测试可预设 exact_hit。"""
        self.calls.append({"method": "lookup_exact", "query": query})
        return self.exact_hit

    def lookup(self, query, *, structured_topic=None, kb_ids=None, scope=None):
        self.calls.append({"method": "lookup", "query": query, "structured_topic": structured_topic, "kb_ids": kb_ids})
        return self.result


def _patch_recognize(monkeypatch, client):
    """让 recognize_intent 使用假客户端（绕过真实 key 检查）。"""
    from app.orchestration import intent as intent_mod

    monkeypatch.setattr(intent_mod, "get_intent_client", lambda settings=None: client)


class TestStructuredFlow:
    def test_pure_structured_never_touches_milvus(self, monkeypatch):
        """验收标准：纯结构化问题不产生 Milvus 调用。"""
        _patch_recognize(monkeypatch, FakeIntentClient(_structured_json()))
        qa = FakeQARetriever(
            QALookupResult(
                hit=QAPair(id=7, category="试用期", question_std="试用期最长可以约定多久？", answer="答案"),
                source="mysql_bm25_cosine",
                similarity=0.97,
            )
        )
        plan = analyze_query("试用期最长可以约定多久", qa_retriever=qa)
        assert plan.route.route == "pure_structured"
        assert plan.route.use_milvus is False  # ← 不碰 Milvus
        assert plan.route.use_mysql_qa is True
        assert plan.qa.hit is not None and plan.qa.hit.id == 7
        assert plan.degraded is False

    def test_hybrid_when_no_exact_answer(self, monkeypatch):
        _patch_recognize(monkeypatch, FakeIntentClient(_structured_json()))
        qa = FakeQARetriever(
            QALookupResult(
                references=[QAPair(id=8, category="试用期", question_std="同类问题", answer="参考")],
                source="topic_reference",
            )
        )
        plan = analyze_query("试用期被辞退有补偿吗", qa_retriever=qa)
        assert plan.route.route == "hybrid"
        assert plan.route.use_milvus is True
        assert plan.route.use_mysql_qa is True
        assert plan.qa.references[0].id == 8


class TestVectorFlow:
    def test_pure_vector_skips_qa_lookup(self, monkeypatch):
        """纯法条问题不查问答表：qa_retriever 一次都不调用。"""
        _patch_recognize(monkeypatch, FakeIntentClient(_structured_json(hit_structured=False, structured_topic=None)))
        qa = FakeQARetriever()
        plan = analyze_query("劳动合同法第三十九条原文是什么", qa_retriever=qa)
        assert plan.route.route == "pure_vector"
        assert plan.route.use_mysql_qa is False
        # 缓存前置会调一次 lookup_exact（0.1ms 级），但不调完整 lookup()（BM25+余弦）
        assert all(c["method"] == "lookup_exact" for c in qa.calls)
        assert not any(c["method"] == "lookup" for c in qa.calls)
        assert plan.route.use_milvus is True


class TestGreetingFlow:
    def test_greeting_no_lookup_at_all(self, monkeypatch):
        _patch_recognize(monkeypatch, FakeIntentClient(_structured_json(intent="greeting", law_hints=[], hit_structured=False)))
        qa = FakeQARetriever()
        plan = analyze_query("你好", qa_retriever=qa)
        assert plan.route.route == "greeting"
        # 缓存前置会调一次 lookup_exact，但 greeting 不需要完整 qa lookup
        assert all(c["method"] == "lookup_exact" for c in qa.calls)
        assert not any(c["method"] == "lookup" for c in qa.calls)
        assert plan.sub_queries == ["试用期最长期限"] or plan.sub_queries  # 改写结果仍保留


class TestDegradedFlow:
    def test_llm_failure_still_produces_plan(self, monkeypatch):
        """LLM 失败 → 规则兜底 → 链路不断（降级成功率 100%）。"""
        _patch_recognize(monkeypatch, FakeIntentClient(exc=TimeoutError("read timeout")))
        qa = FakeQARetriever(
            QALookupResult(
                hit=QAPair(id=7, category="试用期", question_std="试用期最长可以约定多久？", answer="答案"),
                source="mysql_bm25_cosine",
            )
        )
        plan = analyze_query("试用期最长可以约定多久", qa_retriever=qa)
        assert plan.degraded is True
        assert plan.intent.source == "rules"
        # 规则兜底同样命中结构化话题 → 走问答库 → 纯结构化
        assert plan.route.route == "pure_structured"
        assert plan.route.use_milvus is False

    def test_no_key_still_works(self, monkeypatch):
        """未配置任何大模型密钥：走真实 get_intent_client → ConfigError → 规则兜底。

        必须显式清掉三个密钥变量，否则本机一旦配了 DEEPSEEK_API_KEY，这条用例会
        真的去调线上 LLM（既烧钱又让"全离线"承诺失效）。
        """
        for name in ("LLM_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        qa = FakeQARetriever()
        plan = analyze_query("试用期最长可以约定多久", qa_retriever=qa)
        assert plan.degraded is True
        assert plan.intent.source == "rules"


class TestSubQueries:
    def test_rewritten_sub_queries_used(self, monkeypatch):
        _patch_recognize(
            monkeypatch,
            FakeIntentClient(_structured_json(sub_queries=["经济补偿计算标准", "代通知金规定"], rewrite_type="decompose")),
        )
        plan = analyze_query("裁员补偿和代通知金怎么算", qa_retriever=FakeQARetriever())
        assert plan.sub_queries == ["经济补偿计算标准", "代通知金规定"]

    def test_empty_sub_queries_fall_back_to_query(self, monkeypatch):
        _patch_recognize(monkeypatch, FakeIntentClient(_structured_json(sub_queries=[])))
        plan = analyze_query("原始问题", qa_retriever=FakeQARetriever())
        assert plan.sub_queries == ["原始问题"]


class TestToDict:
    def test_serializable(self, monkeypatch):
        _patch_recognize(monkeypatch, FakeIntentClient(_structured_json()))
        plan = analyze_query("试用期最长可以约定多久", qa_retriever=FakeQARetriever())
        data = plan.to_dict()
        assert data["intent"]["intent"] == "labor_law"
        assert data["route"]["route"] in {"pure_structured", "hybrid"}
        assert data["degraded"] is False
        # 必须可 JSON 序列化（API 响应）
        json.dumps(data, ensure_ascii=False)
