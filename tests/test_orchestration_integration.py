"""阶段 4 联调测试：真实 MySQL（ngram BM25）+ Redis（缓存）+ bge-m3（余弦门）。

    RUN_INTEGRATION=1 uv run python -m pytest tests/test_orchestration_integration.py -v -m integration

会真实加载 bge-m3（CPU 上约 10 秒）。意图识别因未配置 DASHSCOPE_API_KEY
自动走规则兜底，不影响结构化链路验证。
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="需要真实 MySQL + Redis + 本地模型，设置 RUN_INTEGRATION=1 启用",
    ),
]

#: 与种子数据 question_std 完全一致的问题（应精确命中）
EXACT_MATCHES = [
    ("试用期最长可以约定多久？", "试用期"),
    ("试用期工资可以随便定吗？", "试用期"),
    ("平时加班费怎么算？", "加班费"),
    ("经济补偿金按什么标准计算？", "经济补偿"),
    ("公司违法辞退员工怎么赔偿？", "合同解除"),
    ("劳动争议仲裁时效是多久？", "仲裁时效"),
    ("用人单位必须给员工缴社保吗？", "社保"),
]

#: 同话题但非标准问法（BM25 应能召回，余弦门由 embedder 判定）
COLLOQUIAL = [
    "试用期一般能给多长",
    "周六上班是不是必须给双倍",
    "被公司裁掉了有赔偿吗",
]

#: 明显不相关的问题（不应命中）
UNRELATED = ["量子计算机的工作原理", "西红柿鸡蛋怎么做"]


@pytest.fixture(scope="module")
def retriever():
    from app.orchestration.qa_cache import get_qa_retriever

    return get_qa_retriever()


@pytest.fixture(scope="module")
def embedder():
    from app.ingest.embedder import get_embedder

    return get_embedder()


class TestRealBM25:
    def test_exact_match_hits(self, retriever, embedder):
        for question, topic in EXACT_MATCHES:
            result = retriever.lookup(question, structured_topic=topic)
            assert result.hit is not None, f"未命中: {question}"
            assert result.similarity >= 0.90, f"相似度不足: {question} -> {result.similarity}"
            assert result.hit.category == topic

    def test_second_lookup_from_cache(self, retriever):
        # 上一次已写 Redis 精确缓存，第二次应走 redis_exact 且不再编码
        result = retriever.lookup("试用期最长可以约定多久？", structured_topic="试用期")
        assert result.source == "redis_exact"
        assert result.hit is not None

    def test_unrelated_no_hit(self, retriever):
        for question in UNRELATED:
            result = retriever.lookup(question, structured_topic=None)
            assert result.hit is None, f"误命中: {question}"

    def test_topic_reference_on_miss(self, retriever):
        result = retriever.lookup("试用期满了公司说不合适", structured_topic="试用期")
        # 不保证命中（口语变体），但命中话题时应给同类参考
        if result.hit is None:
            assert result.source in {"topic_reference", "none"}


class TestRealCosineGate:
    def test_identical_question_similarity(self, embedder):
        from app.orchestration.qa_cache import QARetriever

        retriever = QARetriever(embedder=embedder)
        # 仅差一个句末问号，bge-m3 实测约 0.987（远高于 0.90 门限）
        sim = retriever._cosine("试用期最长可以约定多久", "试用期最长可以约定多久？")
        assert sim is not None and sim >= 0.95

    def test_different_question_similarity(self, embedder):
        from app.orchestration.qa_cache import QARetriever

        retriever = QARetriever(embedder=embedder)
        sim = retriever._cosine("试用期最长可以约定多久", "加班费怎么计算？")
        assert sim is None or sim < 0.90


class TestRealAnalyzeQuery:
    def test_end_to_end_structured(self, monkeypatch):
        """真实依赖下：标准问法 → 纯结构化，不碰 Milvus。

        意图来源不固定：缓存前置命中是 ``cache``，有 LLM key 时是 ``llm``，
        无 key 时才是 ``rules``，因此不断言 ``degraded``。
        """
        from app.orchestration.service import analyze_query

        plan = analyze_query("试用期最长可以约定多久？", selected_kb_ids=["kb_labor"])
        assert plan.intent.source in {"cache", "llm", "rules"}
        assert plan.intent.intent == "labor_law"
        assert plan.route.route == "pure_structured"
        assert plan.route.use_milvus is False
        assert plan.qa.hit is not None

    def test_end_to_end_vector(self):
        """真实依赖下：无结构化命中 → 纯法条路由。"""
        from app.orchestration.service import analyze_query

        plan = analyze_query("劳动合同法第八十二条原文怎么规定", selected_kb_ids=["kb_labor"])
        assert plan.route.route in {"pure_vector", "hybrid"}
        assert plan.route.use_milvus is True
