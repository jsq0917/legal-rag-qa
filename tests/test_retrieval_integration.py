"""阶段 3 联调测试：真实 Milvus 混合检索 + 重排 + 动态 topK + 消融对比。

    RUN_INTEGRATION=1 uv run python -m pytest tests/test_retrieval_integration.py -v -m integration

会真实加载 bge-m3 与 bge-reranker-v2-m3（CPU 上首次约 90 秒），故默认跳过。
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="需要真实 Milvus + 本地模型，设置 RUN_INTEGRATION=1 启用",
    ),
]

#: (问题, 期望命中的法条) —— 期望值用「法名#条号」，跨法同号不混淆
CASES: list[tuple[str, set[str], str]] = [
    ("试用期最长可以约定多久", {"中华人民共和国劳动法#第二十一条", "中华人民共和国劳动合同法#第十九条"}, "simple"),
    ("竞业限制的期限最长是多久", {"中华人民共和国劳动合同法#第二十四条"}, "simple"),
    ("经济补偿按什么标准计算", {"中华人民共和国劳动合同法#第四十七条"}, "medium"),
    ("什么情况下用人单位可以单方解除劳动合同", {"中华人民共和国劳动合同法#第三十九条"}, "medium"),
    ("劳务派遣用工只能用在哪些岗位", {"中华人民共和国劳动合同法#第六十六条"}, "medium"),
    ("女职工在孕期用人单位能不能解除劳动合同", {"中华人民共和国劳动合同法#第四十二条"}, "medium"),
    ("劳动争议申请仲裁的时效期间是多久", {"中华人民共和国劳动法#第八十二条"}, "medium"),
    ("用人单位裁减人员二十人以上需要履行什么程序", {"中华人民共和国劳动合同法#第四十一条"}, "complex"),
]


@pytest.fixture(scope="module")
def components():
    from app.ingest.embedder import get_embedder
    from app.retrieval.rerank import get_reranker

    return get_embedder(), get_reranker()


class TestHybridRetrieval:
    def test_recall_at_5_meets_target(self, components):
        from app.retrieval.service import recall_at_k, retrieve

        embedder, reranker = components
        hits_at_5 = 0
        details = []
        for question, expected, complexity in CASES:
            result = retrieve(
                question, complexity=complexity, embedder=embedder, reranker=reranker
            )
            score = recall_at_k(result.hits, expected, 5)
            hits_at_5 += score
            details.append(
                (question, score, [f"{h.law_name[-6:]}#{h.article_no}" for h in result.hits[:3]])
            )
        recall = hits_at_5 / len(CASES)
        for question, score, top3 in details:
            print(f"  {score:.0f} {question[:24]:<26} top3={top3}")
        assert recall >= 0.75, f"Recall@5={recall:.2f} 低于 0.75"

    def test_dynamic_topk_respected(self, components):
        from app.retrieval.service import retrieve

        embedder, reranker = components
        for complexity, expected in (("simple", 3), ("medium", 5), ("complex", 8)):
            result = retrieve(
                "试用期最长可以约定多久", complexity=complexity,
                embedder=embedder, reranker=reranker,
            )
            assert result.topk == expected
            assert len(result.hits) <= expected

    def test_parent_content_expanded(self, components):
        from app.retrieval.service import retrieve

        embedder, reranker = components
        result = retrieve("经济补偿按什么标准计算", embedder=embedder, reranker=reranker)
        assert result.hits
        assert all(h.parent_content for h in result.hits), "父块回表为空，生成层会拿不到整条法条"

    def test_kb_isolation(self, components):
        from app.retrieval.service import retrieve

        embedder, reranker = components
        result = retrieve(
            "试用期最长可以约定多久", kb_ids=["kb_civil"],
            embedder=embedder, reranker=reranker,
        )
        assert all(h.kb_id == "kb_civil" for h in result.hits)

    def test_multi_kb_search(self, components):
        from app.retrieval.service import retrieve

        embedder, reranker = components
        result = retrieve(
            "试用期最长可以约定多久", kb_ids=["kb_labor", "kb_civil"],
            embedder=embedder, reranker=reranker,
        )
        assert result.hits
        assert any(h.kb_id == "kb_labor" for h in result.hits)

    def test_ablation_dense_vs_hybrid(self, components):
        """消融：仅稠密 vs 混合+重排，后者整体不应更差。"""
        from app.retrieval.service import recall_at_k, retrieve

        embedder, reranker = components
        dense_total = hybrid_total = 0.0
        for question, expected, complexity in CASES:
            dense = retrieve(
                question, complexity=complexity, mode="dense", use_rerank=False,
                embedder=embedder, reranker=reranker,
            )
            full = retrieve(
                question, complexity=complexity, mode="hybrid", use_rerank=True,
                embedder=embedder, reranker=reranker,
            )
            dense_total += recall_at_k(dense.hits, expected, 5)
            hybrid_total += recall_at_k(full.hits, expected, 5)
        dense_recall = dense_total / len(CASES)
        hybrid_recall = hybrid_total / len(CASES)
        print(f"  消融 Recall@5：仅稠密={dense_recall:.3f} 混合+重排={hybrid_recall:.3f}")
        assert hybrid_recall + 1e-9 >= dense_recall - 0.05


class TestPreviewApi:
    def test_preview_returns_scores(self, integration_auth_headers):
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        resp = client.post(
            "/api/retrieval/preview",
            json={"question": "试用期最长可以约定多久", "complexity": "simple"},
            headers=integration_auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["hits"]
        assert body["hits"][0]["article_no"]
        assert body["hits"][0]["rerank_score"] is not None
        assert body["hits"][0]["parent_content"]

    def test_preview_rejects_empty_question(self, integration_auth_headers):
        from fastapi.testclient import TestClient

        from app.main import app

        assert (
            TestClient(app)
            .post(
                "/api/retrieval/preview",
                json={"question": ""},
                headers=integration_auth_headers,
            )
            .status_code
            == 422
        )
