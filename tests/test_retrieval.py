"""检索层单测：过滤器、双路请求构造、RRF 结果映射、去重、动态 topK、Recall/MRR。

全部用假 Milvus 客户端 + DummyEmbedder/DummyReranker，不连真实服务。
"""

from __future__ import annotations

import pytest

from app.ingest.embedder import DummyEmbedder, Embedding
from app.retrieval import hybrid
from app.retrieval.hybrid import Hit, build_filter, build_kb_filter, dedupe_by_parent, to_hit
from app.retrieval.rerank import DummyReranker
from app.retrieval.service import mrr, recall_at_k, resolve_mode, retrieve


def _entity(chunk_id: str, parent: str, content: str, kb: str = "kb_labor",
            law: str = "中华人民共和国劳动合同法", article: str = "第一条") -> dict:
    return {
        "chunk_id": chunk_id,
        "parent_id": parent,
        "kb_id": kb,
        "doc_id": "DOC1",
        "content": content,
        "title": f"{law}·第一章·{article}",
        "article_no": article,
        "chapter_path": "第一章",
        "law_name": law,
        "source_file": "a.md",
    }


class FakeMilvus:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[dict] = []

    def hybrid_search(self, **kwargs):
        self.calls.append({"type": "hybrid", **kwargs})
        return [[{"entity": r, "distance": 0.5 - i * 0.01} for i, r in enumerate(self.rows)]]

    def search(self, **kwargs):
        self.calls.append({"type": "dense", **kwargs})
        return [[{"entity": r, "distance": 0.9 - i * 0.01} for i, r in enumerate(self.rows)]]


class TestFilter:
    def test_empty_means_all(self):
        assert build_kb_filter(None) == ""
        assert build_kb_filter([]) == ""

    def test_single_and_multi(self):
        assert build_kb_filter(["kb_labor"]) == 'kb_id in ["kb_labor"]'
        assert build_kb_filter(["kb_labor", "kb_civil"]) == 'kb_id in ["kb_labor", "kb_civil"]'

    def test_law_hints_combined_with_kb(self):
        """法域预过滤：kb 条件与 law 条件用 and 拼接。"""
        expr = build_filter(["kb_labor"], ["中华人民共和国劳动合同法"])
        assert expr == 'kb_id in ["kb_labor"] and law_name in ["中华人民共和国劳动合同法"]'

    def test_law_hints_alone(self):
        assert build_filter(None, ["中华人民共和国劳动法"]) == (
            'law_name in ["中华人民共和国劳动法"]'
        )

    def test_no_law_hints_keeps_old_behaviour(self):
        """hints 为空（general/other 意图）时不能加过滤，否则误伤召回。"""
        assert build_filter(["kb_labor"], None) == build_kb_filter(["kb_labor"])
        assert build_filter(["kb_labor"], []) == build_kb_filter(["kb_labor"])


class TestRequests:
    def test_hybrid_builds_two_requests(self):
        emb = Embedding(dense=[0.1] * 1024, sparse={1: 0.5, 2: 0.3})
        reqs = hybrid.build_requests(emb, pool=20, expr='kb_id in ["kb_labor"]', mode="hybrid")
        assert len(reqs) == 2
        assert reqs[0].anns_field == "dense_vector"
        assert reqs[1].anns_field == "sparse_vector"
        assert reqs[0].limit == 20

    def test_dense_mode_single_request(self):
        emb = Embedding(dense=[0.1] * 1024, sparse={1: 0.5})
        assert len(hybrid.build_requests(emb, pool=12, expr="", mode="dense")) == 1

    def test_empty_sparse_degrades_to_dense(self):
        emb = Embedding(dense=[0.1] * 1024, sparse={})
        assert len(hybrid.build_requests(emb, pool=12, expr="", mode="hybrid")) == 1


class TestToHit:
    def test_maps_all_traceability_fields(self):
        hit = to_hit({"entity": _entity("D1:p1:c0", "D1:p1", "条文"), "distance": 0.42})
        assert hit.chunk_id == "D1:p1:c0"
        assert hit.parent_id == "D1:p1"
        assert hit.score == pytest.approx(0.42)
        assert hit.law_article == "中华人民共和国劳动合同法#第一条"
        assert hit.embed_text.startswith("中华人民共和国劳动合同法")


class TestDedupe:
    def test_same_parent_keeps_best(self):
        hits = [
            Hit(chunk_id="D1:p1:c0", parent_id="D1:p1", kb_id="k", doc_id="D1",
                content="a", title="t", article_no="第一条", chapter_path="", law_name="L",
                source_file="f", score=0.2),
            Hit(chunk_id="D1:p1:c1", parent_id="D1:p1", kb_id="k", doc_id="D1",
                content="b", title="t", article_no="第一条", chapter_path="", law_name="L",
                source_file="f", score=0.9),
        ]
        result = dedupe_by_parent(hits, topk=5)
        assert len(result) == 1
        assert result[0].chunk_id == "D1:p1:c1"

    def test_respects_topk(self):
        hits = [
            Hit(chunk_id=f"D1:p{i}:c0", parent_id=f"D1:p{i}", kb_id="k", doc_id="D1",
                content="x", title="t", article_no=f"第{i}条", chapter_path="", law_name="L",
                source_file="f", score=1.0 / i)
            for i in range(1, 11)
        ]
        assert len(dedupe_by_parent(hits, topk=3)) == 3

    def test_rerank_score_preferred(self):
        hits = [
            Hit(chunk_id="a", parent_id="a", kb_id="", doc_id="", content="", title="",
                article_no="", chapter_path="", law_name="", source_file="", score=0.9,
                rerank_score=0.1),
            Hit(chunk_id="b", parent_id="b", kb_id="", doc_id="", content="", title="",
                article_no="", chapter_path="", law_name="", source_file="", score=0.1,
                rerank_score=0.9),
        ]
        assert dedupe_by_parent(hits, topk=1)[0].chunk_id == "b"


class TestRetrieve:
    @pytest.fixture
    def fake(self):
        rows = [
            _entity("D1:p19:c0", "D1:p19", "试用期不得超过六个月。", article="第十九条"),
            _entity("D1:p39:c0", "D1:p39", "劳动者有下列情形之一的，用人单位可以解除劳动合同。", article="第三十九条"),
            _entity("D1:p46:c0", "D1:p46", "用人单位应当向劳动者支付经济补偿。", article="第四十六条"),
        ]
        return FakeMilvus(rows)

    def test_dynamic_topk_by_complexity(self, fake):
        for complexity, expected in (("simple", 3), ("medium", 5), ("complex", 8)):
            result = retrieve(
                "试用期多久", complexity=complexity, client=fake,
                embedder=DummyEmbedder(), reranker=DummyReranker(), expand_parent=False,
            )
            assert result.topk == expected
            assert result.pool == max(4 * expected, 12)
            assert len(result.hits) <= expected

    def test_rerank_reorders(self, fake):
        result = retrieve(
            "试用期 经济补偿", client=fake, embedder=DummyEmbedder(),
            reranker=DummyReranker(), expand_parent=False, topk=3,
        )
        assert result.reranked is True
        scores = [h.rerank_score for h in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_dense_mode_uses_client_search(self, fake):
        retrieve("x", client=fake, embedder=DummyEmbedder(), mode="dense",
                 use_rerank=False, expand_parent=False)
        assert fake.calls[0]["type"] == "dense"

    def test_hybrid_mode_uses_hybrid_search(self, fake):
        retrieve("试用期多久", client=fake, embedder=DummyEmbedder(), reranker=DummyReranker())
        assert fake.calls[0]["type"] == "hybrid"

    def test_default_ranker_is_rrf(self, fake, settings_factory):
        """默认等权 RRF：加权（0.75/0.25）在消融里无收益甚至略降，已回退（REPORT.md §4.4）。"""
        s = settings_factory()
        assert s.retrieval_ranker == "rrf"
        retrieve("试用期多久", client=fake, embedder=DummyEmbedder(),
                 reranker=DummyReranker(), settings=s)
        spec = fake.calls[0]["ranker"].dict()
        assert spec["strategy"] == "rrf"

    def test_weighted_ranker_when_configured(self, fake, settings_factory):
        """显式配成 weighted 时走 WeightedRanker，权重取 0.75/0.25。"""
        s = settings_factory(retrieval_ranker="weighted")
        retrieve("试用期多久", client=fake, embedder=DummyEmbedder(),
                 reranker=DummyReranker(), settings=s)
        spec = fake.calls[0]["ranker"].dict()
        assert spec["strategy"] == "weighted"
        assert spec["params"]["weights"] == [0.75, 0.25]

    def test_law_hints_filter_reaches_milvus(self, fake):
        retrieve("试用期多久", kb_ids=["kb_labor"],
                 law_hints=["中华人民共和国劳动合同法"],
                 client=fake, embedder=DummyEmbedder(), use_rerank=False, expand_parent=False)
        expr = fake.calls[0]["reqs"][0].expr
        assert 'law_name in ["中华人民共和国劳动合同法"]' in expr
        assert 'kb_id in ["kb_labor"]' in expr

    def test_kb_filter_passed(self, fake):
        retrieve("试用期多久", kb_ids=["kb_civil"], client=fake, embedder=DummyEmbedder(),
                 use_rerank=False, expand_parent=False)
        call = fake.calls[0]
        assert call["reqs"][0].expr == 'kb_id in ["kb_civil"]'

    def test_empty_question_short_circuits(self, fake):
        result = retrieve("   ", client=fake, embedder=DummyEmbedder())
        assert result.hits == []
        assert fake.calls == []

    def test_elapsed_and_debug(self, fake):
        result = retrieve("试用期多久", client=fake, embedder=DummyEmbedder(),
                          reranker=DummyReranker(), expand_parent=False)
        assert result.elapsed_ms >= 0
        assert result.debug["candidates"] == 3


class TestMetrics:
    def _hits(self, articles: list[str]) -> list[Hit]:
        return [
            Hit(chunk_id=f"c{i}", parent_id=f"p{i}", kb_id="", doc_id="", content="",
                title="", article_no=a, chapter_path="", law_name="L", source_file="")
            for i, a in enumerate(articles)
        ]

    def test_recall_at_k(self):
        hits = self._hits(["第一条", "第二条", "第三条"])
        assert recall_at_k(hits, {"L#第二条"}, 5) == 1.0
        assert recall_at_k(hits, {"L#第三条"}, 2) == 0.0
        assert recall_at_k(hits, {"L#第二条", "L#第九条"}, 5) == 0.5

    def test_recall_empty_expected(self):
        assert recall_at_k(self._hits(["第一条"]), set(), 3) == 0.0

    def test_mrr(self):
        assert mrr(self._hits(["第一条", "第二条"]), {"L#第二条"}) == 0.5
        assert mrr(self._hits(["第九条"]), {"L#第二条"}) == 0.0


# ---------------------------------------------------------------------------
# retrieve_multi：并行分检 + 按法条合并去重（生产与 recall_eval 共用）
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, hits, reranked=True):
        self.hits = hits
        self.reranked = reranked


def _hit(chunk_id, parent, score=0.5, rerank=None, article="第一条"):
    return Hit(
        chunk_id=chunk_id, parent_id=parent, kb_id="kb_labor", doc_id="D",
        content="内容", title="标题", article_no=article, chapter_path="第一章",
        law_name="中华人民共和国劳动合同法", source_file="a.md",
        score=score, rerank_score=rerank,
    )


class TestRetrieveMulti:
    def _settings(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            retrieval=SimpleNamespace(
                topk_for=lambda c: {"simple": 3, "medium": 5, "complex": 8}[c],
                rrf_k=60,
            )
        )

    def test_merge_dedupes_by_parent_not_chunk(self):
        """同一法条（parent 相同）的不同子块从不同子查询召回，只保留最高分一条。"""
        from app.retrieval.service import retrieve_multi

        def fn(q, **kw):
            if "加班" in q:
                return _FakeResult([_hit("c1", "p1", rerank=0.9), _hit("c2", "p2", rerank=0.5)])
            return _FakeResult([_hit("c3", "p1", rerank=0.7), _hit("c4", "p3", rerank=0.4)])

        res = retrieve_multi(
            ["加班费怎么算", "工时有什么限制"],
            complexity="simple",  # topK=3
            settings=self._settings(), retrieve_fn=fn,
        )
        parents = [h.parent_id for h in res.hits]
        assert parents == ["p1", "p2", "p3"]
        # p1 保留的是两个子查询里分数更高的 c1
        assert res.hits[0].chunk_id == "c1"

    def test_truncates_to_final_topk(self):
        from app.retrieval.service import retrieve_multi

        def fn(q, **kw):
            return _FakeResult([_hit(f"c{i}", f"p{i}", rerank=1 - i * 0.1) for i in range(5)])

        res = retrieve_multi(["q1", "q2"], complexity="simple", settings=self._settings(), retrieve_fn=fn)
        assert len(res.hits) == 3  # simple → topK=3

    def test_single_subquery_no_thread_pool(self):
        from app.retrieval.service import retrieve_multi

        def fn(q, **kw):
            assert q == "唯一子查询"
            return _FakeResult([_hit("c1", "p1", rerank=1.0)])

        res = retrieve_multi(["唯一子查询"], complexity="medium", settings=self._settings(), retrieve_fn=fn)
        assert len(res.hits) == 1
        assert res.mode == "multi"

    def test_partial_failure_keeps_rest(self):
        """一个子查询抛异常不拖垮整体；全部失败才抛。"""
        from app.retrieval.service import retrieve_multi

        def fn(q, **kw):
            if "坏" in q:
                raise RuntimeError("milvus down")
            return _FakeResult([_hit("c1", "p1", rerank=0.8)])

        res = retrieve_multi(["坏问题", "好问题"], complexity="simple",
                             settings=self._settings(), retrieve_fn=fn)
        assert [h.chunk_id for h in res.hits] == ["c1"]
        assert res.debug["failed_subqueries"] == 1

        def fn_all_bad(q, **kw):
            raise RuntimeError("milvus down")

        with pytest.raises(RuntimeError):
            retrieve_multi(["q1", "q2"], complexity="simple",
                           settings=self._settings(), retrieve_fn=fn_all_bad)

    def test_empty_subqueries_raises(self):
        from app.retrieval.service import retrieve_multi

        with pytest.raises(ValueError):
            retrieve_multi(["  "], complexity="simple", settings=self._settings(), retrieve_fn=lambda q, **kw: _FakeResult([]))


    def test_anchor_query_joins_fusion(self):
        """原句作为锚点投票人参与 RRF：子查询漏掉的条可被原句路补回。"""
        from app.retrieval.service import retrieve_multi

        seen: list[str] = []

        def fn(q, **kw):
            seen.append(q)
            if q == "原句":
                # 只有原句路召回 p9
                return _FakeResult([_hit("c9", "p9", rerank=0.95)])
            return _FakeResult([_hit("c1", "p1", rerank=0.9)])

        res = retrieve_multi(
            ["子问题A"], anchor_query="原句",
            complexity="simple", settings=self._settings(), retrieve_fn=fn,
        )
        assert set(seen) == {"子问题A", "原句"}
        assert {h.parent_id for h in res.hits} == {"p1", "p9"}

    def test_rrf_merge_rank_based(self):
        """RRF 按名次融合：两路各自的第 1 名并列在前，不受绝对分数量纲影响。"""
        from app.retrieval.service import retrieve_multi

        def fn(q, **kw):
            if q == "q1":
                # 分数量纲整体偏高的一路
                return _FakeResult([_hit("a1", "pa", rerank=0.99), _hit("a2", "pb", rerank=0.98)])
            # 分数整体偏低的一路：pb 是它的第 1 名
            return _FakeResult([_hit("b1", "pb", rerank=0.11), _hit("b2", "pc", rerank=0.10)])

        res = retrieve_multi(["q1", "q2"], complexity="simple",
                             settings=self._settings(), retrieve_fn=fn)
        parents = [h.parent_id for h in res.hits]
        # pa/pb 都是各自路的第 1 名（RRF 并列），pc 两路第 2 → 前二必须是 {pa,pb}
        assert set(parents[:2]) == {"pa", "pb"}
        assert parents[2] == "pc"

    def test_rerank_prefers_parent_best_child(self):
        """retrieve() 重排前按 parent 预去重：同条多子块只送一条进 cross-encoder。"""
        rows = [
            _entity("c1", "p1", "同条子块1"),
            _entity("c2", "p1", "同条子块2"),
            _entity("c3", "p2", "另一条"),
        ]
        reranker = DummyReranker()
        res = retrieve(
            "某问题", complexity="simple",
            embedder=DummyEmbedder(), reranker=reranker, client=FakeMilvus(rows),
            expand_parent=False,
        )
        assert len({h.parent_id for h in res.hits}) == len(res.hits)


# ---------------------------------------------------------------------------
# 法域过滤低置信回退：law_hints 判错法域时不能把正确法条整条筛掉
# ---------------------------------------------------------------------------
class FilterAwareMilvus:
    """按过滤表达式返回不同候选：带 law_name 过滤返回弱相关行，否则返回强相关行。"""

    def __init__(self, filtered_rows, unfiltered_rows):
        self.filtered_rows = filtered_rows
        self.unfiltered_rows = unfiltered_rows
        self.exprs: list[str] = []

    def hybrid_search(self, *, reqs, **kw):
        expr = reqs[0].expr or ""
        self.exprs.append(expr)
        rows = self.filtered_rows if "law_name" in expr else self.unfiltered_rows
        return [[{"entity": r, "distance": 0.5 - i * 0.01} for i, r in enumerate(rows)]]


class TestLawFilterRescue:
    QUERY = "老板让我加班一天最多几小时"

    def _weak_row(self):
        # 与 query 几乎无字面重合（DummyReranker 按字符重合打分 → 低分）
        return _entity("w1", "pw", "诉讼时效中断中止的制度规定", law="中华人民共和国劳动合同法")

    def _strong_row(self):
        return _entity("s1", "ps", "加班时间每日不得超过一小时的小时上限",
                       law="中华人民共和国劳动法", article="第四十一条")

    def test_rescue_triggers_on_low_confidence(self, settings_factory):
        client = FilterAwareMilvus([self._weak_row()], [self._strong_row()])
        res = retrieve(
            self.QUERY, complexity="simple",
            law_hints=["中华人民共和国劳动合同法"],
            embedder=DummyEmbedder(), reranker=DummyReranker(), client=client,
            expand_parent=False,
            settings=settings_factory(retrieval_law_filter=True, retrieval_law_rescue_score=0.5),
        )
        # 两次检索：带法域过滤 + 回退的无过滤
        assert len(client.exprs) == 2
        assert "law_name" in client.exprs[0] and "law_name" not in client.exprs[1]
        # 回退把被错误过滤掉的劳动法第四十一条救了回来
        assert res.hits[0].law_article == "中华人民共和国劳动法#第四十一条"

    def test_no_rescue_when_confident(self, settings_factory):
        strong = _entity("s1", "ps", "老板让我加班一天最多几小时的上限规定",
                         law="中华人民共和国劳动合同法")
        client = FilterAwareMilvus([strong], [self._weak_row()])
        res = retrieve(
            self.QUERY, complexity="simple",
            law_hints=["中华人民共和国劳动合同法"],
            embedder=DummyEmbedder(), reranker=DummyReranker(), client=client,
            expand_parent=False,
            settings=settings_factory(retrieval_law_filter=True, retrieval_law_rescue_score=0.5),
        )
        assert len(client.exprs) == 1  # top1 高置信，不触发回退

    def test_rescue_disabled_by_zero(self, settings_factory):
        client = FilterAwareMilvus([self._weak_row()], [self._strong_row()])
        retrieve(
            self.QUERY, complexity="simple",
            law_hints=["中华人民共和国劳动合同法"],
            embedder=DummyEmbedder(), reranker=DummyReranker(), client=client,
            expand_parent=False,
            settings=settings_factory(retrieval_law_filter=True, retrieval_law_rescue_score=0.0),
        )
        assert len(client.exprs) == 1


# ---------------------------------------------------------------------------
# 检索选路分流：simple / 单跳 dense，多跳 / 跨法 hybrid；低置信回退 hybrid
# ---------------------------------------------------------------------------

class TestResolveMode:
    def test_routing_disabled_always_hybrid(self, settings_factory):
        s = settings_factory(retrieval_mode_routing=False)
        assert resolve_mode(complexity="simple", rewrite_type="direct", settings=s) == "hybrid"
        assert resolve_mode(complexity="complex", rewrite_type="decompose", settings=s) == "hybrid"

    def test_simple_single_hop_uses_dense(self, settings_factory):
        s = settings_factory(retrieval_mode_routing=True)
        assert resolve_mode(complexity="simple", rewrite_type="direct", settings=s) == "dense"
        assert resolve_mode(complexity="medium", rewrite_type="simplify", settings=s) == "dense"

    def test_multi_hop_and_cross_law_use_hybrid(self, settings_factory):
        s = settings_factory(retrieval_mode_routing=True)
        assert resolve_mode(complexity="medium", rewrite_type="decompose", settings=s) == "hybrid"
        assert resolve_mode(complexity="complex", rewrite_type="direct", settings=s) == "hybrid"
        assert resolve_mode(
            complexity="simple", rewrite_type="direct",
            law_hints=["中华人民共和国劳动合同法", "中华人民共和国劳动法"], settings=s,
        ) == "hybrid"
        assert resolve_mode(
            complexity="simple", rewrite_type="direct", route="cross_law", settings=s,
        ) == "hybrid"


class ModeAwareMilvus:
    """dense 检索返回弱相关行，hybrid 检索返回强相关行（用于回退断言）。"""

    def __init__(self, dense_rows, hybrid_rows):
        self.dense_rows = dense_rows
        self.hybrid_rows = hybrid_rows
        self.calls: list[str] = []

    def search(self, **kw):
        self.calls.append("dense")
        return [[{"entity": r, "distance": 0.9 - i * 0.01}
                 for i, r in enumerate(self.dense_rows)]]

    def hybrid_search(self, **kw):
        self.calls.append("hybrid")
        return [[{"entity": r, "distance": 0.5 - i * 0.01}
                 for i, r in enumerate(self.hybrid_rows)]]


class TestModeRoutingRetrieve:
    QUERY = "老板让我加班一天最多几小时"

    def _weak_row(self):
        return _entity("w1", "pw", "诉讼时效中断中止的制度规定")

    def _strong_row(self):
        # 正文包含原句全部字符，保证 DummyReranker 打出高置信分
        return _entity("s1", "ps", f"{self.QUERY} 加班时间每日不得超过一小时的上限规定",
                       law="中华人民共和国劳动法", article="第四十一条")

    def test_auto_simple_uses_dense(self, settings_factory):
        client = ModeAwareMilvus([self._strong_row()], [self._weak_row()])
        res = retrieve(
            self.QUERY, complexity="simple", rewrite_type="direct", mode="auto",
            embedder=DummyEmbedder(), reranker=DummyReranker(), client=client,
            expand_parent=False,
            settings=settings_factory(retrieval_mode_routing=True,
                                      retrieval_dense_rescue_score=0.0),
        )
        assert client.calls == ["dense"]
        assert res.mode == "dense"
        assert res.debug["mode_rescued"] is False

    def test_auto_decompose_uses_hybrid(self, settings_factory):
        client = ModeAwareMilvus([self._weak_row()], [self._strong_row()])
        res = retrieve(
            self.QUERY, complexity="medium", rewrite_type="decompose", mode="auto",
            embedder=DummyEmbedder(), reranker=DummyReranker(), client=client,
            expand_parent=False,
            settings=settings_factory(retrieval_mode_routing=True),
        )
        assert client.calls == ["hybrid"]
        assert res.mode == "hybrid"

    def test_low_confidence_dense_falls_back_to_hybrid(self, settings_factory):
        client = ModeAwareMilvus([self._weak_row()], [self._strong_row()])
        res = retrieve(
            self.QUERY, complexity="simple", rewrite_type="direct", mode="auto",
            embedder=DummyEmbedder(), reranker=DummyReranker(), client=client,
            expand_parent=False,
            settings=settings_factory(retrieval_mode_routing=True,
                                      retrieval_dense_rescue_score=0.5),
        )
        assert client.calls == ["dense", "hybrid"]
        assert res.mode == "hybrid"
        assert res.debug["mode_rescued"] is True
        assert res.hits[0].law_article == "中华人民共和国劳动法#第四十一条"

    def test_confident_dense_skips_fallback(self, settings_factory):
        client = ModeAwareMilvus([self._strong_row()], [self._weak_row()])
        res = retrieve(
            self.QUERY, complexity="simple", rewrite_type="direct", mode="auto",
            embedder=DummyEmbedder(), reranker=DummyReranker(), client=client,
            expand_parent=False,
            settings=settings_factory(retrieval_mode_routing=True,
                                      retrieval_dense_rescue_score=0.5),
        )
        assert client.calls == ["dense"]
        assert res.debug["mode_rescued"] is False

    def test_explicit_dense_never_rescues(self, settings_factory):
        """显式 mode=dense 是消融对照，不能被低置信回退污染。"""
        client = ModeAwareMilvus([self._weak_row()], [self._strong_row()])
        res = retrieve(
            self.QUERY, complexity="simple", mode="dense",
            embedder=DummyEmbedder(), reranker=DummyReranker(), client=client,
            expand_parent=False,
            settings=settings_factory(retrieval_mode_routing=True,
                                      retrieval_dense_rescue_score=0.5),
        )
        assert client.calls == ["dense"]
        assert res.debug["mode_rescued"] is False
