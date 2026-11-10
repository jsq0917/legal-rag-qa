"""法条显式互引：抽取、索引解析、多跳补检。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.retrieval import refs
from app.retrieval.hybrid import Hit, RetrievalResult
from app.retrieval.service import retrieve_multi


def _row(chunk_id, law, article_no, content="", kb_id="labour", title=None):
    return refs.ParentArticle(
        chunk_id=chunk_id,
        parent_id=chunk_id,
        kb_id=kb_id,
        doc_id=f"doc-{law}",
        content=content or f"{law}{article_no}正文",
        title=title or f"{law}·{article_no}",
        article_no=article_no,
        chapter_path="",
        law_name=law,
        source_file=f"{law}.md",
    )


def _hit(chunk_id, law, article_no, content="", score=1.0, parent_content=None):
    return Hit(
        chunk_id=chunk_id,
        parent_id=chunk_id,
        kb_id="labour",
        doc_id=f"doc-{law}",
        content=content,
        title=f"{law}·{article_no}",
        article_no=article_no,
        chapter_path="",
        law_name=law,
        source_file=f"{law}.md",
        score=score,
        parent_content=parent_content or content,
    )


class TestExtractRefs:
    LAW = "中华人民共和国劳动合同法"

    def test_local_article_list(self):
        content = "用人单位依照本法第四十条、第四十一条的规定解除劳动合同。"
        refs_ = refs.extract_refs(content, law_name=self.LAW, article_num=46)
        assert [(r.law_name, r.article_num) for r in refs_] == [(None, 40), (None, 41)]

    def test_cue_only_same_law(self):
        content = "用人单位违反第八十七条规定解除劳动合同的，应当支付赔偿金。"
        refs_ = refs.extract_refs(content, law_name=self.LAW, article_num=48)
        assert [(r.law_name, r.article_num) for r in refs_] == [(None, 87)]

    def test_explicit_law_name(self):
        content = "依照《中华人民共和国劳动合同法》第三十九条规定处理。"
        refs_ = refs.extract_refs(content, law_name="中华人民共和国劳动法", article_num=25)
        assert len(refs_) == 1
        assert refs_[0].law_name == "中华人民共和国劳动合同法"
        assert refs_[0].article_num == 39

    def test_range_keeps_endpoints(self):
        content = "本法第三十六条至第三十八条规定的解除情形。"
        refs_ = refs.extract_refs(content, law_name=self.LAW, article_num=46)
        assert [r.article_num for r in refs_] == [36, 38]

    def test_self_reference_excluded(self):
        content = "依照本法第四十六条规定支付经济补偿。"
        assert refs.extract_refs(content, law_name=self.LAW, article_num=46) == []

    def test_bare_article_without_cue_is_ignored(self):
        content = "第三十九条和第四十条规定的情形，用人单位可以解除。"
        assert refs.extract_refs(content, law_name=self.LAW, article_num=42) == []

    def test_article_item_not_treated_as_article(self):
        content = "依照本法第十四条第二款第三项规定连续订立固定期限劳动合同。"
        refs_ = refs.extract_refs(content, law_name=self.LAW, article_num=97)
        assert [r.article_num for r in refs_] == [14]


class TestArticleIndex:
    def test_resolve_local_and_alias(self):
        index = refs.ArticleIndex([
            _row("c46", "中华人民共和国劳动合同法", "第四十六条"),
            _row("c39", "中华人民共和国劳动合同法", "第三十九条"),
            _row("c25", "中华人民共和国劳动法", "第二十五条"),
        ])
        local = index.resolve(refs.ArticleRef(None, 46, "依照"), "中华人民共和国劳动合同法")
        assert local.chunk_id == "c46"
        alias = index.resolve(
            refs.ArticleRef("《劳动合同法》", 39, "依照"), "中华人民共和国劳动法"
        )
        assert alias.chunk_id == "c39"
        missing = index.resolve(
            refs.ArticleRef("中华人民共和国刑法", 1, "依照"), "中华人民共和国劳动法"
        )
        assert missing is None


class TestExpandHits:
    def _index(self):
        return refs.ArticleIndex([
            _row("c46", "中华人民共和国劳动合同法", "第四十六条"),
            _row("c42", "中华人民共和国劳动合同法", "第四十二条"),
            _row("c41", "中华人民共和国劳动合同法", "第四十一条"),
            _row("c37", "中华人民共和国劳动合同法", "第三十七条"),
            _row("c39", "中华人民共和国劳动合同法", "第三十九条"),
        ])

    def test_inserts_after_source_and_skips_existing(self):
        source = _hit(
            "c46", "中华人民共和国劳动合同法", "第四十六条",
            content="依照本法第四十条、第四十一条规定解除劳动合同。",
        )
        existing = _hit("c37", "中华人民共和国劳动合同法", "第三十七条", score=0.5)
        out, added = refs.expand_hits_with_refs(
            [source, existing], topk=8, index=self._index()
        )
        ids = [h.chunk_id for h in out]
        assert ids[:2] == ["c46", "c41"]
        assert "c37" in ids
        # c40 不在索引里，c41 被补入；c37 已存在不重复
        assert added == 1

    def test_max_added_cap(self):
        source = _hit(
            "c46", "中华人民共和国劳动合同法", "第四十六条",
            content="依照本法第三十九条、第四十一条、第四十二条规定。",
        )
        out, added = refs.expand_hits_with_refs(
            [source], topk=8, max_added=2, index=self._index()
        )
        assert added == 2
        assert len(out) == 3

    def test_kb_filter(self):
        index = refs.ArticleIndex([
            _row("c46", "中华人民共和国劳动合同法", "第四十六条"),
            _row("c41", "中华人民共和国劳动合同法", "第四十一条", kb_id="other"),
        ])
        source = _hit(
            "c46", "中华人民共和国劳动合同法", "第四十六条",
            content="依照本法第四十一条规定。",
        )
        _, added = refs.expand_hits_with_refs(
            [source], topk=8, kb_ids=["labour"], index=index
        )
        assert added == 0


class TestRetrieveMultiRefExpansion:
    def test_multi_hop_adds_referenced_article(self, monkeypatch):
        index = refs.ArticleIndex([
            _row("c46", "中华人民共和国劳动合同法", "第四十六条"),
            _row("c41", "中华人民共和国劳动合同法", "第四十一条"),
        ])
        monkeypatch.setattr(refs, "get_article_index", lambda *a, **kw: index)

        def fake_retrieve(q, **kw):
            if "经济补偿" in q:
                hit = _hit(
                    "c46", "中华人民共和国劳动合同法", "第四十六条",
                    content="依照本法第四十一条规定解除劳动合同。",
                )
            else:
                hit = _hit("c39", "中华人民共和国劳动合同法", "第三十九条")
            return RetrievalResult(
                [hit], q, kw.get("topk", 8), 12, "medium", "hybrid", True, 1,
                {"sub_queries": [q]},
            )

        settings = SimpleNamespace(
            retrieval_ref_expansion=True,
            retrieval_ref_max_sources=3,
            retrieval_ref_max_added=3,
        )
        result = retrieve_multi(
            ["经济补偿如何计算", "解除劳动合同的条件"],
            topk=8,
            settings=settings,
            retrieve_fn=fake_retrieve,
        )
        assert result.debug["refs_added"] == 1
        assert "c41" in [h.chunk_id for h in result.hits]
