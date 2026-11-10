"""引用在线校验测试（纯离线）。

覆盖：引用抽取/法名归一化、来源内/来源外判定、补注格式化与幂等、
剥注后可重复校验，以及与离线 ``ragas_eval`` 共用同一判定口径。
"""

from __future__ import annotations

from app.generation.citations import (
    CITATION_RE,
    append_citation_note,
    check_citations,
    extract_citations,
    format_citation_note,
    normalize_law,
    strip_citation_note,
)

LAW = "中华人民共和国劳动合同法"
SOURCES = [{"law": LAW, "article_no": "第十九条", "kind": "law"}]


class TestExtract:
    def test_normalize_law(self):
        assert normalize_law(LAW) == "劳动合同法"
        assert normalize_law(" 中华人民共和国民法典 ") == "民法典"
        assert normalize_law("") == ""

    def test_extract_with_short_name(self):
        text = "依据《劳动合同法》第十九条，试用期不得超过六个月。"
        assert extract_citations(text) == {("劳动合同法", "第十九条")}

    def test_extract_multiple_and_arabic(self):
        text = "《中华人民共和国民法典》第577条与《劳动合同法》第三十九条。"
        assert extract_citations(text) == {("民法典", "第577条"), ("劳动合同法", "第三十九条")}

    def test_extract_ignores_bare_article(self):
        assert extract_citations("第十九条 试用期不得超过六个月") == set()

    def test_shared_regex_object(self):
        """与 ragas_eval 共用同一个正则对象，口径不会漂移。"""
        from app.eval.ragas_eval import _citation_re

        assert _citation_re() is CITATION_RE


class TestCheck:
    def test_all_citations_verified(self):
        check = check_citations("依据《劳动合同法》第十九条。", SOURCES)
        assert check["ok"] is True
        assert check["cited"] == ["劳动合同法#第十九条"]
        assert check["verified"] == ["劳动合同法#第十九条"]
        assert check["unverified"] == []
        assert check["coverage"] == 1.0
        assert check["source_coverage"] == 1.0

    def test_unverified_citation(self):
        check = check_citations("依据《劳动合同法》第三十九条。", SOURCES)
        assert check["ok"] is False
        assert check["unverified"] == ["劳动合同法#第三十九条"]
        assert check["sources_uncited"] == ["劳动合同法#第十九条"]
        assert check["coverage"] == 0.0

    def test_no_citation_is_ok(self):
        check = check_citations("建议先与单位协商。", SOURCES)
        assert check["ok"] is True
        assert check["cited"] == []
        assert check["coverage"] is None

    def test_other_law_is_unverified(self):
        check = check_citations("依据《中华人民共和国民法典》第五百七十七条。", SOURCES)
        assert check["ok"] is False
        assert check["unverified"] == ["民法典#第五百七十七条"]

    def test_source_without_article_ignored(self):
        sources = [{"law": "", "article_no": ""}, {"law": "高频问答库", "article_no": ""}]
        check = check_citations("依据《劳动合同法》第十九条。", sources)
        assert check["source_coverage"] is None
        assert check["unverified"] == ["劳动合同法#第十九条"]


class TestNote:
    def test_format_note_has_no_brackets(self):
        note = format_citation_note(["劳动合同法#第三十九条"])
        assert "劳动合同法 第三十九条" in note
        assert "《" not in note
        assert CITATION_RE.findall(note) == []  # 补注不会被再次当成引用

    def test_format_note_truncates(self):
        items = [f"劳动合同法#第{i}条" for i in range(1, 9)]
        note = format_citation_note(items)
        assert "等 8 条" in note
        assert note.count("劳动合同法") == 5

    def test_append_and_strip_roundtrip(self):
        check = check_citations("依据《劳动合同法》第三十九条。", SOURCES)
        annotated = append_citation_note("依据《劳动合同法》第三十九条。", check)
        assert "引用校验：" in annotated
        assert strip_citation_note(annotated) == "依据《劳动合同法》第三十九条。"

    def test_append_is_idempotent(self):
        check = check_citations("依据《劳动合同法》第三十九条。", SOURCES)
        once = append_citation_note("依据《劳动合同法》第三十九条。", check)
        twice = append_citation_note(once, check)
        assert once == twice

    def test_append_skipped_when_ok(self):
        answer = "依据《劳动合同法》第十九条。"
        check = check_citations(answer, SOURCES)
        assert append_citation_note(answer, check) == answer

    def test_strip_keeps_body_without_note(self):
        assert strip_citation_note("普通回答") == "普通回答"
