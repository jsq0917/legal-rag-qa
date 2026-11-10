"""法条时效：日期提取、整型互转、检索过滤、入库回填。"""

from __future__ import annotations

from datetime import date

from app.generation.context import SourceRef
from app.generation.prompts import ContextSource
from app.ingest.pipeline import prepare, rebind_ids
from app.ingest.versioning import date_to_int, extract_effective_from, int_to_date, normalize_date
from app.retrieval.hybrid import build_filter
from app.retrieval.service import _as_of_int


class TestDateExtraction:
    def test_arabic_date(self):
        text = "第五十条 本法自2021年1月1日起施行。"
        assert extract_effective_from(text) == "2021-01-01"

    def test_padding_and_last_match(self):
        text = "旧法自1999年1月1日起施行。\n新法自2021年12月3日起施行。"
        assert extract_effective_from(text) == "2021-12-03"

    def test_missing_date(self):
        assert extract_effective_from("本法自公布之日起施行。") == ""

    def test_day_month_padding(self):
        assert extract_effective_from("自2021年1月1日起施行") == "2021-01-01"


class TestDateConversion:
    def test_round_trip(self):
        assert date_to_int("2021-01-01") == 20210101
        assert int_to_date(20210101) == "2021-01-01"

    def test_invalid(self):
        assert date_to_int("") == 0
        assert date_to_int("2021-13-01") == 0
        assert int_to_date(0) == ""
        assert int_to_date(20211301) == "2021-13-01"  # 仅格式化，不校验业务合法性

    def test_normalize_variants(self):
        assert normalize_date("2021/1/1") == "2021-01-01"
        assert normalize_date("2021年1月1日") == "2021-01-01"
        assert normalize_date("2021-02-30") == "2021-02-30"  # 只做格式校验


class TestEffectiveFilter:
    def test_filter_expression(self):
        expr = build_filter(["kb_labor"], ["中华人民共和国劳动法"], effective_as_of=20260101)
        assert "law_name in" in expr
        assert "(effective_from == 0 or effective_from <= 20260101)" in expr
        assert "(effective_to == 0 or effective_to >= 20260101)" in expr

    def test_filter_disabled(self):
        expr = build_filter(["kb_labor"], ["中华人民共和国劳动法"])
        assert "effective_from" not in expr

    def test_as_of_defaults_to_today(self, settings_factory):
        s = settings_factory(retrieval_filter_expired=True, retrieval_effective_as_of="")
        assert _as_of_int(s) == date_to_int(date.today().isoformat())

    def test_as_of_explicit_and_switch(self, settings_factory):
        s = settings_factory(retrieval_filter_expired=True, retrieval_effective_as_of="2020-01-01")
        assert _as_of_int(s) == 20200101
        off = settings_factory(retrieval_filter_expired=False)
        assert _as_of_int(off) == 0


class TestIngestEffectiveDate:
    LAW_TEXT = """# 中华人民共和国测试法

第一条 为了测试，制定本法。

第二条 本法自2021年1月1日起施行。
"""

    def test_prepare_and_rebind(self, tmp_path):
        path = tmp_path / "中华人民共和国测试法.md"
        path.write_text(self.LAW_TEXT, encoding="utf-8")
        prepared = prepare(path)
        assert prepared.effective_from == "2021-01-01"
        assert all(c.effective_from == "" for c in prepared.chunks)  # prepare 阶段未绑定

        bound = rebind_ids(
            prepared,
            doc_id="doc1",
            kb_id="kb_labor",
            effective_from=prepared.effective_from,
            effective_to="",
            version=3,
        )
        assert bound.effective_from == "2021-01-01"
        assert all(c.effective_from == "2021-01-01" for c in bound.chunks)
        assert all(c.version == 3 for c in bound.chunks)

    def test_document_without_date(self, tmp_path):
        path = tmp_path / "中华人民共和国测试法.md"
        path.write_text("# 中华人民共和国测试法\n\n第一条 本法自公布之日起施行。\n", encoding="utf-8")
        assert prepare(path).effective_from == ""


class TestPromptAnnotation:
    def test_context_title_carries_effective_and_version(self):
        src = ContextSource(
            law="中华人民共和国民法典",
            article_no="第七百零五条",
            content="租赁期限不得超过二十年。",
            effective_from="2021-01-01",
            version=2,
        )
        assert src.title == "《中华人民共和国民法典》·第七百零五条（2021-01-01 施行，v2）"

    def test_context_title_unchanged_without_metadata(self):
        src = ContextSource(law="中华人民共和国劳动合同法", article_no="第三十九条", content="x")
        assert src.title == "《中华人民共和国劳动合同法》·第三十九条"

    def test_source_ref_exports_metadata(self):
        ref = SourceRef(
            law="中华人民共和国民法典",
            chapter_path="",
            article_no="第七百零五条",
            score=0.9,
            snippet="租赁期限不得超过二十年。",
            effective_from="2021-01-01",
            version=2,
        )
        data = ref.to_dict()
        assert data["effective_from"] == "2021-01-01"
        assert data["version"] == 2
