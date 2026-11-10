"""LA-N3 双领域能力：法律库 + 非法律产品库（全离线）。

验证法律专属能力可配置：Prompt、条号标签、来源格式、路由分支都由
``kb_profile_map`` 决定；未配置的知识库保持原法律链路不变。
"""

from __future__ import annotations

from pathlib import Path

from app.domain_profiles import (
    GENERIC,
    LEGAL,
    MIXED,
    get_profile,
    parse_kb_profile_map,
    profile_for_kb,
    profile_for_kbs,
)
from app.generation.prompts import ContextSource, build_messages
from app.ingest.pipeline import prepare
from app.orchestration.models import IntentResult
from app.orchestration.router import route as decide_route
from config import Settings

GENERIC_DOC = Path(__file__).resolve().parents[1] / "data" / "generic" / "nova_purifier_manual.md"
LEGAL_DOC = Path(__file__).resolve().parents[1] / "data" / "raw" / "中华人民共和国劳动合同法.md"


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        kb_profile_map='{"kb_labor": "legal", "kb_civil": "legal", "kb_generic": "generic"}',
    )


def test_profile_map_parsing_and_fallback():
    assert parse_kb_profile_map('{"kb_generic": "generic"}') == {"kb_generic": "generic"}
    assert parse_kb_profile_map("{bad json") == {}
    assert parse_kb_profile_map('{"kb_x": "unknown"}') == {}

    settings = _settings()
    assert profile_for_kb("kb_generic", settings).name == GENERIC
    assert profile_for_kb("kb_labor", settings).name == LEGAL
    assert profile_for_kb("kb_unmapped", settings).name == LEGAL
    assert profile_for_kbs(["kb_generic"], settings).name == GENERIC
    assert profile_for_kbs(["kb_labor", "kb_generic"], settings).name == MIXED
    assert profile_for_kbs([], settings).name == LEGAL


def test_generic_prompt_uses_section_citation_not_article_numbers():
    prompt = get_profile(GENERIC).system_prompt
    assert "《资料标题》·章节/小节" in prompt
    assert "禁止编造条号" in prompt
    assert "系统 Prompt" in prompt  # 资料边界仍在
    assert "第X条" in prompt  # 只作为禁止项出现
    messages = build_messages(
        "滤芯多久换一次",
        context_block="【资料1】《Nova P2 说明书》·滤芯更换\n红灯常亮时更换滤芯。",
        system_prompt=prompt,
    )
    assert messages[0]["content"] == prompt
    assert "不可信数据" in messages[-1]["content"]


def test_generic_ingest_uses_sections_and_no_fake_article_numbers():
    prepared = prepare(GENERIC_DOC, profile="generic")
    assert prepared.doc.articles == []
    assert prepared.chunks
    for chunk in prepared.chunks:
        assert chunk.article_no == ""
        assert "第" not in chunk.title or "第" in chunk.chapter_path
        assert not chunk.article_no.startswith("段")
        assert chunk.effective_from == "" and chunk.effective_to == ""

    sections = {c.chapter_path for c in prepared.chunks if c.chunk_type == "parent"}
    assert "滤芯更换" in sections
    assert "安全警告" in sections


def test_legal_ingest_keeps_article_labels():
    prepared = prepare(LEGAL_DOC, profile="legal")
    assert prepared.doc.articles
    parents = [c for c in prepared.chunks if c.chunk_type == "parent"]
    assert parents and all(c.article_no.startswith("第") for c in parents)


def test_generic_context_source_title_omits_law_dates():
    source = ContextSource(
        law="Nova 便携净化器说明书",
        chapter_path="滤芯更换",
        article_no="",
        content="红灯常亮时更换滤芯。",
        domain="generic",
        effective_from="2020-01-01",
        version=3,
    )
    assert source.title == "《Nova 便携净化器说明书》·滤芯更换"
    legal = ContextSource(
        law="中华人民共和国劳动合同法",
        chapter_path="第四章",
        article_no="第三十九条",
        content="x",
        effective_from="2013-07-01",
        version=1,
    )
    assert "施行" in legal.title and "第三十九条" in legal.title


def test_router_treats_generic_kb_as_product_domain():
    settings = _settings()
    generic = decide_route(
        IntentResult(intent="other"),
        selected_kb_ids=["kb_generic"],
        settings=settings,
    )
    assert generic.route == "pure_vector"
    assert generic.kb_ids == ["kb_generic"]
    assert generic.use_milvus is True

    legal = decide_route(
        IntentResult(intent="other"),
        selected_kb_ids=["kb_labor"],
        settings=settings,
    )
    assert legal.route == "no_data"

    # 产品库不做“法域不匹配”拦截：law_hints 非空也照常检索
    hinted = decide_route(
        IntentResult(intent="labor_law", law_hints=["中华人民共和国劳动合同法"]),
        selected_kb_ids=["kb_generic"],
        settings=settings,
    )
    assert hinted.route != "no_data"
    assert hinted.kb_ids == ["kb_generic"]


def test_mixed_selection_keeps_both_citation_styles():
    prompt = get_profile(MIXED).system_prompt
    assert "《法名》第X条" in prompt
    assert "《资料标题》·章节/小节" in prompt
