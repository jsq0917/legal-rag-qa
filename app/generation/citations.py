"""答案引用在线校验（复用 ``ragas_eval --citations`` 的判定口径）。

离线评估里 ``citation_report`` 用「《法名》第X条」正则把答案里的引用抽出来，
再与检索来源 / 期望条号做集合比对。线上生成后做的是同一件事，只是比对基准
换成**本次检索来源**：

* 答案引用的条号落在来源集合内 → 视为已核实；
* 落在集合外 → 追加一行提示（不阻断回答），并写进 SSE ``meta.citation_check``，
  前端可据此打标；
* 答案一条引用都没有 → 无可校验项，``ok=True`` 但 ``cited`` 为空。

这里只做**字面**核对：条号存在性 ≠ 条文内容正确，后者仍由人工/评估兜底。
"""

from __future__ import annotations

import re
from typing import Any

#: 与 ``app/eval/ragas_eval.py`` 共用：答案中的引用（《法名》第X条）
CITATION_RE = re.compile(r"《([^《》]{2,30})》(第[一二三四五六七八九十百零\d]+条)")

#: 补注前缀（同时用作幂等标记：答案里已出现就不再追加）
NOTE_MARK = "引用校验："

#: 补注里最多列出的条号数，超出用「等」收尾，避免长答案尾部噪音
NOTE_MAX_ITEMS = 5


def normalize_law(law: str) -> str:
    """法名归一化：答案常用简写（"劳动合同法"），来源/期望用全称，比对前拉齐。"""
    return (law or "").replace("中华人民共和国", "").strip()


def extract_citations(text: str) -> set[tuple[str, str]]:
    """抽取 ``(归一化法名, 条号)`` 集合。"""
    return {(normalize_law(law), art) for law, art in CITATION_RE.findall(text or "")}


def _source_keys(sources: list[dict[str, Any]] | None) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for src in sources or []:
        law = (src.get("law") or "").strip()
        article = (src.get("article_no") or "").strip()
        if law and article:
            keys.add((normalize_law(law), article))
    return keys


def _key_text(key: tuple[str, str]) -> str:
    return f"{key[0]}#{key[1]}"


def check_citations(
    answer: str,
    sources: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """校验答案条号是否落在检索来源内。

    返回结构（``meta.citation_check`` 原样透出）::

        {
          "ok": True,                  # 无「来源外引用」即 True
          "cited": ["劳动合同法#第十九条"],
          "verified": [...],           # 命中来源的引用
          "unverified": [...],         # 来源外的引用（需补注）
          "sources_uncited": [...],    # 来源里未被引用的条（漏标，仅统计）
          "coverage": 1.0,             # verified / cited
          "source_coverage": 0.5,      # verified / sources
        }
    """
    cited = extract_citations(answer)
    source_keys = _source_keys(sources)
    verified = cited & source_keys
    unverified = cited - source_keys
    return {
        "ok": not unverified,
        "cited": sorted(_key_text(k) for k in cited),
        "verified": sorted(_key_text(k) for k in verified),
        "unverified": sorted(_key_text(k) for k in unverified),
        "sources_uncited": sorted(_key_text(k) for k in source_keys - cited),
        "coverage": round(len(verified) / len(cited), 4) if cited else None,
        "source_coverage": round(len(verified) / len(source_keys), 4) if source_keys else None,
    }


def format_citation_note(unverified: list[str]) -> str:
    """把未核实的条号拼成一段引用提示（不含《》，避免被再次解析成引用）。"""
    items = unverified[:NOTE_MAX_ITEMS]
    text = "、".join(item.replace("#", " ") for item in items)
    if len(unverified) > NOTE_MAX_ITEMS:
        text += f" 等 {len(unverified)} 条"
    return f"\n\n> {NOTE_MARK}以下条号未出现在本次检索资料中，请以原文为准：{text}"


def append_citation_note(answer: str, check: dict[str, Any] | None) -> str:
    """按校验结果追加提示；已补注过则原样返回（幂等）。"""
    text = answer or ""
    if not check or check.get("ok", True) or not check.get("unverified"):
        return text
    if NOTE_MARK in text:
        return text
    return text + format_citation_note(list(check["unverified"]))


def strip_citation_note(answer: str) -> str:
    """剥掉尾部自动补注（缓存回放 / 写缓存前用，避免提示被固化或重复叠加）。"""
    text = answer or ""
    idx = text.rfind(f"\n\n> {NOTE_MARK}")
    if idx < 0:
        return text
    return text[:idx].rstrip()
