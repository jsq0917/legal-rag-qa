"""QA 评估样本的加载与校验（120 条，设计方案 §7.2）。

    uv run python -m app.eval.qa_dataset            # 打印统计 + 校验条号是否存在于语料
    uv run python -m app.eval.qa_dataset --preview  # 额外打印每条期望法条的正文前 40 字

样本字段：``id / type / difficulty / colloquial / kb_ids / question / reference_answer /
expected_articles（法名#条号） / expected_laws / sub_questions``。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "eval" / "qa_test_cases.jsonl"

#: §7.2 的类型配比
EXPECTED_MIX = {"single_hop": 45, "multi_hop": 35, "complex": 25, "no_answer": 15}
REQUIRED_FIELDS = (
    "id",
    "type",
    "question",
    "reference_answer",
    "expected_articles",
    "expected_laws",
)


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """读取 JSONL，逐行解析（空行跳过）。"""
    p = path or DEFAULT_CASES
    cases: list[dict[str, Any]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{p.name}:{lineno} JSON 解析失败：{exc}") from exc
    return cases


def split_article(key: str) -> tuple[str, str]:
    """``法名#条号`` → (法名, 条号)。"""
    law, _, article = key.partition("#")
    if not law or not article:
        raise ValueError(f"expected_articles 格式应为「法名#条号」，实际：{key!r}")
    return law, article


def structural_issues(cases: list[dict[str, Any]]) -> list[str]:
    """不依赖数据库的检查：字段完整、id 唯一、类型配比、条号格式。"""
    issues: list[str] = []
    ids = [c.get("id") for c in cases]
    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        issues.append(f"id 重复：{dup}")

    mix = Counter(c.get("type") for c in cases)
    if dict(mix) != EXPECTED_MIX:
        issues.append(f"类型配比不符：期望 {EXPECTED_MIX}，实际 {dict(mix)}")

    colloquial = sum(1 for c in cases if c.get("colloquial"))
    if colloquial < 8:
        issues.append(f"口语化样本只有 {colloquial} 条（设计要求 ≥8）")

    for c in cases:
        missing = [f for f in REQUIRED_FIELDS if f not in c]
        if missing:
            issues.append(f"{c.get('id')} 缺字段：{missing}")
            continue
        if c["type"] == "no_answer":
            if c["expected_articles"]:
                issues.append(f"{c['id']} 是 no_answer 但给了 expected_articles")
        else:
            if not c["expected_articles"]:
                issues.append(f"{c['id']} 缺少 expected_articles")
            for key in c["expected_articles"]:
                try:
                    split_article(key)
                except ValueError as exc:
                    issues.append(f"{c['id']}：{exc}")
            if not c["expected_laws"]:
                issues.append(f"{c['id']} 缺少 expected_laws")
        if len(c["question"]) < 5:
            issues.append(f"{c['id']} 问题过短")
    return issues


def corpus_article_index() -> dict[str, str]:
    """``法名#条号`` → 父块正文（只取启用中的 parent 块）。"""
    from sqlalchemy import text

    from app.db.mysql import get_engine

    stmt = text(
        "SELECT title, article_no, content FROM document_chunks "
        "WHERE chunk_type = 'parent' AND is_enabled = 1"
    )
    index: dict[str, str] = {}
    with get_engine().connect() as conn:
        for title, article_no, content in conn.execute(stmt):
        # title 形如「中华人民共和国劳动合同法·第四章 …·第四十七条」
            law = (title or "").split("·")[0].strip()
            if law and article_no:
                index[f"{law}#{article_no}"] = content or ""
    return index


def corpus_issues(cases: list[dict[str, Any]], index: dict[str, str]) -> list[str]:
    """期望法条必须在语料里真实存在，否则召回率永远算不上去。"""
    issues: list[str] = []
    for c in cases:
        for key in c.get("expected_articles") or []:
            if key not in index:
                issues.append(f"{c['id']} 期望法条不在语料中：{key}")
    return issues


def report(cases: list[dict[str, Any]], *, preview: bool = False) -> int:
    print(f"样本数：{len(cases)}")
    print(f"类型配比：{dict(Counter(c['type'] for c in cases))}")
    print(f"口语化：{sum(1 for c in cases if c.get('colloquial'))} 条")
    print(f"难度分布：{dict(Counter(c.get('difficulty') for c in cases))}")
    total_articles = sum(len(c.get("expected_articles") or []) for c in cases)
    print(f"期望法条总数：{total_articles}")
    for law, n in Counter(k.split('#')[0] for c in cases for k in c.get("expected_articles") or []).items():
        print(f"  - {law}：{n} 条")

    issues = structural_issues(cases)
    try:
        index = corpus_article_index()
    except Exception as exc:  # noqa: BLE001 - 数据库没起来时只做结构检查
        print(f"\n[跳过语料校验] 数据库不可用：{exc}")
        index = {}
    if index:
        print(f"语料父块：{len(index)} 条")
        issues += corpus_issues(cases, index)

    if preview and index:
        print("\n期望法条预览（正文前 40 字）")
        for c in cases:
            for key in c.get("expected_articles") or []:
                text = index.get(key, "")
                print(f"  {c['id']} {key} | {text[:40].replace(chr(10), ' ')}")

    print()
    if issues:
        print(f"校验未通过（{len(issues)} 项）：")
        for it in issues:
            print(f"  - {it}")
        return 1
    print("校验通过：字段完整、配比正确、期望法条全部存在于语料。")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="QA 评估样本校验")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--preview", action="store_true", help="打印每条期望法条正文片段")
    args = parser.parse_args()
    raise SystemExit(report(load_cases(args.cases), preview=args.preview))


if __name__ == "__main__":
    main()
