"""单查询路径「正确法条到底排第几」探针。

用途：判定单路截断（@5/@8 丢分）能否靠放大 topK 或加深重排池救回来。
做法：用很大的 topk（默认 20，池子会被 pool_for 拉到 400+）跑一次生产同款
单查询检索，打印每个期望法条在重排后的真实位置：

- 位置 ≤ 8：放大 topK 可以救，且前 N 条不变（截断是后缀，前缀稳定）；
- 位置 > 8：@8 永远数不到（recall@8 只看前 8 条），只能靠换查询/换重排；
- 未召回：连候选池里都没有，topK 与重排深度都无效，属查询或法域问题。

    uv run python scripts/single_hop_gold_probe.py q23 q27 q86
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.service import retrieve  # noqa: E402
from config import get_settings  # noqa: E402

CASE_PATH = ROOT / "eval" / "qa_test_cases.jsonl"
INTENT_PATH = ROOT / "eval" / "results" / "intent_pipeline_p8.json"
PROBE_TOPK = 20


def main() -> int:
    ids = sys.argv[1:]
    s = get_settings()
    cases: dict[str, dict] = {}
    for line in CASE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            cases[row["id"]] = row
    intent = json.loads(INTENT_PATH.read_text(encoding="utf-8"))
    targets = ids or [c for c in sorted(cases) if intent[c].get("rewrite_type") != "decompose"]
    for cid in targets:
        case, it = cases[cid], intent[cid]
        hints = list(it.get("law_hints") or []) if s.retrieval_law_filter else None
        res = retrieve(
            case["question"],
            kb_ids=case.get("kb_ids") or None,
            complexity=it.get("complexity") or "medium",
            topk=PROBE_TOPK,
            law_hints=hints,
            rewrite_type=it.get("rewrite_type") or "direct",
            mode="auto",
            settings=s,
        )
        pos = {h.law_article: i for i, h in enumerate(res.hits, start=1)}
        prod_topk = s.retrieval.topk_for(it.get("complexity") or "medium")
        detail = "  ".join(
            f"{a}@{pos.get(a, '未召回')}" for a in case["expected_articles"]
        )
        print(
            f"{cid} 生产topK={prod_topk} 探针池={res.pool} "
            f"探针{PROBE_TOPK}条 gold位置={detail}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
