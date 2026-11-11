"""multi_hop 召回侧候选改动对比（只测召回，不调 LLM、不跑 RAGAS）。

基线 = 线上口径（意图缓存里的 rewrite_type/sub_queries，多跳才走 retrieve_multi）。
候选：
  A  topk+3：多跳问句把最终 topK 放大 3 条（重排后截断是主要丢分点）
  B  改写作第二投票人：非 decompose 时也用意图改写句与原句交错融合
  C  A+B
  D  只对多诉求问句 (含并列连接词) 用 B，其余保持基线
产出每条样本的 recall@5/@8 与命中列表，便于对比哪条样本升降。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.eval.recall_eval import _CachingEmbedder  # noqa: E402
from app.ingest.embedder import get_embedder  # noqa: E402
from app.retrieval import refs  # noqa: E402
from app.retrieval.rerank import get_reranker  # noqa: E402
from app.retrieval.service import (  # noqa: E402
    _as_of_int,
    recall_at_k,
    retrieve,
    retrieve_multi,
)
from config import get_settings  # noqa: E402

CONNECTORS = ("还", "同时", "以及", "另外", "而且", "并且", "、")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def multi_issue(question: str) -> bool:
    """多诉求启发式：含逗号/分号分句，且出现并列/递进连接词或两个问号。"""
    q = question.strip()
    if question.count("？") + question.count("?") > 1:
        return True
    if not any(sep in q for sep in ("，", ",", "；", ";")):
        return False
    return any(c in q for c in CONNECTORS)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval/results/multihop_recall_probe.json"
    settings = get_settings()
    cases = {r["id"]: r for r in load_jsonl(ROOT / "eval/qa_test_cases.jsonl")}
    intent_path = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "eval/results/intent_pipeline.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    ids = [cid for cid, c in cases.items() if c["type"] == (sys.argv[2] if len(sys.argv) > 2 else "multi_hop")]
    embedder = _CachingEmbedder(get_embedder())
    reranker = get_reranker(settings)

    def run(cid: str, variant: str) -> dict:
        case, it = cases[cid], intent[cid]
        expected = set(case["expected_articles"])
        complexity = it.get("complexity") or "medium"
        hints = list(it.get("law_hints") or []) if settings.retrieval_law_filter else None
        subs = list(it.get("sub_queries") or [])
        rewrite = subs[0] if subs else case["question"]
        base_topk = settings.retrieval.topk_for(complexity)
        use_multi = it.get("rewrite_type") == "decompose" and len(subs) > 1
        common = dict(
            kb_ids=case.get("kb_ids") or None,
            complexity=complexity,
            law_hints=hints,
            embedder=embedder,
            reranker=reranker,
        )

        def multi(voters: list[str], topk: int) -> dict:
            return retrieve_multi(
                voters,
                anchor_query=case["question"],
                mode="hybrid",
                topk=topk,
                settings=settings,
                retrieve_fn=lambda q, **kw: retrieve(
                    q, embedder=embedder, reranker=reranker, **kw
                ),
                **{k: v for k, v in common.items() if k not in ("embedder", "reranker")},
            )

        def single(topk: int) -> dict:
            return retrieve(
                case["question"], mode="auto",
                rewrite_type=it.get("rewrite_type") or "direct",
                settings=settings, **common, **({"topk": topk} if topk else {}),
            )

        if variant == "base" or (variant == "D" and not multi_issue(case["question"])):
            res = multi(subs, base_topk) if use_multi else single(base_topk)
        elif variant == "A":
            res = multi(subs, base_topk + 3) if use_multi else single(base_topk + 3)
        elif variant in ("M", "M1", "M2"):
            # 只给多路加 topK：单查询路径不动，用来验证"多路配额"单独的效果
            bonus = {"M": 3, "M1": 1, "M2": 2}[variant]
            res = multi(subs, base_topk + bonus) if use_multi else single(base_topk)
        elif variant in ("B", "C", "D"):
            voters = subs if use_multi else ([rewrite] if rewrite and rewrite != case["question"] else [])
            if not voters:
                res = single(base_topk)
            else:
                res = multi(voters, base_topk + 3 if variant == "C" else base_topk)
        elif variant in ("E", "F"):
            bonus = 3 if variant == "F" else 0
            res = multi(subs, base_topk + bonus) if use_multi else single(base_topk + bonus)
        else:
            raise ValueError(variant)

        hits = res.hits
        refs_added = 0
        if variant in ("E", "F"):
            hits, refs_added = refs.expand_hits_with_refs(
                hits,
                topk=len(hits),
                kb_ids=case.get("kb_ids") or None,
                max_sources=settings.retrieval_ref_max_sources,
                max_added=settings.retrieval_ref_max_added,
                as_of=_as_of_int(settings),
                settings=settings,
            )
        return {
            "retrieved": [h.law_article for h in hits],
            "recall@5": round(recall_at_k(hits, expected, 5), 4),
            "recall@8": round(recall_at_k(hits, expected, 8), 4),
            "ctx_chars": sum(len(h.parent_content or h.content) for h in hits),
            "refs_added": refs_added,
        }

    variants = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("base", "A", "B", "C", "D", "E", "F")
    rows: dict[str, dict] = {}
    started = time.perf_counter()
    for n, cid in enumerate(ids, start=1):
        rows[cid] = {"type": cases[cid]["type"], "question": cases[cid]["question"],
                     "expected": cases[cid]["expected_articles"],
                     "rewrite_type": intent[cid].get("rewrite_type"),
                     "multi_issue": multi_issue(cases[cid]["question"])}
        for v in variants:
            rows[cid][v] = run(cid, v)
        print(
            f"[{n}/{len(ids)}] {cid} " + " ".join(f"{v}={rows[cid][v]['recall@5']:.2f}" for v in variants),
            flush=True,
        )

    summary = {}
    for v in variants:
        summary[v] = {
            "recall@5": round(statistics.fmean(r[v]["recall@5"] for r in rows.values()), 4),
            "recall@8": round(statistics.fmean(r[v]["recall@8"] for r in rows.values()), 4),
            "avg_ctx_chars": round(statistics.fmean(r[v]["ctx_chars"] for r in rows.values()), 1),
            "samples_improved": [], "samples_worse": [],
        }
    baseline = rows
    for cid in ids:
        for v in variants[1:]:
            d = rows[cid][v]["recall@5"] - rows[cid]["base"]["recall@5"]
            if d > 0:
                summary[v]["samples_improved"].append(cid)
            elif d < 0:
                summary[v]["samples_worse"].append(cid)

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "samples": len(ids),
            "intent_source": str(intent_path),
            "variants": {
                "base": "线上口径",
                "A": "最终 topK + 3",
                "B": "非 decompose 时用改写句作第二投票人",
                "C": "A + B",
                "D": "仅多诉求问句用 B",
                "E": "基线 + 沿命中法条互引扩展（补被引条，挤出尾部）",
                "F": "A + E（放大 topK 后再扩展，不挤掉尾部）",
            },
            "seconds": round(time.perf_counter() - started, 1),
        },
        "summary": summary,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
