"""multi_hop 定点诊断：逐阶段复现检索，定位期望法条丢在哪一步。

阶段口径（全部本地检索，不调 LLM、不跑 RAGAS）：
  anchor   —— 生产里 direct/hyde 的实际路径：原句单查询（mode=auto）
  intent   —— 生产里 decompose 的实际路径：意图子查询 + 原句锚点，hybrid 交错融合
  gold     —— 上限参考：数据集给定的 sub_questions + 原句锚点，hybrid 交错融合

对每个期望法条记录：
  - 各路投票人的最高名次（候选池内 / 实际传入融合的名次）
  - 生产路径最终列表里的位置
并分类丢失原因：
  topk_truncate      单查询路径：条子在重排候选池里，但超过最终 topK 被截掉
  per_subquery_quota 多路路径：条子在候选池里，但超过该路 topK，没进融合
  fusion_order       多路路径：已进融合，被交错顺序/最终截断挤出
  law_filter         带法域过滤时不在候选池，去掉过滤能进前 20 → 法域 hint 误杀
  out_of_pool        去掉法域过滤也进不了候选池 → 子查询措辞/向量/语料问题

用法：
  .\\.venv\\Scripts\\python.exe scripts/multihop_stage_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import hybrid  # noqa: E402
from app.retrieval.rerank import get_reranker  # noqa: E402
from app.retrieval.service import (  # noqa: E402
    MULTI_POOL_FACTOR,
    _as_of_int,
    _rerank_candidates,
    _supports_effective_fields,
    _default_client,
    resolve_mode,
    retrieve,
    retrieve_multi,
)
from app.ingest.embedder import get_embedder  # noqa: E402
from config import get_settings  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_of(ranked: list, expected: str) -> int | None:
    for i, hit in enumerate(ranked, start=1):
        if hit.law_article == expected:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/multihop_stage_probe.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    settings = get_settings()
    cases = {r["id"]: r for r in load_jsonl(ROOT / "eval/qa_test_cases.jsonl")}
    intent = json.loads((ROOT / "eval/results/intent_pipeline.json").read_text(encoding="utf-8"))
    mh_ids = [cid for cid, c in cases.items() if c["type"] == "multi_hop"]
    if args.limit:
        mh_ids = mh_ids[: args.limit]

    embedder = get_embedder()
    client = _default_client()
    reranker = get_reranker(settings)
    collection = settings.milvus_collection
    as_of = _as_of_int(settings)
    if as_of and not _supports_effective_fields(client, collection):
        as_of = 0
    law_filter = settings.retrieval_law_filter

    def ranked_list(query: str, *, hints: list[str], mode: str, topk: int) -> list:
        pool = settings.retrieval.pool_for(topk)
        embedding = embedder.encode([query])[0]
        cands = hybrid.search(
            client,
            collection,
            embedding,
            pool=pool,
            mode=mode,
            rrf_k=settings.retrieval.rrf_k,
            ranker=settings.retrieval_ranker,
            dense_weight=settings.retrieval.dense_weight,
            sparse_weight=settings.retrieval.sparse_weight,
            law_hints=hints,
            sparse_pool_ratio=getattr(settings, "retrieval_sparse_pool_ratio", 1.0),
            effective_as_of=as_of,
        )
        if not cands:
            return []
        return _rerank_candidates(cands, query, reranker)

    rows = []
    started = time.perf_counter()
    for n, cid in enumerate(mh_ids, start=1):
        case = cases[cid]
        it = intent[cid]
        gold_subs = [q for q in (case.get("sub_questions") or []) if q]
        intent_subs = [q for q in (it.get("sub_queries") or []) if q]
        complexity = it.get("complexity") or "medium"
        final_topk = settings.retrieval.topk_for(complexity)
        per_topk = max(int(final_topk * MULTI_POOL_FACTOR), final_topk)
        hints = list(it.get("law_hints") or []) if law_filter else []
        expected = list(case.get("expected_articles") or [])
        question = case["question"]
        use_multi = it.get("rewrite_type") == "decompose" and len(intent_subs) > 1

        # 各路投票人的完整重排列表（候选池内，未按 per_topk 截断）
        anchor_mode = resolve_mode(
            complexity=complexity,
            rewrite_type=it.get("rewrite_type") or "direct",
            law_hints=hints,
            settings=settings,
        )
        voters: dict[str, list] = {}
        voters["anchor"] = ranked_list(question, hints=hints, mode=anchor_mode, topk=final_topk)
        for i, q in enumerate(intent_subs, start=1):
            voters[f"intent{i}"] = ranked_list(q, hints=hints, mode="hybrid", topk=per_topk)
        # 上限参考：gold 子问题（anchor 复用上面的结果）
        gold_voters: dict[str, list] = {"anchor": voters["anchor"]}
        for i, q in enumerate(gold_subs, start=1):
            gold_voters[f"gold{i}"] = ranked_list(q, hints=hints, mode="hybrid", topk=per_topk)

        # 生产路径最终结果
        if use_multi:
            prod = retrieve_multi(
                intent_subs,
                complexity=complexity,
                law_hints=hints or None,
                anchor_query=question,
                mode="hybrid",
                settings=settings,
            )
            prod_hits = prod.hits
            prod_mode = "multi"
        else:
            single = retrieve(
                question,
                complexity=complexity,
                mode="auto",
                rewrite_type=it.get("rewrite_type") or "direct",
                law_hints=hints or None,
                settings=settings,
            )
            prod_hits = single.hits
            prod_mode = single.mode

        gold = retrieve_multi(
            gold_subs or intent_subs,
            complexity=complexity,
            law_hints=hints or None,
            anchor_query=question,
            mode="hybrid",
            settings=settings,
        )

        items = []
        # 非多路路径的生产检索只看原句锚点；意图改写句不参与线上召回
        attrib_voters = voters if use_multi else {"anchor": voters["anchor"]}
        for art in expected:
            best_full = None
            best_full_voter = ""
            best_passed = None
            voter_ranks: dict[str, int] = {}
            for name, rl in attrib_voters.items():
                r = rank_of(rl, art)
                if r is None:
                    continue
                voter_ranks[name] = r
                # 只有各投票人前 per_topk 名真正进入融合（anchor 也按 per_topk 计入）
                if r <= per_topk and (best_passed is None or r < best_passed):
                    best_passed = r
                if best_full is None or r < best_full:
                    best_full, best_full_voter = r, name
            prod_pos = rank_of(prod_hits, art)
            gold_pos = rank_of(gold.hits, art)
            gold_full = None
            for name, rl in gold_voters.items():
                r = rank_of(rl, art)
                if r is not None and (gold_full is None or r < gold_full):
                    gold_full = r

            reason = ""
            unfiltered_rank = None
            if prod_pos is not None and prod_pos <= final_topk:
                reason = "hit"
            elif best_full is None:
                # 带法域过滤都没进候选池 → 去掉过滤再看（原句与各子查询都试）
                probe = ranked_list(question, hints=[], mode="hybrid", topk=per_topk)
                unfiltered_rank = rank_of(probe, art)
                for q in intent_subs:
                    r = rank_of(ranked_list(q, hints=[], mode="hybrid", topk=per_topk), art)
                    if r is not None and (unfiltered_rank is None or r < unfiltered_rank):
                        unfiltered_rank = r
                reason = "law_filter" if unfiltered_rank else "out_of_pool"
            elif not use_multi:
                # 单查询路径：条子在重排候选池里，但超过最终 topK 被截掉
                reason = "topk_truncate"
            elif best_passed is None:
                # 多路路径：条子在候选池里，但超过该路 topK，没进融合
                reason = "per_subquery_quota"
            else:
                reason = "fusion_order"

            items.append(
                {
                    "article": art,
                    "prod_pos": prod_pos,
                    "voter_ranks": voter_ranks,
                    "best_pool_rank": best_full,
                    "best_pool_voter": best_full_voter,
                    "best_passed_rank": best_passed,
                    "gold_pos": gold_pos,
                    "gold_best_pool_rank": gold_full,
                    "unfiltered_rank": unfiltered_rank,
                    "reason": reason,
                }
            )

        covered = sum(1 for i in items if i["reason"] == "hit")
        gold_covered = sum(1 for i in items if (i["gold_pos"] or 99) <= final_topk)
        rows.append(
            {
                "id": cid,
                "question": question,
                "complexity": complexity,
                "final_topk": final_topk,
                "per_topk": per_topk,
                "rewrite_type": it.get("rewrite_type"),
                "prod_mode": prod_mode,
                "use_multi": use_multi,
                "intent_subs": intent_subs,
                "gold_subs": gold_subs,
                "expected": expected,
                "prod_hits": [h.law_article for h in prod_hits],
                "gold_hits": [h.law_article for h in gold.hits],
                "prod_recall": covered / len(expected) if expected else 0.0,
                "gold_recall": gold_covered / len(expected) if expected else 0.0,
                "items": items,
            }
        )
        print(
            f"[{n}/{len(mh_ids)}] {cid} {prod_mode:6s} use_multi={use_multi} "
            f"prod={covered}/{len(expected)} gold={gold_covered}/{len(expected)} "
            f"reasons={[i['reason'] for i in items]}",
            flush=True,
        )

    summary = {
        "samples": len(rows),
        "prod_recall": sum(r["prod_recall"] for r in rows) / len(rows) if rows else 0.0,
        "gold_recall": sum(r["gold_recall"] for r in rows) / len(rows) if rows else 0.0,
        "reason_counts": {},
        "by_rewrite": {},
        "seconds": round(time.perf_counter() - started, 1),
    }
    for r in rows:
        for i in r["items"]:
            summary["reason_counts"][i["reason"]] = summary["reason_counts"].get(i["reason"], 0) + 1
        key = r["rewrite_type"] or "unknown"
        bucket = summary["by_rewrite"].setdefault(key, {"samples": 0, "prod_recall": 0.0})
        bucket["samples"] += 1
        bucket["prod_recall"] += r["prod_recall"]
    for bucket in summary["by_rewrite"].values():
        bucket["prod_recall"] = round(bucket["prod_recall"] / bucket["samples"], 4)

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dataset": "eval/qa_test_cases.jsonl (multi_hop)",
            "intent_source": "eval/results/intent_pipeline.json",
            "topk": "per complexity",
            "law_filter": law_filter,
            "command": "python scripts/multihop_stage_probe.py",
        },
        "summary": summary,
        "rows": rows,
    }
    out = ROOT / args.out
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
