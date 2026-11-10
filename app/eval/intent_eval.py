"""意图识别评估脚本（设计方案 §6.1 / §9）。

    uv run python -m app.eval.intent_eval              # 真实 LLM 路径（需密钥）
    uv run python -m app.eval.intent_eval --rules-only # 只跑规则兜底，作基线对比

产出：
- ``eval/results/intent.json``：准确率、macro-F1、逐类 P/R/F1、混淆矩阵、延迟分位、逐条明细；
- 终端打印可直接抄进验收记录的摘要表。

量化目标（§9）：accuracy ≥ 0.90、macro-F1 ≥ 0.88（5 类 40 条样本）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from app.orchestration.intent import recognize_intent
from app.orchestration.intent_rules import rule_based_intent
from app.orchestration.models import INTENT_LABELS
from config import get_llm_api_key, get_llm_api_key_env, get_settings

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "eval" / "datasets" / "intent_samples.json"
DEFAULT_OUT = ROOT / "eval" / "results" / "intent.json"

ACC_TARGET = 0.90
F1_TARGET = 0.88


# ---------------------------------------------------------------------------
# 数据集与指标
# ---------------------------------------------------------------------------
def load_samples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"]
    if not samples:
        raise SystemExit(f"数据集为空：{path}")
    labels = set(data.get("labels") or INTENT_LABELS)
    for s in samples:
        if s["label"] not in labels:
            raise SystemExit(f"样本 {s.get('id')} 的 label 不合法：{s['label']}")
        if not s.get("query"):
            raise SystemExit(f"样本 {s.get('id')} 缺少 query")
    return samples


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(rows: Sequence[dict[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    """准确率、逐类 P/R/F1、macro-F1、混淆矩阵（真实为行、预测为列）。"""
    total = len(rows)
    correct = sum(1 for r in rows if r["label"] == r["prediction"])
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for r in rows if r["label"] == label and r["prediction"] == label)
        fp = sum(1 for r in rows if r["label"] != label and r["prediction"] == label)
        fn = sum(1 for r in rows if r["label"] == label and r["prediction"] != label)
        support = tp + fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    macro_f1 = statistics.fmean(v["f1"] for v in per_class.values())
    confusion = {
        label: {p: 0 for p in labels}
        for label in labels
    }
    for r in rows:
        confusion.setdefault(r["label"], {}).setdefault(r["prediction"], 0)
        confusion[r["label"]][r["prediction"]] += 1
    return {
        "accuracy": round(_safe_div(correct, total), 4),
        "macro_f1": round(macro_f1, 4),
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "confusion": confusion,
    }


def _latency_stats(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": round(statistics.fmean(ordered), 1),
        "p50": float(ordered[len(ordered) // 2]),
        "p95": float(ordered[p95_idx]),
        "max": float(ordered[-1]),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(samples: Sequence[dict[str, Any]], *, rules_only: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(samples, 1):
        started = time.perf_counter()
        if rules_only:
            result = rule_based_intent(s["query"])
        else:
            result = recognize_intent(s["query"])
        elapsed = int((time.perf_counter() - started) * 1000)
        rows.append(
            {
                "id": s["id"],
                "query": s["query"],
                "label": s["label"],
                "prediction": result.intent,
                "ok": result.intent == s["label"],
                "source": result.source,
                "rewrite_type": result.rewrite_type,
                "complexity": result.complexity,
                "hit_structured": result.hit_structured,
                "structured_topic": result.structured_topic,
                "law_hints": result.law_hints,
                "sub_queries": result.sub_queries,
                "latency_ms": elapsed,
            }
        )
        mark = "OK " if rows[-1]["ok"] else "MISS"
        print(
            f"[{i:02d}/{len(samples)}] {mark} {s['id']} "
            f"期望={s['label']:<10} 预测={result.intent:<10} "
            f"src={result.source:<5} {elapsed:>5}ms  {s['query'][:28]}"
        )
    return rows


def summarize(rows: Sequence[dict[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    metrics = compute_metrics(rows, labels)
    metrics["latency_ms"] = _latency_stats([r["latency_ms"] for r in rows])
    metrics["source"] = dict(Counter(r["source"] for r in rows))
    metrics["rewrite_type"] = dict(Counter(r["rewrite_type"] for r in rows))
    metrics["passed"] = (
        metrics["accuracy"] >= ACC_TARGET and metrics["macro_f1"] >= F1_TARGET
    )
    return metrics


def _print_report(
    title: str, m: dict[str, Any], labels: Sequence[str], rows: Sequence[dict[str, Any]]
) -> None:
    print()
    print("-" * 64)
    print(title)
    print(f"  准确率   : {m['accuracy']:.4f}  （目标 ≥ {ACC_TARGET}，{m['correct']}/{m['total']}）")
    print(f"  macro-F1 : {m['macro_f1']:.4f}  （目标 ≥ {F1_TARGET}）")
    if m.get("latency_ms"):
        lat = m["latency_ms"]
        print(f"  耗时(ms) : mean={lat['mean']} p50={lat['p50']} p95={lat['p95']} max={lat['max']}")
    if m.get("source"):
        print(f"  来源     : {m['source']}")
    print()
    print(f"  {'类别':<12}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for label in labels:
        c = m["per_class"][label]
        print(
            f"  {label:<12}{c['precision']:>10.3f}{c['recall']:>9.3f}"
            f"{c['f1']:>8.3f}{c['support']:>9}"
        )
    print()
    print("  混淆矩阵（行=真实，列=预测）")
    print("  " + " " * 12 + "".join(f"{l[:10]:>12}" for l in labels))
    for label in labels:
        cells = "".join(f"{m['confusion'].get(label, {}).get(p, 0):>12}" for p in labels)
        print(f"  {label:<12}{cells}")
    missed = [r for r in rows if not r["ok"]]
    if missed:
        print()
        print("  误判明细")
        for r in missed:
            print(f"  - {r['id']} {r['label']} → {r['prediction']} | {r['query'][:40]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="意图识别评估（准确率 / macro-F1）")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="样本集路径")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="结果 JSON 输出路径")
    parser.add_argument("--rules-only", action="store_true", help="只跑规则兜底（基线对比）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    args = parser.parse_args()

    samples = load_samples(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]
    labels = list(INTENT_LABELS)

    s = get_settings()
    mode = "rules" if args.rules_only else "llm"
    if mode == "llm" and not get_llm_api_key():
        raise SystemExit("未配置密钥：请设置 LLM_API_KEY / DEEPSEEK_API_KEY / DASHSCOPE_API_KEY")

    print("=" * 64)
    print(f"意图识别评估  样本={len(samples)}  模式={mode}")
    if mode == "llm":
        print(f"  provider : {s.llm_base_url}  model={s.llm_default_model}")
        print(f"  密钥来源 : {get_llm_api_key_env()}")
    print("=" * 64)

    rows = run(samples, rules_only=args.rules_only)
    metrics = summarize(rows, labels)

    # 同一批样本的规则兜底基线，用于量化 LLM 带来的提升
    baseline_rows = []
    for r in rows:
        pred = rule_based_intent(r["query"]).intent
        baseline_rows.append(
            {
                "id": r["id"],
                "query": r["query"],
                "label": r["label"],
                "prediction": pred,
                "ok": pred == r["label"],
            }
        )
    baseline = compute_metrics(baseline_rows, labels)

    _print_report("LLM 路径" if mode == "llm" else "规则兜底路径", metrics, labels, rows)
    if mode == "llm":
        _print_report("规则兜底基线（同批样本）", baseline, labels, rows)
        print()
        print(
            f"  提升：accuracy {baseline['accuracy']:.4f} → {metrics['accuracy']:.4f}"
            f"（{(metrics['accuracy'] - baseline['accuracy']) * 100:+.1f} pt），"
            f"macro-F1 {baseline['macro_f1']:.4f} → {metrics['macro_f1']:.4f}"
            f"（{(metrics['macro_f1'] - baseline['macro_f1']) * 100:+.1f} pt）"
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode,
        "dataset": str(args.dataset.relative_to(ROOT)) if args.dataset.is_relative_to(ROOT) else str(args.dataset),
        "provider": {
            "base_url": s.llm_base_url,
            "model": s.llm_default_model,
            "thinking": s.llm_thinking or None,
            "key_env": get_llm_api_key_env(),
        },
        "targets": {"accuracy": ACC_TARGET, "macro_f1": F1_TARGET},
        "labels": labels,
        "metrics": metrics,
        "baseline_rules": baseline,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("=" * 64)
    print(f"结论：{'达标' if metrics['passed'] else '未达标'}  →  {args.out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
