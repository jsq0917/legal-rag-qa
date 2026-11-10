"""补齐评估集的意图识别缓存（分批 + 用量记账）。

120 条样本里只有 q01–q40 的意图结果被缓存过；直接跑 ``recall_eval`` 会因
"缓存缺任意 id 就整批重算"而重复调用前 40 条。本模块只对缺失样本调用
``recognize_intent``，并把结果合并进 ``intent_pipeline.json`` /
``intent_hints.json``，避免重复计费。

    uv run python -m app.eval.intent_cache               # 默认每批 20 条
    uv run python -m app.eval.intent_cache --batch-size 10

每批结束打印并落盘 prompt / completion token 与折算费用；连续 2 条走规则
兜底（LLM 失败降级）即停止，不把预算耗在重试上。

费用口径：仓库没有该模型（deepseek-v4.1-flash）的官方单价，下列单价是
**假设价**，仅用于量级参考，实际以阿里云百炼账单为准。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from app.eval.qa_dataset import load_cases

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CACHE = ROOT / "eval" / "results" / "intent_pipeline.json"
HINTS_CACHE = ROOT / "eval" / "results" / "intent_hints.json"
USAGE_PATH = ROOT / "eval" / "results" / "intent_cache_usage.json"

#: 假设单价（元 / 百万 token）——仓库与百炼控制台无对应价目，仅作量级参考
PRICE_PROMPT_PER_M = 2.0
PRICE_COMPLETION_PER_M = 8.0


def _cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens / 1e6 * PRICE_PROMPT_PER_M
        + completion_tokens / 1e6 * PRICE_COMPLETION_PER_M,
        6,
    )


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _install_usage_counter() -> Callable[[], dict[str, int]]:
    """包一层 ``LLMClient.chat`` 累计 usage（重试只记成功那次的响应）。"""
    from app.generation.llm_client import LLMClient

    counter = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    original = LLMClient.chat

    def wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        resp = original(self, *args, **kwargs)
        usage = resp.usage or {}
        counter["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        counter["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        counter["calls"] += 1
        return resp

    LLMClient.chat = wrapped  # type: ignore[assignment]

    def snapshot() -> dict[str, int]:
        return dict(counter)

    return snapshot


def fill(batch_size: int) -> None:
    from app.orchestration.intent import recognize_intent

    cases = load_cases()
    pipeline = _load(PIPELINE_CACHE)
    hints = _load(HINTS_CACHE)
    missing = [c for c in cases if c["id"] not in pipeline]
    if not missing:
        print("意图缓存已覆盖全部样本，无需补齐")
        return
    print(f"待补齐 {len(missing)} 条（已有 {len(pipeline)} 条），每批 {batch_size} 条")

    snapshot = _install_usage_counter()
    usage_log: list[dict[str, Any]] = []
    total_prompt = total_completion = 0
    consecutive_failures = 0
    started_all = time.perf_counter()

    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        before = snapshot()
        batch_failures: list[str] = []
        for case in batch:
            result = recognize_intent(case["question"])
            fallback = result.source == "rules"
            if fallback:
                # 规则兜底结果不落缓存：它代表 LLM 失败，落盘会把降级结果
                # 固化成后续评估的"意图输入"，掩盖问题
                batch_failures.append(case["id"])
                consecutive_failures += 1
            else:
                pipeline[case["id"]] = {
                    "sub_queries": list(result.sub_queries),
                    "rewrite_type": result.rewrite_type,
                    "complexity": result.complexity,
                    "law_hints": list(result.law_hints),
                }
                hints[case["id"]] = list(result.law_hints)
                consecutive_failures = 0
            print(
                f"  {case['id']} {result.rewrite_type:<10} subs={len(result.sub_queries)} "
                f"complexity={result.complexity:<8} source={result.source}"
            )
            if consecutive_failures >= 2:
                break

        after = snapshot()
        prompt = after["prompt_tokens"] - before["prompt_tokens"]
        completion = after["completion_tokens"] - before["completion_tokens"]
        calls = after["calls"] - before["calls"]
        total_prompt += prompt
        total_completion += completion
        usage_log.append(
            {
                "batch": offset // batch_size + 1,
                "cases": [c["id"] for c in batch],
                "llm_calls": calls,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cost_cny_estimate": _cost(prompt, completion),
                "fallback_ids": batch_failures,
            }
        )
        _save(PIPELINE_CACHE, pipeline)
        _save(HINTS_CACHE, hints)
        print(
            f"[批 {offset // batch_size + 1}] {len(batch)} 条，LLM {calls} 次，"
            f"prompt {prompt} / completion {completion} tokens，"
            f"约 ¥{_cost(prompt, completion):.4f}（假设价）"
            + (f"，规则兜底 {batch_failures}" if batch_failures else "")
        )
        if consecutive_failures >= 2:
            print("连续 2 条 LLM 失败降级，停止补齐（先查原因，不重试）")
            break

    elapsed = time.perf_counter() - started_all
    summary = {
        "model": None,
        "note": "单价为假设价，实际以百炼账单为准",
        "price_cny_per_million": {
            "prompt": PRICE_PROMPT_PER_M,
            "completion": PRICE_COMPLETION_PER_M,
        },
        "cases": len(pipeline),
        "llm_calls": sum(b["llm_calls"] for b in usage_log),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "cost_cny_estimate": _cost(total_prompt, total_completion),
        "elapsed_s": round(elapsed, 1),
        "batches": usage_log,
    }
    _save(USAGE_PATH, summary)
    print(
        f"补齐完成：{len(pipeline)} 条，LLM {summary['llm_calls']} 次，"
        f"prompt {total_prompt} / completion {total_completion} tokens，"
        f"约 ¥{summary['cost_cny_estimate']:.4f}（假设价），耗时 {elapsed:.0f}s"
    )
    print(f"用量明细 → {USAGE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="补齐评估集意图缓存（分批记账）")
    parser.add_argument("--batch-size", type=int, default=20, help="每批条数，默认 20")
    args = parser.parse_args()
    fill(max(1, args.batch_size))


if __name__ == "__main__":
    main()
