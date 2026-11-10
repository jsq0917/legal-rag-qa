"""RAGAS 五项评估（设计方案 §7.4）。

分两步，互不阻塞：

1. **采集**（``--collect``）：对 40 条样本打真实 SSE 接口，落 ``eval/results/qa_answers.jsonl``
   （answer / sources / route / ttft / tokens），并把来源还原成完整法条正文作为 retrieved_contexts。
   已存在的结果会跳过（``--force`` 重采），中断可续跑。
2. **打分**（默认）：读答案文件，用 RAGAS 跑五项指标，写 ``eval/results/ragas.json``。

    uv run python -m app.eval.ragas_eval --collect        # 需要服务已启动
    uv run python -m app.eval.ragas_eval                  # 需要 uv sync --group eval

judge 用配置里的大模型（当前为阿里云百炼 `deepseek-v4.1-flash`，设计稿原写 qwen-max），
embedding 复用本地 bge-m3（与检索同一模型，避免语义空间不一致）。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from app.eval.qa_dataset import load_cases
from app.generation.citations import CITATION_RE as _CITATION_RE
from app.generation.citations import normalize_law as _norm_law

ROOT = Path(__file__).resolve().parents[2]
ANSWERS_PATH = ROOT / "eval" / "results" / "qa_answers.jsonl"
RAGAS_PATH = ROOT / "eval" / "results" / "ragas.json"
USAGE_PATH = ROOT / "eval" / "results" / "ragas_usage.json"

#: 假设单价（元 / 百万 token）——仓库与百炼控制台无对应价目，仅作量级参考
PRICE_PROMPT_PER_M = 2.0
PRICE_COMPLETION_PER_M = 8.0

METRIC_TARGETS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.85,
    "context_precision": 0.80,
    "context_recall": 0.85,
    "context_relevance": 0.80,
}


# ---------------------------------------------------------------------------
# 第一步：采集真实回答
# ---------------------------------------------------------------------------
def _sources_to_contexts(sources: list[dict[str, Any]]) -> list[str]:
    """来源 → RAGAS 的 retrieved_contexts：优先用完整法条正文，退化用 snippet。"""
    from sqlalchemy import text

    from app.db.mysql import get_engine

    keys = []
    for src in sources:
        law = (src.get("law") or "").strip()
        article = (src.get("article_no") or "").strip()
        if law and article:
            keys.append((law, article))
    texts: dict[tuple[str, str], str] = {}
    if keys:
        stmt = text(
            "SELECT title, article_no, content FROM document_chunks "
            "WHERE chunk_type = 'parent' AND is_enabled = 1 "
            "AND article_no IN :arts"
        )
        arts = tuple({a for _, a in keys})
        from sqlalchemy import bindparam

        stmt = stmt.bindparams(bindparam("arts", expanding=True))
        try:
            with get_engine().connect() as conn:
                for title, article_no, content in conn.execute(stmt, {"arts": arts}):
                    law = (title or "").split("·")[0].strip()
                    texts[(law, article_no)] = content or ""
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] 回表取正文失败，退化用 snippet：{exc}")

    contexts = []
    for src in sources:
        key = ((src.get("law") or "").strip(), (src.get("article_no") or "").strip())
        body = texts.get(key)
        if body:
            contexts.append(f"《{key[0]}》{key[1]}：{body}")
        elif src.get("snippet"):
            contexts.append(src["snippet"])
    return contexts


def collect(url: str, *, force: bool = False) -> None:
    import httpx

    cases = load_cases()
    done: dict[str, dict[str, Any]] = {}
    if ANSWERS_PATH.exists() and not force:
        for line in ANSWERS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
        print(f"已有 {len(done)} 条结果，跳过重采（--force 可强制重采）")

    ANSWERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    gen_prompt = gen_completion = 0
    batch_prompt = batch_completion = 0
    batch_count = 0
    batch_usage: list[dict[str, Any]] = []
    with ANSWERS_PATH.open("a" if done else "w", encoding="utf-8") as fh:
        for i, case in enumerate(cases, 1):
            if case["id"] in done:
                continue
            body: dict[str, Any] = {"query": case["question"]}
            if case.get("kb_ids"):
                body["kb_ids"] = case["kb_ids"]
            started = time.perf_counter()
            answer, sources, meta, usage, error = "", [], {}, {}, None
            kind = None
            with httpx.stream("POST", url, json=body, timeout=300.0) as resp:
                if resp.status_code != 200:
                    resp.read()
                    error = f"HTTP {resp.status_code}"
                else:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if line.startswith("event: "):
                            kind = line[7:].strip()
                            continue
                        if not line.startswith("data: "):
                            continue
                        ev = json.loads(line[6:])
                        if kind == "meta":
                            meta = ev
                        elif kind == "sources":
                            sources = ev.get("sources") or []
                        elif kind == "delta":
                            answer += ev.get("text") or ""
                        elif kind == "error":
                            error = f"{ev.get('code')}: {ev.get('message')}"
                        elif kind == "done":
                            usage = ev.get("token_usage") or {}
            elapsed = int((time.perf_counter() - started) * 1000)
            row = {
                "id": case["id"],
                "type": case["type"],
                "question": case["question"],
                "reference": case["reference_answer"],
                "expected_articles": case["expected_articles"],
                "answer": answer,
                "sources": [
                    {k: s.get(k) for k in ("law", "chapter_path", "article_no", "score", "kind")}
                    for s in sources
                ],
                "contexts": _sources_to_contexts(sources),
                "route": meta.get("route"),
                "cached": meta.get("cached"),
                "degraded": meta.get("degraded"),
                "token_usage": usage,
                "elapsed_ms": elapsed,
                "error": error,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
            completion_tokens = int((usage or {}).get("completion_tokens") or 0)
            gen_prompt += prompt_tokens
            gen_completion += completion_tokens
            batch_prompt += prompt_tokens
            batch_completion += completion_tokens
            batch_count += 1
            print(
                f"  [{i:02d}/{len(cases)}] {case['id']} route={row['route']} "
                f"{elapsed}ms 答案 {len(answer)} 字 上下文 {len(row['contexts'])} 条"
                + (f"  tokens {prompt_tokens}/{completion_tokens}" if usage else "")
                + (f"  ERROR {error}" if error else "")
            )
            if batch_count >= 20:
                cost = _judge_cost(batch_prompt, batch_completion)
                batch_usage.append(
                    {
                        "batch": len(batch_usage) + 1,
                        "samples": batch_count,
                        "prompt_tokens": batch_prompt,
                        "completion_tokens": batch_completion,
                        "cost_cny_estimate": cost,
                    }
                )
                print(
                    f"[采集批 {len(batch_usage)}] {batch_count} 条，"
                    f"prompt {batch_prompt} / completion {batch_completion} tokens，"
                    f"约 ¥{cost:.4f}（假设价）"
                )
                batch_prompt = batch_completion = 0
                batch_count = 0
    if batch_count:
        cost = _judge_cost(batch_prompt, batch_completion)
        batch_usage.append(
            {
                "batch": len(batch_usage) + 1,
                "samples": batch_count,
                "prompt_tokens": batch_prompt,
                "completion_tokens": batch_completion,
                "cost_cny_estimate": cost,
            }
        )
        print(
            f"[采集批 {len(batch_usage)}] {batch_count} 条，"
            f"prompt {batch_prompt} / completion {batch_completion} tokens，"
            f"约 ¥{cost:.4f}（假设价）"
        )
    summary = {
        "note": "生成侧 token 来自 SSE done 事件；单价为假设价，实际以百炼账单为准",
        "samples": sum(b["samples"] for b in batch_usage),
        "prompt_tokens": gen_prompt,
        "completion_tokens": gen_completion,
        "cost_cny_estimate": _judge_cost(gen_prompt, gen_completion),
        "batches": batch_usage,
    }
    (ROOT / "eval" / "results" / "ragas_collect_usage.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"采集完成 → {ANSWERS_PATH}；生成 token：prompt {gen_prompt} / "
        f"completion {gen_completion}，约 ¥{summary['cost_cny_estimate']:.4f}（假设价）"
    )


# ---------------------------------------------------------------------------
# 第二步：RAGAS 打分
# ---------------------------------------------------------------------------
def _load_answers() -> list[dict[str, Any]]:
    if not ANSWERS_PATH.exists():
        raise SystemExit(f"缺少答案文件 {ANSWERS_PATH}，先跑 --collect")
    rows = [json.loads(line) for line in ANSWERS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def strip_disclaimer(answer: str) -> str:
    """剥离固定免责尾注后再交给 RAGAS（faithfulness 不该给模板句扣分）。"""
    from app.generation.prompts import DISCLAIMER_TAIL

    text = (answer or "").strip()
    tail = DISCLAIMER_TAIL.strip()
    if text.endswith(tail):
        text = text[: -len(tail)].rstrip()
    return text


#: 引用判定与线上 ``meta.citation_check`` 共用同一份正则/归一化
#: （见 app/generation/citations.py），避免离线评估与在线校验口径漂移。
def _citation_re():
    return _CITATION_RE


def citation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """引用完整性自动校验（离线，无需 LLM）：

    - expected_coverage：期望法条被答案引用的比例（REPORT §6 的"引用命中率"，
      此前靠人工抽查，现在全量自动算）；
    - sources_coverage：检索给出的来源被答案引用的比例（漏标条号的量化）；
    - hallucinated：答案引用了既不在 sources 也不在 expected 里的法条（错引）。
    """
    per_row: list[dict[str, Any]] = []
    expected_cov: list[float] = []
    sources_cov: list[float] = []
    hallucinated_rows = 0
    for r in rows:
        answer = strip_disclaimer(r.get("answer") or "")
        cited = {(_norm_law(law), art) for law, art in _citation_re().findall(answer)}
        sources = {
            (_norm_law(s.get("law") or ""), s.get("article_no") or "")
            for s in r.get("sources") or []
            if s.get("law") and s.get("article_no")
        }
        expected = set()
        for key in r.get("expected_articles") or []:
            law, _, art = key.partition("#")
            expected.add((_norm_law(law), art))
        row = {
            "id": r.get("id"),
            "cited": sorted(f"{law}#{art}" for law, art in cited),
            "expected_missing": sorted(f"{law}#{art}" for law, art in expected - cited),
            "sources_uncited": sorted(f"{law}#{art}" for law, art in sources - cited),
            "hallucinated": sorted(f"{law}#{art}" for law, art in cited - sources - expected),
        }
        if row["hallucinated"]:
            hallucinated_rows += 1
        if expected:
            cov = len(cited & expected) / len(expected)
            expected_cov.append(cov)
            row["expected_coverage"] = round(cov, 4)
        if sources:
            cov = len(cited & sources) / len(sources)
            sources_cov.append(cov)
            row["sources_coverage"] = round(cov, 4)
        per_row.append(row)
    return {
        "rows": len(rows),
        "rows_with_expected": len(expected_cov),
        "expected_coverage_mean": round(statistics.fmean(expected_cov), 4) if expected_cov else None,
        "sources_coverage_mean": round(statistics.fmean(sources_cov), 4) if sources_cov else None,
        "rows_with_hallucinated_citation": hallucinated_rows,
        "per_row": per_row,
    }


def _make_usage_counter():
    """judge 调用的 token 计数器（LangChain callback）。

    RAGAS 内部直接调 judge LLM，不经过项目里的 LLMClient，因此这里挂
    callback 统计 prompt/completion；兼容 ``usage_metadata`` 与
    ``llm_output.token_usage`` 两种挂载形态（LangChain 版本差异）。
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class _Counter(BaseCallbackHandler):
        def __init__(self) -> None:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0

        def on_llm_end(self, response, **kwargs):  # noqa: ANN001
            self.calls += 1
            for gens in getattr(response, "generations", []) or []:
                for gen in gens:
                    usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                    if usage:
                        self.prompt_tokens += int(usage.get("input_tokens") or 0)
                        self.completion_tokens += int(usage.get("output_tokens") or 0)
            info = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            if info:
                self.prompt_tokens += int(info.get("prompt_tokens") or 0)
                self.completion_tokens += int(info.get("completion_tokens") or 0)

        def snapshot(self) -> dict[str, int]:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }

    return _Counter()


def _judge_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens / 1e6 * PRICE_PROMPT_PER_M
        + completion_tokens / 1e6 * PRICE_COMPLETION_PER_M,
        6,
    )


def _build_llm_and_embeddings(callbacks=None):
    """judge LLM 走 OpenAI 兼容接口；embedding 用本地 bge-m3。"""
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from config import get_llm_api_key, get_settings

    s = get_settings()

    class BgeM3Embeddings(Embeddings):
        """把项目里的 bge-m3 embedder 适配成 LangChain Embeddings。"""

        @property
        def _model(self):
            from app.ingest.embedder import get_embedder

            return get_embedder()

        def embed_documents(self, texts):
            return [e.dense for e in self._model.encode(list(texts))]

        def embed_query(self, text):
            return self._model.encode([text])[0].dense

    llm = ChatOpenAI(
        model=s.llm_default_model,
        base_url=s.llm_base_url,
        api_key=get_llm_api_key(),
        temperature=0.0,
        timeout=180,
        max_retries=3,
        callbacks=callbacks or None,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(BgeM3Embeddings())


def score(limit: int = 0, batch_size: int = 20) -> None:
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    rows = _load_answers()
    if limit > 0:
        rows = rows[:limit]
    rows = [r for r in rows if (r.get("answer") or "").strip() and r.get("contexts")]
    if not rows:
        raise SystemExit("没有可打分样本（答案或上下文为空）")
    print(f"待打分样本 {len(rows)} 条，每批 {batch_size} 条")
    # 口径修正：剥离固定免责尾注（模板句不是模型论断，不该计入 faithfulness）
    stripped = sum(1 for r in rows if r["answer"].strip() != strip_disclaimer(r["answer"]).strip())
    if stripped:
        print(f"已剥离 {stripped} 条答案的固定免责尾注")

    import ragas.metrics as ragas_metrics

    # ragas 0.3.0 里前四项以函数/实例形式导出，context_relevance 只有 CamelCase 类名
    metric_objs: dict[str, Any] = {
        "faithfulness": ragas_metrics.faithfulness,
        "answer_relevancy": ragas_metrics.answer_relevancy,
        "context_precision": ragas_metrics.context_precision,
        "context_recall": ragas_metrics.context_recall,
        "context_relevance": getattr(ragas_metrics, "ContextRelevance", None),
    }
    metrics = []
    missing = []
    wanted = list(metric_objs)
    columns: dict[str, str] = {}  # 逻辑指标名 → 结果表列名（ragas 版本间命名有差异）
    for name, obj in metric_objs.items():
        if obj is None:
            missing.append(name)
            continue
        inst = obj() if callable(obj) else obj
        metrics.append(inst)
        columns[name] = getattr(inst, "name", name)
    if missing:
        print(f"[WARN] ragas 未导出这些指标（版本差异）：{missing}")
    print(f"实际打分指标：{[(n, columns[n]) for n in wanted if n in columns]}")

    # judge 是远程 API：并发 8 折中（太高容易被限流重试，太低要跑很久）
    run_config = RunConfig(timeout=180, max_retries=6, max_workers=8)
    started = time.perf_counter()
    per_metric_values: dict[str, list[float]] = {name: [] for name in columns}
    per_sample: list[dict[str, Any]] = []
    judge_totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    batches: list[dict[str, Any]] = []
    consecutive_failures = 0

    for idx, offset in enumerate(range(0, len(rows), batch_size), 1):
        chunk = rows[offset : offset + batch_size]
        counter = _make_usage_counter()
        llm, embeddings = _build_llm_and_embeddings([counter])
        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": r["question"],
                    "response": strip_disclaimer(r["answer"]),
                    "retrieved_contexts": r["contexts"],
                    "reference": r["reference"],
                }
                for r in chunk
            ]
        )
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
                raise_exceptions=False,
                show_progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - 失败即停，不做无脑重试
            print(f"[批 {idx}] 打分失败：{type(exc).__name__}: {exc}")
            consecutive_failures += 1
            if consecutive_failures >= 2:
                raise SystemExit("连续两批打分失败，停止（先查原因，不重试）") from exc
            continue
        consecutive_failures = 0

        df = result.to_pandas()
        for i, r in enumerate(chunk):
            item: dict[str, Any] = {
                "id": r["id"],
                "type": r["type"],
                "question": r["question"],
                "route": r.get("route"),
            }
            for name, col in columns.items():
                if col not in df.columns:
                    continue
                value = df[col].iloc[i]
                score_value = None if value != value else round(float(value), 4)
                item[name] = score_value
                if score_value is not None:
                    per_metric_values[name].append(score_value)
            per_sample.append(item)

        snap = counter.snapshot()
        for key in judge_totals:
            judge_totals[key] += snap[key]
        cost = _judge_cost(snap["prompt_tokens"], snap["completion_tokens"])
        batches.append(
            {
                "batch": idx,
                "samples": len(chunk),
                "ids": [r["id"] for r in chunk],
                **snap,
                "cost_cny_estimate": cost,
            }
        )
        print(
            f"[批 {idx}] {len(chunk)} 条，judge {snap['calls']} 次，"
            f"prompt {snap['prompt_tokens']} / completion {snap['completion_tokens']} tokens，"
            f"约 ¥{cost:.4f}（假设价）"
        )

    elapsed = round(time.perf_counter() - started, 1)
    per_metric: dict[str, float] = {
        name: round(statistics.fmean(values), 4) if values else 0.0
        for name, values in per_metric_values.items()
    }
    verdict = {
        name: {
            "score": per_metric.get(name),
            "target": target,
            "passed": (per_metric.get(name) or 0) >= target,
        }
        for name, target in METRIC_TARGETS.items()
    }

    # 生成侧 token 来自 SSE done 事件的 token_usage；judge 侧来自 callback
    gen_prompt = sum(int((r.get("token_usage") or {}).get("prompt_tokens") or 0) for r in rows)
    gen_completion = sum(
        int((r.get("token_usage") or {}).get("completion_tokens") or 0) for r in rows
    )
    gen_cost = _judge_cost(gen_prompt, gen_completion)
    judge_cost = _judge_cost(judge_totals["prompt_tokens"], judge_totals["completion_tokens"])
    usage = {
        "note": "单价为假设价，实际以百炼账单为准",
        "price_cny_per_million": {
            "prompt": PRICE_PROMPT_PER_M,
            "completion": PRICE_COMPLETION_PER_M,
        },
        "generation": {
            "samples": len(rows),
            "prompt_tokens": gen_prompt,
            "completion_tokens": gen_completion,
            "cost_cny_estimate": gen_cost,
        },
        "judge": {
            **judge_totals,
            "cost_cny_estimate": judge_cost,
        },
        "total_cost_cny_estimate": round(gen_cost + judge_cost, 6),
        "batches": batches,
    }

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "samples": len(rows),
        "elapsed_seconds": elapsed,
        "metrics": per_metric,
        "verdict": verdict,
        "citations": citation_report(rows),
        "per_sample": per_sample,
        "failed_samples": [r for r in per_sample if any(v is not None and v < 0.5 for k, v in r.items() if k in wanted)][:10],
        "usage": usage,
    }
    RAGAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAGAS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    USAGE_PATH.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for name, info in verdict.items():
        flag = "OK " if info["passed"] else "MISS"
        print(f"  {flag} {name:<20} {info['score']}  （目标 ≥{info['target']}）")
    cites = payload["citations"]
    if cites.get("expected_coverage_mean") is not None:
        print(
            f"  引用完整性：期望条覆盖 {cites['expected_coverage_mean']:.2%} / "
            f"来源条被引用 {cites['sources_coverage_mean']:.2%} / "
            f"错引行数 {cites['rows_with_hallucinated_citation']}"
        )
    print(
        f"\n费用（假设价）：生成 ¥{gen_cost:.4f}（prompt {gen_prompt} / completion {gen_completion}）"
        f" + judge ¥{judge_cost:.4f}（prompt {judge_totals['prompt_tokens']} / "
        f"completion {judge_totals['completion_tokens']}） = ¥{usage['total_cost_cny_estimate']:.4f}"
    )
    print(f"\n耗时 {elapsed}s，结果 → {RAGAS_PATH}；用量 → {USAGE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 五项评估")
    parser.add_argument("--collect", action="store_true", help="只采集真实回答")
    parser.add_argument("--citations", action="store_true",
                        help="只跑引用完整性校验（离线，读已有 qa_answers.jsonl，不调 LLM）")
    parser.add_argument("--force", action="store_true", help="重采已有结果")
    parser.add_argument("--url", default="http://127.0.0.1:8101/api/chat/stream")
    parser.add_argument("--limit", type=int, default=0, help="只打分前 N 条（调试）")
    parser.add_argument("--batch-size", type=int, default=20, help="打分每批条数，默认 20")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.citations:
        report = citation_report(_load_answers())
        print(json.dumps({k: v for k, v in report.items() if k != "per_row"},
                         ensure_ascii=False, indent=2))
        bad = [r for r in report["per_row"] if r["expected_missing"] or r["hallucinated"]]
        for r in bad[:10]:
            print(f"  {r['id']} 漏引={r['expected_missing']} 错引={r['hallucinated']}")
        return
    if args.collect:
        collect(args.url, force=args.force)
    else:
        score(args.limit, batch_size=max(1, args.batch_size))


if __name__ == "__main__":
    main()
