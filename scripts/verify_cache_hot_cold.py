"""冷/热缓存验收：同一独立问题连续问两次，记录 miss（冷）与 hit（暖）。

口径
----
* **冷**：Redis 答案缓存未命中，完整走 意图 → 检索 → 重排 → 生成；
* **暖**：同一问题、同一授权身份再次提问，命中整段回答，只做意图识别 + 回放，
  不再调用生成 LLM。
* 问题带随机编号，保证冷启动一定是 miss；不会清空或污染其他会话的缓存。

运行::

    EMBEDDING_DEVICE=cpu uv run python scripts/verify_cache_hot_cold.py

产物：``eval/results/cache_hot_cold.json``（含 ``meta.command`` 与冷/暖两段耗时、
token、成本、回放字符数）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.access import AccessScope, KbGrant, answer_identity  # noqa: E402
from app.core.trace import Trace  # noqa: E402
from app.generation.prompts import PROMPT_VERSION  # noqa: E402
from app.generation.stream import AnswerEngine  # noqa: E402
from app.orchestration.answer_cache import answer_key, get_answer_cache  # noqa: E402
from app.orchestration.qa_cache import normalize_question  # noqa: E402
from app.orchestration.service import analyze_query  # noqa: E402
from config import get_settings  # noqa: E402

DEFAULT_QUERY = "劳动合同试用期最长不得超过多久？"
DEFAULT_OUT = ROOT / "eval" / "results" / "cache_hot_cold.json"


def _command() -> str:
    return "EMBEDDING_DEVICE=cpu uv run python scripts/verify_cache_hot_cold.py"


def _parse(frame: str) -> tuple[str, dict]:
    event = ""
    payload: dict = {}
    for line in frame.splitlines():
        if line.startswith("event: "):
            event = line[len("event: ") :].strip()
        elif line.startswith("data: "):
            try:
                payload = json.loads(line[len("data: ") :])
            except json.JSONDecodeError:
                payload = {}
    return event, payload


def run_once(engine: AnswerEngine, query: str, scope: AccessScope, trace: Trace) -> dict:
    started = time.perf_counter()
    ttft_ms: int | None = None
    done: dict = {}
    chars = 0
    stages: list[str] = []
    error = ""
    for frame in engine.stream(query, scope=scope, trace=trace):
        event, payload = _parse(frame)
        stages.append(event)
        if event == "delta":
            if ttft_ms is None:
                ttft_ms = int((time.perf_counter() - started) * 1000)
            chars += len(str(payload.get("text") or payload.get("delta") or ""))
        elif event == "done":
            done = payload
        elif event == "error":
            error = str(payload.get("code") or payload.get("message") or "error")
    total_ms = int((time.perf_counter() - started) * 1000)

    cache_event = next((e for e in trace.events if e["stage"] == "cache_lookup"), {})
    generate_event = next((e for e in trace.events if e["stage"] == "generate"), {})
    return {
        "cached": bool(done.get("cached")),
        "cache_hit": bool(cache_event.get("hit")),
        "cache_eligible": bool(cache_event.get("eligible")),
        "ttft_ms": ttft_ms or int(done.get("ttft_ms") or 0),
        "latency_ms": int(done.get("latency_ms") or total_ms),
        "total_ms": total_ms,
        "chars": chars,
        "token_usage": done.get("token_usage") or {},
        "cost": generate_event.get("cost"),
        "error": error,
        "events": {
            "delta": stages.count("delta"),
            "replayed": "sources" in stages and done.get("cached") is True,
        },
    }


def _warmup_models() -> int:
    """只预热本地向量/重排模型，避免把进程首次加载权重算进冷请求耗时。"""
    from app.ingest.embedder import get_embedder
    from app.retrieval.rerank import get_reranker

    started = time.perf_counter()
    get_embedder().encode(["缓存冷热验收预热"])
    get_reranker().score("缓存冷热验收预热", ["缓存冷热验收预热"])
    return int((time.perf_counter() - started) * 1000)


def _invalidate(cache, engine: AnswerEngine, query: str, scope: AccessScope) -> bool:
    """只删本次探针问题对应的缓存 key，保证"冷"是真的 miss。

    改写会把随机编号规整掉，所以必须先用 analyze_query 拿到最终 cache_query，
    再按同一 identity 计算 key；不清库、不影响其他问题。
    """
    plan = analyze_query(query, scope=scope, settings=engine.settings)
    if not plan.cache_query:
        return False
    as_of = engine.settings.retrieval_effective_as_of or date.today().isoformat()
    identity = answer_identity(
        scope,
        model=engine.settings.llm_default_model,
        prompt_version=PROMPT_VERSION,
        as_of=as_of,
    )
    key = answer_key(normalize_question(plan.cache_query), identity)
    try:
        cache.redis.delete(key)
    except Exception:  # noqa: BLE001 - Redis 不可用时后面自然会 miss
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--kb", default="kb_labor")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    settings = get_settings()
    warmup_ms = _warmup_models()
    unique = f"{args.query}（验收编号 {uuid.uuid4().hex[:8]}）"
    scope = AccessScope.for_grants(
        [KbGrant(kb_id=args.kb)], tenant_id="default", user_id=1
    )
    cache = get_answer_cache()
    engine = AnswerEngine(
        answer_cache=cache,
        settings=settings,
        persist_fn=lambda *a, **k: True,  # 探针不写聊天历史
    )
    invalidated = _invalidate(cache, engine, unique, scope)

    cold_trace = Trace()
    cold = run_once(engine, unique, scope, cold_trace)
    if cold["error"]:
        print(f"冷启动失败: {cold['error']}")
        return 1
    warm_trace = Trace()
    warm = run_once(engine, unique, scope, warm_trace)

    report = {
        "meta": {
            "mode": "live",
            "data_kind": "cache_hot_cold_probe",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": _command(),
            "model": settings.llm_default_model,
            "embedding_device": settings.embedding_device,
            "kb_id": args.kb,
            "query": unique,
            "cache_namespace": "qa:exact",
            "model_warmup_ms": warmup_ms,
            "cache_key_invalidated": invalidated,
            "note": "冷请求在本地向量/重排模型预热后采样，不含首次加载权重耗时",
        },
        "cold": cold,
        "warm": warm,
        "cache": {
            "cold_miss": cold["cache_hit"] is False and cold["cached"] is False,
            "warm_hit": warm["cache_hit"] is True and warm["cached"] is True,
            "latency_delta_ms": warm["total_ms"] - cold["total_ms"],
            "speedup": round(cold["total_ms"] / warm["total_ms"], 2) if warm["total_ms"] else None,
            "warm_skips_generation": warm["token_usage"].get("completion_tokens") in (None, 0),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"cold: {cold['total_ms']}ms ttft={cold['ttft_ms']}ms cached={cold['cached']} "
        f"tokens={cold['token_usage']}\n"
        f"warm: {warm['total_ms']}ms ttft={warm['ttft_ms']}ms cached={warm['cached']} "
        f"speedup={report['cache']['speedup']}x\n"
        f"写入 {out}"
    )
    return 0 if report["cache"]["cold_miss"] and report["cache"]["warm_hit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
