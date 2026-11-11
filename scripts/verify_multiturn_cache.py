"""N1 真实链路缓存探针：追问在新会话 / 不同上下文是否误命中答案缓存。

背景
----
离线 ``multiturn_eval`` 的缓存探针直接构造 ``AnswerCache``，不经过
``AnswerEngine`` 的「意图识别 → 独立化改写 → 缓存前置判断」顺序，也测不到
「问法相同但上下文不同」的误命中。本脚本补这一条：

* 全程走 ``AnswerEngine.stream``（真实 LLM 改写/生成、真实 Milvus 检索、真实
  Redis 答案缓存）；
* 为让探针落到答案缓存而不是问答库的 ``pure_structured`` 回放，注入一个永不
  命中的 QARetriever（其余链路保持真实）；
* 用独立的语料/索引版本号隔离缓存身份，跑前清理、跑后清理本探针自己的条目，
  不碰其他会话的缓存。

四个场景
--------
A 多轮：高温津贴问题 → 「那如果是我主动辞职呢？」（写入答案缓存）
B 新会话：同一句追问，无历史（是否命中 A 写下的答案）
C 不同上下文：年假问题 → 同一句追问（改写若丢掉上下文就会命中 A 的补偿答案）
D 阳性对照：同一会话内原样重复同一问题（必须命中，证明探针能观测到命中）

用法::

    $env:EMBEDDING_DEVICE='cpu'
    uv run python scripts/verify_multiturn_cache.py --out eval/results/multiturn_cache_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.core.access import AccessScope, KbGrant
from app.core.sse import parse_sse_frames
from app.core.trace import Trace
from app.generation.stream import AnswerEngine
from app.orchestration.answer_cache import normalize_question
from app.orchestration.qa_cache import QALookupResult

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB = "kb_labor"
DEFAULT_OUT = REPO_ROOT / "eval" / "results" / "multiturn_cache_probe.json"

#: 探针专用版本号：与线上语料版本不同，答案缓存身份天然隔离
PROBE_CORPUS_VERSION = 20261111
PROBE_INDEX_VERSION = 20261111
PROBE_USER_ID = 990001

FOLLOW_UP = "那如果是我主动辞职呢？"
SCENARIOS = (
    {
        "id": "A_multi_turn",
        "title": "多轮：先问高温津贴，再追问主动辞职",
        "turns": ["高温津贴的发放标准是什么？", FOLLOW_UP],
    },
    {
        "id": "B_new_session",
        "title": "新会话：无历史直接问同一句追问",
        "turns": [FOLLOW_UP],
    },
    {
        "id": "C_other_context",
        "title": "不同上下文：先问年假，再追问主动辞职",
        "turns": ["年假可以跨年安排吗？", FOLLOW_UP],
    },
    {
        "id": "D_positive_control",
        "title": "阳性对照：同一会话内原样重复同一问题（必须命中）",
        "turns": ["高温津贴的发放标准是什么？", "高温津贴的发放标准是什么？"],
    },
)


class _NoQA:
    """问答库永不命中：避免 pure_structured 回放掩盖答案缓存行为。"""

    def lookup_exact(self, query, *, scope=None):  # noqa: ARG002
        return None

    def lookup(self, query, **kwargs):  # noqa: ARG002
        return QALookupResult()


def _scope(kb_id: str) -> AccessScope:
    return AccessScope.for_grants(
        [
            KbGrant(
                kb_id=kb_id,
                corpus_version=PROBE_CORPUS_VERSION,
                index_version=PROBE_INDEX_VERSION,
                is_public=True,
                tenant_id="default",
            )
        ],
        tenant_id="default",
        user_id=PROBE_USER_ID,
        role="viewer",
    )


def _purge_probe_entries(engine: AnswerEngine, identity: str) -> int:
    """只删除本探针身份下的答案缓存条目（按 cache_meta.identity 精确匹配）。"""
    redis = engine.answer_cache.redis
    removed = 0
    for key in redis.scan_iter("qa:gen:*"):
        try:
            raw = redis.get(key)
            meta = (json.loads(raw).get("cache_meta") or {}) if raw else {}
        except Exception:  # noqa: BLE001 - 坏 key 直接跳过
            continue
        if str(meta.get("identity", "")) == identity:
            redis.delete(key)
            removed += 1
    return removed


def _run_turn(
    engine: AnswerEngine,
    query: str,
    history: list[dict],
    scope: AccessScope,
) -> dict:
    trace = Trace()
    frames = list(
        engine.stream(
            query,
            history=list(history) if history else None,
            selected_kb_ids=list(scope.kb_ids),
            scope=scope,
            trace=trace,
        )
    )
    parsed = parse_sse_frames("".join(frames))
    answer = "".join(data["text"] for event, data in parsed if event == "delta")
    meta = next((data for event, data in parsed if event == "meta"), {})
    cache_events = [e for e in trace.events if e.get("stage") == "cache_lookup"]
    cache = cache_events[-1] if cache_events else {}
    done = next((data for event, data in parsed if event == "done"), {})
    return {
        "query": query,
        "history_turns": len(history),
        "cache_query": cache.get("query", ""),
        "cache_query_note": (
            ""
            if cache.get("query")
            else "多子查询且有历史，按设计不生成缓存键（不读不写）"
        ),
        "route": meta.get("route", ""),
        "intent": meta.get("intent", ""),
        "cache": {
            "eligible": bool(cache.get("eligible")),
            "hit": bool(cache.get("hit")),
            "query": cache.get("query", ""),
        },
        "cached": bool(done.get("cached")),
        "answer": answer.strip(),
    }


def _same_cache_query(left: str, right: str) -> bool | None:
    """两轮缓存键是否一致；任一方为空时返回 None（无从比较）。"""
    if not left or not right:
        return None
    return normalize_question(left) == normalize_question(right)


def run_probe(*, kb_id: str) -> dict:
    engine = AnswerEngine(qa_retriever=_NoQA())
    scope = _scope(kb_id)
    identity = engine._answer_identity(scope)  # noqa: SLF001 - 探针需要同一身份清理缓存
    purged = _purge_probe_entries(engine, identity)

    results: dict[str, dict] = {}
    try:
        for scenario in SCENARIOS:
            history: list[dict] = []
            turns: list[dict] = []
            for query in scenario["turns"]:
                row = _run_turn(engine, query, history, scope)
                turns.append(row)
                history.append({"role": "user", "content": query})
                if row["answer"]:
                    history.append({"role": "assistant", "content": row["answer"]})
            results[scenario["id"]] = {
                "title": scenario["title"],
                "turns": turns,
            }
    finally:
        _purge_probe_entries(engine, identity)

    a_follow = results["A_multi_turn"]["turns"][-1]
    b_follow = results["B_new_session"]["turns"][-1]
    c_follow = results["C_other_context"]["turns"][-1]
    d_repeat = results["D_positive_control"]["turns"][-1]

    same_rewrite_b = _same_cache_query(a_follow["cache_query"], b_follow["cache_query"])
    same_rewrite_c = _same_cache_query(a_follow["cache_query"], c_follow["cache_query"])
    # 命中只允许发生在"改写后的独立问法一致"时；否则就是跨上下文回放
    false_hit_b = b_follow["cache"]["hit"] and same_rewrite_b is False
    false_hit_c = c_follow["cache"]["hit"] and same_rewrite_c is False
    # 年假上下文下改写若仍在问"经济补偿"，说明上下文被丢弃，命中即为答非所问
    context_lost_c = c_follow["cache"]["hit"] and "年假" not in c_follow["cache_query"]
    # 阳性对照：缓存探针本身必须能观测到命中，否则"没命中"没有说服力
    positive_control = d_repeat["cache"]["hit"] and d_repeat["cached"]

    verdict = {
        "positive_control_hit": positive_control,
        "a_followup_cache_key": a_follow["cache_query"],
        "a_followup_cacheable": bool(a_follow["cache_query"]),
        "new_session_hit": b_follow["cache"]["hit"],
        "new_session_same_cache_query": same_rewrite_b,
        "new_session_false_hit": false_hit_b,
        "other_context_hit": c_follow["cache"]["hit"],
        "other_context_same_cache_query": same_rewrite_c,
        "other_context_false_hit": false_hit_c,
        "other_context_lost": context_lost_c,
        "ok": positive_control and not (false_hit_b or false_hit_c or context_lost_c),
        "notes": [
            "cache_query 为空表示该轮被拆成多个子查询且有历史，按设计不读不写答案缓存",
            "same_cache_query 为 null 表示至少一方没有缓存键，命中与否无从比较",
        ],
    }
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": "engine_cache_probe",
            "engine": "AnswerEngine.stream",
            "kb_id": kb_id,
            "qa_retriever": "disabled_for_probe",
            "cache_identity_isolated": True,
            "purged_before": purged,
            "command": "",
        },
        "scenarios": results,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N1 真实链路答案缓存探针")
    parser.add_argument("--kb", default=DEFAULT_KB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    report = run_probe(kb_id=args.kb)
    report["meta"]["command"] = " ".join(
        ["python", "scripts/verify_multiturn_cache.py", *(argv or sys.argv[1:])]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    v = report["verdict"]
    print(f"positive_control_hit={v['positive_control_hit']}")
    print(f"new_session_hit={v['new_session_hit']} same_cache_query={v['new_session_same_cache_query']}")
    print(f"other_context_hit={v['other_context_hit']} same_cache_query={v['other_context_same_cache_query']}")
    print(f"other_context_lost={v['other_context_lost']} ok={v['ok']}")
    print(f"结果已写入 {args.out}")
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
