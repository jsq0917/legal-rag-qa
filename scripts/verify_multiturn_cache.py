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

四个场景（每条追问各跑一遍）
----------------------------
A 多轮：高温津贴问题 → 追问（该追问若可缓存则写入答案缓存）
B 新会话：同一句追问，无历史（是否命中 A 写下的答案）
C 不同上下文：年假问题 → 同一句追问（改写若丢掉上下文就会命中 A 的答案）
D 阳性对照：同一会话内原样重复同一问题（必须命中，证明探针能观测到命中）

判定口径分两层：``verdict.ok`` 只由确定性断言决定，模型波动不改变判定
--------------------------------------------------------------------------

**确定性断言（参与 ``ok``）**：

* ``followup_text_not_replaced``：落到检索的追问文本与原句完全一致（改写只影响
  缓存键/子查询，不得替换用户原话）；
* ``positive_control_hit``：阳性对照在同一会话内原样重复必须命中；
* ``no_hit_without_cache_key``：A/B/C 中任何一轮没有缓存键（``cache_query`` 为空，
  多子查询且有历史时按设计不生成）就不该读缓存，读了即 bug；
* ``no_false_hit_new_session`` / ``no_false_hit_other_context``：有缓存键但改写后
  问法不一致时不得跨上下文命中（命中即答非所问）。

**LLM 依赖的观察项（只输出，不参与 ``ok``）**：本次被拆成几个子查询、
``cache_query`` 是否有值、改写是否稳定（新会话/不同上下文的缓存键是否与 A 一致）、
不同上下文下主题是否保留。这些随模型波动，如实记录但不用于判定通过与否。

两条追问用例（原句「那如果是我主动辞职呢？」预期不可缓存、单点追问
「那主动辞职有没有经济补偿？」预期可缓存）走同一套确定性断言，分开清理缓存条目，
避免互相污染。

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

#: 两条追问用例：原句（预期拆成多子查询、无可缓存键）+ 单点问句（必须可缓存）
FOLLOW_UPS = (
    {
        "id": "original_multi_subquery",
        "text": "那如果是我主动辞职呢？",
        "expect": "uncacheable_multi_subquery",
    },
    {
        "id": "single_issue",
        "text": "那主动辞职有没有经济补偿？",
        "expect": "cacheable_single_query",
    },
)


def _scenarios(follow_up: str) -> tuple[dict, ...]:
    return (
        {
            "id": "A_multi_turn",
            "title": "多轮：先问高温津贴，再追问主动辞职",
            "turns": ["高温津贴的发放标准是什么？", follow_up],
        },
        {
            "id": "B_new_session",
            "title": "新会话：无历史直接问同一句追问",
            "turns": [follow_up],
        },
        {
            "id": "C_other_context",
            "title": "不同上下文：先问年假，再追问主动辞职",
            "turns": ["年假可以跨年安排吗？", follow_up],
        },
    )


POSITIVE_CONTROL = {
    "id": "D_positive_control",
    "title": "阳性对照：同一会话内原样重复同一问题（必须命中）",
    "turns": ["高温津贴的发放标准是什么？", "高温津贴的发放标准是什么？"],
}


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
    rewrite_events = [e for e in trace.events if e.get("stage") == "rewrite"]
    rewrite = rewrite_events[-1] if rewrite_events else {}
    done = next((data for event, data in parsed if event == "done"), {})
    return {
        "query": query,
        "history_turns": len(history),
        "rewrite_type": rewrite.get("rewrite_type", ""),
        "sub_queries": int(rewrite.get("sub_queries") or 0),
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


def _run_scenarios(
    engine: AnswerEngine,
    scope: AccessScope,
    scenarios: tuple[dict, ...],
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for scenario in scenarios:
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
    return results


def _variant_verdict(
    spec: dict,
    results: dict[str, dict],
    *,
    positive_control: bool,
) -> dict:
    """单条追问用例的判定：``ok`` 只看确定性断言，LLM 依赖项归入 ``observations``。"""
    a_follow = results["A_multi_turn"]["turns"][-1]
    b_follow = results["B_new_session"]["turns"][-1]
    c_follow = results["C_other_context"]["turns"][-1]

    key_a = a_follow["cache_query"]
    cacheable = bool(key_a)
    mode = "cacheable_single_query" if cacheable else "uncacheable_multi_subquery"
    same_rewrite_b = _same_cache_query(key_a, b_follow["cache_query"])
    same_rewrite_c = _same_cache_query(key_a, c_follow["cache_query"])
    # 命中只允许发生在"改写后的独立问法一致"时；否则就是跨上下文回放
    false_hit_b = b_follow["cache"]["hit"] and same_rewrite_b is False
    false_hit_c = c_follow["cache"]["hit"] and same_rewrite_c is False
    hits = {
        "a_multi_turn": a_follow["cache"]["hit"],
        "b_new_session": b_follow["cache"]["hit"],
        "c_other_context": c_follow["cache"]["hit"],
    }

    # 确定性断言：与模型输出是否波动无关，任何一条不通过都是真实缺陷。
    criteria = {
        # 改写只影响缓存键/子查询，落到检索的必须是用户原话
        "followup_text_not_replaced": a_follow["query"] == spec["text"],
        "positive_control_hit": positive_control,
        # 没有缓存键就不该读缓存（多子查询且有历史时不生成键是设计行为）
        "no_hit_without_cache_key": all(
            not row["cache"]["hit"]
            for row in (a_follow, b_follow, c_follow)
            if not row["cache_query"]
        ),
        "no_false_hit_new_session": not false_hit_b,
        "no_false_hit_other_context": not false_hit_c,
    }

    # LLM 依赖的观察项：如实输出，不参与 ok（避免模型波动导致忽绿忽红）。
    observations = {
        "mode": mode,
        "mode_matches_expect": mode == spec["expect"],
        "a_cache_query": key_a,
        "a_cacheable": cacheable,
        "a_sub_queries": a_follow["sub_queries"],
        "a_rewrite_type": a_follow["rewrite_type"],
        "b_sub_queries": b_follow["sub_queries"],
        "c_sub_queries": c_follow["sub_queries"],
        "same_cache_query_new_session": same_rewrite_b,
        "same_cache_query_other_context": same_rewrite_c,
        # 年假上下文下改写若不再出现"年假"，说明上下文被丢弃（观察项）
        "other_context_kept_topic": not (
            c_follow["cache"]["hit"] and "年假" not in c_follow["cache_query"]
        ),
        "hits": hits,
    }

    return {
        "id": spec["id"],
        "follow_up": spec["text"],
        "expected_mode": spec["expect"],
        "criteria": criteria,
        "failed_criteria": [name for name, passed in criteria.items() if not passed],
        "ok": all(criteria.values()),
        "observations": observations,
    }


def run_probe(*, kb_id: str) -> dict:
    engine = AnswerEngine(qa_retriever=_NoQA())
    scope = _scope(kb_id)
    identity = engine._answer_identity(scope)  # noqa: SLF001 - 探针需要同一身份清理缓存
    purged = _purge_probe_entries(engine, identity)

    try:
        # 阳性对照先跑：此时缓存为空，能同时证明"写入 + 命中"
        control_results = _run_scenarios(engine, scope, (POSITIVE_CONTROL,))
        control = control_results["D_positive_control"]["turns"][-1]
        positive_control = control["cache"]["hit"] and control["cached"]
        _purge_probe_entries(engine, identity)

        variants: dict[str, dict] = {}
        for spec in FOLLOW_UPS:
            results = _run_scenarios(engine, scope, _scenarios(spec["text"]))
            variants[spec["id"]] = _variant_verdict(
                spec, results, positive_control=positive_control
            )
            results["_positive_control"] = control_results["D_positive_control"]
            variants[spec["id"]]["scenarios"] = results
            _purge_probe_entries(engine, identity)  # 两条用例互不污染
    finally:
        _purge_probe_entries(engine, identity)

    verdict = {
        "positive_control_hit": positive_control,
        "variants": variants,
        "ok": positive_control and all(item["ok"] for item in variants.values()),
        "notes": [
            "ok 只由确定性断言决定（原句未被替换、无缓存键不得命中、有键且改写不一致"
            "不得跨上下文命中、阳性对照同轮命中），模型波动不改判定",
            "sub_queries 条数、cache_query 是否有值、改写是否稳定、上下文是否保留"
            "属 LLM 观察项，见 observations，不参与 ok",
            "same_cache_query 为 null 表示至少一方没有缓存键，命中与否无从比较",
            "每条追问用例跑完后清理本探针身份的缓存条目，避免相互污染",
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
            "follow_ups": [spec["text"] for spec in FOLLOW_UPS],
            "command": "",
        },
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
    for name, item in v["variants"].items():
        obs = item["observations"]
        print(
            f"[{name}] {item['follow_up']} mode={obs['mode']} "
            f"cacheable={obs['a_cacheable']} sub_queries={obs['a_sub_queries']} "
            f"hits={obs['hits']} ok={item['ok']}"
        )
        print(
            "  observations="
            f"{{'mode_matches_expect': {obs['mode_matches_expect']}, "
            f"'same_cache_query_new_session': {obs['same_cache_query_new_session']}, "
            f"'same_cache_query_other_context': {obs['same_cache_query_other_context']}, "
            f"'other_context_kept_topic': {obs['other_context_kept_topic']}}}"
        )
        if not item["ok"]:
            print(f"  failed_criteria={item['failed_criteria']}")
    print(f"ok={v['ok']}")
    print(f"结果已写入 {args.out}")
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
