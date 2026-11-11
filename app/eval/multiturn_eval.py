"""多轮会话 + 缓存正确性评测（LA-N1）。

一条命令产出：逐轮原始结果、汇总指标、失败会话清单。

数据口径
--------
* 数据集 ``eval/datasets/multiturn_sessions.jsonl`` 带 ``dataset_version`` 与
  ``split=dev|holdout``；输出里同时记录文件 SHA-256 与筛选的 split。
* ``fixture`` 模式：``turn.fixture`` 是**人工构造的观测快照**（未参与任何线上调用），
  用于验证评测逻辑、失败定位与缓存隔离；报告 ``data_kind=synthetic_fixture``。
* ``live`` 模式：真实调用 LLM 改写/意图、Milvus 检索与 LLM 生成，报告
  ``data_kind=live_llm`` 并记录模型、调用次数与 token 用量；缺 Key 时退出码 3，
  不产出任何伪造指标。
* 语义判断（LLM-as-judge）默认关闭，单独放在 ``semantic_judge`` 字段，
  **不计入任何通过率**。

缓存正确性对照
--------------
``--identity legacy`` 复刻改造前的缓存 key（不含租户/知识库/版本），
``--identity scoped`` 使用修复后的服务端范围指纹。同一份数据集分别跑两次，
即可稳定复现"同问题不同知识库/租户错误回放"：

    uv run python -m app.eval.multiturn_eval --identity legacy --out eval/results/mt_legacy.json
    uv run python -m app.eval.multiturn_eval --identity scoped --out eval/results/mt_scoped.json
    uv run python -m app.eval.multiturn_eval --identity scoped --compare eval/results/mt_legacy.json

    # 真实链路（需 LLM Key + MySQL/Redis/Milvus）
    uv run python -m app.eval.multiturn_eval --mode live --identity scoped --out eval/results/mt_live_scoped.json
    uv run python -m app.eval.multiturn_eval --mode live --identity legacy --out eval/results/mt_live_legacy.json

命令示例::

    uv run python -m app.eval.multiturn_eval --split all --out eval/results/mt_fixture.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.core.access import AccessScope, KbGrant, answer_identity
from app.generation.prompts import PROMPT_VERSION
from app.orchestration.answer_cache import AnswerCache, normalize_question
from config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "eval" / "datasets" / "multiturn_sessions.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "results"
DEFAULT_MODEL = "deepseek-chat"


class _MemoryRedis:
    """无外部依赖的 Redis 替身：只实现 AnswerCache 用到的 get/set。"""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value: str, ex: int | None = None):  # noqa: ARG002
        self.data[key] = value
        return True

    def setex(self, key: str, ttl: int, value: str):  # noqa: ARG002
        self.data[key] = value
        return True

    def expire(self, key: str, ttl: int):  # noqa: ARG002
        return key in self.data

    def delete(self, *keys: str):
        for key in keys:
            self.data.pop(key, None)
        return len(keys)


def load_sessions(path: Path) -> list[dict]:
    sessions: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            session = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} JSON 解析失败: {exc}") from exc
        for key in ("session_id", "dataset_version", "split", "turns"):
            if key not in session:
                raise ValueError(f"{path}:{lineno} 缺少字段 {key}")
        if session["split"] not in ("dev", "holdout"):
            raise ValueError(f"{path}:{lineno} split 必须是 dev/holdout")
        sessions.append(session)
    return sessions


def _scope_for(session: dict, turn: dict) -> AccessScope:
    tenant = str(session.get("tenant_id") or "default")
    kb_ids = turn.get("kb_ids") or session.get("kb_ids") or []
    grants = [
        KbGrant(
            kb_id=kb_id,
            corpus_version=int(turn.get("corpus_version", 1)),
            index_version=int(turn.get("index_version", 1)),
            # default 租户的库视为公共库；其他租户按私有库建模，命名空间随之隔离
            is_public=(tenant == "default"),
            tenant_id=tenant,
        )
        for kb_id in kb_ids
    ]
    return AccessScope.for_grants(grants, tenant_id=tenant, user_id=1)


def _observed(turn: dict) -> dict:
    """fixture 观测值；缺失字段默认等于期望值（仅用于工具自检）。"""
    fx = dict(turn.get("fixture") or {})
    fx.setdefault("standalone", turn.get("expected_standalone") or turn.get("query", ""))
    fx.setdefault("refs", list(turn.get("expected_refs") or []))
    fx.setdefault("sub_queries", list(turn.get("expected_sub_queries") or []))
    fx.setdefault("refused", bool(turn.get("should_refuse")))
    fx.setdefault("answer", "")
    return fx


def _ref_hits(ref: str, answer: str) -> bool:
    """引用是否出现在答案里：条号必须出现，法名允许简称。"""
    law, _, article = ref.partition("#")
    if not article:
        return True
    short = law.replace("中华人民共和国", "")
    return article in answer and (law in answer or short in answer)


def _rate(values: Iterable[bool]) -> dict:
    values = list(values)
    n = len(values)
    return {"n": n, "hit": sum(1 for v in values if v), "rate": (sum(1 for v in values if v) / n) if n else None}


def evaluate(
    sessions: list[dict],
    *,
    identity_mode: str = "scoped",
    model: str = DEFAULT_MODEL,
    prompt_version: str = PROMPT_VERSION,
) -> dict:
    if identity_mode not in ("scoped", "legacy"):
        raise ValueError("identity_mode 只能是 scoped/legacy")

    # 评测用独立缓存与显式配置：不受本机 .env 的 TTL/长度阈值影响
    cache = AnswerCache(
        redis=_MemoryRedis(),
        settings=Settings(_env_file=None, cache_answer_min_chars=0),
    )
    rows: list[dict] = []
    failed_sessions: dict[str, list[dict]] = {}

    for session in sessions:
        scope = None
        history: list[dict] = []
        for turn in session.get("turns") or []:
            scope = _scope_for(session, turn)
            fx = _observed(turn)
            expected_standalone = str(turn.get("expected_standalone") or "")
            observed_standalone = str(fx.get("standalone") or "")

            # ---- 1) 指代消解/改写正确性 ----
            rewrite_ok = bool(expected_standalone) and (
                normalize_question(observed_standalone) == normalize_question(expected_standalone)
            )

            # ---- 2) 槽位继承 ----
            slots = dict(turn.get("expected_slots") or {})
            slot_missing = [
                name for name, value in slots.items() if str(value).lower() not in observed_standalone.lower()
            ]

            # ---- 3) 子查询覆盖（decompose 类问题）----
            expected_subs = list(turn.get("expected_sub_queries") or [])
            observed_subs = [str(s) for s in (fx.get("sub_queries") or [])]
            subquery_score = None
            if expected_subs:
                covered = sum(
                    1
                    for sub in expected_subs
                    if any(normalize_question(sub) in normalize_question(obs) for obs in observed_subs)
                )
                subquery_score = covered / len(expected_subs)

            # ---- 4) 检索召回（相对期望条款）----
            expected_refs = list(turn.get("expected_refs") or [])
            observed_refs = list(fx.get("refs") or [])
            turn_recall = None
            if expected_refs:
                turn_recall = len(set(expected_refs) & set(observed_refs)) / len(expected_refs)

            # ---- 5) 引用正确性（答案正文是否带对条款）----
            answer = str(fx.get("answer") or "")
            citation_ok = None
            if expected_refs:
                citation_ok = all(_ref_hits(ref, answer) for ref in expected_refs)

            # ---- 6) 拒答判断 ----
            should_refuse = bool(turn.get("should_refuse"))
            refused = bool(fx.get("refused"))
            refusal_ok = refused == should_refuse

            # ---- 7) 缓存命中/错误回放（只在 cache_probe 轮取样）----
            cache_hit = None
            cache_false = None
            if turn.get("cache_probe"):
                identity = (
                    ""
                    if identity_mode == "legacy"
                    else answer_identity(scope, model=model, prompt_version=prompt_version, as_of=0)
                )
                hit = cache.lookup(turn.get("query", ""), history=None, cache_identity=identity)
                cache_hit = hit is not None
                if hit is not None:
                    cache_false = normalize_question(hit.answer) != normalize_question(answer)
                cache.store(
                    turn.get("query", ""),
                    answer=answer,
                    sources=[{"law": ref.partition("#")[0], "article_no": ref.partition("#")[2]} for ref in expected_refs],
                    route="hybrid",
                    intent="labor_law",
                    cache_identity=identity,
                    cache_meta={"kb_ids": list(scope.kb_ids), "tenant": scope.tenant_id},
                )

            checks = {
                "rewrite": rewrite_ok,
                "slot": not slot_missing,
                "citation": citation_ok,
                "refusal": refusal_ok,
                "cache_false_hit": cache_false,
            }
            failures = []
            if not rewrite_ok:
                failures.append(f"rewrite_mismatch: {observed_standalone!r} != {expected_standalone!r}")
            if slot_missing:
                failures.append(f"slot_missing: {','.join(slot_missing)}")
            if turn_recall is not None and turn_recall < 1.0:
                failures.append(f"retrieval_miss: {turn_recall:.2f}")
            if citation_ok is False:
                failures.append("citation_miss")
            if not refusal_ok:
                failures.append(f"refusal_mismatch: expected={should_refuse} observed={refused}")
            if cache_false:
                failures.append("cache_false_hit")

            rows.append(
                {
                    "session_id": session["session_id"],
                    "dataset_version": session["dataset_version"],
                    "split": session["split"],
                    "tenant_id": scope.tenant_id,
                    "kb_ids": list(scope.kb_ids),
                    "turn": turn.get("turn"),
                    "query": turn.get("query"),
                    "observed_standalone": observed_standalone,
                    "expected_standalone": expected_standalone,
                    "observed_refs": observed_refs,
                    "expected_refs": expected_refs,
                    "turn_recall": turn_recall,
                    "subquery_recall": subquery_score,
                    "should_refuse": should_refuse,
                    "observed_refused": refused,
                    "identity_mode": identity_mode,
                    "cache_hit": cache_hit,
                    "cache_false_hit": cache_false,
                    "checks": checks,
                    "failures": failures,
                }
            )
            if failures:
                failed_sessions.setdefault(session["session_id"], []).append(
                    {"turn": turn.get("turn"), "query": turn.get("query"), "failures": failures}
                )
            # 多轮上下文：后续轮次能看到本轮的问答（fixture 快照）
            history.append({"role": "user", "content": turn.get("query", "")})
            if answer:
                history.append({"role": "assistant", "content": answer})

    meta = {
        "mode": "fixture",
        "data_kind": "synthetic_fixture",
        "identity_mode": identity_mode,
        "model": model,
        "prompt_version": prompt_version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "turns": len(rows),
        "sessions": len(sessions),
        "dataset_version": sorted({s["dataset_version"] for s in sessions}),
    }
    return _build_report(rows, sessions, failed_sessions, meta)


def _build_report(
    rows: list[dict], sessions: list[dict], failed_sessions: dict, meta: dict
) -> dict:
    """把逐轮结果汇总成统一报告（fixture / live 共用同一套指标口径）。"""
    rewrite_flags = [r["checks"]["rewrite"] for r in rows]
    self_flags = [
        r["checks"]["rewrite_self_contained"]
        for r in rows
        if r["checks"].get("rewrite_self_contained") is not None
    ]
    slot_flags = [r["checks"]["slot"] for r in rows]
    citation_flags = [r["checks"]["citation"] for r in rows if r["checks"]["citation"] is not None]
    refusal_flags = [
        r["checks"]["refusal"] for r in rows if r["checks"]["refusal"] is not None
    ]
    subquery_scores = [
        r["subquery_recall"] for r in rows if r.get("subquery_recall") is not None
    ]
    recall_scores = [
        r["turn_recall"] for r in rows if r.get("turn_recall") is not None
    ]
    probes = [r for r in rows if r["cache_hit"] is not None]
    false_hits = sum(1 for r in probes if r["cache_false_hit"])

    summary = {
        #: 与 expected_standalone 归一化后完全一致（fixture/live 同口径，偏严格）
        "rewrite_correct": _rate(rewrite_flags),
        #: 仅 live 提供：改写结果不含悬空指代且槽位齐全（更贴近多轮可用性）
        "rewrite_self_contained": _rate(self_flags) if self_flags else None,
        "slot_inheritance": _rate(slot_flags),
        "subquery_recall": {
            "n": len(subquery_scores),
            "mean": (sum(subquery_scores) / len(subquery_scores)) if subquery_scores else None,
        },
        "retrieval_recall": {
            "n": len(recall_scores),
            "mean": (sum(recall_scores) / len(recall_scores)) if recall_scores else None,
            "full_recall": _rate([score == 1.0 for score in recall_scores]),
        },
        "citation_correct": _rate(citation_flags),
        "refusal_correct": _rate(refusal_flags),
        "cache_false_hit": {
            "probes": len(probes),
            "hits": sum(1 for r in probes if r["cache_hit"]),
            "false_hits": false_hits,
            "rate": (false_hits / len(probes)) if probes else None,
        },
        "semantic_judge": {
            "enabled": False,
            "judged": 0,
            "note": "语义判断未启用，不计入任何通过率；需要时单独接入 LLM-as-judge",
        },
    }
    return {
        "meta": meta,
        "summary": summary,
        "by_split": {
            split: _summary_for([r for r in rows if r["split"] == split])
            for split in ("dev", "holdout")
        },
        "raw_turns": rows,
        "failed_sessions": failed_sessions,
    }


def _summary_for(rows: list[dict]) -> dict:
    """按 split 重算核心指标（分母只含该 split）。"""
    self_flags = [
        r["checks"]["rewrite_self_contained"]
        for r in rows
        if r["checks"].get("rewrite_self_contained") is not None
    ]
    return {
        "turns": len(rows),
        "rewrite_correct": _rate([r["checks"]["rewrite"] for r in rows]),
        "rewrite_self_contained": _rate(self_flags) if self_flags else None,
        "slot_inheritance": _rate([r["checks"]["slot"] for r in rows]),
        "citation_correct": _rate([r["checks"]["citation"] for r in rows if r["checks"]["citation"] is not None]),
        "refusal_correct": _rate(
            [r["checks"]["refusal"] for r in rows if r["checks"]["refusal"] is not None]
        ),
        "cache": {
            "probes": sum(1 for r in rows if r["cache_hit"] is not None),
            "false_hits": sum(1 for r in rows if r["cache_false_hit"]),
        },
    }


def evaluate_live(
    sessions: list[dict],
    *,
    identity_mode: str = "scoped",
    model: str | None = None,
    prompt_version: str = PROMPT_VERSION,
    settings=None,
    client=None,
    retrieve_fn=None,
) -> dict:
    """真实链路评测：LLM 改写/意图 → Milvus 检索 → LLM 生成。

    与 fixture 的差别只在观测值来源：这里每一步都真的调用模型与检索服务；
    指标口径、缓存探针与失败定位与 fixture 完全一致，便于前后对照。
    ``client`` / ``retrieve_fn`` 仅用于测试注入，生产运行不要传。
    """
    from app.generation.context import build_context_sources
    from app.generation.llm_client import get_llm_client
    from app.generation.prompts import NO_DATA_REPLY, build_context_block, build_messages
    from app.orchestration.intent import recognize_intent
    from app.orchestration.models import IntentResult
    from app.orchestration.router import route as decide_route
    from app.retrieval.service import retrieve, retrieve_multi
    from app.security.guardrails import is_refusal
    from config import get_settings

    if identity_mode not in ("scoped", "legacy"):
        raise ValueError("identity_mode 只能是 scoped/legacy")

    s = settings or get_settings()
    model = model or getattr(s, "llm_default_model", DEFAULT_MODEL)
    llm = client or get_llm_client(s)
    cache = AnswerCache(
        redis=_MemoryRedis(),
        settings=Settings(_env_file=None, cache_answer_min_chars=0),
    )

    rows: list[dict] = []
    failed_sessions: dict[str, list[dict]] = {}
    errors: list[dict] = []
    llm_calls = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _err(session_id: str, turn: Any, stage: str, exc: Exception) -> None:
        errors.append(
            {
                "session_id": session_id,
                "turn": turn,
                "stage": stage,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }
        )

    for session in sessions:
        history: list[dict] = []
        for turn in session.get("turns") or []:
            scope = _scope_for(session, turn)
            query = str(turn.get("query") or "")
            try:
                intent = recognize_intent(query, history=history, client=llm, settings=s)
            except Exception as exc:  # noqa: BLE001 - 注入假客户端时兜底，保证评测不中断
                intent = IntentResult(intent="other", sub_queries=[query])
                _err(session["session_id"], turn.get("turn"), "intent", exc)

            subs = [str(x).strip() for x in (intent.sub_queries or []) if str(x).strip()]
            if intent.rewrite_type == "decompose" and len(subs) > 1:
                observed_standalone = "，".join(subs)
            else:
                observed_standalone = subs[0] if subs else query
            expected_standalone = str(turn.get("expected_standalone") or "")
            rewrite_ok = bool(expected_standalone) and (
                normalize_question(observed_standalone) == normalize_question(expected_standalone)
            )

            slots = dict(turn.get("expected_slots") or {})
            slot_missing = [
                name
                for name, value in slots.items()
                if str(value).lower() not in observed_standalone.lower()
            ]
            subquery_score = None
            expected_subs = list(turn.get("expected_sub_queries") or [])
            if expected_subs:
                covered = sum(
                    1
                    for sub in expected_subs
                    if any(normalize_question(sub) in normalize_question(obs) for obs in subs)
                )
                subquery_score = covered / len(expected_subs)

            # ---- 路由 + 服务端范围收敛（与 AnswerEngine 同口径）----
            decision = decide_route(intent, selected_kb_ids=list(scope.kb_ids), settings=s)
            route_kb_ids = (
                scope.restrict(decision.kb_ids) if decision.kb_ids else list(scope.kb_ids)
            )
            hits = []
            retrieval_error = ""
            if decision.use_milvus and route_kb_ids:
                try:
                    if retrieve_fn is not None:
                        res = retrieve_fn(
                            observed_standalone,
                            kb_ids=route_kb_ids,
                            complexity=intent.complexity,
                            law_hints=list(intent.law_hints) or None,
                            mode="auto",
                            rewrite_type=intent.rewrite_type,
                            route=decision.route,
                            settings=s,
                        )
                    elif intent.rewrite_type == "decompose" and len(subs) > 1:
                        res = retrieve_multi(
                            subs,
                            kb_ids=route_kb_ids,
                            complexity=intent.complexity,
                            law_hints=list(intent.law_hints) or None,
                            anchor_query=query,
                            mode="hybrid",
                            settings=s,
                        )
                    else:
                        res = retrieve(
                            observed_standalone,
                            kb_ids=route_kb_ids,
                            complexity=intent.complexity,
                            law_hints=list(intent.law_hints) or None,
                            mode="auto",
                            rewrite_type=intent.rewrite_type,
                            route=decision.route,
                            settings=s,
                        )
                    hits = list(getattr(res, "hits", []) or [])
                except Exception as exc:  # noqa: BLE001 - 检索故障显式记录，不伪造召回
                    retrieval_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                    _err(session["session_id"], turn.get("turn"), "retrieve", exc)

            observed_refs = [h.law_article for h in hits]
            expected_refs = list(turn.get("expected_refs") or [])
            turn_recall = None
            if expected_refs:
                turn_recall = len(set(expected_refs) & set(observed_refs)) / len(expected_refs)

            # ---- 生成：真实 LLM；路由判定无数据时用固定拒答话术 ----
            answer = ""
            gen_error = ""
            if hits:
                sources = build_context_sources(hits=hits)
                context_sources = [c for c in sources if c.kind in ("law", "section")]
                block, _ = build_context_block(
                    context_sources, max_chars=getattr(s, "gen_context_max_chars", 6000)
                )
                messages = build_messages(observed_standalone, context_block=block, history=history)
                try:
                    resp = llm.chat(messages, max_tokens=getattr(s, "gen_max_tokens", 2048))
                    answer = (resp.content or "").strip()
                    llm_calls += 1
                    resp_usage = getattr(resp, "usage", None) or {}
                    for key in usage:
                        usage[key] += int(resp_usage.get(key, 0) or 0)
                except Exception as exc:  # noqa: BLE001 - 生成失败按空答案记录
                    gen_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                    _err(session["session_id"], turn.get("turn"), "generate", exc)
            else:
                answer = NO_DATA_REPLY

            should_refuse = bool(turn.get("should_refuse"))
            #: 检索/生成故障的轮次无法判断拒答与引用，置空后不计入分母
            valid = not retrieval_error and not gen_error
            refused = ((not hits) or is_refusal(answer)) if valid else None
            citation_ok = None
            if expected_refs and valid:
                citation_ok = all(_ref_hits(ref, answer) for ref in expected_refs)

            # ---- 缓存探针：真实答案写入，检验跨库/跨租户是否错误回放 ----
            cache_hit = None
            cache_false = None
            if turn.get("cache_probe") and answer:
                identity = (
                    ""
                    if identity_mode == "legacy"
                    else answer_identity(scope, model=model, prompt_version=prompt_version, as_of=0)
                )
                hit = cache.lookup(query, history=None, cache_identity=identity)
                cache_hit = hit is not None
                if hit is not None:
                    cache_false = normalize_question(hit.answer) != normalize_question(answer)
                if answer:
                    cache.store(
                        query,
                        answer=answer,
                        sources=[
                            {"law": ref.partition("#")[0], "article_no": ref.partition("#")[2]}
                            for ref in expected_refs
                        ],
                        route=decision.route,
                        intent=intent.intent,
                        cache_identity=identity,
                        cache_meta={"kb_ids": list(scope.kb_ids), "tenant": scope.tenant_id},
                    )

            dangling = [
                marker
                for marker in ("上述", "该规定", "该法", "该公司", "此规定", "此法", "这家公司", "那家公司", "它")
                if marker in observed_standalone
            ]
            checks = {
                "rewrite": rewrite_ok,
                "rewrite_self_contained": not dangling and not slot_missing,
                "slot": not slot_missing,
                "citation": citation_ok,
                "refusal": (refused == should_refuse) if valid else None,
                "cache_false_hit": cache_false,
            }
            failures = [f"dangling_reference: {','.join(dangling)}"] if dangling else []
            if not rewrite_ok:
                failures.append(
                    f"rewrite_mismatch: {observed_standalone!r} != {expected_standalone!r}"
                )
            if slot_missing:
                failures.append(f"slot_missing: {','.join(slot_missing)}")
            if retrieval_error:
                failures.append(f"retrieval_error: {retrieval_error}")
            if gen_error:
                failures.append(f"generate_error: {gen_error}")
            if turn_recall is not None and turn_recall < 1.0:
                failures.append(f"retrieval_miss: {turn_recall:.2f}")
            if citation_ok is False:
                failures.append("citation_miss")
            if valid and refused != should_refuse:
                failures.append(
                    f"refusal_mismatch: expected={should_refuse} observed={refused}"
                )
            if cache_false:
                failures.append("cache_false_hit")

            rows.append(
                {
                    "session_id": session["session_id"],
                    "dataset_version": session["dataset_version"],
                    "split": session["split"],
                    "tenant_id": scope.tenant_id,
                    "kb_ids": list(scope.kb_ids),
                    "turn": turn.get("turn"),
                    "query": query,
                    "observed_standalone": observed_standalone,
                    "expected_standalone": expected_standalone,
                    "observed_refs": observed_refs,
                    "expected_refs": expected_refs,
                    "turn_recall": turn_recall,
                    "subquery_recall": subquery_score,
                    "intent": intent.intent,
                    "rewrite_type": intent.rewrite_type,
                    "route": decision.route,
                    "hit_count": len(hits),
                    "valid": valid,
                    "retrieval_error": retrieval_error,
                    "generation_error": gen_error,
                    "answer": answer,
                    "should_refuse": should_refuse,
                    "observed_refused": refused,
                    "identity_mode": identity_mode,
                    "cache_hit": cache_hit,
                    "cache_false_hit": cache_false,
                    "checks": checks,
                    "failures": failures,
                }
            )
            if failures:
                failed_sessions.setdefault(session["session_id"], []).append(
                    {"turn": turn.get("turn"), "query": query, "failures": failures}
                )
            history.append({"role": "user", "content": query})
            if answer:
                history.append({"role": "assistant", "content": answer})

    meta = {
        "mode": "live",
        "data_kind": "live_llm",
        "identity_mode": identity_mode,
        "model": model,
        "prompt_version": prompt_version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "turns": len(rows),
        "valid_turns": sum(1 for r in rows if r["valid"]),
        "sessions": len(sessions),
        "dataset_version": sorted({s["dataset_version"] for s in sessions}),
        "llm_calls": llm_calls,
        "llm_usage": usage,
        "errors": errors,
    }
    return _build_report(rows, sessions, failed_sessions, meta)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def print_report(report: dict) -> None:
    meta = report["meta"]
    print(f"dataset={', '.join(meta['dataset_version'])} mode={meta['mode']}({meta['data_kind']}) "
          f"identity={meta['identity_mode']} turns={meta['turns']} sessions={meta['sessions']}")
    s = report["summary"]
    print(f"rewrite_correct     {_fmt(s['rewrite_correct']['rate'])} ({s['rewrite_correct']['hit']}/{s['rewrite_correct']['n']}, exact match)")
    if s.get("rewrite_self_contained"):
        sc = s["rewrite_self_contained"]
        print(f"rewrite_self_contained {_fmt(sc['rate'])} ({sc['hit']}/{sc['n']}, live)")
    print(f"slot_inheritance    {_fmt(s['slot_inheritance']['rate'])} ({s['slot_inheritance']['hit']}/{s['slot_inheritance']['n']})")
    print(f"subquery_recall     {_fmt(s['subquery_recall']['mean'])} (n={s['subquery_recall']['n']})")
    print(f"retrieval_recall    {_fmt(s['retrieval_recall']['mean'])} (n={s['retrieval_recall']['n']})")
    print(f"citation_correct    {_fmt(s['citation_correct']['rate'])} ({s['citation_correct']['hit']}/{s['citation_correct']['n']})")
    print(f"refusal_correct     {_fmt(s['refusal_correct']['rate'])} ({s['refusal_correct']['hit']}/{s['refusal_correct']['n']})")
    print(f"cache_false_hit     {_fmt(s['cache_false_hit']['rate'])} "
          f"(probes={s['cache_false_hit']['probes']} hits={s['cache_false_hit']['hits']} false={s['cache_false_hit']['false_hits']})")
    print(f"semantic_judge      disabled (not counted)")
    if meta.get("mode") == "live":
        print(
            f"llm_calls           {meta.get('llm_calls')} "
            f"usage={meta.get('llm_usage')} errors={len(meta.get('errors') or [])}"
        )
    for split, data in report["by_split"].items():
        print(f"  [{split}] turns={data['turns']} rewrite={_fmt(data['rewrite_correct']['rate'])} "
              f"citation={_fmt(data['citation_correct']['rate'])} cache_false={data['cache']['false_hits']}")
    failed = report["failed_sessions"]
    print(f"failed_sessions     {len(failed)}")
    for sid, items in failed.items():
        for item in items:
            print(f"  - {sid} turn={item['turn']}: {'; '.join(item['failures'])}")


def compare(report: dict, baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    print(f"\n对比基线 {baseline_path}（{baseline['meta']['identity_mode']} → {report['meta']['identity_mode']}）")
    cur, old = report["summary"], baseline["summary"]
    pairs = [
        ("rewrite_correct", cur["rewrite_correct"]["rate"], old["rewrite_correct"]["rate"]),
        ("slot_inheritance", cur["slot_inheritance"]["rate"], old["slot_inheritance"]["rate"]),
        ("retrieval_recall", cur["retrieval_recall"]["mean"], old["retrieval_recall"]["mean"]),
        ("citation_correct", cur["citation_correct"]["rate"], old["citation_correct"]["rate"]),
        ("refusal_correct", cur["refusal_correct"]["rate"], old["refusal_correct"]["rate"]),
        ("cache_false_hit", cur["cache_false_hit"]["rate"], old["cache_false_hit"]["rate"]),
    ]
    for name, new_val, old_val in pairs:
        delta = "" if new_val is None or old_val is None else f" ({new_val - old_val:+.3f})"
        print(f"  {name:20s} {_fmt(old_val):>6s} -> {_fmt(new_val):>6s}{delta}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多轮会话与缓存正确性评测（fixture）")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", default="fixture", choices=["fixture", "live"])
    parser.add_argument("--identity", default="scoped", choices=["scoped", "legacy"])
    parser.add_argument("--split", default="all", choices=["all", "dev", "holdout"])
    parser.add_argument("--model", default=None, help="默认取 LLM_DEFAULT_MODEL")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--compare", type=Path, default=None)
    args = parser.parse_args(argv)

    sessions = load_sessions(args.dataset)
    if args.split != "all":
        sessions = [s for s in sessions if s["split"] == args.split]

    if args.mode == "live":
        from config import get_llm_api_key

        if not get_llm_api_key():
            print(
                "ERROR: live 模式需要真实大模型密钥。请设置 LLM_API_KEY / "
                "DEEPSEEK_API_KEY / DASHSCOPE_API_KEY 后重跑；本工具不会在缺 Key 时伪造指标。",
                file=sys.stderr,
            )
            return 3
        report = evaluate_live(
            sessions,
            identity_mode=args.identity,
            model=args.model,
            prompt_version=args.prompt_version,
        )
    else:
        report = evaluate(
            sessions,
            identity_mode=args.identity,
            model=args.model or DEFAULT_MODEL,
            prompt_version=args.prompt_version,
        )

    report["meta"]["dataset_path"] = str(args.dataset)
    report["meta"]["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    report["meta"]["split"] = args.split
    report["meta"]["command"] = " ".join(["python", "-m", "app.eval.multiturn_eval", *(argv or sys.argv[1:])])

    print_report(report)
    out = args.out or (
        DEFAULT_OUT_DIR / f"multiturn_{args.mode}_{args.identity}_{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    if args.compare:
        compare(report, args.compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
