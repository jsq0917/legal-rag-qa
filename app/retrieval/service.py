"""检索编排：编码 → 双路召回 → (重排) → 去重 → 父块回填。

一次调用的数据流::

    question
      → bge-m3 编码（dense 1024 + sparse）
      → Milvus hybrid_search(dense, sparse) + RRFRanker(k=60)   [候选池 = max(4×topK, 12)]
      → bge-reranker-v2-m3 重排（可选，消融时关掉）
      → 按 parent_id 去重（同条多子块只留最高分）
      → 截断到 topK（simple 3 / medium 5 / complex 8）
      → MySQL 批量回表取整条法条正文（保证进 prompt 的是完整条文）

``mode="auto"`` 按问题形态分流：simple / 非拆解单跳走 dense，多跳（decompose、
complex）或跨法走 hybrid；低置信时回退 hybrid 重查（见 :func:`resolve_mode`）。

mode=dense / hybrid 用于消融实验与显式对照（L1/L3 组），不做自动回退。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Sequence

from sqlalchemy import bindparam, text

from app.core.logger import get_logger
from app.ingest.embedder import Embedder, get_embedder
from app.ingest.versioning import date_to_int, normalize_date
from app.retrieval import hybrid, refs
from app.retrieval.hybrid import Hit, RetrievalResult
from app.retrieval.rerank import Reranker, get_reranker
from config import Settings, get_settings

logger = get_logger("lawanswer.retrieval")

#: 多子查询检索时每个子查询的 topK 放大系数（合并去重后截断回最终 topK）
MULTI_POOL_FACTOR = 1.5


def resolve_mode(
    *,
    complexity: str = "medium",
    rewrite_type: str = "direct",
    law_hints: list[str] | None = None,
    route: str | None = None,
    settings: Settings | None = None,
) -> str:
    """按问题形态选择检索通道（``dense`` / ``hybrid``）。

    依据现有消融：单跳问题纯稠密 Recall@5 0.7405，高于混合 0.7024（−3.8pt）；
    多跳/跨法问题需要稀疏路的术语精确匹配，含 decompose 子查询、complex
    复杂度、cross_law 路由或多法域 hints 时保留 hybrid。

    ``RETRIEVAL_MODE_ROUTING=false`` 时一律 hybrid，等价于改动前的线上行为。
    """
    s = settings or get_settings()
    if not s.retrieval_mode_routing:
        return "hybrid"
    if rewrite_type == "decompose" or complexity == "complex":
        return "hybrid"
    if route == "cross_law" or len(law_hints or []) > 1:
        return "hybrid"
    return "dense"


#: collection 是否已加时效字段（老 collection 没有时跳过 Milvus 过滤，
#: 由 MySQL 回表兜底，避免过滤表达式直接报错）
_EFFECTIVE_SUPPORT: dict[str, bool] = {}


def _as_of_int(settings: Settings) -> int:
    """当前检索基准日（YYYYMMDD）；关闭过滤返回 0。"""
    if not getattr(settings, "retrieval_filter_expired", False):
        return 0
    value = normalize_date(getattr(settings, "retrieval_effective_as_of", "")) or date.today().isoformat()
    return date_to_int(value)


def _supports_effective_fields(client, collection: str) -> bool:
    if collection in _EFFECTIVE_SUPPORT:
        return _EFFECTIVE_SUPPORT[collection]
    try:
        desc = client.describe_collection(collection_name=collection)
        names = {field.get("name") for field in (desc.get("fields") or [])}
        supported = {"effective_from", "effective_to"} <= names
    except Exception:  # noqa: BLE001 - 探测失败按不支持处理，走 MySQL 兜底
        supported = False
    if not supported:
        logger.warning("Milvus collection 缺少时效字段，本次跳过向量侧时效过滤（请执行迁移脚本）")
    _EFFECTIVE_SUPPORT[collection] = supported
    return supported


def _rerank_candidates(hits: list[Hit], question: str, reranker) -> list[Hit]:
    """按 parent 预去重 → cross-encoder 打分 → 分数降序。"""
    hits = hybrid.dedupe_by_parent(hits, topk=len(hits))
    scores = reranker.score(question, [c.embed_text for c in hits])
    for hit, score in zip(hits, scores):
        hit.rerank_score = score
    return sorted(hits, key=hybrid._rank_value, reverse=True)


def _merge_by_parent(*groups: Sequence[Hit]) -> list[Hit]:
    """多路结果按法条（parent）合并，同条保留排名分最高的一条。"""
    by_parent: dict[str, Hit] = {}
    for hits in groups:
        for hit in hits:
            key = hit.parent_id or hit.chunk_id
            if key not in by_parent or hybrid._rank_value(hit) > hybrid._rank_value(by_parent[key]):
                by_parent[key] = hit
    return sorted(by_parent.values(), key=hybrid._rank_value, reverse=True)


def retrieve(
    question: str,
    *,
    kb_ids: list[str] | None = None,
    complexity: str = "medium",
    topk: int | None = None,
    mode: str = "hybrid",
    use_rerank: bool = True,
    expand_parent: bool = True,
    law_hints: list[str] | None = None,
    rewrite_type: str = "direct",
    route: str | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    client=None,
    collection: str | None = None,
    settings: Settings | None = None,
) -> RetrievalResult:
    s = settings or get_settings()
    started = time.perf_counter()
    resolved_topk = topk or s.retrieval.topk_for(complexity)
    pool = s.retrieval.pool_for(resolved_topk)
    collection = collection or s.milvus_collection
    # 意图层给出的法域（1-3 部法律）；为空则全库检索
    hints = list(law_hints or []) if s.retrieval_law_filter else []

    # mode="auto"：按问题形态分流；显式 dense/hybrid 用于消融对照，不做回退。
    auto_routed = mode == "auto"
    effective_mode = (
        resolve_mode(
            complexity=complexity,
            rewrite_type=rewrite_type,
            law_hints=hints,
            route=route,
            settings=s,
        )
        if auto_routed
        else mode
    )

    if not question.strip():
        return RetrievalResult([], question, resolved_topk, pool, complexity, effective_mode, False, 0)

    embedder = embedder or get_embedder()
    client = client or _default_client()

    # 时效过滤基准日；老 collection 没有字段时降级为 MySQL 回表过滤。
    as_of = _as_of_int(s)
    if as_of and not _supports_effective_fields(client, collection):
        as_of = 0

    embedding = embedder.encode([question])[0]
    candidates = hybrid.search(
        client,
        collection,
        embedding,
        pool=pool,
        kb_ids=kb_ids,
        mode=effective_mode,
        rrf_k=s.retrieval.rrf_k,
        ranker=s.retrieval_ranker,
        dense_weight=s.retrieval.dense_weight,
        sparse_weight=s.retrieval.sparse_weight,
        law_hints=hints,
        sparse_pool_ratio=getattr(s, "retrieval_sparse_pool_ratio", 1.0),
        effective_as_of=as_of,
    )
    dense_hits = len(candidates)

    reranked = False
    rescued_mode = False
    rerank_started = time.perf_counter()
    if use_rerank and candidates:
        # 重排前按法条（parent）预去重：同条多子块重复送 cross-encoder 既浪费算力，
        # 半句子块也容易误导打分。每条只保留融合分最高的子块。
        reranker = reranker or get_reranker(s)
        candidates = _rerank_candidates(candidates, question, reranker)
        reranked = True

        # 选路分流低置信回退：auto 判为 dense（单跳）但 top1 重排分过低，
        # 说明单跳判断可能错了（实际需要跨条/术语精确匹配）→ hybrid 重查一次，
        # 按法条合并保留高分。显式 mode=dense 的消融对照不触发。
        if (
            auto_routed
            and effective_mode == "dense"
            and s.retrieval_dense_rescue_score > 0
        ):
            top1_dense = hybrid._rank_value(candidates[0]) if candidates else 0.0
            if top1_dense < s.retrieval_dense_rescue_score:
                rescue = hybrid.search(
                    client,
                    collection,
                    embedding,
                    pool=pool,
                    kb_ids=kb_ids,
                    mode="hybrid",
                    rrf_k=s.retrieval.rrf_k,
                    ranker=s.retrieval_ranker,
                    dense_weight=s.retrieval.dense_weight,
                    sparse_weight=s.retrieval.sparse_weight,
                    law_hints=hints,
                    sparse_pool_ratio=getattr(s, "retrieval_sparse_pool_ratio", 1.0),
                    effective_as_of=as_of,
                )
                if rescue:
                    rescue = _rerank_candidates(rescue, question, reranker)
                    candidates = _merge_by_parent(candidates, rescue)
                effective_mode = "hybrid"
                rescued_mode = True
                logger.info(
                    "检索选路低置信回退：dense top1=%.3f < %.2f，改用 hybrid 重查合并 → %d 条候选",
                    top1_dense, s.retrieval_dense_rescue_score, len(candidates),
                )

        # 法域过滤低置信回退：top1 重排分过低 → 意图大概率判错法域，正确法条
        # 被硬过滤整条筛掉（law_hints 质量随意图模型波动）。无过滤重查一次，
        # 同一 query 的重排分可比，按法条合并保留高分。仅在失败场景多一次检索。
        top1 = hybrid._rank_value(candidates[0]) if candidates else 0.0
        if (
            hints
            and s.retrieval_law_rescue_score > 0
            and top1 < s.retrieval_law_rescue_score
        ):
            rescue = hybrid.search(
                client,
                collection,
                embedding,
                pool=pool,
                kb_ids=kb_ids,
                mode=effective_mode,
                rrf_k=s.retrieval.rrf_k,
                ranker=s.retrieval_ranker,
                dense_weight=s.retrieval.dense_weight,
                sparse_weight=s.retrieval.sparse_weight,
                law_hints=None,
                sparse_pool_ratio=getattr(s, "retrieval_sparse_pool_ratio", 1.0),
                effective_as_of=as_of,
            )
            if rescue:
                rescue = _rerank_candidates(rescue, question, reranker)
                candidates = _merge_by_parent(candidates, rescue)
            logger.info(
                "法域过滤低置信回退：top1 重排分 %.3f < %.2f，无过滤重查合并 → %d 条候选",
                top1, s.retrieval_law_rescue_score, len(candidates),
            )
    rerank_ms = int((time.perf_counter() - rerank_started) * 1000) if reranked else 0

    hits = hybrid.dedupe_by_parent(candidates, topk=resolved_topk)
    kb_versions: dict = {}
    if expand_parent and hits:
        _attach_parents(hits, as_of=as_of, kb_ids=kb_ids)
        # 版本快照：命中带上来源知识库的 corpus/index 版本，供 trace/评估追溯。
        kb_versions = _attach_kb_versions(hits, kb_ids=kb_ids)

    elapsed = int((time.perf_counter() - started) * 1000)
    logger.info(
        "检索完成 q=%.40s kb=%s law_hints=%s complexity=%s topK=%d pool=%d 候选=%d 重排=%s 命中=%d 耗时=%dms",
        question,
        kb_ids or "all",
        hints or "all",
        complexity,
        resolved_topk,
        pool,
        dense_hits,
        reranked,
        len(hits),
        elapsed,
    )
    return RetrievalResult(
        hits=hits,
        question=question,
        topk=resolved_topk,
        pool=pool,
        complexity=complexity,
        mode=effective_mode,
        reranked=reranked,
        elapsed_ms=elapsed,
        debug={
            "candidates": dense_hits,
            "pool": pool,
            "kb_ids": kb_ids or [],
            "law_hints": hints,
            "rrf_k": s.retrieval.rrf_k,
            "ranker": s.retrieval_ranker,
            "mode": effective_mode,
            "mode_requested": mode,
            "mode_rescued": rescued_mode,
            "effective_as_of": as_of,
            "kb_versions": kb_versions,
            "rerank_ms": rerank_ms,
        },
    )


def _attach_parents(
    hits: list[Hit],
    *,
    as_of: int = 0,
    kb_ids: list[str] | None = None,
) -> None:
    """按 parent_id 批量回表取整条法条 + 时效元信息，必要时兜底过滤失效条。

    回表同时校验文档状态与版本：Milvus 里可能残留旧版本/已删除文档的向量，
    父块或文档在 MySQL 中查不到（或 ``chunk.version != doc.version``、
    文档非 ``ready``）时直接丢弃该命中，避免用过期法条作答。

    ``kb_ids`` 非空时，回表 SQL 也带 kb 条件：即使向量过滤被绕过，也不会把
    其他知识库/租户的正文取回来。
    """
    parent_ids = [h.parent_id or h.chunk_id for h in hits]
    from app.db.mysql import get_engine

    base = (
        "SELECT c.id, c.content, c.effective_from, c.effective_to, c.version "
        "FROM document_chunks c JOIN documents d ON d.id = c.doc_id "
        "WHERE c.id IN :ids AND c.chunk_type = 'parent' AND c.is_enabled = 1 "
        "AND d.status = 'ready' AND c.version = d.version"
    )
    if kb_ids:
        stmt = text(
            base + " AND c.kb_id IN :kbs"
        ).bindparams(bindparam("ids", expanding=True), bindparam("kbs", expanding=True))
        params: dict = {"ids": parent_ids, "kbs": list(kb_ids)}
    else:
        stmt = text(base).bindparams(bindparam("ids", expanding=True))
        params = {"ids": parent_ids}
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(stmt, params).fetchall()
    except Exception as exc:  # noqa: BLE001 - 回表失败不应让整次检索失败
        logger.warning("父块回表失败，降级使用子块正文：%s", exc)
        return
    meta = {
        row[0]: (row[1], str(row[2] or ""), str(row[3] or ""), int(row[4] or 0))
        for row in rows
    }
    kept: list[Hit] = []
    for hit in hits:
        row = meta.get(hit.parent_id or hit.chunk_id)
        if row is None:
            # 父块不存在 / 文档已删除 / 版本已过期：向量是脏数据，不能作答。
            continue
        hit.parent_content, hit.effective_from, hit.effective_to, hit.version = row
        # 显式 as_of 时按日期兜底；老数据（日期为空）视为有效，不过度过滤。
        start = date_to_int(hit.effective_from)
        end = date_to_int(hit.effective_to)
        if as_of and ((start and start > as_of) or (end and end < as_of)):
            continue
        kept.append(hit)
    hits[:] = kept


def _attach_kb_versions(
    hits: list[Hit],
    *,
    kb_ids: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """给命中补知识库版本快照；知识库已不存在时丢弃该命中。

    返回 ``{kb_id: {"corpus_version": n, "index_version": n}}``，写入检索 debug
    以便 trace 与评估能追溯"这条结果来自哪个版本"。
    """
    if not hits:
        return {}
    wanted = sorted({h.kb_id for h in hits if h.kb_id})
    if kb_ids:
        allowed = set(kb_ids)
        wanted = [kb for kb in wanted if kb in allowed]
    if not wanted:
        return {}
    from app.db.mysql import get_engine

    stmt = text(
        "SELECT id, corpus_version, index_version FROM knowledge_bases WHERE id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(stmt, {"ids": wanted}).fetchall()
    except Exception as exc:  # noqa: BLE001 - 版本列缺失时降级为 0，不阻断检索
        logger.warning("知识库版本回表失败，命中版本置 0：%s", exc)
        return {}
    versions = {
        row[0]: {"corpus_version": int(row[1] or 0), "index_version": int(row[2] or 0)}
        for row in rows
    }
    kept: list[Hit] = []
    for hit in hits:
        info = versions.get(hit.kb_id)
        if info is None:
            continue
        hit.corpus_version = info["corpus_version"]
        hit.index_version = info["index_version"]
        kept.append(hit)
    hits[:] = kept
    return versions


def _default_client():
    from app.db.milvus_client import get_milvus_db_client

    return get_milvus_db_client()


def recall_at_k(hits: list[Hit], expected: set[str], k: int) -> float:
    """条级 Recall@K：命中任一子块即该条命中。

    ``expected`` 用 ``法名#条号`` 标识，避免两部法律同条号混淆。
    """
    if not expected:
        return 0.0
    got = {h.law_article for h in hits[:k]}
    return len(got & expected) / len(expected)


def mrr(hits: list[Hit], expected: set[str]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if hit.law_article in expected:
            return 1.0 / rank
    return 0.0


def retrieve_multi(
    sub_queries: Sequence[str],
    *,
    kb_ids: list[str] | None = None,
    complexity: str = "medium",
    topk: int | None = None,
    law_hints: list[str] | None = None,
    anchor_query: str | None = None,
    mode: str = "hybrid",
    settings: Settings | None = None,
    retrieve_fn=None,
    max_workers: int = 3,
) -> RetrievalResult:
    """多子查询检索：**并行**分检 → RRF 按名次融合 → 按法条去重 → 全局截断。

    生产链路（``AnswerEngine._retrieve``）与 ``recall_eval`` 共用本函数，
    保证"评测口径 = 线上链路"（REPORT §4.2 假设 C：多跳问题必须拆子查询分检）。

    * 每个子查询取 ``MULTI_POOL_FACTOR × 最终topK`` 条，给融合留余量；
    * **跨子查询融合按名次交错取位（interleave），不能用 max 分数也不能用
      平缓 RRF**：rerank 分数是对各自 query 打的、量纲不可比（max 聚合让
      "分数偏高的子查询"垄断排序）；RRF(k=60) 曲线太平，多跳场景下每条
      期望法条只在对应子查询里排第 1，跨法近义条却在所有投票人里都排 3-5
      名、多票累计反而压过单票第 1。交错取位保证每个子查询的头部法条
      必进最终列表（多跳召回的关键），锚点排最前（保住单跳 MRR）；
    * ``anchor_query``（通常是用户原句）作为第一路投票人参与融合：
      原句直检对单跳强、子查询对多跳强，交错融合两者取并集优势；
    * 合并按 ``parent_id``（同一法条的不同子块只留一条），个别子查询失败
      不致命：只要有任一成功就继续（部分降级），全失败才抛异常。
    """
    s = settings or get_settings()
    fn = retrieve_fn or retrieve
    final_topk = topk or s.retrieval.topk_for(complexity)
    per_topk = max(int(final_topk * MULTI_POOL_FACTOR), final_topk)
    subs = [q.strip() for q in sub_queries if q and q.strip()]
    anchor = (anchor_query or "").strip()
    if anchor and anchor not in subs:
        subs = [anchor] + subs
    if not subs:
        raise ValueError("sub_queries 为空")
    started = time.perf_counter()

    def _one(q: str):
        return fn(
            q,
            kb_ids=kb_ids,
            complexity=complexity,
            topk=per_topk,
            mode=mode,
            law_hints=law_hints,
            settings=s,
        )

    errors: list[Exception] = []
    results: list = []
    if len(subs) == 1:
        results.append(_one(subs[0]))
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(subs))) as ex:
            futures = [(q, ex.submit(_one, q)) for q in subs]
            for q, fut in futures:
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001 - 部分子查询失败不拖垮整体
                    errors.append(exc)
                    logger.warning("子查询检索失败（部分降级）q=%.40s：%s", q, str(exc)[:120])
        if not results:
            raise errors[0]

    # 名次交错融合 + 法条（parent）去重：第 r 轮每个投票人各取自己第 r 名，
    # 锚点（results[0]）每轮先取。已入列的法条跳过，直到凑满 final_topk。
    reranked = False
    ranked_lists: list[list[Hit]] = []
    for r in results:
        reranked = reranked or r.reranked
        ranked_lists.append(sorted(r.hits, key=hybrid._rank_value, reverse=True))
    best_hit: dict[str, Hit] = {}
    ordered_keys: list[str] = []
    for round_rank in range(max((len(rl) for rl in ranked_lists), default=0)):
        for ranked in ranked_lists:
            if round_rank >= len(ranked):
                continue
            hit = ranked[round_rank]
            key = hit.parent_id or hit.chunk_id
            prev = best_hit.get(key)
            if prev is None:
                best_hit[key] = hit
                ordered_keys.append(key)
            elif hybrid._rank_value(hit) > hybrid._rank_value(prev):
                best_hit[key] = hit
        if len(ordered_keys) >= final_topk:
            break
    hits = [best_hit[k] for k in ordered_keys[:final_topk]]

    # 法条互引扩展：多跳才做。沿头部命中的显式引用补入被引条，插在来源条之后，
    # 尾部低名次条被挤出 topK。只扩展前 N 条、最多补 M 条，避免引入噪声。
    refs_added = 0
    if getattr(s, "retrieval_ref_expansion", False) and hits:
        hits, refs_added = refs.expand_hits_with_refs(
            hits,
            topk=final_topk,
            kb_ids=kb_ids,
            max_sources=getattr(s, "retrieval_ref_max_sources", 3),
            max_added=getattr(s, "retrieval_ref_max_added", 3),
            as_of=_as_of_int(s),
            settings=s,
        )

    kb_versions: dict[str, dict[str, int]] = {}
    for hit in hits:
        if hit.kb_id and hit.kb_id not in kb_versions:
            kb_versions[hit.kb_id] = {
                "corpus_version": int(hit.corpus_version or 0),
                "index_version": int(hit.index_version or 0),
            }
    rerank_ms = sum(
        int((getattr(r, "debug", None) or {}).get("rerank_ms", 0) or 0) for r in results
    )
    elapsed = int((time.perf_counter() - started) * 1000)
    logger.info(
        "多子查询检索完成：查询 %d 路 × topK=%d → 交错融合 %d 条法条 → 截断 %d 条"
        "（重排=%s，失败子查询 %d 条）耗时=%dms",
        len(subs), per_topk, len(ordered_keys), len(hits), reranked, len(errors), elapsed,
    )
    return RetrievalResult(
        hits=hits,
        question="｜".join(subs),
        topk=final_topk,
        pool=per_topk,
        complexity=complexity,
        mode="multi",
        reranked=reranked,
        elapsed_ms=elapsed,
        debug={
            "sub_queries": subs,
            "per_subquery_topk": per_topk,
            "failed_subqueries": len(errors),
            "merge": "interleave",
            "retrieval_mode": mode,
            "ref_expansion": bool(getattr(s, "retrieval_ref_expansion", False)),
            "refs_added": refs_added,
            "mode": "multi",
            "kb_ids": list(kb_ids or []),
            "kb_versions": kb_versions,
            "rerank_ms": rerank_ms,
        },
    )
