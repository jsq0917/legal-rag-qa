"""通用知识库（generic）端到端复核：真实 MySQL + Milvus + LLM。

用途：验证 N3 产品资料链路在修复「只保留 kind=law 来源」后是否可用——
检索结果能进 Prompt、模型能引用《手册》·章节、SSE 来源不被去重吞掉。

运行（GPU 不可用时强制 CPU，避免 bge-m3 CUDA illegal instruction）：

    EMBEDDING_DEVICE=cpu uv run python scripts/verify_generic_qa.py \
        --kb kb_generic --query "滤芯指示灯变成红色常亮，应该怎么处理？"

产物：``eval/results/generic_qa_live.json``（含 command 与真实条数/回答）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from app.db.milvus_client import get_milvus_client  # noqa: E402
from app.db.mysql import get_engine  # noqa: E402
from app.generation.context import aggregate_sources, build_context_sources  # noqa: E402
from app.generation.llm_client import get_llm_client  # noqa: E402
from app.generation.prompts import build_context_block, build_messages  # noqa: E402
from app.orchestration.service import analyze_query  # noqa: E402
from app.retrieval.service import retrieve, retrieve_multi  # noqa: E402
from config import get_settings  # noqa: E402

DEFAULT_QUERY = "滤芯指示灯变成红色常亮，应该怎么处理？"


def _mysql_counts(kb_id: str, settings) -> dict:
    with get_engine().connect() as conn:
        kb = conn.execute(
            text(
                "SELECT id, tenant_id, is_public, doc_count, chunk_count, "
                "corpus_version, index_version FROM knowledge_bases WHERE id = :k"
            ),
            {"k": kb_id},
        ).mappings().first()
        docs = conn.execute(
            text(
                "SELECT id, filename, status, article_count, chunk_count "
                "FROM documents WHERE kb_id = :k ORDER BY filename"
            ),
            {"k": kb_id},
        ).mappings().all()
        parents = conn.execute(
            text(
                "SELECT COUNT(*) FROM document_chunks "
                "WHERE kb_id = :k AND chunk_type = 'parent'"
            ),
            {"k": kb_id},
        ).scalar()
    return {
        "kb": dict(kb) if kb else None,
        "documents": [dict(d) for d in docs],
        "parent_chunks": int(parents or 0),
        "milvus_database": settings.milvus.database,
        "collection": settings.milvus_collection,
    }


def _milvus_count(kb_id: str, settings) -> int:
    client = get_milvus_client()
    client.using_database(settings.milvus.database)
    rows = client.query(
        collection_name=settings.milvus_collection,
        filter=f'kb_id == "{kb_id}"',
        output_fields=["count(*)"],
    )
    return int(rows[0]["count(*)"]) if rows else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", default="kb_generic")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--out", default=str(ROOT / "eval" / "results" / "generic_qa_live.json"))
    args = parser.parse_args()

    settings = get_settings()
    command = (
        "EMBEDDING_DEVICE=cpu uv run python scripts/verify_generic_qa.py "
        f'--kb {args.kb} --query "{args.query}"'
    )

    mysql = _mysql_counts(args.kb, settings)
    milvus_entities = _milvus_count(args.kb, settings)

    plan = analyze_query(args.query, selected_kb_ids=[args.kb], settings=settings)
    kb_ids = list(plan.route.kb_ids) or [args.kb]
    kwargs = dict(
        kb_ids=kb_ids,
        complexity=plan.intent.complexity,
        law_hints=plan.intent.law_hints,
        settings=settings,
    )
    if len(plan.sub_queries) > 1:
        retrieved = retrieve_multi(plan.sub_queries, **kwargs)
    else:
        retrieved = retrieve(plan.sub_queries[0], route=plan.route.route, **kwargs)

    context_sources = build_context_sources(hits=retrieved.hits)
    knowledge_sources = [c for c in context_sources if c.kind in ("law", "section")]
    block, used_sources = build_context_block(
        knowledge_sources, max_chars=settings.gen_context_max_chars
    )
    messages = build_messages(args.query, context_block=block)
    response = get_llm_client(settings).chat(messages, max_tokens=settings.gen_max_tokens)
    answer = (response.content or "").strip()

    refs = aggregate_sources(hits=retrieved.hits)
    cited = [c.title for c in used_sources if c.chapter_path and c.chapter_path in answer]
    payload = {
        "command": command,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kb_id": args.kb,
        "query": args.query,
        "mysql": mysql,
        "milvus": {"entities": milvus_entities},
        "route": plan.route.route,
        "degraded": plan.degraded,
        "retrieval": {
            "count": len(retrieved.hits),
            "top": [
                {
                    "law": h.law_name,
                    "chapter_path": h.chapter_path,
                    "article_no": h.article_no,
                    "score": h.rerank_score if h.rerank_score is not None else h.score,
                    "chunk_id": h.chunk_id,
                }
                for h in retrieved.hits[:5]
            ],
        },
        "sses": {
            "aggregated_sources": len(refs),
            "kinds": sorted({r.kind for r in refs}),
        },
        "prompt": {
            "context_chars": len(block),
            "knowledge_sources": len(used_sources),
        },
        "answer": answer,
        "citation": {
            "cited_sections": cited,
            "cited_expected_section": bool(cited),
        },
        "model": response.model,
        "usage": response.usage,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if answer and retrieved.hits and used_sources else 1


if __name__ == "__main__":
    raise SystemExit(main())
