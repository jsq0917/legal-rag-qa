"""写入层：MySQL（事实源）+ Milvus（向量），保证两边不漂移。

写入顺序刻意是 **MySQL 先、Milvus 后**：

* MySQL 是块的权威副本，预览/编辑/删除都以它为准；
* Milvus 写失败时，文档状态置 `failed`，用户可以重试；反过来若 Milvus 先写成功
  而 MySQL 失败，就会出现"检索得到、但预览看不到"的孤儿向量，排查困难。

Milvus 侧只存**子块**（检索单元），父块正文留在 MySQL。
删除一律按 `doc_id` 表达式删，新增用 `upsert`（chunk_id 是主键，天然幂等）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.ingest.embedder import Embedding, embeddings_to_milvus
from app.ingest.models import Chunk
from app.ingest.versioning import date_to_int

MILVUS_BATCH = 256


def now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


# ---------------------------------------------------------------------------
# MySQL
# ---------------------------------------------------------------------------
def get_document_by_sha1(session: Session, kb_id: str, content_sha1: str) -> dict | None:
    row = session.execute(
        text(
            "SELECT id, status, parse_config, article_count, chunk_count, version, "
            "effective_from, effective_to "
            "FROM documents WHERE kb_id = :kb_id AND content_sha1 = :sha1"
        ),
        {"kb_id": kb_id, "sha1": content_sha1},
    ).mappings().first()
    if not row:
        return None
    data = dict(row)
    # MySQL 的 JSON 列经 PyMySQL 读回来是字符串，这里统一解码，
    # 否则调用方 dict(...) 会直接抛 ValueError（踩过）。
    if isinstance(data.get("parse_config"), str):
        try:
            data["parse_config"] = json.loads(data["parse_config"])
        except json.JSONDecodeError:
            data["parse_config"] = {}
    return data


def upsert_document(session: Session, document: dict) -> None:
    session.execute(
        text(
            """
            INSERT INTO documents
                (id, kb_id, filename, file_type, file_size, content_sha1, file_sha1,
                 storage_path, status, progress, parse_config, article_count, chunk_count,
                 version, effective_from, effective_to)
            VALUES
                (:id, :kb_id, :filename, :file_type, :file_size, :content_sha1, :file_sha1,
                 :storage_path, :status, :progress, :parse_config, :article_count, :chunk_count,
                 :version, :effective_from, :effective_to)
            ON DUPLICATE KEY UPDATE
                -- content_sha1 必须一起更新：上传接口先落 pending 行（占位 sha1），
                -- 真实内容算出后若不回写，去重（uk_kb_content）与"内容未变跳过"全部失效
                content_sha1 = VALUES(content_sha1),
                filename = VALUES(filename),
                file_type = VALUES(file_type),
                file_size = VALUES(file_size),
                file_sha1 = VALUES(file_sha1),
                storage_path = VALUES(storage_path),
                status = VALUES(status),
                progress = VALUES(progress),
                parse_config = VALUES(parse_config),
                article_count = VALUES(article_count),
                chunk_count = VALUES(chunk_count),
                version = VALUES(version),
                effective_from = VALUES(effective_from),
                effective_to = VALUES(effective_to),
                error_msg = NULL
            """
        ),
        document,
    )


def update_document_status(
    session: Session,
    doc_id: str,
    *,
    status: str,
    progress: int | None = None,
    error: str | None = None,
    article_count: int | None = None,
    chunk_count: int | None = None,
) -> None:
    session.execute(
        text(
            """
            UPDATE documents
               SET status = :status,
                   progress = COALESCE(:progress, progress),
                   error_msg = :error,
                   article_count = COALESCE(:article_count, article_count),
                   chunk_count = COALESCE(:chunk_count, chunk_count)
             WHERE id = :doc_id
            """
        ),
        {
            "status": status,
            "progress": progress,
            "error": (error[:512] if error else None),
            "article_count": article_count,
            "chunk_count": chunk_count,
            "doc_id": doc_id,
        },
    )


def delete_chunks(session: Session, doc_id: str) -> None:
    session.execute(text("DELETE FROM document_chunks WHERE doc_id = :doc_id"), {"doc_id": doc_id})


def insert_chunks(session: Session, chunks: Sequence[Chunk]) -> None:
    if not chunks:
        return
    payload = [
        {
            "id": c.chunk_id,
            "doc_id": c.doc_id,
            "kb_id": c.kb_id,
            "parent_id": c.parent_id,
            "chunk_type": c.chunk_type,
            "chunk_index": c.chunk_index,
            "article_no": c.article_no,
            "chapter_path": c.chapter_path,
            "title": c.title,
            "content": c.content,
            "char_count": c.char_count,
            "effective_from": c.effective_from or None,
            "effective_to": c.effective_to or None,
            "version": c.version,
        }
        for c in chunks
    ]
    session.execute(
        text(
            """
            INSERT INTO document_chunks
                (id, doc_id, kb_id, parent_id, chunk_type, chunk_index,
                 article_no, chapter_path, title, content, char_count,
                 effective_from, effective_to, version)
            VALUES
                (:id, :doc_id, :kb_id, :parent_id, :chunk_type, :chunk_index,
                 :article_no, :chapter_path, :title, :content, :char_count,
                 :effective_from, :effective_to, :version)
            """
        ),
        payload,
    )


def insert_ingest_job(session: Session, job_id: str, doc_id: str, stage: str) -> None:
    session.execute(
        text("INSERT INTO ingest_jobs (id, doc_id, stage, status) VALUES (:id, :doc_id, :stage, 'running')"),
        {"id": job_id, "doc_id": doc_id, "stage": stage},
    )


def finish_ingest_job(
    session: Session,
    job_id: str,
    *,
    status: str,
    processed: int = 0,
    total: int = 0,
    message: str | None = None,
) -> None:
    session.execute(
        text(
            """
            UPDATE ingest_jobs
               SET status = :status, processed = :processed, total = :total,
                   message = :message, finished_at = CURRENT_TIMESTAMP(3)
             WHERE id = :job_id
            """
        ),
        {
            "status": status,
            "processed": processed,
            "total": total,
            "message": (message[:512] if message else None),
            "job_id": job_id,
        },
    )


def refresh_kb_counts(session: Session, kb_id: str) -> None:
    session.execute(
        text(
            """
            UPDATE knowledge_bases kb
               SET doc_count = (SELECT COUNT(*) FROM documents d
                                 WHERE d.kb_id = kb.id AND d.status = 'ready'),
                   chunk_count = (SELECT COUNT(*) FROM document_chunks c
                                   WHERE c.kb_id = kb.id AND c.chunk_type = 'child')
             WHERE kb.id = :kb_id
            """
        ),
        {"kb_id": kb_id},
    )


# ---------------------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------------------
def chunk_to_entity(
    chunk: Chunk,
    embedding: Embedding,
    *,
    law_name: str,
    source_file: str,
    doc_sha1: str,
) -> dict:
    entity = {
        "chunk_id": chunk.chunk_id,
        "kb_id": chunk.kb_id,
        "doc_id": chunk.doc_id,
        "parent_id": chunk.parent_id,
        "content": chunk.content[:4000],
        "title": chunk.title[:255],
        "law_name": law_name[:64],
        "article_no": chunk.article_no[:32],
        "chapter_path": chunk.chapter_path[:255],
        "source_file": source_file[:255],
        "chunk_index": chunk.chunk_index,
        "char_count": chunk.char_count,
        "doc_sha1": doc_sha1,
        "updated_at": now_ms(),
        "effective_from": date_to_int(chunk.effective_from),
        "effective_to": date_to_int(chunk.effective_to),
        "version": int(chunk.version),
    }
    entity.update(embeddings_to_milvus(embedding))
    return entity


def upsert_vectors(
    client,
    collection: str,
    entities: Sequence[dict],
    batch_size: int = MILVUS_BATCH,
) -> int:
    written = 0
    for start in range(0, len(entities), batch_size):
        batch = list(entities[start : start + batch_size])
        client.upsert(collection_name=collection, data=batch)
        written += len(batch)
    return written


def delete_document_vectors(client, collection: str, doc_id: str) -> None:
    client.delete(collection_name=collection, filter=f'doc_id == "{doc_id}"')


VECTOR_DELETE_BATCH = 200


def delete_chunk_vectors(client, collection: str, chunk_ids: Sequence[str]) -> int:
    """按主键批量删向量（删整个知识库时用）。

    为什么不用 ``kb_id == "xxx"`` 一条表达式删完：``kb_id`` 是 partition key
    字段，Milvus 对分区键过滤的支持随版本/实现（lite vs standalone）差异较大，
    而主键删除在所有实现上都成立，且天然幂等——删到一半失败重试即可。
    """
    total = 0
    ids = [str(cid) for cid in chunk_ids]
    for start in range(0, len(ids), VECTOR_DELETE_BATCH):
        batch = ids[start : start + VECTOR_DELETE_BATCH]
        quoted = ", ".join(f'"{cid}"' for cid in batch)
        client.delete(collection_name=collection, filter=f"chunk_id in [{quoted}]")
        total += len(batch)
    return total


def delete_chunk_vector(client, collection: str, chunk_id: str) -> None:
    client.delete(collection_name=collection, filter=f'chunk_id == "{chunk_id}"')


def count_vectors(client, collection: str, kb_id: str | None = None) -> int:
    expr = f'kb_id == "{kb_id}"' if kb_id else ""
    result = client.query(
        collection_name=collection,
        filter=expr,
        output_fields=["count(*)"],
    )
    return int(result[0]["count(*)"]) if result else 0


def load_chunks(session: Session, doc_id: str) -> list[dict]:
    rows = session.execute(
        text(
            "SELECT id, chunk_type, parent_id, chunk_index, article_no, chapter_path, "
            "title, content, char_count, in_milvus, is_enabled "
            "FROM document_chunks WHERE doc_id = :doc_id ORDER BY chunk_index"
        ),
        {"doc_id": doc_id},
    ).mappings().all()
    return [dict(r) for r in rows]


def parse_config_json(chunk_size: int, overlap: int, retrieval_mode: str) -> str:
    return json.dumps(
        {"chunk_size": chunk_size, "overlap": overlap, "retrieval_mode": retrieval_mode},
        ensure_ascii=False,
        sort_keys=True,
    )


def batches(items: Iterable, size: int):
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
