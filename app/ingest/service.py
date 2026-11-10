"""摄入/文档服务层：供 API 与 CLI 共用，避免业务逻辑写进路由。

职责：
* 上传落盘与校验（扩展名、大小、切块参数）
* 后台入库任务（解析慢，必须异步，前端轮询 documents.status/progress）
* 单块编辑/删除后的向量同步（MySQL 是事实源，Milvus 跟随）
"""

from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.errors import UpstreamError
from app.core.logger import get_logger
from app.ingest.embedder import get_embedder
from app.ingest.gate import IngestBusy, get_ingest_gate
from app.ingest.loaders import SUPPORTED_EXT, UnsupportedDocumentError
from app.ingest.pipeline import ingest, new_id, prepare
from app.ingest.writer import chunk_to_entity, delete_chunk_vector, refresh_kb_counts, update_document_status
from app.ingest.writer import delete_chunk_vectors, delete_document_vectors, upsert_vectors
from config import get_settings

logger = get_logger("lawanswer.ingest")

MAX_UPLOAD_BYTES = 80 * 1024 * 1024


class UploadRejected(ValueError):
    """上传校验失败（前端可直接展示 message）。"""


class KbBusyError(RuntimeError):
    """知识库还有文档在解析中，暂时不能删（后台任务会把行写回来）。"""


def bump_kb_versions(session: Session, kb_id: str, *, corpus: bool = True, index: bool = True) -> None:
    """语料/索引版本 +1：答案缓存 key 随之变化，旧缓存自然失效。"""
    parts = []
    if corpus:
        parts.append("corpus_version = corpus_version + 1")
    if index:
        parts.append("index_version = index_version + 1")
    if not parts:
        return
    session.execute(
        text(f"UPDATE knowledge_bases SET {', '.join(parts)} WHERE id = :kb_id"),
        {"kb_id": kb_id},
    )


def validate_upload(filename: str, size: int, chunk_size: int, overlap: int) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXT:
        raise UploadRejected(
            f"不支持的文件类型 {suffix or '(无扩展名)'}，支持："
            + "、".join(sorted(SUPPORTED_EXT))
        )
    if size <= 0:
        raise UploadRejected("文件为空")
    if size > MAX_UPLOAD_BYTES:
        raise UploadRejected(f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限")
    if chunk_size < 50 or chunk_size > 2000:
        raise UploadRejected("切块大小需在 50–2000 之间")
    if overlap < 0 or overlap >= chunk_size:
        raise UploadRejected("重叠大小必须 >= 0 且小于切块大小")


def save_upload(data: bytes, kb_id: str, filename: str) -> Path:
    settings = get_settings()
    target_dir = settings.upload_path / kb_id
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    path = target_dir / f"{new_id()}_{safe_name}"
    path.write_bytes(data)
    return path


def _placeholder_sha1() -> str:
    """每次唯一的 40 位占位哈希（仅用于 pending 行，入库后会被真实 sha1 覆盖）。"""
    import secrets

    return "p" + secrets.token_hex(19) + "0"


def create_pending_document(
    session: Session,
    *,
    doc_id: str,
    kb_id: str,
    filename: str,
    file_type: str,
    file_size: int,
    storage_path: str,
    chunk_size: int,
    overlap: int,
    retrieval_mode: str,
) -> None:
    import json

    session.execute(
        text(
            """
            INSERT INTO documents
                (id, kb_id, filename, file_type, file_size, content_sha1, file_sha1,
                 storage_path, status, progress, parse_config)
            VALUES
                (:id, :kb_id, :filename, :file_type, :file_size, :placeholder, :placeholder,
                 :storage_path, 'pending', 0, :parse_config)
            """
        ),
        {
            "id": doc_id,
            "kb_id": kb_id,
            "filename": filename,
            "file_type": file_type,
            "file_size": file_size,
            # 占位 sha1 必须每次不同：documents 上有 uk_kb_content(kb_id, content_sha1)，
            # 全用 "0"*40 会让同一知识库的第二条 pending 记录直接撞唯一键（上传即 500）。
            # 解析出真实内容后由入库流程覆盖成 content_sha1。
            "placeholder": _placeholder_sha1(),
            "storage_path": storage_path,
            "parse_config": json.dumps(
                {"chunk_size": chunk_size, "overlap": overlap, "retrieval_mode": retrieval_mode},
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    )


def run_ingest_job(
    path: str | Path,
    *,
    kb_id: str,
    doc_id: str,
    chunk_size: int,
    overlap: int,
    retrieval_mode: str = "hybrid",
    force: bool = False,
    admitted: bool = False,
) -> dict:
    """后台任务入口：任何异常都要落库成 failed，日志留痕。

    ``admitted=True`` 表示上传接口已经通过 :func:`get_ingest_gate` 占好名额，
    本函数结束时负责归还；==False 时（CLI/重试等直接调用）先尝试自行准入，
    队列满则直接标记 failed，不进入解析。
    """
    gate = get_ingest_gate()
    if not admitted and not gate.try_admit():
        error = "解析队列已满，任务未执行，请稍后重新解析"
        logger.warning("入库任务被闸门拒绝 doc=%s 在途=%d", doc_id, gate.outstanding)
        _mark_failed(doc_id, error)
        return {"doc_id": doc_id, "status": "rejected", "error": error}

    try:
        with gate.worker_slot():
            prepared = prepare(
                path,
                chunk_size=chunk_size,
                overlap=overlap,
                retrieval_mode=retrieval_mode,
            )
            result = ingest(
                prepared,
                kb_id=kb_id,
                doc_id=doc_id,
                force=force,
            )
        logger.info(
            "入库完成 doc=%s 文件=%s 状态=%s 条=%d 子块=%d 耗时=%dms",
            result.doc_id,
            Path(path).name,
            result.status,
            result.article_count,
            result.child_count,
            result.elapsed_ms,
        )
        return {
            "doc_id": result.doc_id,
            "status": result.status,
            "article_count": result.article_count,
            "child_count": result.child_count,
        }
    except IngestBusy as exc:
        logger.warning("入库任务等待名额超时 doc=%s: %s", doc_id, exc)
        _mark_failed(doc_id, str(exc))
        return {"doc_id": doc_id, "status": "failed", "error": str(exc)[:300]}
    except Exception as exc:  # noqa: BLE001 - 后台任务必须吞掉异常并落库
        logger.exception("入库失败 文件=%s", Path(path).name)
        _mark_failed(doc_id, str(exc))
        return {"doc_id": doc_id, "status": "failed", "error": str(exc)[:300]}
    finally:
        gate.release()


def _mark_failed(doc_id: str, error: str) -> None:
    """把文档状态落成 failed（后台任务与闸门拒绝共用）。"""
    from app.db.mysql import session_scope

    with session_scope() as db:
        update_document_status(db, doc_id, status="failed", error=error[:300])


def update_chunk_content(session: Session, chunk_id: str, content: str) -> dict:
    """编辑单块：写 MySQL → 重新向量化 → upsert Milvus（同 chunk_id 覆盖）。"""
    row = session.execute(
        text(
            "SELECT c.id, c.doc_id, c.kb_id, c.parent_id, c.chunk_type, c.chunk_index, "
            "c.article_no, c.chapter_path, c.title, d.content_sha1, d.filename "
            "FROM document_chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = :cid"
        ),
        {"cid": chunk_id},
    ).mappings().first()
    if row is None:
        raise LookupError("块不存在")

    session.execute(
        text(
            "UPDATE document_chunks SET content = :content, char_count = :len, in_milvus = 0 "
            "WHERE id = :cid"
        ),
        {"content": content, "len": len(content), "cid": chunk_id},
    )

    settings = get_settings()
    from app.db.milvus_client import get_milvus_db_client

    chunk = _row_to_chunk(row, content)
    embedding = get_embedder().encode([chunk.embed_text])[0]
    entity = chunk_to_entity(
        chunk,
        embedding,
        law_name=_law_name_of(session, row["doc_id"]),
        source_file=row["filename"],
        doc_sha1=row["content_sha1"],
    )
    upsert_vectors(get_milvus_db_client(), settings.milvus_collection, [entity])
    session.execute(
        text("UPDATE document_chunks SET in_milvus = 1 WHERE id = :cid"), {"cid": chunk_id}
    )
    bump_kb_versions(session, row["kb_id"], corpus=True, index=True)
    return {"chunk_id": chunk_id, "char_count": len(content)}


def disable_chunk(session: Session, chunk_id: str) -> None:
    """删除块：软删 MySQL + 物理删向量（检索侧不再命中）。"""
    row = session.execute(
        text("SELECT id, kb_id FROM document_chunks WHERE id = :cid"), {"cid": chunk_id}
    ).mappings().first()
    if row is None:
        raise LookupError("块不存在")
    session.execute(
        text("UPDATE document_chunks SET is_enabled = 0, in_milvus = 0 WHERE id = :cid"),
        {"cid": chunk_id},
    )
    settings = get_settings()
    from app.db.milvus_client import get_milvus_db_client

    delete_chunk_vector(get_milvus_db_client(), settings.milvus_collection, chunk_id)
    bump_kb_versions(session, row["kb_id"], corpus=True, index=True)


def delete_document(session: Session, doc_id: str) -> dict:
    """删除文档：Milvus 向量 → MySQL 行（块靠外键级联）→ 版本号 +1。

    返回 ``{"kb_id": ..., "storage_path": ...}``，源文件由调用方在事务提交
    **成功之后**调用 :func:`purge_document_file` 清理——先删文件再回滚会不可逆。
    """
    row = session.execute(
        text("SELECT kb_id, storage_path FROM documents WHERE id = :id"), {"id": doc_id}
    ).mappings().first()
    if row is None:
        raise LookupError("文档不存在")
    settings = get_settings()
    from app.db.milvus_client import get_milvus_db_client

    delete_document_vectors(get_milvus_db_client(), settings.milvus_collection, doc_id)
    session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
    refresh_kb_counts(session, row["kb_id"])
    bump_kb_versions(session, row["kb_id"], corpus=True, index=True)
    return {"kb_id": row["kb_id"], "storage_path": row["storage_path"]}


def purge_document_file(storage_path: str | None, kb_id: str) -> bool:
    """删掉单个文档的源文件；只允许删上传目录内该知识库下的文件。"""
    if not storage_path:
        return False
    root = get_settings().upload_path.resolve()
    target = Path(storage_path)
    if not target.is_absolute():
        target = (Path.cwd() / target)
    try:
        resolved = target.resolve()
    except OSError:
        return False
    try:
        resolved.relative_to((root / kb_id).resolve())
    except ValueError:
        logger.warning("拒绝删除上传目录外的源文件 path=%s kb=%s", storage_path, kb_id)
        return False
    if not resolved.is_file():
        return False
    resolved.unlink(missing_ok=True)
    return True


def delete_knowledge_base(session: Session, kb_id: str) -> dict:
    """删除整个知识库（文档、块、向量一起走），返回删除前的统计量。

    顺序有讲究：

    * **先删 Milvus 再删 MySQL**。MySQL 是事实源，向量只是派生物；反过来先删库行，
      一旦 Milvus 失败就会留下一批"检索得到、预览看不到"的孤儿向量，只能人工清。
    * documents / document_chunks / ingest_jobs 靠外键 ON DELETE CASCADE 跟着走，
      这里不再逐表 DELETE（逐表删反而容易漏掉 ingest_jobs）。
    * ``qa_pairs.kb_id`` 没有外键约束，必须手动置空，否则结构化检索会指向不存在的库。
    * 解析中的库直接拒绝：后台任务结束时会把状态写回 documents，行已经没了就会报错
      并留下一条 failed 任务，用户看到的是"删了还在解析"。
    * 源文件不在这里删——调用方提交事务成功后再调 :func:`purge_kb_files`，
      避免"文件删了、事务回滚了"这种不可逆的空档。
    """
    kb = session.execute(
        text("SELECT id, name, doc_count, chunk_count FROM knowledge_bases WHERE id = :id"),
        {"id": kb_id},
    ).mappings().first()
    if kb is None:
        raise LookupError(f"知识库 {kb_id} 不存在")

    busy = session.execute(
        text(
            "SELECT COUNT(*) FROM documents WHERE kb_id = :kb_id "
            "AND status IN ('pending','parsing','chunking','embedding')"
        ),
        {"kb_id": kb_id},
    ).scalar()
    if busy:
        raise KbBusyError(
            f"该知识库还有 {int(busy)} 个文档正在解析，请等解析结束或先删除这些文档"
        )

    chunk_ids = [
        row[0]
        for row in session.execute(
            text("SELECT id FROM document_chunks WHERE kb_id = :kb_id"), {"kb_id": kb_id}
        ).fetchall()
    ]

    settings = get_settings()
    from app.db.milvus_client import get_milvus_db_client

    deleted_vectors = 0
    if chunk_ids:
        try:
            deleted_vectors = delete_chunk_vectors(
                get_milvus_db_client(), settings.milvus_collection, chunk_ids
            )
        except Exception as exc:  # noqa: BLE001 - 向量库故障要让用户看懂，而不是 500
            logger.exception("删除知识库向量失败 id=%s", kb_id)
            # 此时 MySQL 一行都没动，库还是完整的，用户重试即可
            raise UpstreamError("向量库删除失败，知识库未删除，请稍后重试") from exc

    session.execute(
        text("UPDATE qa_pairs SET kb_id = NULL WHERE kb_id = :kb_id"), {"kb_id": kb_id}
    )
    session.execute(text("DELETE FROM knowledge_bases WHERE id = :id"), {"id": kb_id})

    logger.info(
        "知识库已删除 id=%s 名称=%s 文档=%s 块=%s 向量=%d",
        kb_id,
        kb["name"],
        kb["doc_count"],
        kb["chunk_count"],
        deleted_vectors,
    )
    return {
        "deleted": kb_id,
        "name": kb["name"],
        "doc_count": int(kb["doc_count"] or 0),
        "chunk_count": int(kb["chunk_count"] or 0),
        "chunk_vectors": deleted_vectors,
    }


def purge_kb_files(kb_id: str) -> int:
    """删掉该库的上传源文件目录，返回删掉的文件数（库已删干净后才调用）。

    只允许删 ``upload_path`` 的**直接子目录**：kb_id 来自 URL，若被构造成
    ``..`` 或 ``a/b``，不做校验就可能顺手删掉整个上传目录。
    """
    root = get_settings().upload_path.resolve()
    target = (root / kb_id).resolve()
    if target.parent != root or not target.is_dir():
        return 0
    removed = sum(1 for p in target.rglob("*") if p.is_file())
    shutil.rmtree(target, ignore_errors=True)
    logger.info("已清理知识库源文件 id=%s 文件数=%d", kb_id, removed)
    return removed


def _law_name_of(session: Session, doc_id: str) -> str:
    name = session.execute(
        text("SELECT filename FROM documents WHERE id = :id"), {"id": doc_id}
    ).scalar()
    if not name:
        return ""
    from app.ingest.structure import law_name_from_filename

    return law_name_from_filename(name)


def _row_to_chunk(row, content: str):
    from app.ingest.models import Chunk

    return Chunk(
        chunk_id=row["id"],
        doc_id=row["doc_id"],
        kb_id=row["kb_id"],
        parent_id=row["parent_id"],
        chunk_type=row["chunk_type"],
        chunk_index=int(row["chunk_index"]),
        article_no=row["article_no"] or "",
        chapter_path=row["chapter_path"] or "",
        title=row["title"] or "",
        content=content,
        char_count=len(content),
    )


def copy_local_file(path: Path, kb_id: str) -> Path:
    """把已在本地磁盘的文件（如 data/raw）复制进上传目录，保持存储路径统一。"""
    target = get_settings().upload_path / kb_id / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return target
