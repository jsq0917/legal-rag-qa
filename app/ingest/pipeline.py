"""摄入流水线：加载 → 清洗 → 结构识别 → 父子分块 → 向量化 → 双写。

增量规则（对应设计文档 §5.2）
------------------------------
* **幂等**：同一知识库里，归一化正文的 SHA1 相同且状态 `ready`、解析参数一致 → 直接 `skipped`，
  不重复向量化（1465 条重跑一次要几十秒，能省就省）。
* **整篇替换**：内容变化（SHA1 不同）或解析参数变化 → 先按 `doc_id` 删旧向量与旧块，再全量重建，
  绝不做"增量补丁"式的半更新，避免父子块与向量错位。
* **就地重建**：同一条文重复入库会命中唯一键 `(kb_id, content_sha1)`，因此参数变更时复用原 `doc_id`、
  `version + 1`，而不是插一条新文档。

命令行：

    uv run python -m app.ingest.pipeline --all
    uv run python -m app.ingest.pipeline --kb kb_labor --path "data/raw/中华人民共和国劳动合同法.md"
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

from app.ingest.chunker import build_chunks
from app.ingest.clean import clean_lines
from app.ingest.embedder import DummyEmbedder, Embedder, get_embedder
from app.ingest.loaders import load_document
from app.ingest.models import Chunk, ParsedDocument
from app.ingest.normalize import normalize_text
from app.ingest.structure import law_name_from_filename, parse_law, validate_article_sequence
from app.ingest.versioning import extract_effective_from, normalize_date
from app.ingest.writer import (
    chunk_to_entity,
    delete_chunks,
    delete_document_vectors,
    finish_ingest_job,
    get_document_by_sha1,
    insert_chunks,
    insert_ingest_job,
    parse_config_json,
    refresh_kb_counts,
    update_document_status,
    upsert_document,
    upsert_vectors,
)
from config import BASE_DIR, get_settings

#: 三部法律与知识库的默认对应关系（决策点 Q5：两个库）
DEFAULT_CORPUS: dict[str, list[str]] = {
    "kb_labor": [
        "中华人民共和国劳动合同法.md",
        "中华人民共和国劳动法_中华人民共和国人力资源和社会保障部.md",
    ],
    "kb_civil": ["中华人民共和国民法典.md"],
}

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id() -> str:
    """26 位 ULID（时间有序，便于列表排序与分页）。"""
    ts = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    chars: list[str] = []
    for _ in range(10):
        chars.append(_ALPHABET[ts & 31])
        ts >>= 5
    for _ in range(16):
        chars.append(_ALPHABET[rand & 31])
        rand >>= 5
    return "".join(reversed(chars))


@dataclass
class PreparedIngest:
    """纯计算产物：不碰数据库、不碰向量库，方便单测。"""

    source_file: str
    file_type: str
    file_size: int
    file_sha1: str
    content_sha1: str
    doc: ParsedDocument
    chunks: list[Chunk]
    parse_config: dict
    #: 从正文末条"自 XXXX 年 X 月 X 日起施行"读出；读不到留空
    effective_from: str = ""
    effective_to: str = ""

    @property
    def parents(self) -> list[Chunk]:
        return [c for c in self.chunks if c.chunk_type == "parent"]

    @property
    def children(self) -> list[Chunk]:
        return [c for c in self.chunks if c.chunk_type == "child"]


@dataclass
class IngestResult:
    doc_id: str
    status: str           # created / rebuilt / skipped / failed
    article_count: int
    parent_count: int
    child_count: int
    elapsed_ms: int
    message: str = ""


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    path: str | Path,
    *,
    chunk_size: int = 300,
    overlap: int = 50,
    retrieval_mode: str = "hybrid",
    law_name: str | None = None,
) -> PreparedIngest:
    """解析 + 清洗 + 结构化 + 分块（纯函数，不产生副作用）。"""
    file_path = Path(path)
    loaded = load_document(file_path)

    # 结构识别保留汉字间空格（判断标题 vs 交叉引用的关键信号）
    raw_lines = loaded.text.splitlines()
    lines = clean_lines(raw_lines)
    resolved_law = law_name or law_name_from_filename(file_path.name)
    doc = parse_law(file_path.name, resolved_law, lines)

    if not doc.articles:
        # 非法律文书（PPT/通用文档）：用归一化后的全文做退化分块
        from app.ingest.models import ParsedDocument as PD

        doc = PD(
            source_file=file_path.name,
            law_name=resolved_law,
            text=normalize_text(loaded.text),
            articles=[],
        )

    chunks = build_chunks(
        doc,
        doc_id="PENDING",     # 真实 doc_id 在 ingest 阶段回填
        kb_id="pending",
        chunk_size=chunk_size,
        overlap=overlap,
    )
    content_sha1 = sha1_text("\n".join(item.text for item in lines))
    return PreparedIngest(
        source_file=file_path.name,
        file_type=loaded.file_type,
        file_size=file_path.stat().st_size,
        file_sha1=sha1_file(file_path),
        content_sha1=content_sha1,
        doc=doc,
        chunks=chunks,
        parse_config={
            "chunk_size": chunk_size,
            "overlap": overlap,
            "retrieval_mode": retrieval_mode,
        },
        effective_from=extract_effective_from(doc.text),
        effective_to="",
    )


def rebind_ids(
    prepared: PreparedIngest,
    *,
    doc_id: str,
    kb_id: str,
    effective_from: str | None = None,
    effective_to: str | None = None,
    version: int = 1,
) -> PreparedIngest:
    """把分块里的占位 id 换成真实 doc_id / kb_id，并写入时效/版本。

    时效与版本取决于"是否重建、库里已有值"，prepare 阶段还不确定，
    因此在 ingest 时统一回填到每个块（父块与子块口径一致）。
    """
    starts = prepared.effective_from if effective_from is None else effective_from
    ends = prepared.effective_to if effective_to is None else effective_to
    rebinded: list[Chunk] = []
    for chunk in prepared.chunks:
        new_parent = chunk.parent_id.replace("PENDING", doc_id)
        rebinded.append(
            Chunk(
                chunk_id=chunk.chunk_id.replace("PENDING", doc_id),
                doc_id=doc_id,
                kb_id=kb_id,
                parent_id=new_parent,
                chunk_type=chunk.chunk_type,
                chunk_index=chunk.chunk_index,
                article_no=chunk.article_no,
                chapter_path=chunk.chapter_path,
                title=chunk.title,
                content=chunk.content,
                char_count=chunk.char_count,
                effective_from=starts,
                effective_to=ends,
                version=version,
            )
        )
    return PreparedIngest(
        source_file=prepared.source_file,
        file_type=prepared.file_type,
        file_size=prepared.file_size,
        file_sha1=prepared.file_sha1,
        content_sha1=prepared.content_sha1,
        doc=prepared.doc,
        chunks=rebinded,
        parse_config=prepared.parse_config,
        effective_from=starts,
        effective_to=ends,
    )


def get_document_naming(db, doc_id: str) -> dict | None:
    """取已有文档的展示名与存储路径（上传流程需要沿用，不能用磁盘存储名覆盖）。"""
    row = db.execute(
        text("SELECT filename, storage_path FROM documents WHERE id = :id"),
        {"id": doc_id},
    ).mappings().first()
    return dict(row) if row else None


def ingest(
    prepared: PreparedIngest,
    *,
    kb_id: str,
    doc_id: str | None = None,
    embedder: Embedder | None = None,
    session=None,
    milvus_client=None,
    collection: str | None = None,
    force: bool = False,
) -> IngestResult:
    """落库。``session`` / ``milvus_client`` 可注入以便测试。"""
    settings = get_settings()
    collection = collection or settings.milvus_collection
    started = time.perf_counter()

    from app.db.mysql import session_scope

    owns_session = session is None
    context = session_scope() if owns_session else _null_context(session)
    with context as db:
        existing = get_document_by_sha1(db, kb_id, prepared.content_sha1)
        same_config = (
            existing is not None
            and dict(existing.get("parse_config") or {}) == prepared.parse_config
        )
        if existing and existing["status"] == "ready" and same_config and not force:
            # 重复上传：本次生成的 pending 占位行没有任何块/向量，直接删掉，
            # 否则列表里会永远留一条"排队中"（走查时踩到）。
            if doc_id and doc_id != existing["id"]:
                db.execute(
                    text(
                        "DELETE FROM documents WHERE id = :id AND status = 'pending' "
                        "AND chunk_count = 0"
                    ),
                    {"id": doc_id},
                )
            return IngestResult(
                doc_id=existing["id"],
                status="skipped",
                article_count=int(existing.get("article_count") or 0),
                parent_count=0,
                child_count=int(existing.get("chunk_count") or 0),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                message="内容与解析参数均未变化，跳过",
            )

        # 上传接口会预生成 doc_id（先落一条 pending 行给前端轮询），此处沿用
        doc_id = existing["id"] if existing else (doc_id or new_id())
        rebuild = existing is not None
        version = int(existing.get("version") or 1) + 1 if rebuild else 1

        # 时效：正文能读到时以正文为准；读不到则保留库里已有值（人工维护的
        # 废止日期尤其不能被重建覆盖）。effective_to 目前正文不产出，靠人工补。
        def _existing_date(key: str) -> str:
            return normalize_date(str(existing.get(key) or "")) if existing else ""

        effective_from = prepared.effective_from or _existing_date("effective_from")
        effective_to = prepared.effective_to or _existing_date("effective_to")
        bound = rebind_ids(
            prepared,
            doc_id=doc_id,
            kb_id=kb_id,
            effective_from=effective_from,
            effective_to=effective_to,
            version=version,
        )

        # prepared.source_file 是磁盘上的存储名（上传时带随机前缀），
        # 展示名与存储路径必须沿用库里已有的值：上传文件在 data/uploads/ 下，
        # 若改写成 data/raw/<存储名>，"重新解析"会找不到源文件。
        naming = get_document_naming(db, doc_id) or {}
        upsert_document(
            db,
            {
                "id": doc_id,
                "kb_id": kb_id,
                "filename": naming.get("filename") or prepared.source_file,
                "file_type": prepared.file_type,
                "file_size": prepared.file_size,
                "content_sha1": prepared.content_sha1,
                "file_sha1": prepared.file_sha1,
                "storage_path": naming.get("storage_path") or str(_relative_storage(prepared.source_file)),
                "status": "parsing",
                "progress": 10,
                "parse_config": parse_config_json(**prepared.parse_config),
                "article_count": len(bound.doc.articles),
                "chunk_count": len(bound.children),
                "version": version,
                "effective_from": effective_from or None,
                "effective_to": effective_to or None,
            },
        )
        job_id = new_id()
        insert_ingest_job(db, job_id, doc_id, "parse")

        # 整篇替换：先清旧的块与向量，避免残留
        delete_chunks(db, doc_id)

        update_document_status(db, doc_id, status="chunking", progress=40)
        insert_chunks(db, bound.chunks)

        errors = validate_article_sequence(bound.doc.articles)
        if errors:
            update_document_status(
                db, doc_id, status="failed", error=f"条号缺失：{errors[:10]}"
            )
            finish_ingest_job(db, job_id, status="failed", message="条号缺失")
            raise RuntimeError(f"{prepared.source_file} 条号缺失：{errors[:10]}")

        update_document_status(db, doc_id, status="embedding", progress=60)

        client = milvus_client or _default_milvus_client()
        try:
            delete_document_vectors(client, collection, doc_id)
            embedder = embedder or get_embedder()
            entities = []
            batch_size = settings.embedding_batch_size
            children = bound.children
            for start in range(0, len(children), batch_size):
                batch = children[start : start + batch_size]
                vectors = embedder.encode(
                    [c.embed_text for c in batch], batch_size=batch_size
                )
                for chunk, vector in zip(batch, vectors):
                    entities.append(
                        chunk_to_entity(
                            chunk,
                            vector,
                            law_name=bound.doc.law_name,
                            source_file=prepared.source_file,
                            doc_sha1=prepared.content_sha1,
                        )
                    )
                done = min(start + batch_size, len(children))
                progress = 60 + int(35 * done / max(len(children), 1))
                update_document_status(db, doc_id, status="embedding", progress=progress)
            upsert_vectors(client, collection, entities)
        except Exception as exc:  # noqa: BLE001
            update_document_status(db, doc_id, status="failed", error=str(exc))
            finish_ingest_job(db, job_id, status="failed", message=str(exc))
            raise

        update_document_status(
            db,
            doc_id,
            status="ready",
            progress=100,
            article_count=len(bound.doc.articles),
            chunk_count=len(bound.children),
        )
        finish_ingest_job(
            db, job_id, status="success", processed=len(children), total=len(children)
        )
        refresh_kb_counts(db, kb_id)

    return IngestResult(
        doc_id=doc_id,
        status="rebuilt" if rebuild else "created",
        article_count=len(bound.doc.articles),
        parent_count=len(bound.parents),
        child_count=len(bound.children),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


class _null_context:
    """session 由调用方传入时，不提交也不关闭（由调用方负责事务）。"""

    def __init__(self, session) -> None:
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.session.commit()
        return False


def _relative_storage(filename: str) -> Path:
    return Path("data/raw") / filename


def _default_milvus_client():
    from app.db.milvus_client import get_milvus_db_client

    return get_milvus_db_client()


def ingest_file(
    path: str | Path,
    *,
    kb_id: str,
    chunk_size: int = 300,
    overlap: int = 50,
    retrieval_mode: str = "hybrid",
    embedder: Embedder | None = None,
    force: bool = False,
) -> IngestResult:
    prepared = prepare(
        path,
        chunk_size=chunk_size,
        overlap=overlap,
        retrieval_mode=retrieval_mode,
    )
    return ingest(prepared, kb_id=kb_id, embedder=embedder, force=force)


def ingest_default_corpus(force: bool = False) -> list[IngestResult]:
    results: list[IngestResult] = []
    for kb_id, filenames in DEFAULT_CORPUS.items():
        for filename in filenames:
            path = BASE_DIR / "data" / "raw" / filename
            if not path.exists():
                print(f"[skip] 文件不存在：{path}")
                continue
            print(f"[ingest] {kb_id} <- {filename}")
            result = ingest_file(path, kb_id=kb_id, force=force)
            print(
                f"         {result.status}: 条 {result.article_count} / 父 {result.parent_count} "
                f"/ 子 {result.child_count} / {result.elapsed_ms} ms"
            )
            results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LawAnswer 离线入库")
    parser.add_argument("--all", action="store_true", help="入库 data/raw 下的三部法律")
    parser.add_argument("--kb", help="知识库 id，如 kb_labor")
    parser.add_argument("--path", help="单个文件路径")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--mode", default="hybrid", choices=["dense", "hybrid"])
    parser.add_argument("--force", action="store_true", help="忽略增量判断，强制重建")
    parser.add_argument("--dry-run", action="store_true", help="只解析不写库（校验条数）")
    args = parser.parse_args(argv)

    if args.dry_run:
        targets = (
            [BASE_DIR / "data" / "raw" / name for names in DEFAULT_CORPUS.values() for name in names]
            if args.all
            else [Path(args.path)] if args.path else []
        )
        for target in targets:
            prepared = prepare(target, chunk_size=args.chunk_size, overlap=args.overlap)
            print(
                f"{target.name}: 条 {len(prepared.doc.articles)} / "
                f"父 {len(prepared.parents)} / 子 {len(prepared.children)} / sha1 {prepared.content_sha1[:12]}"
            )
        return 0

    if args.all:
        ingest_default_corpus(force=args.force)
        return 0
    if not (args.kb and args.path):
        parser.error("需要 --all，或同时提供 --kb 与 --path")
    result = ingest_file(
        args.path,
        kb_id=args.kb,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        retrieval_mode=args.mode,
        force=args.force,
    )
    print(
        f"{result.status}: 条 {result.article_count} / 父 {result.parent_count} / "
        f"子 {result.child_count} / {result.elapsed_ms} ms {result.message}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
