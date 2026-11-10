"""幂等迁移：给法条加时效/版本字段（MySQL + Milvus）。

用法::

    python -m app.db.migrate_law_effective
    python -m app.db.migrate_law_effective --rebuild-milvus

* MySQL：information_schema 判重后 ADD COLUMN，可反复执行；
* 回填：从每篇文档最后一条父块正文读"自 XXXX 年 X 月 X 日起施行"，
  只填空值，不覆盖人工维护的日期；
* Milvus：2.4 不支持给已有 collection 加字段，缺字段时必须重建。
  默认只报告并以退出码 2 结束；加 ``--rebuild-milvus`` 才会 drop +
  重新灌入 ``DEFAULT_CORPUS``（数据可由 data/raw 完整重建）。
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import text

from app.ingest.versioning import extract_effective_from
from config import get_settings

_DOCUMENT_COLUMNS = (
    ("effective_from", "DATE NULL COMMENT '施行日期，空=未知'"),
    ("effective_to", "DATE NULL COMMENT '失效日期，空=长期有效'"),
)
_CHUNK_COLUMNS = (
    ("effective_from", "DATE NULL COMMENT '施行日期，空=未知'"),
    ("effective_to", "DATE NULL COMMENT '失效日期，空=长期有效'"),
    ("version", "INT NOT NULL DEFAULT 1 COMMENT '文档入库版本'"),
)


def _column_exists(conn, table: str, column: str) -> bool:
    row = conn.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = :table "
            "AND column_name = :column"
        ),
        {"table": table, "column": column},
    ).first()
    return row is not None


def ensure_mysql_columns(engine) -> list[str]:
    """补齐 MySQL 字段，返回本次新增的 ``表.字段``。"""
    added: list[str] = []
    with engine.begin() as conn:
        for table, columns in (("documents", _DOCUMENT_COLUMNS), ("document_chunks", _CHUNK_COLUMNS)):
            for column, ddl in columns:
                if _column_exists(conn, table, column):
                    continue
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                added.append(f"{table}.{column}")
    return added


def backfill_dates(engine) -> tuple[int, int]:
    """从正文回填 documents.effective_from，并同步到 document_chunks。

    返回 ``(更新文档数, 同步块数)``。已有人工值的行不覆盖。
    """
    updated_docs = 0
    with engine.begin() as conn:
        docs = conn.execute(text("SELECT id, effective_from FROM documents")).fetchall()
        for doc_id, current in docs:
            if current is not None:
                continue
            row = conn.execute(
                text(
                    "SELECT content FROM document_chunks "
                    "WHERE doc_id = :doc_id AND chunk_type = 'parent' AND is_enabled = 1 "
                    "ORDER BY chunk_index DESC LIMIT 1"
                ),
                {"doc_id": doc_id},
            ).first()
            if not row:
                continue
            found = extract_effective_from(row[0] or "")
            if not found:
                continue
            conn.execute(
                text("UPDATE documents SET effective_from = :d WHERE id = :id"),
                {"d": found, "id": doc_id},
            )
            updated_docs += 1
        synced = conn.execute(
            text(
                "UPDATE document_chunks c JOIN documents d ON d.id = c.doc_id "
                "SET c.effective_from = d.effective_from, "
                "    c.effective_to = d.effective_to, "
                "    c.version = d.version"
            )
        ).rowcount or 0
    return updated_docs, int(synced)


def milvus_missing_fields(client, collection: str) -> list[str]:
    desc = client.describe_collection(collection_name=collection)
    names = {field.get("name") for field in (desc.get("fields") or [])}
    return [name for name in ("effective_from", "effective_to", "version") if name not in names]


def rebuild_milvus(settings) -> None:
    """drop 旧 collection → 按新 schema 重建 → 用 data/raw 重灌默认语料。"""
    from app.db.milvus_client import get_milvus_db_client
    from app.db.milvus_schema import ensure_collection
    from app.ingest.pipeline import ingest_default_corpus

    client = get_milvus_db_client(settings)
    ensure_collection(
        client, settings.milvus_db, settings.milvus_collection, drop=True
    )
    results = ingest_default_corpus(force=True)
    failed = [r for r in results if r.status == "failed"]
    if failed:
        raise RuntimeError(f"重灌失败 {len(failed)} 篇：{[r.doc_id for r in failed]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="法条时效字段幂等迁移")
    parser.add_argument(
        "--rebuild-milvus",
        action="store_true",
        help="Milvus 缺字段时重建 collection 并重灌默认语料（会覆盖向量索引）",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    from app.db.mysql import get_engine

    engine = get_engine(settings)
    added = ensure_mysql_columns(engine)
    updated_docs, synced_chunks = backfill_dates(engine)
    print(f"[mysql] 新增字段 {added or '无'}")
    print(f"[mysql] 回填文档 {updated_docs} 篇，同步块 {synced_chunks} 条")

    from app.db.milvus_client import get_milvus_db_client

    client = get_milvus_db_client(settings)
    missing = milvus_missing_fields(client, settings.milvus_collection)
    if not missing:
        print("[milvus] 时效字段已存在，无需重建")
        return 0
    print(f"[milvus] 缺少字段：{missing}")
    if not args.rebuild_milvus:
        print("[milvus] 未指定 --rebuild-milvus，保留现有索引；检索将走 MySQL 兜底过滤")
        return 2
    print("[milvus] 开始重建 collection 并重灌默认语料……")
    rebuild_milvus(settings)
    print("[milvus] 重建完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
