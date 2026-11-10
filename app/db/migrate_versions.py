"""幂等迁移：知识库语料/索引版本字段（LA-O1 / LA-O3 基础）。

用法::

    python -m app.db.migrate_versions
    uv run python -m app.db.init_db   # 初始化时会自动调用本迁移

只做 ``ADD COLUMN``，通过 information_schema 判重，可反复执行。
"""

from __future__ import annotations

import sys

from sqlalchemy import text

_KB_COLUMNS = (
    ("corpus_version", "BIGINT UNSIGNED NOT NULL DEFAULT 1 COMMENT '语料版本：增删改/重建即 +1'"),
    ("index_version", "BIGINT UNSIGNED NOT NULL DEFAULT 1 COMMENT '索引版本：向量索引重建即 +1'"),
)
_DOC_COLUMNS = (
    ("content_hash", "CHAR(64) NULL COMMENT '归一化正文 SHA-256'"),
    ("index_version", "INT NOT NULL DEFAULT 1 COMMENT '入索引时的知识库 index_version 快照'"),
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
    """补齐版本字段，返回本次新增的 ``表.字段``。"""
    added: list[str] = []
    with engine.begin() as conn:
        for table, columns in (("knowledge_bases", _KB_COLUMNS), ("documents", _DOC_COLUMNS)):
            for column, ddl in columns:
                if _column_exists(conn, table, column):
                    continue
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                added.append(f"{table}.{column}")
    return added


def main(argv: list[str] | None = None) -> int:
    from app.db.mysql import get_engine
    from config import get_settings

    engine = get_engine(get_settings())
    added = ensure_mysql_columns(engine)
    print(f"[migrate_versions] 新增字段 {added or '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
