"""幂等迁移：租户 / 角色 / 知识库归属字段（LA-O2）。

* ``users`` 增加 ``tenant_id``、``role``（默认 editor，可只读/管理员调整）
* ``knowledge_bases`` 增加 ``tenant_id``、``owner_id``、``is_public``
* 历史知识库默认视为公共库（``is_public=1``），与改造前"所有登录用户可见"一致

用法::

    python -m app.db.migrate_access
"""

from __future__ import annotations

import sys

from sqlalchemy import text

_USER_COLUMNS = (
    ("tenant_id", "VARCHAR(32) NOT NULL DEFAULT 'default' COMMENT '租户隔离键'"),
    (
        "role",
        "ENUM('admin','editor','viewer') NOT NULL DEFAULT 'editor' "
        "COMMENT 'admin 管理 / editor 编辑 / viewer 只读'",
    ),
)
_KB_COLUMNS = (
    ("tenant_id", "VARCHAR(32) NOT NULL DEFAULT 'default' COMMENT '归属租户'"),
    ("owner_id", "BIGINT UNSIGNED NULL COMMENT '创建者；种子库为 NULL'"),
    ("is_public", "TINYINT(1) NOT NULL DEFAULT 1 COMMENT '1 公共库（跨租户可读）'"),
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


def _index_exists(conn, table: str, index: str) -> bool:
    row = conn.execute(
        text(
            "SELECT 1 FROM information_schema.statistics "
            "WHERE table_schema = DATABASE() AND table_name = :table AND index_name = :index"
        ),
        {"table": table, "index": index},
    ).first()
    return row is not None


def ensure_access_columns(engine) -> list[str]:
    added: list[str] = []
    with engine.begin() as conn:
        for table, columns in (("users", _USER_COLUMNS), ("knowledge_bases", _KB_COLUMNS)):
            for column, ddl in columns:
                if _column_exists(conn, table, column):
                    continue
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                added.append(f"{table}.{column}")
        if not _index_exists(conn, "knowledge_bases", "idx_tenant_public"):
            conn.execute(text("ALTER TABLE knowledge_bases ADD KEY idx_tenant_public (tenant_id, is_public)"))
            added.append("knowledge_bases.idx_tenant_public")
        # 历史数据明确标记为公共库，避免迁移后突然"谁都看不见"
        conn.execute(
            text("UPDATE knowledge_bases SET is_public = 1 WHERE is_public IS NULL")
        )
        conn.execute(
            text("UPDATE users SET role = 'editor' WHERE role IS NULL")
        )
    return added


def main(argv: list[str] | None = None) -> int:
    from app.db.mysql import get_engine
    from config import get_settings

    added = ensure_access_columns(get_engine(get_settings()))
    print(f"[migrate_access] 新增字段/索引 {added or '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
