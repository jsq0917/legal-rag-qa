"""初始化数据库：建库 + 执行 mysql/init/*.sql（幂等）。

用法::

    uv run python -m app.db.init_db            # 建库 + 建表 + 种子数据
    uv run python -m app.db.init_db --check    # 只检查，不执行 DDL
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymysql

from config import BASE_DIR, get_settings

SQL_DIR = BASE_DIR / "mysql" / "init"
_VALID_DB_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def split_statements(sql: str) -> list[str]:
    """按分号切分 SQL 语句，忽略整行 ``--`` 注释。

    本项目 DDL 中不含存储过程/触发器，故无需处理 ``DELIMITER``；
    字符串字面量里也不存在分号，简单切分即可。
    """
    kept = [ln for ln in sql.splitlines() if not ln.lstrip().startswith("--")]
    statements = (part.strip() for part in "\n".join(kept).split(";"))
    return [s for s in statements if s]


def _connect(database: str | None = None):
    s = get_settings()
    return pymysql.connect(
        host=s.mysql_host,
        port=s.mysql_port,
        user=s.mysql_user,
        password=s.mysql_password,
        database=database,
        charset="utf8mb4",
        autocommit=True,
        connect_timeout=s.mysql_connect_timeout,
    )


def ensure_database() -> str:
    s = get_settings()
    name = s.mysql_db
    if not _VALID_DB_NAME.match(name):
        raise SystemExit(f"MYSQL_DB 非法：{name!r}（只允许字母数字下划线）")
    conn = _connect(database=None)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{name}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
            )
    finally:
        conn.close()
    return name


def apply_files(database: str, verbose: bool = True) -> int:
    files = sorted(SQL_DIR.glob("*.sql"))
    if not files:
        raise SystemExit(f"未找到 SQL 文件：{SQL_DIR}")
    conn = _connect(database=database)
    applied = 0
    try:
        with conn.cursor() as cur:
            for path in files:
                statements = split_statements(path.read_text(encoding="utf-8"))
                if verbose:
                    print(f"[init_db] {path.name}: {len(statements)} 条语句")
                for stmt in statements:
                    cur.execute(stmt)
                    applied += 1
    finally:
        conn.close()
    return applied


def list_tables(database: str) -> list[tuple[str, int]]:
    conn = _connect(database=database)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            tables = [row[0] for row in cur.fetchall()]
            result: list[tuple[str, int]] = []
            for t in tables:
                cur.execute(f"SELECT COUNT(*) FROM `{t}`")
                result.append((t, int(cur.fetchone()[0])))
            return result
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    database = ensure_database()
    if "--check" in argv:
        print(f"[init_db] 数据库 {database} 现有表：")
        for name, count in list_tables(database):
            print(f"  - {name}: {count} 行")
        return 0

    applied = apply_files(database)
    # 旧库补列（新建库在 01_schema.sql 里已包含，迁移会自动跳过）
    from app.db.migrate_versions import ensure_mysql_columns
    from app.db.mysql import get_engine

    added = ensure_mysql_columns(get_engine(get_settings()))
    print(f"[init_db] 完成，共执行 {applied} 条语句；版本字段新增 {added or '无'}；数据库 {database}")
    for name, count in list_tables(database):
        print(f"  - {name}: {count} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
