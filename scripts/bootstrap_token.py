"""为一键启动签发本地 bootstrap JWT（仅本机开发用，不改动鉴权规则）。

背景：``/api/kbs`` 与 ``/api/ingest/local`` 接入强制 JWT 后都需要 Bearer token，
而 ``scripts/bootstrap.ps1`` 是脚本化建库/入库，没有交互式登录（登录接口还带验证码）。
这里直接在本地库里挑一个 admin/editor 账号签发 token；库里没有可用账号时补一个
``bootstrap_local`` 管理员，保证全新环境也能跑通。

用法::

    uv run python -m scripts.bootstrap_token
"""

from __future__ import annotations

import secrets

from sqlalchemy import text

from app.core.security import create_access_token, hash_password
from app.db.mysql import get_engine

BOOTSTRAP_USERNAME = "bootstrap_local"


def main() -> int:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT id, username FROM users "
                "WHERE status = 1 AND role IN ('admin', 'editor') "
                "ORDER BY (role = 'admin') DESC, id ASC LIMIT 1"
            )
        ).mappings().first()
        if row is None:
            conn.execute(
                text(
                    "INSERT INTO users (username, password_hash, tenant_id, role, status) "
                    "VALUES (:username, :password_hash, 'default', 'admin', 1)"
                ),
                {
                    "username": BOOTSTRAP_USERNAME,
                    "password_hash": hash_password(secrets.token_urlsafe(18)),
                },
            )
            row = conn.execute(
                text("SELECT id, username FROM users WHERE username = :username"),
                {"username": BOOTSTRAP_USERNAME},
            ).mappings().first()
    print(create_access_token(int(row["id"]), str(row["username"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
