"""pytest 公共夹具。

原则：单元测试**全离线** —— 不加载真实模型、不调 DashScope、不连 MySQL/Milvus。
需要真实依赖的用例标记 ``@pytest.mark.integration``，默认跳过。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logger import shutdown_logging  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_logging():
    """每个用例结束后移除本模块安装的 handler，避免用例之间互相污染。"""
    yield
    shutdown_logging()


@pytest.fixture(autouse=True)
def _offline_redis(request, monkeypatch):
    """单测全离线：非 integration 用例禁用真实 Redis。

    把 ``get_redis`` 替换为抛错实现，走各调用点的 fail-open 分支
    （意图缓存跳过、限流放行、QARetriever 降级 _NullRedis）。
    不这样做的话，意图缓存/限流会把测试 fixture 数据写进本机真实 Redis
    （db 3 是生产库），既污染测试又污染线上缓存。
    需要真实 Redis 的用例请标 ``@pytest.mark.integration``。
    """
    if request.node.get_closest_marker("integration"):
        yield
        return

    import app.db.redis_client as redis_mod

    def _boom(*_args, **_kw):
        raise ConnectionError("offline test: real redis disabled")

    monkeypatch.setattr(redis_mod, "get_redis", _boom)
    yield


@pytest.fixture
def client():
    """不带 lifespan 的 TestClient：不起日志、不建目录、不连依赖。"""
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def integration_auth_headers():
    """联调接口的登录态：临时造一个 default 租户 editor 用户并签发 JWT。

    阶段 4 接入 JWT 后，``/api/kbs``、``/api/documents``、``/api/retrieval``
    等资源接口都要求登录。联调用例直接落库造用户，跳过验证码/注册流程，
    模块结束后删除该用户。
    """
    import uuid

    from sqlalchemy import text

    from app.core.security import create_access_token, hash_password
    from app.db.mysql import get_engine

    username = f"it_{uuid.uuid4().hex[:10]}"
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO users (username, password_hash, tenant_id, role) "
                "VALUES (:u, :p, 'default', 'editor')"
            ),
            {"u": username, "p": hash_password("secret123")},
        )
        user_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
    token = create_access_token(user_id, username)
    yield {"Authorization": f"Bearer {token}"}
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})


@pytest.fixture
def settings_factory():
    """构造不读真实 .env 的 Settings（避免测试受本机配置影响）。"""
    from config import Settings

    def _make(**kwargs):
        kwargs.setdefault("_env_file", None)
        return Settings(**kwargs)

    return _make
