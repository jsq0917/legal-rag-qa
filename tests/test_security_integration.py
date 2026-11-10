"""LA-P1 真实服务安全集成测试：MySQL 租户授权 + Redis 缓存作用域。

    RUN_INTEGRATION=1 uv run python -m pytest tests/test_security_integration.py -v -m integration

只依赖 MySQL 与 Redis，不加载模型、不连 Milvus，适合作为 CI 集成层的最小
必跑集。删除/重解析与 SSE 的真实依赖用例分别在
``test_upload_flow_integration.py`` / ``test_generation_integration.py``。
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="需要真实 MySQL + Redis，设置 RUN_INTEGRATION=1 启用",
    ),
]


@pytest.fixture(scope="module")
def db():
    from app.db.mysql import get_engine

    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - 依赖未启动时明确 skip，不伪装通过
        pytest.skip(f"MySQL 不可用: {type(exc).__name__}: {exc}")
    return engine


@pytest.fixture(scope="module")
def redis_client():
    from app.db.redis_client import get_redis

    try:
        client = get_redis()
        client.ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Redis 不可用: {type(exc).__name__}: {exc}")
    return client


@pytest.fixture
def temp_kbs(db):
    """临时三库：公共库、同租户私库、其他租户私库；用后删除。"""
    suffix = uuid.uuid4().hex[:8]
    ids = {
        "public": f"kb_it_pub_{suffix}",
        "tenant_a": f"kb_it_a_{suffix}",
        "tenant_b": f"kb_it_b_{suffix}",
    }
    rows = [
        (ids["public"], f"IT公共库{suffix}", 1, "tenant-b", 3, 5),
        (ids["tenant_a"], f"IT租户A私库{suffix}", 0, "tenant-a", 7, 9),
        (ids["tenant_b"], f"IT租户B私库{suffix}", 0, "tenant-b", 11, 13),
    ]
    with db.begin() as conn:
        for kb_id, name, is_public, tenant_id, corpus_version, index_version in rows:
            conn.execute(
                text(
                    "INSERT INTO knowledge_bases "
                    "(id, name, description, embedding_model, tenant_id, is_public, "
                    " doc_count, chunk_count, corpus_version, index_version) "
                    "VALUES (:id, :name, 'integration test', 'bge-m3', :tenant, :pub, 0, 0, :cv, :iv)"
                ),
                {
                    "id": kb_id,
                    "name": name,
                    "tenant": tenant_id,
                    "pub": is_public,
                    "cv": corpus_version,
                    "iv": index_version,
                },
            )
    try:
        yield ids
    finally:
        with db.begin() as conn:
            conn.execute(
                text("DELETE FROM knowledge_bases WHERE id IN (:a, :b, :c)"),
                {"a": ids["public"], "b": ids["tenant_a"], "c": ids["tenant_b"]},
            )


def test_tenant_scope_from_real_mysql(db, temp_kbs):
    """公共库跨租户可见；私库只对同租户可见；越权 kb_id 被收敛为空。"""
    from app.core.access import resolve_access_scope

    user_a = SimpleNamespace(id=1, tenant_id="tenant-a")
    with db.connect() as conn:
        scope_a = resolve_access_scope(conn, user_a, [temp_kbs["tenant_b"]])
        scope_all = resolve_access_scope(conn, user_a)

    assert temp_kbs["public"] in scope_all.kb_ids
    assert temp_kbs["tenant_a"] in scope_all.kb_ids
    assert temp_kbs["tenant_b"] not in scope_all.kb_ids
    assert scope_a.restrict([temp_kbs["tenant_b"]]) == []
    assert scope_a.denied_kb_ids == (temp_kbs["tenant_b"],)

    with db.connect() as conn:
        grants = {g.kb_id: g for g in resolve_access_scope(conn, user_a).grants}
    assert grants[temp_kbs["tenant_a"]].corpus_version == 7
    assert grants[temp_kbs["tenant_a"]].index_version == 9


def test_cache_identity_isolates_kb_and_version(db, redis_client, temp_kbs):
    """真实 Redis：跨知识库/跨版本身份不能互相命中，同身份可命中。"""
    from app.core.access import AccessScope, KbGrant, answer_identity
    from app.orchestration.answer_cache import AnswerCache, answer_key, normalize_question
    from config import Settings

    scope_a = AccessScope.for_grants(
        [KbGrant(kb_id=temp_kbs["tenant_a"], corpus_version=7, index_version=9, is_public=False, tenant_id="tenant-a")],
        tenant_id="tenant-a",
        user_id=1,
    )
    scope_b = AccessScope.for_grants(
        [KbGrant(kb_id=temp_kbs["tenant_b"], corpus_version=11, index_version=13, is_public=False, tenant_id="tenant-b")],
        tenant_id="tenant-b",
        user_id=2,
    )
    scope_a_new_version = AccessScope.for_grants(
        [KbGrant(kb_id=temp_kbs["tenant_a"], corpus_version=8, index_version=10, is_public=False, tenant_id="tenant-a")],
        tenant_id="tenant-a",
        user_id=1,
    )
    query = f"集成测试缓存作用域隔离{uuid.uuid4().hex[:6]}的标准答案"
    identities = {
        "a": answer_identity(scope_a, model="m", prompt_version="p"),
        "b": answer_identity(scope_b, model="m", prompt_version="p"),
        "a2": answer_identity(scope_a_new_version, model="m", prompt_version="p"),
    }
    assert len(set(identities.values())) == 3

    cache = AnswerCache(redis=redis_client, settings=Settings(_env_file=None, cache_answer_min_chars=0))
    keys = [answer_key(normalize_question(query), identity) for identity in identities.values()]
    try:
        assert cache.store(
            query,
            answer="集成测试答案：根据《中华人民共和国劳动合同法》第十九条，试用期不得超过六个月。",
            cache_identity=identities["a"],
        )
        assert cache.lookup(query, cache_identity=identities["a"]) is not None
        assert cache.lookup(query, cache_identity=identities["b"]) is None
        assert cache.lookup(query, cache_identity=identities["a2"]) is None
    finally:
        redis_client.delete(*keys)
