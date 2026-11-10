"""阶段 1 真实依赖联调测试（默认跳过）。

    RUN_INTEGRATION=1 uv run python -m pytest tests/test_integration_stage1.py -v -m integration

前置：docker compose up -d 已全部 healthy，且已执行 init_db / init_milvus。
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="需要真实 MySQL/Redis/Milvus，设置 RUN_INTEGRATION=1 启用",
    ),
]


def test_mysql_schema_applied():
    from sqlalchemy import text

    from app.db.mysql import get_engine

    with get_engine().connect() as conn:
        tables = {
            row[0]
            for row in conn.execute(text("SHOW TABLES")).fetchall()
        }
        assert {
            "users",
            "knowledge_bases",
            "chat_sessions",
            "chat_messages",
            "qa_pairs",
            "documents",
            "document_chunks",
            "ingest_jobs",
        } <= tables
        version = conn.execute(text("SELECT VERSION()")).scalar_one()
        assert version.startswith("8.")

        # 中文全文索引必须用 ngram 解析器
        rows = conn.execute(
            text("SHOW INDEX FROM qa_pairs WHERE Key_name = 'ft_search'")
        ).fetchall()
        assert rows, "qa_pairs 缺少 ft_search 全文索引"
        # 中文全文检索依赖 ngram 解析器插件
        status = conn.execute(
            text(
                "SELECT plugin_status FROM information_schema.plugins "
                "WHERE plugin_name = 'ngram'"
            )
        ).scalar()
        assert status == "ACTIVE", f"ngram 插件未启用：{status}"


def test_seed_knowledge_bases():
    from sqlalchemy import text

    from app.db.mysql import get_engine

    with get_engine().connect() as conn:
        ids = {
            row[0]
            for row in conn.execute(text("SELECT id FROM knowledge_bases")).fetchall()
        }
    assert {"kb_labor", "kb_civil"} <= ids


def test_redis_roundtrip():
    from app.db.redis_client import get_redis

    client = get_redis()
    key = "lawanswer:test:stage1"
    try:
        client.set(key, "劳动法", ex=30)
        assert client.get(key) == "劳动法"
    finally:
        client.delete(key)


def test_milvus_collection_ready():
    from app.db.milvus_client import get_milvus_client
    from app.db.milvus_schema import DENSE_DIM, FIELD_SPECS, INDEX_SPECS
    from config import get_settings

    s = get_settings()
    client = get_milvus_client()
    client.using_database(s.milvus.database)
    assert s.milvus.collection in client.list_collections()

    desc = client.describe_collection(s.milvus.collection)
    fields = {f["name"]: f for f in desc["fields"]}
    # 以 schema 定义为准：实际字段必须覆盖全部定义字段（允许未来新增扩展字段）
    assert {spec.name for spec in FIELD_SPECS} <= set(fields)
    assert fields["dense_vector"]["params"]["dim"] == DENSE_DIM
    assert fields["kb_id"].get("is_partition_key") is True

    indexes = client.list_indexes(s.milvus.collection)
    assert set(indexes) >= {spec.field_name for spec in INDEX_SPECS}


def test_health_reports_all_ok():
    from fastapi.testclient import TestClient

    from app.main import app

    resp = TestClient(app).get("/api/health")
    body = resp.json()
    assert body["status"] == "ok", body
    for name, item in body["services"].items():
        assert item["status"] == "ok", f"{name}: {item}"
