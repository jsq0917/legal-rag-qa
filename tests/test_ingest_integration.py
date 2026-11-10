"""阶段 2 联调测试：真实入库结果、Milvus 向量数、块接口、编辑/删除同步。

    RUN_INTEGRATION=1 uv run python -m pytest tests/test_ingest_integration.py -v -m integration

前置：`uv run python -m app.ingest.pipeline --all` 已执行成功。
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="需要真实 MySQL/Milvus，设置 RUN_INTEGRATION=1 启用",
    ),
]

EXPECTED = {
    "中华人民共和国劳动合同法.md": (98, "kb_labor"),
    "中华人民共和国劳动法_中华人民共和国人力资源和社会保障部.md": (107, "kb_labor"),
    "中华人民共和国民法典.md": (1260, "kb_civil"),
}


@pytest.fixture(scope="module")
def db():
    from app.db.mysql import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        yield conn


@pytest.fixture(scope="module")
def client():
    from app.main import app

    return TestClient(app)


class TestIngestedDocuments:
    def test_all_three_documents_ready(self, db):
        """基线三份法律文档必须 ready；允许后续追加其他语料。"""
        rows = db.execute(
            text("SELECT filename, status, article_count, chunk_count FROM documents")
        ).fetchall()
        by_name = {r[0]: r for r in rows}
        missing = set(EXPECTED) - set(by_name)
        assert not missing, f"缺少基线文档：{sorted(missing)}"
        for filename, (expected, _kb_id) in EXPECTED.items():
            _name, status, articles, chunks = by_name[filename]
            assert status == "ready", f"{filename} 状态={status}"
            assert articles == expected, f"{filename} 条数={articles}，期望 {expected}"
            assert chunks > 0

    def test_article_counts_by_law(self, db):
        for filename, (expected, kb_id) in EXPECTED.items():
            got = db.execute(
                text("SELECT article_count FROM documents WHERE filename = :f AND kb_id = :k"),
                {"f": filename, "k": kb_id},
            ).scalar()
            assert got == expected

    def test_parent_chunks_equal_articles(self, db):
        """法律库每份文档的父块数应等于其条文数（新增语料后按库汇总校验）。"""
        rows = db.execute(
            text(
                "SELECT d.id, d.article_count, "
                "  (SELECT COUNT(*) FROM document_chunks c "
                "   WHERE c.doc_id = d.id AND c.chunk_type = 'parent') AS parents "
                "FROM documents d "
                "WHERE d.kb_id IN ('kb_labor', 'kb_civil') AND d.status = 'ready'"
            )
        ).fetchall()
        assert rows, "法律库应至少有一份 ready 文档"
        for doc_id, articles, parents in rows:
            assert parents == articles, f"{doc_id} 父块={parents}，条文={articles}"

    def test_chunk_types_and_traceability(self, db):
        row = db.execute(
            text(
                "SELECT article_no, chapter_path, title, parent_id, chunk_index "
                "FROM document_chunks WHERE chunk_type='child' LIMIT 1"
            )
        ).fetchone()
        article_no, chapter_path, title, parent_id, chunk_index = row
        assert article_no
        assert chapter_path
        assert "中华人民共和国" in title
        assert parent_id and chunk_index is not None

    def test_kb_counts_refreshed(self, db):
        rows = dict(
            db.execute(text("SELECT id, chunk_count FROM knowledge_bases")).fetchall()
        )
        assert rows["kb_labor"] > 0
        assert rows["kb_civil"] > 0

    def test_vectors_written_to_milvus(self, db):
        from app.db.milvus_client import get_milvus_client
        from config import get_settings

        settings = get_settings()
        client = get_milvus_client()
        client.using_database(settings.milvus.database)
        children = db.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE chunk_type='child'")
        ).scalar()
        result = client.query(
            collection_name=settings.milvus_collection,
            filter="chunk_id != \"\"",
            output_fields=["count(*)"],
        )
        assert int(result[0]["count(*)"]) == children

    def test_unique_vector_ids(self, db):
        from app.db.milvus_client import get_milvus_client
        from config import get_settings

        settings = get_settings()
        client = get_milvus_client()
        client.using_database(settings.milvus.database)
        doc_ids = {
            r[0]
            for r in db.execute(
                text("SELECT id FROM documents WHERE kb_id = 'kb_labor'")
            ).fetchall()
        }
        rows = client.query(
            collection_name=settings.milvus_collection,
            filter='kb_id == "kb_labor"',
            output_fields=["chunk_id", "article_no", "law_name"],
            limit=20,
        )
        assert rows
        # 向量主键唯一，且都能回溯到 kb_labor 下的真实文档
        assert len({r["chunk_id"] for r in rows}) == len(rows)
        assert all(r["chunk_id"].split(":")[0] in doc_ids for r in rows)
        assert all(r["law_name"] for r in rows)

    def test_incremental_skip(self):
        """重复执行同一文件应返回 skipped，而不是重复写入。"""
        from app.ingest.pipeline import ingest_file
        from config import BASE_DIR

        path = BASE_DIR / "data" / "raw" / "中华人民共和国劳动合同法.md"
        result = ingest_file(path, kb_id="kb_labor")
        assert result.status == "skipped"

    def test_chunks_unchanged_after_skip(self):
        """跳过重复入库时，块总数必须保持不变（不因语料扩充而失效）。"""
        from app.ingest.pipeline import ingest_file
        from app.db.mysql import get_engine
        from config import BASE_DIR

        def count_chunks() -> int:
            with get_engine().connect() as conn:
                return conn.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar()

        before = count_chunks()
        result = ingest_file(
            BASE_DIR / "data" / "raw" / "中华人民共和国劳动合同法.md", kb_id="kb_labor"
        )
        assert result.status == "skipped"
        assert count_chunks() == before


class TestChunkApi:
    @pytest.fixture(autouse=True)
    def _auth(self, integration_auth_headers):
        self.headers = integration_auth_headers

    def test_list_documents(self, client):
        resp = client.get("/api/kbs/kb_labor/documents", headers=self.headers)
        assert resp.status_code == 200
        names = {d["filename"] for d in resp.json()}
        assert {
            "中华人民共和国劳动合同法.md",
            "中华人民共和国劳动法_中华人民共和国人力资源和社会保障部.md",
        } <= names

    def test_list_chunks_pagination(self, client, db):
        doc_id = db.execute(
            text("SELECT id FROM documents WHERE filename LIKE '中华人民共和国劳动合同法%'")
        ).scalar()
        resp = client.get(
            f"/api/documents/{doc_id}/chunks?limit=5&offset=0", headers=self.headers
        )
        body = resp.json()
        assert resp.status_code == 200
        assert body["total"] > 0
        assert len(body["items"]) == 5

    def test_filter_children_only(self, client, db):
        doc_id = db.execute(
            text("SELECT id FROM documents WHERE filename LIKE '中华人民共和国民法典%'")
        ).scalar()
        body = client.get(
            f"/api/documents/{doc_id}/chunks?chunk_type=child&limit=3", headers=self.headers
        ).json()
        assert all(item["chunk_type"] == "child" for item in body["items"])

    def test_search_by_article_no(self, client, db):
        doc_id = db.execute(
            text("SELECT id FROM documents WHERE filename LIKE '中华人民共和国劳动合同法%'")
        ).scalar()
        body = client.get(
            f"/api/documents/{doc_id}/chunks?q=第十九条", headers=self.headers
        ).json()
        assert body["total"] >= 1
        assert any("第十九条" in (i["article_no"] or "") for i in body["items"])

    def test_unknown_kb_404(self, client):
        assert (
            client.get("/api/kbs/nope/documents", headers=self.headers).status_code == 404
        )

    def test_unknown_chunk_404(self, client):
        assert (
            client.delete("/api/chunks/not-exist", headers=self.headers).status_code == 404
        )
