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


def _doc_text(articles: list[tuple[str, str]]) -> str:
    lines = ["# 版本一致性测试法", ""]
    for article_no, body in articles:
        lines.extend([f"{article_no} {body}", ""])
    return "\n".join(lines)


def _milvus_rows_client():
    from app.db.milvus_client import get_milvus_client
    from config import get_settings

    settings = get_settings()
    client = get_milvus_client()
    client.using_database(settings.milvus.database)
    return client, settings.milvus_collection


def _milvus_count(doc_id: str) -> int:
    client, collection = _milvus_rows_client()
    rows = client.query(
        collection_name=collection,
        filter=f'doc_id == "{doc_id}"',
        output_fields=["count(*)"],
    )
    return int(rows[0]["count(*)"] or 0) if rows else 0


def _wait_until(predicate, timeout_s: float = 30.0, interval_s: float = 0.5) -> bool:
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


class TestVersionConsistency:
    """LA-O3：CAS 并发保护、重建、删除、编辑的端到端行为（真实 MySQL + Milvus）。"""

    def _ingest(self, tmp_path, filename: str, articles, *, doc_id: str | None = None,
                force: bool = False):
        from app.ingest.pipeline import ingest, prepare

        path = tmp_path / filename
        path.write_text(_doc_text(articles), encoding="utf-8")
        prepared = prepare(
            path, chunk_size=300, overlap=50, retrieval_mode="hybrid", profile="legal"
        )
        return ingest(prepared, kb_id="kb_labor", doc_id=doc_id, force=force)

    def _cleanup(self, doc_id: str | None) -> None:
        if not doc_id:
            return
        from app.db.mysql import session_scope
        from app.ingest.service import delete_document

        try:
            with session_scope() as db:
                delete_document(db, doc_id)
        except LookupError:
            pass

    def _version(self, doc_id: str) -> int:
        from app.db.mysql import session_scope

        with session_scope() as db:
            return int(
                db.execute(
                    text("SELECT version FROM documents WHERE id = :id"), {"id": doc_id}
                ).scalar()
                or 0
            )

    def _chunk_counts(self, doc_id: str) -> dict[str, int]:
        from app.db.mysql import session_scope

        with session_scope() as db:
            rows = db.execute(
                text(
                    "SELECT chunk_type, COUNT(*) FROM document_chunks "
                    "WHERE doc_id = :id GROUP BY chunk_type"
                ),
                {"id": doc_id},
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def test_rebuild_replaces_chunks_and_vectors(self, tmp_path):
        """同名内容变化：复用 doc_id、版本 +1、旧块/旧向量被替换而非追加。"""
        doc_id = None
        filename = f"o3-rebuild-{os.urandom(4).hex()}.md"
        try:
            first = self._ingest(
                tmp_path,
                filename,
                [("第一条", "第一版内容，用于验证重建。"), ("第二条", "第一版第二条。")],
            )
            doc_id = first.doc_id
            assert first.status == "created"
            assert first.article_count == 2
            v1 = self._version(doc_id)
            chunks_before = self._chunk_counts(doc_id)
            assert chunks_before.get("parent") == 2
            child_before = chunks_before["child"]
            assert child_before > 0
            assert _wait_until(lambda: _milvus_count(doc_id) == child_before)

            second = self._ingest(
                tmp_path,
                filename,
                [
                    ("第一条", "第二版内容，已修改。"),
                    ("第二条", "第二版第二条。"),
                    ("第三条", "第二版新增第三条。"),
                ],
                doc_id=doc_id,
            )
            assert second.status == "rebuilt"
            assert second.doc_id == doc_id
            assert second.article_count == 3
            assert self._version(doc_id) == v1 + 1

            chunks_after = self._chunk_counts(doc_id)
            assert chunks_after.get("parent") == 3
            # 旧版本块已被删除，不会出现 2+3 叠加
            assert chunks_after["child"] == second.child_count
            assert _wait_until(lambda: _milvus_count(doc_id) == second.child_count)

            from app.db.mysql import session_scope

            with session_scope() as db:
                contents = [
                    row[0]
                    for row in db.execute(
                        text(
                            "SELECT content FROM document_chunks WHERE doc_id = :id "
                            "AND chunk_type = 'child'"
                        ),
                        {"id": doc_id},
                    ).fetchall()
                ]
            assert all("第一版" not in text_value for text_value in contents)
        finally:
            self._cleanup(doc_id)

    def test_delete_document_removes_rows_chunks_and_vectors(self, tmp_path):
        from app.db.mysql import session_scope
        from app.ingest.service import delete_document

        result = self._ingest(
            tmp_path,
            f"o3-delete-{os.urandom(4).hex()}.md",
            [("第一条", "待删除文档的内容。"), ("第二条", "待删除文档的第二条。")],
        )
        doc_id = result.doc_id
        child_count = self._chunk_counts(doc_id)["child"]
        assert _wait_until(lambda: _milvus_count(doc_id) == child_count)

        with session_scope() as db:
            removed = delete_document(db, doc_id)
        assert removed["kb_id"] == "kb_labor"

        with session_scope() as db:
            assert (
                db.execute(
                    text("SELECT COUNT(*) FROM documents WHERE id = :id"), {"id": doc_id}
                ).scalar()
                == 0
            )
            assert (
                db.execute(
                    text("SELECT COUNT(*) FROM document_chunks WHERE doc_id = :id"),
                    {"id": doc_id},
                ).scalar()
                == 0
            )
        assert _wait_until(lambda: _milvus_count(doc_id) == 0)

    def test_edit_and_disable_chunk_roundtrip(self, tmp_path):
        from app.db.mysql import session_scope
        from app.ingest.service import disable_chunk, update_chunk_content

        result = self._ingest(
            tmp_path,
            f"o3-edit-{os.urandom(4).hex()}.md",
            [("第一条", "编辑前的原始条文内容。")],
        )
        doc_id = result.doc_id
        try:
            with session_scope() as db:
                chunk_id = db.execute(
                    text(
                        "SELECT id FROM document_chunks WHERE doc_id = :id "
                        "AND chunk_type = 'child' LIMIT 1"
                    ),
                    {"id": doc_id},
                ).scalar()
            assert chunk_id

            new_content = "编辑后的条文内容，包含唯一标记 EDIT-O3-TOKEN。"
            with session_scope() as db:
                update_chunk_content(db, chunk_id, new_content)

            with session_scope() as db:
                row = db.execute(
                    text(
                        "SELECT content, char_count, in_milvus FROM document_chunks "
                        "WHERE id = :cid"
                    ),
                    {"cid": chunk_id},
                ).mappings().first()
            assert row["content"] == new_content
            assert row["char_count"] == len(new_content)
            assert int(row["in_milvus"]) == 1

            client, collection = _milvus_rows_client()

            def milvus_content() -> list[dict]:
                return client.query(
                    collection_name=collection,
                    filter=f'chunk_id == "{chunk_id}"',
                    output_fields=["content"],
                )

            assert _wait_until(
                lambda: any(r.get("content") == new_content for r in milvus_content())
            )

            with session_scope() as db:
                disable_chunk(db, chunk_id)
            with session_scope() as db:
                row = db.execute(
                    text(
                        "SELECT is_enabled, in_milvus FROM document_chunks WHERE id = :cid"
                    ),
                    {"cid": chunk_id},
                ).mappings().first()
            assert int(row["is_enabled"]) == 0
            assert int(row["in_milvus"]) == 0
            assert _wait_until(lambda: milvus_content() == [])
        finally:
            self._cleanup(doc_id)

    def test_stale_cas_cannot_overwrite_newer_version(self, tmp_path):
        """旧任务用过期 version 收尾必须 0 行受影响，不得覆盖新任务的块数与状态。"""
        from app.db.mysql import session_scope
        from app.ingest.writer import update_document_status

        result = self._ingest(
            tmp_path,
            f"o3-cas-{os.urandom(4).hex()}.md",
            [("第一条", "CAS 测试条文。")],
        )
        doc_id = result.doc_id
        try:
            with session_scope() as db:
                stale_version = int(
                    db.execute(
                        text("SELECT version FROM documents WHERE id = :id"), {"id": doc_id}
                    ).scalar()
                )
                claim_sql = text(
                    "UPDATE documents SET version = version + 1, status = 'parsing', "
                    "progress = 10, error_msg = NULL WHERE id = :id AND version = :version"
                )
                # 新任务先抢占成功
                assert db.execute(claim_sql, {"id": doc_id, "version": stale_version}).rowcount == 1
                # 旧任务拿着同一个 base_version 再抢，必须失败
                assert db.execute(claim_sql, {"id": doc_id, "version": stale_version}).rowcount == 0

                current = db.execute(
                    text("SELECT version, chunk_count FROM documents WHERE id = :id"),
                    {"id": doc_id},
                ).mappings().first()
                real_chunk_count = int(current["chunk_count"] or 0)
                assert int(current["version"]) == stale_version + 1

                # 旧任务收尾（expected_version 过期）也必须被拒绝
                updated = update_document_status(
                    db,
                    doc_id,
                    status="ready",
                    progress=100,
                    chunk_count=999,
                    expected_version=stale_version,
                )
                assert updated == 0

            with session_scope() as db:
                row = db.execute(
                    text("SELECT version, chunk_count FROM documents WHERE id = :id"),
                    {"id": doc_id},
                ).mappings().first()
            assert int(row["version"]) == stale_version + 1
            assert int(row["chunk_count"] or 0) == real_chunk_count
        finally:
            self._cleanup(doc_id)
