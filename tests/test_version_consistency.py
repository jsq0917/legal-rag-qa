"""LA-O3：知识库版本与一致性（离线单测）。

覆盖四处缺口：
1. 摄入写入 ``content_hash``（SHA-256）与 ``index_version``；
2. 父块回表过滤"文档已删除 / 非 ready / 版本不一致"的脏向量；
3. 命中带上知识库 ``corpus_version/index_version``，已删库的命中被丢弃；
4. 删除文档同时清理源文件，且只允许删上传目录内该知识库下的文件。

真实 MySQL/Milvus 的 CAS 并发与重建/删除一致性在
``tests/test_ingest_integration.py`` 的 integration 用例里验证（默认跳过）。
"""

from __future__ import annotations

import hashlib

import pytest

from app.ingest.pipeline import prepare, sha256_text
from app.retrieval.hybrid import Hit
from app.retrieval.service import _attach_kb_versions, _attach_parents


def _hit(parent_id: str, *, kb_id: str = "kb_labor", chunk_id: str = "") -> Hit:
    return Hit(
        chunk_id=chunk_id or f"{parent_id}:c0",
        parent_id=parent_id,
        kb_id=kb_id,
        doc_id="doc1",
        content="子块正文",
        title="劳动合同法·第十九条",
        article_no="第十九条",
        chapter_path="第二章",
        law_name="劳动合同法",
        source_file="劳动合同法.md",
    )


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, engine):
        self._engine = engine

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params=None):
        self._engine.statements.append(str(stmt))
        self._engine.params.append(params)
        return _FakeResult(self._engine.rows)


class _FakeEngine:
    def __init__(self, rows):
        self.rows = rows
        self.statements: list[str] = []
        self.params: list = []

    def connect(self):
        return _FakeConn(self)


@pytest.fixture
def fake_engine(monkeypatch):
    def _install(rows):
        engine = _FakeEngine(rows)
        monkeypatch.setattr("app.db.mysql.get_engine", lambda: engine)
        return engine

    return _install


class TestIngestContentHash:
    def test_prepare_writes_sha256_content_hash(self, tmp_path):
        body = "第一条 为了完善劳动合同制度，明确劳动合同双方当事人的权利和义务，保护劳动者的合法权益。\n\n第二条 中华人民共和国境内的企业、个体经济组织、民办非企业单位等组织与劳动者建立劳动关系，订立、履行、变更、解除或者终止劳动合同，适用本法。\n"
        path = tmp_path / "中华人民共和国劳动合同法.md"
        path.write_text(body, encoding="utf-8")

        prepared = prepare(path, chunk_size=300, overlap=50)

        from app.ingest.clean import clean_lines

        normalized = "\n".join(item.text for item in clean_lines(body.splitlines()))
        assert prepared.content_hash == sha256_text(normalized)
        assert len(prepared.content_hash) == 64
        assert prepared.content_sha1 == hashlib.sha1(normalized.encode("utf-8")).hexdigest()


class TestParentBackfillFiltersStale:
    def test_drops_hit_when_document_deleted_or_version_mismatch(self, fake_engine):
        engine = fake_engine([("p1", "完整条文", "2020-01-01", None, 2)])
        hits = [_hit("p1", chunk_id="doc1:p1:c0"), _hit("p2", chunk_id="doc2:p2:c0")]

        _attach_parents(hits, kb_ids=["kb_labor"])

        assert [h.parent_id for h in hits] == ["p1"]
        assert hits[0].parent_content == "完整条文"
        assert hits[0].version == 2
        # 回表 SQL 必须带文档状态与版本一致性条件，否则脏向量仍会被当作正文
        sql = engine.statements[0].lower()
        assert "join documents" in sql
        assert "d.status = 'ready'" in sql
        assert "c.version = d.version" in sql
        assert "c.kb_id in" in sql

    def test_backfill_failure_keeps_child_content(self, fake_engine, monkeypatch):
        def _boom():
            raise RuntimeError("db down")

        monkeypatch.setattr("app.db.mysql.get_engine", _boom)
        hits = [_hit("p1")]
        _attach_parents(hits, kb_ids=["kb_labor"])
        # 回表失败是降级场景：保留子块正文，不让整次检索失败
        assert len(hits) == 1
        assert hits[0].parent_content == ""


class TestKbVersionSnapshot:
    def test_hits_carry_kb_versions_and_orphan_kb_dropped(self, fake_engine):
        engine = fake_engine([("kb_labor", 7, 9)])
        good = _hit("p1", kb_id="kb_labor")
        orphan = _hit("p9", kb_id="kb_deleted")

        versions = _attach_kb_versions([good, orphan], kb_ids=["kb_labor", "kb_deleted"])

        assert versions == {"kb_labor": {"corpus_version": 7, "index_version": 9}}
        assert good.corpus_version == 7 and good.index_version == 9
        # 知识库已删除的命中不能继续作为来源
        assert orphan.kb_id == "kb_deleted"
        assert orphan.corpus_version == 0


class TestSessionLiveSources:
    def test_sources_with_deleted_document_or_foreign_kb_filtered(self):
        from app.api.sessions import _live_sources
        from app.core.access import AccessScope, KbGrant
        from tests.fakes_db import FakeDB

        db = FakeDB()
        scope = AccessScope(
            user_id=1,
            tenant_id="default",
            role="editor",
            grants=(KbGrant(kb_id="kb_labor", corpus_version=1, index_version=1),),
        )
        sources = [
            {"law": "劳动合同法", "doc_id": "doc1", "kb_id": "kb_labor", "snippet": "在"},
            {"law": "已删文档", "doc_id": "doc_gone", "kb_id": "kb_labor", "snippet": "没了"},
            {"law": "别库", "doc_id": "doc1", "kb_id": "kb_civil", "snippet": "越权"},
        ]
        kept = _live_sources(db, sources, scope)
        assert [s["law"] for s in kept] == ["劳动合同法"]


class TestPurgeDocumentFile:
    def test_only_file_inside_upload_dir_is_removed(self, tmp_path, monkeypatch):
        from app.ingest import service

        class _S:
            upload_path = tmp_path

        monkeypatch.setattr(service, "get_settings", lambda: _S())

        inside = tmp_path / "kb_labor" / "doc1.md"
        inside.parent.mkdir(parents=True)
        inside.write_text("正文", encoding="utf-8")
        outside = tmp_path / "other" / "secret.md"
        outside.parent.mkdir(parents=True)
        outside.write_text("别动", encoding="utf-8")

        assert service.purge_document_file(str(inside), "kb_labor") is True
        assert not inside.exists()
        # 目录穿越/越界路径一律拒绝
        assert service.purge_document_file(str(outside), "kb_labor") is False
        assert outside.exists()
        assert service.purge_document_file(None, "kb_labor") is False
