"""租户 / 角色权限矩阵（LA-O2 / LA-P2 前置安全用例，全离线）。

策略统一为：

* 资源不可见（跨租户私库、匿名访问私库）→ 404，不泄露存在性；
* 资源可见但角色不足（viewer 写、editor 建公共库）→ 403；
* 检索/缓存/调试端点即使显式传入越权 kb_id，也只会得到空结果或 404。

这些用例跑在 FakeDB 上，只验证路由层的授权判定；真实 MySQL 行为由
``tests/integration`` 下的集成用例覆盖。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.access import ROLE_ADMIN, ROLE_EDITOR, ROLE_VIEWER, resolve_access_scope
from app.core.security import CurrentUser
from app.core.sse import parse_sse_frames
from app.main import app
from tests.fakes_db import FakeDB, override_deps
from tests.test_sse import get_event

SECRET_KB = "kb_secret"
SECRET_DOC = "doc_secret"
SECRET_CHUNK = "c_secret"


def _db() -> FakeDB:
    fake = FakeDB()
    fake.kbs.append(
        {
            "id": SECRET_KB,
            "name": "其他租户私库",
            "description": "t2",
            "embedding_model": "bge-m3",
            "doc_count": 1,
            "chunk_count": 1,
            "updated_at": fake.kbs[0]["updated_at"],
            "corpus_version": 7,
            "index_version": 9,
            "is_public": 0,
            "tenant_id": "tenant-2",
            "owner_id": 99,
        }
    )
    fake.docs.append(
        {
            **{k: v for k, v in fake.docs[0].items()},
            "id": SECRET_DOC,
            "kb_id": SECRET_KB,
            "filename": "secret.md",
        }
    )
    fake.chunks.append({**{k: v for k, v in fake.chunks[0].items()}, "id": SECRET_CHUNK, "kb_id": SECRET_KB})
    return fake


@pytest.fixture
def db() -> FakeDB:
    return _db()


@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setattr(
        "app.api.chat.prepare_scope",
        lambda user, session_id, kb_ids: resolve_access_scope(db, user, kb_ids),
    )
    restore = override_deps(app, db)
    yield TestClient(app)
    restore()


def _client_for(db, *, user_id=1, role=ROLE_EDITOR, tenant="default"):
    user = CurrentUser(id=user_id, username="u", tenant_id=tenant, role=role)
    restore = override_deps(app, db, user=user)
    return TestClient(app), restore


class TestKbVisibility:
    def test_list_only_public_and_own_tenant(self, client):
        items = client.get("/api/kbs").json()
        assert [k["id"] for k in items] == ["kb_labor"]

    def test_other_tenant_admin_sees_own_private_kb(self, db):
        client, restore = _client_for(db, user_id=99, role=ROLE_ADMIN, tenant="tenant-2")
        try:
            ids = {k["id"] for k in client.get("/api/kbs").json()}
            assert ids == {"kb_labor", SECRET_KB}
        finally:
            restore()

    def test_anonymous_only_sees_public(self):
        class _Rows:
            def mappings(self):
                return self

            def all(self):
                return [
                    {
                        "id": "kb_labor",
                        "corpus_version": 1,
                        "index_version": 1,
                        "is_public": 1,
                        "tenant_id": "default",
                        "owner_id": None,
                    },
                    {
                        "id": SECRET_KB,
                        "corpus_version": 1,
                        "index_version": 1,
                        "is_public": 0,
                        "tenant_id": "tenant-2",
                        "owner_id": 99,
                    },
                ]

        class _DB:
            def execute(self, *a, **kw):
                return _Rows()

        scope = resolve_access_scope(_DB(), None)
        assert scope.kb_ids == ("kb_labor",)


class TestResourceAccess:
    def test_foreign_document_is_404(self, client):
        assert client.get(f"/api/documents/{SECRET_DOC}").status_code == 404
        assert client.delete(f"/api/documents/{SECRET_DOC}").status_code == 404

    def test_foreign_chunk_is_404(self, client):
        assert client.patch(f"/api/chunks/{SECRET_CHUNK}", json={"content": "x"}).status_code == 404
        assert client.delete(f"/api/chunks/{SECRET_CHUNK}").status_code == 404

    def test_foreign_document_writes_are_404(self, client):
        assert client.get(f"/api/kbs/{SECRET_KB}/documents").status_code == 404
        assert client.delete(f"/api/kbs/{SECRET_KB}").status_code == 404

    def test_viewer_cannot_write(self, db):
        client, restore = _client_for(db, role=ROLE_VIEWER)
        try:
            assert client.delete("/api/kbs/kb_labor").status_code == 403
            assert client.patch("/api/chunks/c1", json={"content": "x"}).status_code == 403
            assert client.delete("/api/documents/doc1").status_code == 403
        finally:
            restore()

    def test_viewer_cannot_create_kb(self, db):
        client, restore = _client_for(db, role=ROLE_VIEWER)
        try:
            resp = client.post("/api/kbs", json={"id": "kb_new", "name": "新库"})
            assert resp.status_code == 403
        finally:
            restore()

    def test_editor_cannot_create_public_kb(self, client):
        resp = client.post("/api/kbs", json={"id": "kb_new", "name": "新库", "is_public": True})
        assert resp.status_code == 403

    def test_admin_can_create_public_kb(self, db):
        client, restore = _client_for(db, role=ROLE_ADMIN)
        try:
            resp = client.post("/api/kbs", json={"id": "kb_new", "name": "新库", "is_public": True})
            assert resp.status_code == 201
            assert resp.json()["is_public"] is True
        finally:
            restore()


class TestChatAndDebugScope:
    def test_chat_with_foreign_kb_never_retrieves(self, client):
        resp = client.post(
            "/api/chat/stream",
            json={"query": "秘密条款是什么", "kb_ids": [SECRET_KB], "history": []},
        )
        assert resp.status_code == 200
        meta = get_event(parse_sse_frames(resp.text), "meta")
        assert meta["route"] == "no_data"
        assert meta["cached"] is False
        assert SECRET_KB not in resp.text

    def test_intent_preview_foreign_kb_404(self, client):
        resp = client.post(
            "/api/intent/preview",
            json={"query": "劳动合同解除", "kb_ids": [SECRET_KB]},
        )
        assert resp.status_code == 404

    def test_retrieval_preview_foreign_kb_404(self, client):
        resp = client.post(
            "/api/retrieval/preview",
            json={"question": "劳动合同解除", "kb_ids": [SECRET_KB]},
        )
        assert resp.status_code == 404

    def test_session_with_foreign_kb_404(self, client):
        resp = client.post("/api/sessions", json={"title": "t", "kb_ids": [SECRET_KB]})
        assert resp.status_code == 404

    def test_session_allowed_kb_is_stored(self, client):
        resp = client.post("/api/sessions", json={"title": "t", "kb_ids": ["kb_labor"]})
        assert resp.status_code == 201
        assert resp.json()["kb_ids"] == ["kb_labor"]
