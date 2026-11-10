"""LA-P2 安全边界矩阵（全离线）：未登录 / 恶意参数 / 已删除资源 / 历史来源。

策略（与 ``app/api/sessions.py``、``app/core/errors.py`` 保持一致）：

* 未登录访问受保护接口 → 401，统一错误信封，不返回业务数据；
* 资源不可见（跨租户/不存在/已删除）→ 404，不用 403 泄露存在性；
* 资源可见但角色不足 → 403；
* 旧消息里的来源按"当前仍存在 + 当前仍可见"过滤，历史快照不能绕过删除/撤权。

本文件的 ``PROTECTED_ENDPOINTS`` 是**白名单式矩阵**：新增 ``/api`` 端点若
既不在公开清单、也不在矩阵里，``test_all_api_routes_are_classified`` 会失败，
强迫作者补授权用例（默认拒绝）。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.access import ROLE_EDITOR
from app.core.security import CurrentUser
from app.main import app
from tests.fakes_db import FakeDB, override_deps

SECRET_KB = "kb_secret"
SECRET_DOC = "doc_secret"

#: 公开端点：无需登录（登录/注册/健康检查/指标/匿名可用的问答入口）。
PUBLIC_ENDPOINTS: set[tuple[str, str]] = {
    ("GET", "/"),
    ("GET", "/metrics"),
    ("GET", "/api/health"),
    ("GET", "/api/health/live"),
    ("POST", "/api/auth/captcha"),
    ("GET", "/api/auth/check-username"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/register"),
    ("POST", "/api/chat/stream"),
    ("POST", "/api/chat/suggestions"),
}

#: 受保护端点矩阵：方法 + OpenAPI 路径模板（路径参数统一用 dummy 值请求）。
PROTECTED_ENDPOINTS: set[tuple[str, str]] = {
    ("GET", "/api/auth/me"),
    ("GET", "/api/kbs"),
    ("POST", "/api/kbs"),
    ("DELETE", "/api/kbs/{kb_id}"),
    ("GET", "/api/kbs/{kb_id}/documents"),
    ("POST", "/api/kbs/{kb_id}/documents"),
    ("GET", "/api/documents/{doc_id}"),
    ("DELETE", "/api/documents/{doc_id}"),
    ("GET", "/api/documents/{doc_id}/chunks"),
    ("POST", "/api/documents/{doc_id}/reingest"),
    ("PATCH", "/api/chunks/{chunk_id}"),
    ("DELETE", "/api/chunks/{chunk_id}"),
    ("POST", "/api/ingest/local"),
    ("POST", "/api/intent/preview"),
    ("POST", "/api/retrieval/preview"),
    ("GET", "/api/sessions"),
    ("POST", "/api/sessions"),
    ("PATCH", "/api/sessions/{session_id}"),
    ("DELETE", "/api/sessions/{session_id}"),
    ("GET", "/api/sessions/{session_id}/messages"),
    ("GET", "/api/traces/{trace_id}"),
}


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
    return fake


@pytest.fixture
def db() -> FakeDB:
    return _db()


@pytest.fixture
def client(db):
    restore = override_deps(app, db, user=CurrentUser(id=1, username="editor", tenant_id="default", role=ROLE_EDITOR))
    yield TestClient(app)
    restore()


def _url(path: str) -> str:
    return (
        path.replace("{kb_id}", "kb_nonexistent")
        .replace("{doc_id}", "doc_nonexistent")
        .replace("{chunk_id}", "chunk_nonexistent")
        .replace("{session_id}", "sess_nonexistent")
        .replace("{trace_id}", "trace_nonexistent")
    )


def test_all_api_routes_are_classified():
    """任何新增 /api 端点必须显式归入公开或受保护矩阵，否则默认拒绝。"""
    paths = app.openapi()["paths"]
    api_routes = {
        (method.upper(), path)
        for path, operations in paths.items()
        if path.startswith("/api")
        for method in operations
    }
    classified = PUBLIC_ENDPOINTS | PROTECTED_ENDPOINTS
    missing = sorted(api_routes - classified)
    assert not missing, f"新增端点未纳入安全矩阵: {missing}"


@pytest.mark.parametrize("method,path", sorted(PROTECTED_ENDPOINTS))
def test_unauthenticated_requests_are_401_without_data(method: str, path: str):
    with TestClient(app) as unauthenticated:
        resp = unauthenticated.request(method, _url(path), json={})
    assert resp.status_code == 401, f"{method} {path} -> {resp.status_code}"
    body = resp.json()
    assert set(body) == {"code", "message", "detail", "request_id"}
    assert body["code"] == "unauthorized"
    assert body["request_id"]
    text = resp.text.lower()
    for leaked in ("password", "sk-", "traceback", "sqlalchemy", "select "):
        assert leaked not in text, f"401 响应泄露 {leaked}"


class TestMaliciousInput:
    def test_sql_injection_in_path_is_404_not_500(self, client):
        resp = client.get("/api/documents/x%27%20OR%20%271%27%3D%271")
        assert resp.status_code == 404
        assert resp.json()["code"] == "not_found"
        assert "select" not in resp.text.lower()

    def test_path_traversal_is_rejected(self, client):
        resp = client.get("/api/documents/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (404, 422)
        assert "root:" not in resp.text

    def test_sql_injection_in_kb_ids_is_404(self, client):
        resp = client.post(
            "/api/retrieval/preview",
            json={"question": "劳动合同解除", "kb_ids": ["kb_labor' OR '1'='1"]},
        )
        assert resp.status_code == 404
        assert resp.json()["code"] == "not_found"

    def test_oversized_question_is_422(self, client):
        resp = client.post("/api/retrieval/preview", json={"question": "法" * 501})
        assert resp.status_code == 422
        assert resp.json()["code"] == "validation_failed"

    def test_malformed_json_is_422(self, client):
        resp = client.post(
            "/api/retrieval/preview",
            content="{not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "validation_failed"

    def test_control_chars_in_chunk_content_do_not_500(self, client):
        resp = client.patch("/api/chunks/c1", json={"content": "正常\x00内容\x07"})
        assert resp.status_code in (200, 404)
        assert resp.status_code != 500


class TestDeletedOrMissingResources:
    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/api/documents/doc_nonexistent"),
            ("DELETE", "/api/documents/doc_nonexistent"),
            ("GET", "/api/documents/doc_nonexistent/chunks"),
            ("POST", "/api/documents/doc_nonexistent/reingest"),
            ("PATCH", "/api/chunks/chunk_nonexistent"),
            ("DELETE", "/api/chunks/chunk_nonexistent"),
            ("DELETE", "/api/kbs/kb_nonexistent"),
            ("GET", "/api/kbs/kb_nonexistent/documents"),
            ("PATCH", "/api/sessions/sess_nonexistent"),
            ("DELETE", "/api/sessions/sess_nonexistent"),
            ("GET", "/api/sessions/sess_nonexistent/messages"),
            ("GET", "/api/traces/trace_nonexistent"),
        ],
    )
    def test_missing_resources_are_404(self, client, method: str, path: str):
        # PATCH /api/chunks 的 body 必填，先通过校验再断言资源不存在
        payload = {"content": "x"} if method == "PATCH" and "/chunks/" in path else {}
        resp = client.request(method, path, json=payload)
        assert resp.status_code == 404, f"{method} {path} -> {resp.status_code}"
        assert resp.json()["code"] == "not_found"

    def test_deleted_session_is_404(self, client):
        created = client.post("/api/sessions", json={"title": "t", "kb_ids": ["kb_labor"]})
        assert created.status_code == 201
        session_id = created.json()["id"]
        assert client.delete(f"/api/sessions/{session_id}").status_code == 200
        assert client.get(f"/api/sessions/{session_id}/messages").status_code == 404


class TestHistoricalSources:
    def test_history_sources_filter_deleted_doc_and_revoked_kb(self, db, client):
        """旧消息来源：存在且仍可见才返回；删除文档/跨租户库的来源被过滤。"""
        db.add_session("s_hist", 1, kb_ids=["kb_labor"])
        db.add_message(
            "s_hist",
            1,
            "assistant",
            "历史回答",
            sources=[
                {"doc_id": "doc1", "kb_id": "kb_labor", "title": "劳动合同法.docx"},
                {"doc_id": "doc_deleted", "kb_id": "kb_labor", "title": "已删除.pdf"},
                {"doc_id": SECRET_DOC, "kb_id": SECRET_KB, "title": "别家资料.md"},
            ],
        )
        resp = client.get("/api/sessions/s_hist/messages")
        assert resp.status_code == 200
        sources = resp.json()["items"][0]["sources"]
        assert [s["doc_id"] for s in sources] == ["doc1"]
        assert SECRET_KB not in resp.text
        assert "别家资料" not in resp.text
        assert "已删除" not in resp.text

    def test_history_message_without_sources_stays_none(self, db, client):
        db.add_session("s_none", 1, kb_ids=["kb_labor"])
        db.add_message("s_none", 1, "user", "问题", sources=None)
        resp = client.get("/api/sessions/s_none/messages")
        assert resp.status_code == 200
        assert resp.json()["items"][0]["sources"] is None
