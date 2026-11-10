"""前后端契约测试（阶段 6 验收项）：前端调用的每个接口，字段名/类型/状态码逐一锁定。

为什么值得单独写：TS 接口是手抄的，后端改字段名时编译器不会报错，
只会在浏览器里静默变成 `undefined`。这个文件把"抄错"变成 CI 里的一条红。
字段清单与 `frontend/src/api/*.ts` 的 interface 一一对应。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.security import CurrentUser, hash_password
from app.main import app
from tests.fakes_db import FakeDB, override_deps

SID = "01J" + "A" * 23
DOC = "doc1"


def assert_keys(obj: dict, spec: dict[str, type | tuple], where: str = "") -> None:
    """校验字段存在且类型匹配（int 可接受 bool 之外的数值，float 接受 int）。"""
    for key, expected in spec.items():
        assert key in obj, f"{where} 缺少字段 {key}；实际字段={sorted(obj)}"
        value = obj[key]
        if isinstance(expected, tuple):
            assert value is None or isinstance(value, expected), f"{where}.{key} 类型应为 {expected}，实际 {type(value)}"
        elif expected is float:
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"{where}.{key} 应为数值，实际 {type(value)}"
            )
        else:
            assert isinstance(value, expected), f"{where}.{key} 类型应为 {expected.__name__}，实际 {type(value)}"


@pytest.fixture
def db():
    return FakeDB()


@pytest.fixture
def client(db, monkeypatch):
    from app.core.access import AccessScope

    monkeypatch.setattr(
        "app.api.chat.prepare_scope", lambda user, session_id, kb_ids: AccessScope.legacy_all()
    )
    restore = override_deps(app, db)
    yield TestClient(app)
    restore()


# ---------------------------------------------------------------------------
# 认证（frontend/src/api/auth.ts）
# ---------------------------------------------------------------------------
class TestAuthContract:
    def test_captcha(self, client, monkeypatch):
        monkeypatch.setattr(
            "app.api.auth.captcha_svc.new_captcha",
            lambda **kw: ("cap-1234", "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="),
        )
        data = client.post("/api/auth/captcha").json()
        assert_keys(data, {"captcha_id": str, "image": str, "expires_in": int}, "captcha")
        assert data["image"].startswith("data:image/svg+xml;base64,")

    def test_check_username(self, client):
        data = client.get("/api/auth/check-username", params={"username": "someone"}).json()
        assert_keys(data, {"available": bool, "reason": (str,)}, "check-username")

    def test_register(self, client, monkeypatch):
        monkeypatch.setattr("app.api.auth.captcha_svc.verify_captcha", lambda *a, **k: True)
        resp = client.post(
            "/api/auth/register",
            json={"username": "newbie", "password": "secret123", "captcha_id": "cap-1234", "captcha_code": "1234"},
        )
        assert resp.status_code == 201
        assert_keys(resp.json(), {"id": int, "username": str}, "register")

    def test_login(self, client, db, monkeypatch):
        monkeypatch.setattr("app.api.auth.captcha_svc.verify_captcha", lambda *a, **k: True)
        db.add_user("alice", hash_password("secret123"))
        data = client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "secret123", "captcha_id": "cap-1234", "captcha_code": "1234"},
        ).json()
        assert_keys(data, {"access_token": str, "token_type": str, "expires_in": int}, "login")
        assert_keys(data["user"], {"id": int, "username": str}, "login.user")

    def test_me(self, client):
        data = client.get("/api/auth/me").json()
        assert_keys(data, {"id": int, "username": str}, "me")


# ---------------------------------------------------------------------------
# 会话（frontend/src/api/sessions.ts）
# ---------------------------------------------------------------------------
SESSION_SPEC = {
    "id": str,
    "title": str,
    "kb_ids": list,
    "model": str,
    "temperature": float,
    "top_p": float,
    "max_tokens": int,
    "history_rounds": int,
    "message_count": int,
    "created_at": (str,),
    "updated_at": (str,),
}


class TestSessionsContract:
    def test_create(self, client):
        resp = client.post("/api/sessions", json={})
        assert resp.status_code == 201
        assert_keys(resp.json(), SESSION_SPEC, "sessions.create")

    def test_list_and_patch(self, client, db):
        db.add_session(SID, user_id=1)
        items = client.get("/api/sessions").json()
        assert isinstance(items, list) and items
        assert_keys(items[0], SESSION_SPEC, "sessions.list")

        patched = client.patch(f"/api/sessions/{SID}", json={"title": "新标题"}).json()
        assert patched["title"] == "新标题"
        assert_keys(patched, SESSION_SPEC, "sessions.patch")

    def test_delete(self, client, db):
        db.add_session(SID, user_id=1)
        assert client.delete(f"/api/sessions/{SID}").json() == {"deleted": SID}

    def test_messages(self, client, db):
        db.add_session(SID, user_id=1)
        db.add_message(SID, 1, "user", "问题")
        db.add_message(SID, 2, "assistant", "回答", sources=[], cached=0)
        data = client.get(f"/api/sessions/{SID}/messages").json()
        assert_keys(data, {"count": int, "items": list}, "messages")
        assert_keys(
            data["items"][1],
            {
                "id": int,
                "seq": int,
                "role": str,
                "content": str,
                "sources": (list,),
                "intent": (str,),
                "rewrite_type": (str,),
                "route": (str,),
                "cached": bool,
                "latency_ms": (int,),
                "ttft_ms": (int,),
                "token_usage": (dict,),
                "error": (str,),
                "created_at": str,
            },
            "messages.items[]",
        )


# ---------------------------------------------------------------------------
# SSE（frontend/src/api/chat.ts）
# ---------------------------------------------------------------------------
class TestSseContract:
    def test_event_payloads(self, client, monkeypatch):
        monkeypatch.setattr("app.api.chat.get_owned_session", lambda *a, **kw: object())

        from app.generation.stream import AnswerEngine
        from app.orchestration.models import IntentResult, RouteDecision
        from app.orchestration.qa_cache import QALookupResult
        from app.orchestration.service import QueryPlan
        from tests.test_sse import FakeLLMStream, fake_retrieve_fn, make_hit

        plan = QueryPlan(
            query="试用期最长多久",
            intent=IntentResult(intent="labor_law", sub_queries=["试用期最长多久"], source="llm"),
            route=RouteDecision(route="pure_vector", kb_ids=["kb_labor"]),
            qa=QALookupResult(),
            sub_queries=["试用期最长多久"],
        )
        monkeypatch.setattr("app.generation.stream.analyze_query", lambda *a, **kw: plan)
        monkeypatch.setattr(
            "app.api.chat.AnswerEngine",
            lambda **kw: AnswerEngine(llm_client=FakeLLMStream(), retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=2.0)])),
        )

        from app.core.sse import parse_sse_frames

        resp = client.post(
            "/api/chat/stream",
            json={"query": "试用期最长多久", "session_id": SID},
        )
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames = parse_sse_frames(resp.text)
        by_event = {}
        for name, data in frames:
            by_event.setdefault(name, []).append(data)

        assert_keys(
            by_event["meta"][0],
            {
                "message_id": str,
                "intent": str,
                "route": str,
                "rewrite_type": str,
                "complexity": str,
                "cached": bool,
                "degraded": (bool, str),
            },
            "sse.meta",
        )
        assert_keys(by_event["stage"][0], {"stage": str}, "sse.stage")
        assert by_event["stage"][0]["stage"] in {"intent", "retrieval", "rerank", "generate"}

        src = by_event["sources"][0]["sources"][0]
        assert_keys(
            src,
            {
                "law": str,
                "chapter_path": str,
                "article_no": str,
                "score": float,
                "snippet": str,
                "kind": str,
            },
            "sse.sources[]",
        )
        assert_keys(by_event["delta"][0], {"text": str}, "sse.delta")
        assert_keys(
            by_event["done"][0],
            {"ttft_ms": int, "latency_ms": int, "token_usage": dict, "sources_count": int, "cached": bool},
            "sse.done",
        )

    def test_error_event_shape(self, client, monkeypatch):
        from app.generation.llm_client import LLMError
        from app.generation.stream import AnswerEngine
        from app.orchestration.models import IntentResult, RouteDecision
        from app.orchestration.qa_cache import QALookupResult
        from app.orchestration.service import QueryPlan
        from tests.test_sse import FakeLLMStream, fake_retrieve_fn, make_hit

        plan = QueryPlan(
            query="试用期最长多久",
            intent=IntentResult(intent="labor_law", sub_queries=["试用期最长多久"], source="llm"),
            route=RouteDecision(route="pure_vector", kb_ids=["kb_labor"]),
            qa=QALookupResult(),
            sub_queries=["试用期最长多久"],
        )
        monkeypatch.setattr("app.generation.stream.analyze_query", lambda *a, **kw: plan)
        monkeypatch.setattr(
            "app.api.chat.AnswerEngine",
            lambda **kw: AnswerEngine(
                llm_client=FakeLLMStream(raises=LLMError("限流")),
                retrieve_fn=fake_retrieve_fn([make_hit("c1", 0.5, rerank=1.0)]),
            ),
        )

        from app.core.sse import parse_sse_frames

        resp = client.post("/api/chat/stream", json={"query": "试用期最长多久"})
        frames = parse_sse_frames(resp.text)
        err = next(d for n, d in frames if n == "error")
        assert_keys(err, {"code": str, "message": str, "retriable": bool}, "sse.error")

    def test_session_requires_login(self, db):
        restore = override_deps(app, db, user=None)
        try:
            resp = TestClient(app).post("/api/chat/stream", json={"query": "你好", "session_id": SID})
            assert resp.status_code == 401
        finally:
            restore()


# ---------------------------------------------------------------------------
# 知识库 / 文档 / 块（frontend/src/api/kb.ts）
# ---------------------------------------------------------------------------
class TestKbContract:
    def test_list_kbs(self, client):
        items = client.get("/api/kbs").json()
        assert isinstance(items, list) and items
        assert_keys(
            items[0],
            {
                "id": str,
                "name": str,
                "description": (str,),
                "embedding_model": str,
                "doc_count": int,
                "chunk_count": int,
                "updated_at": (str,),
            },
            "kbs[]",
        )

    def test_create_kb(self, client):
        resp = client.post("/api/kbs", json={"id": "kb_test", "name": "测试库"})
        assert resp.status_code == 201
        assert_keys(resp.json(), {"id": str, "name": str}, "kbs.create")

    def test_documents(self, client):
        items = client.get("/api/kbs/kb_labor/documents").json()
        assert isinstance(items, list) and items
        assert_keys(
            items[0],
            {
                "id": str,
                "filename": str,
                "file_type": str,
                "file_size": int,
                "status": str,
                "progress": int,
                "article_count": int,
                "chunk_count": int,
                "created_at": (str,),
                "updated_at": (str,),
            },
            "documents[]",
        )
        assert items[0]["status"] in {"pending", "parsing", "ready", "failed"}

    def test_document_detail(self, client):
        data = client.get(f"/api/documents/{DOC}").json()
        assert_keys(data, {"id": str, "kb_id": str, "filename": str, "parse_config": (dict,)}, "document")

    def test_document_delete_and_reingest(self, client, monkeypatch):
        monkeypatch.setattr("app.api.knowledge.delete_document_service", lambda db, doc_id: None)
        assert client.delete(f"/api/documents/{DOC}").json() == {"deleted": DOC}

        monkeypatch.setattr("app.api.knowledge.run_ingest_job", lambda *a, **k: None)

        class _ExistingPath:
            def exists(self) -> bool:
                return True

        monkeypatch.setattr("app.api.knowledge._resolve", lambda storage_path: _ExistingPath())
        resp = client.post(f"/api/documents/{DOC}/reingest")
        assert resp.status_code == 202
        assert_keys(resp.json(), {"doc_id": str, "status": str}, "documents.reingest")

    def test_upload_shape(self, client, monkeypatch):
        """上传返回 {accepted:[{doc_id, filename, status}], count}；解析任务被替身跳过。"""
        monkeypatch.setattr("app.api.knowledge.validate_upload", lambda *a, **k: None)
        monkeypatch.setattr("app.api.knowledge.save_upload", lambda data, kb, name: __import__("pathlib").Path("data/uploads/x.md"))
        monkeypatch.setattr("app.api.knowledge.create_pending_document", lambda *a, **k: None)
        monkeypatch.setattr("app.api.knowledge.run_ingest_job", lambda *a, **k: None)

        resp = client.post(
            "/api/kbs/kb_labor/documents",
            files={"files": ("测试.md", b"# hello", "text/markdown")},
            data={"chunk_size": "300", "overlap": "50", "retrieval_mode": "hybrid"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert_keys(data, {"accepted": list, "count": int}, "documents.upload")
        assert_keys(data["accepted"][0], {"doc_id": str, "filename": str, "status": str}, "documents.upload.accepted[]")

    def test_chunks(self, client):
        data = client.get(f"/api/documents/{DOC}/chunks").json()
        assert_keys(data, {"total": int, "limit": int, "offset": int, "items": list}, "chunks")
        assert_keys(
            data["items"][0],
            {
                "id": str,
                "chunk_type": str,
                "parent_id": (str,),
                "chunk_index": int,
                "article_no": (str,),
                "chapter_path": (str,),
                "content": str,
                "char_count": int,
                "in_milvus": int,
                "is_enabled": int,
            },
            "chunks.items[]",
        )
        assert data["items"][0]["chunk_type"] in {"parent", "child"}

    def test_chunk_patch_and_delete(self, client, monkeypatch):
        monkeypatch.setattr("app.api.knowledge.update_chunk_content", lambda db, cid, content: {"id": cid, "content": content})
        resp = client.patch("/api/chunks/c1", json={"content": "新内容"})
        assert resp.status_code == 200
        assert resp.json()["content"] == "新内容"

        monkeypatch.setattr("app.api.knowledge.disable_chunk", lambda db, cid: None)
        assert client.delete("/api/chunks/c1").json() == {"deleted": "c1"}
