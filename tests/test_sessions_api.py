"""会话 / 消息接口测试（阶段 6，离线，假 MySQL）。

重点验证三件事：
1. **契约**：字段名与类型（前端 TypeScript 接口照此声明）；
2. **归属隔离**：访问他人会话一律 404（不泄露存在性）；
3. **参数校验**：温度/轮数等越界由 pydantic 拦下（422）。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.security import CurrentUser, hash_password
from app.main import app
from tests.fakes_db import FakeDB, override_deps


@pytest.fixture
def db():
    fake = FakeDB()
    # 会话默认勾选两个种子库；FakeDB 默认只造了一个，这里补齐第二个
    fake.kbs.append(
        {
            "id": "kb_civil",
            "name": "民法典知识库",
            "description": "民法典",
            "embedding_model": "bge-m3",
            "doc_count": 0,
            "chunk_count": 0,
            "updated_at": fake.kbs[0]["updated_at"],
            "corpus_version": 1,
            "index_version": 1,
            "is_public": 1,
            "tenant_id": "default",
            "owner_id": None,
        }
    )
    return fake


@pytest.fixture
def client(db):
    restore = override_deps(app, db)
    yield TestClient(app)
    restore()


class TestSessionList:
    def test_lists_only_own_sessions(self, client, db):
        db.add_session("S1" + "0" * 24, user_id=1, title="我的对话")
        db.add_session("S2" + "0" * 24, user_id=2, title="别人的对话")
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["title"] == "我的对话"

    def test_session_shape(self, client, db):
        db.add_session("S1" + "0" * 24, user_id=1, kb_ids=["kb_labor"])
        item = client.get("/api/sessions").json()[0]
        assert isinstance(item["id"], str)
        assert isinstance(item["kb_ids"], list) and item["kb_ids"] == ["kb_labor"]
        assert isinstance(item["temperature"], float)
        assert isinstance(item["history_rounds"], int)
        assert item["model"] == "deepseek-v4.1-flash"
        assert set(item) >= {
            "id", "title", "kb_ids", "model", "temperature", "top_p",
            "max_tokens", "history_rounds", "message_count", "created_at", "updated_at",
        }

    def test_soft_deleted_hidden(self, client, db):
        from datetime import datetime

        db.add_session("S1" + "0" * 24, user_id=1, deleted_at=datetime(2026, 1, 1))
        assert client.get("/api/sessions").json() == []

    def test_requires_login(self, db):
        restore = override_deps(app, db, user=None)  # 不覆盖鉴权 → 无 token 应 401
        try:
            resp = TestClient(app).get("/api/sessions")
            assert resp.status_code == 401
        finally:
            restore()


class TestSessionCreate:
    def test_create_defaults(self, client):
        resp = client.post("/api/sessions", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["id"]) == 26
        assert data["title"] == "新对话"
        assert data["kb_ids"] == ["kb_labor", "kb_civil"]
        assert data["temperature"] == 0.3
        assert data["top_p"] == 0.8
        assert data["max_tokens"] == 2048
        assert data["history_rounds"] == 5
        assert data["message_count"] == 0

    def test_create_with_params(self, client):
        resp = client.post(
            "/api/sessions",
            json={"title": "裁员补偿", "kb_ids": ["kb_labor"], "model": "qwen-plus", "temperature": 0.7, "history_rounds": 3},
        )
        data = resp.json()
        assert data["title"] == "裁员补偿"
        assert data["kb_ids"] == ["kb_labor"]
        assert data["model"] == "qwen-plus"
        assert data["temperature"] == 0.7
        assert data["history_rounds"] == 3

    @pytest.mark.parametrize(
        "payload",
        [
            {"temperature": 2.0},
            {"top_p": 1.5},
            {"max_tokens": 100},
            {"history_rounds": 30},
            {"title": ""},
        ],
    )
    def test_validation(self, client, payload):
        assert client.post("/api/sessions", json=payload).status_code == 422


class TestSessionUpdate:
    def test_rename(self, client, db):
        db.add_session("S1" + "0" * 24, user_id=1)
        resp = client.patch(f"/api/sessions/{'S1' + '0' * 24}", json={"title": "改名了"})
        assert resp.status_code == 200
        assert resp.json()["title"] == "改名了"

    def test_update_params(self, client, db):
        db.add_session("S1" + "0" * 24, user_id=1)
        sid = "S1" + "0" * 24
        resp = client.patch(f"/api/sessions/{sid}", json={"temperature": 0.9, "top_p": 0.5, "kb_ids": ["kb_civil"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["temperature"] == 0.9
        assert data["top_p"] == 0.5
        assert data["kb_ids"] == ["kb_civil"]

    def test_update_other_user_404(self, client, db):
        db.add_session("S2" + "0" * 24, user_id=2, title="别人")
        resp = client.patch(f"/api/sessions/{'S2' + '0' * 24}", json={"title": "劫持"})
        assert resp.status_code == 404
        assert resp.json()["code"] == "not_found"

    def test_update_unknown_404(self, client):
        assert client.patch("/api/sessions/" + "X" * 26, json={"title": "x"}).status_code == 404


class TestSessionDelete:
    def test_soft_delete(self, client, db):
        sid = "S1" + "0" * 24
        db.add_session(sid, user_id=1)
        resp = client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json() == {"deleted": sid}
        assert client.get("/api/sessions").json() == []

    def test_delete_other_user_404(self, client, db):
        db.add_session("S2" + "0" * 24, user_id=2)
        assert client.delete(f"/api/sessions/{'S2' + '0' * 24}").status_code == 404


class TestMessages:
    def test_messages_asc_with_parsed_json(self, client, db):
        sid = "S1" + "0" * 24
        db.add_session(sid, user_id=1)
        db.add_message(sid, 1, "user", "试用期多久")
        db.add_message(
            sid, 2, "assistant", "最长六个月",
            sources=[{"law": "中华人民共和国劳动合同法", "article_no": "第十九条", "score": 0.9, "snippet": "…", "chapter_path": "", "kind": "law"}],
            intent="labor_law", rewrite_type="direct", route="pure_structured",
            cached=0, ttft_ms=32, latency_ms=35, token_usage={"total_tokens": 10},
        )
        resp = client.get(f"/api/sessions/{sid}/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        roles = [m["role"] for m in data["items"]]
        assert roles == ["user", "assistant"]
        assistant = data["items"][1]
        assert isinstance(assistant["sources"], list) and assistant["sources"][0]["article_no"] == "第十九条"
        assert assistant["cached"] is False  # TINYINT → bool
        assert assistant["ttft_ms"] == 32
        assert isinstance(assistant["token_usage"], dict)
        assert set(assistant) >= {
            "id", "seq", "role", "content", "sources", "intent", "rewrite_type",
            "route", "cached", "latency_ms", "ttft_ms", "token_usage", "error", "created_at",
        }

    def test_messages_other_user_404(self, client, db):
        db.add_session("S2" + "0" * 24, user_id=2)
        assert client.get(f"/api/sessions/{'S2' + '0' * 24}/messages").status_code == 404
