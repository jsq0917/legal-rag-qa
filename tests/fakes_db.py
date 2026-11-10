"""离线测试用的"假 MySQL"：只按 SQL 形态返回预置行，不连数据库。

为什么不用 SQLite：DDL 用了 MySQL 专有语法（JSON / FULLTEXT ngram /
ON UPDATE CURRENT_TIMESTAMP），SQLite 跑不了；而这些测试关心的**不是 SQL
执行结果，而是接口契约**（字段名、类型、状态码）。所以按 SQL 特征分发，
既能覆盖路由层的组装逻辑，又保持全离线、毫秒级。
"""

from __future__ import annotations

from datetime import datetime

from app.core.security import CurrentUser


class _Mappings:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def all(self) -> list[dict]:
        return list(self._rows)

    def first(self) -> dict | None:
        return self._rows[0] if self._rows else None


class _Result:
    def __init__(self, rows: list[dict] | None = None, *, scalar=None, lastrowid=None):
        self._rows = rows or []
        self._scalar = scalar
        self.lastrowid = lastrowid

    def mappings(self) -> _Mappings:
        return _Mappings(self._rows)

    def fetchall(self):
        return [(r.get("id"),) for r in self._rows]

    def scalar(self):
        if self._scalar is not None:
            return self._scalar
        return self._rows[0].get("id") if self._rows else None


def _now() -> datetime:
    return datetime(2026, 10, 23, 8, 0, 0)


class FakeDB:
    """会话 / 用户 / 知识库 / 文档 / 块 的极简内存实现。"""

    def __init__(self):
        self.users: list[dict] = []
        self.sessions: list[dict] = []
        self.messages: list[dict] = []
        self.kbs: list[dict] = [
            {
                "id": "kb_labor",
                "name": "劳动法知识库",
                "description": "劳动合同法 + 劳动法",
                "embedding_model": "bge-m3",
                "doc_count": 2,
                "chunk_count": 320,
                "updated_at": _now(),
                "corpus_version": 1,
                "index_version": 1,
                "is_public": 1,
                "tenant_id": "default",
                "owner_id": None,
            }
        ]
        self.docs: list[dict] = [
            {
                "id": "doc1",
                "kb_id": "kb_labor",
                "filename": "劳动合同法.md",
                "file_type": "md",
                "file_size": 18542,
                "status": "ready",
                "progress": 100,
                "error_msg": None,
                "article_count": 98,
                "chunk_count": 150,
                "version": 1,
                "content_sha1": "abcdef1234567890",
                "parse_config": {"chunk_size": 300, "overlap": 50, "retrieval_mode": "hybrid"},
                "storage_path": "data/uploads/a.md",
                "created_at": _now(),
                "updated_at": _now(),
            }
        ]
        self.chunks: list[dict] = [
            {
                "id": "c1",
                "doc_id": "doc1",
                "kb_id": "kb_labor",
                "chunk_type": "parent",
                "parent_id": None,
                "chunk_index": 0,
                "article_no": "第十九条",
                "chapter_path": "第二章 劳动合同的订立",
                "title": "中华人民共和国劳动合同法·第十九条",
                "content": "劳动合同期限三个月以上不满一年的，试用期不得超过一个月。",
                "char_count": 26,
                "in_milvus": 1,
                "is_enabled": 1,
            }
        ]
        self.qa_detached: list[str] = []
        self.commits = 0

    # -- 写入辅助（测试直接调用造数据） -------------------------------------
    def add_user(self, username: str, password_hash: str, user_id: int = 1, status: int = 1) -> dict:
        row = {"id": user_id, "username": username, "password_hash": password_hash, "status": status}
        self.users.append(row)
        return row

    def add_session(self, session_id: str, user_id: int, **kw) -> dict:
        row = {
            "id": session_id,
            "user_id": user_id,
            "title": kw.get("title", "新对话"),
            "kb_ids": kw.get("kb_ids", ["kb_labor", "kb_civil"]),
            "model": kw.get("model", "deepseek-v4.1-flash"),
            "temperature": kw.get("temperature", 0.3),
            "top_p": kw.get("top_p", 0.8),
            "max_tokens": kw.get("max_tokens", 2048),
            "history_rounds": kw.get("history_rounds", 5),
            "message_count": kw.get("message_count", 0),
            "deleted_at": kw.get("deleted_at"),
            "created_at": _now(),
            "updated_at": _now(),
        }
        self.sessions.append(row)
        return row

    def add_message(self, session_id: str, seq: int, role: str, content: str, **kw) -> dict:
        row = {
            "id": len(self.messages) + 1,
            "session_id": session_id,
            "seq": seq,
            "role": role,
            "content": content,
            "sources": kw.get("sources"),
            "intent": kw.get("intent"),
            "rewrite_type": kw.get("rewrite_type"),
            "route": kw.get("route"),
            "cached": kw.get("cached", 0),
            "latency_ms": kw.get("latency_ms"),
            "ttft_ms": kw.get("ttft_ms"),
            "token_usage": kw.get("token_usage"),
            "error": kw.get("error"),
            "created_at": _now(),
        }
        self.messages.append(row)
        return row

    # -- SQLAlchemy Session 接口 -------------------------------------------
    def execute(self, stmt, params: dict | None = None):
        sql = " ".join(str(stmt).split())
        p = params or {}
        low = sql.lower()

        # ---- users ----
        if low.startswith("select id, username, password_hash, status from users"):
            rows = [u for u in self.users if u["username"] == p.get("u")]
            return _Result(rows)
        if low.startswith("select id, username, status, tenant_id, role from users"):
            rows = [
                {
                    "id": u["id"],
                    "username": u["username"],
                    "status": u["status"],
                    "tenant_id": u.get("tenant_id", "default"),
                    "role": u.get("role", "editor"),
                }
                for u in self.users
                if u["id"] == p.get("id")
            ]
            return _Result(rows)
        if low.startswith("select id, username, status from users"):
            rows = [u for u in self.users if u["id"] == p.get("id")]
            return _Result(rows)
        if "insert into users" in low:
            uid = (max([u["id"] for u in self.users], default=0)) + 1
            self.users.append({"id": uid, "username": p.get("u"), "password_hash": p.get("p"), "status": 1})
            return _Result(lastrowid=uid)
        if low.startswith("update users set last_login_at"):
            return _Result()

        # ---- chat_sessions ----
        if low.startswith("select") and "from chat_sessions" in low:
            rows = [
                s
                for s in self.sessions
                if s["user_id"] == p.get("uid") and s["deleted_at"] is None
            ]
            if p.get("sid"):
                rows = [s for s in rows if s["id"] == p.get("sid")]
            if "order by updated_at" in low:
                rows = sorted(rows, key=lambda s: s["updated_at"], reverse=True)
            return _Result(rows)
        if low.startswith("insert into chat_sessions"):
            self.add_session(
                p["id"],
                p["uid"],
                title=p.get("title", "新对话"),
                kb_ids=_as_list(p.get("kb_ids")),
                model=p.get("model", "deepseek-v4.1-flash"),
                temperature=p.get("temperature", 0.3),
                top_p=p.get("top_p", 0.8),
                max_tokens=p.get("max_tokens", 2048),
                history_rounds=p.get("history_rounds", 5),
            )
            return _Result()
        if low.startswith("update chat_sessions set") and "deleted_at = current_timestamp" in low:
            for s in self.sessions:
                if s["id"] == p.get("sid") and s["user_id"] == p.get("uid"):
                    s["deleted_at"] = _now()
            return _Result()
        if low.startswith("update chat_sessions set"):
            for s in self.sessions:
                if s["id"] != p.get("sid") or s["user_id"] != p.get("uid"):
                    continue
                for key in ("title", "model", "temperature", "top_p", "max_tokens", "history_rounds"):
                    if f"{key} = :{key}" in low and key in p:
                        s[key] = p[key]
                if "kb_ids = cast(:kb_ids as json)" in low:
                    s["kb_ids"] = _as_list(p.get("kb_ids"))
            return _Result()

        # ---- chat_messages ----
        if low.startswith("select id, seq, role, content") and "from chat_messages" in low:
            rows = sorted(
                [m for m in self.messages if m["session_id"] == p.get("sid")],
                key=lambda m: m["seq"],
            )
            return _Result(rows[: int(p.get("limit", 100))])

        # ---- knowledge_bases ----
        if low.startswith("select id, corpus_version, index_version, is_public") and "from knowledge_bases" in low:
            return _Result(
                [
                    {
                        "id": k["id"],
                        "corpus_version": k.get("corpus_version", 1),
                        "index_version": k.get("index_version", 1),
                        "is_public": k.get("is_public", 1),
                        "tenant_id": k.get("tenant_id", "default"),
                        "owner_id": k.get("owner_id"),
                    }
                    for k in self.kbs
                ]
            )
        if low.startswith("select id, corpus_version, index_version from knowledge_bases"):
            return _Result(
                [
                    {"id": k["id"], "corpus_version": k.get("corpus_version", 1), "index_version": k.get("index_version", 1)}
                    for k in self.kbs
                ]
            )
        if low.startswith("select id, name, description") and "from knowledge_bases" in low:
            ids = p.get("ids")
            rows = self.kbs if not ids else [k for k in self.kbs if k["id"] in ids]
            return _Result(rows)
        if low.startswith("select 1 from knowledge_bases"):
            return _Result([{"id": 1}]) if any(k["id"] == p.get("id") for k in self.kbs) else _Result([])
        if "insert into knowledge_bases" in low:
            self.kbs.append(
                {
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "description": p.get("description"),
                    "embedding_model": "bge-m3",
                    "doc_count": 0,
                    "chunk_count": 0,
                    "updated_at": _now(),
                    "corpus_version": 1,
                    "index_version": 1,
                    "is_public": p.get("is_public", 0),
                    "tenant_id": p.get("tenant_id", "default"),
                    "owner_id": p.get("owner_id", 1),
                }
            )
            return _Result()
        if low.startswith("select id, name, doc_count, chunk_count from knowledge_bases"):
            return _Result([k for k in self.kbs if k["id"] == p.get("id")])
        if low.startswith("delete from knowledge_bases"):
            kb_id = p.get("id")
            self.kbs = [k for k in self.kbs if k["id"] != kb_id]
            # 真库靠外键 ON DELETE CASCADE 带走文档/块，假库这里手动模拟
            self.docs = [d for d in self.docs if d["kb_id"] != kb_id]
            self.chunks = [c for c in self.chunks if c.get("kb_id") != kb_id]
            return _Result()
        if low.startswith("update qa_pairs"):
            self.qa_detached.append(p.get("kb_id"))
            return _Result()

        # ---- documents ----
        if low.startswith("select count(*) from documents"):
            busy = sum(
                1
                for d in self.docs
                if d["kb_id"] == p.get("kb_id")
                and d["status"] in ("pending", "parsing", "chunking", "embedding")
            )
            return _Result(scalar=busy)
        if low.startswith("select id, filename") and "from documents" in low:
            return _Result([d for d in self.docs if d["kb_id"] == p.get("kb_id")])
        if low.startswith("select id, kb_id, filename") and "from documents" in low:
            rows = [d for d in self.docs if d["id"] == p.get("id")]
            return _Result(rows)
        if low.startswith("select kb_id from documents where id"):
            return _Result([d for d in self.docs if d["id"] == p.get("id")])
        if low.startswith("select 1 from documents"):
            return _Result([{"id": 1}]) if any(d["id"] == p.get("id") for d in self.docs) else _Result([])
        if low.startswith("select kb_id, storage_path, parse_config from documents"):
            rows = [d for d in self.docs if d["id"] == p.get("id")]
            return _Result(rows)
        if low.startswith("select id from documents where id in"):
            ids = set(p.get("ids") or [])
            return _Result([{"id": d["id"]} for d in self.docs if d["id"] in ids])

        # ---- document_chunks ----
        if low.startswith("select count(*) from document_chunks"):
            return _Result(scalar=len(self.chunks))
        if low.startswith("select kb_id from document_chunks where id"):
            return _Result([c for c in self.chunks if c["id"] == p.get("id")])
        if low.startswith("select id from document_chunks where kb_id"):
            return _Result([c for c in self.chunks if c.get("kb_id") == p.get("kb_id")])
        if "from document_chunks" in low and "where doc_id" in low:
            return _Result(self.chunks[: int(p.get("limit", 50))])

        # 兜底：返回空，避免测试因未知 SQL 崩掉（断言会指出真实问题）
        return _Result([])

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def close(self):
        pass


def _as_list(value):
    import json

    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def override_deps(app, db: FakeDB, user: CurrentUser | None = CurrentUser(id=1, username="alice")):
    """把 get_db / 鉴权依赖换成假件，返回还原函数。"""
    from app.core.security import get_current_user, get_current_user_optional
    from app.db.mysql import get_db

    app.dependency_overrides[get_db] = lambda: db
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_current_user_optional] = lambda: user

    def restore():
        for dep in (get_db, get_current_user, get_current_user_optional):
            app.dependency_overrides.pop(dep, None)

    return restore
