"""对话会话与消息历史（设计方案 §2.2、§2.3、§4.2 P3）。

::

    GET    /api/sessions                     会话列表（按最近活跃排序，不含软删）
    POST   /api/sessions                     新建会话（返回完整会话对象）
    PATCH  /api/sessions/{id}                重命名 / 改参数 / 改知识库
    DELETE /api/sessions/{id}                软删除
    GET    /api/sessions/{id}/messages       消息历史（升序，供刷新后恢复对话）

**归属隔离**：所有查询都带 ``user_id`` 条件；访问他人会话一律 404（不暴露是否存在）。
参数（model/temperature/top_p/max_tokens/history_rounds）与会话绑定，
前端切会话时回填——这是 P3 参数抽屉的数据源。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_access_scope
from app.core.access import AccessScope
from app.core.errors import NotFoundError
from app.core.ids import new_ulid
from app.core.logger import get_logger
from app.core.security import CurrentUser, get_current_user
from app.db.mysql import get_db

logger = get_logger("lawanswer.sessions")

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

#: 会话字段（列表与详情共用，避免 SELECT * ）
_SESSION_COLUMNS = (
    "id, title, kb_ids, model, temperature, top_p, max_tokens, history_rounds, "
    "message_count, created_at, updated_at"
)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class SessionCreate(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=128)
    kb_ids: list[str] = Field(default_factory=lambda: ["kb_labor", "kb_civil"], max_length=20)
    model: str = Field(default="deepseek-v4.1-flash", max_length=64)
    temperature: float = Field(default=0.3, ge=0.0, le=1.5)
    top_p: float = Field(default=0.8, ge=0.1, le=1.0)
    max_tokens: int = Field(default=2048, ge=256, le=8192)
    history_rounds: int = Field(default=5, ge=0, le=20)


class SessionUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=128)
    kb_ids: list[str] | None = Field(default=None, max_length=20)
    model: str | None = Field(default=None, max_length=64)
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    top_p: float | None = Field(default=None, ge=0.1, le=1.0)
    max_tokens: int | None = Field(default=None, ge=256, le=8192)
    history_rounds: int | None = Field(default=None, ge=0, le=20)


class SessionOut(BaseModel):
    id: str
    title: str
    kb_ids: list[str]
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    history_rounds: int
    message_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# 数据访问
# ---------------------------------------------------------------------------
def _as_kb_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value)
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def _iso(value: Any) -> str | None:
    return value.isoformat(sep=" ", timespec="seconds") if value else None


def _to_session(row: dict) -> SessionOut:
    return SessionOut(
        id=row["id"],
        title=row["title"],
        kb_ids=_as_kb_ids(row.get("kb_ids")),
        model=row["model"],
        temperature=float(row["temperature"]),
        top_p=float(row["top_p"]),
        max_tokens=int(row["max_tokens"]),
        history_rounds=int(row["history_rounds"]),
        message_count=int(row.get("message_count") or 0),
        created_at=_iso(row.get("created_at")),
        updated_at=_iso(row.get("updated_at")),
    )


def get_owned_session(db: Session, session_id: str, user_id: int) -> SessionOut:
    """取当前用户的会话；不存在/已删/他人会话 → 404（不泄露存在性）。"""
    row = db.execute(
        text(
            f"SELECT {_SESSION_COLUMNS} FROM chat_sessions "
            "WHERE id = :sid AND user_id = :uid AND deleted_at IS NULL"
        ),
        {"sid": session_id, "uid": user_id},
    ).mappings().first()
    if row is None:
        raise NotFoundError("会话不存在")
    return _to_session(dict(row))


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@router.get("", summary="会话列表（按最近活跃排序）")
def list_sessions(user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)) -> list[SessionOut]:
    rows = db.execute(
        text(
            f"SELECT {_SESSION_COLUMNS} FROM chat_sessions "
            "WHERE user_id = :uid AND deleted_at IS NULL "
            "ORDER BY updated_at DESC LIMIT 100"
        ),
        {"uid": user.id},
    ).mappings().all()
    return [_to_session(dict(r)) for r in rows]


@router.post("", status_code=201, response_model=SessionOut, summary="新建会话")
def create_session(
    payload: SessionCreate,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    scope: AccessScope = Depends(get_access_scope),
) -> SessionOut:
    kb_ids = scope.restrict(payload.kb_ids)
    if scope.has_grants and payload.kb_ids and not kb_ids:
        raise NotFoundError("知识库不存在")
    session_id = new_ulid()
    db.execute(
        text(
            "INSERT INTO chat_sessions "
            "(id, user_id, title, kb_ids, model, temperature, top_p, max_tokens, history_rounds) "
            "VALUES (:id, :uid, :title, CAST(:kb_ids AS JSON), :model, :temperature, :top_p, :max_tokens, :history_rounds)"
        ),
        {
            "id": session_id,
            "uid": user.id,
            "title": payload.title.strip() or "新对话",
            "kb_ids": json.dumps(kb_ids, ensure_ascii=False),
            "model": payload.model,
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "max_tokens": payload.max_tokens,
            "history_rounds": payload.history_rounds,
        },
    )
    db.commit()
    logger.info("新建会话 user=%s session=%s", user.id, session_id)
    return get_owned_session(db, session_id, user.id)


@router.patch("/{session_id}", response_model=SessionOut, summary="重命名 / 修改参数")
def update_session(
    session_id: str,
    payload: SessionUpdate,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    scope: AccessScope = Depends(get_access_scope),
) -> SessionOut:
    get_owned_session(db, session_id, user.id)  # 归属校验（404）
    if payload.kb_ids is not None:
        allowed = scope.restrict(payload.kb_ids)
        if scope.has_grants and payload.kb_ids and not allowed:
            raise NotFoundError("知识库不存在")
        payload = payload.model_copy(update={"kb_ids": allowed})

    fields: list[str] = []
    params: dict[str, Any] = {"sid": session_id, "uid": user.id}
    if payload.title is not None:
        fields.append("title = :title")
        params["title"] = payload.title.strip() or "新对话"
    if payload.kb_ids is not None:
        fields.append("kb_ids = CAST(:kb_ids AS JSON)")
        params["kb_ids"] = json.dumps(payload.kb_ids, ensure_ascii=False)
    for key in ("model", "temperature", "top_p", "max_tokens", "history_rounds"):
        value = getattr(payload, key)
        if value is not None:
            fields.append(f"{key} = :{key}")
            params[key] = value

    if fields:
        db.execute(
            text(
                f"UPDATE chat_sessions SET {', '.join(fields)} "
                "WHERE id = :sid AND user_id = :uid AND deleted_at IS NULL"
            ),
            params,
        )
        db.commit()
    return get_owned_session(db, session_id, user.id)


@router.delete("/{session_id}", summary="删除会话（软删）")
def delete_session(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    get_owned_session(db, session_id, user.id)
    db.execute(
        text(
            "UPDATE chat_sessions SET deleted_at = CURRENT_TIMESTAMP(3) "
            "WHERE id = :sid AND user_id = :uid AND deleted_at IS NULL"
        ),
        {"sid": session_id, "uid": user.id},
    )
    db.commit()
    logger.info("删除会话 user=%s session=%s", user.id, session_id)
    return {"deleted": session_id}


@router.get("/{session_id}/messages", summary="消息历史（升序）")
def list_messages(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    get_owned_session(db, session_id, user.id)
    rows = db.execute(
        text(
            "SELECT id, seq, role, content, sources, intent, rewrite_type, route, cached, "
            "latency_ms, ttft_ms, token_usage, error, created_at "
            "FROM chat_messages WHERE session_id = :sid ORDER BY seq ASC LIMIT :limit"
        ),
        {"sid": session_id, "limit": limit},
    ).mappings().all()
    items = []
    for row in rows:
        data = dict(row)
        for key in ("sources", "token_usage"):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    data[key] = None
        data["cached"] = bool(data.get("cached"))
        data["created_at"] = _iso(data.get("created_at"))
        items.append(data)
    return {"session_id": session_id, "count": len(items), "items": items}
