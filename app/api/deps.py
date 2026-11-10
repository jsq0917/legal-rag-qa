"""API 依赖：把"登录用户"解析成"服务端授权范围"。

所有资源型接口都应通过本依赖拿 :class:`AccessScope`，不要在路由里各自
拼 ``kb_ids``；否则很容易出现"列表过滤了、详情没过滤"的旁路。
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.access import AccessScope, resolve_access_scope
from app.core.security import CurrentUser, get_current_user
from app.db.mysql import get_db


def get_access_scope(
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> AccessScope:
    return resolve_access_scope(db, user)


__all__ = ["get_access_scope"]
