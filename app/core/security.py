"""密码哈希与 JWT 鉴权（设计方案 §2.1 / §4.2 P1）。

约定：
* 密码用 **bcrypt cost=12**（自带盐，不存明文也不存可逆哈希）；
* JWT 用 HS256 + ``JWT_SECRET``，payload 只放 ``sub``（用户 id，字符串）、
  ``username``、``exp``；不放权限位——权限变化需要即时生效时以 DB 为准；
* ``get_current_user`` 是 FastAPI 依赖：解析 ``Authorization: Bearer <token>``，
  用户不存在/被禁用一律 401（不泄露"用户是否存在"）。

安全细节：
* 登录失败统一提示"账号或密码错误"，不区分账号不存在与密码错误；
* 密码哈希在校验失败时也做一次"哑算"，避免用响应时间枚举账号。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.errors import AuthError
from app.core.logger import get_logger
from app.db.mysql import get_db
from config import Settings, get_settings

logger = get_logger("lawanswer.security")

#: bcrypt 成本因子（§2.1 规定 12）
BCRYPT_ROUNDS = 12

#: 用户名规则：3–32 位字母数字下划线
USERNAME_PATTERN = r"^[A-Za-z0-9_]{3,32}$"

#: 密码规则：≥8 位，且同时含字母与数字
PASSWORD_MIN_LEN = 8
PASSWORD_MAX_LEN = 64


class TokenError(AuthError):
    code = "unauthorized"
    status_code = 401
    message = "未登录或登录已过期"


@dataclass(frozen=True)
class CurrentUser:
    """已认证用户（依赖注入到路由）。"""

    id: int
    username: str
    status: int = 1
    tenant_id: str = "default"
    role: str = "editor"

    def to_dict(self) -> dict:
        return {"id": self.id, "username": self.username}


# ---------------------------------------------------------------------------
# 密码
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """bcrypt 哈希（cost=12）。"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """校验密码；任何异常（哈希格式损坏）都当失败处理。"""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def dummy_verify() -> None:
    """账号不存在时也做一次等价开销的哈希校验，防响应时间侧信道枚举账号。"""
    verify_password("dummy-password-for-timing", hash_password("dummy-password-for-timing"))


def validate_username(username: str) -> str:
    """校验并返回规范化用户名（去空白）。不合规抛 :class:`AuthError`（422）。"""
    import re

    u = (username or "").strip()
    if not re.match(USERNAME_PATTERN, u):
        raise AuthError("账号需为 3–32 位字母、数字或下划线", status_code=422)
    return u


def validate_password(password: str) -> None:
    """密码强度：≥8 位且同时含字母与数字。"""
    if not (PASSWORD_MIN_LEN <= len(password or "") <= PASSWORD_MAX_LEN):
        raise AuthError(f"密码长度需为 {PASSWORD_MIN_LEN}–{PASSWORD_MAX_LEN} 位", status_code=422)
    has_alpha = any(ch.isalpha() for ch in password)
    has_digit = any(ch.isdigit() for ch in password)
    if not (has_alpha and has_digit):
        raise AuthError("密码需同时包含字母和数字", status_code=422)


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------
def create_access_token(
    user_id: int,
    username: str,
    *,
    settings: Settings | None = None,
    expires_hours: int | None = None,
) -> str:
    """签发 access token。"""
    s = settings or get_settings()
    hours = s.jwt_expire_hours if expires_hours is None else expires_hours
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + hours * 3600,
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_access_token(token: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """解析 token；过期/签名错/结构错统一抛 :class:`TokenError`。"""
    s = settings or get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("登录已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("登录凭证无效") from exc
    if not payload.get("sub"):
        raise TokenError("登录凭证缺少用户标识")
    return payload


# ---------------------------------------------------------------------------
# 依赖
# ---------------------------------------------------------------------------
def _extract_bearer(request: Request) -> str | None:
    header = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _load_user(db: Session, user_id: int) -> CurrentUser | None:
    row = db.execute(
        text("SELECT id, username, status FROM users WHERE id = :id"),
        {"id": user_id},
    ).mappings().first()
    if row is None:
        return None
    return CurrentUser(id=int(row["id"]), username=row["username"], status=int(row["status"]))


def get_current_user(request: Request, db: Session = Depends(get_db)) -> CurrentUser:
    """必须登录的依赖：无 token/无效 token/用户禁用 → 401。"""
    token = _extract_bearer(request)
    if not token:
        raise TokenError("请先登录")
    payload = decode_access_token(token)
    try:
        user_id = int(payload["sub"])
    except (TypeError, ValueError) as exc:
        raise TokenError("登录凭证无效") from exc
    user = _load_user(db, user_id)
    if user is None or user.status != 1:
        raise TokenError("账号不存在或已被禁用")
    return user


def get_current_user_optional(request: Request, db: Session = Depends(get_db)) -> CurrentUser | None:
    """可选登录的依赖（chat 无状态模式下不强制登录）。"""
    token = _extract_bearer(request)
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        user_id = int(payload["sub"])
    except (TokenError, TypeError, ValueError):
        return None
    user = _load_user(db, user_id)
    return user if user and user.status == 1 else None
