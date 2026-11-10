"""统一异常体系与全局异常处理。

前端只需要认一种错误结构::

    {"code": "not_found", "message": "会话不存在", "request_id": "ab12...", "detail": null}

这样前端错误提示、后端日志排查都只对接一处。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logger import current_request_id, get_logger
from app.security.guardrails import redact_sensitive

logger = get_logger("lawanswer.error")


class AppError(Exception):
    """业务异常基类。"""

    code: str = "app_error"
    status_code: int = 500
    message: str = "服务内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        detail: Any = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.detail = detail
        super().__init__(self.message)


class ValidationFailed(AppError):
    code = "validation_failed"
    status_code = 422
    message = "参数校验失败"


class AuthError(AppError):
    code = "unauthorized"
    status_code = 401
    message = "未登录或登录已过期"


class PermissionDenied(AppError):
    code = "forbidden"
    status_code = 403
    message = "无权访问该资源"


class NotFoundError(AppError):
    code = "not_found"
    status_code = 404
    message = "资源不存在"


class ConflictError(AppError):
    code = "conflict"
    status_code = 409
    message = "资源冲突"


class RateLimited(AppError):
    code = "too_many_requests"
    status_code = 429
    message = "请求过于频繁，请稍后再试"


class UpstreamError(AppError):
    code = "upstream_error"
    status_code = 502
    message = "上游服务不可用"


def _redact_value(value: Any) -> Any:
    """错误响应递归脱敏：异常消息/详情里可能夹带路径、连接串或密钥。"""
    if isinstance(value, str):
        return redact_sensitive(value)[0]
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    return value


def _envelope(code: str, message: str, detail: Any = None) -> dict[str, Any]:
    return {
        "code": code,
        "message": _redact_value(str(message)),
        "detail": _redact_value(detail),
        "request_id": current_request_id(),
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("%s %s -> %s: %s", request.method, request.url.path, exc.code, exc.message)
        else:
            logger.warning("%s %s -> %s: %s", request.method, request.url.path, exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.detail),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found"}.get(
            exc.status_code, "http_error"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(code, str(exc.detail), None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # 只回传精简信息，避免把内部字段结构全部暴露给前端
        errors = [
            {"field": ".".join(str(p) for p in e.get("loc", [])), "msg": e.get("msg", "")}
            for e in exc.errors()
        ]
        logger.warning("%s %s 参数校验失败: %s", request.method, request.url.path, errors)
        return JSONResponse(
            status_code=422,
            content=_envelope("validation_failed", "参数校验失败", errors),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常 %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_envelope("internal_error", "服务内部错误，请稍后重试"),
        )
