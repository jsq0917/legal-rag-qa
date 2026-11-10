"""链路 trace 查询（LA-O4）。

按 ``trace_id`` 还原一次问答的完整阶段顺序与各段成本。trace 里只有
阶段名、耗时、模型、Prompt/语料/索引版本、token 与降级原因，不含问题、
回答正文或密钥；仍按"只能看自己的请求"收紧：非本人且非管理员一律 404，
避免用 trace_id 探测他人请求是否存在。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.access import ROLE_ADMIN
from app.core.errors import NotFoundError
from app.core.security import CurrentUser, get_current_user
from app.core.trace import load_trace

router = APIRouter(prefix="/api", tags=["trace"])


@router.get("/traces/{trace_id}", summary="按 trace_id 还原一次请求的调用链")
def get_trace(trace_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    data = load_trace(trace_id)
    if data is None:
        raise NotFoundError("trace 不存在或已过期")
    owner = data.get("user_id")
    if owner != user.id and user.role != ROLE_ADMIN:
        # 与"不存在"同响应，防止跨用户探测
        raise NotFoundError("trace 不存在或已过期")
    return data
