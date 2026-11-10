"""意图与路由调试接口：一眼看清"这个问题被识别成什么、走了哪条链路"。

不参与正式问答链路（正式链路在阶段 5 的 /api/chat/stream），
但与 /api/retrieval/preview 对偶：一个看检索质量，一个看路由决策。
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_access_scope
from app.core.access import AccessScope
from app.core.errors import NotFoundError
from app.core.metrics import record_permission_denied
from app.core.ratelimit import rate_limit_dep
from app.core.security import CurrentUser, get_current_user
from app.orchestration.service import analyze_query

router = APIRouter(
    prefix="/api",
    tags=["orchestration"],
    dependencies=[Depends(get_current_user), Depends(rate_limit_dep)],
)


class IntentPreviewRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    history: list[dict] = Field(default_factory=list)
    kb_ids: list[str] | None = Field(default=None, description="前端勾选的知识库，空=全部")


class IntentPreviewResponse(BaseModel):
    query: str
    intent: dict
    route: dict
    qa: dict
    sub_queries: list[str]
    degraded: bool
    elapsed_ms: int


@router.post(
    "/intent/preview",
    response_model=IntentPreviewResponse,
    summary="意图识别与路由预览（含问答库判定）",
)
def preview(
    payload: IntentPreviewRequest,
    scope: AccessScope = Depends(get_access_scope),
) -> IntentPreviewResponse:
    allowed = scope.restrict(payload.kb_ids)
    if scope.has_grants and not allowed:
        record_permission_denied("intent_preview")
        raise NotFoundError("知识库不存在")
    plan = analyze_query(
        payload.query,
        history=payload.history or None,
        selected_kb_ids=allowed,
        scope=scope,
    )
    data = plan.to_dict()
    return IntentPreviewResponse(**data)
