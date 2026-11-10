"""问答 SSE 端点：``POST /api/chat/stream``（设计方案 §6.5）。

事件协议::

    event: meta     data: {message_id, intent, route, rewrite_type, complexity, cached, degraded}
    event: stage    data: {stage: intent|retrieval|rerank|generate}
    event: sources  data: {sources: [{law, chapter_path, article_no, score, snippet, kind}]}
    event: delta    data: {text: "..."}
    event: done     data: {ttft_ms, latency_ms, token_usage, sources_count, cached}
    event: error    data: {code, message, retriable}

工程要点：
* ``AnswerEngine`` 是同步生成器，用 ``iterate_in_threadpool`` 包装成异步流，
  不阻塞事件循环；
* 客户端断开（AbortController）→ 异步任务被取消 → 关闭同步生成器 →
  ``AnswerEngine`` 的 ``finally`` 照常落库、不发 error（§6.5 容错第 4 层）；
* ``session_id`` 可选：传了且会话存在才落库（阶段 6 接入登录/会话后即为默认路径）。

另有 ``POST /api/chat/suggestions``：回答结束后预测 3 条追问。
它**刻意不并入 SSE 流**——并进去会拖慢 ``done`` 事件（用户看到答案却还在等流，
输入框一直锁着），且预测失败会污染主链路；独立端点失败只返回空列表。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.responses import StreamingResponse

from app.api.sessions import get_owned_session
from app.core.access import AccessScope, resolve_access_scope
from app.core.ratelimit import check_rate_limit
from app.core.security import CurrentUser, TokenError, get_current_user_optional
from app.core.sse import SSE_HEADERS, SSE_MEDIA_TYPE
from app.db.mysql import session_scope
from app.generation.stream import AnswerEngine
from app.generation.suggestions import generate_suggestions

router = APIRouter(prefix="/api", tags=["chat"])


class ChatStreamRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list, description="最近对话 [{role, content}]")
    kb_ids: list[str] | None = Field(default=None, description="前端勾选的知识库，空=全部")
    session_id: str | None = Field(default=None, description="会话 ID（ULID）；传了才落库")


def prepare_scope(
    user: CurrentUser | None,
    session_id: str | None,
    kb_ids: list[str] | None,
) -> AccessScope:
    """短连接内完成"会话归属 + 授权范围"解析；SSE 流期间不占 MySQL 连接。

    单独抽成模块级函数，便于测试注入；生产路径始终走数据库解析。
    """
    with session_scope() as db:
        if session_id:
            get_owned_session(db, session_id, user.id)  # type: ignore[union-attr]
        return resolve_access_scope(db, user, kb_ids)


@router.post("/chat/stream", summary="SSE 流式问答（六段 Prompt + 双源路由）")
async def chat_stream(
    payload: ChatStreamRequest,
    request: Request,
    user: CurrentUser | None = Depends(get_current_user_optional),
) -> Any:
    # 会话归属校验 + 授权范围解析共用一次短连接。
    # 客户端传入的 kb_ids 只作为"缩小范围"的请求，最终范围由服务端解析。
    if payload.session_id and user is None:
        raise TokenError("会话模式需要登录")

    scope = await run_in_threadpool(prepare_scope, user, payload.session_id, payload.kb_ids)

    # 限流：匿名按 IP 20 次/分，登录按 uid 60 次/分（防 LLM 费用被刷爆）。
    # Redis 是同步客户端，必须入线程池——直接调用会在 Redis 抖动时（最坏 5s）
    # 冻结整个事件循环，殃及所有请求。
    await run_in_threadpool(
        check_rate_limit, request, user_id=user.id if user else None
    )

    engine = AnswerEngine()
    frames = engine.stream(
        payload.query,
        history=payload.history or None,
        selected_kb_ids=payload.kb_ids,
        session_id=payload.session_id,
        scope=scope,
    )

    async def event_stream():
        try:
            async for frame in iterate_in_threadpool(frames):
                if await request.is_disconnected():
                    frames.close()  # 触发 AnswerEngine.finally → 落库
                    return
                yield frame
        finally:
            frames.close()

    return StreamingResponse(event_stream(), media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS)


# ---------------------------------------------------------------------------
# 追问预测（问题推荐）
# ---------------------------------------------------------------------------
class SuggestionsRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="用户刚问的问题")
    answer: str = Field(min_length=1, max_length=8000, description="助手刚给出的回答")
    law_names: list[str] = Field(default_factory=list, description="回答涉及的法律名（可选，用于收敛方向）")


class SuggestionsResponse(BaseModel):
    suggestions: list[str] = Field(default_factory=list, description="最多 3 条；预测失败为空列表")


@router.post(
    "/chat/suggestions",
    response_model=SuggestionsResponse,
    summary="追问预测（回答后推荐 3 个可能的下一个问题）",
)
async def chat_suggestions(
    payload: SuggestionsRequest,
    request: Request,
    user: CurrentUser | None = Depends(get_current_user_optional),
) -> SuggestionsResponse:
    """**不挂在 SSE 流里**：回答渲染完成后前端另行调用。

    这样预测慢几秒也不会拖住 ``done`` 事件（否则输入框会被一直锁着）；
    预测失败只返回空列表，前端不展示推荐区，**不影响任何主链路**。
    """
    await run_in_threadpool(
        check_rate_limit, request, user_id=user.id if user else None
    )
    items = await run_in_threadpool(
        generate_suggestions,
        payload.query,
        payload.answer,
        law_names=payload.law_names or None,
    )
    return SuggestionsResponse(suggestions=items)

