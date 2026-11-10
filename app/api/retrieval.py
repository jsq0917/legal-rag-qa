"""检索调试接口：直接看"问了什么、召回哪几条、各多少分"。

不参与正式问答链路（正式链路在阶段 5 的 /api/chat/stream），
但它是调参与答辩演示的关键窗口：能直观展示 RRF 融合、重排前后的排序变化。
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
from app.retrieval.service import retrieve
from app.security.guardrails import safe_source_file

router = APIRouter(
    prefix="/api",
    tags=["kbs"],
    dependencies=[Depends(get_current_user), Depends(rate_limit_dep)],
)


class PreviewRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    kb_ids: list[str] | None = None
    complexity: Literal["simple", "medium", "complex"] = "medium"
    topk: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["dense", "hybrid"] = "hybrid"
    use_rerank: bool = True


@router.post("/retrieval/preview", summary="检索预览（含融合分与重排分）")
def preview(
    payload: PreviewRequest,
    scope: AccessScope = Depends(get_access_scope),
) -> dict:
    allowed = scope.restrict(payload.kb_ids)
    if scope.has_grants and not allowed:
        record_permission_denied("retrieval_preview")
        raise NotFoundError("知识库不存在")
    result = retrieve(
        payload.question,
        kb_ids=allowed or None,
        complexity=payload.complexity,
        topk=payload.topk,
        mode=payload.mode,
        use_rerank=payload.use_rerank,
    )
    return {
        "question": result.question,
        "complexity": result.complexity,
        "topk": result.topk,
        "pool": result.pool,
        "mode": result.mode,
        "reranked": result.reranked,
        "elapsed_ms": result.elapsed_ms,
        "debug": result.debug,
        "hits": [
            {
                "chunk_id": h.chunk_id,
                "kb_id": h.kb_id,
                "law_name": h.law_name,
                "article_no": h.article_no,
                "chapter_path": h.chapter_path,
                "title": h.title,
                "score": round(h.score, 6),
                "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 4),
                "content": h.content,
                "parent_content": h.parent_content,
                # 调试接口也不回传服务器路径/随机前缀，只给 basename
                "source_file": safe_source_file(h.source_file),
            }
            for h in result.hits
        ],
    }
