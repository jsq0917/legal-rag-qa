"""健康检查接口。

判据：三个依赖全部 ok 才算 ``ok``，任一不可用降级为 ``degraded``（HTTP 仍返回 200）。
前端/运维据此提示"向量库暂不可用，本次仅使用高频问答"，而不是整站白屏。
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter

from app.core import gpu_guard
from app.core.logger import current_request_id
from app.db.milvus_client import check_milvus
from app.db.mysql import check_mysql
from app.db.redis_client import check_redis
from config import get_llm_api_key, get_llm_api_key_env, get_settings

router = APIRouter(tags=["meta"])

SERVICE_TIMEOUT_S = 8


async def _run_check(fn) -> dict:
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn), timeout=SERVICE_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {
            "status": "down",
            "latency_ms": SERVICE_TIMEOUT_S * 1000,
            "detail": None,
            "error": "TimeoutError: 健康检查超时",
        }


@router.get("/health", summary="依赖健康检查")
async def health() -> dict:
    settings = get_settings()
    mysql, redis, milvus = await asyncio.gather(
        _run_check(check_mysql),
        _run_check(check_redis),
        _run_check(check_milvus),
    )
    services = {"mysql": mysql, "redis": redis, "milvus": milvus}
    healthy = all(item["status"] == "ok" for item in services.values())
    return {
        "status": "ok" if healthy else "degraded",
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
        "time": datetime.now().isoformat(timespec="seconds"),
        "request_id": current_request_id(),
        # 只暴露"是否配置"与来源变量名，绝不返回密钥内容
        "llm_key_configured": bool(get_llm_api_key()),
        "llm_key_source": get_llm_api_key_env() or None,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_default_model,
        # GPU 粘性故障后进程级降级到 CPU；响应显式标注，避免把 CPU 延迟误判为退化
        "inference_device": "fallback" if gpu_guard.is_poisoned() else "primary",
        "services": services,
    }


@router.get("/health/live", summary="存活探针（不探测依赖）")
async def liveness() -> dict:
    return {"status": "alive"}
