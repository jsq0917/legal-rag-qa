"""FastAPI 应用装配。

启动顺序：加载配置 → 安全自检（密钥不得出现在 .env）→ 初始化日志 → 建运行时目录
         → 注册中间件/异常处理/路由。

注意：**依赖不可用不阻止启动**（MySQL/Milvus 没起来也能起服务，由 /api/health 暴露），
只有"配置非法/密钥写错位置"这类错误才拒绝启动。
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.intent import router as intent_router
from app.api.knowledge import router as knowledge_router
from app.api.metrics import router as metrics_router
from app.api.retrieval import router as retrieval_router
from app.api.sessions import router as sessions_router
from app.core.errors import register_exception_handlers
from app.core.logger import get_logger, setup_logging, shutdown_logging
from app.core.metrics import MetricsMiddleware, check_disk_watermark
from app.core.middleware import RequestIDMiddleware
from app.db.milvus_client import close_milvus
from app.db.mysql import dispose_engines
from app.db.redis_client import close_redis
from config import (
    Settings,
    assert_secrets_not_in_dotenv,
    get_llm_api_key,
    get_llm_api_key_env,
    get_settings,
    mask_secret,
)

logger = get_logger("lawanswer.app")

OPENAPI_TAGS = [
    {"name": "meta", "description": "健康检查与元信息"},
    {"name": "auth", "description": "注册、登录、图形验证码"},
    {"name": "sessions", "description": "对话会话管理"},
    {"name": "chat", "description": "SSE 流式问答"},
    {"name": "kbs", "description": "知识库与文档"},
    {"name": "chunks", "description": "文档块预览、编辑、删除"},
    {"name": "orchestration", "description": "意图识别与路由预览"},
]


def _startup_checks(settings: Settings) -> None:
    """启动自检：配置非法就不要带病启动。"""
    assert_secrets_not_in_dotenv()
    settings.ensure_dirs()
    if not get_llm_api_key():
        logger.warning(
            "未检测到大模型密钥（LLM_API_KEY / DEEPSEEK_API_KEY / DASHSCOPE_API_KEY）："
            "检索链路可用，但问答生成会失败。请设置环境变量后重开终端。"
        )
    else:
        logger.info(
            "大模型密钥已配置（来源 %s，base_url=%s，model=%s）：%s",
            get_llm_api_key_env(),
            settings.llm_base_url,
            settings.llm_default_model,
            mask_secret(get_llm_api_key()),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(
        log_dir=settings.log_path,
        level=settings.log_level,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    # 调大 anyio 线程池容量：SSE 每帧经 iterate_in_threadpool 抢线程，
    # 默认 40 在并发 20+ 时会被长占的意图/生成调用耗尽，殃及所有端点。
    import anyio.to_thread

    anyio.to_thread.current_default_thread_limiter().total_tokens = settings.server_thread_tokens
    logger.info("线程池容量：%d（anyio to_thread）", settings.server_thread_tokens)
    logger.info(
        "启动 %s v%s (env=%s, python=%s)",
        settings.app_name,
        settings.app_version,
        settings.app_env,
        ".".join(map(str, sys.version_info[:3])),
    )
    _startup_checks(settings)
    check_disk_watermark(settings=settings)
    logger.info(
        "依赖地址 mysql=%s:%s/%s redis=%s:%s/%s milvus=%s:%s/%s",
        settings.mysql_host, settings.mysql_port, settings.mysql_db,
        settings.redis_host, settings.redis_port, settings.redis_db,
        settings.milvus_host, settings.milvus_port, settings.milvus_db,
    )
    try:
        yield
    finally:
        logger.info("正在关闭：释放 MySQL 连接池 / Redis / Milvus 客户端")
        dispose_engines()
        close_redis()
        close_milvus()
        shutdown_logging()
        logging.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.app_version,
        description="面向法律条文的多路 RAG 问答系统（SSE 流式输出）",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    register_exception_handlers(app)
    app.include_router(metrics_router)
    app.include_router(health_router, prefix="/api")
    app.include_router(knowledge_router)
    app.include_router(retrieval_router)
    app.include_router(intent_router)
    app.include_router(auth_router)
    app.include_router(sessions_router)
    app.include_router(chat_router)

    @app.get("/", tags=["meta"], summary="服务信息")
    async def root() -> dict:
        return {
            "app": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
