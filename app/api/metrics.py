"""Prometheus 指标端点（``GET /metrics``）。

不鉴权：与 ``/api/health`` 同级，只暴露聚合计数与"依赖是否可用"，
不含密钥、请求正文、SQL 或文档内容。需要限制访问时在网关层做 IP 白名单。
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.core.metrics import (
    check_disk_watermark,
    collect_dependencies,
    disk_snapshot,
    render_prometheus,
)
from config import get_settings

router = APIRouter(tags=["meta"])

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get("/metrics", summary="Prometheus 指标", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    settings = get_settings()
    dependencies = await collect_dependencies()
    disk = disk_snapshot(settings)
    exceeded = check_disk_watermark(disk, settings)
    return PlainTextResponse(
        render_prometheus(dependencies, disk, exceeded),
        media_type=PROMETHEUS_CONTENT_TYPE,
    )
