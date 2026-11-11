"""进程内轻量指标：Prometheus 文本格式，不引入 prometheus_client。

只保留排障真正会用到的六类信号：

* HTTP 请求数与耗时（按路由模板，避免 URL 里的 id 把标签基数打爆）
* LLM 调用数（按模型/结果；重试算多次，对应真实请求次数）
* 限流 429（按来源：问答限流 / ingest 队列）
* 依赖健康（MySQL / Redis / Milvus，带 10s 缓存）
* 磁盘水位（数据盘使用率、剩余空间、是否越过阈值）

刻意不做直方图分桶与多进程聚合：单实例部署下计数器的价值远高于实现复杂度，
需要分位数时用日志/评估脚本算，不在进程内维护。
"""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from collections import defaultdict
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logger import get_logger
from config import Settings, get_settings

logger = get_logger("lawanswer.metrics")

_LOCK = threading.Lock()

#: (method, path, status) -> count
_HTTP_REQUESTS: dict[tuple[str, str, str], int] = defaultdict(int)
#: (method, path) -> seconds
_HTTP_DURATION_SUM: dict[tuple[str, str], float] = defaultdict(float)
_HTTP_DURATION_COUNT: dict[tuple[str, str], int] = defaultdict(int)
#: (model, outcome) -> count
_LLM_CALLS: dict[tuple[str, str], int] = defaultdict(int)
#: (model, kind) -> count（统一错误分类，kind 见 llm_client.LLM_ERROR_KINDS）
_LLM_FAILURES: dict[tuple[str, str], int] = defaultdict(int)
#: reason -> count
_RATE_LIMITED: dict[str, int] = defaultdict(int)
#: resource -> count（越权拒绝；不含具体资源 id，避免日志/指标泄露）
_PERMISSION_DENIED: dict[str, int] = defaultdict(int)
#: kind -> count（输出护栏命中：api_key/internal_path/system_prompt 等）
_GUARDRAIL_HITS: dict[str, int] = defaultdict(int)
#: reason -> count（GPU 粘性故障降级：异常类型）
_GPU_FALLBACKS: dict[str, int] = defaultdict(int)

DEPENDENCY_NAMES = ("mysql", "redis", "milvus")
_DEP_TIMEOUT_S = 2.0
_DEP_CACHE_TTL_S = 10.0
_dep_lock = threading.Lock()
_dep_cached_at = 0.0
_dep_cached: dict[str, dict] = {}

_disk_lock = threading.Lock()
_disk_alerted = False


# ---------------------------------------------------------------------------
# 记录
# ---------------------------------------------------------------------------
def record_http_request(method: str, path: str, status: int, duration_s: float) -> None:
    key = (method.upper(), path, str(status))
    with _LOCK:
        _HTTP_REQUESTS[key] += 1
        _HTTP_DURATION_SUM[(method.upper(), path)] += max(0.0, float(duration_s))
        _HTTP_DURATION_COUNT[(method.upper(), path)] += 1


def record_llm_failure(model: str, kind: str) -> None:
    """LLM 失败按统一错误分类计数（timeout/rate_limit/auth/…/circuit_open）。"""
    _LLM_FAILURES[(model, kind)] = _LLM_FAILURES.get((model, kind), 0) + 1


def record_llm_call(model: str, outcome: str) -> None:
    """``outcome``：success / error（每次重试各算一次真实请求）。"""
    with _LOCK:
        _LLM_CALLS[(model or "unknown", outcome)] += 1


def record_rate_limited(reason: str) -> None:
    """``reason``：chat（问答限流）/ ingest（解析队列满）。"""
    with _LOCK:
        _RATE_LIMITED[reason] += 1


def record_guardrail_hit(kind: str) -> None:
    with _LOCK:
        _GUARDRAIL_HITS[kind or "unknown"] += 1


def record_permission_denied(resource: str) -> None:
    """``resource``：kb_write / document_write / chunk_write / debug_preview 等。"""
    with _LOCK:
        _PERMISSION_DENIED[resource or "unknown"] += 1


def record_gpu_fallback(reason: str) -> None:
    """GPU 粘性故障降级到 CPU；``reason`` 为异常类型名。"""
    with _LOCK:
        _GPU_FALLBACKS[reason or "unknown"] += 1


# ---------------------------------------------------------------------------
# 中间件
# ---------------------------------------------------------------------------
class MetricsMiddleware(BaseHTTPMiddleware):
    """请求结束（含 SSE 流式响应体发送完毕）后记录计数与耗时。"""

    async def dispatch(self, request: Request, call_next) -> Response:
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            record_http_request(
                request.method,
                _route_template(request),
                status,
                time.perf_counter() - started,
            )


def _route_template(request: Request) -> str:
    """把实际路径还原成"带路由前缀的模板"，作为指标标签。

    本项目的路由挂在子 ``APIRouter`` 上，命中后 ``scope["route"].path`` 不含
    ``include_router`` 的 ``/api`` 前缀；直接用它会让不同前缀下的同名路径
    混在一起。这里按段数把实际路径的前缀补回去，同时用 ``{param}`` 代替
    具体 id，避免每个文档 id 都生成一条时间序列。
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if not template:
        return "unmatched"
    actual_segments = request.url.path.split("/")
    template_segments = template.split("/")
    if len(actual_segments) < len(template_segments):
        return template
    prefix = "/".join(actual_segments[: len(actual_segments) - len(template_segments) + 1])
    if template == "/":
        return "/"
    return prefix + template


# ---------------------------------------------------------------------------
# 依赖健康
# ---------------------------------------------------------------------------
async def collect_dependencies() -> dict[str, dict]:
    """探测三个依赖；10 秒内复用上次结果，避免每次抓取都打一遍连接。"""
    global _dep_cached_at, _dep_cached
    now = time.monotonic()
    with _dep_lock:
        if _dep_cached and now - _dep_cached_at < _DEP_CACHE_TTL_S:
            return {name: dict(info) for name, info in _dep_cached.items()}

    result = await _probe_dependencies()
    with _dep_lock:
        _dep_cached = result
        _dep_cached_at = time.monotonic()
    return {name: dict(info) for name, info in result.items()}


def _dependency_checks() -> dict[str, Callable[[], dict]]:
    # 延迟导入：/metrics 的调用方（含单测）可以替换 app.db.* 里的探测函数
    from app.db.milvus_client import check_milvus
    from app.db.mysql import check_mysql
    from app.db.redis_client import check_redis

    return {"mysql": check_mysql, "redis": check_redis, "milvus": check_milvus}


async def _probe_dependencies() -> dict[str, dict]:
    checks = _dependency_checks()

    async def one(name: str, fn: Callable[[], dict]) -> dict:
        started = time.perf_counter()
        try:
            info = await asyncio.wait_for(asyncio.to_thread(fn), timeout=_DEP_TIMEOUT_S)
        except asyncio.TimeoutError:
            return {"up": False, "latency_s": _DEP_TIMEOUT_S, "error": "timeout"}
        except Exception as exc:  # noqa: BLE001 - 指标端点不能因探测异常而 500
            return {
                "up": False,
                "latency_s": round(time.perf_counter() - started, 4),
                "error": f"{type(exc).__name__}: {str(exc)[:120]}",
            }
        return {
            "up": info.get("status") == "ok",
            "latency_s": round(float(info.get("latency_ms") or 0) / 1000, 4),
            "error": info.get("error"),
        }

    started = time.perf_counter()
    results = await asyncio.gather(*(one(n, f) for n, f in checks.items()))
    logger.debug("依赖探测完成 %.0fms", (time.perf_counter() - started) * 1000)
    return dict(zip(checks, results))


# ---------------------------------------------------------------------------
# 磁盘水位
# ---------------------------------------------------------------------------
def disk_snapshot(settings: Settings | None = None) -> dict:
    """返回数据盘容量；路径不可用时 ``available=False``。"""
    path = (settings or get_settings()).data_path
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {"path": str(path), "available": False, "error": str(exc)[:120]}
    ratio = (usage.used / usage.total) if usage.total else 0.0
    return {
        "path": str(path),
        "available": True,
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "used_ratio": round(ratio, 4),
    }


def check_disk_watermark(
    disk: dict | None = None, settings: Settings | None = None
) -> bool:
    """越阈值时只告警一次（回落时再记一条恢复日志），避免每次抓取刷日志。"""
    global _disk_alerted
    s = settings or get_settings()
    snap = disk or disk_snapshot(s)
    exceeded = bool(snap.get("available")) and snap["used_ratio"] >= s.metrics_disk_warn_ratio

    with _disk_lock:
        if exceeded and not _disk_alerted:
            _disk_alerted = True
            logger.warning(
                "磁盘水位告警：%s 使用率 %.1f%% ≥ 阈值 %.1f%%，剩余 %.1f GB",
                snap["path"],
                snap["used_ratio"] * 100,
                s.metrics_disk_warn_ratio * 100,
                snap["free_bytes"] / 1024**3,
            )
        elif not exceeded and _disk_alerted:
            _disk_alerted = False
            logger.info(
                "磁盘水位恢复：%s 使用率 %.1f%% < 阈值 %.1f%%",
                snap.get("path", "?"),
                snap["used_ratio"] * 100,
                s.metrics_disk_warn_ratio * 100,
            )
    return exceeded


# ---------------------------------------------------------------------------
# 渲染（Prometheus text exposition）
# ---------------------------------------------------------------------------
def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(items: dict[str, Any]) -> str:
    if not items:
        return ""
    body = ",".join(f'{k}="{_escape(str(v))}"' for k, v in items.items())
    return "{" + body + "}"


def _family(lines: list[str], name: str, mtype: str, help_text: str) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {mtype}")


def render_prometheus(
    dependencies: dict[str, dict] | None = None,
    disk: dict | None = None,
    disk_exceeded: bool = False,
) -> str:
    with _LOCK:
        http_requests = dict(_HTTP_REQUESTS)
        duration_sum = dict(_HTTP_DURATION_SUM)
        duration_count = dict(_HTTP_DURATION_COUNT)
        llm_calls = dict(_LLM_CALLS)
        llm_failures = dict(_LLM_FAILURES)
        rate_limited = dict(_RATE_LIMITED)
        permission_denied = dict(_PERMISSION_DENIED)
        guardrail_hits = dict(_GUARDRAIL_HITS)
        gpu_fallbacks = dict(_GPU_FALLBACKS)

    lines: list[str] = []
    _family(lines, "lawanswer_http_requests_total", "counter", "HTTP requests by method, route and status.")
    for (method, path, status), count in sorted(http_requests.items()):
        lines.append(
            f"lawanswer_http_requests_total"
            f"{_labels({'method': method, 'path': path, 'status': status})} {count}"
        )

    _family(
        lines,
        "lawanswer_http_request_duration_seconds",
        "summary",
        "HTTP request duration in seconds (sum/count; compute mean as sum/count).",
    )
    for (method, path), count in sorted(duration_count.items()):
        labels = _labels({"method": method, "path": path})
        lines.append(f"lawanswer_http_request_duration_seconds_count{labels} {count}")
        lines.append(
            f"lawanswer_http_request_duration_seconds_sum{labels} "
            f"{duration_sum.get((method, path), 0.0):.6f}"
        )

    _family(lines, "lawanswer_llm_calls_total", "counter", "LLM API requests by model and outcome.")
    for (model, outcome), count in sorted(llm_calls.items()):
        lines.append(
            f"lawanswer_llm_calls_total{_labels({'model': model, 'outcome': outcome})} {count}"
        )

    _family(lines, "lawanswer_llm_failures_total", "counter", "LLM failures by unified error kind.")
    for (model, kind), count in sorted(llm_failures.items()):
        lines.append(
            f"lawanswer_llm_failures_total{_labels({'model': model, 'kind': kind})} {count}"
        )

    _family(lines, "lawanswer_rate_limited_total", "counter", "Rejected requests by reason (chat rate limit / ingest queue).")
    for reason, count in sorted(rate_limited.items()):
        lines.append(f"lawanswer_rate_limited_total{_labels({'reason': reason})} {count}")

    _family(
        lines,
        "lawanswer_permission_denied_total",
        "counter",
        "Rejected resource operations by resource type (no ids, no titles).",
    )
    for resource, count in sorted(permission_denied.items()):
        lines.append(
            f"lawanswer_permission_denied_total{_labels({'resource': resource})} {count}"
        )

    _family(
        lines,
        "lawanswer_guardrail_hits_total",
        "counter",
        "Sensitive-info guardrail hits by kind (api_key/internal_path/system_prompt/...).",
    )
    for kind, count in sorted(guardrail_hits.items()):
        lines.append(f"lawanswer_guardrail_hits_total{_labels({'kind': kind})} {count}")

    _family(
        lines,
        "lawanswer_gpu_fallbacks_total",
        "counter",
        "Sticky GPU failures that forced a one-way CPU fallback, by exception type.",
    )
    for reason, count in sorted(gpu_fallbacks.items()):
        lines.append(f"lawanswer_gpu_fallbacks_total{_labels({'reason': reason})} {count}")

    if dependencies is not None:
        _family(lines, "lawanswer_dependency_up", "gauge", "Dependency health (1=ok, 0=down).")
        for name in DEPENDENCY_NAMES:
            info = dependencies.get(name, {})
            lines.append(
                f"lawanswer_dependency_up{_labels({'service': name})} {1 if info.get('up') else 0}"
            )
        _family(
            lines,
            "lawanswer_dependency_latency_seconds",
            "gauge",
            "Last dependency probe latency in seconds.",
        )
        for name in DEPENDENCY_NAMES:
            info = dependencies.get(name, {})
            lines.append(
                f"lawanswer_dependency_latency_seconds{_labels({'service': name})} "
                f"{float(info.get('latency_s') or 0.0):.4f}"
            )

    if disk is not None and disk.get("available"):
        path = str(disk["path"])
        labels = _labels({"path": path})
        _family(lines, "lawanswer_disk_used_ratio", "gauge", "Data disk used ratio (0-1).")
        lines.append(f"lawanswer_disk_used_ratio{labels} {float(disk['used_ratio']):.4f}")
        _family(lines, "lawanswer_disk_free_bytes", "gauge", "Data disk free bytes.")
        lines.append(f"lawanswer_disk_free_bytes{labels} {int(disk['free_bytes'])}")
        _family(lines, "lawanswer_disk_total_bytes", "gauge", "Data disk total bytes.")
        lines.append(f"lawanswer_disk_total_bytes{labels} {int(disk['total_bytes'])}")
        _family(lines, "lawanswer_disk_watermark_exceeded", "gauge", "1 when disk usage crossed the warn threshold.")
        lines.append(f"lawanswer_disk_watermark_exceeded{labels} {1 if disk_exceeded else 0}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def reset_metrics() -> None:
    """清空计数器与缓存（仅测试使用）。"""
    global _dep_cached_at, _dep_cached, _disk_alerted
    with _LOCK:
        _HTTP_REQUESTS.clear()
        _HTTP_DURATION_SUM.clear()
        _HTTP_DURATION_COUNT.clear()
        _LLM_CALLS.clear()
        _LLM_FAILURES.clear()
        _RATE_LIMITED.clear()
        _PERMISSION_DENIED.clear()
        _GUARDRAIL_HITS.clear()
        _GPU_FALLBACKS.clear()
    with _dep_lock:
        _dep_cached = {}
        _dep_cached_at = 0.0
    with _disk_lock:
        _disk_alerted = False
