"""GPU 健康闸门：一次 GPU 故障 → 进程级 CPU 降级，而不是整进程永久不可用。

背景（2026-11-11 gap-analysis 缺口 1）
--------------------------------------
RTX 4060 / torch cu130 在**并发** embed + rerank 时会偶发 GPU 层异常
（``CUDA error: an illegal instruction`` / ``illegal memory access``）。
CUDA context 一旦被销毁，``torch.cuda.is_available()`` 仍返回 True，
旧代码不会降级，后续所有前向都级联成 ``'list' object has no attribute 'keys'``。

本模块提供三件事：
1. 进程级状态 ``unknown → healthy / poisoned``，**只降级一次、不回切 GPU**；
2. embed 与 rerank 共用一把推理锁，把"两模型并发"变回已验证稳定的串行；
3. 粘性 GPU 故障识别与错误透明化：CPU 重建后重试一次，仍失败则抛
   ``CUDA context poisoned``，不再让调用方看到误导性的 keys 错误。

OOM 不算粘性故障（按 batch 收缩处理），避免把普通显存不足误判成设备损坏。
"""

from __future__ import annotations

import threading
from typing import Callable, TypeVar

from app.core.logger import get_logger

logger = get_logger("lawanswer.gpu_guard")

T = TypeVar("T")

UNKNOWN = "unknown"
HEALTHY = "healthy"
POISONED = "poisoned"

_state = UNKNOWN
_state_lock = threading.Lock()
#: embed / rerank 共用的 GPU 推理锁（非可重入，避免嵌套获取）
_gpu_lock = threading.Lock()

#: 粘性故障关键字：命中即认为 CUDA context 已损坏
_STICKY_MARKERS = (
    "illegal instruction",
    "illegal memory access",
    "device-side assert",
    "cuda error",
    "no kernel image is available",
    "'list' object has no attribute 'keys'",
)


def lock() -> threading.Lock:
    """返回进程级 GPU 推理锁；embed 与 rerank 必须共用。"""
    return _gpu_lock


def state() -> str:
    return _state


def is_poisoned() -> bool:
    return _state == POISONED


def reset() -> None:
    """仅测试/运维重启用：把闸门恢复为 unknown。"""
    global _state
    with _state_lock:
        _state = UNKNOWN


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "outofmemory" in text


def is_sticky(exc: BaseException) -> bool:
    """判断异常是否代表 CUDA context 已损坏（OOM 排除在外）。"""
    if _is_oom(exc):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _STICKY_MARKERS)


def run(fn: Callable[[], T], *, rebuild: Callable[[], None] | None = None) -> T:
    """在 GPU 闸门下执行一次推理。

    :param fn: 实际前向调用
    :param rebuild: 设备降级回调（在 CPU 上重建模型）；命中粘性故障时调用一次
    """
    global _state

    with _gpu_lock:
        if _state == POISONED:
            # 已降级：模型应是 CPU 版；再失败就明确标注 context 已损坏
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"CUDA context poisoned: {exc}") from exc

        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            if not is_sticky(exc):
                raise
            with _state_lock:
                _state = POISONED
            logger.error("检测到粘性 GPU 故障，device=fallback，降级到 CPU 并重试一次：%s", exc)
            try:
                from app.core.metrics import record_gpu_fallback

                record_gpu_fallback(type(exc).__name__)
            except Exception:  # noqa: BLE001 - 指标失败不能影响主链路
                pass
            if rebuild is not None:
                rebuild()
            try:
                return fn()
            except Exception as retry_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"CUDA context poisoned; CPU fallback failed: {retry_exc}"
                ) from retry_exc

        with _state_lock:
            if _state == UNKNOWN:
                _state = HEALTHY
        return result
