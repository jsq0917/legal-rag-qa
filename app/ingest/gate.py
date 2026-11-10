"""ingest 并发闸门：限制同时解析/入库的文档数，并给排队长度设上限。

为什么需要：上传接口用 FastAPI ``BackgroundTasks`` 起解析任务，每个任务在
anyio 线程池里跑完整解析 + 向量化（单文件数十秒到分钟级）。不设上限时，
N 个并发上传会把线程池、显存与内存一起打满，殃及问答链路。

语义（三件事分开）：

* :meth:`IngestGate.try_admit`：上传入口**不阻塞**地占一个"在途"名额
  （运行中 + 排队中）。名额用尽返回 ``False``，由接口层转成 429。
* :meth:`IngestGate.worker_slot`：后台任务在工作线程里等待一个"并发"名额。
  只限制真正在跑的任务数，排队中的任务不占线程。
* :meth:`IngestGate.release`：任务结束（成功/失败/超时）归还名额。

名额是**进程内**的：多 worker 部署时每个进程各自限流，总并发为
``进程数 × INGEST_MAX_CONCURRENCY``。需要跨进程硬上限时应换成独立队列
（arq/RQ），这里刻意不引入。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from app.core.logger import get_logger
from config import Settings, get_settings

logger = get_logger("lawanswer.ingest.gate")


class IngestBusy(RuntimeError):
    """并发名额与排队名额都已用尽，或等待并发名额超时。"""


class IngestGate:
    """线程安全的两级闸门：在途容量（并发 + 排队）与并发上限。"""

    def __init__(
        self,
        max_concurrency: int,
        max_queue: int = 0,
        slot_timeout_s: float = 1800.0,
    ) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_queue = max(0, int(max_queue))
        self.slot_timeout_s = float(slot_timeout_s)
        self._cond = threading.Condition()
        self._slots = threading.BoundedSemaphore(self.max_concurrency)
        self._outstanding = 0
        self._active = 0

    # ------------------------------------------------------------------
    # 只读状态（指标/运维排查用）
    # ------------------------------------------------------------------
    @property
    def outstanding(self) -> int:
        """已准入但尚未结束的任务数（运行中 + 排队中）。"""
        with self._cond:
            return self._outstanding

    @property
    def active(self) -> int:
        """正在解析/入库的任务数。"""
        with self._cond:
            return self._active

    # ------------------------------------------------------------------
    # 准入 / 归还
    # ------------------------------------------------------------------
    def try_admit(self) -> bool:
        """尝试占一个在途名额，不阻塞；容量用尽返回 False。"""
        with self._cond:
            if self._outstanding >= self.max_concurrency + self.max_queue:
                return False
            self._outstanding += 1
            return True

    def release(self) -> None:
        """归还一个在途名额（幂等：下限为 0，不会把计数减成负数）。"""
        with self._cond:
            if self._outstanding > 0:
                self._outstanding -= 1
            self._cond.notify_all()

    @contextmanager
    def worker_slot(self, timeout: float | None = None) -> Iterator[None]:
        """占用一个并发名额；等待超过 ``timeout`` 秒抛 :class:`IngestBusy`。

        只应在已经 ``try_admit()`` 成功（或等价地受调用方保护）的任务里使用。
        """
        wait_s = self.slot_timeout_s if timeout is None else float(timeout)
        acquired = self._slots.acquire(timeout=wait_s)
        if not acquired:
            raise IngestBusy(
                f"等待解析名额超时（{wait_s:.0f}s，并发上限 {self.max_concurrency}）"
            )
        with self._cond:
            self._active += 1
        try:
            yield
        finally:
            with self._cond:
                self._active = max(0, self._active - 1)
            self._slots.release()


@lru_cache(maxsize=1)
def get_ingest_gate() -> IngestGate:
    """进程级单例，参数来自 Settings（可被环境变量覆盖）。"""
    return _build_gate(get_settings())


def _build_gate(settings: Settings) -> IngestGate:
    return IngestGate(
        max_concurrency=settings.ingest_max_concurrency,
        max_queue=settings.ingest_max_queue,
        slot_timeout_s=settings.ingest_slot_timeout_s,
    )


def reset_ingest_gate() -> None:
    """测试用：清空单例（改配置后需要重建）。"""
    get_ingest_gate.cache_clear()
