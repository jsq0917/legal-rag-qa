"""重排：bge-reranker-v2-m3（cross-encoder）。

为什么 RRF 之后还要重排
----------------------
RRF 只融合"排名"，不看"查询与文本的交互"，法律条文之间用词高度相似
（第39条/第40条/第46条都在讲"解除"），融合分常常咬得很紧。cross-encoder 把
(query, 条文) 拼起来做一次完整注意力，能把"试用期被辞退有没有补偿"这种
跨条推理的问题顶到正确条文上。实测：相关条文 0.79 vs 无关条文 0.03，区分度足够。

CPU 上的成本：20 条候选约 2–4 秒。所以只在候选池上跑（≤ 4×topK），
且懒加载、批处理，避免每次请求都付模型加载代价。
"""

from __future__ import annotations

import threading
from typing import Protocol, Sequence

from app.core import gpu_guard
from app.core.logger import get_logger
from config import Settings, get_settings

logger = get_logger("lawanswer.rerank")


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class DummyReranker:
    """离线测试用：按字符重合度打分，确定性且不需要模型。"""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        q = set(query)
        results: list[float] = []
        for doc in documents:
            overlap = len(q & set(doc)) / max(len(q), 1)
            results.append(round(overlap, 4))
        return results


class BgeReranker:
    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        fp16: bool = False,
        batch_size: int = 16,
        max_length: int = 256,
        max_text_chars: int = 200,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.fp16 = fp16
        self.batch_size = batch_size
        self.max_length = max_length
        #: 送进 cross-encoder 的正文裁剪长度。重排是粗筛，前 200 字已足够判断相关性；
        #: 不裁剪的话，300 字的条文会把 CPU 上的耗时拉到 9 秒级。
        self.max_text_chars = max_text_chars
        self._model = None
        # 加载锁：与 embedder 同理，防并发首请求双载模型
        self._load_lock = threading.Lock()
        # 推理锁由 app.core.gpu_guard 统一提供：embed 与 rerank 共用一把，
        # 避免两个模型同时在 GPU 上前向（实测会触发 CUDA illegal instruction）。

    def _fallback_to_cpu(self) -> None:
        """GPU 粘性故障后重建为 CPU 模型；本进程不再回切 GPU。"""
        with self._load_lock:
            self._model = None
            self.device = "cpu"
        logger.error("device=fallback：bge-reranker 模型已降级到 CPU（本进程不再回到 GPU）")

    def _resolve_device(self) -> str:
        device = (self.device or "cpu").lower()
        if device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return device
            except Exception:  # noqa: BLE001
                pass
            logger.warning("重排模型配置为 %s 但 CUDA 不可用，回退 CPU（会明显变慢）", device)
            return "cpu"
        return device

    def _load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:  # 双检：等锁期间别的线程已加载完
                return self._model
            from FlagEmbedding import FlagReranker

            device = self._resolve_device()
            self._model = FlagReranker(
                self.model_path,
                use_fp16=self.fp16 and device.startswith("cuda"),
                device=device,
            )
            logger.info("重排模型已加载：%s (device=%s)", self.model_path, device)
            return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return gpu_guard.run(
            lambda: self._score_once(query, documents),
            rebuild=self._fallback_to_cpu,
        )

    def _score_once(self, query: str, documents: Sequence[str]) -> list[float]:
        model = self._load()
        pairs = [[query, doc[: self.max_text_chars]] for doc in documents]
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            raw = model.compute_score(batch, normalize=True, max_length=self.max_length)
            if isinstance(raw, (int, float)):
                raw = [raw]
            scores.extend(float(x) for x in raw)
        return scores


#: 进程级单例：与 embedder 一样，加载代价高（约 45 秒）
_RERANKER: Reranker | None = None
_RERANKER_LOCK = threading.Lock()


def get_reranker(settings: Settings | None = None) -> Reranker:
    global _RERANKER
    if _RERANKER is None:
        with _RERANKER_LOCK:
            if _RERANKER is None:
                s = settings or get_settings()
                _RERANKER = BgeReranker(
                    str(s.reranker_model_dir),
                    device=s.embedding_device,
                    # GPU 上开 fp16：4060 级别显卡单次重排从秒级降到几十毫秒；CPU 上 fp16 反而更慢
                    fp16=s.embedding_fp16,
                    batch_size=s.reranker_batch_size,
                    max_length=s.reranker_max_length,
                    max_text_chars=s.reranker_max_text_chars,
                )
    return _RERANKER


def reset_reranker() -> None:
    global _RERANKER
    _RERANKER = None
