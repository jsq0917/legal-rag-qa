"""向量化：bge-m3 一次前向同时产出稠密 + 稀疏。

* 稠密向量（1024 维）负责语义相似；稀疏向量（learned lexical weights）负责
  "代通知金""竞业限制"这类法律术语的精确命中，两者在 Milvus 里分别建索引后 RRF 融合。
* 稀疏向量词表 25 万维，写入前过滤 `weight < 0.05` 的噪声维度，
  每条约保留 60–200 维，索引体积小一个量级。
* `DummyEmbedder` 让全部单测离线运行：确定性哈希伪向量，不加载 2 GB 权重。
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Protocol, Sequence

from config import Settings, get_settings
from app.core import gpu_guard
from app.core.logger import get_logger

logger = get_logger("lawanswer.embedder")

#: 稀疏向量最小保留权重
SPARSE_MIN_WEIGHT = 0.05


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: dict[int, float]

    def sparse_arrays(self) -> tuple[list[int], list[float]]:
        """Milvus 需要的 (indices, values) 形式。"""
        if not self.sparse:
            return [], []
        items = sorted(self.sparse.items())
        return [i for i, _ in items], [v for _, v in items]


class Embedder(Protocol):
    dim: int
    #: 是否具备稠密语义能力。无能力的实现（bm25）dense 恒为零向量，
    #: 余弦相似度没有意义，调用方必须换判据——见 qa_cache 的字面双门。
    supports_dense: bool

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> list[Embedding]: ...


class DummyEmbedder:
    """确定性伪向量：同样的文本永远得到同样的向量，便于断言。"""

    supports_dense = True

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _dense(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [b / 255.0 - 0.5 for b in digest]
        # 用哈希字节循环铺满维度，再归一化
        vector = [raw[i % len(raw)] for i in range(self.dim)]
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def _sparse(self, text: str) -> dict[int, float]:
        weights: dict[int, float] = {}
        for i in range(len(text) - 1):
            gram = text[i : i + 2]
            idx = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:6], 16) % 250_000
            weights[idx] = weights.get(idx, 0.0) + 0.1
        return {k: round(v, 3) for k, v in weights.items() if v >= SPARSE_MIN_WEIGHT}

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> list[Embedding]:
        return [Embedding(dense=self._dense(t), sparse=self._sparse(t)) for t in texts]


class BgeM3Embedder:
    """bge-m3 封装（FlagEmbedding.BGEM3FlagModel）。"""

    supports_dense = True

    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        fp16: bool = True,
        max_length: int = 512,
        dim: int = 1024,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.fp16 = fp16
        self.max_length = max_length
        self.dim = dim
        self._model = None
        # 加载锁：启动后前两个并发请求可能同时触发 _load，无锁会双载 2GB 权重
        self._load_lock = threading.Lock()
        # 推理锁由 app.core.gpu_guard 统一提供：embed 与 rerank 共用一把，
        # 避免两个模型同时在 GPU 上前向（实测会触发 CUDA illegal instruction）。

    def _fallback_to_cpu(self) -> None:
        """GPU 粘性故障后重建为 CPU 模型；本进程不再回切 GPU。"""
        with self._load_lock:
            self._model = None
            self.device = "cpu"
        logger.error("device=fallback：bge-m3 向量化模型已降级到 CPU（本进程不再回到 GPU）")

    def _resolve_device(self) -> str:
        device = (self.device or "cpu").lower()
        if device.startswith("cuda"):
            try:
                import torch
            except Exception:  # noqa: BLE001
                logger.warning("未安装 torch，向量化回退到 CPU")
                return "cpu"
            if not torch.cuda.is_available():
                logger.warning(
                    "配置为 %s 但当前 torch 不支持 CUDA（%s），向量化回退到 CPU；"
                    "大批量入库会明显变慢",
                    device,
                    torch.__version__,
                )
                return "cpu"
        return device

    def _load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:  # 双检：等锁期间别的线程已加载完
                return self._model
            from FlagEmbedding import BGEM3FlagModel

            device = self._resolve_device()
            self._model = BGEM3FlagModel(
                self.model_path,
                use_fp16=self.fp16 and device.startswith("cuda"),
                device=device,
            )
            return self._model

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> list[Embedding]:
        if not texts:
            return []
        return gpu_guard.run(
            lambda: self._encode_once(texts, batch_size),
            rebuild=self._fallback_to_cpu,
        )

    def _encode_once(self, texts: Sequence[str], batch_size: int) -> list[Embedding]:
        model = self._load()
        output = model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = output["dense_vecs"]
        lexical = output["lexical_weights"]
        results: list[Embedding] = []
        for dense, weights in zip(dense_vecs, lexical):
            sparse = {
                int(k): float(v) for k, v in weights.items() if float(v) >= SPARSE_MIN_WEIGHT
            }
            results.append(Embedding(dense=[float(x) for x in dense], sparse=sparse))
        return results


#: 进程级单例：bge-m3 加载一次约 40 秒，绝不能每个请求都重建
_EMBEDDER: Embedder | None = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(settings: Settings | None = None) -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                s = settings or get_settings()
                _EMBEDDER = BgeM3Embedder(
                    str(s.embedding_model_dir),
                    device=s.embedding_device,
                    fp16=s.embedding_fp16,
                    max_length=512,
                )
    return _EMBEDDER


def reset_embedder() -> None:
    global _EMBEDDER
    _EMBEDDER = None


def embeddings_to_milvus(embedding: Embedding) -> dict:
    """转成 Milvus 实体里的两个向量字段。

    空稀疏向量（极短文本）会被 Milvus 拒绝，这时给一个最小占位，检索侧会自然拿到低分。
    """
    indices, values = embedding.sparse_arrays()
    if not indices:
        indices, values = [0], [0.0]
    return {"dense_vector": embedding.dense, "sparse_vector": dict(zip(indices, values))}
