"""GPU 健康闸门：粘性故障一次降级、OOM 不误判、错误透明化（全离线）。"""

from __future__ import annotations

import pytest

from app.core import gpu_guard
from app.retrieval.rerank import BgeReranker


@pytest.fixture(autouse=True)
def _reset_guard():
    gpu_guard.reset()
    yield
    gpu_guard.reset()


def test_sticky_gpu_error_falls_back_to_cpu_and_retries():
    calls: list[int] = []
    rebuilt: list[int] = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA error: an illegal instruction was encountered")
        return ["cpu-result"]

    assert gpu_guard.run(fn, rebuild=lambda: rebuilt.append(1)) == ["cpu-result"]
    assert rebuilt == [1]
    assert gpu_guard.is_poisoned()
    # 降级只发生一次：后续调用直接走 CPU，不再触发重建
    assert gpu_guard.run(lambda: ["ok"]) == ["ok"]
    assert rebuilt == [1]


def test_oom_is_not_treated_as_poison():
    def fn():
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(RuntimeError, match="out of memory"):
        gpu_guard.run(fn, rebuild=lambda: None)
    assert not gpu_guard.is_poisoned()


def test_cpu_retry_failure_is_explicit():
    calls: list[int] = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        raise AttributeError("'list' object has no attribute 'keys'")

    with pytest.raises(RuntimeError, match="CUDA context poisoned"):
        gpu_guard.run(fn, rebuild=lambda: None)
    assert gpu_guard.is_poisoned()


def test_poisoned_state_labels_later_errors():
    def _boom():
        raise RuntimeError("CUDA error: illegal instruction")

    with pytest.raises(RuntimeError, match="CUDA context poisoned"):
        gpu_guard.run(_boom)
    assert gpu_guard.is_poisoned()
    # 后续调用失败时必须明确标注 context 已损坏，而不是原始错误
    with pytest.raises(RuntimeError, match="CUDA context poisoned"):
        gpu_guard.run(lambda: (_ for _ in ()).throw(RuntimeError("boom")))


def test_reranker_returns_cpu_result_after_gpu_failure(monkeypatch):
    class Model:
        def __init__(self, fail_first: bool) -> None:
            self.calls = 0
            self.fail_first = fail_first

        def compute_score(self, batch, **kwargs):
            self.calls += 1
            if self.fail_first and self.calls == 1:
                raise RuntimeError("CUDA error: an illegal instruction was encountered")
            return [0.9 for _ in batch]

    reranker = BgeReranker("dummy", device="cuda")
    reranker._model = Model(fail_first=True)
    cpu_model = Model(fail_first=False)
    monkeypatch.setattr(reranker, "_load", lambda: reranker._model)

    def _rebuild() -> None:
        reranker.device = "cpu"
        reranker._model = cpu_model

    monkeypatch.setattr(reranker, "_fallback_to_cpu", _rebuild)

    assert reranker.score("q", ["d1", "d2"]) == [0.9, 0.9]
    assert reranker.device == "cpu"


def test_reranker_keys_error_is_transparent(monkeypatch):
    class BadModel:
        def compute_score(self, batch, **kwargs):
            raise AttributeError("'list' object has no attribute 'keys'")

    reranker = BgeReranker("dummy", device="cuda")
    reranker._model = BadModel()
    monkeypatch.setattr(reranker, "_load", lambda: reranker._model)
    monkeypatch.setattr(reranker, "_fallback_to_cpu", lambda: setattr(reranker, "device", "cpu"))

    with pytest.raises(RuntimeError, match="CUDA context poisoned"):
        reranker.score("q", ["d1"])
