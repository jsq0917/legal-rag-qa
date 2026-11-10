"""ingest 并发闸门测试：准入上限、并发上限、超时与接口 429。

全部离线：不解析真实文档、不连数据库。
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.ingest.service as service_mod
from app.ingest.gate import IngestBusy, IngestGate, _build_gate, get_ingest_gate, reset_ingest_gate
from app.main import app
from tests.fakes_db import FakeDB, override_deps


@pytest.fixture(autouse=True)
def _fresh_gate():
    """每个用例前后重置进程级单例，避免名额泄漏影响后续用例。"""
    reset_ingest_gate()
    yield
    reset_ingest_gate()


class TestAdmission:
    def test_capacity_is_concurrency_plus_queue(self):
        gate = IngestGate(max_concurrency=2, max_queue=1)
        assert [gate.try_admit() for _ in range(3)] == [True, True, True]
        assert gate.outstanding == 3
        assert gate.try_admit() is False

    def test_release_frees_a_slot(self):
        gate = IngestGate(max_concurrency=1, max_queue=0)
        assert gate.try_admit()
        assert gate.try_admit() is False
        gate.release()
        assert gate.try_admit() is True

    def test_release_is_idempotent(self):
        gate = IngestGate(max_concurrency=1, max_queue=0)
        gate.release()
        gate.release()
        assert gate.outstanding == 0

    def test_settings_are_wired(self, settings_factory):
        settings = settings_factory(ingest_max_concurrency=3, ingest_max_queue=4, ingest_slot_timeout_s=60)
        gate = _build_gate(settings)
        assert (gate.max_concurrency, gate.max_queue, gate.slot_timeout_s) == (3, 4, 60.0)

    def test_defaults_come_from_settings(self):
        gate = get_ingest_gate()
        assert gate.max_concurrency >= 1
        assert gate.max_queue >= 0


class TestConcurrency:
    def test_worker_slot_limits_parallel_jobs(self):
        """6 个已准入任务同时跑，真正并发的不得超过 2。"""
        gate = IngestGate(max_concurrency=2, max_queue=10)
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}
        barrier = threading.Barrier(6)

        def job() -> None:
            assert gate.try_admit()
            barrier.wait()
            try:
                with gate.worker_slot(timeout=5):
                    with lock:
                        state["active"] += 1
                        state["peak"] = max(state["peak"], state["active"])
                    time.sleep(0.05)
                    with lock:
                        state["active"] -= 1
            finally:
                gate.release()

        threads = [threading.Thread(target=job) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert all(not t.is_alive() for t in threads)
        assert state["peak"] == 2
        assert gate.active == 0
        assert gate.outstanding == 0

    def test_worker_slot_times_out(self):
        gate = IngestGate(max_concurrency=1, max_queue=0)
        started = threading.Event()
        finish = threading.Event()

        def holder() -> None:
            with gate.worker_slot(timeout=2):
                started.set()
                finish.wait(2)

        t = threading.Thread(target=holder)
        t.start()
        assert started.wait(2)
        try:
            with pytest.raises(IngestBusy):
                with gate.worker_slot(timeout=0.1):
                    pass
        finally:
            finish.set()
            t.join(timeout=3)
        assert gate.active == 0


class TestRunIngestJob:
    def test_rejects_without_touching_pipeline_when_full(self, monkeypatch):
        gate = IngestGate(max_concurrency=1, max_queue=0)
        assert gate.try_admit()
        monkeypatch.setattr(service_mod, "get_ingest_gate", lambda: gate)
        failed: list[tuple[str, str]] = []
        monkeypatch.setattr(service_mod, "_mark_failed", lambda doc_id, error: failed.append((doc_id, error)))
        monkeypatch.setattr(
            service_mod, "prepare", lambda *a, **k: pytest.fail("队列满时不应进入解析")
        )

        result = service_mod.run_ingest_job(
            "x.md", kb_id="kb", doc_id="doc-1", chunk_size=300, overlap=50
        )

        assert result["status"] == "rejected"
        assert failed and failed[0][0] == "doc-1"
        # 被拒绝的任务没有占用名额，准入它的那个仍保留
        assert gate.outstanding == 1

    def test_releases_slot_after_failure(self, monkeypatch):
        gate = IngestGate(max_concurrency=1, max_queue=0)
        monkeypatch.setattr(service_mod, "get_ingest_gate", lambda: gate)
        monkeypatch.setattr(service_mod, "_mark_failed", lambda doc_id, error: None)

        def boom(*_a, **_k):
            raise RuntimeError("解析炸了")

        monkeypatch.setattr(service_mod, "prepare", boom)
        result = service_mod.run_ingest_job(
            "x.md", kb_id="kb", doc_id="doc-2", chunk_size=300, overlap=50
        )

        assert result["status"] == "failed"
        assert "解析炸了" in result["error"]
        assert gate.outstanding == 0


class TestUploadQueueFull:
    def test_upload_returns_429_when_queue_full(self, monkeypatch):
        restore = override_deps(app, FakeDB())
        try:
            gate = IngestGate(max_concurrency=1, max_queue=0)
            assert gate.try_admit()
            monkeypatch.setattr("app.api.knowledge.get_ingest_gate", lambda: gate)
            monkeypatch.setattr("app.api.knowledge.validate_upload", lambda *a, **k: None)

            resp = TestClient(app).post(
                "/api/kbs/kb_labor/documents",
                files={"files": ("测试.md", b"# hello", "text/markdown")},
                data={"chunk_size": "300", "overlap": "50", "retrieval_mode": "hybrid"},
            )
        finally:
            restore()

        assert resp.status_code == 429
        body = resp.json()
        assert body["code"] == "too_many_requests"
        assert body["detail"]["max_concurrency"] == 1
