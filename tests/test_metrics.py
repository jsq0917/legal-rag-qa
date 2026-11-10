"""指标端点测试：请求计数/耗时、LLM 调用、429、依赖健康、磁盘水位。

依赖探测全部替换为桩函数，不连真实 MySQL/Redis/Milvus。
"""

from __future__ import annotations

import collections
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.core.metrics as metrics_mod
from app.core.errors import RateLimited
from app.generation.llm_client import LLMClient
from app.main import create_app


@pytest.fixture(autouse=True)
def _clean_metrics():
    metrics_mod.reset_metrics()
    yield
    metrics_mod.reset_metrics()


def _stub_deps(monkeypatch, **statuses: str) -> None:
    import app.db.milvus_client as milvus_mod
    import app.db.mysql as mysql_mod
    import app.db.redis_client as redis_mod

    modules = {"mysql": mysql_mod, "redis": redis_mod, "milvus": milvus_mod}
    for name, mod in modules.items():
        status = statuses.get(name, "ok")
        monkeypatch.setattr(
            mod,
            f"check_{name}",
            lambda st=status: {
                "status": st,
                "latency_ms": 2.0,
                "detail": None,
                "error": None if st == "ok" else "ConnectionRefused",
            },
        )


class TestEndpoint:
    def test_reports_requests_dependencies_and_disk(self, monkeypatch):
        _stub_deps(monkeypatch)
        client = TestClient(create_app())

        assert client.get("/api/health/live").status_code == 200
        resp = client.get("/metrics")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain; version=0.0.4")
        text = resp.text
        assert (
            'lawanswer_http_requests_total{method="GET",path="/api/health/live",status="200"} 1'
            in text
        )
        assert "lawanswer_http_request_duration_seconds_count" in text
        assert 'lawanswer_dependency_up{service="mysql"} 1' in text
        assert 'lawanswer_dependency_latency_seconds{service="mysql"}' in text
        assert "# TYPE lawanswer_disk_used_ratio gauge" in text
        assert "lawanswer_disk_watermark_exceeded{path=" in text

    def test_dependency_down_is_reported(self, monkeypatch):
        _stub_deps(monkeypatch, milvus="down")
        text = TestClient(create_app()).get("/metrics").text
        assert 'lawanswer_dependency_up{service="milvus"} 0' in text
        assert 'lawanswer_dependency_up{service="mysql"} 1' in text

    def test_probe_exception_does_not_break_metrics(self, monkeypatch):
        def boom() -> dict:
            raise RuntimeError("mysql 驱动炸了")

        import app.db.mysql as mysql_mod

        monkeypatch.setattr(mysql_mod, "check_mysql", boom)
        resp = TestClient(create_app()).get("/metrics")
        assert resp.status_code == 200
        assert 'lawanswer_dependency_up{service="mysql"} 0' in resp.text

    def test_dependency_probe_is_cached(self, monkeypatch):
        calls = {"n": 0}

        def count() -> dict:
            calls["n"] += 1
            return {"status": "ok", "latency_ms": 1.0, "detail": None, "error": None}

        import app.db.milvus_client as milvus_mod
        import app.db.mysql as mysql_mod
        import app.db.redis_client as redis_mod

        monkeypatch.setattr(mysql_mod, "check_mysql", count)
        monkeypatch.setattr(milvus_mod, "check_milvus", count)
        monkeypatch.setattr(redis_mod, "check_redis", count)

        client = TestClient(create_app())
        client.get("/metrics")
        client.get("/metrics")
        # 三个依赖只探测一轮（10s 缓存），第二次抓取复用结果
        assert calls["n"] == 3


class TestCounters:
    def test_llm_calls_by_model_and_outcome(self):
        metrics_mod.record_llm_call("deepseek-v4.1-flash", "success")
        metrics_mod.record_llm_call("deepseek-v4.1-flash", "error")
        text = metrics_mod.render_prometheus()
        assert (
            'lawanswer_llm_calls_total{model="deepseek-v4.1-flash",outcome="success"} 1'
            in text
        )
        assert (
            'lawanswer_llm_calls_total{model="deepseek-v4.1-flash",outcome="error"} 1'
            in text
        )

    def test_llm_client_records_each_request(self):
        client = LLMClient(api_key="test-key", base_url="http://localhost", model="m1")

        class _Completions:
            def create(self, **kwargs):
                return "ok"

        client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        assert client._create(model="m1") == "ok"

        class _Broken:
            def create(self, **kwargs):
                raise RuntimeError("upstream 500")

        client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Broken()))
        with pytest.raises(RuntimeError):
            client._create(model="m1")

        text = metrics_mod.render_prometheus()
        assert 'lawanswer_llm_calls_total{model="m1",outcome="success"} 1' in text
        assert 'lawanswer_llm_calls_total{model="m1",outcome="error"} 1' in text

    def test_rate_limit_increments_429_counter(self, monkeypatch, settings_factory):
        import app.core.ratelimit as rl

        class _FakeRedis:
            def eval(self, *_args, **_kwargs):
                return 999

        monkeypatch.setattr(rl, "get_redis", lambda *a, **k: _FakeRedis())
        settings = settings_factory(rate_limit_enabled=True, rate_limit_anonymous=1)

        with pytest.raises(RateLimited):
            rl._enforce("203.0.113.9", None, settings)

        assert 'lawanswer_rate_limited_total{reason="chat"} 1' in metrics_mod.render_prometheus()

    def test_http_labels_use_route_template(self, monkeypatch):
        """带 id 的路径必须归并到路由模板，不能把每个 id 都变成一条时间序列。"""
        _stub_deps(monkeypatch)
        client = TestClient(create_app())
        client.get("/api/documents/doc-abc")
        text = client.get("/metrics").text
        assert 'path="/api/documents/{doc_id}"' in text
        assert 'path="/api/documents/doc-abc"' not in text


class TestDisk:
    def test_watermark_detects_and_renders(self, monkeypatch, settings_factory):
        usage = collections.namedtuple("usage", "total used free")(100, 90, 10)
        monkeypatch.setattr(metrics_mod.shutil, "disk_usage", lambda _path: usage)
        settings = settings_factory(metrics_disk_warn_ratio=0.85)

        disk = metrics_mod.disk_snapshot(settings)
        assert disk["used_ratio"] == 0.9
        assert metrics_mod.check_disk_watermark(disk, settings) is True

        text = metrics_mod.render_prometheus(disk=disk, disk_exceeded=True)
        assert "lawanswer_disk_used_ratio{path=" in text
        assert "lawanswer_disk_free_bytes{path=" in text
        assert "lawanswer_disk_watermark_exceeded{path=" in text
        assert text.rstrip().endswith("1")

    def test_watermark_below_threshold(self, monkeypatch, settings_factory):
        usage = collections.namedtuple("usage", "total used free")(100, 10, 90)
        monkeypatch.setattr(metrics_mod.shutil, "disk_usage", lambda _path: usage)
        settings = settings_factory(metrics_disk_warn_ratio=0.85)
        assert metrics_mod.check_disk_watermark(settings=settings) is False
