"""上传链路回归（阶段 6 走查发现的问题）：展示名与存储路径不得被磁盘存储名覆盖。

    RUN_INTEGRATION=1 uv run python -m pytest tests/test_upload_flow_integration.py -v -m integration

背景：`ingest()` 原先用 `prepared.source_file`（磁盘名，上传时带 `<随机ID>_` 前缀）
同时写入 `documents.filename` 与 `storage_path`，造成两个后果：
1. 前端按文件名匹配进度永远匹配不上（列表里显示的是随机前缀名）；
2. 上传文件实际在 data/uploads/ 下，storage_path 被改写成 data/raw/ 后
   "重新解析"找不到源文件。
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="需要真实 MySQL + Milvus + 本地模型，设置 RUN_INTEGRATION=1 启用",
    ),
]


@pytest.fixture(scope="module")
def client():
    from app.main import app

    return TestClient(app)


def _wait_ready(
    client: TestClient, doc_id: str, headers: dict | None = None, timeout_s: float = 180.0
) -> dict:
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        detail = client.get(f"/api/documents/{doc_id}", headers=headers).json()
        if detail["status"] in {"ready", "failed"}:
            return detail
        time.sleep(1.5)
    raise AssertionError(f"文档 {doc_id} 在 {timeout_s}s 内未完成解析")


class TestUploadNaming:
    @pytest.fixture(autouse=True)
    def _auth(self, integration_auth_headers):
        self.headers = integration_auth_headers

    def test_filename_and_storage_path_preserved(self, client):
        filename = f"e2e-{uuid.uuid4().hex[:8]}.md"
        content = (
            "# 上传链路回归样例\n\n"
            f"第一条 本文件用于验证上传文档的展示名与存储路径（{uuid.uuid4().hex[:6]}）。\n\n"
            "第二条 测试结束后会被删除，不影响正式语料。\n"
        )
        resp = client.post(
            "/api/kbs/kb_labor/documents",
            files={"files": (filename, content.encode("utf-8"), "text/markdown")},
            data={"chunk_size": "300", "overlap": "50", "retrieval_mode": "hybrid"},
            headers=self.headers,
        )
        assert resp.status_code == 202, resp.text
        doc_id = resp.json()["accepted"][0]["doc_id"]

        detail = _wait_ready(client, doc_id, self.headers)
        assert detail["status"] == "ready", detail.get("error_msg")
        # 1) 展示名保持原名（不带随机前缀）
        assert detail["filename"] == filename
        # 2) 存储路径指向 data/uploads/（不是 data/raw/）
        assert detail["storage_path"].startswith("data/uploads/"), detail["storage_path"]
        assert detail["chunk_count"] >= 1
        assert detail["article_count"] == 2

        # 3) 重新解析：源文件应能被找到（storage_path 正确）
        reingest = client.post(f"/api/documents/{doc_id}/reingest", headers=self.headers)
        assert reingest.status_code == 202, reingest.text
        again = _wait_ready(client, doc_id, self.headers)
        assert again["status"] == "ready"

        # 清理
        assert client.delete(f"/api/documents/{doc_id}", headers=self.headers).status_code == 200

    def test_duplicate_upload_skips_without_phantom_row(self, client):
        """同内容重复上传 → 跳过，且不留永久"排队中"的占位行。"""
        filename = f"e2e-dup-{uuid.uuid4().hex[:8]}.md"
        content = f"# 重复上传样例\n\n第一条 重复内容测试（{uuid.uuid4().hex[:6]}）。\n"
        first = client.post(
            "/api/kbs/kb_labor/documents",
            files={"files": (filename, content.encode("utf-8"), "text/markdown")},
            data={"chunk_size": "300", "overlap": "50", "retrieval_mode": "hybrid"},
            headers=self.headers,
        ).json()["accepted"][0]["doc_id"]
        assert _wait_ready(client, first, self.headers)["status"] == "ready"

        second = client.post(
            "/api/kbs/kb_labor/documents",
            files={"files": (filename, content.encode("utf-8"), "text/markdown")},
            data={"chunk_size": "300", "overlap": "50", "retrieval_mode": "hybrid"},
            headers=self.headers,
        ).json()["accepted"][0]["doc_id"]
        assert second != first

        import time

        time.sleep(3)
        docs = client.get("/api/kbs/kb_labor/documents", headers=self.headers).json()
        phantom = [d for d in docs if d["id"] == second]
        assert phantom == [], "重复上传的占位行应被清理，不能留下永久 pending"
        assert any(d["id"] == first and d["status"] == "ready" for d in docs)

        assert client.delete(f"/api/documents/{first}", headers=self.headers).status_code == 200
