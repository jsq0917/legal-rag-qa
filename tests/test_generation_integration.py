"""阶段 5 联调测试：真实 MySQL + Redis + Milvus 检索 + bge-m3，假 LLM 流。

    RUN_INTEGRATION=1 uv run python -m pytest tests/test_generation_integration.py -v -m integration

验证生成链路在真实依赖下的行为：
* 结构化命中 → SSE 回放标准答案，**零检索调用**（验收硬标准）；
* 纯法条问题 → 真实混合检索出法条 → sources 带真实法名/条号；
* 无资料问题 → 引导话术，不调 LLM；
* 落库：造一个临时 session（user_id 关联已有用户或新建）后跑完整流，
  断言 chat_messages 两行入库、seq 连续、sources JSON 完整。
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

from app.core.sse import parse_sse_frames
from app.generation import stream as stream_mod
from app.generation.llm_client import StreamChunk
from app.generation.stream import AnswerEngine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="需要真实 MySQL + Redis + Milvus + 本地模型，设置 RUN_INTEGRATION=1 启用",
    ),
]


class FakeLLMStream:
    """假流式客户端：固定产出 + 记录调用。"""

    def __init__(self, chunks=None):
        self.chunks = chunks or [
            StreamChunk(text="结论："),
            StreamChunk(text="试用期最长不得超过六个月。"),
            StreamChunk(usage={"prompt_tokens": 500, "completion_tokens": 30, "total_tokens": 530}),
        ]
        self.calls = []

    def chat_stream(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        yield from self.chunks


def run_frames(engine, query, **kw):
    raw = "".join(engine.stream(query, **kw))
    return parse_sse_frames(raw)


def deltas(frames):
    return "".join(d["text"] for n, d in frames if n == "delta")


def get_event(frames, name):
    return next((d for n, d in frames if n == name), None)


@pytest.fixture(scope="module")
def session_id():
    """造一个真实会话（复用/新建联调用户），产出 sid；结束后清理消息、会话与临时用户。"""
    from sqlalchemy import text

    from app.db.mysql import get_engine

    sid = "IT" + uuid.uuid4().hex[:23].upper()
    created_user = False
    with get_engine().begin() as conn:
        row = conn.execute(text("SELECT id FROM users ORDER BY id LIMIT 1")).fetchone()
        if row is None:
            conn.execute(
                text("INSERT INTO users (username, password_hash) VALUES (:u, :p)"),
                {"u": f"it_{uuid.uuid4().hex[:8]}", "p": "integration-only"},
            )
            user_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
            created_user = True
        else:
            user_id = row[0]
        conn.execute(
            text(
                "INSERT INTO chat_sessions (id, user_id, kb_ids) "
                "VALUES (:sid, :uid, CAST('[\"kb_labor\"]' AS JSON))"
            ),
            {"sid": sid, "uid": user_id},
        )
    yield sid
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM chat_messages WHERE session_id = :sid"), {"sid": sid})
        conn.execute(text("DELETE FROM chat_sessions WHERE id = :sid"), {"sid": sid})
        if created_user:
            conn.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})


class TestStructuredReplayReal:
    def test_structured_replay_zero_retrieval(self):
        """标准问法 → 纯结构化回放；检索函数若被调用即失败（mock 计数）。"""

        def counting_retrieve(*args, **kwargs):
            raise AssertionError("纯结构化问题不应触发检索")

        engine = AnswerEngine(llm_client=FakeLLMStream(), retrieve_fn=counting_retrieve)
        frames = run_frames(engine, "试用期最长可以约定多久？", selected_kb_ids=["kb_labor"])
        assert "六个月" in deltas(frames)
        done = get_event(frames, "done")
        assert done["sources_count"] >= 1
        sources = get_event(frames, "sources")["sources"]
        assert sources[0]["kind"] == "qa"
        assert sources[0]["law"] == "中华人民共和国劳动合同法"

    def test_second_call_hits_redis_cache(self):
        """同一问题第二次 → redis_exact，done.cached=True。"""

        def boom(*a, **k):
            raise AssertionError("no retrieve")

        engine = AnswerEngine(llm_client=FakeLLMStream(), retrieve_fn=boom)
        frames = run_frames(engine, "试用期最长可以约定多久？", selected_kb_ids=["kb_labor"])
        assert get_event(frames, "done")["cached"] is True


class TestVectorGenerationReal:
    def test_real_retrieval_sources(self):
        """纯法条问题：真实混合检索 → sources 带真实法名/条号/章节。

        kb_labor 除《劳动合同法》《劳动法》外还收录了实施条例、工伤保险条例等，
        因此只约束"来源必须落在所选库内且能命中劳动合同法"，不写死法名白名单。
        """
        engine = AnswerEngine(llm_client=FakeLLMStream())
        frames = run_frames(engine, "劳动合同法第八十二条原文怎么规定", selected_kb_ids=["kb_labor"])
        sources = get_event(frames, "sources")["sources"]
        assert sources, "真实检索应有法条来源"
        assert all(s["kb_id"] == "kb_labor" for s in sources)
        assert all(s["law"] for s in sources)
        assert any(s["law"] == "中华人民共和国劳动合同法" for s in sources)
        assert all(s["article_no"].startswith("第") for s in sources)
        assert all(s["chapter_path"] for s in sources)
        assert deltas(frames) == "结论：试用期最长不得超过六个月。"  # 假 LLM 固定输出
        assert get_event(frames, "done")["token_usage"]["total_tokens"] == 530

    def test_context_contains_law_text(self):
        """进 prompt 的上下文含法条正文（父块回填）。

        top1 可能是《劳动合同法实施条例》等同库资料，因此只要求"资料块 + 正文"
        存在，不绑定具体法名。
        """
        llm = FakeLLMStream()
        engine = AnswerEngine(llm_client=llm)
        run_frames(engine, "劳动合同法第八十二条原文怎么规定", selected_kb_ids=["kb_labor"])
        user_msg = llm.calls[0]["messages"][-1]["content"]
        assert "【参考资料】" in user_msg
        assert "【资料1】《" in user_msg
        assert "第八十二条" in user_msg
        assert len(user_msg) > 200

    def test_no_matching_kb_guidance(self):
        """知识库不匹配（勾选了不含相关法律的库）→ 引导话术，不调 LLM。

        真实 Milvus 没有相关度下限，任何问题都能召回 topK，因此"无资料"
        只能由**路由层**触发：勾选库与问题所属法律无交集 → no_data。
        """
        llm = FakeLLMStream()
        engine = AnswerEngine(llm_client=llm)
        frames = run_frames(engine, "试用期最长可以约定多久？", selected_kb_ids=["kb_not_exist"])
        assert llm.calls == []
        assert "暂未在所选知识库中找到" in deltas(frames)
        assert get_event(frames, "done")["sources_count"] == 0
        assert get_event(frames, "meta")["route"] == "no_data"

    def test_unrelated_question_routes_to_no_data(self, monkeypatch):
        """与法律无关的问题由路由层拦截，不检索、不生成。

        意图用假客户端固定为 ``other``，避免真实 LLM 分类波动影响断言。
        """
        import json

        from app.generation.llm_client import LLMResponse
        from app.orchestration import intent as intent_mod

        class FakeIntentClient:
            def chat(self, messages, **kwargs):
                data = {
                    "intent": "other",
                    "rewrite_type": "direct",
                    "sub_queries": ["如何用 Python 实现快速排序"],
                    "complexity": "simple",
                    "hit_structured": False,
                    "structured_topic": None,
                    "law_hints": [],
                    "need_history": False,
                }
                return LLMResponse(
                    content=json.dumps(data, ensure_ascii=False),
                    model="fake",
                    latency_ms=1,
                    usage={},
                )

        monkeypatch.setattr(intent_mod, "get_intent_client", lambda settings=None: FakeIntentClient())
        llm = FakeLLMStream()
        engine = AnswerEngine(llm_client=llm)
        frames = run_frames(engine, "如何用 Python 实现快速排序", selected_kb_ids=["kb_labor"])
        assert get_event(frames, "meta")["route"] == "no_data"
        assert get_event(frames, "done")["sources_count"] == 0
        assert llm.calls == []
        assert "暂未在所选知识库中找到" in deltas(frames)


class TestPersistenceReal:
    def test_exchange_persisted(self, session_id):
        engine = AnswerEngine(llm_client=FakeLLMStream())
        frames = run_frames(engine, "试用期最长可以约定多久？", selected_kb_ids=["kb_labor"], session_id=session_id)
        assert get_event(frames, "done")["sources_count"] >= 1

        from sqlalchemy import text

        from app.db.mysql import get_engine

        with get_engine().connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT seq, role, content, sources, intent, route, cached, ttft_ms, token_usage, error "
                    "FROM chat_messages WHERE session_id = :sid ORDER BY seq"
                ),
                {"sid": session_id},
            ).fetchall()
        assert len(rows) == 2
        user_row, assistant_row = rows
        assert user_row[1] == "user" and user_row[0] == 1
        assert assistant_row[1] == "assistant" and assistant_row[0] == 2
        assert "六个月" in assistant_row[2]
        assert assistant_row[3] is not None  # sources JSON
        parsed = json.loads(assistant_row[3]) if isinstance(assistant_row[3], str) else assistant_row[3]
        assert parsed and parsed[0]["law"] == "中华人民共和国劳动合同法"
        assert assistant_row[4] == "labor_law"
        assert assistant_row[5] == "pure_structured"
        assert assistant_row[7] is not None  # ttft_ms
        assert assistant_row[8] is None or "prompt_tokens" in str(assistant_row[8])
        assert assistant_row[9] is None  # 无错误
