"""追问预测（问题推荐）：回答结束后预测用户最可能继续问的 3 个问题。

为什么要单独一次调用而不是塞进生成流：
* 生成流里追加会**拖慢 done 事件**——用户看到答案却还在等流结束，输入框一直锁着；
* 独立端点让前端在回答渲染完成后异步取，慢几秒无感，失败也只是不展示推荐，
  不影响任何主链路（**失败即静默**，不抛错、不弹错误提示）。

数据流::

    query + answer 摘要 + 涉及法条
      → LLM（JSON 模式, 温度 0.7 取多样性, 短超时）
      → _coerce() 校正（去重 / 去空 / 去与原问重复 / 截断到 3 条）
      → list[str]

任何异常（缺 key / 超时 / 网络 / JSON 非法 / 字段缺失）一律返回空列表。
"""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from app.core.logger import get_logger
from app.generation.llm_client import LLMClient
from config import Settings, get_llm_api_key, get_settings

logger = get_logger("lawanswer.suggestions")

#: 预测条数上限（产品固定 3 条，多给会让界面变成一堵墙）
SUGGEST_LIMIT = 3

#: 送进 prompt 的回答摘要长度：足够判断"这条回答讲了什么"，又不浪费 token
ANSWER_EXCERPT_CHARS = 1200

SUGGEST_SYSTEM_PROMPT = """你是法律问答系统的「追问预测」模块。根据用户上一个问题和助手的回答，预测用户接下来最可能继续追问的 3 个问题，供页面上的快捷选项使用。

要求：
- 每个问题必须是这次问答的**自然延伸**，优先覆盖：具体计算方式、需要哪些证据材料、时效期限、例外情形、下一步流程、与相邻情形的区别；
- 每问 8-25 个字，用普通用户的日常说法，不要出现"用户""请问"这类前缀，不要写成陈述句；
- 三问之间角度不能重复，也不能把原问题换个说法重复一遍；
- 必须能被现有知识库（劳动合同法、劳动法、民法典）回答；回答里已明确说"现有资料未涉及"的方向不要出；
- 只输出 JSON 对象，不要 Markdown 代码块、不要任何解释。

输出格式：
{"questions": ["问题一", "问题二", "问题三"]}"""


def _strip_json_fences(text: str) -> str:
    """容忍 ```json 包裹与前后杂讯：截取第一个 { 到最后一个 }。"""
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def _normalize(text: str) -> str:
    """去空白与常见标点后比较，用于判重（"能赔多少？"与"能赔多少"视为同一条）。"""
    drop = set(" 　\t\n，,。.？?！!：:；;、\"'“”‘’（）()【】[]《》")
    return "".join(ch for ch in text if ch not in drop).lower()


def build_messages(
    query: str,
    answer: str,
    law_names: Sequence[str] | None = None,
) -> list[dict]:
    """装配追问预测的两条消息（纯函数，单测可离线断言）。"""
    excerpt = (answer or "").strip()[:ANSWER_EXCERPT_CHARS]
    parts = [f"【用户上一个问题】\n{query.strip()}"]
    if law_names:
        names = "、".join(n for n in law_names if n)[:200]
        if names:
            parts.append(f"【回答涉及的法律】\n{names}")
    parts.append(f"【助手回答（节选）】\n{excerpt}")
    return [
        {"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def coerce_suggestions(data: Any, query: str, limit: int = SUGGEST_LIMIT) -> list[str]:
    """把 LLM 的 JSON 校正为最多 ``limit`` 条去重问题。

    兼容 ``questions`` / ``suggestions`` / ``items`` 三种字段名（模型偶尔自由发挥），
    过滤非字符串、空串、过长（> 60 字，明显跑偏）、与原问题重复的条目。
    """
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = None
        for key in ("questions", "suggestions", "items", "result"):
            if isinstance(data.get(key), list):
                raw = data[key]
                break
        if raw is None:
            raw = []
    else:
        return []

    base = _normalize(query or "")
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        q = item.strip().strip("-·•0123456789.、）) ")
        if not q or len(q) > 60:
            continue
        norm = _normalize(q)
        if not norm or norm == base or norm in seen:
            continue
        seen.add(norm)
        out.append(q)
        if len(out) >= limit:
            break
    return out


def get_suggest_client(settings: Settings | None = None) -> LLMClient:
    """追问预测专用客户端：短超时 + 不重试（这是**锦上添花**的调用，不该拖时间）。"""
    from app.generation.llm_client import _get_cached_client

    s = settings or get_settings()
    return _get_cached_client(
        get_llm_api_key(),
        s.llm_base_url,
        s.llm_default_model,
        float(s.suggest_timeout_s),
        s.suggest_max_retries,
        s.llm_thinking or "",
        max_concurrency=getattr(s, "llm_max_concurrency", 8),
        circuit_fail_threshold=getattr(s, "llm_circuit_fail_threshold", 5),
        circuit_cooldown_s=float(getattr(s, "llm_circuit_cooldown_s", 30)),
    )


def generate_suggestions(
    query: str,
    answer: str,
    *,
    law_names: Sequence[str] | None = None,
    client: LLMClient | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """预测 3 条追问；**任何失败都返回空列表**，绝不向上抛异常。

    :param client: 注入 LLM 客户端（测试用假件）
    """
    q = (query or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return []

    s = settings or get_settings()
    if not s.suggest_enabled:
        return []

    started = time.perf_counter()
    try:
        llm = client or get_suggest_client(s)
        resp = llm.chat(
            build_messages(q, a, law_names),
            temperature=0.7,  # 高一点换取三个不同角度，低温度会给出三条近似句
            max_tokens=256,
            json_mode=True,
        )
        data = json.loads(_strip_json_fences(resp.content))
        items = coerce_suggestions(data, q)
        logger.info(
            "追问预测 q=%.30s 条数=%d 耗时=%dms",
            q,
            len(items),
            int((time.perf_counter() - started) * 1000),
        )
        return items
    except Exception as exc:  # noqa: BLE001 - 推荐失败必须静默
        logger.warning(
            "追问预测失败（%s: %s），本次不展示推荐；q=%.30s",
            type(exc).__name__,
            str(exc)[:160],
            q,
        )
        return []
