"""意图识别 + Query 改写（单次 LLM 调用，JSON 模式）。

数据流::

    query + history
      → LLM（deepseek-v4.1-flash, JSON 模式, 3s 超时, 重试 1 次）
      → _validate() 字段校正（非法值回落安全默认）
      → IntentResult(source="llm")
      —— 任一环节失败（缺 key/超时/网络/JSON 解析失败）——
      → rule_based_intent() 规则兜底 → IntentResult(source="rules")

兜底保证链路 100% 可用（设计方案 §6.1 量化目标：降级成功率 100%）。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Sequence

from app.core.logger import get_logger
from app.generation.llm_client import LLMClient, LLMResponseError
from app.orchestration.intent_rules import rule_based_intent
from app.orchestration.models import (
    ALL_LAWS,
    COMPLEXITIES,
    INTENT_LABELS,
    REWRITE_TYPES,
    STRUCTURED_TOPICS,
    IntentResult,
)
from config import Settings, get_llm_api_key, get_settings

logger = get_logger("lawanswer.intent")

#: 意图 prompt 版本：模板或字段语义变化时 +1，旧意图缓存随之失效。
INTENT_PROMPT_VERSION = "2026-11-11.1"

#: 一次 LLM 调用同时产出 6 类信息的 system prompt（设计方案 §6.1）
INTENT_SYSTEM_PROMPT = """你是法律问答系统的「意图识别与查询改写」模块。根据用户问题（可附最近对话历史），一次性输出 JSON 对象。只输出 JSON，不要输出解释、不要用 Markdown 代码块。

字段与取值：
{
  "intent": "labor_law | civil_code | general | greeting | other",
  "rewrite_type": "direct | history | hyde | simplify | decompose",
  "sub_queries": ["1-3 条检索用查询"],
  "complexity": "simple | medium | complex",
  "hit_structured": true,
  "structured_topic": "试用期 | 加班费 | 经济补偿 | 竞业限制 | 劳务派遣 | 裁员 | 三期保护 | 工伤 | 社保 | 仲裁时效 | 合同订立 | 合同解除 | null",
  "law_hints": ["中华人民共和国劳动合同法"],
  "need_history": false
}

判定规则：
- intent：劳动/劳动合同/工伤/仲裁等 → labor_law；民法典领域（合同、婚姻、继承、侵权、物权、借款、租赁、遗嘱）→ civil_code；跨两部以上或泛法律 → general；纯问候寒暄（你好/在吗/谢谢等，不含任何实质问题）→ greeting；**与法律完全无关的生活话题（天气、影视、美食、闲聊）→ other 而非 greeting**。注意"你能做什么/有哪些功能/你是谁"属能力与身份询问，归 other。
- rewrite_type：
  - direct：问题完整、无指代、单跳；
  - history：含"那/这个/具体/继续/还有呢"等指代，需结合历史补全实体；
  - hyde：口语化且术语缺失（如"老板不给钱怎么办"），生成一段假设性法条表述用于检索；
  - simplify：问题过长且混杂多意图，抽取主干；
  - decompose：完整回答需要分别检索两条以上不同法条或不同法律时使用，拆成 2-3 个子查询。**不要求出现"分别/以及"**：只有一个问句但含两个并列诉求（如"能同时要赔偿金和经济补偿吗""工伤医疗费谁付、能不能仲裁""年假没休也没给钱怎么办"）也要拆。
- sub_queries：把口语改写成法律术语表述，保留关键实体与数字；除 decompose 外通常 1 条；改写结果要能独立读懂。
- complexity：单点短问 → simple；含条件/多要素 → medium；多问、跨条、含计算（要按工资/伤残等级/年限套公式算出金额）→ complex。
- hit_structured：**仅当问题命中上述 12 个劳动法高频话题之一且用户在问该话题本身的规定/标准/计算时 → true** 并给出 topic。民法典问题（婚姻、继承、物权、租赁等）即使涉及"合同"字样也一律 false——12 个话题全部对齐劳动法问答库，civil 问题没有结构化答案可回放。拿不准 → false。
- law_hints：从「中华人民共和国劳动合同法、中华人民共和国劳动法、中华人民共和国民法典、工伤保险条例、中华人民共和国劳动争议调解仲裁法、中华人民共和国劳动合同法实施条例、职工带薪年休假条例」中选 1-3 部（按问题实际涉及的法域选，不要只选劳动合同法）。对应关系：工伤认定/待遇/伤残补助/工亡 → 工伤保险条例；仲裁时效/受理/裁决/终局裁决/举证 → 中华人民共和国劳动争议调解仲裁法；二倍工资起算/经济补偿月工资口径/无固定期限合同细则 → 中华人民共和国劳动合同法实施条例；年休假天数/未休补偿 → 职工带薪年休假条例。跨法域可选 2-3 部（如工伤且未缴社保 → 工伤保险条例 + 中华人民共和国劳动合同法）。greeting/other 输出 []。
- need_history：是否需要结合对话历史才能理解 → true/false。"""


def _format_history(history: Sequence[dict] | None, max_turns: int) -> str:
    """把最近 N 轮对话压成文本行；无历史返回空串。"""
    if not history or max_turns <= 0:
        return ""
    lines: list[str] = []
    for msg in list(history)[-(max_turns * 2) :]:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        speaker = "用户" if role == "user" else "助手"
        lines.append(f"{speaker}：{content[:200]}")
    return "\n".join(lines)


def _build_messages(query: str, history: Sequence[dict] | None, max_turns: int) -> list[dict]:
    history_text = _format_history(history, max_turns)
    user_parts: list[str] = []
    if history_text:
        user_parts.append(f"【最近对话】\n{history_text}")
    user_parts.append(f"【用户问题】\n{query.strip()}")
    return [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _strip_json_fences(text: str) -> str:
    """容忍 ```json 包裹与前后杂讯：截取第一个 { 到最后一个 }。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json|JSON)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def _coerce_sub_queries(value: Any, query: str) -> list[str]:
    """LLM 可能给字符串/空列表/超长/重复——统一校正为 1-3 条非空去重查询。"""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return [query] if query.strip() else []
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        q = item.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) >= 3:
            break
    return out or ([query] if query.strip() else [])


def _validate(data: dict, query: str) -> IntentResult:
    """把 LLM 原始 JSON 校正为合法的 :class:`IntentResult`。

    非法字段不抛错，逐一回落到安全默认值——单字段抽风不应让整次识别失败。
    """
    intent = data.get("intent")
    intent = intent if intent in INTENT_LABELS else "general"

    rewrite_type = data.get("rewrite_type")
    rewrite_type = rewrite_type if rewrite_type in REWRITE_TYPES else "direct"

    complexity = data.get("complexity")
    complexity = complexity if complexity in COMPLEXITIES else "medium"

    sub_queries = _coerce_sub_queries(data.get("sub_queries"), query)

    hit_structured = bool(data.get("hit_structured"))
    topic = data.get("structured_topic")
    if topic not in STRUCTURED_TOPICS:
        topic = None
        # topic 不在枚举内时不能声称命中
        if hit_structured and data.get("structured_topic"):
            logger.warning("LLM 返回未知 structured_topic=%r，按未命中处理", data.get("structured_topic"))

    law_hints = [law for law in (data.get("law_hints") or []) if law in ALL_LAWS]
    if intent in {"labor_law", "civil_code", "general"} and not law_hints:
        # LLM 漏给 law_hints：按 intent 兜底，避免检索范围失控
        from app.orchestration.intent_rules import rule_law_hints

        law_hints = rule_law_hints(intent)

    return IntentResult(
        intent=intent,
        rewrite_type=rewrite_type,
        sub_queries=sub_queries,
        complexity=complexity,
        hit_structured=hit_structured,
        structured_topic=topic,
        law_hints=law_hints,
        need_history=bool(data.get("need_history")),
        source="llm",
        raw=data,
    )


def _recognize_via_llm(
    query: str,
    *,
    client: LLMClient,
    history: Sequence[dict] | None,
    max_turns: int,
) -> IntentResult:
    messages = _build_messages(query, history, max_turns)
    resp = client.chat(messages, json_mode=True, temperature=0.1)
    data = json.loads(_strip_json_fences(resp.content))
    if not isinstance(data, dict):
        raise LLMResponseError(f"LLM JSON 顶层不是对象: {type(data).__name__}")
    return _validate(data, query)


def _build_intent_client(
    api_key: str,
    base_url: str,
    model: str,
    timeout_s: float,
    max_retries: int,
    thinking: str,
) -> LLMClient:
    from app.generation.llm_client import _get_cached_client

    return _get_cached_client(api_key, base_url, model, timeout_s, max_retries, thinking)


def get_intent_client(settings: Settings | None = None) -> LLMClient:
    """意图专用客户端：短超时 + 重试 1 次，快速失败走规则兜底。

    ``thinking`` 必须与主链路一致下发：DeepSeek 的思考模式默认开启，若不显式
    disabled，意图调用会跑到 3-10s（hidden reasoning 计入 completion_tokens），
    既超时又让 temperature 失效。
    """
    s = settings or get_settings()
    return _build_intent_client(
        get_llm_api_key(),
        s.llm_base_url,
        s.llm_default_model,
        float(s.intent_timeout_s),
        s.intent_max_retries,
        s.llm_thinking or "",
    )


def reset_intent_client() -> None:
    from app.generation.llm_client import reset_llm_client

    reset_llm_client()


# ---------------------------------------------------------------------------
# 意图结果缓存：同一问题（无历史）重复提问时跳过 ~1s 的 LLM 调用
# ---------------------------------------------------------------------------

def _intent_cache_key(query: str, s: Settings | None = None) -> str:
    """归一化后取 SHA1；key 带 prompt 版本与模型名，换模型/改 prompt 即失效。"""
    from app.orchestration.qa_cache import normalize_question

    settings = s or get_settings()
    norm = normalize_question(query)
    raw = f"v2|{INTENT_PROMPT_VERSION}|{settings.llm_default_model}|{norm}"
    return "intent:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _intent_cache_get(query: str, s: Settings) -> IntentResult | None:
    """读意图缓存；Redis 故障/解析失败一律返回 None（fail-open，不影响链路）。"""
    try:
        from app.db.redis_client import get_redis

        raw = get_redis(s).get(_intent_cache_key(query, s))
    except Exception:  # noqa: BLE001 - 缓存不可用就当没命中
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
        result = IntentResult(**data)
        # 命中缓存也要能区分"真调了 LLM"还是"复用结果"
        object.__setattr__(result, "source", "intent_cache")
        return result
    except Exception:  # noqa: BLE001 - 脏数据当未命中
        return None


def _intent_cache_put(query: str, result: IntentResult, s: Settings) -> None:
    """只缓存 LLM 成功结果（source=llm）；规则兜底/降级结果不缓存。"""
    try:
        from app.db.redis_client import get_redis

        get_redis(s).set(
            _intent_cache_key(query, s),
            json.dumps(result.to_dict(), ensure_ascii=False),
            ex=s.intent_cache_ttl,
        )
    except Exception:  # noqa: BLE001 - 写失败不影响主链路
        pass


#: 纯问候语模式：整句只由问候词 + 语气词/标点构成才算 greeting。
#: 必须整句匹配，不能 startswith 前缀匹配——"请问试用期被辞退有补偿吗"这类
#: "礼貌前缀 + 真问题"会被误杀成固定话术（test_qa_cache / test_intent_rules
#: 都认定"请问，"是正常法律问题的前缀）。
_GREETING_RE = re.compile(
    r"^(?:你好|您好|嗨|哈喽|hello|hi|hey|在吗|谢谢|多谢)[呀啊哈!！?？。.~～\s]*$",
    re.IGNORECASE,
)


def _fast_rule_precheck(query: str) -> IntentResult | None:
    """快速规则前置：**整句纯问候语**直接路由 greeting，跳过 LLM 意图识别（~1s）。

    匹配上返回 IntentResult(source="fast_rules")；没匹配上返回 None，继续走 LLM 路径。
    只做问候语短路：法律问题关键词不前置——意图识别结果影响路由判断
    （hybrid/pure_structured/pure_vector），规则猜测的 intent 与 LLM 实际判断
    可能不一致，破坏路由逻辑。高频问题的正确省时手段是 Redis 缓存意图结果。
    """
    q = query.strip()
    if not q:
        return None

    if _GREETING_RE.match(q):
        return IntentResult(
            intent="greeting",
            rewrite_type="direct",
            sub_queries=[q],
            complexity="simple",
            hit_structured=False,
            structured_topic=None,
            law_hints=[],
            need_history=False,
            source="fast_rules",
        )
    return None


def recognize_intent(
    query: str,
    *,
    history: Sequence[dict] | None = None,
    client: LLMClient | None = None,
    settings: Settings | None = None,
) -> IntentResult:
    """意图识别入口：快速规则前置 → LLM → 规则兜底。

    :param query: 用户当前问题
    :param history: 最近对话，``[{"role": "user"|"assistant", "content": "..."}]``
    :param client: 注入 LLM 客户端（测试用 mock）
    """
    query = (query or "").strip()
    if not query:
        return rule_based_intent(query, list(history) if history else None)

    # 0) 快速规则前置：常见问题直接分类，跳过 LLM（~1s）
    #    注意：显式注入 client 时跳过（测试场景要测 LLM 路径）
    if client is None:
        pre = _fast_rule_precheck(query)
        if pre is not None:
            logger.info(
                "意图识别(快速规则) q=%.40s intent=%s complexity=%s source=fast_rules",
                query, pre.intent, pre.complexity,
            )
            return pre

    s = settings or get_settings()
    # 0.5) 意图结果缓存：无历史的重复问题直接复用上次 LLM 结果，省 ~1s。
    #      带历史的问题意图依赖上下文，不缓存也不读缓存（key 只有 query）。
    use_cache = client is None and not history and s.intent_cache_ttl > 0
    if use_cache:
        cached = _intent_cache_get(query, s)
        if cached is not None:
            logger.info(
                "意图识别(缓存) q=%.40s intent=%s complexity=%s 命中 intent 缓存",
                query, cached.intent, cached.complexity,
            )
            return cached

    started = time.perf_counter()
    try:
        llm = client or get_intent_client(s)
        result = _recognize_via_llm(
            query,
            client=llm,
            history=history,
            max_turns=s.intent_history_turns,
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info(
            "意图识别(LLM) q=%.40s intent=%s rewrite=%s complexity=%s topic=%s 耗时=%dms",
            query,
            result.intent,
            result.rewrite_type,
            result.complexity,
            result.structured_topic,
            elapsed,
        )
        if use_cache:
            _intent_cache_put(query, result, s)
        return result
    except Exception as exc:  # noqa: BLE001 - 兜底必须吞掉一切 LLM 异常
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "意图识别 LLM 失败（%s: %s），降级规则兜底；q=%.40s 耗时=%dms",
            type(exc).__name__,
            str(exc)[:120],
            query,
            elapsed,
        )
        return rule_based_intent(query, list(history) if history else None)
