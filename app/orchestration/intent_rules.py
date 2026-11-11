"""意图识别的规则兜底（LLM 失败/超时时启用，保证链路 100% 可用）。

纯函数、无 IO、无第三方依赖，可直接单测。判定规则与设计方案 §6.1
兜底表逐条对应：

=============  ==========================================================
判定项          规则
=============  ==========================================================
intent          劳动关键词 → labor_law；民法典关键词 → civil_code；
                短问候 → greeting；能力询问 → other；其余 → general
complexity      长度 <15 且无连接词 → simple；含连接词或 15–40 字 → medium；
                长度 >40 或多问号 → complex
rewrite_type    有指代词 → history；否则 direct
hit_structured  命中结构化话题关键词 → true，并给出 topic
=============  ==========================================================
"""

from __future__ import annotations

import re

from app.orchestration.models import (
    COMPLEXITIES,
    INTENT_LABELS,
    LAW_CIVIL,
    LAW_LABOR,
    LAW_LABOR_CONTRACT,
    STRUCTURED_TOPICS,
    IntentResult,
)

# ---------------------------------------------------------------------------
# 关键词表
# ---------------------------------------------------------------------------
#: 劳动法/劳动合同法关键词（含口语说法）
LABOR_KEYWORDS: tuple[str, ...] = (
    "试用期", "试用", "加班", "工资", "薪水", "薪酬", "待遇", "辞退", "裁员",
    "社保", "保险", "工伤", "竞业", "年假", "休假", "仲裁", "劳动", "离职",
    "入职", "辞职", "开除", "补偿", "赔偿", "派遣", "三期", "产假", "病假",
    "绩效", "工时", "调岗", "降薪", "拖欠", "克扣", "押金", "服务期", "保密",
    "集体合同", "劳务", "hr", "老板", "打工人", "上班", "劳动合同", "劳动法",
    "经济补偿", "伤残", "哺乳期", "怀孕", "孕期", "失业",
)

#: 民法典关键词
CIVIL_KEYWORDS: tuple[str, ...] = (
    "婚姻", "结婚", "离婚", "继承", "遗产", "侵权", "物权", "借款", "借钱",
    "租赁", "租房", "遗嘱", "抚养", "赡养", "赠与", "抵押", "担保", "定金",
    "违约金", "监护", "收养", "人格权", "隐私", "名誉", "相邻关系", "共有",
    "宅基地", "承包", "质权", "留置", "债权", "债务", "不当得利", "无因管理",
    "肖像权", "姓名权", "赠与", "合同", "借贷", "夫妻", "财产分割",
)

#: 问候语（整句去标点后完全相等才算，避免"你好"误伤"你好，我想问"）
GREETING_WORDS: tuple[str, ...] = (
    "你好", "您好", "在吗", "谢谢", "多谢", "再见", "拜拜", "嗨", "hello",
    "hi", "早上好", "下午好", "晚上好", "辛苦",
)

#: 能力/身份询问 → other
CAPABILITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^你(能|可以)做什么"),
    re.compile(r"^你是谁"),
    re.compile(r"^介绍?一下?你"),
    re.compile(r"^(有什么|有哪些)功能"),
    re.compile(r"^怎么(使用|用)你"),
)

#: 指代词（触发 history 改写）
ANAPHORA_WORDS: tuple[str, ...] = (
    "那个", "这个", "这些", "那些", "上面", "前述", "刚才", "继续", "还有呢",
    "具体呢", "他呢", "她呢", "它呢", "那", "此",
)

#: 复杂度连接词
COMPLEX_CONNECTORS: tuple[str, ...] = (
    "分别", "以及", "并且", "同时", "还", "怎么算", "如何计算", "计算",
    "依据", "依据是什么", "怎么办", "吗", "呢",
)

#: 多问题特征（多个问号或"……？……？"）
_MULTI_QUESTION_RE = re.compile(r"[?？].*[?？]")

#: 多诉求连接词：问句里出现"还要/同时/以及"等，通常对应两处以上规定
MULTI_ISSUE_CONNECTORS: tuple[str, ...] = ("还", "同时", "以及", "另外", "而且", "并且", "、")

#: 分句分隔符（英文逗号也常见于用户输入）
_CLAUSE_SEPARATORS: tuple[str, ...] = ("，", ",", "；", ";")

#: 子查询拆分分隔符：只按**显式**分隔符切（分号/句号/以及/并且/同时/还有）。
#: 不能按"和"切——"和解协议"这类词会被误切，破坏检索语义。
_SUBQUERY_SPLIT_RE = re.compile(r"[；;。]|以及|并且|同时|还有")

# ---------------------------------------------------------------------------
# 结构化话题 → 触发词（命中即 hit_structured）
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "试用期": ("试用期", "试用"),
    "加班费": ("加班费", "加班工资", "加班", "1.5倍", "1.5 倍", "双倍工资", "三倍"),
    "经济补偿": ("经济补偿", "补偿金", "n+1", "经济赔偿", "赔偿金"),
    "竞业限制": ("竞业", "竞业限制", "保密协议"),
    "劳务派遣": ("派遣", "劳务派遣", "外包"),
    "裁员": ("裁员", "经济性裁员", "倒闭", "优化"),
    "三期保护": ("三期", "孕期", "产假", "哺乳期", "怀孕"),
    "工伤": ("工伤", "伤残", "职业病"),
    "社保": ("社保", "五险", "养老保险", "医疗保险", "公积金", "失业保险"),
    "仲裁时效": ("仲裁时效", "诉讼时效", "一年内"),
    "合同订立": ("不签合同", "没签合同", "未签合同", "不订立", "入职"),
    "合同解除": ("解除", "辞职", "离职", "开除", "辞退"),
}


def normalize_query(query: str) -> str:
    """去空白/标点的轻量归一化，供关键词匹配使用。

    三引号避免字符串内 ASCII 单双引号截断正则字面量。
    """
    return re.sub(
        r'''[\s，。！？、；：“”‘’（）()\[\]【】{}…~·!?.,;:"'`\-_/\\|@#$%^&*+=<>]+''',
        "",
        query or "",
    )


def _count_hits(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for kw in keywords if kw and kw in text)


# ---------------------------------------------------------------------------
# 各判定项
# ---------------------------------------------------------------------------
def rule_intent(query: str) -> tuple[str, float]:
    """返回 ``(intent, confidence)``。confidence 仅供日志，不参与决策。"""
    text = normalize_query(query)
    if not text:
        return "general", 0.0
    if len(text) <= 12 and text in GREETING_WORDS:
        return "greeting", 0.95
    if any(p.search(text) for p in CAPABILITY_PATTERNS):
        return "other", 0.85

    labor_hits = _count_hits(text, LABOR_KEYWORDS)
    civil_hits = _count_hits(text, CIVIL_KEYWORDS)
    # "劳动合同"同时含"合同"（民法典词），劳动优先
    if labor_hits > 0 and labor_hits >= civil_hits:
        return "labor_law", min(0.6 + 0.1 * labor_hits, 0.95)
    if civil_hits > 0:
        return "civil_code", min(0.6 + 0.1 * civil_hits, 0.95)
    return "general", 0.4


def rule_complexity(query: str) -> str:
    text = normalize_query(query)
    if not text:
        return "simple"
    if len(text) > 40 or _MULTI_QUESTION_RE.search(query or ""):
        return "complex"
    if len(text) < 15 and not any(c in text for c in COMPLEX_CONNECTORS):
        return "simple"
    return "medium"


def is_multi_issue(query: str) -> bool:
    """多诉求问句：多个问号，或含分句且带并列/递进连接词。

    与 ``rule_complexity`` 的区别：后者判"难不难"，本函数判"是不是要分别检索
    两处规定"。仅用于检索配额（多跳定点分析：单查询路径下正确法条常落在
    重排第 6-13 名，被最终 topK 截掉）。
    """
    q = (query or "").strip()
    if not q:
        return False
    if q.count("？") + q.count("?") > 1:
        return True
    if not any(sep in q for sep in _CLAUSE_SEPARATORS):
        return False
    return any(c in q for c in MULTI_ISSUE_CONNECTORS)


#: 二次拆解触发信号：只在「第二处诉求/救济」有显式措辞时才值得再发一次拆解调用。
#: 必须收窄到"连接词 + 诉求名词"，不能沿用 :data:`MULTI_ISSUE_CONNECTORS` 的裸
#: 单字「还」——"劳动关系还在""离职后还要遵守吗"只有一个诉求，误触发会把单跳
#: 问句改成多路检索，反而丢召回（多跳定点：q75 由 1.0 掉到 0.5）。
_SECOND_DEMAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:还要|还不|还有|另外|而且|并且)(?:补偿|赔偿|钱|主张|要求|权利|责任|待遇)"),
    re.compile(r"同时(?:主张|要求|获得|追究|起诉|承担)"),
    re.compile(r"主张(?:什么|哪些)|(?:哪些|什么)(?:赔偿|权利|钱)"),
    re.compile(r"申请撤销"),
)


def needs_second_decompose(query: str) -> bool:
    """单句是否疑似含两处以上诉求，值得再发一次拆解调用（仅首拆未拆时使用）。

    与 :func:`is_multi_issue` 的分工：后者判"要不要放大检索配额"，取宽口径；
    本函数判"要不要再花一次 LLM 调用重拆"，取窄口径——宁可漏拆也不误拆，
    因为重拆会换掉检索入口，代价比放大 topK 大得多。
    """
    q = (query or "").strip()
    if not q:
        return False
    if q.count("？") + q.count("?") > 1:
        return True
    return any(p.search(q) for p in _SECOND_DEMAND_PATTERNS)


def rule_rewrite_type(query: str, history: list[dict] | None = None) -> str:
    """有指代词 → history；其余 → direct（与 §6.1 兜底表一致）。"""
    text = normalize_query(query)
    if any(w in text for w in ANAPHORA_WORDS):
        return "history"
    return "direct"


def rule_sub_queries(query: str) -> list[str]:
    """规则兜底的多跳拆解：LLM 降级时尽量保住 decompose 能力。

    按显式分隔符（；;。/以及/并且/同时/还有）和多个问号切 2-3 条子查询，
    每条至少 4 个字符；切不出 ≥2 条有效子查询就退回原句单查询。
    保守策略：宁可不拆，也不把"和解协议"这类词切碎。
    """
    q = (query or "").strip()
    if not q:
        return []
    parts: list[str] = []
    for seg in _SUBQUERY_SPLIT_RE.split(q):
        parts.extend(x.strip() for x in re.split(r"[?？]", seg))
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        part = part.strip(" ，,、")
        if len(part) >= 4 and part not in seen:
            seen.add(part)
            out.append(part)
        if len(out) >= 3:
            break
    return out if len(out) >= 2 else ([q] if q else [])


def rule_structured_topic(query: str) -> str | None:
    """命中结构化话题返回话题名，否则 None。长关键词优先匹配。"""
    text = normalize_query(query)
    if not text:
        return None
    best: tuple[int, str | None] = (0, None)  # (命中关键词长度, 话题)
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text and len(kw) > best[0]:
                best = (len(kw), topic)
    return best[1]


def rule_law_hints(intent: str) -> list[str]:
    if intent == "labor_law":
        return [LAW_LABOR_CONTRACT, LAW_LABOR]
    if intent == "civil_code":
        return [LAW_CIVIL]
    return list(STANDBY_LAWS)


#: general 意图不预设法律，交给全库检索
STANDBY_LAWS: tuple[str, ...] = (LAW_LABOR_CONTRACT, LAW_LABOR, LAW_CIVIL)


def rule_based_intent(query: str, history: list[dict] | None = None) -> IntentResult:
    """完整规则兜底：产出与 LLM 路径同构的 :class:`IntentResult`。"""
    query = (query or "").strip()
    intent, _conf = rule_intent(query)
    complexity = rule_complexity(query)
    rewrite_type = rule_rewrite_type(query, history)
    topic = rule_structured_topic(query)
    need_history = rewrite_type == "history" and bool(history)
    return IntentResult(
        intent=intent,
        rewrite_type=rewrite_type,
        sub_queries=rule_sub_queries(query) or ([query] if query else []),
        complexity=complexity if complexity in COMPLEXITIES else "medium",
        hit_structured=topic is not None,
        structured_topic=topic if topic in STRUCTURED_TOPICS else None,
        law_hints=rule_law_hints(intent) if intent != "greeting" else [],
        need_history=need_history,
        source="rules",
        raw=None,
    )
