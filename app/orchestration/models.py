"""编排层数据模型：意图识别结果与路由决策。

这些模型横跨 intent / router / service / api 四层，单独成文件避免循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: 意图五分类（设计方案 §6.1）
INTENT_LABELS = ("labor_law", "civil_code", "general", "greeting", "other")

#: 查询改写类型
REWRITE_TYPES = ("direct", "history", "hyde", "simplify", "decompose")

#: 问题复杂度
COMPLEXITIES = ("simple", "medium", "complex")

#: 路由六分支（设计方案 §6.3）
ROUTES = ("greeting", "pure_structured", "pure_vector", "hybrid", "cross_law", "no_data")

#: 高频问答话题（与 qa_pairs.category 对齐）
STRUCTURED_TOPICS = (
    "试用期",
    "加班费",
    "经济补偿",
    "竞业限制",
    "劳务派遣",
    "裁员",
    "三期保护",
    "工伤",
    "社保",
    "仲裁时效",
    "合同订立",
    "合同解除",
)

#: 语料覆盖的七部法律名（与 Milvus/MySQL 里的 title 前缀一致）
LAW_LABOR_CONTRACT = "中华人民共和国劳动合同法"
LAW_LABOR = "中华人民共和国劳动法"
LAW_CIVIL = "中华人民共和国民法典"
LAW_INJURY = "工伤保险条例"
LAW_ARBITRATION = "中华人民共和国劳动争议调解仲裁法"
LAW_CONTRACT_IMPL = "中华人民共和国劳动合同法实施条例"
LAW_ANNUAL_LEAVE = "职工带薪年休假条例"
ALL_LAWS = (
    LAW_LABOR_CONTRACT,
    LAW_LABOR,
    LAW_CIVIL,
    LAW_INJURY,
    LAW_ARBITRATION,
    LAW_CONTRACT_IMPL,
    LAW_ANNUAL_LEAVE,
)


@dataclass(frozen=True)
class IntentResult:
    """一次意图识别的产出。``source`` 标记来自 LLM 还是规则兜底。"""

    intent: str = "general"
    rewrite_type: str = "direct"
    sub_queries: list[str] = field(default_factory=list)
    complexity: str = "medium"
    hit_structured: bool = False
    structured_topic: str | None = None
    law_hints: list[str] = field(default_factory=list)
    need_history: bool = False
    source: str = "rules"  # "llm" | "rules"
    raw: dict | None = None  # LLM 原始 JSON，调试/回归用
    #: JSON 校验问题：``schema:*``（缺字段/类型错）与 ``business:*``（枚举越界，
    #: 已回落默认值）。解析失败不进这里——那是 LLMResponseError(parse_error)。
    validation_errors: list[str] = field(default_factory=list)
    #: LLM 调用成本（命中缓存/规则兜底时为空）
    token_usage: dict = field(default_factory=dict)
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "rewrite_type": self.rewrite_type,
            "sub_queries": list(self.sub_queries),
            "complexity": self.complexity,
            "hit_structured": self.hit_structured,
            "structured_topic": self.structured_topic,
            "law_hints": list(self.law_hints),
            "need_history": self.need_history,
            "source": self.source,
            "validation_errors": list(self.validation_errors),
            "token_usage": dict(self.token_usage or {}),
            "latency_ms": int(self.latency_ms or 0),
        }


@dataclass(frozen=True)
class RouteDecision:
    """路由矩阵的一格：决定走哪些数据源。"""

    route: str  # ROUTES 之一
    kb_ids: list[str] = field(default_factory=list)  # 检索的知识库（空=不过滤）
    use_mysql_qa: bool = False  # 是否查 qa_pairs 结构化链路
    use_milvus: bool = False  # 是否走向量检索
    reason: str = ""  # 人类可读的决策理由（调试/日志）
