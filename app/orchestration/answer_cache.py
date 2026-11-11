"""答案级缓存：把"实际生成过的整段回答"按问题哈希落 Redis。

为什么需要这一层（与 :mod:`app.orchestration.qa_cache` 的分工）::

    qa_cache   只回放**问答库里预置**的标准答案（65 条 / 325 种问法）
               命中前提是先过余弦门或字面双门 —— 库里没有的问题永远过不了
    answer_cache  回放**用户真实问过并已生成成功**的回答，键就是问题自身
               的 sha1，不需要任何相似度判定 → 第二次提同一问题必然命中

没有这一层时的真实故障：用户问「申请支付令需要准备哪些材料」，BM25 召回的
top1 是别的问题，字面门不达标 → 不写缓存 → 第二次提问仍走完整链路
（意图识别 1–2s + 检索 + 重排 + LLM 十几秒）。"秒答"只在问答库恰好覆盖
该问法时才成立，这与用户感知的"同样的问题问两次应该一样快"完全脱节。

安全边界（缓存直出等于跳过 LLM，错一次就是答非所问）：

1. **只在问题自包含时命中**：带指代（"那这个呢""还有呢"）的问题答案依赖
   上文，缓存回来必然张冠李戴；
2. **多轮对话要求"确实重复问过"**：``history`` 非空时，只有当前问题在历史
   用户消息中出现过才命中 —— 这正是"同一对话里重复提问"的定义，也排除了
   "同一句话但用户中途补充了新事实"的情形；
3. **只缓存完整、无错误的生成**：断流 / LLM 报错 / 无资料来源一律不写，
   否则一条残缺回答会被固化 7 天；
4. **指代问题、纯结构化、问候、无数据引导话术都不写**（后三者已有各自的
   确定性路径，写进来只是重复占内存）。

Redis key::

    qa:gen:{sha1(normalize_question(query))} → JSON（TTL 见 cache_answer_ttl）
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.access import CACHE_SCHEMA_VERSION
from app.core.logger import get_logger
from app.db.redis_client import get_redis
from app.orchestration.qa_cache import normalize_question
from config import Settings, get_settings

logger = get_logger("lawanswer.answer_cache")

#: 答案缓存 key 前缀（与 qa:exact 区分：那边是库里标准答案，这边是生成结果）
REDIS_ANSWER_PREFIX = "qa:gen:"

#: 依赖上文的指代词 / 追问用语。出现任一个就不进缓存、也不从缓存命中。
#: 注意：这里的词必须能扛住 ``normalize_question``（它会去掉"呢/吗/的/了"
#: 等停用词），所以写"然后"而不是"然后呢"。
_PRONOUNS: tuple[str, ...] = (
    "这个", "那个", "这条", "那条", "这部", "那部", "这家", "那家",
    "这些", "那些", "这样", "那样", "上述", "上面", "前面说", "刚才说",
    "该公司", "该条", "此条", "该案", "此案", "该法", "此法",
    "它", "他们", "她们", "然后", "接着说",
    # 多轮实测漏判的追问（dev/holdout 回放发现）：
    # "那如果是我主动辞职呢？" 归一化后不含旧表任何指代词
    "那如果", "如果我", "那我",
)

#: 条件覆盖型追问：换了计算口径或追问 N+1，答案依赖上文前提
#: 例："工资按8000算，N+1是多少？"
_CONDITION_OVERRIDE_RE = re.compile(r"按\s*\d+\s*[算计]|n\s*[+＋]\s*1", re.IGNORECASE)

#: 自包含问题的最短归一化长度（"工伤"两个字太泛，答案要结合具体情形）
_MIN_SELF_CONTAINED_CHARS = 8

_WS_RE = re.compile(r"\s+")


def answer_key(norm_query: str, cache_identity: str = "") -> str:
    """缓存 key。

    ``cache_identity`` 由 :func:`app.core.access.answer_identity` 生成，
    包含租户/知识库/版本/模型/prompt/as-of。为空时保留旧 key 形态，
    仅用于离线单测；线上链路始终传非空值，因此旧格式缓存不会被读到。
    """
    if not cache_identity:
        return REDIS_ANSWER_PREFIX + hashlib.sha1(norm_query.encode("utf-8")).hexdigest()
    raw = f"v{CACHE_SCHEMA_VERSION}|{cache_identity}|{norm_query}"
    return REDIS_ANSWER_PREFIX + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def is_self_contained(query: str) -> bool:
    """问题是否自包含（不依赖上文即可独立回答）。

    只做保守判断：出现指代词直接判否，长度不足也判否。
    """
    norm = normalize_question(query)
    if len(norm) < _MIN_SELF_CONTAINED_CHARS:
        return False
    if _CONDITION_OVERRIDE_RE.search(norm):
        return False
    return not any(p in norm for p in _PRONOUNS)


def _norm_text(text_: str) -> str:
    return _WS_RE.sub("", (text_ or "").strip())


@dataclass(frozen=True)
class CachedAnswer:
    """一条已生成回答的缓存快照。"""

    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    route: str = ""
    intent: str = ""
    rewrite_type: str = "direct"
    complexity: str = "simple"
    created_at: int = 0
    cache_meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "route": self.route,
            "intent": self.intent,
            "rewrite_type": self.rewrite_type,
            "complexity": self.complexity,
            "created_at": self.created_at,
            "cache_meta": self.cache_meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CachedAnswer":
        return cls(
            question=str(data.get("question", "")),
            answer=str(data.get("answer", "")),
            sources=list(data.get("sources") or []),
            route=str(data.get("route", "")),
            intent=str(data.get("intent", "")),
            rewrite_type=str(data.get("rewrite_type", "direct")),
            complexity=str(data.get("complexity", "simple")),
            created_at=int(data.get("created_at", 0)),
            cache_meta=dict(data.get("cache_meta") or {}),
        )


class _NullRedis:
    """Redis 不可用时的占位：读写全部静默失败（缓存故障不能拖垮主链路）。"""

    def get(self, *_a: Any, **_k: Any) -> None:
        return None

    def set(self, *_a: Any, **_k: Any) -> Any:
        return None

    def expire(self, *_a: Any, **_k: Any) -> Any:
        return None


class AnswerCache:
    """答案级缓存的读写门面。依赖可注入，测试用假实现替换。"""

    def __init__(self, *, redis: Any = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._redis = redis

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def redis(self) -> Any:
        if self._redis is None:
            try:
                self._redis = get_redis(self._settings)
            except Exception as exc:  # noqa: BLE001 - Redis 不可用则整层禁用
                logger.warning("Redis 连接失败，答案缓存降级为不缓存: %s", exc)
                self._redis = _NullRedis()
        return self._redis

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def lookup(
        self,
        query: str,
        *,
        history: list[dict] | None = None,
        cache_identity: str = "",
    ) -> CachedAnswer | None:
        """命中返回快照，否则 None（None 即"照常走完整链路"，不是错误）。

        ``cache_identity`` 必须与写入时一致：租户/知识库/语料版本/模型/
        prompt/as-of 任一变化都视为不同问题，旧缓存自然不可达。
        """
        if not self.settings.cache_answer_enabled:
            return None
        query = (query or "").strip()
        if not query:
            return None
        norm = normalize_question(query)
        if not norm:
            return None
        # 指代式追问永不命中：答案依赖上文，回放必然错位
        if not is_self_contained(query):
            return None
        # 多轮对话：只有"历史上确实问过同一句"才算重复提问
        if history and not _repeated_in_history(norm, history):
            return None

        try:
            raw = self.redis.get(answer_key(norm, cache_identity))
        except Exception as exc:  # noqa: BLE001
            logger.warning("答案缓存读取失败，按未命中处理: %s", exc)
            return None
        if not raw:
            return None
        try:
            cached = CachedAnswer.from_dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("答案缓存内容损坏，忽略: %s", exc)
            return None
        if not cached.answer.strip():
            return None
        # 旧格式（无 cache_meta）一律视为未命中：上线即失效，避免跨租户回放。
        meta = cached.cache_meta
        if int(meta.get("schema") or 0) != CACHE_SCHEMA_VERSION:
            logger.info("答案缓存 schema 不匹配，忽略旧格式 q=%.40s", query)
            return None
        if str(meta.get("identity", "")) != cache_identity:
            logger.info("答案缓存身份不匹配，忽略 q=%.40s", query)
            return None
        self._touch(norm, cache_identity)
        logger.info("答案缓存命中 q=%.40s route=%s 跳过意图/检索/LLM", query, cached.route)
        return cached

    def _touch(self, norm: str, cache_identity: str = "") -> None:
        """命中续期：热问题不能在第 7 天准时失效。"""
        ttl = self.settings.cache_answer_ttl
        if ttl <= 0:
            return
        try:
            self.redis.expire(answer_key(norm, cache_identity), ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("答案缓存续期失败: %s", exc)

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def store(
        self,
        query: str,
        *,
        answer: str,
        sources: list[dict] | None = None,
        route: str = "",
        intent: str = "",
        rewrite_type: str = "direct",
        complexity: str = "simple",
        error: str | None = None,
        cache_identity: str = "",
        cache_meta: dict | None = None,
    ) -> bool:
        """写入一条已生成回答。返回是否真的写了（便于测试与日志）。"""
        if not self.settings.cache_answer_enabled:
            return False
        if error:
            return False  # 断流/报错的残缺回答绝不固化
        query = (query or "").strip()
        answer = (answer or "").strip()
        if not query or len(answer) < self.settings.cache_answer_min_chars:
            return False
        if not is_self_contained(query):
            logger.info("问题依赖上文，不写入答案缓存 q=%.40s", query)
            return False
        norm = normalize_question(query)
        if not norm:
            return False
        snapshot = CachedAnswer(
            question=query,
            answer=answer,
            sources=list(sources or []),
            route=route,
            intent=intent,
            rewrite_type=rewrite_type or "direct",
            complexity=complexity or "simple",
            created_at=int(time.time()),
            cache_meta={
                "schema": CACHE_SCHEMA_VERSION,
                "identity": cache_identity,
                **dict(cache_meta or {}),
            },
        )
        try:
            self.redis.set(
                answer_key(norm, cache_identity),
                json.dumps(snapshot.to_dict(), ensure_ascii=False),
                ex=self.settings.cache_answer_ttl,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("答案缓存写入失败: %s", exc)
            return False
        logger.info("答案缓存已写入 q=%.40s route=%s 字数=%d", query, route, len(answer))
        return True


def _repeated_in_history(norm: str, history: list[dict]) -> bool:
    """历史用户消息里是否出现过同一句（归一化后比对）。"""
    for msg in reversed(history or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        if normalize_question(str(msg.get("content", ""))) == norm:
            return True
    return False


def get_answer_cache(*, redis: Any = None) -> AnswerCache:
    """进程级单例（Redis 懒加载；传 redis 时返回非共享实例，供测试注入）。"""
    if redis is not None:
        return AnswerCache(redis=redis)
    return _build_answer_cache()


_CACHE: AnswerCache | None = None


def _build_answer_cache() -> AnswerCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = AnswerCache()
    return _CACHE


def reset_answer_cache() -> None:
    global _CACHE
    _CACHE = None
