"""一次问答请求的全链路 trace。

设计目标（LA-O4）：

* 一个 ``trace_id`` 串联 cache / intent / rewrite / route / retrieve / rerank /
  generate / citation_check / persistence；
* 每段记录模型、Prompt 版本、语料/索引版本、token、成本、耗时、缓存命中、
  重试与降级原因，事后可还原完整调用顺序；
* **不写密钥、完整文档正文或可识别用户隐私**：字段名命中黑名单的值一律打码，
  字符串值截断，问题与回答正文永不进入 trace。

存储：Redis（key ``trace:{trace_id}``，默认 7 天，失败 fail-open）。
生产若未配置 Redis，trace 仍会随 ``done`` 事件返回并写一行结构化日志，
不影响主链路。
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.core.logger import get_logger

logger = get_logger("lawanswer.trace")

#: 字段名命中即打码（大小写不敏感）；宁可多打，不可漏 key
_SENSITIVE_KEYS = ("api_key", "apikey", "authorization", "password", "secret", "token_key")
#: 字符串值最大长度：trace 只留可判定信息，不留正文
_MAX_STR = 160


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def cost_fields(
    tokens: dict | None, *, prompt_per_1k: float, completion_per_1k: float
) -> dict:
    """把 token 用量换算成可比较的成本字段（人民币元）。

    只做估算：单价来自配置，默认值仅用于横向比较，不作为账单依据。
    """
    tokens = dict(tokens or {})
    prompt = int(tokens.get("prompt_tokens") or 0)
    completion = int(tokens.get("completion_tokens") or 0)
    prompt_cost = prompt / 1000 * float(prompt_per_1k or 0)
    completion_cost = completion / 1000 * float(completion_per_1k or 0)
    return {
        "cost": {
            "currency": "CNY",
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt": round(prompt_cost, 6),
            "completion": round(completion_cost, 6),
            "total": round(prompt_cost + completion_cost, 6),
        }
    }


def _sanitize_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(bad in lowered for bad in _SENSITIVE_KEYS):
        return "***"
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(key, item) for item in value][:20]
    if isinstance(value, dict):
        return {str(k): _sanitize_value(str(k), v) for k, v in list(value.items())[:20]}
    return str(value)[:_MAX_STR]


@dataclass
class Trace:
    """请求级 trace（线程内使用；不跨请求共享）。"""

    trace_id: str = field(default_factory=new_trace_id)
    user_id: int | None = None
    events: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record(self, stage: str, *, latency_ms: int = 0, **fields: Any) -> dict:
        """直接写一条已完成的阶段记录（用于无法用上下文管理器包住的调用）。"""
        event = {
            "stage": stage,
            "latency_ms": int(latency_ms or 0),
            "ok": bool(fields.pop("ok", True)),
            **{str(k): _sanitize_value(str(k), v) for k, v in fields.items()},
        }
        self.events.append(event)
        return event

    @contextmanager
    def stage(self, stage: str, **fields: Any) -> Iterator[dict]:
        """包住一段调用：自动记录耗时、成功/异常类型，并把阶段字段写进 trace。"""
        started = time.perf_counter()
        event_fields = {str(k): _sanitize_value(str(k), v) for k, v in fields.items()}
        try:
            yield event_fields
        except Exception as exc:
            event_fields["error"] = type(exc).__name__
            event_fields["ok"] = False
            raise
        else:
            event_fields["ok"] = True
        finally:
            event_fields["stage"] = stage
            event_fields["latency_ms"] = int((time.perf_counter() - started) * 1000)
            self.events.append(event_fields)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "started_at": round(self.started_at, 3),
            "total_ms": int((time.time() - self.started_at) * 1000),
            "events": list(self.events),
        }

    def log(self) -> None:
        """一行结构化日志（JSON），不含问题/回答正文。"""
        try:
            logger.info("trace %s", json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")))
        except Exception:  # noqa: BLE001 - 日志失败不影响请求
            pass

    def save(self) -> None:
        """写 Redis（TTL 可配，失败 fail-open）；供 trace 查询接口还原。"""
        try:
            from app.db.redis_client import get_redis
            from config import get_settings

            settings = get_settings()
            if not getattr(settings, "trace_store_enabled", True):
                return
            get_redis(settings).setex(
                f"trace:{self.trace_id}",
                max(int(settings.trace_ttl_s), 60),
                json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")),
            )
        except Exception:  # noqa: BLE001 - trace 持久化失败不影响请求
            pass


def load_trace(trace_id: str) -> dict | None:
    """读取 trace；不存在/Redis 不可用时返回 None。"""
    try:
        from app.db.redis_client import get_redis
        from config import get_settings

        raw = get_redis(get_settings()).get(f"trace:{trace_id}")
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None
