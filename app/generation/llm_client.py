"""DashScope（OpenAI 兼容模式）LLM 客户端。

意图识别与生成链路各持一个按参数缓存的实例（同库不同超时/重试）：

* **密钥只从系统环境变量读**，缺 key 直接报
  :class:`~config.ConfigError`，不静默降级——调用方自行决定是否走兜底。
* **超时与重试可配**：意图识别 5s/重试 1 次（快速失败走规则兜底），
  生成 60s/重试 2 次（保质量）。指数退避 ``0.5s * 2^n``。
* **JSON 模式**：``response_format={"type": "json_object"}`` 强约束输出，
  调用方仍须校验字段（模型可能吐非法 JSON 或漏字段）。
* 客户端进程级缓存，复用底层 HTTP 连接池；缓存键只用密钥指纹。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.core.logger import get_logger
from app.core.metrics import record_llm_call
from config import ConfigError, Settings, get_llm_api_key, get_settings

logger = get_logger("lawanswer.llm")


class LLMError(RuntimeError):
    """LLM 调用失败（网络/超时/限流/服务端错误）。"""


class LLMResponseError(LLMError):
    """LLM 返回内容不可用（空内容、JSON 解析失败）。"""


class LLMStreamInterrupted(LLMError):
    """流式传输中途断流（首 token 已产出后的网络/上游异常）。

    ``partial`` 保存已收到的文本，调用方可选择"保留半截回答 + 落库"而不是
    整轮丢弃（设计方案 §6.5 流式中断容错）。
    """

    def __init__(self, partial: str, message: str) -> None:
        super().__init__(message)
        self.partial = partial


@dataclass(frozen=True)
class StreamChunk:
    """一次流式增量。``usage`` 仅在最后一个 chunk（``finish_reason`` 就绪）携带。"""

    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """一次成功的 LLM 调用结果。"""

    content: str
    model: str
    latency_ms: int
    usage: dict[str, int] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        """把内容解析为 JSON 对象；失败抛 :class:`LLMResponseError`。"""
        try:
            data = json.loads(self.content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMResponseError(f"LLM 返回非合法 JSON: {exc}; 内容前 200 字: {self.content[:200]!r}") from exc
        if not isinstance(data, dict):
            raise LLMResponseError(f"LLM 返回的 JSON 不是对象: {type(data).__name__}")
        return data


class LLMClient:
    """薄封装：chat / chat_json 两个入口，内置重试与计时。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        thinking: str = "",
    ) -> None:
        if not api_key:
            raise ConfigError(
                "大模型密钥为空：请设置 LLM_API_KEY / DEEPSEEK_API_KEY / "
                "DASHSCOPE_API_KEY 之一（系统环境变量），并确认 .env 的 "
                "LLM_BASE_URL 与 LLM_DEFAULT_MODEL 与该厂商匹配"
            )
        # max_retries=0：重试由本类的指数退避统一负责，避免 SDK 内重试叠加
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        self._model = model
        self._base_url = base_url or ""
        self._max_retries = max_retries
        #: 思考模式开关（enabled/disabled/空=不传），按 provider 适配参数形态：
        #: - DeepSeek 官方（api.deepseek.com）：默认 enabled，思考模式下 temperature 无效、
        #:   输出上限 64K、延迟与费用显著上升，参数形态 ``thinking: {type: ...}``；
        #: - 阿里云百炼（DashScope 兼容模式，含 DeepSeek-V4 系 deepseek-v4.1-flash 与
        #:   qwen3 系）：**默认开思考**，RAG 生成实测首 token 前 17-26s 隐藏推理
        #:   （TTFT≈总延迟），参数形态 ``enable_thinking``；
        #: - 其他 OpenAI 兼容厂商：不传。
        self._thinking = thinking.strip().lower()

    def _create(self, **kwargs: Any):
        """统一的请求出口：按"真实 HTTP 请求"计数（重试各算一次）。"""
        model = str(kwargs.get("model") or self._model)
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception:
            record_llm_call(model, "error")
            raise
        record_llm_call(model, "success")
        return resp

    def _thinking_kwargs(self) -> dict[str, Any]:
        if self._thinking not in {"enabled", "disabled"}:
            return {}
        if "dashscope" in self._base_url:
            return {"extra_body": {"enable_thinking": self._thinking == "enabled"}}
        return {"extra_body": {"thinking": {"type": self._thinking}}}

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """普通对话；``json_mode=True`` 时要求模型输出 JSON 对象。"""
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(self._thinking_kwargs())

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                resp = self._create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - 统一转成 LLMError 再重试
                last_exc = exc
                if attempt < self._max_retries:
                    _backoff_sleep(attempt)
                    continue
                raise LLMError(f"LLM 调用失败（重试 {self._max_retries} 次后仍失败）: {exc}") from exc

            latency_ms = int((time.perf_counter() - started) * 1000)
            content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
            if not content:
                last_exc = LLMResponseError("LLM 返回空内容")
                if attempt < self._max_retries:
                    _backoff_sleep(attempt)
                    continue
                raise last_exc
            usage: dict[str, int] = {}
            if resp.usage is not None:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens or 0,
                    "completion_tokens": resp.usage.completion_tokens or 0,
                    "total_tokens": resp.usage.total_tokens or 0,
                }
            return LLMResponse(
                content=content,
                model=resp.model or kwargs["model"],
                latency_ms=latency_ms,
                usage=usage,
            )
        # 理论不可达（循环内必定 return 或 raise）
        raise LLMError(f"LLM 调用异常退出: {last_exc}")

    def chat_json(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """JSON 模式调用，返回 ``(解析后的 dict, 原始响应)``。"""
        resp = self.chat(messages, json_mode=True, **kwargs)
        return resp.json(), resp

    def chat_stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> Iterator[StreamChunk]:
        """流式对话，逐段产出文本增量（SSE ``delta`` 事件的上游）。

        * 重试只覆盖**首个 chunk 之前**的失败（连接/鉴权/限流），
          首 token 后的断流不再重试——避免重复计费与内容错乱；
        * 断流抛 :class:`LLMStreamInterrupted`，``partial`` 为已产出文本；
        * ``stream_options={"include_usage": True}`` 让末 chunk 带 token 用量
          （DashScope OpenAI 兼容模式支持）。
        """
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs.update(self._thinking_kwargs())

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            partial = ""
            try:
                stream = self._create(**kwargs)
                for chunk in stream:
                    usage = _extract_usage(chunk)
                    if not chunk.choices:
                        # OpenAI 风格：单独的 usage-only 块（choices 为空）
                        if usage:
                            yield StreamChunk(usage=usage)
                        continue
                    choice = chunk.choices[0]
                    text = choice.delta.content or ""
                    finish = choice.finish_reason
                    if text:
                        partial += text
                    # DeepSeek 风格：usage 挂在最后一个内容块上（choices 非空、finish_reason 非空），
                    # 所以这里三种信号任一出现都要产出，避免丢掉用量统计
                    if text or finish or usage:
                        yield StreamChunk(text=text, finish_reason=finish, usage=usage)
                return
            except Exception as exc:  # noqa: BLE001 - 统一转 LLMError 再决定重试
                last_exc = exc
                if partial:
                    # 已产出部分内容：断流不重试，保留 partial 供上层落库
                    raise LLMStreamInterrupted(
                        partial, f"LLM 流式传输中断（已收 {len(partial)} 字）: {exc}"
                    ) from exc
                if attempt < self._max_retries:
                    _backoff_sleep(attempt)
                    continue
                raise LLMError(f"LLM 流式调用失败（重试 {self._max_retries} 次后仍失败）: {exc}") from exc
        # 理论不可达
        raise LLMError(f"LLM 流式调用异常退出: {last_exc}")


def _extract_usage(chunk) -> dict[str, int]:
    """从流式 chunk 里取 usage；不同厂商挂载位置不同（OpenAI 独立块 / DeepSeek 末个内容块）。"""
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
    }


def _backoff_sleep(attempt: int) -> None:
    """指数退避：0.5s、1s、2s……"""
    time.sleep(0.5 * (2**attempt))


def _key_fingerprint(api_key: str) -> str:
    """密钥指纹：缓存键用指纹而非明文，避免 key 留存在缓存键里。"""
    return hashlib.sha1(api_key.encode("utf-8")).hexdigest()[:12]


#: 进程级客户端缓存：键含密钥指纹（不含明文），值持有真实 key 的客户端实例。
#: 手写 dict 而非 lru_cache，因为 lru_cache 的缓存键就是入参本身——
#: 构造函数要收明文 key 就必然把明文留在缓存键里，两者不可兼得。
_CLIENT_CACHE: dict[tuple, LLMClient] = {}
_CLIENT_CACHE_MAX = 4


def _get_cached_client(
    api_key: str, base_url: str, model: str, timeout_s: float, max_retries: int, thinking: str = ""
) -> LLMClient:
    cache_key = (_key_fingerprint(api_key), base_url, model, timeout_s, max_retries, thinking)
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            thinking=thinking,
        )
        _CLIENT_CACHE[cache_key] = client
        while len(_CLIENT_CACHE) > _CLIENT_CACHE_MAX:  # 与原 lru_cache(maxsize=4) 行为一致
            _CLIENT_CACHE.pop(next(iter(_CLIENT_CACHE)))
    return client


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """进程级单例。缺 key 抛 :class:`~config.ConfigError`。"""
    s = settings or get_settings()
    return _get_cached_client(
        get_llm_api_key(),
        s.llm_base_url,
        s.llm_default_model,
        float(s.llm_timeout_s),
        s.llm_max_retries,
        s.llm_thinking,
    )


def reset_llm_client() -> None:
    """测试用：清空单例。"""
    _CLIENT_CACHE.clear()
