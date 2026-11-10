"""LawAnswer 全局配置。

单一配置源：所有可调参数集中在这里，通过 pydantic-settings 从 ``.env`` 加载
（字段名大写即环境变量名，如 ``MYSQL_HOST``）。

安全红线
--------
大模型密钥**只从系统环境变量读取**，不写入 ``.env``、不写入代码、不写入仓库：

* 读取入口：:func:`get_llm_api_key`（按 ``LLM_API_KEY`` → ``DEEPSEEK_API_KEY``
  → ``DASHSCOPE_API_KEY`` 的顺序取第一个非空值，兼容多家 OpenAI 兼容服务：
  换厂商只需改 ``LLM_BASE_URL`` 与 ``LLM_DEFAULT_MODEL``）
* 启动自检：:func:`assert_secrets_not_in_dotenv`，若 ``.env`` 中出现上述任一
  密钥赋值，应用直接拒绝启动（不是警告），从机制上杜绝"图省事写进配置文件"。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

#: 只允许存在于系统环境变量中的密钥名
SECRET_ENV_NAMES = (
    "LLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "API_KEY",
    "OPENAI_API_KEY",
)

#: 匹配"真正的赋值行"，注释里提到变量名不算
_SECRET_ASSIGN_RE = re.compile(
    r"^\s*(?:export\s+)?(" + "|".join(SECRET_ENV_NAMES) + r")\s*=",
)

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(RuntimeError):
    """配置非法，启动阶段就应失败。"""


class SecretInDotEnvError(ConfigError):
    """密钥被写进了 .env —— 违反安全红线。"""


# ---------------------------------------------------------------------------
# 分组视图（只读，便于 settings.mysql.host 这样使用）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    pool_size: int
    connect_timeout: int
    charset: str = "utf8mb4"

    @property
    def dsn(self) -> str:
        pwd = quote_plus(self.password)
        return (
            f"mysql+pymysql://{self.user}:{pwd}@{self.host}:{self.port}"
            f"/{self.database}?charset={self.charset}"
        )


@dataclass(frozen=True)
class RedisConfig:
    host: str
    port: int
    db: int
    password: str = ""

    @property
    def url(self) -> str:
        auth = f":{quote_plus(self.password)}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class MilvusConfig:
    host: str
    port: int
    database: str
    collection: str
    timeout: int = 5

    @property
    def uri(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    default_model: str
    timeout_s: int
    max_retries: int


@dataclass(frozen=True)
class IntentConfig:
    """意图识别专用：短超时（3s）+ 重试 1 次，失败即走规则兜底。"""

    timeout_s: int
    max_retries: int
    history_turns: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model_path: str
    device: str
    fp16: bool
    batch_size: int
    dense_dim: int = 1024


@dataclass(frozen=True)
class RerankerConfig:
    model_path: str
    batch_size: int
    max_length: int


@dataclass(frozen=True)
class RetrievalConfig:
    rrf_k: int
    pool_factor: int
    pool_min: int
    topk_simple: int
    topk_medium: int
    topk_complex: int
    # 融合策略不在这里：由顶层 `retrieval_ranker` 统一控制（"rrf" | "weighted"），
    # 消融结论是等权 RRF 优于加权，见 REPORT.md §4.4。
    dense_weight: float = 0.75
    sparse_weight: float = 0.25

    def topk_for(self, complexity: str) -> int:
        return {
            "simple": self.topk_simple,
            "medium": self.topk_medium,
            "complex": self.topk_complex,
        }.get(complexity, self.topk_medium)

    def pool_for(self, topk: int) -> int:
        return max(self.pool_factor * topk, self.pool_min)


@dataclass(frozen=True)
class GenerationConfig:
    """生成链路参数（阶段 5）。上下文预算与历史预算均按字符/token 估算口径。"""

    context_max_chars: int      # 参考资料总字符预算，超预算丢低分资料
    history_max_rounds: int     # 历史窗口轮数（对齐 chat_sessions.history_rounds 默认 5）
    history_token_budget: int   # 历史 token 预算（估算口径）
    max_tokens: int             # 生成长度上限
    temperature: float          # 采样温度（会话默认 0.30）
    replay_chunk_chars: int     # 结构化答案回放的分片大小（打字机效果）
    sources_limit: int          # SSE sources / prompt 资料条数上限


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """全部配置项。字段默认值即可跑通本地开发环境。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- 应用 ----
    app_name: str = "LawAnswer"
    app_env: str = "dev"
    app_version: str = "0.1.0"
    debug: bool = False
    server_host: str = "127.0.0.1"
    server_port: int = Field(default=8101, ge=1, le=65535)
    #: anyio 默认线程池容量（默认 40）。每条 SSE 流的每帧都经
    #: iterate_in_threadpool 抢一个线程，意图（最坏 ~10.5s）+ 生成（60s）长占线程，
    #: 40 条并发流即耗尽线程池且殃及所有端点。
    server_thread_tokens: int = Field(default=120, ge=10, le=1000)
    cors_origins: str = "http://localhost:5180,http://127.0.0.1:5180"

    # ---- 鉴权 ----
    jwt_secret: str = "dev-only-change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = Field(default=24, ge=1, le=24 * 30)
    # 验证码与登录失败锁定（阶段 6）
    captcha_ttl_s: int = Field(default=120, ge=30, le=600)
    # Redis 故障时验证码校验是否放行。dev 默认 True（本地不起 Redis 也能登录）；
    # prod 由启动校验强制 False——否则缓存一挂，登录验证就形同虚设。
    captcha_fail_open: bool = Field(
        default=True,
        description="Redis 故障时验证码校验是否放行；生产必须 false",
    )
    login_fail_limit: int = Field(default=5, ge=1, le=100)
    login_fail_window_s: int = Field(default=60, ge=10, le=3600)
    # 同一出口 IP 的失败上限（跨用户名累计）。只按用户名计数时，攻击者轮询不同
    # 用户名可以无限爆破，还能把任意账号逐个锁死；因此必须有 IP 维度兜底。
    # 30 次/5 分钟对 NAT 后多人共用的正常场景足够宽松。
    login_fail_ip_limit: int = Field(default=30, ge=1, le=1000)
    login_fail_ip_window_s: int = Field(default=300, ge=10, le=3600)

    # ---- MySQL ----
    mysql_host: str = "127.0.0.1"
    mysql_port: int = Field(default=3307, ge=1, le=65535)
    mysql_user: str = "lawanswer"
    mysql_password: str = "lawanswer_dev_pwd"
    mysql_root_password: str = "root_dev_pwd"
    mysql_db: str = "lawanswer"
    mysql_pool_size: int = Field(default=10, ge=1, le=100)
    mysql_connect_timeout: int = Field(default=5, ge=1, le=60)

    # ---- Redis ----
    redis_host: str = "127.0.0.1"
    redis_port: int = Field(default=6380, ge=1, le=65535)
    redis_db: int = Field(default=3, ge=0, le=15)
    redis_password: str = ""

    # ---- Milvus ----
    milvus_host: str = "127.0.0.1"
    milvus_port: int = Field(default=19531, ge=1, le=65535)
    milvus_db: str = "lawanswer"
    milvus_collection: str = "law_chunks"
    milvus_timeout: int = Field(default=5, ge=1, le=60)

    # ---- LLM ----
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_default_model: str = "deepseek-v4.1-flash"
    llm_thinking: str = "disabled"  # 百炼 DeepSeek-V4/qwen3 默认开思考，须显式 disabled；空=不传（其他厂商）
    llm_timeout_s: int = Field(default=60, ge=5, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # ---- 意图识别（短超时，失败走规则兜底）----
    # 5s 而非设计初稿的 3s：关掉 thinking 后 DeepSeek 实测 p50≈1.0s / p95≈1.1s，
    # 留 4 倍余量抗网络抖动；超时即降级规则兜底（准确率 1.00→0.55，故别设太紧）。
    intent_timeout_s: int = Field(default=5, ge=1, le=30)
    intent_max_retries: int = Field(default=1, ge=0, le=3)
    intent_history_turns: int = Field(default=2, ge=0, le=10)
    intent_cache_ttl: int = Field(
        default=86400,
        ge=0,
        le=30 * 86400,
        description="意图识别结果 Redis 缓存 TTL（秒）；0 关闭。只缓存无历史问题的 LLM 成功结果",
    )

    # ---- Embedding / Rerank ----
    embedding_model_path: str = (
        r"C:/Users/Administrator/.cache/huggingface/hub/models--BAAI--bge-m3"
    )
    embedding_device: str = "cuda"
    embedding_fp16: bool = True
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    reranker_model_path: str = "models/bge-reranker-v2-m3"
    reranker_batch_size: int = Field(default=16, ge=1, le=128)
    reranker_max_length: int = Field(default=256, ge=64, le=8192)
    reranker_max_text_chars: int = Field(default=200, ge=50, le=2000)

    # ---- 分块 / 检索 ----
    chunk_default_size: int = Field(default=300, ge=50, le=2000)
    chunk_default_overlap: int = Field(default=50, ge=0, le=500)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)
    retrieval_pool_factor: int = Field(default=4, ge=1, le=20)
    retrieval_pool_min: int = Field(default=12, ge=1, le=200)
    retrieval_topk_simple: int = Field(default=3, ge=1, le=50)
    retrieval_topk_medium: int = Field(default=5, ge=1, le=50)
    retrieval_topk_complex: int = Field(default=8, ge=1, le=50)
    # 混合检索融合策略：rrf（等权，默认）/ weighted（稠密主导，实测无收益，保留作对照）
    retrieval_ranker: str = "rrf"
    retrieval_dense_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    retrieval_sparse_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    # 用意图层的 law_hints 做检索法域预过滤（false = 全库检索，用于消融对照）
    retrieval_law_filter: bool = True
    # 法域过滤低置信回退：过滤检索后若 top1 重排分 < 阈值，说明意图可能判错法域、
    # 正确法条被硬过滤筛掉（实测 law_hints 质量随意图模型能力波动，判错法域时
    # single_hop 整条掉 0），此时无过滤重查一次并按 parent 合并。0 = 关闭。
    retrieval_law_rescue_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # 稀疏路候选池比例：<1 时稀疏少取候选，避免挤占稠密正确条的候选名额
    # （REPORT §4.4 证伪加权 RRF 后的真实怀疑点：名额之争而非权重之争）
    retrieval_sparse_pool_ratio: float = Field(default=1.0, ge=0.1, le=1.0)
    # 检索选路分流：simple / 非拆解单跳走纯稠密（消融 L1 R@5 0.7405 高于
    # 混合 L3 0.7024），decompose 多跳、complex、跨法则走混合（稀疏路对
    # 跨法术语互补）。关闭后一律 hybrid，与旧行为一致。
    retrieval_mode_routing: bool = Field(
        default=True,
        description="按复杂度/改写类型/法域数自动选择 dense 或 hybrid 检索",
    )
    # 自动选路走 dense 时的低置信回退：重排后 top1 < 阈值 → 再用 hybrid 检索
    # 一次并按法条合并。bge-reranker-v2-m3 normalize 后相关条约 0.79、无关条
    # 约 0.03，0.40 取两者中点偏保守；0 = 关闭回退。
    # 仅作用于 auto 选路，显式 mode=dense 的消融对照不受影响。
    retrieval_dense_rescue_score: float = Field(default=0.40, ge=0.0, le=1.0)
    # 法条互引扩展（task 2）：多跳检索沿"依照本法第X条"等显式边补检被引条。
    # 只扩展头部 max_sources 条命中、最多补 max_added 条，已命中条不重复。
    retrieval_ref_expansion: bool = Field(default=True)
    retrieval_ref_max_sources: int = Field(default=3, ge=1, le=10)
    retrieval_ref_max_added: int = Field(default=3, ge=1, le=10)
    # 法条时效（task 3）：检索时按 as_of 过滤未生效/已失效条文。
    # as_of 留空 = 今天；格式 YYYY-MM-DD。老数据 effective_from/to=0 不过滤。
    retrieval_filter_expired: bool = Field(default=True)
    retrieval_effective_as_of: str = Field(default="")
    qa_similarity_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    qa_bm25_topn: int = Field(default=3, ge=1, le=20)
    cache_qa_ttl: int = Field(default=604800, ge=60)
    # 无稠密能力（EMBEDDING_PROVIDER=bm25）时的字面双门。
    # bm25 的 dense 是固定零向量 → 余弦恒为 None → 缓存永远写不进去；
    # 此时改用「字符序列相似度 + 2-gram Jaccard」双条件，只放行"几乎同一句"。
    # 两个阈值必须同时达标才回放标准答案：宁可退回 RAG，也不能答非所问。
    qa_text_gate_enabled: bool = Field(default=True)
    qa_text_gate_ratio: float = Field(default=0.90, ge=0.0, le=1.0)
    # ------------------------------------------------------------------
    # 答案级缓存（把"实际生成过的整段回答"按问题哈希落 Redis）
    # 为什么需要它：qa_pairs 只有预置的几十条标准问法，用户问的绝大多数问题
    # 都不在库里 → 旧实现永远写不进缓存 → "同一问题问第二次仍要十几秒"。
    # 命中即整段回放，跳过意图识别 / 检索 / 重排 / LLM，TTFT 从秒级降到毫秒级。
    # ------------------------------------------------------------------
    cache_answer_enabled: bool = Field(default=True)
    cache_answer_ttl: int = Field(default=604800, ge=60)
    #: 低于该字数不缓存（"你好""嗯"之类的短回答回放没有价值，还占内存）
    cache_answer_min_chars: int = Field(default=80, ge=0)
    qa_text_gate_jaccard: float = Field(default=0.65, ge=0.0, le=1.0)

    # ---- 生成链路（阶段 5）----
    gen_context_max_chars: int = Field(default=6000, ge=500, le=50000)
    gen_history_max_rounds: int = Field(default=5, ge=0, le=20)
    gen_history_token_budget: int = Field(default=2000, ge=100, le=50000)
    gen_max_tokens: int = Field(default=2048, ge=256, le=8192)
    gen_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    gen_replay_chunk_chars: int = Field(default=30, ge=5, le=500)
    gen_sources_limit: int = Field(default=10, ge=1, le=50)

    # ---- 追问预测（回答后推荐 3 个可能的下一个问题）----
    suggest_enabled: bool = Field(
        default=True,
        description="是否在每次回答后预测 3 条追问；关闭可省一次小模型调用",
    )
    suggest_timeout_s: int = Field(
        default=8,
        ge=2,
        le=60,
        description="追问预测超时；这是锦上添花的调用，超时即放弃，不重试不报错",
    )
    suggest_max_retries: int = Field(default=0, ge=0, le=3)

    # ---- 限流（chat 接口，防 LLM 费用被刷爆）----
    rate_limit_enabled: bool = Field(
        default=True,
        description="限流总开关；压测时以 RATE_LIMIT_ENABLED=false 启动服务，否则 30 次/档的压测会被 429 卡死",
    )
    rate_limit_anonymous: int = Field(default=20, ge=1, le=600, description="匿名用户每分钟最大请求数")
    rate_limit_authed: int = Field(default=60, ge=1, le=1000, description="登录用户每分钟最大请求数")

    # ---- 反向代理（决定"真实客户端 IP"从哪来）----
    # 关着时一律用 TCP 对端地址；nginx 同机反代时必须打开，否则 client.host 恒为
    # 127.0.0.1，匿名限流与登录失败计数会把全站当成同一个人。
    trust_proxy: bool = Field(
        default=False,
        description="是否信任反向代理下发的 X-Real-IP / X-Forwarded-For",
    )
    trusted_proxies: str = Field(
        default="127.0.0.1,::1",
        description="可信代理地址，逗号分隔、支持 CIDR；仅当 TCP 对端在此列表内才读取 IP 头（防伪造）",
    )

    # ---- 日志 ----
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    log_backup_count: int = Field(default=5, ge=1, le=50)

    # ---- 存储 ----
    data_dir: str = "data"
    upload_dir: str = "data/uploads"
    minio_enabled: bool = True
    minio_endpoint: str = "127.0.0.1:9002"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "lawanswer"

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        v = (v or "").upper()
        if v not in _LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL 必须是 {sorted(_LOG_LEVELS)} 之一，收到 {v!r}")
        return v

    @field_validator("server_host", "mysql_host", "redis_host", "milvus_host")
    @classmethod
    def _check_host(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("主机地址不能为空")
        return v

    @model_validator(mode="after")
    def _check_cross_fields(self) -> "Settings":
        if self.chunk_default_overlap >= self.chunk_default_size:
            raise ValueError(
                f"CHUNK_DEFAULT_OVERLAP({self.chunk_default_overlap}) 必须小于 "
                f"CHUNK_DEFAULT_SIZE({self.chunk_default_size})"
            )
        if not (self.retrieval_topk_simple <= self.retrieval_topk_medium <= self.retrieval_topk_complex):
            raise ValueError("topK 必须满足 simple <= medium <= complex")
        if self.app_env not in {"dev", "test", "prod"}:
            raise ValueError(f"APP_ENV 只能是 dev/test/prod，收到 {self.app_env!r}")
        # 生产环境必须显式设置 JWT_SECRET，不能用默认值（否则 JWT 可被伪造）
        if self.app_env == "prod" and self.jwt_secret == "dev-only-change-me-in-production":
            raise ConfigError(
                "生产环境必须通过环境变量 JWT_SECRET 设置一个强随机密钥，"
                "不能使用默认值 dev-only-change-me-in-production"
            )
        # 生产环境不许验证码"存储故障时放行"：那等于把登录验证关掉。
        # 开发环境保留 fail-open，避免本地没起 Redis 就完全登不进去。
        if self.app_env == "prod" and self.captcha_fail_open:
            raise ConfigError(
                "生产环境必须设置 CAPTCHA_FAIL_OPEN=false：验证码校验在 Redis 故障时"
                "若放行，等于任何人无需验证码即可尝试登录"
            )
        return self

    # ------------------------------------------------------------------
    # 分组视图
    # ------------------------------------------------------------------
    @property
    def mysql(self) -> MySQLConfig:
        return MySQLConfig(
            host=self.mysql_host,
            port=self.mysql_port,
            user=self.mysql_user,
            password=self.mysql_password,
            database=self.mysql_db,
            pool_size=self.mysql_pool_size,
            connect_timeout=self.mysql_connect_timeout,
        )

    @property
    def redis(self) -> RedisConfig:
        return RedisConfig(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
        )

    @property
    def milvus(self) -> MilvusConfig:
        return MilvusConfig(
            host=self.milvus_host,
            port=self.milvus_port,
            database=self.milvus_db,
            collection=self.milvus_collection,
            timeout=self.milvus_timeout,
        )

    @property
    def llm(self) -> LLMConfig:
        return LLMConfig(
            base_url=self.llm_base_url,
            default_model=self.llm_default_model,
            timeout_s=self.llm_timeout_s,
            max_retries=self.llm_max_retries,
        )

    @property
    def intent(self) -> IntentConfig:
        return IntentConfig(
            timeout_s=self.intent_timeout_s,
            max_retries=self.intent_max_retries,
            history_turns=self.intent_history_turns,
        )

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            model_path=self.embedding_model_path,
            device=self.embedding_device,
            fp16=self.embedding_fp16,
            batch_size=self.embedding_batch_size,
        )

    @property
    def reranker(self) -> RerankerConfig:
        return RerankerConfig(
            model_path=self.reranker_model_path,
            batch_size=self.reranker_batch_size,
            max_length=self.reranker_max_length,
        )

    @property
    def retrieval(self) -> RetrievalConfig:
        return RetrievalConfig(
            rrf_k=self.retrieval_rrf_k,
            pool_factor=self.retrieval_pool_factor,
            pool_min=self.retrieval_pool_min,
            topk_simple=self.retrieval_topk_simple,
            topk_medium=self.retrieval_topk_medium,
            topk_complex=self.retrieval_topk_complex,
            dense_weight=self.retrieval_dense_weight,
            sparse_weight=self.retrieval_sparse_weight,
        )

    @property
    def generation(self) -> GenerationConfig:
        return GenerationConfig(
            context_max_chars=self.gen_context_max_chars,
            history_max_rounds=self.gen_history_max_rounds,
            history_token_budget=self.gen_history_token_budget,
            max_tokens=self.gen_max_tokens,
            temperature=self.gen_temperature,
            replay_chunk_chars=self.gen_replay_chunk_chars,
            sources_limit=self.gen_sources_limit,
        )

    # ------------------------------------------------------------------
    # 路径（相对路径一律相对项目根）
    # ------------------------------------------------------------------
    @property
    def log_path(self) -> Path:
        return self._abs(self.log_dir)

    @property
    def data_path(self) -> Path:
        return self._abs(self.data_dir)

    @property
    def upload_path(self) -> Path:
        return self._abs(self.upload_dir)

    @property
    def embedding_model_dir(self) -> Path:
        return Path(self.embedding_model_path).expanduser()

    @property
    def reranker_model_dir(self) -> Path:
        """相对路径相对项目根，绝对路径原样返回。"""
        p = Path(self.reranker_model_path).expanduser()
        return p if p.is_absolute() else (BASE_DIR / p)

    def _abs(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (BASE_DIR / p)

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        """创建运行时目录（日志、上传、数据）。"""
        for p in (self.log_path, self.upload_path, self.data_path):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 密钥：只认系统环境变量
# ---------------------------------------------------------------------------
def assert_secrets_not_in_dotenv(path: Path | None = None) -> None:
    """若 ``.env`` 中出现密钥赋值行，抛 :class:`SecretInDotEnvError`。

    只检查非注释行，因此 `.env.example` 里"密钥应放环境变量"的说明文字不会误报。
    """
    env_path = Path(path) if path is not None else ENV_FILE
    if not env_path.exists():
        return
    for lineno, line in enumerate(
        env_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
    ):
        if line.lstrip().startswith("#"):
            continue
        m = _SECRET_ASSIGN_RE.match(line)
        if m:
            raise SecretInDotEnvError(
                f"{env_path.name} 第 {lineno} 行出现了 {m.group(1)} 赋值。"
                "密钥必须只放在系统环境变量里（LLM_API_KEY / DEEPSEEK_API_KEY / "
                "DASHSCOPE_API_KEY 任选其一），禁止写入配置文件。"
            )


#: 密钥环境变量按优先级排列：通用名优先，其次各厂商约定名。
#: 之所以支持多个名字：底层是 OpenAI 兼容协议，换厂商只需换 base_url + 模型名，
#: 密钥变量名沿用各厂商习惯即可（DeepSeek 用户直接用 DEEPSEEK_API_KEY）。
LLM_API_KEY_ENV_NAMES: tuple[str, ...] = (
    "LLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
)


def get_llm_api_key() -> str:
    """读取大模型密钥（任一家 OpenAI 兼容服务）；未配置返回空串。"""
    for name in LLM_API_KEY_ENV_NAMES:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def get_llm_api_key_env() -> str:
    """返回实际生效的密钥变量名（只暴露名字、不暴露值，供健康检查/日志展示）。"""
    for name in LLM_API_KEY_ENV_NAMES:
        if os.getenv(name, "").strip():
            return name
    return ""


def require_llm_api_key() -> str:
    """需要真正调用大模型时使用：缺失则明确报错，而不是静默降级。"""
    key = get_llm_api_key()
    if not key:
        raise ConfigError(
            "未检测到大模型密钥。请设置下列任一系统环境变量后重启服务："
            "LLM_API_KEY / DEEPSEEK_API_KEY / DASHSCOPE_API_KEY；"
            "并确认 .env 中 LLM_BASE_URL 与 LLM_DEFAULT_MODEL 与该厂商匹配"
            "（DeepSeek: https://api.deepseek.com + deepseek-flash，并设 LLM_THINKING=disabled；"
            "DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1 + deepseek-v4.1-flash，"
            "并设 LLM_THINKING=disabled）。"
        )
    return key


def get_dashscope_api_key() -> str:
    """兼容旧调用点：等价于 :func:`get_llm_api_key`（现已支持 DeepSeek 等）。"""
    return get_llm_api_key()


def require_dashscope_api_key() -> str:
    """兼容旧调用点：等价于 :func:`require_llm_api_key`。"""
    return require_llm_api_key()


def mask_secret(value: str, keep: int = 4) -> str:
    """日志里打印密钥时使用，如 ``sk-1****cdef``。"""
    if not value:
        return "<empty>"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}****{value[-keep:]}"


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """测试或热更新时重建单例。"""
    get_settings.cache_clear()
    return get_settings()
