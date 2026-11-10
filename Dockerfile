# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# 基础镜像
# ---------------------------------------------------------------------------
# python:3.13-slim：与 pyproject 的 requires-python >=3.13 对齐。
# 依赖里的 torch 从 PyPI 安装的是自带 CUDA 运行时的 Linux 轮子，因此默认镜像
# 不选 nvidia/cuda；需要 GPU 时再基于 nvidia/cuda 派生，并以
#   EMBEDDING_DEVICE=cuda
# 运行（同时需要 --gpus all）。CPU 推理把 EMBEDDING_DEVICE 保持为 cpu。
FROM python:3.13-slim

# uv 只用于构建期装依赖，运行期直接用 venv 里的 uvicorn。
# 从 PyPI 装固定版本而不是 COPY ghcr 镜像：部分网络环境到 ghcr 极慢/不可达。
RUN pip install --no-cache-dir uv==0.12.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# libgomp1：torch CPU 算子依赖 OpenMP。
# APT_MIRROR 可选：官方源慢的网络可用 --build-arg APT_MIRROR=mirrors.aliyun.com
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖清单 + 项目本体（hatchling 需要 README.md）
COPY pyproject.toml uv.lock README.md ./
COPY config.py ./
COPY app ./app

# --frozen：严格按 uv.lock 安装；--no-dev：镜像不带 pytest 等开发依赖
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# 运行期目录（上传文件、日志、缓存）在宿主机上持久化：
#   -v lawanswer-data:/app/data -v lawanswer-logs:/app/logs
RUN useradd --uid 10001 --create-home appuser \
    && mkdir -p /app/data /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# ---------------------------------------------------------------------------
# 模型资产：不进镜像
# ---------------------------------------------------------------------------
# bge-m3（约 2.3GB）与 bge-reranker-v2-m3（约 2.3GB）都不复制进镜像：
# 镜像会多出数 GB，且每次构建都要重传。运行时把宿主机目录挂到 /models：
#   docker run -v /srv/lawanswer/models:/models ...
# 目录约定：
#   /models/bge-m3/...
#   /models/bge-reranker-v2-m3/...
# 可用 scripts/download_models.py 或 huggingface-cli 预先下载；
# 没有模型时服务能启动，但检索/入库在首次 encode 时失败（/api/health 不会报，
# 因为模型不在依赖探测范围内），因此生产必须挂载。
ENV EMBEDDING_MODEL_PATH=/models/bge-m3 \
    RERANKER_MODEL_PATH=/models/bge-reranker-v2-m3 \
    EMBEDDING_DEVICE=cpu \
    EMBEDDING_FP16=false

VOLUME ["/models"]

EXPOSE 8101

# 存活探针不探测 MySQL/Redis/Milvus，依赖故障由 /api/health 暴露，
# 避免依赖抖动导致容器被反复重启。用 Python 发请求，镜像里不必装 curl。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8101/api/health/live', timeout=3).status == 200 else 1)"

# 生产部署还需显式提供：
#   APP_ENV=prod JWT_SECRET=<强随机> CAPTCHA_FAIL_OPEN=false
#   LLM_API_KEY（或 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY，只走环境变量）
#   MYSQL_HOST / REDIS_HOST / MILVUS_HOST 指向可达的地址
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8101"]
