FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# 构建阶段
FROM base AS builder

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

# 创建虚拟环境并安装依赖
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# 创建缓存目录
RUN mkdir -p /app/.cache/torch /app/.cache/huggingface

# 设置环境变量指向新的缓存位置
ENV TORCH_HOME=/app/.cache/torch \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# 预下载模型
COPY pre_download.py .
RUN python3 pre_download.py

# 生产阶段
FROM base AS prod

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

# 创建缓存目录
RUN mkdir -p /app/.cache/torch /app/.cache/huggingface

# 设置环境变量指向缓存位置
ENV TORCH_HOME=/app/.cache/torch \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    PATH="/app/venv/bin:$PATH"

# 从构建阶段复制虚拟环境和模型
COPY --from=builder /app/venv /app/venv
COPY --from=builder /app/.cache/torch/ /app/.cache/torch/
COPY --from=builder /app/.cache/huggingface/ /app/.cache/huggingface/

COPY main.py .
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE ${PORT}

CMD [ "entrypoint.sh" ]
