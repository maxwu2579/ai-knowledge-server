# AI Knowledge Server 运行时镜像
#
# - python:3.12-slim（稳定、体积小；torch 对 3.12 支持成熟）
# - torch 用 CPU 版 wheel（容器内无 GPU 需求，避免拉 2GB+ CUDA 包）
# - 非 root 用户运行（useradd appuser）
# - 只复制运行时必需的源码；.env/.git/缓存/向量库由 .dockerignore 排除
# - API key 不从镜像获取：运行时通过环境变量或 compose env_file 注入

FROM python:3.12-slim

# 非 root 运行用户（固定 uid 便于卷权限）
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# 依赖层（利用 Docker 层缓存：requirements.txt 不变则不重装）
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

# 运行时源码（评估/实验脚本不进入镜像）
COPY api.py ask.py chunker.py store.py reranker.py ./

# HuggingFace 模型缓存目录（compose 可挂载持久卷，避免每次重建重复下载）
ENV HF_HOME=/models
RUN mkdir -p /models && chown -R appuser:appuser /models

USER appuser

EXPOSE 8000

# .env 不存在于镜像：DEEPSEEK_API_KEY 等由环境变量提供（ask.py 读取 os.environ）
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
