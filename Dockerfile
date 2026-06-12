# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── system deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl \
        git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── python deps (torch first to leverage layer cache) ─────────────────────────
RUN pip install --no-cache-dir --upgrade pip

COPY config/requirements.txt config/requirements.txt
RUN pip install --no-cache-dir \
        torch==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r config/requirements.txt

# ── app source ────────────────────────────────────────────────────────────────
COPY app.py .
COPY src/ src/

# ── model cache ───────────────────────────────────────────────────────────────
# Mount a persistent volume here so weights survive container restarts:
#   docker run -v /your/local/path:/cache/huggingface ...
#   RunPod: attach a Network Volume and set its mount point to /cache/huggingface
ENV HF_HOME=/cache/huggingface
VOLUME /cache/huggingface

# ── runtime ───────────────────────────────────────────────────────────────────
EXPOSE ${PORT:-5000}
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-5000}/health || exit 1

# RUNPOD_ENABLED=true  → python app.py  (RunPod serverless handler)
# RUNPOD_ENABLED=false → gunicorn       (standard HTTP server)
# PRELOAD_MODELS=true  → download & load all models before serving any request
#
# 1 worker: the model is not thread-safe and holds most GPU memory.
# timeout 600: accounts for first-run model download + generation time.
CMD ["sh", "-c", \
     "if [ \"$RUNPOD_ENABLED\" = 'true' ]; then \
        python app.py; \
      else \
        gunicorn app:app \
          --bind 0.0.0.0:${PORT:-5000} \
          --workers 1 \
          --threads 1 \
          --timeout 600 \
          --log-level info; \
      fi"]
