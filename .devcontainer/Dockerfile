FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

RUN python -m pip install --upgrade pip setuptools wheel
