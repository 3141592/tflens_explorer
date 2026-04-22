FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HISTFILE=/commandhistory/.bash_history \
    HISTSIZE=10000 \
    HISTFILESIZE=20000

RUN echo 'export PROMPT_COMMAND="history -a; history -n"' >> /root/.bashrc

ENV HISTFILE=/commandhistory/.bash_history \
    HISTSIZE=10000 \
    HISTFILESIZE=20000

RUN apt-get update \
    && rm -rf /var/lib/apt/lists/* \
    && printf '\nexport PROMPT_COMMAND="${PROMPT_COMMAND:+$PROMPT_COMMAND; }history -a; history -n"\n' >> /root/.bashrc \
    && printf '"\\e[A": history-search-backward\n"\\e[B": history-search-forward\n' >> /root/.inputrc

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    openssh-client \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

RUN python -m pip install --upgrade pip setuptools wheel
