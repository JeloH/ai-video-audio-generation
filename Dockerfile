FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    ffmpeg git libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    torch==2.13.0+cu129 \
    torchaudio==2.11.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /inputs /outputs /cache/huggingface /cache/torch

CMD ["bash"]
