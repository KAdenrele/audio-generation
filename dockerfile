# Use NVIDIA's PyTorch image (Best for GPU drivers & Flash Attention)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Clone LuxTTS
RUN git clone https://github.com/ysharma3501/LuxTTS.git

# 2. Install Build Tools
RUN uv pip install --system uv-build ninja setuptools wheel

# 3. Install Dependencies & UPGRADE Transformers
# FIX: Added "transformers>=4.48.0" to force an upgrade over the NVIDIA default.
# FIX: Added "accelerate>=0.26.0" as Qwen3 often needs newer accelerate too.
RUN uv pip install --system --no-build-isolation \
    "numpy<2" \
    "transformers>=4.48.0" \
    "accelerate>=0.26.0" \
    flash-attn \
    soundfile \
    tqdm \
    pandas \
    huggingface_hub \
    qwen-tts \
    ./LuxTTS

# 4. Pre-download Model Weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

COPY . .

CMD ["python", "main.py"]