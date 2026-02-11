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

# --- ENV 1: MAIN SYSTEM (Qwen, Lux, Flash-Attn) ---
# We force numpy<2 to keep NVIDIA/Flash-Attn happy
RUN git clone https://github.com/ysharma3501/LuxTTS.git && \
    uv pip install --system --no-build-isolation \
    "numpy<2" \
    ninja \
    flash-attn \
    transformers \
    soundfile \
    tqdm \
    pandas \
    accelerate \
    huggingface_hub \
    qwen-tts \
    ./LuxTTS

# Pre-download Lux weights (for main env)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

COPY . .

CMD ["python", "main.py"]