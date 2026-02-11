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

# 2. CRITICAL FIX: Install Build Tools FIRST
# We install 'uv-build' here so it exists before we try to compile LuxTTS.
RUN uv pip install --system uv-build ninja setuptools wheel

# 3. Install Dependencies & LuxTTS
# Now that uv-build is installed, --no-build-isolation will work perfectly.
# We force "numpy<2" to keep the NVIDIA container happy.
RUN uv pip install --system --no-build-isolation \
    "numpy<2" \
    flash-attn \
    transformers \
    soundfile \
    tqdm \
    pandas \
    accelerate \
    huggingface_hub \
    qwen-tts \
    ./LuxTTS

# 4. Pre-download Model Weights (Optional but recommended)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

COPY . .

CMD ["python", "main.py"]