# Use NVIDIA's PyTorch image (Best for GPU drivers)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# 1. Install uv (The magic tool)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE

# 2. Install System Dependencies (ffmpeg, git, ninja)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Clone LuxTTS
RUN git clone https://github.com/ysharma3501/LuxTTS.git

# 4. Use `uv` to install everything
# --system: Installs into the current environment (keeps NVIDIA's PyTorch)
# --no-build-isolation: Critical for Flash Attention to see CUDA
# "numpy<2": Explicitly prevents the binary incompatibility error
RUN uv pip install --system --no-build-isolation \
    "numpy<2" \
    ninja \
    flash-attn \
    transformers \
    soundfile \
    tqdm \
    pandas \
    accelerate \
    pocket-tts \
    huggingface_hub \
    qwen-tts \
    ./LuxTTS

# 5. Pre-download Model Weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

# Copy your scripts
COPY . .

# Run
CMD ["python", "main.py"]