# Use NVIDIA's PyTorch image (PyTorch 2.3.0 | CUDA 12.4)
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

# 3. Uninstall conflicting packages (Safety First)
# We remove the system transformers/torchvision so we can install clean versions
RUN pip uninstall -y transformers torchvision torchaudio

# 4. Install Main Dependencies (with upgrades)
# We use standard PyPI for these
RUN uv pip install --system --no-build-isolation \
    "numpy<2" \
    "transformers>=4.48.0" \
    "accelerate>=0.28.0" \
    librosa \
    sentencepiece \
    protobuf \
    scipy \
    flash-attn \
    soundfile \
    tqdm \
    pandas \
    huggingface_hub \
    qwen-tts \
    ./LuxTTS

# 5. CRITICAL FIX: Repair Torchvision & Torchaudio
# We force-reinstall the specific versions compatible with PyTorch 2.3.0
# using the official CUDA wheel index. This fixes the 'nms' error.
RUN pip install --no-cache-dir \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 6. Pre-download Lux Weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

COPY . .

CMD ["python", "main.py"]