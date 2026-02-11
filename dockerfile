# Use NVIDIA's PyTorch image (Best for GPU drivers & Flash Attention)
# Base: PyTorch 2.3.0 | CUDA 12.4
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
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Clone LuxTTS
RUN git clone https://github.com/ysharma3501/LuxTTS.git

# 2. Install Build Tools
RUN uv pip install --system uv-build ninja setuptools wheel

# 3. Clean environment (Remove conflicting versions)
RUN pip uninstall -y transformers torchvision torchaudio flash-attn

# 4. Install Core Dependencies & Audio Backends
# We force upgrade transformers/accelerate and pin numpy
RUN uv pip install --system --no-build-isolation \
    "numpy<2" \
    "transformers>=4.48.0" \
    "accelerate>=0.28.0" \
    librosa \
    sentencepiece \
    protobuf \
    scipy \
    soundfile \
    tqdm \
    pandas \
    huggingface_hub \
    qwen-tts

# 5. Fix Torch Vision/Audio (Reinstall compatible versions)
RUN uv pip install --system \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 6. Install Flash Attention (Force Build)
RUN pip install flash-attn --no-build-isolation

# 7. CRITICAL FIX: Install LuxTTS manually
# Instead of relying on 'pip install .', we add it to PYTHONPATH
# This guarantees 'from zipvoice.luxtts import LuxTTS' works.
ENV PYTHONPATH="${PYTHONPATH}:/app/LuxTTS"
# We also install its requirements just in case
RUN uv pip install --system -r LuxTTS/requirements.txt

# 8. Pre-download Lux Weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

COPY . .

CMD ["python", "main.py"]