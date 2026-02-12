# Use NVIDIA's PyTorch image (PyTorch 2.3.0 | CUDA 12.4)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Install uv (The fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ninja-build \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Clone LuxTTS to /opt (Safe from volume overrides)
RUN git clone https://github.com/ysharma3501/LuxTTS.git /opt/LuxTTS

# 3. Install Build Tools
# We install these first so that when we build LuxTTS later, the tools exist.
RUN uv pip install --system uv-build ninja setuptools wheel

# 4. Uninstall Pre-installed Packages (Prevent conflicts)
RUN pip uninstall -y transformers torchvision torchaudio flash-attn

# 5. FAST INSTALL: Main Dependencies
# REMOVED: --no-build-isolation (This was causing the slow 12min build)
# uv will now download binary wheels for these, taking ~30 seconds instead of 12 minutes.
RUN uv pip install --system \
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

# 6. Fix Torch Vision/Audio (Reinstall compatible versions)
RUN uv pip install --system \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 7. Install LuxTTS Dependencies
RUN uv pip install --system -r /opt/LuxTTS/requirements.txt

# 8. CRITICAL FIX: Install LuxTTS as a Package
# Instead of PYTHONPATH, we install it in "editable" mode or direct mode.
# --no-build-isolation is allowed HERE because we installed uv-build in Step 3.
RUN uv pip install --system --no-build-isolation -e /opt/LuxTTS

# 9. Pre-download Weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

COPY . .

CMD ["python", "main.py"]