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

# 2. Clone LuxTTS to /opt (Safe from volume overrides in /app)
# This ensures it survives when you mount your local dev folder to /app
RUN git clone https://github.com/ysharma3501/LuxTTS.git /opt/LuxTTS

# 3. Install Build Tools
RUN uv pip install --system uv-build ninja setuptools wheel

# 4. Uninstall Pre-installed Packages (Prevent conflicts)
RUN pip uninstall -y transformers torchvision torchaudio flash-attn

# 5. FAST INSTALL: Main Dependencies
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
    qwen-tts \

# 6. Fix Torch Vision/Audio (Reinstall compatible versions)
RUN uv pip install --system \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 7. Install LuxTTS Dependencies
RUN uv pip install --system -r /opt/LuxTTS/requirements.txt

# 8. THE GLOBAL IMPORT FIX
# Adding /opt/LuxTTS to PYTHONPATH makes the 'zipvoice' module 
# available to any script running in the container.
ENV PYTHONPATH="/opt/LuxTTS:${PYTHONPATH}"

# 9. Pre-download Weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

# 10. Copy your local scripts (main.py, etc.)
COPY . .

CMD ["python", "main.py"]