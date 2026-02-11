# Use NVIDIA's PyTorch image
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Clone LuxTTS
RUN git clone https://github.com/ysharma3501/LuxTTS.git

# Step 2: Install Build Tools
RUN pip install --upgrade pip && \
    pip install uv-build ninja

# Step 3: Install Python dependencies
# CRITICAL FIX: Added "numpy<2.0" to prevent binary incompatibility
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    flash-attn --no-build-isolation \
    transformers \
    soundfile \
    tqdm \
    pandas \
    accelerate \
    pocket-tts \
    huggingface_hub \
    qwen-tts \
    -r LuxTTS/requirements.txt

# Step 4: Install LuxTTS
RUN pip install --no-cache-dir --no-build-isolation ./LuxTTS

# Step 5: Pre-download Weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

COPY . .

CMD ["python", "main.py"]