# Use NVIDIA's PyTorch image (includes CUDA 12.4+, cuDNN, and tuned PyTorch)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables for performance
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

# Step 1: Clone LuxTTS repo
RUN git clone https://github.com/ysharma3501/LuxTTS.git

# Step 2: Upgrade pip and install the missing build backend (uv-build)
# We also install 'ninja' here so it's ready for Flash Attention
RUN pip install --upgrade pip && \
    pip install uv-build ninja

# Step 3: Install dependencies
# We split this into two parts:
# Part A: General dependencies + Flash Attention
RUN pip install --no-cache-dir \
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

# Part B: Install LuxTTS
# CRITICAL FIX: We use --no-build-isolation so it uses the 'uv-build' 
# package we installed in Step 2, rather than trying to find it in a fresh env.
RUN pip install --no-cache-dir --no-build-isolation ./LuxTTS

# Step 4: Pre-download Model Weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

# Copy your local generation scripts
COPY . .

# Set entrypoint
CMD ["python", "main.py"]