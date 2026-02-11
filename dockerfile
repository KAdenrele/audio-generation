# Use NVIDIA's PyTorch image (includes CUDA 12.4+, cuDNN, and tuned PyTorch)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables for performance
ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE

# Install system dependencies
# Added 'git' so we can clone LuxTTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Clone LuxTTS repo to get its requirements
RUN git clone https://github.com/ysharma3501/LuxTTS.git

# Step 2: Install all Python dependencies in a single layer using pip
# This avoids numpy/pandas ABI incompatibility issues by letting pip resolve
# all dependencies at once.
RUN pip install --no-cache-dir \
    ninja \
    flash-attn --no-build-isolation \
    uv_build \
    uvloop \
    transformers \
    soundfile \
    tqdm \
    pandas \
    accelerate \
    pocket-tts \ 
    huggingface_hub \
    qwen-tts \
    -r LuxTTS/requirements.txt && \
    pip install --no-cache-dir ./LuxTTS

# Step 3: Pre-download Model Weights
# This ensures the 500-sample run doesn't wait for downloads at runtime
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

# Copy your local generation scripts (like the batch_generate.py we wrote)
COPY . .

# Set entrypoint
CMD ["python", "main.py"]