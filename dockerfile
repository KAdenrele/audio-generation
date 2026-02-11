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

# Install uv - a fast Python package installer from Astral
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Step 1: Install Python dependencies using uv
# We combine all pip installs into one layer for efficiency.
RUN uv pip install --no-cache-dir \
    ninja \
    flash-attn --no-build-isolation \
    transformers \
    soundfile \
    tqdm \
    pandas \
    accelerate \
    pocket-tts \
    huggingface_hub \
    qwen-tts

# Step 2: Download and Install LuxTTS and its dependencies
# Cloning from the official repo and installing in editable mode/package mode
RUN git clone https://github.com/ysharma3501/LuxTTS.git && \
    cd LuxTTS && \
    uv pip install --no-cache-dir -r requirements.txt && \
    uv pip install --no-cache-dir .

# Step 3: Pre-download Model Weights
# This ensures the 500-sample run doesn't wait for downloads at runtime
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

# Copy your local application scripts
COPY . .


# Set entrypoint
CMD ["python", "main.py"]