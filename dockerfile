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
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Clone LuxTTS
RUN git clone https://github.com/ysharma3501/LuxTTS.git

# Step 2: UPGRADE PIP & SETUPTOOLS (Crucial Step)
# This fixes the "metadata-generation-failed" error by ensuring 
# we can read modern package formats.
RUN pip install --upgrade pip setuptools wheel && \
    pip install uv-build ninja

# Step 3: Install Dependencies
# We use --only-binary to prevent the "compiling from source" error.
# We force numpy<2.0 to ensure compatibility with the NVIDIA container.
RUN pip install --no-cache-dir --only-binary=:all: \
    "numpy<2.0" \
    pandas \
    soundfile \
    tqdm \
    accelerate \
    huggingface_hub \
    transformers && \
    # Install the rest normally
    pip install --no-cache-dir \
    flash-attn --no-build-isolation \
    pocket-tts \
    qwen-tts \
    -r LuxTTS/requirements.txt

# Step 4: Install LuxTTS
RUN pip install --no-cache-dir --no-build-isolation ./LuxTTS

# Step 5: Pre-download Weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='YatharthS/LuxTTS', allow_patterns=['*.bin', '*.json', '*.pth'])"

COPY . .

CMD ["python", "main.py"]