FROM dustynv/pytorch:2.7-r36.4.0

# Set working directory
WORKDIR /app

# Install build dependencies for CTranslate2
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    cmake \
    libopenblas-dev \
    patch \
    && rm -rf /var/lib/apt/lists/*

# Build CTranslate2 from source with CUDA support (required for ARM64/Jetson)
RUN cd /tmp && \
    git clone --recursive https://github.com/OpenNMT/CTranslate2.git && \
    cd CTranslate2 && \
    mkdir build && cd build && \
    cmake -DWITH_CUDA=ON \
    -DWITH_MKL=OFF \
    -DWITH_OPENBLAS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    .. && \
    make -j$(nproc) && \
    make install && \
    cd ../python && \
    PIP_INDEX_URL=https://pypi.org/simple pip3 install . && \
    cd / && rm -rf /tmp/CTranslate2

# Install Python dependencies
# We use the base image's PyTorch (2.7.0) and NumPy (1.26.4)
# Install dependencies first to cache them
RUN pip3 install --index-url https://pypi.org/simple --no-cache-dir \
    gradio \
    jetson-stats \
    "faster-whisper==1.2.1" \
    "huggingface-hub<1.0"

# Install WhisperX without dependencies initially to avoid breaking PyTorch
# Then install its missing requirements, pinning torch components to prevent upgrade
RUN pip3 install --index-url https://pypi.org/simple --no-cache-dir --no-deps "whisperx==3.7.4" && \
    pip3 install --index-url https://pypi.org/simple --no-cache-dir \
    "torch==2.7.0" \
    "torchaudio==2.7.0" \
    nltk \
    pandas \
    transformers \
    pyannote.audio

# CRITICAL FIXES FOR COMPATIBILITY
# 1. Downgrade NumPy to 1.x to satisfy whisperx/numba constraints avoiding 2.x conflicts
RUN pip3 install --index-url https://pypi.org/simple --no-cache-dir "numpy<2.0"

# 2. Patch pyannote.audio to compatible with huggingface-hub >= 0.20
# Replace 'use_auth_token=use_auth_token' with 'token=use_auth_token' in pyannote core files
# This fixes "TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'"
COPY pyannote-compatibility.patch /tmp/pyannote-compatibility.patch
RUN cd /usr/local/lib/python3.10/dist-packages && \
    patch -p1 < /tmp/pyannote-compatibility.patch && \
    rm /tmp/pyannote-compatibility.patch

# Copy application code
COPY whisperx_gradio.py .

# Create outputs directory
RUN mkdir -p outputs && chmod 777 outputs

# Expose Gradio port
EXPOSE 7861

# Run the application
CMD ["python3", "whisperx_gradio.py"]
