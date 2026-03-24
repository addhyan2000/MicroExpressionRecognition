# syntax=docker/dockerfile:1

# ==========================================================================
# Master's Thesis: Micro-Expression Recognition
# Containerised Research Framework Targeting NVIDIA RTX 4080 (Ada Lovelace)
#
# Base Image  : NVIDIA PyTorch Container (NGC)
# Target GPU  : RTX 4080 — 4th Gen Tensor Cores, Ada Lovelace, 16GB VRAM
#
# IMPORTANT: Run with shared memory for 5D tensor DataLoader workers:
#
#   docker run --gpus all --shm-size=2g \
#       -v /host/data:/data \
#       -v /host/outputs:/outputs \
#       mer-thesis ablate --amp --compile
#
# ==========================================================================
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Environment Variables — Scientific Bounds
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Headless Matplotlib for t-SNE / visualisation generation
ENV MPLBACKEND=Agg

# Path-neutral volume mount points (consumed by main.py via os.environ.get)
ENV DATASET_PATH=/data
ENV OUTPUT_DIR=/outputs

WORKDIR /app

# ---------------------------------------------------------------------------
# OS Dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python Dependencies
# The NGC base image ships torch, torchvision, numpy, scipy.
# We install the remaining thesis-level packages only.
# ---------------------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Source Code
# ---------------------------------------------------------------------------
COPY . /app

# ---------------------------------------------------------------------------
# Volume Declarations for Docker Compose / Orchestrators
# ---------------------------------------------------------------------------
VOLUME ["/data", "/outputs"]

# ---------------------------------------------------------------------------
# Default Entry Point
# ---------------------------------------------------------------------------
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
