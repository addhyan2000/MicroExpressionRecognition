# syntax=docker/dockerfile:1

# Master's Thesis: Micro-Expression Recognition
# Base Image: NVIDIA PyTorch Container (Targeting High-End Compute clusters)
# Note: For strict determinism and shape-matched library architectures.
FROM nvcr.io/nvidia/pytorch:24.05-py3

# MAINTAINER Addhyan Pant

# Environment Variables mapping scientific bounds
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set MPLBACKEND=Agg for completely headless t-SNE / Matplotlib generation 
# This prevents backend crashes natively during deep execution bounds
ENV MPLBACKEND=Agg

WORKDIR /app

# Standard OS Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Thesis-Level Python Dependencies Native 
# The base image already inherently mounts torch, torchvision, numpy, etc.
# Scipy/Sklearn bounds strictly evaluated mathematically mapping.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# System Code Import
COPY . /app

# IMPORTANT: SHARED MEMORY VRAM WARNING 
# WARNING: When executing this container dynamically, you MUST pass `--shm-size=2g`
# or larger natively. 5D Spatiotemporal video tensors require extensive shared memory.
# Failure to expand `--shm-size` will result in catastrophic "Bus Errors" via PyTorch DataLoader workers.

# Default Orchestrator Command bounds natively evaluated
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
