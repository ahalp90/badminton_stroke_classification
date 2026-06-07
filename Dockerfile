# Backend Dockerfile, used for both dev and prod (prod overrides CMD in docker-compose.prod.yml).
# Bumped from 3.10 to 3.11, required by mediapipe and several other deps in requirements.txt.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps needed by OpenCV (libgl1, libsm6 etc) and the data pipeline (ffmpeg, cmake)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only PyTorch before the rest of requirements.txt.
# Without this, pip pulls the default CUDA-enabled torch (~2GB of nvidia packages)
# which is useless on a machine with no GPU. The CPU wheel is ~300MB instead.
# When requirements.txt runs, torch is already satisfied so pip skips it.
RUN python -m pip install --upgrade pip && \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
      pip install -r requirements.txt

COPY . .

# Run as a non-root user. No fixed UID so the image is portable across team members.
# Dev compose overrides this with the host user's UID/GID (see docker-compose.yml).
# Prod uses this user directly since there is no bind mount to worry about.
RUN useradd -m appuser && \
    mkdir -p /app/uploads && \
    chown appuser:appuser /app/uploads
USER appuser

# Dev default: --reload watches for file changes (code is bind-mounted in docker-compose.yml).
# Prod overrides this with --workers 2 and no --reload (see docker-compose.prod.yml).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]