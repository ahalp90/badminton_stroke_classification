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
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . .

# Dev default: --reload watches for file changes (code is bind-mounted in docker-compose.yml).
# Prod overrides this with --workers 2 and no --reload (see docker-compose.prod.yml).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]