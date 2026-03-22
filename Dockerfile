# --- Base Image ---
FROM python:3.10-slim

# --- System Dependencies ---
# ADDED build-essential, g++, and python3-dev for fairseq compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    python3-dev \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

# --- Working Directory ---
WORKDIR /app

# --- Core Dependencies Fix ---
RUN python -m pip install --upgrade pip
RUN python -m pip install "pip<24.1" "setuptools<70.0.0"

# --- Install Requirements ---
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# --- Project Files ---
COPY . .

# --- Runtime Configuration ---
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8

CMD ["bash"]
