# 🎙 Universal Voice Changer CLI (Academic Edition)

A modular command-line interface for RVC (Retrieval-based Voice Conversion) and GSS engines. Optimized for performance and cross-platform compatibility (macOS ARM64, Windows, Linux).

---

## 🛠 Prerequisites

### 1. Environment Setup (Python 3.10)
**Python 3.10 is strictly required**. Use **Conda** for a stable setup:

```bash
conda create -n hrvc_env python=3.10 -y
conda activate hrvc_env
```

### 2. Dependency Installation
Always use the `python -m pip` prefix to ensure packages are installed in the correct environment.

#### For macOS (Apple Silicon ARM64)
```bash
python -m pip install "pip<24.1" "setuptools<70.0.0"
python -m pip install -r requirements.txt
```

#### For Windows / Linux (NVIDIA GPU)
```bash
# 🚨 CRITICAL: Run this FIRST to enable GPU (CUDA) acceleration:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install the rest:
python -m pip install "pip<24.1" "setuptools<70.0.0"
python -m pip install -r requirements.txt
```

---

## 🐳 Running with Docker (Optional)

This project is fully containerized for reproducibility and server-side testing.

```bash
# Build the image and start the container in background
docker-compose up -d --build

# Enter the container environment
docker exec -it voice_changer_cli bash

# Execute voice conversion inside Docker (using generic paths)
docker exec -it voice_changer_cli python voice_cli.py -adapt hrvc_adapter -engine "./engine" -model "./model"
```
> [!WARNING]  
> GPU acceleration and interactive Audio I/O might be limited inside Docker on macOS. Use for environment testing and server-side inference.

---

## 🚀 Usage

Run the CLI using your specific adapter and model paths:

```bash
python voice_cli.py -adapt hrvc_adapter -engine "./engine" -model "./model"
```

### Command Arguments:
- `-adapt`: Engine adapter (e.g., `hrvc_adapter`).
- `-engine`: Path to the engine source folder.
- `-model`: Path to the voice model folder.
- `--pitch`: Semitone shift (-12 to 12).
- `--index`: Feature retrieval ratio (0.0 to 1.0).
- `--protect`: Voiceless protection threshold.

---

## 🏗 Project Architecture
- `voice_cli.py`: Interactive interface with real-time dB monitoring.
- `adapters/`: Modular engine adapters (decoupled logic).
- `last_devices.json`: Auto-saving your audio I/O configuration.

## 📄 License
MIT License. Optimized for Research and Academic Submission. 🏛
