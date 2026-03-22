# 🎙 Universal Voice Changer CLI

A modular command-line interface for RVC and GSS neural voice conversion engines. Optimized for cross-platform compatibility (macOS ARM64, Windows, Linux).

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

## 🚀 Usage

Run the CLI using your specific adapter and model paths:

```bash
python voice_cli.py -adapt hrvc_adapter -engine "./engine" -model "./model"
```

### Command Arguments:
- `-adapt`: Engine adapter (e.g., `hrvc_adapter`).
- `-engine`: Path to the engine folder.
- `-model`: Path to the voice model folder.

---

## 🏗 Project Architecture
- `voice_cli.py`: Interactive interface with dB monitoring.
- `adapters/`: Modular engine adapters.
- `last_devices.json`: Auto-saving your audio I/O configuration.

## 📄 License
MIT License. Optimized for Research and Academic use.
