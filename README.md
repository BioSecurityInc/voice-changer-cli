# 🎙 hRVC: Modular Neural Voice Conversion CLI
### Academic Research Project in Neural Speech Synthesis (NSS)

---

## 📄 Abstract
The hRVC project presents a modular command-line interface designed to decouple the neural voice conversion (VC) engine logic from the user interaction layer. By implementing the **Adapter Software Design Pattern**, hRVC achieves an engine-agnostic architecture.

## 🏛 Methodology & Architecture
The system is divided into three distinct abstraction layers:
1. **Runner Core (`voice_cli.py`)**: Hardware abstraction and signal orchestration.
2. **Adapter Layer (`adapters/`)**: Mediation between the CLI and neural processing units.
3. **Neural Engine**: Proprietary or open-source inference models.

## 🛠 Installation & Deployment Guide

### 1. Conda Environment (Recommended)
```bash
conda env create -f environment.yml
conda activate hrvc_env
```

### 2. Manual venv Setup (Alternative)
```bash
# Create environment with Python 3.10
python3.10 -m venv venv
source venv/bin/activate

# Downgrade pip to bypass fairseq metadata conflict
pip install "pip<24.1" "setuptools<70.0.0"

# Install PyTorch 2.0.x first (critical — newer versions are incompatible)
pip install torch==2.0.0 torchaudio==2.0.1

# Install remaining dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Workflow

```bash
python3 voice_cli.py \
  -adapt hrvc_adapter \
  -engine "/path/to/engine_source" \
  -model "/path/to/model_weights"
```

### Command Arguments Specification:
- `-adapt`: Identifier for the target adapter in `adapters/`.
- `-engine`: Path to the neural processing unit source directory.
- `-model`: Path to the directory containing pre-trained weights and `model.json`.
- `-s`, `--select`: Override device persistence and enforce manual selection.

---

## 📈 Research Features
- **Deterministic Device Persistence**: Caches hardware IDs in `last_devices.json`.
- **Hardware Acceleration**: Automatic utilization of **Apple Metal (MPS)** for local GPU inference.
- **Pitch Management**: Integrated support for real-time transposition via `--pitch`.

## 📜 Intellectual Property & Licensing
This software is developed for research purposes and adheres to the original licenses provided by the underlying neural implementations. See the `LICENSE` file.
