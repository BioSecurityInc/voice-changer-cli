# 🎙 Voice Changer CLI — Setup Guide

Система: **HalimGSS v4.2** + модель **Vlad (group1_vlad_EN_48k)**  
Запуск: микрофон → голосовой движок → динамики (real-time)

---

## Структура проекта

```
voice cli/
├── vcclient-dev/          ← виртуальное окружение Python 3.10
├── server/                ← w-okada voice-changer (не используется напрямую)
├── voice_cli.py           ← наш CLI (использует w-okada, НЕ для Vlad)
├── model_source/
│   ├── HalimGSSv4.2.228/  ← исходный код движка Vlad
│   │   ├── inference_online.py   ← ГЛАВНЫЙ скрипт для запуска Vlad!
│   │   ├── config_timbre.yaml    ← конфиг (нужно настроить пути)
│   │   ├── download_assets.py    ← скрипт скачивания моделей
│   │   └── assets/               ← создаётся после скачивания
│   │       ├── mHuBERT-147/      ← фонетический энкодер
│   │       ├── contentvec700/    ← ContentVec фичи
│   │       └── wavlm/            ← WavLM speaker encoder
│   └── Abdul Halim Bin Abdul Rahman - group1_vlad_EN_48k/
│       ├── model.json             ← конфиг архитектуры модели
│       ├── TimbreNode/
│       │   ├── G_Best_EMA.pth    ← ЛУЧШИЙ чекпоинт генератора ✅
│       │   ├── G_Final_EMA.pth   ← финальный чекпоинт
│       │   └── FiLM_latest.pth   ← FiLM модули (style, speaker, energy)
│       ├── faiss_index/
│       │   └── added_fused_IVF2937_Flat_nprobe_1_V2.index  ← FAISS индекс
│       ├── model_assets/
│       │   └── wavlm_speaker_embedding.pth  ← embedding спикера
│       └── style/
│           └── style_vec.npy    ← style vector
```

---

## Шаг 1 — Создать виртуальное окружение

```bash
cd "/Users/banan/Documents/voice cli"
python3.10 -m venv vcclient-dev
source vcclient-dev/bin/activate
```

---

## Шаг 2 — Установить зависимости

```bash
# Основные
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Аудио и ML
pip install sounddevice librosa numpy scipy soundfile

# HalimGSS зависимости
pip install transformers speechbrain pyyaml munch einops einops_exts \
            noisereduce pyloudnorm pydub webrtcvad pymediainfo \
            faiss-cpu fairseq psutil

# ONNX (для w-okada компонентов)
pip install onnxruntime
```

> ⚠️ `fairseq` нужен для ContentVec. Если не устанавливается:
> ```bash
> pip install fairseq --no-build-isolation
> ```

---

## Шаг 3 — Скачать модели (assets)

```bash
cd "/Users/banan/Documents/voice cli/model_source/HalimGSSv4.2.228"
mkdir -p assets/mHuBERT-147 assets/contentvec700 assets/wavlm
```

### ContentVec700 (~360 MB)
```bash
curl -L --progress-bar \
  "https://huggingface.co/lengyue233/content-vec-best/resolve/main/checkpoint_best_legacy_500.pt" \
  -o "assets/contentvec700/checkpoint_best_legacy_500.pt"
```

### mHuBERT-147 (~350 MB)
```bash
# Нужен git-lfs: brew install git-lfs && git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/utter-project/mHuBERT-147 assets/mHuBERT-147
cd assets/mHuBERT-147 && git lfs pull && cd ../..
```

### WavLM-Large (опционально, ~1.2 GB)
> Нужен только для анализа сходства спикеров. **Для просто запуска inference — не обязателен.**

```bash
curl -L --progress-bar \
  "https://huggingface.co/microsoft/wavlm-large/resolve/main/pytorch_model.bin" \
  -o "assets/wavlm/pytorch_model.bin"
curl -L \
  "https://huggingface.co/microsoft/wavlm-large/resolve/main/config.json" \
  -o "assets/wavlm/config.json"
```

| Файл | Размер | Обязателен? |
|------|--------|-------------|
| `assets/mHuBERT-147/` | ~350 MB | ✅ Да |
| `assets/contentvec700/checkpoint_best_legacy_500.pt` | ~360 MB | ✅ Да |
| `assets/wavlm/pytorch_model.bin` | ~1.2 GB | ⚪ Нет |

---

## Шаг 4 — Настроить config_timbre.yaml

Открой `/Users/banan/Documents/voice cli/model_source/HalimGSSv4.2.228/config_timbre.yaml`  
и измени следующие строки:

```yaml
# Имя модели (Vlad)
model_name: "group1_vlad_EN_48k"

# Пути к данным модели
dataset_path: "/Users/banan/Documents/voice cli/model_source/"

# Пути к pretrained энкодерам
hubert_model_path:       "./assets/mHuBERT-147"
contentvec_model_path:   "./assets/contentvec700/checkpoint_best_legacy_500.pt"

# Устройство (CPU для Mac)
device: "cpu"
```

---

## Шаг 5 — Запустить CLI

```bash
cd "/Users/banan/Documents/voice cli/model_source/HalimGSSv4.2.228"
source ../../vcclient-dev/bin/activate

# Terminal режим (без curses)
python inference_online.py --mode terminal

# Или GUI режим (TUI с курсорными клавишами)
python inference_online.py --mode gui
```

При запуске:
1. Выводит список устройств → выбираешь микрофон и динамик
2. Загружает модель Vlad (~30-60 сек на CPU)
3. `SPACE` → начало записи → `SPACE` → стоп → обработка → воспроизведение

---

## Горячие клавиши (terminal mode)

| Клавиша | Действие |
|---------|----------|
| `SPACE` | Старт / стоп записи |
| `R`     | Record-then-Infer режим |
| `Q`     | Выход |

---

## Какой чекпоинт использовать?

| Файл | Когда использовать |
|------|--------------------|
| `G_Best_EMA.pth` | **Рекомендуется** — лучший по валидационному loss |
| `G_Final_EMA.pth` | Финальный эпох (может быть немного хуже) |
| `G_best_EMA_by_MCD` | Лучший по MCD метрике |

По умолчанию `inference_online.py` ищет лучший чекпоинт автоматически.

---

## Возможные проблемы

### ❌ `No module named 'fairseq'`
```bash
pip install fairseq --no-build-isolation
```

### ❌ `No module named 'header_silent'`
Нужно запускать из папки `HalimGSSv4.2.228/`:
```bash
cd "/Users/banan/Documents/voice cli/model_source/HalimGSSv4.2.228"
python inference_online.py --mode terminal
```

### ❌ `ContentVec checkpoint not found`
Запусти `python download_assets.py` и убедись что файл скачался:
```
assets/contentvec700/checkpoint_best_legacy_500.pt
```

### ❌ MPS / Metal ошибки на Apple Silicon
В `config_timbre.yaml` поставь:
```yaml
device: "cpu"
```

### ❌ PortAudio ошибка
```bash
brew install portaudio
```

---

## Технические детали модели

| Параметр | Значение |
|---------|---------|
| Архитектура | HalimGSS SpeechCore v4.2 (форк RVC v2) |
| Фонетический энкодер | mHuBERT-147 (768-dim) |
| FAISS индекс | Fused (HuBERT + ContentVec) |
| Speaker embedding | WavLM-based (109 спикеров) |
| Sample rate | 48 kHz |
| F0 детектор | RMVPE |
| FiLM модули | Style + Speaker + Energy + Prosody |
