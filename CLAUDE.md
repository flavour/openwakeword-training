# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local training pipeline for custom OpenWakeWord wake word models. Uses Kokoro TTS for synthetic voice generation combined with real voice recordings to produce `.onnx` models.

## Key Commands

### Docker (recommended)
```bash
docker compose build trainer                  # Build training image
docker compose run --rm trainer ./setup-data.sh  # Download ~17GB training data
docker compose run --rm trainer python train.py --wake-word "hey cal" --data-dir /app/data
```

### Host (mic access needed)
```bash
python record_samples.py --wake-word "hey cal"                  # Record voice samples
python test_model.py --model my_custom_model/hey_cal.onnx       # Test model
```

### Manual (no Docker)
```bash
./setup.sh                                    # One-time setup (downloads ~17GB)
source venv/bin/activate
python train.py --wake-word "hey cal"
```

## Architecture

**Docker container** handles training (the dependency-heavy part):
- `train.py` orchestrates the full pipeline
- Generates positive/negative WAV samples via Kokoro TTS API (67 English voices, speed 0.7-1.3x)
- Copies real voice recordings from mounted `my_real_samples/` (3x weighted)
- Creates `training_config.yaml` from OpenWakeWord's template
- Shells out to `openwakeword/openwakeword/train.py --augment_clips` then `--train_model`
- Outputs `.onnx` model to mounted `my_custom_model/`

**Host** handles mic-dependent tasks:
- `record_samples.py` - records real voice samples (PyAudio)
- `test_model.py` - live mic testing of trained models

### Docker volume mounts
- `./data` → `/app/data` - 17GB feature files, background audio, impulse responses
- `./my_real_samples` → `/app/my_real_samples` - user voice recordings
- `./my_custom_model` → `/app/my_custom_model` - trained model output

### Kokoro TTS
- Runs as separate Docker service via `docker-compose.yml`
- trainer container connects via `http://kokoro:8880` (set by KOKORO_URL env var)
- train.py also accepts `--kokoro-url` flag

## Important Design Decisions

- **Negatives must be clearly different** from the wake word. Similar-sounding phrases ("hey call", "hey carl") hurt performance. Use only distinct phrases ("hello", "alexa", "hey siri").
- All audio is 16kHz, 16-bit, mono WAV.
- Real voice samples are copied 3x to weight them higher in training.
- `--data-dir` flag lets train.py work both inside Docker (`/app/data`) and on host (`.`).
