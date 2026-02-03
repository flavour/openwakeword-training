# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local training pipeline for custom OpenWakeWord wake word models. Uses Kokoro TTS for synthetic voice generation combined with real voice recordings to produce `.onnx` models.

## Key Commands

```bash
./setup.sh                                    # One-time setup (downloads ~17GB)
source venv/bin/activate                      # Activate environment
python record_samples.py --wake-word "hey cal" # Record voice samples
python train.py --wake-word "hey cal"          # Train model (4-8 hours)
python test_model.py --model my_custom_model/hey_cal.onnx  # Test model
```

## Architecture

`train.py` orchestrates the full pipeline:
1. Generates positive/negative WAV samples via Kokoro TTS API (67 English voices, speed 0.7-1.3x)
2. Copies real voice recordings from `my_real_samples/` (3x weighted)
3. Creates `training_config.yaml` from OpenWakeWord's template
4. Calls `openwakeword/openwakeword/train.py --augment_clips` (adds noise, reverb)
5. Calls `openwakeword/openwakeword/train.py --train_model` (50K steps)
6. Outputs `.onnx` model to `my_custom_model/`

## Important Design Decisions

- **Negatives must be clearly different** from the wake word. Similar-sounding phrases ("hey call", "hey carl") hurt performance. Use only distinct phrases ("hello", "alexa", "hey siri").
- All audio is 16kHz, 16-bit, mono WAV.
- Real voice samples are copied 3x to weight them higher in training.
- `setup.sh` handles cloning openwakeword and downloading the 17GB feature dataset - these are not committed to the repo.
