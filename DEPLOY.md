# Wakeword Training - Forge Deployment

Train custom "Freyja" wakeword models on the forge VM (RTX 5070).

## Quick Start

```bash
# 1. Copy to forge (via sync or scp)
cd ~/code/openwakeword-training

# 2. Build Docker image & download training data (~17GB, one-time)
docker compose build trainer
docker compose run --rm trainer ./setup-data.sh

# 3. Train all three candidates (overnight job, 12-24h)
docker compose run --rm trainer bash train_all.sh
```

Or train one at a time:
```bash
docker compose run --rm trainer python train.py --wake-word "freyja" --data-dir /app/data
docker compose run --rm trainer python train.py --wake-word "hey freyja" --data-dir /app/data
docker compose run --rm trainer python train.py --wake-word "okay freyja" --data-dir /app/data
```

## What Happens

1. **Kokoro TTS** generates ~13K synthetic speech samples per candidate (67 voices × 200 samples)
2. **Negative samples** generated for clearly different phrases ("hello", "hey siri", "alexa")
3. **Augmentation** adds noise, reverb, mixing from AudioSet/FMA/MIT RIR data
4. **Training** produces an .onnx model (~400KB)

## Output

```
my_custom_model/
├── freyja.onnx
├── hey_freyja.onnx
└── okay_freyja.onnx
```

## Testing

Needs a microphone — test on laptop, not forge:
```bash
pip install openwakeword pyaudio numpy
python test_model.py --model my_custom_model/freyja.onnx
```

## GPU Notes

- Kokoro TTS needs GPU for sample generation
- Training itself uses GPU
- Both run sequentially per candidate, so 16GB VRAM is fine
- Make sure no other GPU services (ASR, etc.) are loaded during training

## Disk Space

- Training data: ~17GB (one-time download, shared across all candidates)
- Per candidate: ~2-3GB temp data during training
- Final models: ~400KB each
