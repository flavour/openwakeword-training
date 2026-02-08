#!/bin/bash
# Download training data (~17GB) into data/ directory.
# Run once before training. Skips files that already exist.
#
# Docker:  docker compose run --rm trainer ./setup-data.sh
# Manual:  ./setup-data.sh

set -e

# Use /app/data inside container, ./data on host
DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"

echo "=== Downloading training data to $DATA_DIR ==="
echo ""

# Pre-computed features (~17GB)
if [ ! -f "$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" ]; then
    echo "Downloading ACAV100M features (17GB - this takes a while)..."
    curl -L -o "$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" \
        'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy'
else
    echo "ACAV100M features already exist, skipping."
fi

# Validation features (~177MB)
if [ ! -f "$DATA_DIR/validation_set_features.npy" ]; then
    echo "Downloading validation features..."
    curl -L -o "$DATA_DIR/validation_set_features.npy" \
        'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy'
else
    echo "Validation features already exist, skipping."
fi

# MIT Room Impulse Responses
if [ ! -d "$DATA_DIR/mit_rirs" ]; then
    echo "Downloading room impulse responses..."
    git lfs install
    git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses "$DATA_DIR/MIT_environmental_impulse_responses_tmp"

    mkdir -p "$DATA_DIR/mit_rirs"
    # Just copy the 16kHz WAVs directly — no need for datasets library
    cp "$DATA_DIR/MIT_environmental_impulse_responses_tmp/16khz/"*.wav "$DATA_DIR/mit_rirs/"
    echo "Copied $(ls "$DATA_DIR/mit_rirs/"*.wav | wc -l) RIR files"
    rm -rf "$DATA_DIR/MIT_environmental_impulse_responses_tmp"
else
    echo "MIT RIRs already exist, skipping."
fi

# AudioSet background audio (parquet format on HuggingFace)
if [ ! -d "$DATA_DIR/audioset_16k" ]; then
    echo "Downloading AudioSet background audio..."
    mkdir -p "$DATA_DIR/audioset_16k"

    python3 << 'PYEOF'
import os, struct
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import numpy as np
import scipy.io.wavfile

data_dir = os.environ.get("DATA_DIR", "./data")
out_dir = Path(f"{data_dir}/audioset_16k")

# Download one parquet shard (~700MB, contains ~2000 clips)
print("Downloading AudioSet parquet shard...")
path = hf_hub_download(
    repo_id="agkphysics/AudioSet",
    filename="data/bal_train/00.parquet",
    repo_type="dataset",
)

print("Extracting audio from parquet...")
table = pq.read_table(path, columns=["audio"])
count = 0
for i in range(min(len(table), 500)):  # 500 clips is plenty for background
    try:
        audio_struct = table.column("audio")[i].as_py()
        audio_bytes = audio_struct["bytes"]
        # Audio is encoded (typically FLAC) - write to temp and convert via scipy
        tmp_path = out_dir / f"_tmp_{i}"
        wav_path = out_dir / f"audioset_{i:04d}.wav"
        tmp_path.write_bytes(audio_bytes)
        # Use ffmpeg to convert to 16kHz WAV
        os.system(f'ffmpeg -y -i "{tmp_path}" -ar 16000 -ac 1 "{wav_path}" -loglevel error 2>/dev/null')
        tmp_path.unlink(missing_ok=True)
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            count += 1
    except Exception as e:
        pass

print(f"Extracted {count} AudioSet clips to 16kHz WAV")
PYEOF
else
    echo "AudioSet already exists, skipping."
fi

# Music background samples (MUSAN music subset - direct download, no HF dataset API)
if [ ! -d "$DATA_DIR/fma" ]; then
    echo "Downloading background music (MUSAN)..."
    mkdir -p "$DATA_DIR/fma"

    # MUSAN is ~10GB total but we only extract the music portion (~1.5GB)
    # Using openslr mirror which is reliable
    curl -L -o "$DATA_DIR/musan_music.tar.gz" \
        'https://www.openslr.org/resources/17/musan.tar.gz'

    echo "Extracting music subset..."
    tar -xzf "$DATA_DIR/musan_music.tar.gz" -C "$DATA_DIR" musan/music 2>/dev/null || true

    # Convert to 16kHz WAV (take up to 120 files)
    count=0
    for f in $(find "$DATA_DIR/musan/music" -name "*.wav" -type f 2>/dev/null | head -120); do
        name="music_$(basename "$f")"
        if ffmpeg -y -i "$f" -ar 16000 -ac 1 "$DATA_DIR/fma/$name" -loglevel error 2>/dev/null; then
            count=$((count+1))
        fi
    done
    echo "Converted $count music files to 16kHz WAV"
    rm -rf "$DATA_DIR/musan" "$DATA_DIR/musan_music.tar.gz"
else
    echo "Music background already exists, skipping."
fi

echo ""
echo "=== Data download complete! ==="
echo ""
du -sh "$DATA_DIR"/*
