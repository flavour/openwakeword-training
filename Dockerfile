FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git git-lfs curl build-essential libsndfile1 portaudio19-dev ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# PyTorch with CUDA 12.8 (minimum version for Blackwell support)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Training dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install OpenWakeWord (in /opt so volume mounts on /app don't hide it)
RUN git clone https://github.com/dscripka/openWakeWord /opt/openwakeword \
    && pip install --no-cache-dir -e /opt/openwakeword \
    && sed -i 's/torchaudio.set_audio_backend("soundfile")/#torchaudio.set_audio_backend("soundfile")/' \
       /usr/local/lib/python3.10/dist-packages/torch_audiomentations/utils/io.py 2>/dev/null || true

# Download embedding models (small, safe to bake into image)
RUN mkdir -p /opt/openwakeword/openwakeword/resources/models \
    && curl -L -o /opt/openwakeword/openwakeword/resources/models/embedding_model.onnx \
        'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx' \
    && curl -L -o /opt/openwakeword/openwakeword/resources/models/melspectrogram.onnx \
        'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx'

# Patch upstream train.py to emit METRIC: lines for MLflow parsing
RUN TRAIN=/opt/openwakeword/openwakeword/train.py \
    && sed -i '/self\.history\["loss"\]\.append(/a\                    print(f"METRIC:loss={loss.detach().cpu().numpy()}:step={step_ndx}", flush=True)' "$TRAIN" \
    && sed -i '/self\.history\["recall"\]\.append(/a\                    print(f"METRIC:recall={self.history[\"recall\"][-1]}:step={step_ndx}", flush=True)' "$TRAIN" \
    && sed -i '/self\.history\["val_accuracy"\]\.append(/a\                print(f"METRIC:val_accuracy={val_acc.detach().cpu().numpy()}:step={step_ndx}", flush=True)' "$TRAIN" \
    && sed -i '/self\.history\["val_recall"\]\.append(/a\                print(f"METRIC:val_recall={val_recall}:step={step_ndx}", flush=True)' "$TRAIN" \
    && sed -i '/self\.history\["val_fp_per_hr"\]\.append(/a\                print(f"METRIC:val_fp_per_hr={val_fp_per_hr}:step={step_ndx}", flush=True)' "$TRAIN" \
    && sed -i '/self\.history\["val_n_fp"\]\.append(/a\                print(f"METRIC:val_n_fp={val_fp.detach().cpu().numpy()}:step={step_ndx}", flush=True)' "$TRAIN"

# Copy training scripts
COPY train.py .
COPY setup-data.sh .
RUN chmod +x setup-data.sh
