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
# Use a Python script to avoid sed quoting issues with f-strings
RUN TRAIN=/opt/openwakeword/openwakeword/train.py \
    && python3 -c "
import re
with open('$TRAIN') as f:
    code = f.read()

# Insert print statements after each history append
patches = [
    ('self.history[\"loss\"].append(loss.detach().cpu().numpy())',
     'self.history[\"loss\"].append(loss.detach().cpu().numpy())\n                    _m = self.history[\"loss\"][-1]; print(f\"METRIC:loss={_m}:step={step_ndx}\", flush=True)'),
    ('self.history[\"recall\"].append(self.recall(accumulated_predictions, accumulated_labels).detach().cpu().numpy())',
     'self.history[\"recall\"].append(self.recall(accumulated_predictions, accumulated_labels).detach().cpu().numpy())\n                    _m = self.history[\"recall\"][-1]; print(f\"METRIC:recall={_m}:step={step_ndx}\", flush=True)'),
    ('self.history[\"val_accuracy\"].append(val_acc.detach().cpu().numpy())',
     'self.history[\"val_accuracy\"].append(val_acc.detach().cpu().numpy())\n                _m = self.history[\"val_accuracy\"][-1]; print(f\"METRIC:val_accuracy={_m}:step={step_ndx}\", flush=True)'),
    ('self.history[\"val_recall\"].append(val_recall)',
     'self.history[\"val_recall\"].append(val_recall)\n                _m = self.history[\"val_recall\"][-1]; print(f\"METRIC:val_recall={_m}:step={step_ndx}\", flush=True)'),
    ('self.history[\"val_fp_per_hr\"].append(val_fp_per_hr)',
     'self.history[\"val_fp_per_hr\"].append(val_fp_per_hr)\n                _m = self.history[\"val_fp_per_hr\"][-1]; print(f\"METRIC:val_fp_per_hr={_m}:step={step_ndx}\", flush=True)'),
    ('self.history[\"val_n_fp\"].append(val_fp.detach().cpu().numpy())',
     'self.history[\"val_n_fp\"].append(val_fp.detach().cpu().numpy())\n                _m = self.history[\"val_n_fp\"][-1]; print(f\"METRIC:val_n_fp={_m}:step={step_ndx}\", flush=True)'),
]
for old, new in patches:
    code = code.replace(old, new, 1)
with open('$TRAIN', 'w') as f:
    f.write(code)
print('Patched', len(patches), 'metric lines')
"

# Copy training scripts
COPY train.py .
COPY setup-data.sh .
RUN chmod +x setup-data.sh
