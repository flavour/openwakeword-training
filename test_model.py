#!/usr/bin/env python3
"""
Test a trained OpenWakeWord model with microphone input.

Usage:
    python test_model.py --model my_custom_model/hey_cal.onnx
    python test_model.py --model my_custom_model/hey_cal.onnx --threshold 0.3
"""

import argparse
import time
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll

# Suppress ALSA warnings on Linux
try:
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def _alsa_error_handler(filename, line, function, err, fmt):
        pass
    _c_error_handler = ERROR_HANDLER_FUNC(_alsa_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(_c_error_handler)
except Exception:
    pass

import numpy as np
import pyaudio
from openwakeword.model import Model

# Audio settings - OpenWakeWord expects 80ms chunks at 16kHz
CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


def main():
    parser = argparse.ArgumentParser(description="Test a wake word model with microphone")
    parser.add_argument("--model", required=True, help="Path to .onnx model file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0-1.0)")
    args = parser.parse_args()

    print("Loading model...")
    start = time.time()
    oww_model = Model(wakeword_model_paths=[args.model])
    print(f"Model loaded in {time.time() - start:.2f}s")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print(f"\nListening (threshold: {args.threshold}) - Ctrl+C to stop")
    print("=" * 50)

    try:
        while True:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            start = time.time()
            prediction = oww_model.predict(audio_array)
            inference_ms = (time.time() - start) * 1000

            for model_name, score in prediction.items():
                if score > args.threshold:
                    print(f"DETECTED: {model_name} (score: {score:.3f}, inference: {inference_ms:.1f}ms)")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
