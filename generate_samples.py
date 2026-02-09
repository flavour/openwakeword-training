# Dummy module — openwakeword's train.py imports this unconditionally
# but we use Kokoro TTS for sample generation, not Piper.

def generate_samples(*args, **kwargs):
    raise RuntimeError("Piper sample generation not available. Use Kokoro TTS via train.py instead.")
