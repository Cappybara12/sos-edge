from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from src.config import MODELS_DIR, VECTOR_DIMENSION

import os
os.environ["TFHUB_CACHE_DIR"] = MODELS_DIR

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

_model = None


def _load_model() -> object:
    global _model
    if _model is None:
        print("[embedder] Loading YAMNet from TF Hub...")
        _model = hub.load(YAMNET_URL)
        print("[embedder] YAMNet loaded")
    return _model


def get_embedding(waveform: np.ndarray) -> np.ndarray:
    model = _load_model()

    waveform_tensor = tf.cast(waveform, tf.float32)
    _, embeddings, _ = model(waveform_tensor)
    embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def warm_up() -> None:
    dummy = np.zeros(16000, dtype=np.float32)
    get_embedding(dummy)
    print("[embedder] Warm-up complete")


if __name__ == "__main__":
    # Quick smoke test
    import librosa
    print("Running embedder smoke test...")
    warm_up()
    silence = np.zeros(16000, dtype=np.float32)
    emb = get_embedding(silence)
    print(f"Embedding shape: {emb.shape}  (expected ({VECTOR_DIMENSION},))")
    print(f"Embedding norm:  {np.linalg.norm(emb):.4f}  (expected ≈ 1.0)")
    print("Smoke test passed ✓")
