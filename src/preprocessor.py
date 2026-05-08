from __future__ import annotations

import numpy as np
import librosa

from src.config import SAMPLE_RATE, CHUNK_SAMPLES


def preprocess_waveform(waveform: np.ndarray, source_sr: int = SAMPLE_RATE) -> np.ndarray:
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=-1)

    if source_sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform.astype(np.float32), orig_sr=source_sr, target_sr=SAMPLE_RATE)

    waveform = waveform.astype(np.float32)

    if len(waveform) < CHUNK_SAMPLES:
        waveform = np.pad(waveform, (0, CHUNK_SAMPLES - len(waveform)))
    elif len(waveform) > CHUNK_SAMPLES:
        waveform = waveform[:CHUNK_SAMPLES]

    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak

    return waveform


def load_audio_file(filepath: str) -> np.ndarray:
    waveform, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    return preprocess_waveform(waveform, source_sr=SAMPLE_RATE)


def preprocess_mic_chunk(chunk: np.ndarray) -> np.ndarray:
    return preprocess_waveform(chunk.flatten(), source_sr=SAMPLE_RATE)
