from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Generator
import numpy as np
import sounddevice as sd
from src.config import SAMPLE_RATE, CHUNK_SAMPLES, STEP_SAMPLES

class AudioCapturer:

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
        step_samples: int = STEP_SAMPLES,
    ) -> None:
        self._sr          = sample_rate
        self._chunk_size  = chunk_samples
        self._step_size   = step_samples
        self._buffer: deque[float] = deque(maxlen=chunk_samples * 4)
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self._running     = False
        self._stream      = None
        self._thread      = None
        self._samples_accumulated = 0

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[audio_capture] Status: {status}")

        # Add new samples to rolling buffer
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        self._buffer.extend(mono.tolist())
        self._samples_accumulated += len(mono)

        # Emit a chunk whenever we have accumulated enough NEW samples (step_size)
        if self._samples_accumulated >= self._step_size and len(self._buffer) >= self._chunk_size:
            chunk = np.array(list(self._buffer)[-self._chunk_size:], dtype=np.float32)
            self._samples_accumulated = 0
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=int(self._sr * 0.01),
            callback=self._callback,
        )
        self._stream.start()
        print(f"[audio_capture] Mic stream started @ {self._sr} Hz ✓")
        print(f"[audio_capture] Chunk: {self._chunk_size} samples ({self._chunk_size/self._sr:.2f}s)")
        print(f"[audio_capture] Step:  {self._step_size} samples  ({self._step_size/self._sr:.2f}s)")

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[audio_capture] Mic stream stopped ✓")

    def chunks(self) -> Generator[np.ndarray, None, None]:
        while self._running:
            try:
                chunk = self._queue.get(timeout=1.0)
                yield chunk
            except queue.Empty:
                continue

    def queue_size(self) -> int:
        return self._queue.qsize()
