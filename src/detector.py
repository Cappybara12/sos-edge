from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from src.config import (
    SIMILARITY_THRESHOLD,
    SMOOTHING_HITS_REQUIRED,
    SMOOTHING_WINDOW_SECONDS,
)
from src.embedder import get_embedding
from src.preprocessor import preprocess_mic_chunk
from src.vector_store import VectorStore

# Skip chunks below this RMS — filters out very quiet ambient noise
_AMPLITUDE_GATE_RMS: float = 0.03

# Per-class similarity thresholds.
# Broadband ambient noise (fan/AC) scores 0.87–0.93 against car_horn and siren,
# so those classes require a higher threshold than acoustically distinct ones.
_CLASS_THRESHOLD_OVERRIDES: dict[str, float] = {
    "car_horn":    0.90,
    "scream":      0.80,  # Lowered threshold so screams are detected easier
}

# Impulsive sounds (gunshot, glass break) are short — only require 2 hits
# instead of the global default (3) to fire an alert.
_CLASS_HITS_REQUIRED: dict[str, int] = {
    "scream":      2,     # Requires fewer hits since screams can be short
}


@dataclass
class DetectionEvent:
    timestamp: float
    score: float
    sound_type: str
    severity: str
    description: str = ""
    hit_count: int = 1


class Detector:

    def __init__(
        self,
        store: VectorStore,
        on_alert: Callable[[DetectionEvent], None],
        threshold: float              = SIMILARITY_THRESHOLD,
        hits_required: int            = SMOOTHING_HITS_REQUIRED,
        window_secs: int              = SMOOTHING_WINDOW_SECONDS,
    ) -> None:
        self._store        = store
        self._on_alert     = on_alert
        self._threshold    = threshold
        self._hits_required = hits_required
        self._window_secs  = window_secs

        self._hit_window: deque[tuple[float, DetectionEvent]] = deque()
        self.chunks_processed = 0
        self.total_hits       = 0
        self.total_alerts     = 0

    def process_chunk(self, raw_chunk: np.ndarray) -> Optional[DetectionEvent]:
        self.chunks_processed += 1
        
        # Amplitude gate — skip embedding for very quiet chunks
        raw_rms = float(np.sqrt(np.mean(raw_chunk.astype(np.float32) ** 2)))
        if raw_rms < _AMPLITUDE_GATE_RMS:
            return None

        waveform = preprocess_mic_chunk(raw_chunk)

        # ── 2. Embed
        embedding = get_embedding(waveform)
        results = self._store.search(embedding, top_k=5, only_alerts=True)

        if not results:
            return None

        best = results[0]
        best_score = best["score"]
        sound_type_early = best.get("payload", {}).get("sound_type", "")
        effective_threshold = _CLASS_THRESHOLD_OVERRIDES.get(
            sound_type_early, self._threshold
        )

        if best_score < effective_threshold:
            return None

        self.total_hits += 1
        payload = best.get("payload", {})
        event = DetectionEvent(
            timestamp  = time.time(),
            score      = best_score,
            sound_type = payload.get("sound_type", "unknown"),
            severity   = payload.get("severity", "unknown"),
            description= payload.get("description", ""),
        )

        self._hit_window.append((time.time(), event))

        now = time.time()
        while self._hit_window and (now - self._hit_window[0][0]) > self._window_secs:
            self._hit_window.popleft()

        recent_hits = len(self._hit_window)
        hits_required = _CLASS_HITS_REQUIRED.get(event.sound_type, self._hits_required)

        print(
            f"[detector] Hit! score={best_score:.3f}  "
            f"type={event.sound_type}  severity={event.severity}  "
            f"recent_hits={recent_hits}/{hits_required}"
        )

        if recent_hits >= hits_required:
            self.total_alerts += 1
            event.hit_count = recent_hits
            self._hit_window.clear()
            self._on_alert(event)
            return event

        return None

    def stats(self) -> dict:
        return {
            "chunks_processed": self.chunks_processed,
            "total_hits":       self.total_hits,
            "total_alerts":     self.total_alerts,
        }
