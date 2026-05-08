from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.alert_usage import Alerter, send_test_telegram
from src.config import SAMPLE_RATE, CHUNK_SAMPLES, SHARD_DIRECTORY
from src.detector import Detector, DetectionEvent
from src.embedder import warm_up
from src.vector_store import VectorStore


def make_synthetic_distress() -> np.ndarray:
    t  = np.linspace(0, 1.0, CHUNK_SAMPLES, dtype=np.float32)
    waveform = 0.9 * np.sin(2 * np.pi * 2000 * t)
    # Add harmonics for richer content
    waveform += 0.3 * np.sin(2 * np.pi * 4000 * t)
    waveform += 0.1 * np.sin(2 * np.pi * 6000 * t)
    return waveform.astype(np.float32)


def make_synthetic_silence() -> np.ndarray:
    return np.zeros(CHUNK_SAMPLES, dtype=np.float32)


def run_tests(audio_file: str | None = None) -> None:
    print("\n" + "="*60)
    print("  SOS Detector — Pipeline Test")
    print("="*60 + "\n")

    # ── Test 1: Telegram ──────────────────────────────────────────────────────
    print("[test] Sending Telegram test message...")
    ok = send_test_telegram(
        "SOS Bot Test Pipeline is online and working!\n"
        "This is a test message from the in-car SOS detector."
    )
    if ok:
        print("[test] Telegram connected successfully!")
    else:
        print("[test] Telegram failed — check your token and chat ID in .env")

    # ── Test 2: Load shard ────────────────────────────────────────────────────
    print(f"\n[test] Opening Qdrant Edge shard at {SHARD_DIRECTORY}...")
    store = VectorStore()
    try:
        store.open()
    except Exception as e:
        print(f"[test] Could not open shard: {e}")
        print("[test] Run `python scripts/indexing_sounds.py` first to build the shard.")
        return

    print(f"[test] Shard opened. Total points: {store.count()}")

    print("\n[test] Loading YAMNet model...")
    warm_up()

    # ── Test 4: Detection test ────────────────────────────────────────────────
    alerts_triggered = []

    def on_alert(event: DetectionEvent) -> None:
        alerts_triggered.append(event)
        print(f"[test] ALERT TRIGGERED: {event.sound_type} (score={event.score:.4f})")

    alerter = Alerter()
    detector = Detector(
        store       = store,
        on_alert    = lambda event: (alerter.handle(event), on_alert(event)),
        hits_required = 1,    # Lowered to 1 for testing (real system uses 3)
        window_secs  = 10,
    )

    # Prepare test audio
    if audio_file:
        print(f"\n[test] Loading audio file: {audio_file}")
        from src.preprocessor import load_audio_file
        chunks = [load_audio_file(audio_file)]
        print(f"[test] Running detection on real audio file...")
    else:
        print("\n[test] Using synthetic audio (silence + distress burst)...")
        chunks = [
            make_synthetic_silence(),   # should not trigger
            make_synthetic_distress(),  # might trigger depending on shard content
            make_synthetic_silence(),   # should not trigger
        ]

    for i, chunk in enumerate(chunks):
        print(f"\n[test] Processing chunk {i+1}/{len(chunks)}...")
        result = detector.process_chunk(chunk)
        if result:
            print(f"[test] Detection result: {result.sound_type}, score={result.score:.4f}")
        else:
            print(f"[test] No alert (below threshold or not enough hits)")
        time.sleep(0.2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    stats = detector.stats()
    print(f"  Chunks processed: {stats['chunks_processed']}")
    print(f"  Total hits:       {stats['total_hits']}")
    print(f"  Alerts triggered: {stats['total_alerts']}")
    print("="*60)

    store.close()


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_tests(audio_file)
