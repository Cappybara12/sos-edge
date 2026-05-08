from __future__ import annotations

import signal
import sys
import time
from datetime import datetime

from src.alert_usage import Alerter
from src.AudioCapturing import AudioCapturer
from src.config import (
    ALERT_COOLDOWN_SECONDS,
    CHUNK_DURATION_SECONDS,
    OVERLAP_SECONDS,
    SAMPLE_RATE,
    SHARD_DIRECTORY,
    SIMILARITY_THRESHOLD,
    SMOOTHING_HITS_REQUIRED,
    SMOOTHING_WINDOW_SECONDS,
)
from src.detector import Detector, DetectionEvent
from src.embedder import warm_up
from src.vector_store import VectorStore

BANNER = r"""
  ____  ___  ____     ___  _  _    _      ___  ____  ____
 / ___||_ _||  _ \   / _ \| \| |  | |    |_ _||  _ \| ___|
 \___ \ | | | |_) | | | | |  ` |  | |     | | | | | | _|
  ___) || | |  __/  | |_| | |\  |  | |___ | | | |_| | |___
 |____/|___||_|      \___/|_| \_|  |_____|___||____/|_____|

  In-Car SOS Detector · On-Device Vector Search · Qdrant Edge
"""


def print_config() -> None:
    print(f"""
  Configuration:
  ─────────────────────────────────────────────
  Sample rate:       {SAMPLE_RATE} Hz
  Chunk duration:    {CHUNK_DURATION_SECONDS}s (overlap: {OVERLAP_SECONDS}s)
  Shard directory:   {SHARD_DIRECTORY}
  ─────────────────────────────────────────────
  Similarity threshold:  {SIMILARITY_THRESHOLD}
  Hits required:         {SMOOTHING_HITS_REQUIRED} within {SMOOTHING_WINDOW_SECONDS}s
  Alert cooldown:        {ALERT_COOLDOWN_SECONDS}s
  ─────────────────────────────────────────────
""")


def main() -> None:
    print(BANNER)
    print_config()

    print("[main] Opening Qdrant Edge shard...")
    store = VectorStore()
    try:
        store.open()
    except Exception as e:
        print(f"\n[main] Could not open shard: {e}")
        print("[main] Run this first: python scripts/indexing_sounds.py\n")
        sys.exit(1)

    count = store.count()
    if count == 0:
        print("[main] Shard is empty — run `python scripts/indexing_sounds.py` first.")
        store.close()
        sys.exit(1)
    print(f"[main] Shard ready — {count} indexed sound vectors")

    print("[main] Loading YAMNet model...")
    warm_up()

    alerter = Alerter()

    def on_alert(event: DetectionEvent) -> None:
        alerter.handle(event)

    detector = Detector(
        store     = store,
        on_alert  = on_alert,
    )

    capturer = AudioCapturer()

    def shutdown(sig=None, frame=None) -> None:
        print("\n\n[main] Shutting down...")
        capturer.stop()
        store.close()
        stats = detector.stats()
        print(f"\n[main] Session stats:")
        print(f"  Chunks processed: {stats['chunks_processed']}")
        print(f"  Total hits:       {stats['total_hits']}")
        print(f"  Alerts sent:      {stats['total_alerts']}")
        print("[main] Goodbye.\n")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    capturer.start()

    print(f"\n[main] Listening for distress sounds... (Ctrl+C to stop)\n")

    last_stats_time = time.time()
    STATS_INTERVAL  = 30   # print stats every 30 seconds

    for chunk in capturer.chunks():
        detector.process_chunk(chunk)

        # Periodic status line
        now = time.time()
        if now - last_stats_time >= STATS_INTERVAL:
            stats = detector.stats()
            ts    = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{ts}] Running — "
                f"chunks={stats['chunks_processed']}  "
                f"hits={stats['total_hits']}  "
                f"alerts={stats['total_alerts']}  "
                f"queue={capturer.queue_size()}"
            )
            last_stats_time = now


if __name__ == "__main__":
    main()
