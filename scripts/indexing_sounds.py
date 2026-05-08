from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from src.config import SHARD_DIRECTORY
from src.embedder import get_embedding, warm_up
from src.preprocessor import load_audio_file
from src.vector_store import VectorStore

DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
CUSTOM_DIR     = DATA_DIR / "custom"
ENGINE_NOISE   = DATA_DIR / "car_engine_idle.mp3"
ESC50_URL      = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ESC50_ZIP      = DATA_DIR / "ESC-50.zip"
ESC50_DIR      = DATA_DIR / "ESC-50"

ALERT_CLASSES = {
    "crying_baby":    {"sound_type": "crying",        "severity": "medium"},
    "screaming":      {"sound_type": "scream",         "severity": "high"},
    "glass_breaking": {"sound_type": "glass_break",   "severity": "high"},
    "car_horn":       {"sound_type": "car_horn",       "severity": "medium"},
    "siren":          {"sound_type": "siren",          "severity": "high"},
    "gunshot":        {"sound_type": "collision",      "severity": "high"},
    "dog":            {"sound_type": "dog_bark",       "severity": "low"},
    "tire_screech":   {"sound_type": "tire_screech",  "severity": "high"},   # custom samples
}

NEGATIVE_CLASSES = {
    "talking":        {"sound_type": "speech",         "severity": "none"},
    "laughing":       {"sound_type": "speech",         "severity": "none"},
    "clapping":       {"sound_type": "clapping",       "severity": "none"},
    "helicopter":     {"sound_type": "engine",         "severity": "none"},
    "chainsaw":       {"sound_type": "engine",         "severity": "none"},
    "engine":         {"sound_type": "engine",         "severity": "none"},
    "rain":           {"sound_type": "ambient",        "severity": "none"},
    "wind":           {"sound_type": "ambient",        "severity": "none"},
    "crickets":       {"sound_type": "ambient",        "severity": "none"},
}


def download_esc50() -> bool:
    if ESC50_DIR.exists() and any(ESC50_DIR.rglob("*.wav")):
        print(f"[index] ESC-50 already exists at {ESC50_DIR}")
        return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[index] Downloading ESC-50 dataset...")
    print(f"[index] URL: {ESC50_URL}")

    try:
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                print(f"\r[index] Downloading... {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(ESC50_URL, ESC50_ZIP, reporthook=progress)
        print()

        print(f"[index] Extracting ZIP...")
        with zipfile.ZipFile(ESC50_ZIP, "r") as z:
            z.extractall(DATA_DIR)

        extracted = DATA_DIR / "ESC-50-master"
        if extracted.exists():
            extracted.rename(ESC50_DIR)

        ESC50_ZIP.unlink(missing_ok=True)
        print(f"[index] ESC-50 extracted to {ESC50_DIR}")
        return True

    except Exception as e:
        print(f"[index] Download failed: {e}")
        return False


def find_audio_files(class_name: str) -> list[Path]:
    audio_dir = ESC50_DIR / "audio"
    if not audio_dir.exists():
        return []

    meta_file = ESC50_DIR / "meta" / "esc50.csv"
    if not meta_file.exists():
        return []

    matched = []
    with open(meta_file, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            filename = parts[0]
            category = parts[3].lower().replace(" ", "_")
            if class_name.lower().replace(" ", "_") in category:
                wav = audio_dir / filename
                if wav.exists():
                    matched.append(wav)

    return matched


def index_class(
    store: VectorStore,
    class_name: str,
    metadata: dict,
    is_alert: bool,
    max_files: int = 40,
) -> int:
    files = find_audio_files(class_name)
    if not files:
        print(f"[index] No files found for class '{class_name}'")
        return 0

    files = files[:max_files]
    embeddings, payloads = [], []

    for wav_path in files:
        try:
            waveform = load_audio_file(str(wav_path))
            emb      = get_embedding(waveform)
            payload  = {
                "alert_class":    "alert" if is_alert else "negative",
                "sound_type":     metadata["sound_type"],
                "severity":       metadata["severity"],
                "source_dataset": "ESC-50",
                "source_class":   class_name,
                "description":    f"{class_name} from ESC-50",
            }
            embeddings.append(emb)
            payloads.append(payload)
        except Exception as e:
            print(f"[index] Skipping {wav_path.name}: {e}")

    if embeddings:
        store.upsert_batch(embeddings, payloads)

    return len(embeddings)


def _augment_with_engine(waveform: np.ndarray) -> np.ndarray:
    """Overlay the waveform on top of engine noise to simulate in-car conditions."""
    if not ENGINE_NOISE.exists():
        return waveform
    try:
        from src.preprocessor import load_audio_file as _load
        engine = _load(str(ENGINE_NOISE))
        # Loop/trim engine noise to match waveform length
        if len(engine) < len(waveform):
            engine = np.tile(engine, int(np.ceil(len(waveform) / len(engine))))
        engine = engine[:len(waveform)]
        mixed = np.clip(0.75 * waveform + 0.25 * engine, -1.0, 1.0).astype(np.float32)
        return mixed
    except Exception:
        return waveform


def index_custom_files(store: VectorStore, class_name: str, metadata: dict) -> int:
    """Index all MP3/WAV files from data/custom/<class_name>/, with engine-noise augmentation."""
    source_dir = CUSTOM_DIR / class_name
    if not source_dir.exists():
        print(f"[index] Custom folder not found: {source_dir}")
        return 0

    files = list(source_dir.glob("*.wav")) + list(source_dir.glob("*.mp3"))
    if not files:
        print(f"[index] No audio files in {source_dir}")
        return 0

    embeddings, payloads = [], []

    for audio_path in files:
        try:
            waveform = load_audio_file(str(audio_path))
            base_payload = {
                "alert_class":    "alert",
                "sound_type":     metadata["sound_type"],
                "severity":       metadata["severity"],
                "source_dataset": "custom",
                "source_class":   class_name,
                "description":    f"{class_name} (custom sample)",
            }

            # Original clip
            embeddings.append(get_embedding(waveform))
            payloads.append(base_payload.copy())

            # Augmented with engine noise
            augmented = _augment_with_engine(waveform)
            aug_payload = base_payload.copy()
            aug_payload["description"] = f"{class_name} (custom + engine noise)"
            embeddings.append(get_embedding(augmented))
            payloads.append(aug_payload)

        except Exception as e:
            print(f"[index] Skipping {audio_path.name}: {e}")

    if embeddings:
        store.upsert_batch(embeddings, payloads)
        print(f"[index]     Indexed {len(embeddings)} embeddings ({len(files)} files × 2 variants)")

    return len(embeddings)


def run_indexing() -> None:
    print("\n" + "="*60)
    print("  SOS Detector — Audio Indexing Pipeline")
    print("="*60 + "\n")

    if not download_esc50():
        print("[index] Could not get dataset. Aborting.")
        sys.exit(1)

    print("\n[index] Loading YAMNet model...")
    warm_up()

    print(f"\n[index] Opening Qdrant Edge shard at {SHARD_DIRECTORY}...")
    store = VectorStore()
    store.open()

    total = 0

    print("\n[index] Indexing ALERT classes...")
    for class_name, metadata in ALERT_CLASSES.items():
        print(f"[index]   → {class_name} ({metadata['sound_type']}, severity={metadata['severity']})")
        n = index_class(store, class_name, metadata, is_alert=True)
        print(f"[index]     Indexed {n} embeddings")
        total += n

    print("\n[index] Indexing NEGATIVE classes (for contrast)...")
    for class_name, metadata in NEGATIVE_CLASSES.items():
        print(f"[index]   → {class_name} ({metadata['sound_type']})")
        n = index_class(store, class_name, metadata, is_alert=False, max_files=20)
        print(f"[index]     Indexed {n} embeddings")
        total += n

    # Index custom sound classes (tire screech etc.)
    custom_classes = {k: v for k, v in ALERT_CLASSES.items() if k not in [c for c in ALERT_CLASSES if k != "tire_screech"]}
    print("\n[index] Indexing CUSTOM classes...")
    for class_name, metadata in ALERT_CLASSES.items():
        if (CUSTOM_DIR / class_name).exists():
            print(f"[index]   → {class_name} ({metadata['sound_type']}, severity={metadata['severity']})")
            n = index_custom_files(store, class_name, metadata)
            total += n

    print(f"\n[index] Total points indexed: {total}")
    print(f"[index] Total points in shard: {store.count()}")
    print("\n[index] Running shard optimizer...")
    store.optimize()
    store.close()

    print("\n" + "="*60)
    print("  Indexing complete! Shard ready for inference.")
    print(f"  Shard location: {SHARD_DIRECTORY}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_indexing()
