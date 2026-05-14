#  In-Car SOS with On-Device Vector Search

**Detecting distress sounds passively using Qdrant Edge — no cloud, no raw audio upload, no manual SOS button.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Qdrant Edge](https://img.shields.io/badge/Qdrant-Edge-red.svg)](https://qdrant.tech/documentation/edge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This system continuously monitors in-car audio and detects distress sounds (screams, glass breaking, tire screeches, collisions) using **on-device vector similarity search** powered by Qdrant Edge. When distress is confirmed, an alert is sent via Telegram.

```
USB Mic → YAMNet (1024-d embeddings) → Qdrant Edge ANN Search → Telegram Alert
```

**Privacy guarantee:** Raw audio never leaves the device. Only outbound call is the Telegram alert.

---

## Architecture
<img width="3296" height="7197" alt="NLLM Codebase Embedding-2026-05-07-190145" src="https://github.com/user-attachments/assets/24175d28-35f2-44d2-ad17-5e3a5d0d6600" />

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Telegram bot token and chat ID
```

### 3. Index sound library (one-time)

```bash
python scripts/indexing_sounds.py
```

This downloads the ESC-50 dataset, incorporates any custom samples from `data/custom/`, generates YAMNet embeddings for all distress + negative sounds (with engine noise augmentation for custom samples), and builds the local Qdrant Edge shard (~6 MB).

### 4. Test the pipeline

```bash
python scripts/test_detection.py
```

Verifies Telegram connectivity, loads the shard, and runs synthetic audio through the detector.

### 5. Start the live detector

```bash
python main.py
```

---

## Configuration

All tunable parameters live in `.env`:

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | Your bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | — | Your personal or group chat ID |
| `SIMILARITY_THRESHOLD` | `0.80` | Cosine similarity cutoff (0–1) |
| `SMOOTHING_HITS_REQUIRED` | `3` | Hits needed within window to alert |
| `SMOOTHING_WINDOW_SECONDS` | `5` | Time window for hit counting |
| `ALERT_COOLDOWN_SECONDS` | `30` | Min seconds between consecutive alerts |
| `CHUNK_DURATION_SECONDS` | `1` | Audio window size |
| `OVERLAP_SECONDS` | `0.5` | Overlap between windows |

### Advanced False-Positive Filtering (`src/detector.py`)
To prevent false alarms in noisy environments, the system uses custom per-class overrides:
- **Amplitude Gating**: The system strictly ignores low-volume ambient noises (like AC or engine hums) before they even reach the AI model.
- **Strict Mode (`car_horn`, `siren`)**: Broadband noises like wind can mimic sirens/horns. These require a very high similarity score (e.g., `0.90` or `0.96`) to trigger.
- **Sensitive Mode (`scream`, `gunshot`)**: Since emergency human screams and gunshots can be brief or muffled, they are highly sensitive (score `0.80` to `0.85`) and require fewer consecutive hits to alert you instantly.

---

## Sound Classes

**Alert sounds (trigger SOS):**
- `scream` / `crying` → severity: high / medium *(includes custom datasets)*
- `glass_break` → severity: high
- `collision` / `gunshot` → severity: high *(includes custom datasets)*
- `siren` → severity: high
- `car_horn` → severity: medium

**Negative sounds (do NOT trigger):**
- Normal speech, music, engine noise, ambient sounds

---

## Tech Stack

| Component | Technology |
|---|---|
| Vector database | [Qdrant Edge](https://qdrant.tech/documentation/edge/) (`qdrant-edge-py`) |
| Audio embedding | [YAMNet](https://tfhub.dev/google/yamnet/1) via TF Hub (1024-d) |
| Audio preprocessing | [librosa](https://librosa.org/) |
| Audio capture | [sounddevice](https://python-sounddevice.readthedocs.io/) |
| Alert delivery | Telegram Bot API |
| Datasets | [ESC-50](https://github.com/karoldvl/ESC-50), custom `.wav`/`.mp3` folders |

---

## Screenshots
<img width="678" height="177" alt="Screenshot 2026-05-09 at 2 37 06 AM" src="https://github.com/user-attachments/assets/56b5261e-9bc4-4d32-a5a4-df256c66cc07" />
<img width="693" height="207" alt="Screenshot 2026-05-09 at 2 37 17 AM" src="https://github.com/user-attachments/assets/8ad20746-7fc5-4975-9797-0348c47ae19d" />

<img width="694" height="523" alt="Screenshot 2026-05-09 at 2 36 54 AM" src="https://github.com/user-attachments/assets/c90158fa-19ee-4363-8162-4601482c3ff3" />


## Project Structure

```
qdr-edge/
├── main.py                    # Entry point
├── requirements.txt
├── .env.example               # Config template
├── scripts/
│   ├── index_sounds.py        # One-time indexing pipeline
│   └── test_detection.py      # End-to-end test
└── src/
    ├── config.py              # Environment config
    ├── audio_capture.py       # Real-time mic capture
    ├── preprocessor.py        # Audio normalization
    ├── embedder.py            # YAMNet wrapper
    ├── vector_store.py        # Qdrant Edge interface
    ├── detector.py            # Detection + smoothing
    └── alerter.py             # Telegram + console alerts
```

---

## References

- [Qdrant Edge Docs](https://qdrant.tech/documentation/edge/)
- [YAMNet on TF Hub](https://tfhub.dev/google/yamnet/1)
- [ESC-50 Dataset](https://github.com/karoldvl/ESC-50)
- [Telegram Bot API](https://core.telegram.org/bots/api)
