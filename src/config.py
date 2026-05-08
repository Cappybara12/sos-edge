"""
config.py — Central configuration loaded from .env
All modules import from here; nothing reads .env directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (two levels up from src/)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# Telegram 
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Detection 
SIMILARITY_THRESHOLD: float   = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))
SMOOTHING_HITS_REQUIRED: int  = int(os.getenv("SMOOTHING_HITS_REQUIRED", "3"))
SMOOTHING_WINDOW_SECONDS: int = int(os.getenv("SMOOTHING_WINDOW_SECONDS", "5"))
ALERT_COOLDOWN_SECONDS: int   = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))

# ── Audio 
SAMPLE_RATE: int             = int(os.getenv("SAMPLE_RATE", "16000"))
CHUNK_DURATION_SECONDS: float = float(os.getenv("CHUNK_DURATION_SECONDS", "1"))
OVERLAP_SECONDS: float        = float(os.getenv("OVERLAP_SECONDS", "0.5"))

# ── Storage 
SHARD_DIRECTORY: str = os.getenv("SHARD_DIRECTORY", "./qdrant-edge-shard")
MODELS_DIR: str      = os.getenv("MODELS_DIR", "./models_cache")

# ── Qdrant Edge 
VECTOR_NAME      = "audio_embedding"
VECTOR_DIMENSION = 1024      # YAMNet output dimension
COLLECTION_NAME  = "distress_sounds"

# ── Derived 
CHUNK_SAMPLES    = int(SAMPLE_RATE * CHUNK_DURATION_SECONDS)   # 16000
OVERLAP_SAMPLES  = int(SAMPLE_RATE * OVERLAP_SECONDS)          # 8000
STEP_SAMPLES     = CHUNK_SAMPLES - OVERLAP_SAMPLES             # 8000  (new audio per step)
