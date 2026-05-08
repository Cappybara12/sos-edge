from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
import requests

from src.config import (
    ALERT_COOLDOWN_SECONDS,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)
from src.detector import DetectionEvent
# ANSI colours for terminal output
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

class Alerter:

    def __init__(
        self,
        cooldown_seconds: int = ALERT_COOLDOWN_SECONDS,
    ) -> None:
        self._cooldown  = cooldown_seconds
        self._last_alert_time: float = 0.0

    def handle(self, event: DetectionEvent) -> None:
        now = time.time()
        if (now - self._last_alert_time) < self._cooldown:
            remaining = int(self._cooldown - (now - self._last_alert_time))
            print(f"[alerter] Alert suppressed (cooldown: {remaining}s remaining)")
            return

        self._last_alert_time = now

        self._console_alert(event)
        self._telegram_alert(event)
        self._beep()

    def _console_alert(self, event: DetectionEvent) -> None:
        ts  = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        sev = event.severity.upper()
        colour = _RED if sev == "HIGH" else _YELLOW

        print(
            f"\n{_BOLD}{colour}"
            f"{'='*60}\n"
            f"IN-CAR SOS ALERT\n"
            f"{'='*60}{_RESET}\n"
            f"  Time:        {ts}\n"
            f"  Sound:       {event.sound_type}\n"
            f"  Severity:    {sev}\n"
            f"  Score:       {event.score:.4f}\n"
            f"  Hits:        {event.hit_count}\n"
            f"  Description: {event.description}\n"
            f"{colour}{'='*60}{_RESET}\n"
        )
    def _telegram_alert(self, event: DetectionEvent) -> bool:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("[alerter] Telegram not configured — skipping")
            return False

        ts  = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        message = (
            f"IN-CAR SOS ALERT\n\n"
            f"Severity: {event.severity.upper()}\n"
            f"Sound Detected: {event.sound_type}\n"
            f"Match Score: {event.score:.4f}\n"
            f"Hits in Window: {event.hit_count}\n"
            f"Time: {ts}\n\n"
            f"Powered by Qdrant Edge · On-device detection"
        )

        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            response = requests.post(
                url,
                json={
                    "chat_id":    TELEGRAM_CHAT_ID,
                    "text":       message,
                },
                timeout=10,
            )

            if response.status_code == 200:
                print(f"[alerter] Telegram alert sent to chat {TELEGRAM_CHAT_ID}")
                return True
            else:
                print(f"[alerter] Telegram error: {response.status_code} — {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"[alerter] Telegram request failed: {e}")
            return False
    def _beep(self) -> None:
        try:
            if sys.platform == "darwin":
                subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"], check=False, timeout=3)
            elif sys.platform.startswith("linux"):
                result = subprocess.run(
                    ["paplay", "/usr/share/sounds/alsa/Front_Center.wav"],
                    check=False, timeout=3, capture_output=True
                )
                if result.returncode != 0:
                    subprocess.run(["aplay", "-q", "/usr/share/sounds/alsa/Front_Center.wav"],
                                   check=False, timeout=3)
            print("[alerter] Audio alert triggered")
        except Exception as e:
            print(f"[alerter] Beep failed (non-critical): {e}")
def send_test_telegram(message: str = "SOS Bot test connection OK!") -> bool:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[alerter] Test message failed: {e}")
        return False
