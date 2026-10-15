"""
Sign-triggered desktop actions: holding a configured static sign for `hold_seconds`
opens the mapped app/website (e.g. W -> WhatsApp, Y -> YouTube, L -> Google).

The sign -> action table comes from config.yaml (actions.signs). Only http(s) URLs and the
whatsapp: protocol are accepted, so a config typo cannot launch arbitrary programs.
"""

import re
import subprocess
import webbrowser
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("SignActions")

DEFAULT_ACTIONS = {
    "W": {"name": "WhatsApp", "url": "https://web.whatsapp.com"},
    "Y": {"name": "YouTube", "url": "https://www.youtube.com"},
    "L": {"name": "Google", "url": "https://www.google.com"},
}

_SIGN_LETTER = re.compile(r"Sign '([^']+)'")


def sign_key(label: str) -> Optional[str]:
    """Extract the sign key from a display label: "Sign 'W' (WhatsApp)" -> "W"; other labels -> None."""
    m = _SIGN_LETTER.search(label or "")
    return m.group(1).strip().upper() if m else None


def open_url(url: str) -> bool:
    """Open a URL in the default browser (falls back to the Windows shell for whatsapp:)."""
    if url.startswith("whatsapp:"):
        subprocess.Popen(["cmd", "/c", "start", "", url])
        return True
    return bool(webbrowser.open(url))


@dataclass
class SignAction:
    key: str
    name: str
    url: str


class SignActionTrigger:
    """Tracks how long one mapped sign has been held and fires its action once per cooldown."""

    def __init__(self, signs: Optional[Dict[str, Dict[str, str]]] = None, hold_seconds: float = 1.0,
                 cooldown_seconds: float = 5.0, min_confidence: float = 0.70,
                 opener: Callable[[str], bool] = open_url):
        self.actions: Dict[str, SignAction] = {}
        for key, spec in (signs if signs is not None else DEFAULT_ACTIONS).items():
            url = str(spec["url"])
            if not (url.startswith("https://") or url.startswith("http://") or url.startswith("whatsapp:")):
                raise ValueError(f"Action for sign '{key}' has unsupported URL '{url}' (http(s):// or whatsapp: only)")
            self.actions[str(key).upper()] = SignAction(str(key).upper(), str(spec.get("name", key)), url)
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        self._opener = opener
        self._held_key: Optional[str] = None
        self._held_since = 0.0
        self._last_fired = -1e9

    def update(self, label: str, confidence: float, now: float) -> Tuple[Optional[SignAction], float, Optional[SignAction]]:
        """
        Feed the current prediction. Returns (action being held or None, hold progress 0..1,
        action fired on this call or None).
        """
        key = sign_key(label)
        action = self.actions.get(key) if key else None
        if action is None or confidence < self.min_confidence:
            self._held_key = None
            return None, 0.0, None
        if key != self._held_key:
            self._held_key, self._held_since = key, now
        progress = min(1.0, (now - self._held_since) / self.hold_seconds) if self.hold_seconds > 0 else 1.0
        if progress >= 1.0 and now - self._last_fired > self.cooldown_seconds:
            self._last_fired = now
            self._held_key = None
            try:
                self._opener(action.url)
                logger.info(f"Sign '{key}' held -> opened {action.name} ({action.url})")
            except OSError as e:
                logger.error(f"Failed to open {action.name}: {e}")
            return action, 0.0, action
        return action, progress, None


def build_trigger_from_config(cfg: dict) -> Optional[SignActionTrigger]:
    """Create the trigger from the config 'actions' section; None when disabled."""
    a = cfg.get("actions", {}) or {}
    if not a.get("enabled", True):
        return None
    return SignActionTrigger(
        signs=a.get("signs") or DEFAULT_ACTIONS,
        hold_seconds=float(a.get("hold_seconds", 1.0)),
        cooldown_seconds=float(a.get("cooldown_seconds", 5.0)),
        min_confidence=float(a.get("min_confidence", 0.70)),
    )
