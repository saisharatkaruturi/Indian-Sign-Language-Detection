"""
Guided live sign test: the app shows a target sign (A-Z by default), the user signs it and
presses SPACE, the recognizer's labels are collected for `capture_seconds`, and the
majority label is compared with the target. Results are saved as CSV + JSON summary under
evaluation_results/live_tests/ so the outcome can be shown as evidence.

Run:  python src/main.py --mode letter_test
"""

import json
import re
import string
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import pandas as pd

from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger("LiveSignTest")

DEFAULT_TARGETS = list(string.ascii_uppercase)
NUMBER_TARGETS = [str(i) for i in range(6)]
_KEY = re.compile(r"(?:Sign|Number) '([^']+)'")


def target_key(label: str) -> str:
    """Sign 'A' (Fist) -> "A", Number '3' -> "3"; other labels unchanged."""
    m = _KEY.search(label or "")
    return m.group(1).strip().upper() if m else (label or "")


class SignTestSession:
    """UI-independent bookkeeping for a guided live test (unit-testable)."""

    def __init__(self, targets: Optional[List[str]] = None, capture_seconds: float = 2.0):
        self.targets = [t.upper() for t in (targets or DEFAULT_TARGETS)]
        self.capture_seconds = capture_seconds
        self.index = 0
        self.results: List[Dict[str, object]] = []
        self._capturing_since: Optional[float] = None
        self._labels: List[str] = []

    @property
    def done(self) -> bool:
        return self.index >= len(self.targets)

    @property
    def current(self) -> Optional[str]:
        return None if self.done else self.targets[self.index]

    @property
    def capturing(self) -> bool:
        return self._capturing_since is not None

    def start_capture(self, now: float):
        if not self.done:
            self._capturing_since, self._labels = now, []

    def feed(self, label: str, now: float) -> Optional[Dict[str, object]]:
        """Add one frame's label while capturing; returns the result once the window closes."""
        if not self.capturing:
            return None
        self._labels.append(target_key(label))
        if now - self._capturing_since < self.capture_seconds:
            return None
        return self._finish("captured")

    def skip(self) -> Optional[Dict[str, object]]:
        if self.done:
            return None
        self._labels = []
        return self._finish("skipped")

    def _finish(self, status: str) -> Dict[str, object]:
        counts = Counter(self._labels)
        majority, votes = counts.most_common(1)[0] if counts else ("", 0)
        target = self.targets[self.index]
        result = {
            "target": target, "status": status, "recognised": majority,
            "correct": status == "captured" and str(majority).upper() == target,
            "agreement": round(votes / len(self._labels), 3) if self._labels else 0.0,
            "frames": len(self._labels),
        }
        self.results.append(result)
        self.index += 1
        self._capturing_since, self._labels = None, []
        return result

    def summary(self) -> Dict[str, object]:
        tested = [r for r in self.results if r["status"] == "captured"]
        correct = [r for r in tested if r["correct"]]
        return {
            "targets": len(self.targets), "tested": len(tested), "skipped": len(self.results) - len(tested),
            "correct": len(correct), "accuracy_on_tested": round(len(correct) / len(tested), 4) if tested else None,
            "failed": [f"{r['target']}->{r['recognised']}" for r in tested if not r["correct"]],
        }

    def save(self, out_dir: str = "evaluation_results/live_tests", prefix: str = "letter_test") -> str:
        out = resolve_path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out / f"{prefix}_{stamp}.csv"
        pd.DataFrame(self.results).to_csv(csv_path, index=False)
        with open(out / f"{prefix}_{stamp}_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
        return str(csv_path)


def run_letter_test(config_path: str = "config.yaml", targets: Optional[List[str]] = None, numbers: bool = False):
    """Interactive webcam test (A-Z, or 0-5 with numbers=True). Keys: SPACE = capture, S = skip, Q = finish."""
    from src.camera.camera import CameraManager
    from src.inference.gesture_translator import ISLGestureTranslator
    from src.utils.config import load_config

    cam_cfg = load_config(config_path).get("camera", {})
    cam = CameraManager(camera_index=cam_cfg.get("index", 0), width=cam_cfg.get("width", 640),
                        height=cam_cfg.get("height", 480), target_fps=cam_cfg.get("fps", 30))
    if not cam.start():
        logger.error("Failed to start camera hardware.")
        return None
    translator = ISLGestureTranslator()
    translator.numbers_mode = numbers
    session = SignTestSession(targets or (NUMBER_TARGETS if numbers else DEFAULT_TARGETS))
    last = None
    try:
        while not session.done:
            ok, raw = cam.read_frame()
            if not ok or raw is None:
                logger.error("Empty frame received from camera feed.")
                break
            frame = cv2.flip(raw, 1)
            out, label, conf, _, _, _ = translator.process_frame(frame)
            now = time.time()
            res = session.feed(label, now)
            if res is not None:
                last = res
                logger.info(f"{res['target']}: recognised {res['recognised']!r} -> {'PASS' if res['correct'] else 'FAIL'}")

            h, w, _ = out.shape
            cv2.rectangle(out, (0, 0), (w, 110), (20, 20, 25), -1)
            cv2.putText(out, f"Sign this: {session.current}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(out, f"Seen now: {label} ({conf * 100:.0f}%)", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            state = "CAPTURING..." if session.capturing else "SPACE = capture | S = skip | Q = finish"
            cv2.putText(out, f"{session.index + 1}/{len(session.targets)}  {state}", (15, 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120) if session.capturing else (180, 180, 180), 1)
            if last is not None:
                verdict = "PASS" if last["correct"] else ("SKIPPED" if last["status"] == "skipped" else "FAIL")
                cv2.putText(out, f"Last: {last['target']} -> {last['recognised'] or '-'}  {verdict}", (15, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if last["correct"] else (0, 0, 255), 2)
            cv2.imshow("ISL Number Test" if numbers else "ISL Letter Test", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and not session.capturing:
                session.start_capture(now)
            elif key == ord("s") and not session.capturing:
                last = session.skip()
            elif key == ord("q"):
                break
    finally:
        cam.release()
        translator.close()
    path = session.save(prefix="number_test" if numbers else "letter_test")
    summary = session.summary()
    print(f"\n========== {'NUMBER' if numbers else 'LETTER'} TEST SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print(f"Per-letter results: {path}")
    return summary
