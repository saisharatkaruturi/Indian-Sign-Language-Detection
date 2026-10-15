"""
One-shot dynamic sign recognition by reference matching.

Only one reference clip per sign is available (Sample Videos/, 61 clips), which is too
little to train the CNN + Bi-GRU model honestly. Instead, each clip is encoded with a frozen
ImageNet MobileNetV2 (no training) into a sequence of per-frame embeddings, and a query
clip is labelled with the reference sequence it matches best under dynamic time warping
(DTW) on cosine distance.

The returned score is a DTW similarity in [-1, 1], NOT a probability.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.preprocessing.video_preprocessing import extract_frames_from_video, sample_or_pad_sequence
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger("OneShotDynamic")

DEFAULT_GALLERY = "models/dynamic_oneshot/gallery.npz"
EMBED_SIZE = (224, 224)
BACKBONE_NAME = "MobileNetV2-ImageNet-avgpool"


class FrameEncoder:
    """Frozen MobileNetV2 (ImageNet weights) mapping RGB frames to L2-normalised 1280-d vectors."""

    def __init__(self):
        import tensorflow as tf
        self._tf = tf
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=EMBED_SIZE + (3,), include_top=False, weights="imagenet", pooling="avg"
        )
        self.model.trainable = False

    def encode(self, frames_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """frames_rgb: list of (H, W, 3) uint8 RGB frames -> (N, 1280) float32, unit length."""
        if len(frames_rgb) == 0:
            raise ValueError("No frames to encode")
        batch = np.stack([
            cv2.resize(f, (EMBED_SIZE[1], EMBED_SIZE[0]), interpolation=cv2.INTER_LINEAR) for f in frames_rgb
        ]).astype(np.float32)
        batch = self._tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        feats = np.asarray(self.model(batch, training=False), dtype=np.float32)
        return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)


def dtw_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    DTW over cosine distance between two unit-normalised embedding sequences (Ta, D), (Tb, D).
    Returns 1 - (accumulated cost / warping-path length), i.e. a cosine-like similarity.
    """
    cost = 1.0 - a @ b.T
    ta, tb = cost.shape
    acc = np.full((ta + 1, tb + 1), np.inf)
    steps = np.zeros((ta + 1, tb + 1))
    acc[0, 0] = 0.0
    for i in range(1, ta + 1):
        for j in range(1, tb + 1):
            prev = min((acc[i - 1, j - 1], steps[i - 1, j - 1]),
                       (acc[i - 1, j], steps[i - 1, j]),
                       (acc[i, j - 1], steps[i, j - 1]))
            acc[i, j] = cost[i - 1, j - 1] + prev[0]
            steps[i, j] = prev[1] + 1
    return float(1.0 - acc[ta, tb] / steps[ta, tb])


def aligned_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity of frame-aligned sequences of equal length (ablation baseline)."""
    return float(np.mean(np.sum(a * b, axis=1)))


def file_md5(path: Union[str, Path]) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def embed_sequence(encoder: FrameEncoder, frames_rgb: List[np.ndarray], sequence_length: int) -> np.ndarray:
    """Uniformly sample sequence_length frames (order preserved) and encode them -> (T, 1280)."""
    return encoder.encode(sample_or_pad_sequence(frames_rgb, target_len=sequence_length))


class OneShotDynamicRecognizer:
    """Labels a video / frame sequence with the best-matching reference clip from the gallery."""

    def __init__(self, gallery_path: str = DEFAULT_GALLERY, strict: bool = False,
                 encoder: Optional[FrameEncoder] = None):
        self.gallery_path = str(resolve_path(gallery_path))
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.labels: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.sequence_length = 0
        self._encoder = encoder
        try:
            self._load()
        except (FileNotFoundError, ValueError, KeyError, OSError) as e:
            self.load_error = str(e)
            if strict:
                raise
            logger.warning(f"One-shot dynamic gallery unavailable: {e}")

    def _load(self):
        if not Path(self.gallery_path).exists():
            raise FileNotFoundError(f"One-shot gallery not found: {self.gallery_path} "
                                    f"(build it with: python src/training/build_oneshot_gallery.py)")
        data = np.load(self.gallery_path, allow_pickle=False)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.labels = [str(x) for x in data["labels"]]
        self.sequence_length = int(data["sequence_length"])
        backbone = str(data["backbone"])
        if backbone != BACKBONE_NAME:
            raise ValueError(f"Gallery built with backbone '{backbone}', expected '{BACKBONE_NAME}'")
        if self.embeddings.ndim != 3 or self.embeddings.shape[0] != len(self.labels) \
                or self.embeddings.shape[1] != self.sequence_length:
            raise ValueError(f"Gallery embeddings shape {self.embeddings.shape} inconsistent with "
                             f"{len(self.labels)} labels and T={self.sequence_length}")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Gallery contains duplicate labels")
        if self._encoder is None:
            self._encoder = FrameEncoder()
        self.is_loaded = True
        logger.info(f"One-shot gallery loaded: {len(self.labels)} signs, T={self.sequence_length}, {backbone}")

    def embed(self, video_or_frames: Union[str, List[np.ndarray]]) -> np.ndarray:
        frames = extract_frames_from_video(video_or_frames) if isinstance(video_or_frames, str) else video_or_frames
        return embed_sequence(self._encoder, frames, self.sequence_length)

    def rank(self, query_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """All gallery labels sorted by DTW similarity to the query embedding sequence."""
        sims = [dtw_similarity(query_embedding, ref) for ref in self.embeddings]
        order = np.argsort(sims)[::-1]
        return [(self.labels[i], float(sims[i])) for i in order]

    def predict(self, video_or_frames: Union[str, List[np.ndarray]]) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Video path or list of RGB frames -> (label, DTW similarity, top-3)."""
        if not self.is_loaded:
            return "Waiting for reference gallery...", 0.0, []
        ranking = self.rank(self.embed(video_or_frames))
        return ranking[0][0], ranking[0][1], ranking[:3]


def build_gallery(clip_paths: Dict[str, Path], sequence_length: int, out_path: Union[str, Path],
                  encoder: Optional[FrameEncoder] = None) -> Path:
    """Encode one reference clip per label and save the gallery (.npz, no pickled objects)."""
    encoder = encoder or FrameEncoder()
    labels, embs, sources, hashes = [], [], [], []
    for label in sorted(clip_paths):
        path = Path(clip_paths[label])
        embs.append(embed_sequence(encoder, extract_frames_from_video(str(path)), sequence_length))
        labels.append(label)
        sources.append(path.name)
        hashes.append(file_md5(path))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=np.stack(embs).astype(np.float32), labels=np.array(labels),
             sources=np.array(sources), source_md5=np.array(hashes),
             sequence_length=np.int64(sequence_length), backbone=np.array(BACKBONE_NAME))
    logger.info(f"Saved one-shot gallery ({len(labels)} signs, T={sequence_length}) to {out_path}")
    return out_path


class LiveDynamicMatcher:
    """
    Rolling webcam window for a sequence recognizer exposing is_loaded / load_error /
    predict(frames) -> (label, score, top3) (OneShotDynamicRecognizer or DynamicInferenceEngine). Frames (RGB, un-mirrored) are buffered
    for window_seconds; every match_every_seconds, if a hand was visible in at least
    min_hand_fraction of the buffered frames, the window is matched in a background thread
    so the UI loop is not blocked. latest_result holds (label, similarity, top3, timestamp).
    """

    def __init__(self, recognizer, window_seconds: float = 4.0,
                 match_every_seconds: float = 1.0, min_hand_fraction: float = 0.6):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        if not recognizer.is_loaded:
            raise ValueError(f"Recognizer not loaded: {recognizer.load_error}")
        self.recognizer = recognizer
        self.window_seconds = window_seconds
        self.match_every_seconds = match_every_seconds
        self.min_hand_fraction = min_hand_fraction
        self._buffer = deque()  # (timestamp, rgb_frame, hand_present)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = None
        self._last_submit = 0.0
        self.latest_result: Optional[Tuple[str, float, List[Tuple[str, float]], float]] = None

    def push(self, frame_rgb: np.ndarray, timestamp: float, hand_present: bool):
        small = cv2.resize(frame_rgb, (EMBED_SIZE[1], EMBED_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        self._buffer.append((timestamp, small, bool(hand_present)))
        while self._buffer and timestamp - self._buffer[0][0] > self.window_seconds:
            self._buffer.popleft()

    def hand_fraction(self) -> float:
        return sum(h for _, _, h in self._buffer) / len(self._buffer) if self._buffer else 0.0

    def poll(self, now: float) -> bool:
        """Collect a finished match and start a new one if due. Returns True when a new result arrived."""
        new_result = False
        if self._pending is not None and self._pending.done():
            label, sim, top3 = self._pending.result()
            self.latest_result = (label, sim, top3, now)
            self._pending = None
            new_result = True
        span = self._buffer[-1][0] - self._buffer[0][0] if len(self._buffer) > 1 else 0.0
        if (self._pending is None and now - self._last_submit >= self.match_every_seconds
                and span >= 0.5 * self.window_seconds and self.hand_fraction() >= self.min_hand_fraction):
            frames = [f for _, f, _ in self._buffer]
            self._pending = self._executor.submit(self.recognizer.predict, frames)
            self._last_submit = now
        return new_result

    def clear(self):
        self._buffer.clear()
        self.latest_result = None

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)
