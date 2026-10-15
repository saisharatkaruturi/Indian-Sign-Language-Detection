"""
One-shot dynamic recognizer tests.
- Unit tests use a FAKE encoder (mean colour per frame) to test matching/gating logic only.
- The REAL-GALLERY smoke test loads models/dynamic_oneshot/gallery.npz with the real frozen
  MobileNetV2; a reference clip matching itself is a sanity check, not an accuracy result.
"""

import os
import tempfile
import time
import unittest

import numpy as np

from src.inference.oneshot_dynamic import (
    BACKBONE_NAME, DEFAULT_GALLERY, LiveDynamicMatcher, OneShotDynamicRecognizer,
    aligned_similarity, dtw_similarity,
)
from src.utils.config import resolve_path


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class FakeEncoder:
    """MOCK encoder: embeds each frame as its unit-normalised mean RGB plus a constant."""

    def encode(self, frames):
        return _unit(np.array([np.append(f.reshape(-1, 3).mean(0), 1.0) for f in frames], dtype=np.float32))


def _save_gallery(path, embeddings, labels, backbone=BACKBONE_NAME):
    np.savez(path, embeddings=embeddings.astype(np.float32), labels=np.array(labels),
             sources=np.array(labels), source_md5=np.array(labels),
             sequence_length=np.int64(embeddings.shape[1]), backbone=np.array(backbone))


def _clip(rgb, n=12):
    return [np.full((32, 32, 3), rgb, np.uint8) for _ in range(n)]


class TestSimilarity(unittest.TestCase):

    def test_identical_sequences_score_one(self):
        a = _unit(np.random.default_rng(0).normal(size=(16, 8)))
        self.assertAlmostEqual(dtw_similarity(a, a), 1.0, places=6)
        self.assertAlmostEqual(aligned_similarity(a, a), 1.0, places=6)

    def test_dtw_tolerates_time_shift_better_than_alignment(self):
        t = np.linspace(0, 1, 16)
        a = _unit(np.stack([np.sin(3 * t), np.cos(3 * t), t + 0.1], axis=1))
        b = _unit(np.stack([np.sin(3 * t ** 1.8), np.cos(3 * t ** 1.8), t ** 1.8 + 0.1], axis=1))
        self.assertGreater(dtw_similarity(a, b), aligned_similarity(a, b))


class TestGalleryLoading(unittest.TestCase):

    def test_missing_gallery(self):
        with self.assertRaises(FileNotFoundError):
            OneShotDynamicRecognizer("no_such_gallery.npz", strict=True, encoder=FakeEncoder())
        rec = OneShotDynamicRecognizer("no_such_gallery.npz", encoder=FakeEncoder())
        self.assertEqual(rec.predict(_clip((1, 2, 3))), ("Waiting for reference gallery...", 0.0, []))

    def test_wrong_backbone_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "g.npz")
            _save_gallery(path, np.ones((2, 4, 4)), ["A", "B"], backbone="SomethingElse")
            with self.assertRaises(ValueError):
                OneShotDynamicRecognizer(path, strict=True, encoder=FakeEncoder())


class TestMatchingWithFakeEncoder(unittest.TestCase):
    """MOCK-encoder tests of the nearest-reference logic."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        enc = FakeEncoder()
        refs = {"Red": (220, 20, 20), "Green": (20, 220, 20), "Blue": (20, 20, 220)}
        emb = np.stack([enc.encode(_clip(c, 4)) for c in refs.values()])
        cls.path = os.path.join(cls.tmp.name, "g.npz")
        _save_gallery(cls.path, emb, list(refs))
        cls.rec = OneShotDynamicRecognizer(cls.path, strict=True, encoder=enc)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_predict_nearest_reference(self):
        label, sim, top3 = self.rec.predict(_clip((30, 200, 40), n=50))
        self.assertEqual(label, "Green")
        self.assertEqual([l for l, _ in top3][0], "Green")
        self.assertGreaterEqual(top3[0][1], top3[1][1])

    def test_live_matcher_requires_hand_and_returns_result(self):
        m = LiveDynamicMatcher(self.rec, window_seconds=1.0, match_every_seconds=0.0)
        try:
            t0 = 1000.0
            for i in range(20):  # 1 s of frames without a hand: never matched
                m.push(np.full((48, 64, 3), (20, 20, 220), np.uint8), t0 + i * 0.05, hand_present=False)
                m.poll(t0 + i * 0.05)
            self.assertIsNone(m.latest_result)
            for i in range(20, 60):  # hand visible: a match is produced
                m.push(np.full((48, 64, 3), (20, 20, 220), np.uint8), t0 + i * 0.05, hand_present=True)
                m.poll(t0 + i * 0.05)
            deadline = time.time() + 10
            while m.latest_result is None and time.time() < deadline:
                m.poll(t0 + 3.0)
                time.sleep(0.01)
            self.assertIsNotNone(m.latest_result)
            self.assertEqual(m.latest_result[0], "Blue")
        finally:
            m.close()


@unittest.skipUnless(resolve_path(DEFAULT_GALLERY).exists() and resolve_path("Sample Videos/Hello.mp4").exists(),
                     "real gallery or Sample Videos not present")
class TestRealGallery(unittest.TestCase):
    """REAL-GALLERY smoke test (frozen MobileNetV2)."""

    def test_gallery_metadata_and_self_match(self):
        rec = OneShotDynamicRecognizer(strict=True)
        self.assertIn(len(rec.labels), (60, 61))  # 61 when built from all Video_Dataset folders (incl. "Still")
        self.assertEqual(rec.sequence_length, 16)
        self.assertEqual(len(set(rec.labels)), len(rec.labels))
        label, sim, top3 = rec.predict(str(resolve_path("Sample Videos/Hello.mp4")))
        self.assertEqual(label, "Hello")
        self.assertAlmostEqual(sim, 1.0, places=4)
        self.assertEqual(len(top3), 3)


if __name__ == "__main__":
    unittest.main()
