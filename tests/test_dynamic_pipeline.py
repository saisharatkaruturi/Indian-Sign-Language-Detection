"""
Dynamic pipeline tests: video decoding, temporal sampling, and engine error handling.
No trained dynamic model exists in the repository, so real-model dynamic inference
cannot be tested here.
"""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from src.inference.dynamic_inference import DynamicInferenceEngine
from src.preprocessing.video_preprocessing import (
    extract_frames_from_video, preprocess_video_sequence, sample_or_pad_sequence,
)


def _write_video(path, n_frames, size=(64, 48)):
    """Write an MJPG .avi whose frame i is filled with gray level 6*i (encodes frame order, n_frames <= 42)."""
    if n_frames > 42:
        raise ValueError("gray-level encoding supports at most 42 frames")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV cannot write MJPG video in this environment")
    try:
        for i in range(n_frames):
            writer.write(np.full((size[1], size[0], 3), 6 * i, np.uint8))
    finally:
        writer.release()


class TestSampling(unittest.TestCase):

    def _frames(self, n):
        return [np.full((4, 4, 3), i, np.uint8) for i in range(n)]

    def test_long_clip_uniform_spans_whole_clip_in_order(self):
        sampled = sample_or_pad_sequence(self._frames(100), target_len=16)
        ids = [int(f[0, 0, 0]) for f in sampled]
        self.assertEqual(len(ids), 16)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[-1], 99)
        self.assertEqual(ids, sorted(ids))

    def test_short_clip_padded_with_last_frame(self):
        ids = [int(f[0, 0, 0]) for f in sample_or_pad_sequence(self._frames(5), target_len=8)]
        self.assertEqual(ids, [0, 1, 2, 3, 4, 4, 4, 4])

    def test_empty_and_invalid_length_rejected(self):
        with self.assertRaises(ValueError):
            sample_or_pad_sequence([], target_len=8)
        with self.assertRaises(ValueError):
            sample_or_pad_sequence(self._frames(3), target_len=0)


class TestVideoDecoding(unittest.TestCase):

    def test_decode_and_sequence_preserves_order(self):
        with tempfile.TemporaryDirectory(prefix="dir with spaces ") as d:
            path = os.path.join(d, "clip.avi")
            _write_video(path, 20)
            frames = extract_frames_from_video(path)
            self.assertEqual(len(frames), 20)
            seq = preprocess_video_sequence(path, target_len=10, target_size=(32, 32), normalize=False)
            self.assertEqual(seq.shape, (10, 32, 32, 3))
            means = [float(f.mean()) for f in seq]
            self.assertEqual(means, sorted(means))

    def test_fast_sampler_matches_reference_path(self):
        from src.preprocessing.video_preprocessing import sample_frames_from_video
        with tempfile.TemporaryDirectory() as d:
            for n in (40, 7):  # long clip (uniform sampling) and short clip (last-frame padding)
                path = os.path.join(d, f"clip{n}.avi")
                _write_video(path, n)
                ref = sample_or_pad_sequence(extract_frames_from_video(path), target_len=16)
                fast = sample_frames_from_video(path, 16)
                self.assertEqual(len(fast), 16)
                for a, b in zip(ref, fast):
                    self.assertTrue(np.array_equal(a, b))

    def test_missing_video(self):
        with self.assertRaises(FileNotFoundError):
            extract_frames_from_video("no_such_video.mp4")

    def test_corrupt_video(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "corrupt.mp4")
            with open(path, "wb") as f:
                f.write(b"\x00garbage" * 100)
            with self.assertRaises(ValueError):
                extract_frames_from_video(path)


class TestDynamicEngine(unittest.TestCase):

    def test_missing_model_strict_raises(self):
        with self.assertRaises(FileNotFoundError):
            DynamicInferenceEngine(model_path="no_such_model.keras", strict=True)

    def test_missing_model_non_strict(self):
        eng = DynamicInferenceEngine(model_path="no_such_model.keras")
        self.assertFalse(eng.is_loaded)
        self.assertEqual(eng.predict_sequence([np.zeros((8, 8, 3), np.uint8)]), ("Waiting for trained model...", 0.0))

    def test_missing_mapping_strict_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            model_path = f.name
        try:
            with self.assertRaises(FileNotFoundError):
                DynamicInferenceEngine(model_path=model_path, mapping_path="no_such_map.json", strict=True)
        finally:
            os.remove(model_path)


class TestDynamicEngineWithArchitecture(unittest.TestCase):
    """
    Uses an UNTRAINED tiny model (random weights) purely to exercise the engine's shape
    handling: sequence length taken from the model, output/mapping validation. Not a
    recognition test.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not installed")
        from src.models.dynamic_model import build_dynamic_model
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_path = os.path.join(cls.tmp.name, "untrained.keras")
        build_dynamic_model("GRU", sequence_length=6, input_shape=(32, 32, 3), num_classes=3, backbone="Custom").save(cls.model_path)
        cls.map_path = os.path.join(cls.tmp.name, "map.json")
        with open(cls.map_path, "w", encoding="utf-8") as f:
            json.dump({"0": "Hello", "1": "Thank you", "2": "Yes"}, f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sequence_length_and_output_from_model(self):
        eng = DynamicInferenceEngine(self.model_path, self.map_path, strict=True)
        self.assertEqual(eng.sequence_length, 6)
        self.assertEqual(eng.frame_size, (32, 32))
        frames = [np.random.randint(0, 256, (48, 64, 3), np.uint8) for _ in range(20)]
        label, conf = eng.predict_sequence(frames)
        self.assertIn(label, {"Hello", "Thank you", "Yes"})
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_mapping_mismatch_rejected(self):
        bad = os.path.join(self.tmp.name, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump({"0": "Hello"}, f)
        with self.assertRaises(ValueError):
            DynamicInferenceEngine(self.model_path, bad, strict=True)


if __name__ == "__main__":
    unittest.main()
