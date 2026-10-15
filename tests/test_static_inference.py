"""
Static inference tests.
REAL-MODEL tests load model/custom_cnn.tflite (skipped if the artifact is absent).
They verify loading, shapes, finiteness and label mapping only; synthetic inputs say
nothing about recognition accuracy.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.inference.static_inference import (
    DEFAULT_STATIC_MAPPING, DEFAULT_STATIC_MODEL, StaticInferenceEngine, prepare_static_input,
)
from src.utils.config import resolve_path

HAS_MODEL = resolve_path(DEFAULT_STATIC_MODEL).exists()


class TestStaticPreprocessing(unittest.TestCase):

    def test_bgr_to_rgb_resize_and_range(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[..., 0] = 255  # pure blue in BGR
        batch = prepare_static_input(frame, (224, 224))
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        # Model expects raw [0, 255] RGB: blue ends up in channel 2, not rescaled
        self.assertEqual(float(batch[0, 0, 0, 2]), 255.0)
        self.assertEqual(float(batch[0, 0, 0, 0]), 0.0)

    def test_grayscale_and_bgra_accepted(self):
        self.assertEqual(prepare_static_input(np.zeros((50, 60), np.uint8)).shape, (1, 224, 224, 3))
        self.assertEqual(prepare_static_input(np.zeros((50, 60, 4), np.uint8)).shape, (1, 224, 224, 3))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(TypeError):
            prepare_static_input("not an array")
        with self.assertRaises(ValueError):
            prepare_static_input(np.zeros((10, 10, 2), np.uint8))


class TestStaticEngineErrors(unittest.TestCase):

    def test_missing_model_strict_raises(self):
        with self.assertRaises(FileNotFoundError):
            StaticInferenceEngine(model_path="does_not_exist.tflite", strict=True)

    def test_missing_model_non_strict_reports(self):
        eng = StaticInferenceEngine(model_path="does_not_exist.tflite")
        self.assertFalse(eng.is_loaded)
        self.assertIn("not found", eng.load_error)
        self.assertEqual(eng.predict(np.zeros((10, 10, 3), np.uint8)), ("Waiting for trained model...", 0.0, []))

    def test_corrupt_artifact_raises(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.tflite")
            with open(bad, "wb") as f:
                f.write(b"not a flatbuffer")
            with self.assertRaises((ValueError, RuntimeError)):
                StaticInferenceEngine(model_path=bad, strict=True)

    def test_unsupported_format_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(ValueError):
                StaticInferenceEngine(model_path=path, strict=True)
        finally:
            os.remove(path)


@unittest.skipUnless(HAS_MODEL, f"{DEFAULT_STATIC_MODEL} not present")
class TestRealStaticModel(unittest.TestCase):
    """REAL-MODEL smoke tests against model/custom_cnn.tflite."""

    @classmethod
    def setUpClass(cls):
        cls.engine = StaticInferenceEngine(strict=True)

    def test_model_metadata_matches_mapping(self):
        self.assertTrue(self.engine.is_loaded)
        self.assertEqual(self.engine.input_size, (224, 224))
        self.assertEqual(self.engine.num_classes, 26)
        with open(resolve_path(DEFAULT_STATIC_MAPPING), encoding="utf-8") as f:
            mapping = json.load(f)
        self.assertEqual([mapping[str(i)] for i in range(26)], [chr(ord("A") + i) for i in range(26)])

    def test_output_is_valid_probability_vector(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (300, 260, 3), dtype=np.uint8)
        probs = self.engine.predict_probabilities(frame)
        self.assertEqual(probs.shape, (26,))
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all(probs >= 0))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)

    def test_predict_maps_argmax_to_label(self):
        frame = np.full((224, 224, 3), 128, np.uint8)
        probs = self.engine.predict_probabilities(frame)
        label, conf, top3 = self.engine.predict(frame)
        self.assertEqual(label, self.engine.mapping.get_label(int(np.argmax(probs))))
        self.assertAlmostEqual(conf, float(probs.max()), places=6)
        self.assertEqual(len(top3), 3)
        self.assertGreaterEqual(top3[0][1], top3[1][1])
        self.assertGreaterEqual(top3[1][1], top3[2][1])

    def test_empty_frame_handled(self):
        self.assertEqual(self.engine.predict(np.zeros((0, 0, 3), np.uint8)), ("No Hand", 0.0, []))

    def test_repeated_inference_reuses_loaded_model(self):
        with mock.patch("src.inference.static_inference._make_tflite_interpreter") as factory:
            for _ in range(3):
                self.engine.predict(np.zeros((100, 100, 3), np.uint8))
            factory.assert_not_called()

    def test_mapping_size_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bad_map = os.path.join(d, "map.json")
            with open(bad_map, "w", encoding="utf-8") as f:
                json.dump({"0": "A", "1": "B"}, f)
            with self.assertRaises(ValueError):
                StaticInferenceEngine(mapping_path=bad_map, strict=True)


if __name__ == "__main__":
    unittest.main()
