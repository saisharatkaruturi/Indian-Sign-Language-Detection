"""
REAL-MODEL tests for the trained dynamic classifier (models/dynamic/dynamic_model.keras).
Skipped when the model, its evaluation outputs, or the test clips are not present locally.
These check loading, shapes, label mapping and train/inference parity; accuracy itself is
reported in evaluation_results/dynamic_trained/metrics.json, not asserted here.
"""

import json
import unittest
from pathlib import PureWindowsPath

import numpy as np
import pandas as pd

from src.inference.dynamic_inference import DEFAULT_DYNAMIC_MAPPING, DEFAULT_DYNAMIC_MODEL, DynamicInferenceEngine
from src.utils.config import resolve_path

PRED_CSV = resolve_path("evaluation_results/dynamic_trained/predictions_test.csv")
HAVE_MODEL = resolve_path(DEFAULT_DYNAMIC_MODEL).exists() and PRED_CSV.exists()


@unittest.skipUnless(HAVE_MODEL, "trained dynamic model / evaluation outputs not present")
class TestTrainedDynamicModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = DynamicInferenceEngine(strict=True)
        cls.preds = pd.read_csv(PRED_CSV)
        cls.preds["local"] = cls.preds["video_path"].map(lambda p: resolve_path(PureWindowsPath(p).as_posix()))
        cls.preds = cls.preds[cls.preds["local"].map(lambda p: p.exists())]
        if cls.preds.empty:
            raise unittest.SkipTest("held-out test clips not present locally")

    def test_model_contract_matches_mapping_and_config(self):
        with open(resolve_path(DEFAULT_DYNAMIC_MAPPING), encoding="utf-8") as f:
            mapping = json.load(f)
        self.assertEqual(self.engine.num_classes, len(mapping))
        self.assertEqual(self.engine.sequence_length, 16)
        self.assertEqual(self.engine.frame_size, (224, 224))
        self.assertEqual(sorted(mapping.values()), [mapping[str(i)] for i in range(len(mapping))])

    def test_engine_reproduces_training_time_prediction(self):
        row = self.preds.iloc[0]
        label, conf = self.engine.predict_sequence(str(row["local"]))
        self.assertEqual(label, row["pred_label"])
        self.assertAlmostEqual(conf, float(row["confidence"]), places=3)

    def test_output_is_probability(self):
        row = self.preds.iloc[-1]
        label, conf = self.engine.predict_sequence(str(row["local"]))
        self.assertIn(label, self.engine.mapping.idx_to_label.values())
        self.assertTrue(0.0 <= conf <= 1.0 and np.isfinite(conf))


if __name__ == "__main__":
    unittest.main()
