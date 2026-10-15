import unittest
import collections
import numpy as np
from src.utils.config import load_config
from src.utils.class_mapping import ClassMappingManager
from src.inference.static_inference import StaticInferenceEngine
from src.inference.dynamic_inference import DynamicInferenceEngine
from src.inference.realtime_inference import smooth_predictions

class TestSystemIntegration(unittest.TestCase):

    def test_config_loading(self):
        cfg = load_config("config.yaml")
        self.assertIn("camera", cfg)
        self.assertIn("preprocessing", cfg)
        self.assertIn("model", cfg)

    def test_class_mapping_manager(self):
        labels = ["HELLO", "THANK YOU", "YES", "NO"]
        mgr = ClassMappingManager.from_list(labels)
        self.assertEqual(mgr.num_classes(), 4)
        self.assertEqual(mgr.get_label(0), "HELLO")
        self.assertEqual(mgr.get_idx("YES"), 3)

    def test_inference_engines_fallback(self):
        static_eng = StaticInferenceEngine(model_path="non_existent.keras", mapping_path="non_existent.json")
        self.assertFalse(static_eng.is_loaded)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        label, conf, top3 = static_eng.predict(dummy_frame)
        self.assertEqual(label, "Waiting for trained model...")
        self.assertEqual(conf, 0.0)
        self.assertEqual(top3, [])

        dynamic_eng = DynamicInferenceEngine(model_path="non_existent.keras", mapping_path="non_existent.json")
        self.assertFalse(dynamic_eng.is_loaded)
        dummy_seq = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
        label, conf = dynamic_eng.predict_sequence(dummy_seq)
        self.assertEqual(label, "Waiting for trained model...")
        self.assertEqual(conf, 0.0)

    def test_translator_processes_frame_and_closes(self):
        from src.inference.gesture_translator import ISLGestureTranslator
        translator = ISLGestureTranslator()
        out = translator.process_frame(np.full((480, 640, 3), 40, dtype=np.uint8))
        self.assertEqual(len(out), 6)
        translator.close()
        translator.close()  # idempotent
        self.assertIsNone(translator.recognizer)

    def test_prediction_smoothing(self):
        history = collections.deque(["HELLO", "HELLO", "NO", "HELLO"], maxlen=5)
        smoothed = smooth_predictions(history)
        self.assertEqual(smoothed, "HELLO")

if __name__ == "__main__":
    unittest.main()
