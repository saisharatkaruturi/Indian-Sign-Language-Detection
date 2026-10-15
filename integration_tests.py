"""
Integration Test Suite
Verifies repository setup, data pipeline contracts, and mock module compliance.
NOTE: run_static/run_dynamic are MOCKS (integration/mock_*.py) returning hard-coded
confidences (0.95 / 0.92). Passing these tests says nothing about model accuracy.
"""

import unittest
import os
import numpy as np
from data_pipeline.static_preprocessing import preprocess_image
from data_pipeline.dynamic_preprocessing import prepare_sequence
from integration.mock_static import run_static
from integration.mock_dynamic import run_dynamic

class TestIntegrationContracts(unittest.TestCase):

    def setUp(self):
        # Create synthetic test frame (240, 320, 3)
        self.synthetic_frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        # Create synthetic frame buffer (20 frames)
        self.synthetic_buffer = [
            np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
            for _ in range(20)
        ]

    def test_static_preprocessing_contract(self):
        """Test static preprocessing output shape and normalization."""
        tensor = preprocess_image(self.synthetic_frame, target_size=(224, 224), normalize=True)
        self.assertEqual(tensor.shape, (224, 224, 3))
        self.assertGreaterEqual(tensor.min(), 0.0)
        self.assertLessEqual(tensor.max(), 1.0)
        print(" [PASS] Static Preprocessing Contract")

    def test_dynamic_preprocessing_contract(self):
        """Test dynamic sequence preprocessing output shape and sampling."""
        seq_tensor = prepare_sequence(self.synthetic_buffer, sequence_length=16, target_size=(224, 224), normalize=True)
        self.assertEqual(seq_tensor.shape, (16, 224, 224, 3))
        self.assertGreaterEqual(seq_tensor.min(), 0.0)
        self.assertLessEqual(seq_tensor.max(), 1.0)
        print(" [PASS] Dynamic Preprocessing Contract")

    def test_mock_static_contract(self):
        """Test the static recognizer contract run_static(frame)."""
        label, conf = run_static(self.synthetic_frame)
        self.assertIsInstance(label, str)
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        print(f" [PASS] MOCK Static Recognizer contract: Label='{label}', Conf={conf:.2f} (hard-coded)")

    def test_mock_dynamic_contract(self):
        """Test the dynamic recognizer contract run_dynamic(frame_buffer)."""
        label, conf = run_dynamic(self.synthetic_buffer)
        self.assertIsInstance(label, str)
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        print(f" [PASS] MOCK Dynamic Recognizer contract: Label='{label}', Conf={conf:.2f} (hard-coded)")

    def test_actual_video_pipeline(self):
        """Decode a real video file through the MOCK dynamic recognizer (tests decoding only, not recognition)."""
        sample_video_path = os.path.join(os.path.dirname(__file__), "Sample Videos", "Hello.mp4")
        if not os.path.exists(sample_video_path):
            self.skipTest("Sample Videos/Hello.mp4 not found")
        label, conf = run_dynamic(sample_video_path)
        self.assertIsInstance(label, str)
        print(f" [PASS] Real-video decoding via MOCK recognizer ('Hello.mp4'): Label='{label}' (mock, not a prediction)")

if __name__ == "__main__":
    print("\n==============================================")
    print("  RUNNING INTEGRATION TEST SUITE             ")
    print("==============================================\n")
    unittest.main()
