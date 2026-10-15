"""
Integration Mock for Static Sign Recognition
Contract: run_static(frame) -> (label, confidence)
"""

import numpy as np
from typing import Tuple
from data_pipeline.static_preprocessing import preprocess_image

STATIC_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

class MockStaticRecognizer:
    def __init__(self):
        self.classes = STATIC_CLASSES
        print(f"[MockStaticRecognizer] Initialized with {len(self.classes)} classes (A-Z).")

    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Run static prediction pipeline mock.
        Args:
            frame: Input RGB/BGR image array
        Returns:
            Tuple of (predicted_class_label, confidence_score)
        """
        # Test preprocessing contract
        processed_tensor = preprocess_image(frame, target_size=(224, 224), normalize=True)
        assert processed_tensor.shape == (224, 224, 3), f"Invalid tensor shape: {processed_tensor.shape}"

        # Deterministic mock prediction based on mean pixel brightness
        idx = int(np.mean(processed_tensor) * 1000) % len(self.classes)
        label = self.classes[idx]
        confidence = 0.95

        return label, confidence

_static_mock_instance = MockStaticRecognizer()

def run_static(frame: np.ndarray) -> Tuple[str, float]:
    """Module-level entry point matching the recognizer contract."""
    return _static_mock_instance.predict(frame)
