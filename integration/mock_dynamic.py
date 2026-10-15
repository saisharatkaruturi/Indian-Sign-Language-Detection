"""
Integration Mock for Dynamic Sign Recognition
Contract: run_dynamic(frame_buffer) -> (label, confidence)
"""

import numpy as np
from typing import Union, List, Tuple
from data_pipeline.dynamic_preprocessing import prepare_sequence

DYNAMIC_SAMPLE_CLASSES = [
    "Hello", "Thank you", "Good morning", "Help", "Yes", "No",
    "Please", "Water", "Food", "Name", "What", "Goodbye"
]

class MockDynamicRecognizer:
    def __init__(self):
        self.classes = DYNAMIC_SAMPLE_CLASSES
        print(f"[MockDynamicRecognizer] Initialized with {len(self.classes)} mock dynamic sign classes.")

    def predict(self, frame_buffer: Union[List[np.ndarray], np.ndarray]) -> Tuple[str, float]:
        """
        Run dynamic prediction pipeline mock.
        Args:
            frame_buffer: List of RGB frame arrays or 4D sequence array
        Returns:
            Tuple of (predicted_class_label, confidence_score)
        """
        # Test dynamic preprocessing contract
        seq_tensor = prepare_sequence(frame_buffer, sequence_length=16, target_size=(224, 224), normalize=True)
        assert seq_tensor.shape == (16, 224, 224, 3), f"Invalid sequence tensor shape: {seq_tensor.shape}"

        # Deterministic mock prediction
        idx = int(np.mean(seq_tensor) * 1000) % len(self.classes)
        label = self.classes[idx]
        confidence = 0.92

        return label, confidence

_dynamic_mock_instance = MockDynamicRecognizer()

def run_dynamic(frame_buffer: Union[List[np.ndarray], np.ndarray]) -> Tuple[str, float]:
    """Module-level entry point matching the recognizer contract."""
    return _dynamic_mock_instance.predict(frame_buffer)
