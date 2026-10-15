import collections
import numpy as np
from typing import List, Optional
from src.preprocessing.image_preprocessing import preprocess_static_image

class RollingSequenceBuffer:
    """Rolling frame sequence buffer for real-time dynamic gesture recognition."""

    def __init__(self, sequence_length: int = 30, target_size: tuple = (224, 224), normalize: bool = True):
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.normalize = normalize
        self.buffer = collections.deque(maxlen=sequence_length)

    def append(self, frame: np.ndarray):
        """Preprocess single frame and append to rolling buffer."""
        processed = preprocess_static_image(frame, target_size=self.target_size, normalize=self.normalize)
        self.buffer.append(processed)

    def is_full(self) -> bool:
        """Check if buffer has accumulated target sequence length."""
        return len(self.buffer) == self.sequence_length

    def get_sequence_tensor(self) -> Optional[np.ndarray]:
        """
        Return sequence tensor of shape (sequence_length, height, width, 3).
        If buffer is not full, pads by repeating earliest frame.
        """
        if len(self.buffer) == 0:
            return None
        frames = list(self.buffer)
        while len(frames) < self.sequence_length:
            frames.insert(0, frames[0].copy())
        return np.array(frames, dtype=np.float32 if self.normalize else np.uint8)

    def count(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
