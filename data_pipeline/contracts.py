"""
Data Pipeline & Preprocessing
Data Interface Contracts for Static and Dynamic ISL Recognition Modules.
"""

from typing import Union, List, Tuple
import numpy as np

def validate_image_shape(image: np.ndarray) -> bool:
    """Validate that input image is a valid HWC image array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image as numpy.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[2] not in (1, 3):
        raise ValueError(f"Expected image with shape (H, W, 3) or (H, W, 1), got {image.shape}")
    return True

def validate_sequence_shape(sequence: Union[List[np.ndarray], np.ndarray]) -> bool:
    """Validate that input sequence is a valid T x H x W x C array/list."""
    if isinstance(sequence, list):
        if len(sequence) == 0:
            raise ValueError("Sequence frame list cannot be empty.")
        for idx, img in enumerate(sequence):
            validate_image_shape(img)
    elif isinstance(sequence, np.ndarray):
        if sequence.ndim != 4:
            raise ValueError(f"Expected sequence array with shape (T, H, W, C), got {sequence.shape}")
    else:
        raise TypeError(f"Expected sequence as list of arrays or 4D array, got {type(sequence)}")
    return True
