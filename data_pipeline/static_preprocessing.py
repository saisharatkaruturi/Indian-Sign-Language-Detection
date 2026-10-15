"""
Static Image Preprocessing Pipeline
Provides reusable preprocessing for static ISL recognition.
Contract: preprocess_image(image) -> image_tensor
"""

import os
import cv2
import numpy as np
from typing import Union
from data_pipeline.contracts import validate_image_shape

def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk in BGR/RGB format."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image at: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(
    image: Union[str, np.ndarray],
    target_size: tuple = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess static sign image.
    Args:
        image: File path or numpy image array (H, W, 3)
        target_size: (height, width) tuple (default: 224, 224)
        normalize: Scale pixels to [0, 1] (default: True)
    Returns:
        Processed image tensor of shape (224, 224, 3)
    """
    if isinstance(image, str):
        img_arr = load_image(image)
    elif isinstance(image, np.ndarray):
        img_arr = image.copy()
        if img_arr.ndim == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    validate_image_shape(img_arr)
    
    # Resize to target dimension
    resized = cv2.resize(img_arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    # Normalize to [0.0, 1.0]
    if normalize:
        processed = resized.astype(np.float32) / 255.0
    else:
        processed = resized

    return processed
