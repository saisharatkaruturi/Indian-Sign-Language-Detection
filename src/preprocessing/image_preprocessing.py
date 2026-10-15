import os
import cv2
import numpy as np
from typing import Union

def validate_image(image: np.ndarray) -> bool:
    """Validate image format, channels, and dimensions."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be a numpy.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[2] not in (1, 3):
        raise ValueError(f"Image array shape invalid: expected (H, W, 3/1), got {image.shape}")
    return True

def preprocess_static_image(
    image: Union[str, np.ndarray],
    target_size: tuple = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess static sign image.
    Pipeline: Load/Format -> Validate -> Resize (224x224) -> RGB -> Normalize [0.0, 1.0].
    """
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path not found: {image}")
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise ValueError(f"Failed to read image at: {image}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        img_rgb = image.copy()
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    validate_image(img_rgb)

    # Resize to target dimension
    resized = cv2.resize(img_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    # Min-max normalization
    if normalize:
        processed = resized.astype(np.float32) / 255.0
    else:
        processed = resized

    return processed
