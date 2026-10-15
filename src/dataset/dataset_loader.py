"""
Static image dataset loading for training/evaluation.

Uses the same tf.keras.utils.image_dataset_from_directory settings as train_cnn.py
(the script that produced model/custom_cnn.tflite): RGB uint8->float32 pixels in
[0, 255], integer labels, class indices from alphabetically sorted folder names.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.utils.logger import get_logger

logger = get_logger("DatasetLoader")

PathLike = Union[str, Path]


def _load_split(directory: PathLike, img_size: Tuple[int, int], batch_size: int, shuffle: bool, seed: int):
    import tensorflow as tf

    return tf.keras.utils.image_dataset_from_directory(
        str(directory),
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=shuffle,
        seed=seed,
    )


def load_static_tensorflow_datasets(
    train_dir: PathLike,
    val_dir: Optional[PathLike] = None,
    test_dir: Optional[PathLike] = None,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[object, Optional[object], Optional[object], List[str]]:
    """
    Load Training (shuffled) and optional Validation/Testing (unshuffled) splits.
    Returns (train_ds, val_ds, test_ds, class_names). Raises FileNotFoundError for a
    missing directory and ValueError if the splits do not share identical class folders,
    which would silently misalign label indices between splits.
    """
    for d in (train_dir, val_dir, test_dir):
        if d is not None and not Path(d).is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {d}")

    train_ds = _load_split(train_dir, img_size, batch_size, shuffle=True, seed=seed)
    class_names = list(train_ds.class_names)

    val_ds = test_ds = None
    if val_dir is not None:
        val_ds = _load_split(val_dir, img_size, batch_size, shuffle=False, seed=seed)
        if list(val_ds.class_names) != class_names:
            raise ValueError(f"Validation classes {val_ds.class_names} differ from training classes {class_names}")
    if test_dir is not None:
        test_ds = _load_split(test_dir, img_size, batch_size, shuffle=False, seed=seed)
        if list(test_ds.class_names) != class_names:
            raise ValueError(f"Testing classes {test_ds.class_names} differ from training classes {class_names}")

    logger.info(f"Loaded static datasets with {len(class_names)} classes: {class_names}")
    return train_ds, val_ds, test_ds, class_names
