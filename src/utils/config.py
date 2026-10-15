import copy
from pathlib import Path
from typing import Any, Dict, Union
from src.utils.logger import get_logger

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = get_logger("ConfigLoader")

# Repository root (the folder containing src/, config.yaml and model/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Kept in sync with config.yaml so a missing file does not silently change behaviour
DEFAULT_CONFIG = {
    "camera": {"index": 0, "width": 640, "height": 480, "fps": 30},
    "preprocessing": {"image_width": 224, "image_height": 224, "channels": 3, "normalization": True},
    "dynamic": {"sequence_length": 16, "inference_interval": 3,
                "live_method": "trained", "oneshot_gallery": "models/dynamic_oneshot/gallery.npz",
                "live_window_seconds": 4.0, "live_match_every_seconds": 1.0},
    "model": {"backbone": "CustomCNN", "temporal_model": "GRU"},
    "inference": {"confidence_threshold": 0.40, "smoothing": True, "smoothing_window_size": 5},
    "training": {"batch_size": 32, "epochs": 30, "learning_rate": 0.001, "seed": 42}
}


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a relative path against the CWD if it exists there, otherwise against PROJECT_ROOT."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return PROJECT_ROOT / p


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file (merged over defaults) or returns defaults."""
    defaults = copy.deepcopy(DEFAULT_CONFIG)
    path = resolve_path(config_path)
    if not HAS_YAML:
        logger.warning("PyYAML not installed. Falling back to default configuration.")
        return defaults
    if not path.exists():
        logger.warning(f"Config file not found at {path}. Falling back to default settings.")
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.error(f"Error reading config file {path}: {e}. Falling back to default settings.")
        return defaults
    logger.info(f"Loaded configuration from {path}")
    for section, values in cfg.items():
        if isinstance(values, dict) and isinstance(defaults.get(section), dict):
            defaults[section].update(values)
        else:
            defaults[section] = values
    return defaults
