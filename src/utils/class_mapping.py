import os
import json
from typing import Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger("ClassMapping")

class ClassMappingManager:
    """Manages class ID <-> Label mappings for static and dynamic models."""
    
    def __init__(self, mapping_dict: Optional[Dict[int, str]] = None):
        self.idx_to_label: Dict[int, str] = mapping_dict if mapping_dict else {}
        self.label_to_idx: Dict[str, int] = {v: int(k) for k, v in self.idx_to_label.items()}

    @classmethod
    def from_list(cls, label_list: List[str]):
        mapping = {idx: label for idx, label in enumerate(sorted(label_list))}
        return cls(mapping)

    def save_json(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.idx_to_label.items()}, f, indent=4)
        logger.info(f"Saved class mapping ({len(self.idx_to_label)} classes) to {filepath}")

    @classmethod
    def load_json(cls, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Class mapping file not found at {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = {int(k): v for k, v in data.items()}
        logger.info(f"Loaded class mapping ({len(mapping)} classes) from {filepath}")
        return cls(mapping)

    def get_label(self, idx: int) -> str:
        return self.idx_to_label.get(idx, "Unknown")

    def get_idx(self, label: str) -> int:
        return self.label_to_idx.get(label, -1)

    def num_classes(self) -> int:
        return len(self.idx_to_label)
