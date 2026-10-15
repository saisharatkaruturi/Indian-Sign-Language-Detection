"""
Static image dataset integrity checks.

Consolidates the logic of the original standalone analysis scripts
(isl_dataset_analysis.py, isl_leakage_check.py) so it can be reused by
src/main.py (--mode inspect) and the unit tests.
"""

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

from PIL import Image, UnidentifiedImageError

from src.utils.logger import get_logger

logger = get_logger("DatasetValidator")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ["Training", "Validation", "Testing"]

PathLike = Union[str, Path]


def _iter_images(root: PathLike):
    """Yield every image file under root (recursive), in a deterministic order."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
            yield path


def hash_file(path: PathLike) -> str:
    """Return the MD5 hex digest of a file's bytes (same scheme as isl_leakage_check.py)."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_corrupted_images(root: PathLike) -> List[Tuple[str, str]]:
    """Return [(path, error_message)] for every image under root that PIL cannot verify."""
    corrupted = []
    for path in _iter_images(root):
        try:
            with Image.open(path) as im:
                im.verify()
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            corrupted.append((str(path), str(e)))
    return corrupted


def find_duplicate_images(root: PathLike) -> Dict[str, List[str]]:
    """Return {md5: [paths]} for byte-identical images that occur more than once under root."""
    hash_map: Dict[str, List[str]] = defaultdict(list)
    for path in _iter_images(root):
        hash_map[hash_file(path)].append(str(path))
    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def find_cross_split_leaks(base_dir: PathLike, splits: List[str] = SPLITS) -> Dict[str, List[Tuple[str, str]]]:
    """Return {md5: [(split, path)]} for images that appear in more than one split folder."""
    base_dir = Path(base_dir)
    hash_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.is_dir():
            logger.warning(f"Split folder not found, skipping: {split_dir}")
            continue
        for path in _iter_images(split_dir):
            hash_map[hash_file(path)].append((split, str(path)))
    return {h: locs for h, locs in hash_map.items() if len({s for s, _ in locs}) > 1}


def count_images_per_class(split_dir: PathLike) -> Dict[str, int]:
    """Return {class_folder_name: image_count} for the immediate class folders of split_dir."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    counts = {}
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        counts[class_dir.name] = sum(
            1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
    return counts


def inspect_dataset_directory(target_dir: PathLike) -> Dict[str, object]:
    """
    Print and return a summary of a class-folder image dataset:
    per-class counts, empty classes, corrupted files and within-directory duplicates.
    """
    target_dir = Path(target_dir)
    counts = count_images_per_class(target_dir)
    corrupted = find_corrupted_images(target_dir)
    duplicates = find_duplicate_images(target_dir)
    empty = [c for c, n in counts.items() if n == 0]

    print("\n==============================================")
    print(f" DATASET INSPECTION: {target_dir}")
    print("==============================================")
    for cls, n in counts.items():
        print(f"  {cls:<12} {n:>6} images")
    print(f"  {'TOTAL':<12} {sum(counts.values()):>6} images across {len(counts)} classes")
    print(f"\n  Empty classes      : {empty if empty else 'none'}")
    print(f"  Corrupted images   : {len(corrupted)}")
    for path, err in corrupted[:10]:
        print(f"    - {path} ({err})")
    print(f"  Duplicate groups   : {len(duplicates)}")
    print("==============================================\n")

    return {
        "class_counts": counts,
        "empty_classes": empty,
        "corrupted": corrupted,
        "duplicates": duplicates,
    }
