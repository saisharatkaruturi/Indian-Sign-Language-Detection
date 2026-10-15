"""
Build the one-shot dynamic reference gallery from Sample Videos/ (one clip per sign).

Class names: the 60 classes of dynamic_video_splits.csv plus any extra class folders under
Video_Dataset/Video_Dataset/ (e.g. "Still" -> 61), matched to clip file names
case-insensitively (e.g. "Good morning.mp4" -> "Good Morning").

Usage (from the repository root):
    python src/training/build_oneshot_gallery.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from pathlib import Path

import pandas as pd

from src.inference.oneshot_dynamic import DEFAULT_GALLERY, build_gallery
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("BuildOneShotGallery")


def match_clips_to_classes(clip_dir: Path, class_names) -> dict:
    """Return {class_name: clip_path}; raises if a class has no clip."""
    clips = {p.stem.strip().lower(): p for p in clip_dir.glob("*.mp4")}
    mapping, missing = {}, []
    for cls in sorted(class_names):
        path = clips.get(cls.strip().lower())
        if path is None:
            missing.append(cls)
        else:
            mapping[cls] = path
    if missing:
        raise FileNotFoundError(f"No reference clip in {clip_dir} for classes: {missing}")
    unused = sorted(set(clips) - {c.strip().lower() for c in class_names})
    if unused:
        logger.info(f"Reference clips not in the dynamic class list (ignored): {unused}")
    return mapping


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Build one-shot dynamic sign gallery")
    parser.add_argument("--clips", default="Sample Videos")
    parser.add_argument("--manifest", default="dynamic_video_splits.csv")
    parser.add_argument("--dataset", default="Video_Dataset/Video_Dataset")
    parser.add_argument("--sequence-length", type=int, default=cfg["dynamic"]["sequence_length"])
    parser.add_argument("--out", default=DEFAULT_GALLERY)
    args = parser.parse_args()

    # Union of the manifest's 60 dynamic classes and any local dataset folders (e.g. "Still");
    # folder names win for spelling, so labels match the training data exactly
    classes = {c.lower(): c for c in pd.read_csv(resolve_path(args.manifest))["class_label"].unique()}
    dataset_dir = resolve_path(args.dataset)
    if dataset_dir.is_dir():
        classes.update({p.name.lower(): p.name for p in dataset_dir.iterdir() if p.is_dir()})
    classes = sorted(classes.values())
    mapping = match_clips_to_classes(resolve_path(args.clips), classes)
    build_gallery(mapping, args.sequence_length, resolve_path(args.out))


if __name__ == "__main__":
    main()
