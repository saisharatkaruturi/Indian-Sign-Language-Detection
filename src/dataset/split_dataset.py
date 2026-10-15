"""
Stratified train/val/test splitting for manifest DataFrames (e.g. dynamic_video_splits.csv),
plus a check for recording-level leakage across splits.

split_dataframe_stratified reproduces the per-file 70/15/15 procedure used to create
dynamic_video_splits.csv (dynamic_datapreprocessing.py, cell 0).
"""

import re
from pathlib import PureWindowsPath
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataframe_stratified(
    df: pd.DataFrame,
    label_column: str = "class_label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into stratified train/val/test frames, each tagged with a 'split' column."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in DataFrame columns {list(df.columns)}")

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), stratify=df[label_column], random_state=random_state
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_ratio, stratify=temp_df[label_column], random_state=random_state
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return train_df, val_df, test_df


# Augmentation suffixes observed in Video_Dataset file names, e.g.
#   WIN_20231102_12_42_54_Pro.mp4 / ..._Pro_left_tilt.mp4 / ..._Pro_Trim_cropped_right_tilt.mp4
_AUGMENT_SUFFIX = re.compile(r"(_Trim)?(_cropped)?(_(left|right)_tilt)?$")


def recording_id_from_path(video_path: str) -> str:
    """Return '<class>/<base recording>' with augmentation suffixes (tilt/trim/crop) removed."""
    p = PureWindowsPath(video_path)
    stem = _AUGMENT_SUFFIX.sub("", p.stem)
    return f"{p.parent.name}/{stem}"


def split_dataframe_grouped_stratified(
    df: pd.DataFrame,
    label_column: str = "class_label",
    path_column: str = "video_path",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Leakage-free split: whole source recordings (all augmented variants together) are
    assigned to one split, stratified per class. Returns df with a 'split' column.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    df = df.drop(columns=["split"], errors="ignore").copy()
    df["_recording"] = df[path_column].map(recording_id_from_path)
    if (df.groupby("_recording")[label_column].nunique() > 1).any():
        raise ValueError("A recording id maps to more than one class label")

    assignment = {}
    for _, cls_df in df.groupby(label_column, sort=True):
        recordings = pd.Series(sorted(cls_df["_recording"].unique()))
        recordings = recordings.sample(frac=1.0, random_state=random_state).tolist()
        n_val = int(round(len(recordings) * val_ratio))
        n_test = int(round(len(recordings) * test_ratio))
        n_train = len(recordings) - n_val - n_test
        for i, rid in enumerate(recordings):
            assignment[rid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    df["split"] = df["_recording"].map(assignment)
    order = {"train": 0, "val": 1, "test": 2}
    df = df.sort_values(["split", label_column, path_column], key=lambda s: s.map(order) if s.name == "split" else s)
    return df.drop(columns=["_recording"]).reset_index(drop=True)


def find_cross_split_recordings(df: pd.DataFrame, path_column: str = "video_path") -> Dict[str, List[str]]:
    """Return {recording_id: sorted splits} for source recordings whose variants occur in more than one split."""
    groups: Dict[str, set] = {}
    for path, split in zip(df[path_column], df["split"]):
        groups.setdefault(recording_id_from_path(path), set()).add(split)
    return {rid: sorted(s) for rid, s in groups.items() if len(s) > 1}
