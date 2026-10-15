"""
Unit tests for src.dataset package:
- Dataset loader
- Integrity validator
- Stratified splitter
"""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np
from PIL import Image

from src.dataset.split_dataset import (
    split_dataframe_stratified, split_dataframe_grouped_stratified,
    recording_id_from_path, find_cross_split_recordings,
)
from src.utils.config import resolve_path
from src.dataset.dataset_validator import hash_file, find_corrupted_images, find_duplicate_images
from src.dataset.dataset_loader import load_static_tensorflow_datasets


class TestDatasetPackage(unittest.TestCase):

    def test_stratified_split(self):
        # Create a synthetic dataset with 100 rows across 5 classes
        data = {
            "video_path": [f"video_{i}.mp4" for i in range(100)],
            "class_label": [f"Class_{i % 5}" for i in range(100)]
        }
        df = pd.DataFrame(data)

        train_df, val_df, test_df = split_dataframe_stratified(
            df,
            label_column="class_label",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )

        self.assertEqual(len(train_df), 70)
        self.assertEqual(len(val_df), 15)
        self.assertEqual(len(test_df), 15)
        self.assertEqual(set(train_df["split"]), {"train"})
        self.assertEqual(set(val_df["split"]), {"val"})
        self.assertEqual(set(test_df["split"]), {"test"})

    def test_dataset_validator_integrity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid image
            img_path = os.path.join(temp_dir, "valid.jpg")
            img = Image.new("RGB", (32, 32), color="red")
            img.save(img_path)

            # Create an identical duplicate
            dup_path = os.path.join(temp_dir, "dup.jpg")
            img.save(dup_path)

            # Create a corrupt image file
            bad_path = os.path.join(temp_dir, "bad.jpg")
            with open(bad_path, "wb") as f:
                f.write(b"not an image data")

            # Check hashing
            h1 = hash_file(img_path)
            h2 = hash_file(dup_path)
            self.assertEqual(h1, h2)

            # Check duplicate detection
            dups = find_duplicate_images(temp_dir)
            self.assertIn(h1, dups)
            self.assertEqual(len(dups[h1]), 2)

            # Check corrupt detection
            corrupt = find_corrupted_images(temp_dir)
            self.assertEqual(len(corrupt), 1)
            self.assertTrue(corrupt[0][0].endswith("bad.jpg"))

    def test_dataset_loader_missing_dir_error(self):
        with self.assertRaises(FileNotFoundError):
            load_static_tensorflow_datasets("non_existent_directory_12345")

    def test_recording_id_strips_augmentation_suffixes(self):
        base = r"Video_Dataset\Video_Dataset\Bear\WIN_20231102_12_42_54_Pro"
        ids = {recording_id_from_path(base + suffix + ".mp4")
               for suffix in ("", "_left_tilt", "_right_tilt", "_Trim_cropped", "_Trim_cropped_left_tilt")}
        self.assertEqual(ids, {"Bear/WIN_20231102_12_42_54_Pro"})

    def test_cross_split_recording_detection(self):
        df = pd.DataFrame({
            "video_path": [r"A\x_Pro.mp4", r"A\x_Pro_left_tilt.mp4", r"A\y_Pro.mp4"],
            "split": ["train", "test", "train"],
        })
        self.assertEqual(find_cross_split_recordings(df), {"A/x_Pro": ["test", "train"]})

    def test_grouped_split_keeps_recordings_together(self):
        rows = []
        for c in range(3):
            for r in range(20):
                for suffix in ("", "_left_tilt", "_right_tilt"):
                    rows.append({"video_path": rf"V\C{c}\WIN_{r:02d}_Pro{suffix}.mp4", "class_label": f"C{c}"})
        out = split_dataframe_grouped_stratified(pd.DataFrame(rows), random_state=42)
        self.assertEqual(find_cross_split_recordings(out), {})
        self.assertEqual(out["split"].value_counts().to_dict(), {"train": 126, "val": 27, "test": 27})
        again = split_dataframe_grouped_stratified(pd.DataFrame(rows), random_state=42)
        self.assertTrue(out.equals(again))


class TestDynamicManifestIntegrity(unittest.TestCase):
    """DATA-INTEGRITY test on the committed dynamic_video_splits.csv."""

    def test_no_recording_leaks_across_splits(self):
        manifest = resolve_path("dynamic_video_splits.csv")
        if not manifest.exists():
            self.skipTest("dynamic_video_splits.csv not present")
        df = pd.read_csv(manifest)
        self.assertEqual(find_cross_split_recordings(df), {})
        self.assertEqual(len(df), 3600)
        self.assertEqual(df["split"].value_counts().to_dict(), {"train": 2520, "val": 540, "test": 540})


if __name__ == "__main__":
    unittest.main()
