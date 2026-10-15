"""
Evaluate the one-shot dynamic recognizer on the locally available Video_Dataset clips.

Protocol (fixed before evaluation):
- Gallery: one reference clip per class (Sample Videos/), 60 classes.
- Queries: every clip under Video_Dataset/Video_Dataset/ (same recording-level split tags
  as train_dynamic.py, for reference). Nothing is trained on these clips, so all of them are
  unseen queries regardless of their split tag.
- Leakage guard: a query recording (original + tilt variants share one recording id) is
  flagged as a possible copy of its class's reference clip when any variant has the same
  frame count as the reference AND frame-aligned similarity >= DUP_SIM. Flagged recordings
  are excluded from the headline metrics and listed in the report.
- Primary method: DTW similarity at T = gallery sequence length (16). The other method/T
  combinations are reported as ablations only.

Usage (from the repository root):
    python src/training/evaluate_oneshot.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import json
from pathlib import PureWindowsPath

import numpy as np
import pandas as pd

from src.dataset.split_dataset import recording_id_from_path
from src.inference.oneshot_dynamic import (
    DEFAULT_GALLERY, FrameEncoder, OneShotDynamicRecognizer, aligned_similarity, dtw_similarity, embed_sequence,
)
from src.preprocessing.video_preprocessing import extract_frames_from_video
from src.training.build_oneshot_gallery import match_clips_to_classes
from src.training.train_dynamic import collect_local_clips
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger("EvaluateOneShot")

DUP_SIM = 0.97
ABLATION_T = 30


def variant_of(path: str) -> str:
    stem = PureWindowsPath(path).stem
    return "left_tilt" if stem.endswith("_left_tilt") else "right_tilt" if stem.endswith("_right_tilt") else "original"


def main():
    parser = argparse.ArgumentParser(description="Evaluate one-shot dynamic recognizer")
    parser.add_argument("--dataset", default="Video_Dataset/Video_Dataset")
    parser.add_argument("--clips", default="Sample Videos")
    parser.add_argument("--gallery", default=DEFAULT_GALLERY)
    parser.add_argument("--out", default="evaluation_results/dynamic_oneshot")
    args = parser.parse_args()

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = FrameEncoder()
    rec = OneShotDynamicRecognizer(args.gallery, strict=True, encoder=encoder)
    T = rec.sequence_length
    labels = rec.labels

    # Ablation gallery at a second sequence length, built in memory from the same clips
    clip_map = match_clips_to_classes(resolve_path(args.clips), labels)
    # One clip at a time: holding every decoded reference frame at once needs many GB of RAM
    ref_counts, ablation_embs = {}, []
    for c in labels:
        frames = extract_frames_from_video(str(clip_map[c]))
        ref_counts[c] = len(frames)
        ablation_embs.append(embed_sequence(encoder, frames, ABLATION_T))
        del frames
    gallery = {T: rec.embeddings, ABLATION_T: np.stack(ablation_embs)}

    available = collect_local_clips(args.dataset)
    available = available[available["class_label"].isin(labels)].reset_index(drop=True)
    logger.info(f"Queries available locally: {len(available)} clips, {available['class_label'].nunique()} classes")

    rows = []
    for i, r in available.iterrows():
        frames = extract_frames_from_video(str(r["local_path"]))
        q = {t: embed_sequence(encoder, frames, t) for t in gallery}
        cls_idx = labels.index(r["class_label"])
        row = {
            "video_path": r["video_path"], "split_tag": r["split"], "true_label": r["class_label"],
            "variant": variant_of(r["video_path"]), "recording_id": recording_id_from_path(r["video_path"]),
            "n_frames": len(frames),
            "same_frame_count_as_reference": len(frames) == ref_counts[r["class_label"]],
            "aligned_sim_to_own_reference": aligned_similarity(q[T], gallery[T][cls_idx]),
        }
        for t in gallery:
            for method, fn in (("dtw", dtw_similarity), ("aligned", aligned_similarity)):
                sims = np.array([fn(q[t], g) for g in gallery[t]])
                order = np.argsort(sims)[::-1]
                key = f"{method}{t}"
                row[f"pred_{key}"] = labels[order[0]]
                row[f"score_{key}"] = float(sims[order[0]])
                row[f"top3_{key}"] = "|".join(labels[j] for j in order[:3])
                row[f"rank_{key}"] = int(np.where(order == cls_idx)[0][0]) + 1
        rows.append(row)
        if (i + 1) % 25 == 0:
            logger.info(f"Evaluated {i + 1}/{len(available)}")

    df = pd.DataFrame(rows)
    dup = df[df["same_frame_count_as_reference"] & (df["aligned_sim_to_own_reference"] >= DUP_SIM)]
    flagged = sorted(dup["recording_id"].unique())
    df["excluded_possible_reference_copy"] = df["recording_id"].isin(flagged)
    df.to_csv(out_dir / "predictions.csv", index=False)

    clean = df[~df["excluded_possible_reference_copy"]]
    present = sorted(clean["true_label"].unique())
    primary = f"dtw{T}"

    def summarize(d, key):
        correct = d[f"pred_{key}"] == d["true_label"]
        per_class = d.assign(c=correct).groupby("true_label")["c"].mean()
        return {"n": int(len(d)), "top1_accuracy": float(correct.mean()),
                "top3_accuracy": float((d[f"rank_{key}"] <= 3).mean()),
                "macro_recall_present_classes": float(per_class.mean()), "mean_rank": float(d[f"rank_{key}"].mean())}

    metrics = {
        "protocol": "one-shot nearest reference (1 clip/class), frozen ImageNet MobileNetV2 frame embeddings",
        "gallery_classes": len(labels), "chance_top1": 1.0 / len(labels),
        "primary_method": primary,
        "queries_total": int(len(df)), "queries_after_leakage_exclusion": int(len(clean)),
        "flagged_recordings": flagged,
        "classes_with_queries": present,
        "primary": summarize(clean, primary),
        "primary_by_variant": {v: summarize(g, primary) for v, g in clean.groupby("variant")},
        "ablations": {k: summarize(clean, k) for k in (f"aligned{T}", f"dtw{ABLATION_T}", f"aligned{ABLATION_T}")},
        "per_class_recall_primary": clean.assign(c=clean[f"pred_{primary}"] == clean["true_label"])
                                         .groupby("true_label")["c"].mean().round(4).to_dict(),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = pd.crosstab(clean["true_label"], clean[f"pred_{primary}"]).reindex(columns=labels, fill_value=0)
    cm.to_csv(out_dir / "confusion_matrix_primary.csv")

    print(json.dumps({k: metrics[k] for k in ("queries_total", "queries_after_leakage_exclusion",
                                              "flagged_recordings", "primary", "primary_by_variant", "ablations")},
                     indent=2))
    logger.info(f"Wrote predictions.csv, metrics.json, confusion_matrix_primary.csv to {out_dir}")


if __name__ == "__main__":
    main()
