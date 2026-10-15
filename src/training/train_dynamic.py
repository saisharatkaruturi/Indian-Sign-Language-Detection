"""
Train the dynamic (video) sign classifier: frozen ImageNet MobileNetV2 frame features ->
2-layer Bidirectional GRU head (src/models/gru_model.py).

Data: every class folder present under Video_Dataset/Video_Dataset/ (including "Still").
Split: recording-level 70/15/15 per class over the clips actually present
(src.dataset.split_dataset.split_dataframe_grouped_stratified), so the original and tilt
variants of one recording never cross splits and every class has train/val/test clips.
The split used is saved next to the evaluation outputs. The trained classes are written to
models/dynamic/class_mapping.json, so the model is labelled with exactly what it learned.

Train/inference parity: frames are prepared bit-identically to
src.preprocessing.video_preprocessing.preprocess_video_sequence(normalize=False), the call
DynamicInferenceEngine makes, and the saved model takes raw [0, 255] RGB frames
(batch, T, 224, 224, 3) end to end.

Usage (from the repository root):
    python src/training/train_dynamic.py
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import json
import random
from pathlib import Path, PureWindowsPath

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from src.dataset.split_dataset import find_cross_split_recordings, split_dataframe_grouped_stratified
from src.preprocessing.image_preprocessing import preprocess_static_image
from src.preprocessing.video_preprocessing import sample_frames_from_video
# TensorFlow is imported inside the functions that need it, so the parallel video-decoding
# worker processes (which re-import this module on Windows) start in seconds.
from src.utils.class_mapping import ClassMappingManager
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("TrainDynamic")

FRAME_SIZE = (224, 224)


def build_frozen_backbone():
    """Raw RGB [0, 255] frame -> 1280-d MobileNetV2 (ImageNet, frozen) average-pooled features."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    inputs = layers.Input(shape=FRAME_SIZE + (3,))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)  # == mobilenet_v2.preprocess_input
    base = tf.keras.applications.MobileNetV2(input_shape=FRAME_SIZE + (3,), include_top=False,
                                             weights="imagenet", pooling="avg")
    base.trainable = False
    outputs = base(x, training=False)
    model = models.Model(inputs, outputs, name="Frozen_MobileNetV2_Features")
    model.trainable = False
    return model


def collect_local_clips(dataset_dir: str = "Video_Dataset/Video_Dataset", seed: int = 42) -> pd.DataFrame:
    """All local .mp4 clips (one class per folder) with a recording-level train/val/test split."""
    root = resolve_path(dataset_dir)
    if not root.is_dir():
        raise SystemExit(f"Dataset folder not found: {root}")
    rows = [{"video_path": str(PureWindowsPath(*p.relative_to(resolve_path(".")).parts)),
             "class_label": p.parent.name, "local_path": p}
            for p in sorted(root.glob("*/*.mp4"))]
    if not rows:
        raise SystemExit(f"No .mp4 clips under {root}")
    df = split_dataframe_grouped_stratified(pd.DataFrame(rows), random_state=seed)
    if find_cross_split_recordings(df):
        raise RuntimeError("Recording-level leakage in the generated split")
    counts = df.groupby(["class_label", "split"]).size().unstack(fill_value=0)
    thin = [c for c in counts.index if min(counts.loc[c].get(s, 0) for s in ("train", "val", "test")) == 0]
    if thin:
        logger.warning(f"Classes without clips in every split (too few recordings), dropped: {thin}")
        df = df[~df["class_label"].isin(thin)]
    return df.reset_index(drop=True)


def _decode_sequence(path: str, seq_len: int) -> np.ndarray:
    """
    (T, 224, 224, 3) uint8, bit-identical to preprocess_video_sequence(path, normalize=False),
    the call DynamicInferenceEngine makes (tests/test_dynamic_pipeline.py checks this), but
    only the sampled frames are converted.
    """
    frames = sample_frames_from_video(path, seq_len)
    return np.array([preprocess_static_image(f, target_size=FRAME_SIZE, normalize=False) for f in frames], dtype=np.uint8)


def extract_features(df: pd.DataFrame, backbone, seq_len: int, cache: Path, workers: int = 1) -> np.ndarray:
    """
    (N, T, 1280) features; cached per video path + T so reruns skip decoding. Videos are
    decoded/sampled in parallel worker processes; the frozen backbone runs in this process.
    """
    cached = {}
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if int(data["sequence_length"]) == seq_len:
            cached = dict(zip([str(p) for p in data["paths"]], data["features"]))
    todo = [(r["video_path"], str(r["local_path"])) for _, r in df.iterrows() if r["video_path"] not in cached]
    logger.info(f"Features: {len(df) - len(todo)} cached, {len(todo)} to extract with {workers} decoder processes")
    if todo:
        keys, paths = [k for k, _ in todo], [p for _, p in todo]
        if workers > 1:
            ex = ProcessPoolExecutor(max_workers=workers)
            sequences = ex.map(partial(_decode_sequence, seq_len=seq_len), paths, chunksize=2)
        else:
            ex = None
            sequences = (_decode_sequence(p, seq_len) for p in paths)
        try:
            for i, (key, seq) in enumerate(zip(keys, sequences), 1):
                cached[key] = np.asarray(backbone(seq.astype(np.float32), training=False), dtype=np.float32)
                if i % 25 == 0 or i == len(todo):
                    logger.info(f"Features: {i}/{len(todo)}")
                if i % 200 == 0:  # checkpoint the cache
                    _save_cache(cache, cached, seq_len)
        finally:
            if ex is not None:
                ex.shutdown(cancel_futures=True)
    _save_cache(cache, cached, seq_len)
    return np.stack([cached[k] for k in df["video_path"]])


def _save_cache(cache: Path, cached: dict, seq_len: int):
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, paths=np.array(list(cached.keys())), features=np.stack(list(cached.values())),
             sequence_length=np.int64(seq_len))


def run_dynamic_training(config_path: str = "config.yaml", dataset_dir: str = "Video_Dataset/Video_Dataset",
                         out_dir: str = "models/dynamic", eval_dir: str = "evaluation_results/dynamic_trained",
                         cache: str = "features/dynamic_mobilenetv2_features.npz", epochs: int = 150,
                         workers: int = 1, features_only: bool = False):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from src.models.gru_model import build_gru_temporal_head

    cfg = load_config(config_path)
    train_cfg = cfg.get("training", {})
    seq_len = int(cfg.get("dynamic", {}).get("sequence_length", 16))
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    df = collect_local_clips(dataset_dir, seed)
    classes = sorted(df["class_label"].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = df["class_label"].map(label_to_idx).to_numpy()
    logger.info(f"Training CNN(frozen MobileNetV2) + Bi-GRU on {len(classes)} classes, T={seq_len}: {classes}")
    logger.info("Clips per split: " + json.dumps(df["split"].value_counts().to_dict()))

    backbone = build_frozen_backbone()
    X = extract_features(df, backbone, seq_len, resolve_path(cache), workers=workers)
    if features_only:
        logger.info(f"Features cached for {len(df)} clips; stopping (--features-only).")
        return None
    idx = {s: np.where(df["split"].to_numpy() == s)[0] for s in ("train", "val", "test")}

    head = build_gru_temporal_head(seq_len, X.shape[-1], len(classes))
    head.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(train_cfg.get("learning_rate", 0.001))),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = head.fit(
        X[idx["train"]], y[idx["train"]], validation_data=(X[idx["val"]], y[idx["val"]]),
        epochs=epochs, batch_size=int(train_cfg.get("batch_size", 32)), verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
    )

    # End-to-end inference model: raw frames -> frozen backbone per frame -> trained Bi-GRU head
    frames_in = layers.Input(shape=(seq_len,) + FRAME_SIZE + (3,))
    full = models.Model(frames_in, head(layers.TimeDistributed(backbone)(frames_in)), name="Dynamic_ISL_MobileNetV2_BiGRU")
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full.save(out / "dynamic_model.keras")
    ClassMappingManager({i: c for c, i in label_to_idx.items()}).save_json(str(out / "class_mapping.json"))

    # Held-out evaluation (test recordings never seen in training or model selection)
    ev = resolve_path(eval_dir)
    ev.mkdir(parents=True, exist_ok=True)
    df[["video_path", "class_label", "split"]].to_csv(ev / "split_used.csv", index=False)
    probs = head.predict(X[idx["test"]], verbose=0)
    pred = probs.argmax(axis=1)
    test_df = df.iloc[idx["test"]].assign(
        true_label=lambda d: d["class_label"], pred_label=[classes[p] for p in pred],
        confidence=probs.max(axis=1))[["video_path", "true_label", "pred_label", "confidence"]]
    test_df.to_csv(ev / "predictions_test.csv", index=False)
    pd.DataFrame(history.history).to_csv(ev / "history.csv", index_label="epoch")
    cm = pd.crosstab(test_df["true_label"], test_df["pred_label"]).reindex(index=classes, columns=classes, fill_value=0)
    cm.to_csv(ev / "confusion_matrix_test.csv")

    from sklearn.metrics import f1_score
    correct = test_df["true_label"] == test_df["pred_label"]
    best_epoch = int(np.argmin(history.history["val_loss"]))
    metrics = {
        "model": "frozen ImageNet MobileNetV2 (per frame) + 2-layer Bi-GRU head",
        "sequence_length": seq_len, "classes": classes, "num_classes": len(classes),
        "chance_accuracy": 1.0 / len(classes),
        "clips": {s: int(len(v)) for s, v in idx.items()},
        "split": "recording-level 70/15/15 per class over locally available clips (split_used.csv)",
        "best_epoch": best_epoch + 1, "epochs_run": len(history.history["loss"]),
        "train_accuracy_at_best": float(history.history["accuracy"][best_epoch]),
        "val_accuracy_at_best": float(history.history["val_accuracy"][best_epoch]),
        "test_accuracy": float(correct.mean()),
        "test_macro_f1": float(f1_score(test_df["true_label"], test_df["pred_label"], average="macro")),
        "test_per_class_recall": test_df.assign(c=correct).groupby("true_label")["c"].mean().round(4).to_dict(),
    }
    with open(ev / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Test metrics: " + json.dumps({k: metrics[k] for k in ("clips", "test_accuracy", "test_macro_f1")}))
    logger.info(f"Saved model to {out / 'dynamic_model.keras'} and evaluation to {ev}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the dynamic ISL classifier")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--workers", type=int, default=1, help="parallel video-decoding processes (1 = in-process)")
    parser.add_argument("--features-only", action="store_true", help="only extract/cache features")
    args = parser.parse_args()
    run_dynamic_training(epochs=args.epochs, workers=args.workers, features_only=args.features_only)
