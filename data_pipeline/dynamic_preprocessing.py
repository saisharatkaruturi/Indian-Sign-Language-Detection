"""
Dynamic Video Preprocessing Pipeline
Provides reusable frame sequence sampling and preprocessing for dynamic ISL recognition.
Contract: prepare_sequence(video_or_frames) -> sequence_tensor
"""

import os
import cv2
import numpy as np
from typing import Union, List
from data_pipeline.contracts import validate_sequence_shape
from data_pipeline.static_preprocessing import preprocess_image

def extract_frames_from_video(video_path: str) -> List[np.ndarray]:
    """Extract all frames from a video file as RGB numpy arrays."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No valid frames extracted from video: {video_path}")
    return frames

def sample_uniform_sequence(frames: List[np.ndarray], target_len: int = 16) -> List[np.ndarray]:
    """Sample target_len frames uniformly across the frame list."""
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError("Cannot sample from an empty frame list.")

    if total_frames <= target_len:
        # Pad by repeating the last frame if video is shorter than target_len
        indices = np.linspace(0, total_frames - 1, total_frames, dtype=int).tolist()
        sampled = [frames[i] for i in indices]
        while len(sampled) < target_len:
            sampled.append(frames[-1].copy())
    else:
        indices = np.linspace(0, total_frames - 1, target_len, dtype=int)
        sampled = [frames[i] for i in indices]

    return sampled

def prepare_sequence(
    video_or_frames: Union[str, List[np.ndarray], np.ndarray],
    sequence_length: int = 16,
    target_size: tuple = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Convert video path or frame list into a (T, H, W, C) preprocessed sequence tensor.
    Args:
        video_or_frames: Path to video file, or list/array of RGB frames
        sequence_length: Number of uniform frames in output sequence (default: 16)
        target_size: (height, width) frame dimension (default: 224, 224)
        normalize: Scale pixel values to [0.0, 1.0] (default: True)
    Returns:
        Numpy array of shape (sequence_length, target_size[0], target_size[1], 3)
    """
    if isinstance(video_or_frames, str):
        raw_frames = extract_frames_from_video(video_or_frames)
    elif isinstance(video_or_frames, list):
        raw_frames = video_or_frames
    elif isinstance(video_or_frames, np.ndarray):
        if video_or_frames.ndim == 4:
            raw_frames = [video_or_frames[i] for i in range(video_or_frames.shape[0])]
        else:
            raise ValueError(f"4D array expected for sequence array, got shape {video_or_frames.shape}")
    else:
        raise TypeError(f"Unsupported sequence input type: {type(video_or_frames)}")

    validate_sequence_shape(raw_frames)

    # Uniform temporal sampling
    sampled_frames = sample_uniform_sequence(raw_frames, target_len=sequence_length)

    # Spatial preprocessing per frame
    processed_sequence = [
        preprocess_image(f, target_size=target_size, normalize=normalize)
        for f in sampled_frames
    ]

    return np.array(processed_sequence, dtype=np.float32 if normalize else np.uint8)
