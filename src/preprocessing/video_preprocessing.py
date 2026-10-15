import os
import cv2
import numpy as np
from typing import Union, List
from src.preprocessing.image_preprocessing import preprocess_static_image, validate_image

def extract_frames_from_video(video_path: str) -> List[np.ndarray]:
    """Read video file using OpenCV and extract frames as RGB numpy arrays."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from video file: {video_path}")
    return frames

def sample_or_pad_sequence(frames: List[np.ndarray], target_len: int = 30) -> List[np.ndarray]:
    """
    Uniformly sample or pad sequence to fixed target_len frames.
    Preserves temporal information without destroying sequence order.
    Long clips: evenly spaced, rounded indices spanning the whole clip (as in
    dynamic_datapreprocessing.uniform_temporal_sample). Short clips: all frames,
    then the last frame repeated.
    """
    total = len(frames)
    if total == 0:
        raise ValueError("Frame list is empty.")
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")

    if total >= target_len:
        # Uniform temporal sampling
        indices = np.round(np.linspace(0, total - 1, target_len)).astype(int)
        return [frames[i] for i in indices]
    else:
        # Padding by extending final frame
        sampled = list(frames)
        last_frame = frames[-1]
        while len(sampled) < target_len:
            sampled.append(last_frame.copy())
        return sampled

def sample_frames_from_video(video_path: str, target_len: int) -> List[np.ndarray]:
    """
    Fast equivalent of sample_or_pad_sequence(extract_frames_from_video(path), target_len):
    frames are decoded with grab() and only the sampled ones are retrieved/converted to RGB,
    so long videos are not held in memory. Uses the container's frame count to pick the
    indices and falls back to the full decode if that count turns out to be inaccurate.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            return sample_or_pad_sequence(extract_frames_from_video(video_path), target_len)
        if n >= target_len:
            wanted = np.round(np.linspace(0, n - 1, target_len)).astype(int)
        else:
            wanted = np.arange(n)
        wanted_set = set(wanted.tolist())
        picked = {}
        for i in range(n):
            if not cap.grab():
                break
            if i in wanted_set:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                picked[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        exact = len(picked) == len(wanted_set) and not cap.grab()
    finally:
        cap.release()
    if not exact:  # container frame count was wrong: use the reference implementation
        return sample_or_pad_sequence(extract_frames_from_video(video_path), target_len)
    frames = [picked[i] for i in wanted]
    return sample_or_pad_sequence(frames, target_len) if n < target_len else frames


def preprocess_video_sequence(
    video_or_frames: Union[str, List[np.ndarray], np.ndarray],
    target_len: int = 30,
    target_size: tuple = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Convert video file path or frame list into model-compatible sequence tensor.
    Returns array of shape (target_len, target_size[0], target_size[1], 3).
    """
    if isinstance(video_or_frames, str):
        raw_frames = extract_frames_from_video(video_or_frames)
    elif isinstance(video_or_frames, list):
        raw_frames = video_or_frames
    elif isinstance(video_or_frames, np.ndarray):
        if video_or_frames.ndim == 4:
            raw_frames = [video_or_frames[i] for i in range(video_or_frames.shape[0])]
        else:
            raise ValueError(f"Expected 4D array for sequence input, got shape {video_or_frames.shape}")
    else:
        raise TypeError(f"Unsupported sequence input type: {type(video_or_frames)}")

    for f in raw_frames:
        validate_image(f)

    sampled = sample_or_pad_sequence(raw_frames, target_len=target_len)

    processed_frames = [
        preprocess_static_image(f, target_size=target_size, normalize=normalize)
        for f in sampled
    ]

    return np.array(processed_frames, dtype=np.float32 if normalize else np.uint8)
