import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List
from src.preprocessing.video_preprocessing import preprocess_video_sequence
from src.utils.class_mapping import ClassMappingManager
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger("DynamicInferenceEngine")

# No trained dynamic model exists in the repository. src/training/train_dynamic.py only
# builds an untrained architecture. A trained model placed here must accept raw RGB
# frames in [0, 255] shaped (batch, T, H, W, 3), like the src/models builders.
DEFAULT_DYNAMIC_MODEL = "models/dynamic/dynamic_model.keras"
DEFAULT_DYNAMIC_MAPPING = "models/dynamic/class_mapping.json"


class DynamicInferenceEngine:
    """Inference engine for dynamic sequence ISL recognition (Keras CNN + Bi-GRU/LSTM model)."""

    def __init__(self, model_path: str = DEFAULT_DYNAMIC_MODEL, mapping_path: str = DEFAULT_DYNAMIC_MAPPING,
                 strict: bool = False):
        """
        strict=False leaves is_loaded=False (with load_error set) when the artifact is missing
        or invalid; strict=True raises instead.
        """
        self.model_path = str(resolve_path(model_path))
        self.mapping_path = str(resolve_path(mapping_path))
        self.model = None
        self.mapping: Optional[ClassMappingManager] = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.sequence_length: Optional[int] = None
        self.frame_size: Tuple[int, int] = (224, 224)
        self.num_classes = 0
        try:
            self._load()
        except (FileNotFoundError, ValueError, OSError) as e:
            self.load_error = str(e)
            self.is_loaded = False
            if strict:
                raise
            logger.warning(f"Dynamic model unavailable: {e}")

    def _load(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Dynamic model file not found: {self.model_path}")
        self.mapping = ClassMappingManager.load_json(self.mapping_path)

        import tensorflow as tf
        self.model = tf.keras.models.load_model(self.model_path)
        in_shape = tuple(self.model.input_shape)
        out_shape = tuple(self.model.output_shape)
        if len(in_shape) != 5 or in_shape[4] != 3:
            raise ValueError(f"Dynamic model input shape {in_shape} is not (batch, T, H, W, 3)")
        if in_shape[1] is None:
            raise ValueError("Dynamic model has no fixed sequence length; cannot determine frames to sample")
        self.sequence_length = int(in_shape[1])
        self.frame_size = (int(in_shape[2]), int(in_shape[3]))
        self.num_classes = int(out_shape[-1])
        if self.num_classes != self.mapping.num_classes():
            raise ValueError(
                f"Dynamic model outputs {self.num_classes} classes but {self.mapping_path} "
                f"maps {self.mapping.num_classes()} labels"
            )
        self.is_loaded = True
        logger.info(f"Dynamic model loaded from {self.model_path} | input {in_shape} -> {self.num_classes} classes")

    def _probabilities(self, sequence_input: Union[np.ndarray, List[np.ndarray], str]) -> np.ndarray:
        seq_tensor = preprocess_video_sequence(
            sequence_input,
            target_len=self.sequence_length,
            target_size=self.frame_size,
            normalize=False
        ).astype(np.float32)
        batch_seq = np.expand_dims(seq_tensor, axis=0)

        preds = np.asarray(self.model(batch_seq, training=False))[0]
        if preds.shape != (self.num_classes,) or not np.all(np.isfinite(preds)):
            raise RuntimeError(f"Invalid dynamic model output (shape {preds.shape})")
        return preds

    def predict_sequence(self, sequence_input: Union[np.ndarray, List[np.ndarray], str]) -> Tuple[str, float]:
        """
        Run dynamic sequence prediction on a video path or list/array of RGB frames.
        The sequence length and frame size come from the loaded model's input shape.
        Returns Tuple: (Predicted_Sign_Label, Softmax_Confidence)
        """
        if not self.is_loaded:
            return "Waiting for trained model...", 0.0
        preds = self._probabilities(sequence_input)
        max_idx = int(np.argmax(preds))
        return self.mapping.get_label(max_idx), float(preds[max_idx])

    def predict(self, sequence_input: Union[np.ndarray, List[np.ndarray], str]) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Same as predict_sequence plus the top-3 (label, probability) list (live-matcher interface)."""
        if not self.is_loaded:
            return "Waiting for trained model...", 0.0, []
        preds = self._probabilities(sequence_input)
        order = np.argsort(preds)[::-1][:3]
        top3 = [(self.mapping.get_label(int(i)), float(preds[i])) for i in order]
        return top3[0][0], top3[0][1], top3
