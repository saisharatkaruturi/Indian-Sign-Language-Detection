import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from src.utils.class_mapping import ClassMappingManager
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger("StaticInferenceEngine")

# The only trained static artifact in the repository. It was exported from the
# train_cnn.py custom CNN (3x Conv2D + Dense 128 + Dense 26 softmax). Its
# Rescaling(1/255) layer is folded into the first convolution, so it expects
# float32 RGB pixels in [0, 255] with shape (1, 224, 224, 3).
DEFAULT_STATIC_MODEL = "model/custom_cnn.tflite"
DEFAULT_STATIC_MAPPING = "models/static/class_mapping.json"


def _make_tflite_interpreter(model_path: str):
    """Create a TFLite interpreter, preferring LiteRT and falling back to tf.lite."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=model_path)


def prepare_static_input(frame_bgr: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Convert an OpenCV BGR/BGRA/grayscale frame to the static model input:
    RGB, resized (bilinear, as tf.keras.utils.image_dataset_from_directory does during
    training) to target_size=(height, width), float32 in [0, 255], with a batch axis.
    """
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError(f"Frame must be a numpy.ndarray, got {type(frame_bgr)}")
    if frame_bgr.ndim == 2:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported frame shape {frame_bgr.shape}; expected (H, W), (H, W, 3) or (H, W, 4)")
    resized = cv2.resize(rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(resized.astype(np.float32), axis=0)


class StaticInferenceEngine:
    """Inference engine for static single-frame ISL recognition (.tflite or .keras artifact)."""

    def __init__(self, model_path: str = DEFAULT_STATIC_MODEL, mapping_path: str = DEFAULT_STATIC_MAPPING,
                 strict: bool = False):
        """
        strict=False keeps the real-time UI usable without a model (is_loaded=False and
        load_error set). strict=True raises on a missing/corrupt/mismatched artifact.
        """
        self.model_path = str(resolve_path(model_path))
        self.mapping_path = str(resolve_path(mapping_path))
        self.mapping: Optional[ClassMappingManager] = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.input_size: Tuple[int, int] = (224, 224)
        self.num_classes = 0
        self._keras_model = None
        self._interpreter = None
        self._input_index = None
        self._output_index = None
        try:
            self._load()
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
            self.load_error = str(e)
            self.is_loaded = False
            if strict:
                raise
            logger.warning(f"Static CNN unavailable: {e}")

    def _load(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Static model file not found: {self.model_path}")
        self.mapping = ClassMappingManager.load_json(self.mapping_path)

        suffix = Path(self.model_path).suffix.lower()
        if suffix == ".tflite":
            self._interpreter = _make_tflite_interpreter(self.model_path)
            self._interpreter.allocate_tensors()
            in_det = self._interpreter.get_input_details()[0]
            out_det = self._interpreter.get_output_details()[0]
            in_shape, out_shape = tuple(in_det["shape"]), tuple(out_det["shape"])
            if in_det["dtype"] != np.float32:
                raise ValueError(f"Unsupported TFLite input dtype {in_det['dtype']} (expected float32)")
            self._input_index, self._output_index = in_det["index"], out_det["index"]
        elif suffix in (".keras", ".h5"):
            import tensorflow as tf
            self._keras_model = tf.keras.models.load_model(self.model_path)
            in_shape = tuple(self._keras_model.input_shape)
            out_shape = tuple(self._keras_model.output_shape)
        else:
            raise ValueError(f"Unsupported static model format '{suffix}' ({self.model_path})")

        if len(in_shape) != 4 or in_shape[3] != 3:
            raise ValueError(f"Static model input shape {in_shape} is not (batch, H, W, 3)")
        self.input_size = (int(in_shape[1]), int(in_shape[2]))
        self.num_classes = int(out_shape[-1])
        if self.num_classes != self.mapping.num_classes():
            raise ValueError(
                f"Static model outputs {self.num_classes} classes but {self.mapping_path} "
                f"maps {self.mapping.num_classes()} labels"
            )
        self.is_loaded = True
        logger.info(f"Static model loaded from {self.model_path} | input {in_shape} -> {self.num_classes} classes")

    def predict_probabilities(self, frame: np.ndarray) -> np.ndarray:
        """Return the softmax probability vector (num_classes,) for one BGR frame."""
        if not self.is_loaded:
            raise RuntimeError(f"Static model is not loaded: {self.load_error}")
        batch = prepare_static_input(frame, self.input_size)
        if self._interpreter is not None:
            self._interpreter.set_tensor(self._input_index, batch)
            self._interpreter.invoke()
            probs = self._interpreter.get_tensor(self._output_index)[0]
        else:
            probs = np.asarray(self._keras_model(batch, training=False))[0]
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape != (self.num_classes,) or not np.all(np.isfinite(probs)):
            raise RuntimeError(f"Invalid static model output (shape {probs.shape})")
        return probs

    def predict(self, frame: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Run static prediction pipeline on a BGR frame (e.g. a hand crop).
        Returns Tuple: (Predicted_Sign_Label, Softmax_Confidence, Top_3_List)
        """
        if not self.is_loaded:
            return "Waiting for trained model...", 0.0, []

        if frame is None or frame.size == 0:
            return "No Hand", 0.0, []

        probs = self.predict_probabilities(frame)
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [(self.mapping.get_label(int(i)), float(probs[i])) for i in top3_indices]
        label, confidence = top3[0]
        return label, confidence, top3
