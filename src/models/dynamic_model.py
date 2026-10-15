import tensorflow as tf
from typing import Tuple
from src.models.lstm_model import build_lstm_dynamic_model
from src.models.gru_model import build_gru_dynamic_model
from src.utils.logger import get_logger

logger = get_logger("DynamicModelFactory")

def build_dynamic_model(
    temporal_model_type: str = "GRU",
    sequence_length: int = 30,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 20,
    backbone: str = "MobileNetV2"
) -> tf.keras.Model:
    """
    Factory function for building dynamic models.
    Supports config-driven selection between GRU and LSTM temporal architectures.
    """
    t_type = temporal_model_type.upper()
    logger.info(f"Building dynamic gesture model with temporal architecture: {t_type}")

    if t_type == "LSTM":
        return build_lstm_dynamic_model(
            sequence_length=sequence_length,
            input_shape=input_shape,
            num_classes=num_classes,
            backbone=backbone
        )
    elif t_type == "GRU":
        return build_gru_dynamic_model(
            sequence_length=sequence_length,
            input_shape=input_shape,
            num_classes=num_classes,
            backbone=backbone
        )
    else:
        raise ValueError(f"Unsupported temporal model type '{temporal_model_type}'. Choose 'LSTM' or 'GRU'.")
