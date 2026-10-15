import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple
from src.models.cnn_model import build_cnn_feature_extractor

def build_static_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 26,
    backbone: str = "MobileNetV2"
) -> tf.keras.Model:
    """
    Builds static sign classifier model.
    Pipeline: Input Image [0, 255] -> CNN Spatial Extractor (backbone-specific scaling) -> Dropout -> Softmax.
    """
    inputs = layers.Input(shape=input_shape)

    extractor = build_cnn_feature_extractor(input_shape=input_shape, backbone=backbone, embedding_dim=256)
    features = extractor(inputs)
    
    x = layers.Dropout(0.3)(features)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="static_classification")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Static_ISL_Model")
    return model
