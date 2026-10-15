import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

def build_cnn_feature_extractor(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    backbone: str = "MobileNetV2",
    embedding_dim: int = 512
) -> tf.keras.Model:
    """
    Builds CNN backbone for spatial feature extraction.
    Spatial features capture hand shape, finger arrangement, and gesture position.
    Expects raw RGB pixels in [0, 255] (same contract as model/custom_cnn.tflite) and
    applies the scaling each backbone requires internally.
    """
    inputs = layers.Input(shape=input_shape)

    if backbone.lower() == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False
        # MobileNetV2 ImageNet weights expect pixels in [-1, 1]
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(embedding_dim, activation="relu", name="feature_embedding")(x)
    else:
        # Custom CNN spatial backbone
        x = layers.Rescaling(1.0 / 255.0)(inputs)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(embedding_dim, activation="relu", name="feature_embedding")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="CNN_Feature_Extractor")
