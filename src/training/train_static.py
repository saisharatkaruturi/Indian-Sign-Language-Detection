import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import tensorflow as tf
from src.utils.config import load_config
from src.utils.class_mapping import ClassMappingManager
from src.utils.logger import get_logger
from src.dataset.dataset_loader import load_static_tensorflow_datasets
from src.models.static_model import build_static_model

logger = get_logger("TrainStatic")

def run_static_training(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    train_cfg = cfg.get("training", {})
    prep_cfg = cfg.get("preprocessing", {})
    model_cfg = cfg.get("model", {})

    train_dir = "Dataset/Training"
    val_dir = "Dataset/Validation"
    test_dir = "Dataset/Testing"

    if not os.path.exists(train_dir):
        logger.error(f"Training dataset directory not found at {train_dir}")
        return

    tf.random.set_seed(train_cfg.get("seed", 42))
    batch_size = train_cfg.get("batch_size", 32)
    epochs = train_cfg.get("epochs", 30)
    img_size = (prep_cfg.get("image_height", 224), prep_cfg.get("image_width", 224))

    train_ds, val_ds, test_ds, class_names = load_static_tensorflow_datasets(
        train_dir, val_dir, test_dir, img_size=img_size, batch_size=batch_size
    )

    # Save class mapping
    out_dir = "models/static"
    os.makedirs(out_dir, exist_ok=True)
    mapping_mgr = ClassMappingManager.from_list(class_names)
    mapping_mgr.save_json(os.path.join(out_dir, "class_mapping.json"))

    model = build_static_model(
        input_shape=img_size + (3,),
        num_classes=len(class_names),
        backbone=model_cfg.get("backbone", "MobileNetV2")
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg.get("learning_rate", 0.001)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info("Starting static model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model_path = os.path.join(out_dir, "static_model.keras")
    model.save(model_path)
    logger.info(f"Static model successfully trained and saved to {model_path}")

if __name__ == "__main__":
    run_static_training()
