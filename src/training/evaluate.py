import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from src.utils.logger import get_logger

logger = get_logger("EvaluateModel")

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list = None):
    """Computes and prints evaluation metrics: Accuracy, Precision, Recall, F1, and Confusion Matrix."""
    acc = np.mean(y_true == y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n==============================================")
    print("           MODEL EVALUATION REPORT            ")
    print("==============================================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    if class_names:
        print("\n--- Detailed Classification Report ---")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("\n--- Confusion Matrix ---")
    print(cm)
    print("==============================================\n")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "confusion_matrix": cm}

if __name__ == "__main__":
    logger.info("Evaluation module ready.")
