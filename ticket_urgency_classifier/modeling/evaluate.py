"""Evaluates the trained model using per-class threshold-based predictions."""

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import typer

from ticket_urgency_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main():
    """Evaluate the trained model using per-class threshold-based predictions."""
    logger.info("Starting model evaluation with threshold-based predictions...")

    test_file = PROCESSED_DATA_DIR / "test_features.csv"
    if not test_file.exists():
        logger.error("Processed test data not found. Run data preparation first.")
        return

    df_test = pd.read_csv(test_file)
    logger.info(f"Loaded processed test data. Shape: {df_test.shape}")

    y_test = df_test["priority"]
    X_test = df_test.drop(columns=["priority"])
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    model_file = MODELS_DIR / "best_rf_model.joblib"
    encoder_file = MODELS_DIR / "label_encoder.joblib"
    threshold_file = MODELS_DIR / "best_threshold.joblib"

    if not model_file.exists() or not encoder_file.exists():
        logger.error("Model or encoder not found. Run model training first.")
        return

    model = joblib.load(model_file)
    le = joblib.load(encoder_file)

    if not threshold_file.exists():
        logger.warning("Threshold file not found, using argmax predictions.")
        best_thresholds = {}
    else:
        best_thresholds = joblib.load(threshold_file)
    logger.info("Model, label encoder, and thresholds loaded successfully.")

    # Get probabilities
    y_proba = model.predict_proba(X_test)

    # Apply per-class thresholds
    default_class = "low"
    default_idx = list(le.classes_).index(default_class) if default_class in le.classes_ else 0

    if isinstance(best_thresholds, dict) and best_thresholds:
        y_pred_threshold = np.full(len(X_test), default_idx, dtype=int)
        for class_name, thresh_val in best_thresholds.items():
            if class_name not in le.classes_:
                continue
            class_idx = list(le.classes_).index(class_name)
            mask = y_proba[:, class_idx] >= thresh_val
            y_pred_threshold[mask] = class_idx
    elif isinstance(best_thresholds, (int, float)):
        # Scalar threshold: predict 'low' if low prob >= threshold, else argmax of others
        y_pred_threshold = np.full(len(X_test), default_idx, dtype=int)
        low_mask = y_proba[:, default_idx] >= best_thresholds
        y_pred_threshold[low_mask] = default_idx
        not_low_mask = ~low_mask
        temp_proba = y_proba[not_low_mask].copy()
        temp_proba[:, default_idx] = 0
        if temp_proba.shape[0] > 0:
            remaining_preds = np.argmax(temp_proba, axis=1)
            y_pred_threshold[not_low_mask] = remaining_preds
    else:
        y_pred_threshold = model.predict(X_test)

    # Baseline (argmax)
    y_pred_initial = model.predict(X_test)

    # Reports
    logger.info(f"\n{'='*60}")
    logger.info("THRESHOLD-BASED PREDICTIONS")
    logger.info(f"{'='*60}")
    threshold_accuracy = accuracy_score(y_test, y_pred_threshold)
    threshold_f1 = f1_score(y_test, y_pred_threshold, average="weighted")
    logger.info(f"Accuracy:  {threshold_accuracy:.4f}")
    logger.info(f"F1 (weighted): {threshold_f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred_threshold, target_names=le.classes_)}")

    logger.info(f"{'='*60}")
    logger.info("ARGMAX (BASELINE) PREDICTIONS")
    logger.info(f"{'='*60}")
    initial_accuracy = accuracy_score(y_test, y_pred_initial)
    initial_f1 = f1_score(y_test, y_pred_initial, average="weighted")
    logger.info(f"Accuracy:  {initial_accuracy:.4f}")
    logger.info(f"F1 (weighted): {initial_f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred_initial, target_names=le.classes_)}")

    logger.info(f"{'='*60}")
    logger.info("COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy improvement:  {threshold_accuracy - initial_accuracy:+.4f}")
    logger.info(f"F1 (weighted) improvement: {threshold_f1 - initial_f1:+.4f}")


if __name__ == "__main__":
    app()
