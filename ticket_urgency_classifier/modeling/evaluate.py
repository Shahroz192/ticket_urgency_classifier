"""Evaluates the trained model using the threshold-based prediction logic."""

from loguru import logger
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import typer

from ticket_urgency_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR
from ticket_urgency_classifier.modeling.predict import (
    load_label_encoder,
    load_model,
    load_threshold,
    load_top_tags,
    predict,
)

app = typer.Typer()


@app.command()
def main():
    """Evaluate the trained model using the threshold-based prediction logic."""
    logger.info("Starting model evaluation with threshold-based predictions...")

    # 1. Load processed final test data
    test_file = PROCESSED_DATA_DIR / "test_features.csv"
    if not test_file.exists():
        logger.error("Processed test data not found. Run data preparation first.")
        return

    df_test = pd.read_csv(test_file)
    logger.info(f"Loaded processed test data. Shape: {df_test.shape}")

    # 2. Separate features and target
    y_test = df_test["priority"]
    X_test = df_test.drop(columns=["priority"])
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 3. Load the trained model, label encoder, top tags, and threshold
    model_file = MODELS_DIR / "best_rf_model.joblib"
    encoder_file = MODELS_DIR / "label_encoder.joblib"
    top_tags_file = MODELS_DIR / "top_tags.joblib"
    threshold_file = MODELS_DIR / "best_threshold.joblib"

    if not model_file.exists() or not encoder_file.exists() or not top_tags_file.exists():
        logger.error("Model, encoder, or top tags not found. Run model training first.")
        return

    model = load_model(model_file)
    le = load_label_encoder(encoder_file)
    top_tags = load_top_tags(top_tags_file)
    threshold = load_threshold(threshold_file)
    logger.info("Model, label encoder, top tags, and threshold loaded successfully.")

    # 4. Prepare data for prediction using the same feature engineering as in predict.py
    df_for_prediction = X_test.copy()

    # 5. Make predictions using the threshold-based predict function
    y_pred_threshold = predict(df_for_prediction, model, top_tags, threshold)
    threshold_accuracy = accuracy_score(y_test, y_pred_threshold)
    logger.info(f"Threshold-based model accuracy: {threshold_accuracy:.4f}")
    logger.info("Threshold-based classification report:")
    logger.info(f"\n{classification_report(y_test, y_pred_threshold, target_names=le.classes_)}")

    # 6. Make initial predictions (for comparison)
    y_pred_initial = model.predict(X_test)
    initial_accuracy = accuracy_score(y_test, y_pred_initial)
    logger.info(f"Initial model accuracy: {initial_accuracy:.4f}")
    logger.info("Initial classification report:")
    logger.info(f"\n{classification_report(y_test, y_pred_initial, target_names=le.classes_)}")

    # 7. Compare the results
    logger.info(f"Accuracy improvement: {threshold_accuracy - initial_accuracy:.4f}")


if __name__ == "__main__":
    app()
