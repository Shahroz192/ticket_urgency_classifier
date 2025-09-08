from pathlib import Path
from typing import Any, List

import joblib
from loguru import logger
import mlflow.pyfunc
import numpy as np
import pandas as pd
import typer

from ticket_urgency_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR
from ticket_urgency_classifier.features import (
    add_tag_features,
    engineer_features,
    generate_sentence_transformer_embeddings,
)

app = typer.Typer()


def load_model(model_uri: str = "models:/ticket_urgency_classifier@challenger") -> Any:
    """Load the trained model from MLflow Model Registry with error handling."""
    try:
        logger.info(f"Loading model from MLflow registry: {model_uri}...")
        model = mlflow.pyfunc.load_model(model_uri)
        # Check if the model has predict_proba method
        if hasattr(model, "predict_proba"):
            logger.success("Model loaded successfully from MLflow registry.")
            return model
        else:
            logger.warning(
                "MLflow model doesn't have predict_proba method, falling back to local model file..."
            )
    except Exception as e:
        logger.error(f"Failed to load model from registry: {e}")
        logger.info("Falling back to local model file...")

    model_path = MODELS_DIR / "best_rf_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        logger.success("Model loaded from local file as fallback.")
        return model
    else:
        raise FileNotFoundError(f"Model not found in registry or locally: {model_uri}")


def load_top_tags(top_tags_path: Path) -> List:
    """Load the top tags list from the specified path with error handling."""
    if not top_tags_path.exists():
        logger.error(f"Top tags file not found: {top_tags_path}")
        raise FileNotFoundError(f"Top tags file not found: {top_tags_path}")
    try:
        logger.info(f"Loading top tags from {top_tags_path}...")
        top_tags = joblib.load(top_tags_path)
        logger.success("Top tags loaded successfully.")
        return top_tags
    except Exception as e:
        logger.error(f"Failed to load top tags: {e}")
        raise


def load_label_encoder(label_encoder_path: Path) -> Any:
    """Load the label encoder from the specified path with error handling."""
    if not label_encoder_path.exists():
        logger.error(f"Label encoder file not found: {label_encoder_path}")
        raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
    try:
        logger.info(f"Loading label encoder from {label_encoder_path}...")
        label_encoder = joblib.load(label_encoder_path)
        logger.success("Label encoder loaded successfully.")
        return label_encoder
    except Exception as e:
        logger.error(f"Failed to load label encoder: {e}")
        raise


def load_threshold(threshold_path: Path) -> float:
    """Load the threshold from the specified path with error handling."""
    if not threshold_path.exists():
        logger.warning(f"Threshold file not found: {threshold_path}. Using default 0.5.")
        return 0.5
    try:
        logger.info(f"Loading threshold from {threshold_path}...")
        threshold = joblib.load(threshold_path)
        logger.success("Threshold loaded successfully.")
        return threshold
    except Exception as e:
        logger.error(f"Failed to load threshold: {e}")
        return 0.5


def predict(
    df: pd.DataFrame,
    model: Any,
    top_tags: List,
    threshold: float,
) -> np.ndarray:
    """
    Predict ticket urgency using the trained model and a custom threshold.
    """
    logger.info("Engineering features for prediction...")
    df = engineer_features(df.copy())
    df = add_tag_features(df, top_tags)
    df_embeddings = generate_sentence_transformer_embeddings(df, "")
    df = pd.concat([df, df_embeddings], axis=1)
    logger.success("Features engineered successfully.")

    cols_to_drop = ["subject", "body", "full_text"] + [f"tag_{i}" for i in range(1, 9)]
    df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    X = df
    if X.empty:
        logger.error("No feature columns found after processing.")
        raise ValueError("No feature columns found for prediction.")

    logger.info("Making predictions...")
    # Load label encoder first to determine number of classes
    label_encoder_path = MODELS_DIR / "label_encoder.joblib"
    label_encoder = load_label_encoder(label_encoder_path)

    # Check if model has predict_proba method
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    else:
        logger.warning("Model doesn't have predict_proba method, using predict method instead.")
        y_pred = model.predict(X)
        n_classes = len(label_encoder.classes_)
        y_proba = np.zeros((len(y_pred), n_classes))
        for i, pred in enumerate(y_pred):
            y_proba[i, pred] = 1.0

    class_names = label_encoder.classes_
    try:
        low_class_indices = np.where(class_names == "low")[0]
        if len(low_class_indices) > 0:
            low_class_index = low_class_indices[0]
        else:
            raise ValueError("Could not find 'low' class in label encoder.")
    except (IndexError, ValueError):
        logger.error("Error: Could not find 'low' class in label encoder.")
        raise

    y_pred = np.zeros(len(X), dtype=int)

    low_mask = y_proba[:, low_class_index] >= threshold

    y_pred[low_mask] = low_class_index

    not_low_mask = ~low_mask
    temp_proba = y_proba[not_low_mask].copy()
    temp_proba[:, low_class_index] = 0

    if temp_proba.shape[0] > 0:
        remaining_preds = np.argmax(temp_proba, axis=1)
        y_pred[not_low_mask] = remaining_preds

    logger.success("Predictions made successfully.")
    return y_pred


@app.command("batch")
def predict_batch(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to the input data CSV file."),
    model_uri: str = typer.Option(
        "models:/ticket_urgency_classifier@challenger", "--model", "-m", help="MLflow model URI"
    ),
    top_tags_path: Path = MODELS_DIR / "top_tags.joblib",
    label_encoder_path: Path = MODELS_DIR / "label_encoder.joblib",
    threshold_path: Path = MODELS_DIR / "best_threshold.joblib",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """
    Main function to perform predictions on input data.
    """
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise typer.Exit(code=1)

    model = load_model(model_uri)
    top_tags = load_top_tags(top_tags_path)
    label_encoder = load_label_encoder(label_encoder_path)
    threshold = load_threshold(threshold_path)

    logger.info(f"Loading input data from {input_path}...")
    df = pd.read_csv(input_path)
    logger.success("Input data loaded successfully.")

    predictions_encoded = predict(df, model, top_tags, threshold)
    predictions = label_encoder.inverse_transform(predictions_encoded)

    df["predicted_urgency"] = predictions
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")


@app.command("single")
def predict_single(
    model_uri: str = typer.Option(
        "models:/ticket_urgency_classifier@challenger", "--model", "-m", help="MLflow model URI"
    ),
    subject: str = typer.Option(..., "--subject", "-s", help="Ticket subject"),
    body: str = typer.Option(..., "--body", "-b", help="Ticket body"),
    queue: str = typer.Option(..., "--queue", "-q", help="Ticket queue"),
    ticket_type: str = typer.Option(..., "--type", "-t", help="Ticket type"),
    language: str = typer.Option("en", "--language", "-l", help="Ticket language"),
    tags: List[str] = typer.Option([], "--tag", help="Ticket tags (can be multiple)"),
    top_tags_path: Path = MODELS_DIR / "top_tags.joblib",
    label_encoder_path: Path = MODELS_DIR / "label_encoder.joblib",
    threshold_path: Path = MODELS_DIR / "best_threshold.joblib",
):
    """
    Perform prediction on a single ticket.
    """
    model = load_model(model_uri)
    top_tags = load_top_tags(top_tags_path)
    label_encoder = load_label_encoder(label_encoder_path)
    threshold = load_threshold(threshold_path)

    tags_padded = tags + [""] * (8 - len(tags))
    tag_dict = {f"tag_{i + 1}": tag for i, tag in enumerate(tags_padded)}

    df = pd.DataFrame(
        [
            {
                "subject": subject,
                "body": body,
                "queue": queue,
                "type": ticket_type,
                "language": language,
                **tag_dict,
            }
        ]
    )

    prediction_encoded = predict(df, model, top_tags, threshold)
    human_readable_label = label_encoder.inverse_transform(prediction_encoded)[0]

    logger.success(f"Prediction: {human_readable_label}")
    print(f"Predicted Urgency: {human_readable_label}")


if __name__ == "__main__":
    app()
