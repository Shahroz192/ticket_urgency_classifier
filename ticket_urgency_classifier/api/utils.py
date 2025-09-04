from typing import Any, List, Optional

import joblib
from loguru import logger
import mlflow.pyfunc

from ticket_urgency_classifier.config import MODELS_DIR

# Paths
model_path = MODELS_DIR / "best_rf_model.joblib"
label_encoder_path = MODELS_DIR / "label_encoder.joblib"
top_tags_path = MODELS_DIR / "top_tags.joblib"
threshold_path = MODELS_DIR / "best_threshold.joblib"

# Globals for loaded objects
model: Optional[Any] = None
label_encoder: Optional[Any] = None
top_tags: Optional[List[str]] = None
threshold: Optional[float] = None


def load_resources():
    """Load model, label encoder, top tags, and threshold with error handling."""
    global model, label_encoder, top_tags, threshold
    try:
        logger.info("Loading model from MLflow registry...")
        model = mlflow.pyfunc.load_model("models:/ticket_urgency_classifier@challenger")
        logger.success("Model loaded successfully from MLflow registry.")
    except Exception as e:
        logger.error(f"Failed to load model from registry: {e}")
        logger.info("Falling back to local model file...")
        # Fallback to local file if registry fails
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            model = None
        else:
            try:
                logger.info(f"Loading model from {model_path}...")
                model = joblib.load(model_path)
                logger.success("Model loaded from local file as fallback.")
            except Exception as load_e:
                logger.error(f"Failed to load model from local file: {load_e}")
                model = None

    try:
        if not label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
        logger.info(f"Loading label encoder from {label_encoder_path}...")
        label_encoder = joblib.load(label_encoder_path)
        logger.success("Label encoder loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load label encoder: {e}")
        label_encoder = None

    try:
        if not top_tags_path.exists():
            raise FileNotFoundError(f"Top tags file not found: {top_tags_path}")
        logger.info(f"Loading top tags from {top_tags_path}...")
        top_tags = joblib.load(top_tags_path)
        logger.success("Top tags loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load top tags: {e}")
        top_tags = None

    try:
        if not threshold_path.exists():
            logger.warning(f"Threshold file not found: {threshold_path}. Using default 0.5.")
            threshold = 0.5
        else:
            logger.info(f"Loading threshold from {threshold_path}...")
            threshold = joblib.load(threshold_path)
            logger.success("Threshold loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load threshold: {e}")
        threshold = 0.5
