"""Trains the best Random Forest model with MLflow experiment tracking."""

from pathlib import Path

import joblib
from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer
import yaml

from ticket_urgency_classifier.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from ticket_urgency_classifier.modeling.thresholds import apply_thresholds, tune_thresholds

app = typer.Typer()


def _split_for_threshold_calibration(X_train, y_train, random_state=42):
    """Reserve training rows for threshold calibration, separate from the test set."""
    return train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )


def _tune_thresholds(model, X_val, y_val, le):
    """Tune per-class thresholds jointly on the validation set.

    Maximizes weighted F1 over all classes (including 'low') via coordinate
    ascent — no silent default class, no dict-order overwrite.
    """
    y_proba = model.predict_proba(X_val)
    class_names = list(le.classes_)

    y_pred_original = model.predict(X_val)
    original_weighted_f1 = f1_score(y_val, y_pred_original, average="weighted")
    logger.info(f"Original (argmax) Weighted F1-Score: {original_weighted_f1:.4f}")

    best_thresholds = tune_thresholds(y_proba, y_val, class_names)

    y_pred_thresholded = apply_thresholds(y_proba, best_thresholds, class_names)
    thresholded_f1 = f1_score(y_val, y_pred_thresholded, average="weighted")
    logger.info(f"Threshold-optimized Weighted F1-Score: {thresholded_f1:.4f}")
    if thresholded_f1 > original_weighted_f1:
        logger.info(f"Improvement over argmax: {thresholded_f1 - original_weighted_f1:.4f}")
    else:
        logger.info("No improvement over argmax.")

    return best_thresholds, thresholded_f1


@app.command()
def main(
    tune_only: bool = typer.Option(False, "--tune-only", help="Skip training, only re-tune thresholds on the existing model."),
):
    """Train the best Random Forest model with MLflow tracking."""
    if tune_only is True:
        logger.info("Tune-only mode: loading existing model and re-tuning thresholds.")
        model_file = MODELS_DIR / "best_rf_model.joblib"
        if not model_file.exists():
            logger.error(f"Existing model not found at {model_file}. Run full training first.")
            return
        best_model = joblib.load(model_file)

        train_file = PROCESSED_DATA_DIR / "train_features.csv"
        if not train_file.exists():
            logger.error(f"Training data not found at {train_file}. Run data preparation first.")
            return
        df_train = pd.read_csv(train_file)
        X_train = df_train.drop(columns=["priority"])
        y_train = df_train["priority"]
        X_fit, X_val, y_fit, y_val = _split_for_threshold_calibration(X_train, y_train)
        threshold_model = joblib.load(model_file)
        threshold_model.fit(X_fit, y_fit)

        le = joblib.load(MODELS_DIR / "label_encoder.joblib")
        best_thresholds, _ = _tune_thresholds(threshold_model, X_val, y_val, le)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        threshold_file = MODELS_DIR / "best_threshold.joblib"
        joblib.dump(best_thresholds, threshold_file)
        logger.success(f"Per-class thresholds saved to {threshold_file}")
        return

    logger.info("Starting Random Forest model training...")

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    RANDOM_FOREST_PARMS = config["random_forest"]
    random_search_config = config["random_search"]
    RANDOM_SEARCH_PARMS_DICT = {
        "model__n_estimators": randint(
            random_search_config["model__n_estimators"]["min"],
            random_search_config["model__n_estimators"]["max"],
        ),
        "model__max_depth": random_search_config["model__max_depth"],
        "model__min_samples_split": randint(
            random_search_config["model__min_samples_split"]["min"],
            random_search_config["model__min_samples_split"]["max"],
        ),
        "model__min_samples_leaf": randint(
            random_search_config["model__min_samples_leaf"]["min"],
            random_search_config["model__min_samples_leaf"]["max"],
        ),
        "model__max_features": random_search_config["model__max_features"],
    }

    # Set up MLflow experiment
    mlflow.set_experiment("ticket_urgency_classifier")
    with mlflow.start_run():
        train_file = PROCESSED_DATA_DIR / "train_features.csv"
        if not train_file.exists():
            logger.error("Processed training data not found. Run data preparation first.")
            return

        df_train = pd.read_csv(train_file)
        logger.info(f"Loaded processed training data. Shape: {df_train.shape}")
        y_train = df_train["priority"]
        X_train = df_train.drop(columns=["priority"])
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        X_fit, X_val, y_fit, y_val = _split_for_threshold_calibration(
            X_train,
            y_train,
            random_state=random_search_config["random_state"],
        )

        categorical_features = ["language", "queue", "type", "queue_type_interaction"]
        numerical_features = [col for col in X_train.columns if col not in categorical_features]

        logger.info(f"Categorical features: {len(categorical_features)}")
        logger.info(f"Numerical features: {len(numerical_features)}")

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
                ("num", StandardScaler(), numerical_features),
            ],
            remainder="passthrough",
        )

        rf_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(**RANDOM_FOREST_PARMS)),
            ]
        )

        param_dist = RANDOM_SEARCH_PARMS_DICT
        cv = StratifiedKFold(
            n_splits=random_search_config["cv_folds"],
            shuffle=True,
            random_state=random_search_config["random_state"],
        )

        logger.info("Setting up RandomizedSearchCV for Random Forest...")
        random_search = RandomizedSearchCV(
            estimator=rf_pipeline,
            param_distributions=param_dist,
            n_iter=random_search_config["n_iter"],
            scoring=random_search_config["scoring"],
            cv=cv,
            n_jobs=-1,
            verbose=2,
            random_state=random_search_config["random_state"],
        )

        logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")
        mlflow.sklearn.autolog()
        random_search.fit(X_fit, y_fit)
        logger.success("Hyperparameter tuning complete.")

        best_model = random_search.best_estimator_

        le = joblib.load(MODELS_DIR / "label_encoder.joblib")
        best_thresholds, thresholded_f1 = _tune_thresholds(best_model, X_val, y_val, le)

        # Calibrate thresholds on held-out training rows, then use all training
        # rows for the final model. The test split stays untouched until evaluation.
        best_model.fit(X_train, y_train)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        threshold_file = MODELS_DIR / "best_threshold.joblib"
        joblib.dump(best_thresholds, threshold_file)
        logger.success(f"Best per-class thresholds saved to {threshold_file}")

        best_model_file = MODELS_DIR / "best_rf_model.joblib"
        joblib.dump(best_model, best_model_file)
        logger.success(f"Best Random Forest model saved to {best_model_file}")

        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_f1_weighted", random_search.best_score_)
        mlflow.log_metric("thresholded_f1_weighted", thresholded_f1)
        mlflow.log_metrics({f"threshold_{k}": v for k, v in best_thresholds.items()})
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_artifact(str(threshold_file))
        mlflow.log_artifact(str(best_model_file))
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "ticket_urgency_classifier")
        logger.info(f"Best parameters found: {random_search.best_params_}")
        logger.info(f"Best cross-validation F1-weighted score: {random_search.best_score_:.4f}")
        logger.info("MLflow run completed. View results with 'mlflow ui'")
        logger.info("Model registered in MLflow Registry as 'ticket_urgency_classifier'")


if __name__ == "__main__":
    app()
