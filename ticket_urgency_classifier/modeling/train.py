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
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer
import yaml

from ticket_urgency_classifier.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()


@app.command()
def main():
    """Train the best Random Forest model with MLflow tracking."""
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
        df_test = pd.read_csv(PROCESSED_DATA_DIR / "test_features.csv")
        logger.info(f"Loaded processed test data. Shape: {df_test.shape}")

        y_train = df_train["priority"]
        X_train = df_train.drop(columns=["priority"])
        y_val = df_test["priority"]
        X_val = df_test.drop(columns=["priority"])
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

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
        random_search.fit(X_train, y_train)
        logger.success("Hyperparameter tuning complete.")

        best_model = random_search.best_estimator_

        label_encoder_file = MODELS_DIR / "label_encoder.joblib"
        le = joblib.load(label_encoder_file)
        y_proba = best_model.predict_proba(X_val)

        class_names = le.classes_
        try:
            low_class_index = np.where(class_names == "low")[0][0]
            _ = np.where(class_names == "high")[0][0]
            _ = np.where(class_names == "medium")[0][0]
        except IndexError:
            logger.error(
                "Error: Could not find 'low', 'high', or 'medium' class in label encoder."
            )
            raise

        thresholds = np.arange(0.10, 0.51, 0.01)

        best_threshold = 0.0
        best_weighted_f1 = 0.0
        original_weighted_f1 = None

        y_pred_original = best_model.predict(X_val)
        original_weighted_f1 = f1_score(y_val, y_pred_original, average="weighted")

        logger.info(f"Original Weighted F1-Score: {original_weighted_f1:.4f}")
        for thresh in thresholds:
            y_pred_thresh = np.zeros(len(y_val), dtype=int)
            low_mask = y_proba[:, low_class_index] >= thresh
            y_pred_thresh[low_mask] = low_class_index
            not_low_mask = ~low_mask
            temp_proba = y_proba[not_low_mask].copy()
            temp_proba[:, low_class_index] = 0
            if temp_proba.shape[0] > 0:
                remaining_preds = np.argmax(temp_proba, axis=1)
                y_pred_thresh[not_low_mask] = remaining_preds
            current_weighted_f1 = f1_score(y_val, y_pred_thresh, average="weighted")
            if current_weighted_f1 > best_weighted_f1:
                best_weighted_f1 = current_weighted_f1
                best_threshold = thresh

        if best_weighted_f1 <= original_weighted_f1:
            best_threshold = 0.5
            logger.info("No improvement found, best threshold remains at 0.5 (default).")
        else:
            logger.info(f"Best Threshold for 'low' class: {best_threshold:.2f}")
            logger.info(f"Best Weighted F1-Score achieved: {best_weighted_f1:.4f}")
            logger.info(f"Improvement: {best_weighted_f1 - original_weighted_f1:.4f}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        threshold_file = MODELS_DIR / "best_threshold.joblib"
        joblib.dump(best_threshold, threshold_file)
        logger.success(f"Best threshold ({best_threshold:.4f}) saved to {threshold_file}")

        best_model_file = MODELS_DIR / "best_rf_model.joblib"
        joblib.dump(best_model, best_model_file)
        logger.success(f"Best Random Forest model saved to {best_model_file}")

        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_f1_weighted", random_search.best_score_)
        mlflow.log_metric("best_threshold", best_threshold)
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
