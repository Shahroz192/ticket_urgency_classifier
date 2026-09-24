"""Evaluates the trained model using joint per-class threshold predictions."""

import json

import joblib
from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import typer

from ticket_urgency_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from ticket_urgency_classifier.modeling.thresholds import apply_thresholds

app = typer.Typer()

EVAL_DIR = PROJ_ROOT / "reports" / "evaluation"


def _save_cm(y_true, y_pred, class_names, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


@app.command()
def main():
    """Evaluate the trained model using joint threshold-based predictions."""
    logger.info("Starting model evaluation with threshold-based predictions...")

    test_file = PROCESSED_DATA_DIR / "test_features.csv"
    if not test_file.exists():
        logger.error("Processed test data not found. Run data preparation first.")
        return

    df_test = pd.read_csv(test_file)
    logger.info(f"Loaded processed test data. Shape: {df_test.shape}")

    y_test = df_test["priority"].to_numpy()
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
    class_names = list(le.classes_)

    if not threshold_file.exists():
        logger.warning("Threshold file not found, using argmax predictions.")
        best_thresholds: dict | float = {}
    else:
        best_thresholds = joblib.load(threshold_file)
    logger.info(f"Model, label encoder, and thresholds loaded successfully.")
    logger.info(f"Thresholds: {best_thresholds}")

    y_proba = model.predict_proba(X_test)
    y_pred_threshold = apply_thresholds(y_proba, best_thresholds, class_names)
    y_pred_initial = model.predict(X_test)

    # --- Metrics ---
    threshold_accuracy = accuracy_score(y_test, y_pred_threshold)
    threshold_f1 = f1_score(y_test, y_pred_threshold, average="weighted")
    initial_accuracy = accuracy_score(y_test, y_pred_initial)
    initial_f1 = f1_score(y_test, y_pred_initial, average="weighted")

    report_th = classification_report(
        y_test, y_pred_threshold, target_names=class_names, output_dict=True
    )
    report_am = classification_report(
        y_test, y_pred_initial, target_names=class_names, output_dict=True
    )

    # Dummy baselines (floor): fit on train labels only — never on y_test,
    # otherwise test frequencies leak into the floor and understate it.
    # DummyClassifier checks len(X)==len(y), so pass train-shaped X (zeros).
    y_train = pd.read_csv(
        PROCESSED_DATA_DIR / "train_features.csv", usecols=["priority"]
    )["priority"].to_numpy()
    X_train_dummy = np.zeros((len(y_train), 1))
    dummy_prior = DummyClassifier(strategy="prior").fit(X_train_dummy, y_train)
    dummy_strat = DummyClassifier(strategy="stratified", random_state=42).fit(
        X_train_dummy, y_train
    )
    dummy_prior_acc = accuracy_score(y_test, dummy_prior.predict(X_test))
    dummy_prior_f1 = f1_score(y_test, dummy_prior.predict(X_test), average="weighted")
    dummy_strat_f1 = f1_score(y_test, dummy_strat.predict(X_test), average="weighted")

    logger.info(f"\n{'='*60}")
    logger.info("THRESHOLD-BASED PREDICTIONS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:  {threshold_accuracy:.4f}")
    logger.info(f"F1 (weighted): {threshold_f1:.4f}")
    logger.info(
        f"\n{classification_report(y_test, y_pred_threshold, target_names=class_names)}"
    )

    logger.info(f"{'='*60}")
    logger.info("ARGMAX (BASELINE) PREDICTIONS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:  {initial_accuracy:.4f}")
    logger.info(f"F1 (weighted): {initial_f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred_initial, target_names=class_names)}")

    logger.info(f"{'='*60}")
    logger.info("DUMMY FLOORS")
    logger.info(f"{'='*60}")
    logger.info(f"Majority-class acc={dummy_prior_acc:.4f} wF1={dummy_prior_f1:.4f}")
    logger.info(f"Stratified dummy wF1={dummy_strat_f1:.4f}")

    logger.info(f"{'='*60}")
    logger.info("COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy improvement:  {threshold_accuracy - initial_accuracy:+.4f}")
    logger.info(f"F1 (weighted) improvement: {threshold_f1 - initial_f1:+.4f}")

    # --- Persist artifacts ---
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_test": int(len(y_test)),
        "threshold_accuracy": float(threshold_accuracy),
        "threshold_f1_weighted": float(threshold_f1),
        "argmax_accuracy": float(initial_accuracy),
        "argmax_f1_weighted": float(initial_f1),
        "accuracy_improvement": float(threshold_accuracy - initial_accuracy),
        "f1_improvement": float(threshold_f1 - initial_f1),
        "dummy_prior_accuracy": float(dummy_prior_acc),
        "dummy_prior_f1_weighted": float(dummy_prior_f1),
        "dummy_stratified_f1_weighted": float(dummy_strat_f1),
        "thresholds": {
            k: float(v) for k, v in best_thresholds.items()
        }
        if isinstance(best_thresholds, dict)
        else {"_scalar": float(best_thresholds)},
        "per_class_threshold": report_th,
        "per_class_argmax": report_am,
    }
    with open(EVAL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Log test metrics to MLflow so feature changes are comparable run-to-run
    # (train.py logs val CV/thresholded F1; evaluate was silent until now).
    try:
        mlflow.set_experiment("ticket_urgency_classifier")
        with mlflow.start_run(run_name="evaluate-test"):
            mlflow.log_metrics(
                {
                    "test_threshold_accuracy": float(threshold_accuracy),
                    "test_threshold_f1_weighted": float(threshold_f1),
                    "test_argmax_accuracy": float(initial_accuracy),
                    "test_argmax_f1_weighted": float(initial_f1),
                    "test_dummy_prior_f1_weighted": float(dummy_prior_f1),
                    "test_dummy_stratified_f1_weighted": float(dummy_strat_f1),
                }
            )
            if isinstance(best_thresholds, dict):
                mlflow.log_metrics(
                    {f"threshold_{k}": float(v) for k, v in best_thresholds.items()}
                )
            mlflow.log_artifact(str(EVAL_DIR / "metrics.json"))
    except Exception as exc:  # MLflow must never fail the evaluation itself
        logger.warning(f"MLflow test-metric logging skipped: {exc}")

    pd.DataFrame(report_th).T.to_csv(EVAL_DIR / "classification_report_threshold.csv")
    pd.DataFrame(report_am).T.to_csv(EVAL_DIR / "classification_report_argmax.csv")

    cm_th = _save_cm(
        y_test, y_pred_threshold, class_names, "Threshold", EVAL_DIR / "confusion_matrix_threshold.png"
    )
    cm_am = _save_cm(
        y_test, y_pred_initial, class_names, "Argmax", EVAL_DIR / "confusion_matrix_argmax.png"
    )
    pd.DataFrame(cm_th, index=class_names, columns=class_names).to_csv(
        EVAL_DIR / "confusion_matrix_threshold.csv"
    )
    pd.DataFrame(cm_am, index=class_names, columns=class_names).to_csv(
        EVAL_DIR / "confusion_matrix_argmax.csv"
    )

    logger.success(f"Evaluation artifacts written to {EVAL_DIR}")


if __name__ == "__main__":
    app()
