"""Diagnostic PR / F1 curve for the joint per-class 'low' threshold.

Scans only the 'low' cutoff on the *train* split (never test) while other
class thresholds stay frozen at the val-tuned values in best_threshold.joblib.
The saved operating point is marked; the train-scan peak is diagnostic only
and must not be used to retune or report test performance.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from ticket_urgency_classifier.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from ticket_urgency_classifier.modeling.thresholds import apply_thresholds


def main():
    model_file = MODELS_DIR / "best_rf_model.joblib"
    model = joblib.load(model_file)

    # Train split only — selecting a "best" threshold on test would leak labels.
    df = pd.read_csv(PROCESSED_DATA_DIR / "train_features.csv")
    y = df["priority"]
    X = df.drop(columns=["priority"])

    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    thr_file = MODELS_DIR / "best_threshold.joblib"
    saved = joblib.load(thr_file) if thr_file.exists() else {}
    if not isinstance(saved, dict):
        saved = {}

    y_proba = model.predict_proba(X)
    class_names = list(le.classes_)
    if "low" not in class_names:
        raise ValueError("label encoder has no 'low' class")
    low_idx = class_names.index("low")

    # Joint mode: freeze non-low thresholds at saved val-tuned values (missing => 0)
    base_thr = {name: float(saved.get(name, 0.0)) for name in class_names}
    saved_low = float(saved.get("low", 0.50))
    thresholds = np.arange(0.10, 0.91, 0.02)

    precisions, recalls, f1_lows, weighted_f1s = [], [], [], []
    for t in thresholds:
        thr = dict(base_thr)
        thr["low"] = float(t)
        y_pred = apply_thresholds(y_proba, thr, class_names)
        precisions.append(precision_score(y, y_pred, average="weighted", zero_division=0))
        recalls.append(recall_score(y, y_pred, average="weighted", zero_division=0))
        f1_lows.append(
            f1_score(y, y_pred, average=None, zero_division=0)[low_idx]
        )
        weighted_f1s.append(f1_score(y, y_pred, average="weighted", zero_division=0))

    # Diagnostic peak on train only — not a calibration value
    peak_idx = int(np.argmax(weighted_f1s))
    peak_t = float(thresholds[peak_idx])
    # Saved operating point may not sit on the grid; show its x position anyway
    saved_y = (
        float(weighted_f1s[int(np.argmin(np.abs(thresholds - saved_low)))])
        if thresholds.size
        else 0.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1 = axes[0]
    ax1.plot(thresholds, precisions, label="Precision (weighted)", color="#2E86AB", linewidth=2)
    ax1.plot(thresholds, recalls, label="Recall (weighted)", color="#A23B72", linewidth=2)
    ax1.plot(thresholds, weighted_f1s, label="Weighted F1", color="#F18F01", linewidth=2)
    ax1.axvline(
        x=saved_low,
        color="green",
        linestyle="--",
        alpha=0.85,
        label=f"Saved (val) low thr = {saved_low:.2f}",
    )
    ax1.axvline(
        x=peak_t,
        color="purple",
        linestyle=":",
        alpha=0.7,
        label=f"Train-scan peak = {peak_t:.2f} (diagnostic)",
    )
    ax1.scatter([saved_low], [saved_y], color="green", s=80, zorder=5)
    ax1.set_xlabel("Threshold (low class; others frozen at saved)")
    ax1.set_ylabel("Score")
    ax1.set_title("Precision / Recall / F1 vs low threshold (train diagnostic)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(thresholds, f1_lows, label="F1 (low class)", color="#A23B72", linewidth=2)
    ax2.axvline(
        x=saved_low,
        color="green",
        linestyle="--",
        alpha=0.85,
        label=f"Saved (val) low thr = {saved_low:.2f}",
    )
    ax2.axvline(
        x=peak_t,
        color="purple",
        linestyle=":",
        alpha=0.7,
        label=f"Train-scan peak = {peak_t:.2f} (diagnostic)",
    )
    ax2.set_xlabel("Threshold (low class; others frozen at saved)")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 for 'low' by threshold (train diagnostic)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Joint threshold diagnostic — saved low="
        f"{saved_low:.2f} (train wF1@saved≈{saved_y:.4f}); "
        f"other classes frozen. Train peak {peak_t:.2f} is NOT a test score.",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "pr_curve_threshold.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")

    print(f"\nSaved thresholds (val-tuned): {saved}")
    print(f"Saved low threshold:          {saved_low:.2f}")
    print(f"Train-scan peak (diagnostic): {peak_t:.2f} (wF1={weighted_f1s[peak_idx]:.4f})")
    print(f"Train wF1 @ saved low thr:    {saved_y:.4f}")
    print("\nNote: curve is train-only diagnostic; never select thresholds on test.")
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1-low':>10} {'W-F1':>10}")
    print("-" * 52)
    for i, t in enumerate(thresholds):
        marker = ""
        if abs(t - saved_low) < 1e-9:
            marker = " ← saved"
        elif i == peak_idx:
            marker = " ← train peak"
        print(
            f"{t:>10.2f} {precisions[i]:>10.4f} {recalls[i]:>10.4f} "
            f"{f1_lows[i]:>10.4f} {weighted_f1s[i]:>10.4f}{marker}"
        )


if __name__ == "__main__":
    main()
