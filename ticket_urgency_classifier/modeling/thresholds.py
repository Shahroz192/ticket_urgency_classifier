"""Joint multi-class thresholding: tune and apply per-class cutoffs together.

The legacy scheme treated 'low' as a silent default and tuned other classes
one-vs-rest, then overwrote by dict order. That was designed for 3-class
(low/medium/high) and collides when 2+ classes pass. Here every class gets a
threshold; predictions pick the best passer by probability (fallback: argmax).
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.metrics import f1_score

DEFAULT_THRESHOLD_GRID = np.arange(0.10, 0.91, 0.02)


def apply_thresholds(
    y_proba: np.ndarray,
    thresholds: dict[str, float] | float,
    class_names: list[str] | np.ndarray,
) -> np.ndarray:
    """Map probabilities to class indices using per-class thresholds.

    Dict mode: a class is eligible when p >= t (missing key => t=0, always
    eligible). Among eligible classes, highest probability wins. If none are
    eligible, fall back to argmax(p).

    Scalar mode (legacy): gate on 'low', else argmax of remaining classes.
    """
    class_names = list(class_names)
    n = y_proba.shape[0]

    if isinstance(thresholds, dict):
        # Build threshold vector aligned with class order
        t_vec = np.array(
            [float(thresholds.get(name, 0.0)) for name in class_names],
            dtype=float,
        )
        eligible = y_proba >= t_vec  # (n, C)
        any_eligible = eligible.any(axis=1)
        # Mask non-eligible to -inf so argmax only sees passers
        masked = np.where(eligible, y_proba, -np.inf)
        # Rows with no passer: fall back to raw argmax
        fallback = y_proba.argmax(axis=1)
        winners = masked.argmax(axis=1)
        return np.where(any_eligible, winners, fallback).astype(int)

    # Scalar legacy path
    default_class = "low"
    if default_class not in class_names:
        raise ValueError(f"Default class '{default_class}' not found in class_names.")
    default_idx = class_names.index(default_class)
    y_pred = np.full(n, default_idx, dtype=int)
    low_mask = y_proba[:, default_idx] >= float(thresholds)
    y_pred[low_mask] = default_idx
    not_low = ~low_mask
    if not_low.any():
        temp = y_proba[not_low].copy()
        temp[:, default_idx] = -np.inf
        y_pred[not_low] = temp.argmax(axis=1)
    return y_pred


def _weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def tune_thresholds(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str] | np.ndarray,
    threshold_grid: np.ndarray | None = None,
    max_passes: int = 3,
) -> dict[str, float]:
    """Coordinate-ascent search for per-class thresholds maximizing weighted F1.

    Starts from one-vs-rest F1-optimal cutoffs per class, then sweeps each
    class threshold while holding others fixed until weighted F1 stops improving.
    """
    class_names = list(class_names)
    grid = DEFAULT_THRESHOLD_GRID if threshold_grid is None else np.asarray(threshold_grid)
    n_classes = len(class_names)

    # Init: per-class OVR F1 peak (includes every class, not just non-default)
    thresholds: dict[str, float] = {}
    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred_bin = (y_proba[:, i] >= t).astype(int)
            f1 = f1_score(y_bin, pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[name] = best_t

    def score(thr: dict[str, float]) -> float:
        return _weighted_f1(y_true, apply_thresholds(y_proba, thr, class_names))

    current = score(thresholds)
    logger.info(f"Threshold init weighted F1 (OVR): {current:.4f}")

    for pass_idx in range(max_passes):
        improved = False
        for i, name in enumerate(class_names):
            best_t, best_score = thresholds[name], current
            for t in grid:
                trial = dict(thresholds)
                trial[name] = float(t)
                s = score(trial)
                if s > best_score + 1e-9:
                    best_score, best_t = s, float(t)
            if best_t != thresholds[name]:
                thresholds[name] = best_t
                current = best_score
                improved = True
                logger.info(
                    f"  pass {pass_idx + 1}: '{name}' -> {best_t:.2f} (weighted F1 {current:.4f})"
                )
        if not improved:
            break

    logger.info(f"Joint-tuned thresholds: {thresholds}")
    logger.info(f"Final threshold weighted F1: {current:.4f}")
    return thresholds
