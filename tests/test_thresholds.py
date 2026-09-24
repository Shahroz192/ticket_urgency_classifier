"""Joint multi-class threshold tests (replaces 3-class default-class scheme)."""

import numpy as np
from sklearn.metrics import f1_score

from ticket_urgency_classifier.modeling.thresholds import apply_thresholds, tune_thresholds

CLASSES = ["critical", "high", "low", "medium", "very_low"]


def test_apply_among_passers_picks_highest_probability():
    # critical and medium both pass; medium has higher p and must win
    # regardless of dict order
    y_proba = np.array([[0.40, 0.10, 0.20, 0.45, 0.05]])
    thresholds = {"critical": 0.30, "high": 0.50, "low": 0.30, "medium": 0.35, "very_low": 0.30}
    pred = apply_thresholds(y_proba, thresholds, CLASSES)
    assert pred[0] == CLASSES.index("medium")


def test_apply_no_passer_falls_back_to_argmax():
    y_proba = np.array([[0.20, 0.25, 0.30, 0.15, 0.10]])
    thresholds = {c: 0.90 for c in CLASSES}
    pred = apply_thresholds(y_proba, thresholds, CLASSES)
    assert pred[0] == CLASSES.index("low")  # argmax of probs


def test_apply_low_is_tunable_not_silent_default():
    # high barely passes its bar but low has much higher probability
    y_proba = np.array([[0.10, 0.31, 0.55, 0.04, 0.00]])
    thresholds = {"critical": 0.5, "high": 0.30, "low": 0.50, "medium": 0.5, "very_low": 0.5}
    # low does NOT pass (0.55 >= 0.50 does pass actually)
    # set low bar above p so low is ineligible
    thresholds["low"] = 0.60
    pred = apply_thresholds(y_proba, thresholds, CLASSES)
    # low ineligible; high passes (0.31>=0.30); medium/ critical/very_low fail
    assert pred[0] == CLASSES.index("high")


def test_apply_missing_key_means_always_eligible():
    y_proba = np.array([[0.4, 0.6, 0.0, 0.0, 0.0]])
    thresholds = {"critical": 0.5}  # high missing -> t=0 always eligible
    pred = apply_thresholds(y_proba, thresholds, CLASSES)
    # both critical (fail) and high (pass) — high wins
    assert pred[0] == CLASSES.index("high")


def test_tune_thresholds_includes_low_and_returns_all_classes():
    rng = np.random.default_rng(0)
    n = 400
    # Synthetic: low is confused with medium under pure argmax
    y = rng.choice(5, size=n, p=[0.1, 0.3, 0.25, 0.25, 0.1])
    y_proba = rng.random((n, 5))
    y_proba /= y_proba.sum(axis=1, keepdims=True)
    # Boost true class slightly so signal exists
    y_proba[np.arange(n), y] += 0.25
    y_proba /= y_proba.sum(axis=1, keepdims=True)

    thr = tune_thresholds(y_proba, y, CLASSES, max_passes=2)
    assert set(thr.keys()) == set(CLASSES)
    assert all(0.0 <= v <= 1.0 for v in thr.values())

    pred = apply_thresholds(y_proba, thr, CLASSES)
    # Joint tuning must be at least as good as naive all-0.5 passers+argmax
    naive_thr = {c: 0.5 for c in CLASSES}
    naive_f1 = f1_score(y, apply_thresholds(y_proba, naive_thr, CLASSES), average="weighted")
    tuned_f1 = f1_score(y, pred, average="weighted")
    assert tuned_f1 >= naive_f1 - 1e-9


def test_scalar_legacy_path_still_works():
    y_proba = np.array([[0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    classes3 = ["high", "high2", "low"]  # not used - need 'low' present
    classes3 = ["critical", "high", "low"]
    y_proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
    pred = apply_thresholds(y_proba, 0.5, classes3)
    # row0: low 0.1 < 0.5 -> argmax non-low = critical
    # row1: low 0.5 >= 0.5 -> low
    assert pred[0] == classes3.index("critical")
    assert pred[1] == classes3.index("low")
