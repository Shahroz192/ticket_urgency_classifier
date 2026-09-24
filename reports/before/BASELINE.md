# Baseline before German feature fixes

Snapshot: 2026-09-24 (before bilingual keywords / DE sentiment / type imputation)

## Test set (n=10,700) — joint thresholds

| Metric | Value |
|--------|------:|
| Threshold accuracy | **0.7754** |
| Threshold weighted F1 | **0.7746** |
| Argmax accuracy | 0.7455 |
| Argmax weighted F1 | 0.7385 |
| Accuracy improvement (thr−argmax) | +0.0299 |
| F1 improvement (thr−argmax) | +0.0361 |

## Dummy floors (train-fit)

| Metric | Value |
|--------|------:|
| Majority prior accuracy | 0.3756 |
| Majority prior weighted F1 | 0.2051 |
| Stratified dummy weighted F1 | 0.3133 |

## Per-class F1 (threshold mode)

| Class | F1 | Recall |
|-------|---:|-------:|
| critical | 0.867 | 0.911 |
| high | 0.779 | 0.777 |
| low | 0.733 | 0.661 |
| medium | 0.771 | 0.813 |
| very_low | 0.928 | 0.904 |

## Thresholds

`critical=0.20, high=0.42, low=0.28, medium=0.42, very_low=0.24`

## MLflow

Experiment `ticket_urgency_classifier` logs **train/val** metrics only:
- `best_cv_f1_weighted` (CV on fit split)
- `thresholded_f1_weighted` (val, joint thresholds)
- per-class `threshold_*`

`evaluate` does **not** log test metrics to MLflow. After feature changes, compare:
1. MLflow run `best_cv_f1_weighted` / `thresholded_f1_weighted` (val signal)
2. This report vs `reports/evaluation/metrics.json` (test signal)

## Full artifacts

- `reports/before/metrics_before_de_features.json`
- `reports/evaluation/metrics.json` (same numbers until re-evaluated)

## MLflow baseline (last full train before DE features)

Run `fd3befbe` (awesome-wren-280):
- `best_cv_f1_weighted` = 0.6883
- `thresholded_f1_weighted` (val) = 0.7187

Later toy/test runs (cv=0.99) are not comparable.

After DE feature changes: retrain (full-pipeline train) → new MLflow run
will show val CV + thresholded F1; evaluate updates test metrics.json.
Compare both to this baseline.
