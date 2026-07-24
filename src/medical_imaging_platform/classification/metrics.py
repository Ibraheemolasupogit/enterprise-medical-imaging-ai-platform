"""Binary classification metrics and calibration diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as sk_metrics


def compute_classification_metrics(
    labels: list[int], probabilities: list[float], threshold: float
) -> dict[str, object]:
    """Compute aggregate metrics for binary synthetic lesion-presence classification."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(np.count_nonzero((pred == 1) & (y == 1)))
    tn = int(np.count_nonzero((pred == 0) & (y == 0)))
    fp = int(np.count_nonzero((pred == 1) & (y == 0)))
    fn = int(np.count_nonzero((pred == 0) & (y == 1)))
    return {
        "sensitivity": _divide(tp, tp + fn),
        "recall": _divide(tp, tp + fn),
        "specificity": _divide(tn, tn + fp),
        "precision": _divide(tp, tp + fp),
        "positive_predictive_value": _divide(tp, tp + fp),
        "negative_predictive_value": _divide(tn, tn + fn),
        "f1": _divide(2 * tp, 2 * tp + fp + fn),
        "accuracy": _divide(tp + tn, len(y)),
        "auroc": _safe_auroc(y, p),
        "auprc": _safe_auprc(y, p),
        "brier_score": float(sk_metrics.brier_score_loss(y, p)) if len(y) else None,
        "expected_calibration_error": expected_calibration_error(y, p),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "false_positive_count": fp,
        "false_negative_count": fn,
    }


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 5
) -> float:
    if labels.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= lower) & (probabilities < upper)
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
            )
    return ece


def calibration_slope_intercept(
    labels: list[int], probabilities: list[float]
) -> dict[str, float | None]:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if len(set(labels)) < 2:
        return {"slope": None, "intercept": None}
    logits = np.log(p / (1 - p))
    slope, intercept = np.polyfit(logits, y, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def _safe_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    return float(sk_metrics.roc_auc_score(labels, probabilities))


def _safe_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    return float(sk_metrics.average_precision_score(labels, probabilities))


def _divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)
