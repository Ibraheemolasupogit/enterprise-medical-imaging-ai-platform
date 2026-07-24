"""Operating-point selection from validation data only."""

from __future__ import annotations

import numpy as np

from medical_imaging_platform.classification.metrics import compute_classification_metrics
from medical_imaging_platform.classification.models import ClassificationConfig


def _metric_float(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        return 0.0
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"Expected numeric metric {name}, got {type(value).__name__}")


def _metric_int(metrics: dict[str, object], name: str) -> int:
    value = metrics.get(name)
    if value is None:
        return 0
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"Expected integer metric {name}, got {type(value).__name__}")


def select_threshold(
    labels: list[int], probabilities: list[float], config: ClassificationConfig
) -> dict[str, object]:
    """Select a threshold from validation labels/probabilities only."""
    if config.threshold_method == "fixed":
        threshold = config.fixed_threshold
    else:
        candidates = sorted(set(np.linspace(0.0, 1.0, 101).tolist() + probabilities))
        scored = [
            (candidate, compute_classification_metrics(labels, probabilities, candidate))
            for candidate in candidates
        ]
        if config.threshold_method == "youden":
            threshold = max(
                scored,
                key=lambda item: _metric_float(item[1], "sensitivity")
                + _metric_float(item[1], "specificity")
                - 1.0,
            )[0]
        elif config.threshold_method == "minimum_sensitivity":
            feasible = [
                item
                for item in scored
                if _metric_float(item[1], "sensitivity") >= config.minimum_sensitivity
            ]
            threshold = max(feasible or scored, key=lambda item: item[0])[0]
        elif config.threshold_method == "minimum_npv":
            threshold = max(
                scored, key=lambda item: _metric_float(item[1], "negative_predictive_value")
            )[0]
        else:
            feasible = [
                item
                for item in scored
                if _metric_int(item[1], "false_positive_count") <= config.maximum_false_positives
            ]
            threshold = min(feasible or scored, key=lambda item: item[0])[0]
    metrics = compute_classification_metrics(labels, probabilities, float(threshold))
    return {
        "method": config.threshold_method,
        "selected_threshold": float(threshold),
        "validation_metrics": metrics,
        "constraints": {
            "minimum_sensitivity": config.minimum_sensitivity,
            "maximum_false_positives": config.maximum_false_positives,
        },
        "source_split": "validation",
    }
