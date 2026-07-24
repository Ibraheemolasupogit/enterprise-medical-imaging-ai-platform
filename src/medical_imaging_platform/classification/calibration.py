"""Probability calibration using validation data only."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from medical_imaging_platform.classification.metrics import calibration_slope_intercept
from medical_imaging_platform.classification.models import ClassificationConfig


def _parameters(artefact: dict[str, object]) -> dict[str, object]:
    params = artefact.get("parameters")
    if not isinstance(params, dict):
        return {}
    return params


def _parameter_float(params: dict[str, object], name: str, default: float) -> float:
    value = params.get(name, default)
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"Calibration parameter must be numeric: {name}")


def fit_calibration(
    logits: list[float], labels: list[int], config: ClassificationConfig
) -> dict[str, object]:
    """Fit calibration artefact from validation logits only."""
    raw = sigmoid(np.asarray(logits, dtype=float))
    if config.calibration_method == "none" or len(set(labels)) < 2:
        return {
            "method": "none",
            "status": "fallback_insufficient_validation_data",
            "parameters": {},
            "diagnostics": calibration_slope_intercept(labels, raw.tolist()),
        }
    y = np.asarray(labels, dtype=int)
    if config.calibration_method == "isotonic" and len(labels) >= 8:
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(raw, y)
        return {
            "method": "isotonic",
            "status": "fitted",
            "parameters": {
                "x_thresholds": model.X_thresholds_.tolist(),
                "y_thresholds": model.y_thresholds_.tolist(),
            },
            "diagnostics": calibration_slope_intercept(labels, model.predict(raw).tolist()),
        }
    lr = LogisticRegression(solver="lbfgs", random_state=config.random_seed)
    lr.fit(np.asarray(logits, dtype=float).reshape(-1, 1), y)
    calibrated = lr.predict_proba(np.asarray(logits, dtype=float).reshape(-1, 1))[:, 1]
    return {
        "method": "platt",
        "status": "fitted",
        "parameters": {
            "coef": float(lr.coef_[0][0]),
            "intercept": float(lr.intercept_[0]),
        },
        "diagnostics": calibration_slope_intercept(labels, calibrated.tolist()),
    }


def apply_calibration(logit: float, artefact: dict[str, object]) -> float:
    method = artefact.get("method")
    if method == "platt":
        params = _parameters(artefact)
        coef = _parameter_float(params, "coef", 1.0)
        intercept = _parameter_float(params, "intercept", 0.0)
        return float(sigmoid(np.asarray([coef * logit + intercept]))[0])
    if method == "isotonic":
        params = _parameters(artefact)
        x = np.asarray(params.get("x_thresholds", [0.0, 1.0]), dtype=np.float64)
        y = np.asarray(params.get("y_thresholds", [0.0, 1.0]), dtype=np.float64)
        return float(np.interp(float(sigmoid(np.asarray([logit]))[0]), x, y))
    return float(sigmoid(np.asarray([logit]))[0])


def sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = 1.0 / (1.0 + np.exp(-values))
    return np.asarray(result, dtype=np.float64)
