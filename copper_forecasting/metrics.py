"""Error metrics shared by the one-step and multi-step experiment runners."""

from __future__ import annotations

import numpy as np


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_tol: float = 1e-8,
) -> dict[str, float | None]:
    """Compute MSE, RMSE, MAE, and percentage MAPE on original-scale values.

    Inputs may be vectors or matching multi-output arrays. MAPE is unavailable
    when any actual value is at or below ``zero_tol`` in absolute value.
    """

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if actual.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true and y_pred must contain only finite values")
    if not np.isfinite(zero_tol) or zero_tol < 0:
        raise ValueError("zero_tol must be a finite, non-negative value")

    errors = predicted - actual
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    if np.any(np.abs(actual) <= zero_tol):
        mape: float | None = None
    else:
        mape = float(np.mean(np.abs(errors / actual)) * 100.0)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "mape": mape,
    }
