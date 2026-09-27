from __future__ import annotations

import numpy as np
import pytest

from copper_forecasting.metrics import evaluate_forecast


def test_perfect_forecast_has_zero_errors() -> None:
    scores = evaluate_forecast([2.0, 4.0], [2.0, 4.0])

    assert scores == {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0}


def test_metrics_match_hand_calculated_values() -> None:
    scores = evaluate_forecast([1.0, 2.0, 4.0], [2.0, 2.0, 1.0])

    assert scores["mse"] == pytest.approx(10.0 / 3.0)
    assert scores["rmse"] == pytest.approx(np.sqrt(10.0 / 3.0))
    assert scores["mae"] == pytest.approx(4.0 / 3.0)
    assert scores["mape"] == pytest.approx(7.0 / 12.0 * 100.0)


def test_metrics_reject_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="same shape"):
        evaluate_forecast([1.0, 2.0], [1.0])


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([1.0, np.nan], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, np.inf]),
    ],
    ids=["nan-actual", "infinite-prediction"],
)
def test_metrics_reject_non_finite_values(y_true, y_pred) -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_forecast(y_true, y_pred)


def test_mape_is_unavailable_for_near_zero_actuals() -> None:
    scores = evaluate_forecast([1.0, 1e-10], [1.1, 1e-10])

    assert scores["mape"] is None
    assert scores["mae"] == pytest.approx(0.05)


def test_metrics_reject_mismatched_shapes_even_when_element_counts_match() -> None:
    with pytest.raises(ValueError, match="same shape"):
        evaluate_forecast([[1.0, 2.0]], [[1.0], [2.0]])
