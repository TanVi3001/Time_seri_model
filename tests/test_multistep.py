from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from copper_forecasting.data import prepare_data
from copper_forecasting.multistep import (
    direct_predict,
    evaluate_by_horizon,
    iterative_predict,
    run_multistep_comparison,
)


class IncrementByOneModel:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def predict(self, windows: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.inputs.append(windows.copy())
        return windows[:, -1, :1] + 1.0


class ThreeStepDirectModel:
    def predict(self, windows: np.ndarray, verbose: int = 0) -> np.ndarray:
        return np.tile(np.array([[1.0, 2.0, 3.0]]), (len(windows), 1))


def make_copper_frame(n: int = 60) -> pd.DataFrame:
    step = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=n, freq="B"),
            "closed_copper_price": 3.0 + step * 0.01 + np.sin(step / 3.0) * 0.02,
        }
    )


def test_multistep_origins_stay_inside_test_with_complete_horizon() -> None:
    frame = make_copper_frame(23)
    prepared = prepare_data(
        frame,
        ["closed_copper_price"],
        "closed_copper_price",
        lookback=3,
        horizon=3,
        train_ratio=0.6,
        validation_ratio=0.2,
    )

    assert prepared.test.origin_indices.tolist() == [17, 18, 19]
    assert prepared.test.target_indices.tolist() == [[18, 19, 20], [19, 20, 21], [20, 21, 22]]
    assert np.all(prepared.test.target_indices >= prepared.bounds.validation_end)
    assert np.all(prepared.test.target_indices < prepared.bounds.total_rows)


def test_ims_feeds_its_own_predictions_into_the_next_window() -> None:
    model = IncrementByOneModel()
    initial_windows = np.array([[[1.0], [2.0], [3.0]], [[10.0], [20.0], [30.0]]])

    predictions = iterative_predict(model, initial_windows, horizon=3)

    np.testing.assert_array_equal(predictions, [[4.0, 5.0, 6.0], [31.0, 32.0, 33.0]])
    np.testing.assert_array_equal(model.inputs[1], [[[2.0], [3.0], [4.0]], [[20.0], [30.0], [31.0]]])


def test_direct_predict_returns_one_vector_per_origin() -> None:
    predictions = direct_predict(ThreeStepDirectModel(), np.zeros((4, 5, 1)))

    assert predictions.shape == (4, 3)
    np.testing.assert_array_equal(predictions[0], [1.0, 2.0, 3.0])


def test_evaluate_by_horizon_inverse_transforms_and_reports_aggregate_metrics() -> None:
    scaler = MinMaxScaler().fit(np.array([[0.0], [10.0]]))
    y_true = np.array([[0.2, 0.4], [0.5, 0.6]])
    y_pred = np.array([[0.3, 0.2], [0.5, 0.8]])

    by_horizon, aggregate = evaluate_by_horizon(y_true, y_pred, scaler)

    assert by_horizon["horizon"].tolist() == [1, 2]
    assert by_horizon["mae"].tolist() == pytest.approx([0.5, 2.0])
    assert by_horizon["mape"].tolist() == pytest.approx([25.0, 125.0 / 3.0])
    assert aggregate["mae"] == pytest.approx(1.25)
    assert aggregate["mse"] == pytest.approx(2.25)
    assert aggregate["rmse"] == pytest.approx(1.5)


def test_ims_and_dms_use_the_same_origins_and_write_horizon_metrics(tmp_path: Path) -> None:
    results = run_multistep_comparison(
        make_copper_frame(),
        output_dir=tmp_path,
        seeds=(42,),
        lookback=5,
        horizon=3,
        epochs=1,
        batch_size=8,
        patience=1,
    )

    assert [result.metadata["strategy"] for result in results] == ["IMS", "DMS"]
    ims, dms = results
    np.testing.assert_array_equal(ims.predictions["origin_index"], dms.predictions["origin_index"])
    np.testing.assert_array_equal(ims.predictions["target_index"], dms.predictions["target_index"])
    assert set(ims.predictions["horizon"]) == {1, 2, 3}
    assert len(ims.predictions) == len(dms.predictions)
    for result in results:
        assert result.metadata["seed"] == 42
        assert result.metadata["horizon"] == 3
        assert len(result.metadata["metrics_by_horizon"]) == 3
        dates = make_copper_frame()["date"]
        assert result.metadata["split_dates"] == {
            "train": {"start": dates.iloc[0].isoformat(), "end": dates.iloc[47].isoformat()},
            "validation": {
                "start": dates.iloc[48].isoformat(),
                "end": dates.iloc[53].isoformat(),
            },
            "test": {"start": dates.iloc[54].isoformat(), "end": dates.iloc[59].isoformat()},
        }
        assert (tmp_path / result.run_id / "metrics_by_horizon.csv").is_file()
        saved_metadata = pd.read_json(
            tmp_path / result.run_id / "metadata.json", typ="series"
        )
        assert saved_metadata["split_dates"] == result.metadata["split_dates"]
