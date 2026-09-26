from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from copper_forecasting.experiments import run_architecture_comparison
from copper_forecasting.metrics import evaluate_forecast


def make_market_frame(n: int = 60) -> pd.DataFrame:
    step = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "closed_copper_price": 3.0 + step * 0.01 + np.sin(step / 4.0) * 0.02,
            "close_wti_oil": 60.0 + step * 0.1,
            "close_gold": 1500.0 + step * 0.5,
            "close_silver": 18.0 + step * 0.03,
        }
    )


def test_architecture_comparison_uses_same_dates_original_units_and_saves_artifacts(
    tmp_path: Path,
) -> None:
    frame = make_market_frame()

    results = run_architecture_comparison(
        frame,
        output_dir=tmp_path,
        seeds=(42,),
        epochs=1,
        batch_size=32,
        patience=1,
    )

    assert [result.metadata["model"] for result in results] == [
        "SimpleRNN",
        "LSTM",
        "GRU",
        "Bi-LSTM",
    ]
    expected_dates = frame.loc[54:, "date"].tolist()
    for result in results:
        assert pd.to_datetime(result.predictions["date"]).tolist() == expected_dates
        expected_actual = frame.loc[result.predictions["target_index"], "closed_copper_price"].to_numpy()
        np.testing.assert_allclose(result.predictions["actual"], expected_actual)
        expected_metrics = evaluate_forecast(expected_actual, result.predictions["prediction"])
        for name, value in expected_metrics.items():
            assert result.metrics[name] == pytest.approx(value)
        assert result.metadata["seed"] == 42
        assert result.metadata["best_epoch"] == 1
        assert result.metadata["parameter_count"] > 0
        assert result.metadata["training_seconds"] >= 0.0
        assert result.metadata["inference_seconds"] >= 0.0

        run_dir = tmp_path / result.run_id
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        saved_predictions = pd.read_csv(run_dir / "predictions.csv")
        saved_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metadata["seed"] == 42
        assert len(saved_predictions) == len(expected_dates)
        assert saved_metrics == result.metrics
