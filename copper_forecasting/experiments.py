"""One-step model comparison runner and run metadata."""

from __future__ import annotations

import platform
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

from .data import PreparedData, prepare_data, split_date_ranges
from .metrics import evaluate_forecast
from .models import build_one_step_model, set_seed
from .training import fit_model

INPUT_COLUMNS = (
    "closed_copper_price",
    "close_wti_oil",
    "close_gold",
    "close_silver",
)
TARGET_COLUMN = "closed_copper_price"
MODEL_NAMES = ("SimpleRNN", "LSTM", "GRU", "Bi-LSTM")
COLUMN_UNITS = {
    "closed_copper_price": "USD per pound",
    "close_wti_oil": "USD per barrel",
    "close_gold": "USD per troy ounce",
    "close_silver": "USD per troy ounce",
}


@dataclass(frozen=True)
class RunResult:
    run_id: str
    metadata: dict[str, Any]
    metrics: dict[str, float | None]
    predictions: pd.DataFrame


def _scaler_metadata(scaler: Any) -> dict[str, list[float]]:
    return {
        name: np.asarray(getattr(scaler, name), dtype=float).tolist()
        for name in ("data_min_", "data_max_", "scale_", "min_")
    }


def _date_value(value: Any) -> str | int | float | None:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float)):
        return value
    return None


def run_one_step_experiment(
    model_name: str,
    prepared_data: PreparedData,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
) -> RunResult:
    """Train one model, score its original-unit test predictions, and record metadata."""

    set_seed(seed)
    model = build_one_step_model(
        model_name,
        input_shape=(prepared_data.lookback, len(prepared_data.input_columns)),
    )
    outcome = fit_model(
        model,
        prepared_data.train.X,
        prepared_data.train.y,
        prepared_data.validation.X,
        prepared_data.validation.y,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )

    test_batch = prepared_data.test
    inference_started = time.perf_counter()
    scaled_prediction = outcome.model.predict(test_batch.X, verbose=0)
    inference_seconds = time.perf_counter() - inference_started
    actual = prepared_data.target_scaler.inverse_transform(
        test_batch.y.reshape(-1, 1)
    ).reshape(-1)
    prediction = prepared_data.target_scaler.inverse_transform(
        np.asarray(scaled_prediction).reshape(-1, 1)
    ).reshape(-1)
    metrics = evaluate_forecast(actual, prediction)

    dates = np.asarray(prepared_data.dates)[test_batch.target_indices[:, 0]]
    predictions = pd.DataFrame(
        {
            "date": dates,
            "origin_index": test_batch.origin_indices,
            "target_index": test_batch.target_indices[:, 0],
            "actual": actual,
            "prediction": prediction,
            "residual": prediction - actual,
        }
    )
    run_id = f"{re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')}_one_step_seed{seed}"
    date_start = _date_value(prepared_data.dates[0])
    date_end = _date_value(prepared_data.dates[-1])
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "model": model_name,
        "strategy": "One-step",
        "seed": int(seed),
        "dataset_rows": int(prepared_data.bounds.total_rows),
        "dataset_source": prepared_data.dataset_source or "provided DataFrame",
        "dataset_version": {
            "rows": int(prepared_data.bounds.total_rows),
            "date_start": date_start,
            "date_end": date_end,
        },
        "date_start": date_start,
        "date_end": date_end,
        "input_columns": list(prepared_data.input_columns),
        "input_units": {
            column: COLUMN_UNITS.get(column, "as represented in source CSV")
            for column in prepared_data.input_columns
        },
        "target_column": prepared_data.target_column,
        "target_unit": COLUMN_UNITS.get(
            prepared_data.target_column, "as represented in source CSV"
        ),
        "lookback": int(prepared_data.lookback),
        "horizon": 1,
        "split_bounds": {
            "train_end": int(prepared_data.bounds.train_end),
            "validation_end": int(prepared_data.bounds.validation_end),
            "total_rows": int(prepared_data.bounds.total_rows),
        },
        "split_dates": split_date_ranges(prepared_data.dates, prepared_data.bounds),
        "scaler": {
            "type": "MinMaxScaler",
            "fit_rows": [0, int(prepared_data.bounds.train_end)],
            "feature": _scaler_metadata(prepared_data.feature_scaler),
            "target": _scaler_metadata(prepared_data.target_scaler),
        },
        "loss": "mse",
        "optimizer": {"name": "Adam", "learning_rate": 0.001},
        "batch_size": int(batch_size),
        "max_epochs": int(epochs),
        "patience": int(patience),
        "best_epoch": int(outcome.best_epoch),
        "epochs_ran": int(outcome.epochs_ran),
        "training_seconds": float(outcome.training_seconds),
        "inference_seconds": float(inference_seconds),
        "parameter_count": int(outcome.model.count_params()),
        "metrics": metrics,
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    return RunResult(
        run_id=run_id,
        metadata=metadata,
        metrics=metrics,
        predictions=predictions,
    )


def run_architecture_comparison(
    frame: pd.DataFrame,
    output_dir: str,
    seeds: tuple[int, ...] = (42,),
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
) -> list[RunResult]:
    """Run all four recurrent architectures on one shared prepared split."""

    from pathlib import Path

    from .artifacts import save_run_result

    prepared_data = prepare_data(
        frame,
        input_columns=INPUT_COLUMNS,
        target_column=TARGET_COLUMN,
        lookback=30,
        horizon=1,
    )
    results: list[RunResult] = []
    for seed in seeds:
        for model_name in MODEL_NAMES:
            result = run_one_step_experiment(
                model_name,
                prepared_data,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
            )
            save_run_result(result, Path(output_dir))
            results.append(result)
    return results
