"""Iterative and direct multi-step forecasts with shared test origins."""

from __future__ import annotations

import platform
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

from .data import PreparedData, prepare_data
from .experiments import RunResult, TARGET_COLUMN
from .metrics import evaluate_forecast
from .models import build_direct_gru, build_one_step_model, set_seed
from .training import fit_model


def iterative_predict(
    model: Any, initial_windows: np.ndarray, horizon: int
) -> np.ndarray:
    """Forecast repeatedly, feeding each scaled prediction into the next window."""

    windows = np.asarray(initial_windows, dtype=np.float32).copy()
    if windows.ndim != 3 or windows.shape[0] == 0 or windows.shape[1] == 0:
        raise ValueError("initial_windows must be a non-empty [origins, lookback, features] array")
    if windows.shape[2] != 1:
        raise ValueError("iterative forecasting supports one input feature")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")

    forecasts = np.empty((len(windows), int(horizon)), dtype=np.float32)
    for step in range(int(horizon)):
        prediction = np.asarray(model.predict(windows, verbose=0), dtype=np.float32)
        if prediction.shape == (len(windows), 1):
            prediction = prediction[:, 0]
        elif prediction.shape != (len(windows),):
            raise ValueError("one-step model must return exactly one value per origin")
        forecasts[:, step] = prediction
        next_value = prediction.reshape(-1, 1, 1)
        windows = np.concatenate((windows[:, 1:, :], next_value), axis=1)
    return forecasts


def direct_predict(model: Any, initial_windows: np.ndarray) -> np.ndarray:
    """Return the direct model's complete horizon vector for every origin."""

    windows = np.asarray(initial_windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[0] == 0 or windows.shape[1] == 0:
        raise ValueError("initial_windows must be a non-empty [origins, lookback, features] array")
    predictions = np.asarray(model.predict(windows, verbose=0), dtype=float)
    if predictions.ndim != 2 or predictions.shape[0] != windows.shape[0]:
        raise ValueError("direct model must return a [origins, horizon] matrix")
    if predictions.shape[1] == 0 or not np.isfinite(predictions).all():
        raise ValueError("direct model must return finite predictions for at least one lead")
    return predictions


def evaluate_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_scaler: Any,
    zero_tol: float = 1e-8,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Inverse-transform multi-step arrays and score each lead and all pairs."""

    actual_scaled = np.asarray(y_true, dtype=float)
    prediction_scaled = np.asarray(y_pred, dtype=float)
    if actual_scaled.ndim != 2 or actual_scaled.shape != prediction_scaled.shape:
        raise ValueError("y_true and y_pred must have the same [origins, horizon] shape")
    if actual_scaled.size == 0:
        raise ValueError("multi-step arrays must not be empty")
    actual = target_scaler.inverse_transform(actual_scaled.reshape(-1, 1)).reshape(
        actual_scaled.shape
    )
    prediction = target_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).reshape(prediction_scaled.shape)

    rows: list[dict[str, int | float | None]] = []
    for column in range(actual.shape[1]):
        row: dict[str, int | float | None] = {"horizon": column + 1}
        row.update(evaluate_forecast(actual[:, column], prediction[:, column], zero_tol))
        rows.append(row)
    by_horizon = pd.DataFrame(rows)
    aggregate = evaluate_forecast(actual.reshape(-1), prediction.reshape(-1), zero_tol)
    return by_horizon, aggregate


def _metadata_date(value: Any) -> str | int | float | None:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    return value if isinstance(value, (str, int, float)) else None


def _metrics_records(frame: pd.DataFrame) -> list[dict[str, int | float | None]]:
    records: list[dict[str, int | float | None]] = []
    for _, row in frame.iterrows():
        record: dict[str, int | float | None] = {"horizon": int(row["horizon"])}
        for name in ("mse", "rmse", "mae", "mape"):
            value = row[name]
            record[name] = float(value) if pd.notna(value) else None
        records.append(record)
    return records


def _run_result(
    *,
    strategy: str,
    model: tf.keras.Model,
    prepared_data: PreparedData,
    seed: int,
    outcome: Any,
    epochs: int,
    batch_size: int,
    patience: int,
    predictions_scaled: np.ndarray,
    inference_seconds: float,
) -> tuple[RunResult, pd.DataFrame]:
    test = prepared_data.test
    by_horizon, aggregate = evaluate_by_horizon(
        test.y, predictions_scaled, prepared_data.target_scaler
    )
    actual = prepared_data.target_scaler.inverse_transform(test.y.reshape(-1, 1)).reshape(
        test.y.shape
    )
    prediction = prepared_data.target_scaler.inverse_transform(
        np.asarray(predictions_scaled).reshape(-1, 1)
    ).reshape(predictions_scaled.shape)
    target_indices = test.target_indices
    origins = np.repeat(test.origin_indices, prepared_data.horizon)
    flattened_indices = target_indices.reshape(-1)
    dates = np.asarray(prepared_data.dates)[flattened_indices]
    date_start = _metadata_date(prepared_data.dates[0])
    date_end = _metadata_date(prepared_data.dates[-1])
    predictions = pd.DataFrame(
        {
            "date": dates,
            "origin_index": origins,
            "target_index": flattened_indices,
            "horizon": np.tile(np.arange(1, prepared_data.horizon + 1), len(test.X)),
            "actual": actual.reshape(-1),
            "prediction": prediction.reshape(-1),
            "residual": (prediction - actual).reshape(-1),
        }
    )
    run_id = f"gru_{strategy.lower()}_h{prepared_data.horizon}_seed{seed}"
    scaler = prepared_data.target_scaler
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "model": "GRU",
        "strategy": strategy,
        "seed": int(seed),
        "dataset_rows": int(prepared_data.bounds.total_rows),
        "dataset_source": prepared_data.dataset_source or "provided DataFrame",
        "date_start": date_start,
        "date_end": date_end,
        "dataset_version": {
            "rows": int(prepared_data.bounds.total_rows),
            "date_start": date_start,
            "date_end": date_end,
        },
        "input_columns": list(prepared_data.input_columns),
        "target_column": prepared_data.target_column,
        "target_unit": "USD per pound",
        "lookback": int(prepared_data.lookback),
        "horizon": int(prepared_data.horizon),
        "split_bounds": {
            "train_end": int(prepared_data.bounds.train_end),
            "validation_end": int(prepared_data.bounds.validation_end),
            "total_rows": int(prepared_data.bounds.total_rows),
        },
        "scaler": {
            "type": "MinMaxScaler",
            "fit_rows": [0, int(prepared_data.bounds.train_end)],
            "target_data_min": np.asarray(scaler.data_min_, dtype=float).tolist(),
            "target_data_max": np.asarray(scaler.data_max_, dtype=float).tolist(),
            "feature_data_min": np.asarray(
                prepared_data.feature_scaler.data_min_, dtype=float
            ).tolist(),
            "feature_data_max": np.asarray(
                prepared_data.feature_scaler.data_max_, dtype=float
            ).tolist(),
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
        "parameter_count": int(model.count_params()),
        "metrics": aggregate,
        "metrics_by_horizon": _metrics_records(by_horizon),
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    result = RunResult(
        run_id=run_id,
        metadata=metadata,
        metrics=aggregate,
        predictions=predictions,
    )
    return result, by_horizon


def run_multistep_comparison(
    frame: pd.DataFrame,
    output_dir: str | Path,
    seeds: tuple[int, ...] = (42,),
    lookback: int = 30,
    horizon: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
) -> list[RunResult]:
    """Train IMS and direct multi-output GRUs on the same copper-only split."""

    from .artifacts import save_run_result

    one_step_data = prepare_data(
        frame,
        input_columns=(TARGET_COLUMN,),
        target_column=TARGET_COLUMN,
        lookback=lookback,
        horizon=1,
    )
    direct_data = prepare_data(
        frame,
        input_columns=(TARGET_COLUMN,),
        target_column=TARGET_COLUMN,
        lookback=lookback,
        horizon=horizon,
    )
    if one_step_data.bounds != direct_data.bounds:
        raise RuntimeError("IMS and DMS must share identical split boundaries")
    one_step_windows = {
        int(origin): window
        for origin, window in zip(
            one_step_data.test.origin_indices, one_step_data.test.X, strict=True
        )
    }
    try:
        common_windows = np.stack(
            [one_step_windows[int(origin)] for origin in direct_data.test.origin_indices]
        )
    except KeyError as exc:
        raise RuntimeError("DMS test origins must also exist in the one-step test set") from exc

    results: list[RunResult] = []
    for seed in seeds:
        set_seed(seed)
        ims_model = build_one_step_model("GRU", (lookback, 1))
        ims_outcome = fit_model(
            ims_model,
            one_step_data.train.X,
            one_step_data.train.y,
            one_step_data.validation.X,
            one_step_data.validation.y,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        started = time.perf_counter()
        ims_prediction = iterative_predict(ims_outcome.model, common_windows, horizon)
        ims_inference_seconds = time.perf_counter() - started
        ims_result, ims_by_horizon = _run_result(
            strategy="IMS",
            model=ims_outcome.model,
            prepared_data=direct_data,
            seed=seed,
            outcome=ims_outcome,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            predictions_scaled=ims_prediction,
            inference_seconds=ims_inference_seconds,
        )
        ims_dir = save_run_result(ims_result, Path(output_dir))
        ims_by_horizon.to_csv(ims_dir / "metrics_by_horizon.csv", index=False)
        results.append(ims_result)

        set_seed(seed)
        dms_model = build_direct_gru((lookback, 1), horizon=horizon)
        dms_outcome = fit_model(
            dms_model,
            direct_data.train.X,
            direct_data.train.y,
            direct_data.validation.X,
            direct_data.validation.y,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        started = time.perf_counter()
        dms_prediction = direct_predict(dms_outcome.model, direct_data.test.X)
        dms_inference_seconds = time.perf_counter() - started
        dms_result, dms_by_horizon = _run_result(
            strategy="DMS",
            model=dms_outcome.model,
            prepared_data=direct_data,
            seed=seed,
            outcome=dms_outcome,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            predictions_scaled=dms_prediction,
            inference_seconds=dms_inference_seconds,
        )
        dms_dir = save_run_result(dms_result, Path(output_dir))
        dms_by_horizon.to_csv(dms_dir / "metrics_by_horizon.csv", index=False)
        results.append(dms_result)
    return results
