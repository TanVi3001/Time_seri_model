"""Dataset validation, chronological splitting, scaling, and window creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class SplitBounds:
    """Raw-row split boundaries using end-exclusive indices."""

    train_end: int
    validation_end: int
    total_rows: int


@dataclass(frozen=True)
class WindowBatch:
    """Sliding windows and their raw-row origin and target indices."""

    X: np.ndarray
    y: np.ndarray
    origin_indices: np.ndarray
    target_indices: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    """Chronological train, validation, and test windows with fitted scalers."""

    train: WindowBatch
    validation: WindowBatch
    test: WindowBatch
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    bounds: SplitBounds
    dates: np.ndarray
    input_columns: tuple[str, ...]
    target_column: str
    lookback: int
    horizon: int


def read_dataset(path: Path) -> pd.DataFrame:
    """Read and validate a date-indexed numeric CSV, returning dates ascending."""

    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError("Dataset must contain a 'date' column")
    if len(frame.columns) < 2:
        raise ValueError("Dataset must contain at least one numeric value column")

    try:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Dataset contains an invalid date") from exc
    if frame["date"].isna().any():
        raise ValueError("Dataset contains a missing date")
    if frame["date"].duplicated().any():
        raise ValueError("Dataset contains duplicate dates")

    value_columns = [column for column in frame.columns if column != "date"]
    for column in value_columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dataset column {column!r} must be numeric") from exc
    numeric_values = frame[value_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Dataset contains missing or non-finite values")

    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def split_bounds(
    n_rows: int, train_ratio: float = 0.8, validation_ratio: float = 0.1
) -> SplitBounds:
    """Return chronological raw-row boundaries, with floor rounding."""

    if n_rows < 3:
        raise ValueError("At least three rows are required for train/validation/test")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio and validation_ratio must sum to less than 1")

    train_end = int(np.floor(n_rows * train_ratio))
    validation_end = int(np.floor(n_rows * (train_ratio + validation_ratio)))
    if train_end < 1 or validation_end <= train_end or validation_end >= n_rows:
        raise ValueError("Ratios must leave at least one row in each split")
    return SplitBounds(train_end, validation_end, n_rows)


def prepare_data(
    frame: pd.DataFrame,
    input_columns: Sequence[str],
    target_column: str,
    lookback: int,
    horizon: int = 1,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> PreparedData:
    """Fit MinMax scalers on raw training rows and create split-safe windows."""

    inputs = tuple(input_columns)
    if not inputs:
        raise ValueError("At least one input column is required")
    if len(set(inputs)) != len(inputs):
        raise ValueError("input_columns must not contain duplicates")
    missing = sorted((set(inputs) | {target_column}) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required data columns: {', '.join(missing)}")
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive integers")

    try:
        feature_values = frame.loc[:, inputs].to_numpy(dtype=float)
        target_values = frame.loc[:, [target_column]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Input and target columns must contain numeric values") from exc
    if not np.isfinite(feature_values).all() or not np.isfinite(target_values).all():
        raise ValueError("Input and target columns must contain only finite values")

    bounds = split_bounds(len(frame), train_ratio, validation_ratio)
    if "date" in frame.columns:
        try:
            dates = pd.to_datetime(frame["date"], errors="raise").to_numpy()
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset contains an invalid date") from exc
    else:
        dates = np.arange(len(frame), dtype=int)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(feature_values[: bounds.train_end])
    target_scaler.fit(target_values[: bounds.train_end])
    scaled_features = feature_scaler.transform(feature_values)

    def build_batch(split_start: int, split_end: int) -> WindowBatch:
        min_origin = max(lookback - 1, split_start - 1)
        max_origin = min(split_end - horizon - 1, len(frame) - horizon - 1)
        origins = np.arange(min_origin, max_origin + 1, dtype=int)
        if origins.size == 0:
            raise ValueError(
                f"No complete lookback={lookback}, horizon={horizon} windows in split"
            )

        X = np.stack(
            [scaled_features[origin - lookback + 1 : origin + 1] for origin in origins]
        )
        target_indices = origins[:, np.newaxis] + np.arange(1, horizon + 1)
        raw_targets = np.stack(
            [target_values[indices, 0] for indices in target_indices]
        )
        y = target_scaler.transform(raw_targets.reshape(-1, 1)).reshape(-1, horizon)
        return WindowBatch(
            X=X.astype(np.float32),
            y=y.astype(np.float32),
            origin_indices=origins,
            target_indices=target_indices,
        )

    train = build_batch(0, bounds.train_end)
    validation = build_batch(bounds.train_end, bounds.validation_end)
    test = build_batch(bounds.validation_end, bounds.total_rows)
    return PreparedData(
        train=train,
        validation=validation,
        test=test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        bounds=bounds,
        dates=dates,
        input_columns=inputs,
        target_column=target_column,
        lookback=lookback,
        horizon=horizon,
    )
