from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from copper_forecasting.data import prepare_data, read_dataset, split_bounds


def make_frame(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "copper": np.arange(n, dtype=float),
            "other": np.arange(n, dtype=float) * 10,
        }
    )


def test_read_dataset_parses_dates_and_sorts_rows(tmp_path: Path) -> None:
    frame = make_frame(3).iloc[[2, 0, 1]]
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)

    loaded = read_dataset(path)

    assert loaded["date"].tolist() == list(pd.date_range("2024-01-01", periods=3, freq="D"))
    assert loaded["copper"].tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
        lambda frame: frame.assign(copper=[0.0, np.nan, *range(2, 20)]),
        lambda frame: frame.assign(copper=[0.0, np.inf, *range(2, 20)]),
    ],
    ids=["duplicate-date", "missing-price", "infinite-price"],
)
def test_read_dataset_rejects_duplicate_dates_and_invalid_values(
    tmp_path: Path, mutate
) -> None:
    path = tmp_path / "invalid.csv"
    mutate(make_frame()).to_csv(path, index=False)

    with pytest.raises(ValueError):
        read_dataset(path)


def test_split_bounds_use_chronological_floor_boundaries() -> None:
    assert split_bounds(23) == split_bounds(23, 0.8, 0.1)
    bounds = split_bounds(23)

    assert (bounds.train_end, bounds.validation_end, bounds.total_rows) == (18, 20, 23)


def test_scalers_fit_train_rows_only() -> None:
    frame = make_frame()
    frame.loc[16:, "copper"] = 10_000
    frame.loc[16:, "other"] = 20_000

    prepared = prepare_data(frame, ["copper", "other"], "copper", lookback=3)

    assert prepared.feature_scaler.data_max_.tolist() == [15.0, 150.0]
    assert prepared.target_scaler.data_max_.tolist() == [15.0]


def test_prepare_data_sorts_unsorted_dataframe_before_splitting() -> None:
    frame = make_frame().iloc[[*range(19, -1, -1)]]

    prepared = prepare_data(frame, ["copper", "other"], "copper", lookback=3)

    assert prepared.dates.tolist() == pd.date_range(
        "2024-01-01", periods=20, freq="D"
    ).to_numpy().tolist()
    assert prepared.feature_scaler.data_max_.tolist() == [15.0, 150.0]
    assert prepared.train.target_indices[:, 0].max() == 15


def test_prepare_data_rejects_duplicate_dates() -> None:
    frame = pd.concat([make_frame(), make_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate dates"):
        prepare_data(frame, ["copper", "other"], "copper", lookback=3)


def test_windows_are_assigned_by_target_indices() -> None:
    prepared = prepare_data(make_frame(), ["copper", "other"], "copper", lookback=3)

    assert prepared.train.target_indices[:, 0].min() == 3
    assert prepared.train.target_indices[:, 0].max() == 15
    assert prepared.validation.target_indices[:, 0].tolist() == [16, 17]
    assert prepared.test.target_indices[:, 0].tolist() == [18, 19]
    assert np.array_equal(prepared.validation.origin_indices, [15, 16])


def test_windows_include_only_values_at_or_before_origin() -> None:
    frame = make_frame(12)
    prepared = prepare_data(
        frame,
        ["copper"],
        "copper",
        lookback=3,
        train_ratio=0.5,
        validation_ratio=0.25,
    )

    test_row = int(prepared.test.origin_indices[0])
    actual_window = prepared.test.X[0, :, 0]
    expected_raw_window = frame.loc[test_row - 2 : test_row, "copper"].to_numpy()
    expected_scaled_window = prepared.feature_scaler.transform(
        expected_raw_window.reshape(-1, 1)
    )[:, 0]

    assert prepared.test.target_indices[0, 0] == test_row + 1
    np.testing.assert_allclose(actual_window, expected_scaled_window)


def test_direct_multioutput_window_keeps_full_horizon_inside_each_split() -> None:
    prepared = prepare_data(make_frame(), ["copper"], "copper", lookback=3, horizon=2)

    assert prepared.validation.target_indices.tolist() == [[16, 17]]
    assert prepared.test.target_indices.tolist() == [[18, 19]]
    assert np.all(prepared.train.target_indices < prepared.bounds.train_end)
