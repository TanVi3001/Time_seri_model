"""Headless plots for the saved experiment metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiments import RunResult


def plot_architecture_comparison(
    results: Sequence[RunResult], output_path: str | Path
) -> Path:
    """Plot test RMSE and MAE for each one-step architecture run."""

    if not results:
        raise ValueError("at least one one-step result is required")
    labels = [f"{result.metadata['model']}\nseed={result.metadata['seed']}" for result in results]
    rmse = [float(result.metrics["rmse"]) for result in results]
    mae = [float(result.metrics["mae"]) for result in results]
    x = np.arange(len(results))
    width = 0.38

    figure, axis = plt.subplots(figsize=(max(7.5, 1.5 * len(results)), 4.5))
    axis.bar(x - width / 2, rmse, width, label="RMSE")
    axis.bar(x + width / 2, mae, width, label="MAE")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Error (copper price unit)")
    axis.set_title("One-step copper-price test error")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def plot_multistep_errors(
    by_horizon: pd.DataFrame, output_path: str | Path
) -> Path:
    """Plot average test RMSE by lead time for IMS and DMS."""

    required = {"strategy", "horizon", "rmse"}
    if not required.issubset(by_horizon.columns) or by_horizon.empty:
        raise ValueError("by_horizon must contain strategy, horizon, and rmse rows")
    grouped = by_horizon.groupby(["strategy", "horizon"], as_index=False)["rmse"].mean()

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for strategy, rows in grouped.groupby("strategy", sort=False):
        rows = rows.sort_values("horizon")
        axis.plot(rows["horizon"], rows["rmse"], marker="o", label=strategy)
    axis.set_xlabel("Forecast lead (trading sessions)")
    axis.set_ylabel("RMSE (copper price unit)")
    axis.set_title("Multi-step copper-price RMSE by horizon")
    axis.set_xticks(sorted(grouped["horizon"].unique()))
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
