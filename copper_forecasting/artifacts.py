"""Persistence helpers for experiment results."""

from __future__ import annotations

import json
from pathlib import Path

from .experiments import RunResult


def save_run_result(result: RunResult, output_dir: Path) -> Path:
    """Write one result's JSON metadata, metrics, and prediction CSV."""

    run_dir = Path(output_dir) / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result.predictions.to_csv(run_dir / "predictions.csv", index=False)
    return run_dir
