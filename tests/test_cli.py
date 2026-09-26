from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_cli_smoke_writes_expected_artifacts(tmp_path: Path) -> None:
    step = np.arange(100, dtype=float)
    dataset = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods=100, freq="B"),
            "closed_copper_price": 3.0 + step * 0.01 + np.sin(step / 4.0) * 0.03,
            "close_wti_oil": 60.0 + step * 0.2,
            "close_gold": 1700.0 + step * 0.7,
            "close_silver": 21.0 + step * 0.05,
        }
    )
    dataset_path = tmp_path / "smoke_dataset.csv"
    output_dir = tmp_path / "results"
    dataset.to_csv(dataset_path, index=False)
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "copper_forecasting",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--seeds",
            "7",
            "--epochs",
            "1",
            "--models",
            "GRU",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert (output_dir / "one_step_metrics.csv").is_file()
    assert (output_dir / "multistep_metrics_by_horizon.csv").is_file()
    assert (output_dir / "architecture_comparison.png").stat().st_size > 0
    assert (output_dir / "multistep_errors.png").stat().st_size > 0
    for run_id in ("gru_one_step_seed7", "gru_ims_h5_seed7", "gru_dms_h5_seed7"):
        run_dir = output_dir / run_id
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "predictions.csv").is_file()
        assert metadata["seed"] == 7
    assert (output_dir / "gru_ims_h5_seed7" / "metrics_by_horizon.csv").is_file()
    assert (output_dir / "gru_dms_h5_seed7" / "metrics_by_horizon.csv").is_file()
