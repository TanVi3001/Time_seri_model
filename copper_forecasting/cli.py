"""Command-line entry point for repeatable copper forecasting experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .artifacts import save_run_result
from .data import prepare_data, read_dataset
from .experiments import INPUT_COLUMNS, MODEL_NAMES, TARGET_COLUMN, run_one_step_experiment
from .multistep import run_multistep_comparison
from .plotting import plot_architecture_comparison, plot_multistep_errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run copper-price forecasting experiments.")
    parser.add_argument("--dataset", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Artifact directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="One or more random seeds")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum training epochs per model"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="One-step architectures to compare",
    )
    parser.add_argument("--lookback", type=int, default=30, help="Input window length")
    parser.add_argument("--horizon", type=int, default=5, help="Multi-step forecast horizon")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience")
    parser.add_argument(
        "--skip-multistep",
        action="store_true",
        help="Run the one-step architecture comparison only",
    )
    return parser


def _one_step_metrics(results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": result.run_id,
                "model": result.metadata["model"],
                "strategy": result.metadata["strategy"],
                "seed": result.metadata["seed"],
                **result.metrics,
                "parameter_count": result.metadata["parameter_count"],
                "best_epoch": result.metadata["best_epoch"],
                "epochs_ran": result.metadata["epochs_ran"],
                "training_seconds": result.metadata["training_seconds"],
                "inference_seconds": result.metadata["inference_seconds"],
            }
            for result in results
        ]
    )


def _multistep_metrics(results) -> pd.DataFrame:
    rows = []
    for result in results:
        for lead in result.metadata["metrics_by_horizon"]:
            rows.append(
                {
                    "run_id": result.run_id,
                    "model": result.metadata["model"],
                    "strategy": result.metadata["strategy"],
                    "seed": result.metadata["seed"],
                    **lead,
                    "parameter_count": result.metadata["parameter_count"],
                    "training_seconds": result.metadata["training_seconds"],
                    "inference_seconds": result.metadata["inference_seconds"],
                }
            )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise SystemExit("epochs and batch-size must be positive; patience cannot be negative")
    if args.lookback < 1 or args.horizon < 1:
        raise SystemExit("lookback and horizon must be positive")

    frame = read_dataset(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_data = prepare_data(
        frame,
        input_columns=INPUT_COLUMNS,
        target_column=TARGET_COLUMN,
        lookback=args.lookback,
        horizon=1,
    )
    one_step_results = []
    for seed in args.seeds:
        for model_name in dict.fromkeys(args.models):
            result = run_one_step_experiment(
                model_name,
                prepared_data,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
            )
            save_run_result(result, args.output_dir)
            one_step_results.append(result)

    one_step_metrics = _one_step_metrics(one_step_results)
    one_step_metrics.to_csv(args.output_dir / "one_step_metrics.csv", index=False)
    plot_architecture_comparison(
        one_step_results, args.output_dir / "architecture_comparison.png"
    )

    if not args.skip_multistep:
        multistep_results = run_multistep_comparison(
            frame,
            output_dir=args.output_dir,
            seeds=tuple(args.seeds),
            lookback=args.lookback,
            horizon=args.horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        multistep_metrics = _multistep_metrics(multistep_results)
        multistep_metrics.to_csv(
            args.output_dir / "multistep_metrics_by_horizon.csv", index=False
        )
        plot_multistep_errors(
            multistep_metrics, args.output_dir / "multistep_errors.png"
        )

    print(f"Dataset: {len(frame)} rows, {frame['date'].min().date()} to {frame['date'].max().date()}")
    print(f"One-step runs: {len(one_step_results)}")
    if not args.skip_multistep:
        print(f"IMS/DMS runs: {len(multistep_results)}")
    print(f"Artifacts: {args.output_dir.resolve()}")
    return 0
