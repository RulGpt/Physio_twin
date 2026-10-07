from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .parameter_recovery import run_parameter_recovery
from .state_space import StateSpaceConfig


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

HISTORY_LENGTHS = (
    30,
    60,
    100,
    180,
    240,
    500,
    1000,
)

SEEDS = (
    42,
    43,
    44,
    45,
    46,
)

MAXITER = 600


# ---------------------------------------------------------------------------
# Synthetic reference configuration
# ---------------------------------------------------------------------------

def make_reference_config() -> StateSpaceConfig:
    """Return the fixed synthetic configuration used for M2.2."""

    return StateSpaceConfig(
        A=np.array(
            [
                [0.70, 0.05, 0.03],
                [0.03, 0.75, 0.02],
                [0.02, 0.03, 0.60],
            ],
            dtype=float,
        ),
        B=np.array(
            [
                [0.10, 0.08, 0.05],
                [-0.04, -0.03, -0.02],
                [0.08, 0.12, 0.04],
            ],
            dtype=float,
        ),
        C=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        Q=np.diag(
            [0.01, 0.01, 0.02]
        ),
        R=np.diag(
            [0.04, 0.04]
        ),
        initial_state=np.array(
            [0.5, 0.5, 0.2],
            dtype=float,
        ),
        initial_covariance=np.diag(
            [0.10, 0.10, 0.20]
        ),
    )


# ---------------------------------------------------------------------------
# Single experiment matrix
# ---------------------------------------------------------------------------

def run_experiment_matrix(
    config: StateSpaceConfig | None = None,
    history_lengths: tuple[int, ...] = HISTORY_LENGTHS,
    seeds: tuple[int, ...] = SEEDS,
    maxiter: int = MAXITER,
) -> pd.DataFrame:
    """Run all history-length × seed combinations.

    Returns one row per experiment.
    """

    if config is None:
        config = make_reference_config()

    rows: list[dict] = []

    total_runs = len(history_lengths) * len(seeds)
    completed = 0

    for days in history_lengths:
        for seed in seeds:
            completed += 1

            print(
                f"[{completed:02d}/{total_runs:02d}] "
                f"days={days}, seed={seed}"
            )

            metrics = run_parameter_recovery(
                config=config,
                days=days,
                seed=seed,
                maxiter=maxiter,
            )

            row = asdict(metrics)

            rows.append(row)

            print(
                f"    converged={metrics.converged} "
                f"iterations={metrics.optimizer_iterations} "
                f"total_rmse={metrics.total_parameter_rmse:.6f}"
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_experiments(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate experiment results by history length."""

    summary = (
        results
        .groupby("days")
        .agg(
            n_runs=("seed", "count"),

            convergence_rate=(
                "converged",
                "mean",
            ),

            mean_total_parameter_rmse=(
                "total_parameter_rmse",
                "mean",
            ),

            std_total_parameter_rmse=(
                "total_parameter_rmse",
                "std",
            ),

            median_total_parameter_rmse=(
                "total_parameter_rmse",
                "median",
            ),

            min_total_parameter_rmse=(
                "total_parameter_rmse",
                "min",
            ),

            max_total_parameter_rmse=(
                "total_parameter_rmse",
                "max",
            ),

            mean_A_rmse=(
                "A_rmse",
                "mean",
            ),

            mean_B_rmse=(
                "B_rmse",
                "mean",
            ),

            mean_Q_rmse=(
                "Q_rmse",
                "mean",
            ),

            mean_R_rmse=(
                "R_rmse",
                "mean",
            ),

            mean_open_loop_observation_rmse=(
                "open_loop_observation_rmse",
                "mean",
            ),

            mean_spectral_radius=(
                "spectral_radius",
                "mean",
            ),

            mean_log_likelihood=(
                "log_likelihood",
                "mean",
            ),

            mean_optimizer_iterations=(
                "optimizer_iterations",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run M2.2 and save raw + summary results."""

    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data"
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        data_dir
        / "m2_2_parameter_recovery.csv"
    )

    summary_path = (
        data_dir
        / "m2_2_parameter_recovery_summary.csv"
    )

    config = make_reference_config()

    print("=" * 72)
    print("PHYSIO-TWIN M2.2 PARAMETER RECOVERY EXPERIMENT")
    print("=" * 72)

    print()
    print("History lengths:")
    print(HISTORY_LENGTHS)

    print()
    print("Seeds:")
    print(SEEDS)

    print()
    print(f"Maximum optimizer iterations: {MAXITER}")
    print()

    results = run_experiment_matrix(
        config=config,
        history_lengths=HISTORY_LENGTHS,
        seeds=SEEDS,
        maxiter=MAXITER,
    )

    summary = summarize_experiments(
        results
    )

    results.to_csv(
        raw_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("M2.2 EXPERIMENT COMPLETE")
    print("=" * 72)

    print()
    print(f"Raw results:")
    print(raw_path)

    print()
    print(f"Summary:")
    print(summary_path)

    print()
    print("SUMMARY")
    print("-" * 72)

    display_columns = [
        "days",
        "n_runs",
        "convergence_rate",
        "mean_total_parameter_rmse",
        "std_total_parameter_rmse",
        "median_total_parameter_rmse",
        "mean_A_rmse",
        "mean_B_rmse",
        "mean_Q_rmse",
        "mean_R_rmse",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()