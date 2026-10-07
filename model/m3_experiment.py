from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .parameter_recovery import run_parameter_recovery
from .predictive_validation import (
    PredictionMetrics,
    chronological_split,
    persistence_predictions,
    state_space_one_step_predictions,
)
from .synthetic import make_synthetic_dataset
from .state_space import StateSpaceConfig


HISTORY_LENGTHS = (
    180,
    240,
    500,
)

SEEDS = (
    42,
    43,
    44,
    45,
    46,
)

TRAIN_RATIO = 0.75
MAXITER = 600


def make_reference_config() -> StateSpaceConfig:
    """Return the fixed synthetic configuration used for M3."""

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


def _metrics_to_dict(
    metrics: PredictionMetrics,
) -> dict:
    """Convert prediction metrics into a dictionary."""

    return asdict(metrics)


def evaluate_population_model(
    config: StateSpaceConfig,
    train_inputs: np.ndarray,
    train_observations: np.ndarray,
    test_inputs: np.ndarray,
    test_observations: np.ndarray,
) -> PredictionMetrics:
    """
    Evaluate the fixed population model on the held-out period.

    The model is not refitted using test observations.
    """

    combined_inputs = np.vstack(
        [
            train_inputs,
            test_inputs,
        ]
    )

    combined_observations = np.vstack(
        [
            train_observations,
            test_observations,
        ]
    )

    # Run filtering over the training history only.
    state = config.initial_state.copy()
    covariance = config.initial_covariance.copy()

    from .state_space import filter_step

    for t in range(len(train_observations)):

        state, covariance, _, _ = filter_step(
            state=state,
            covariance=covariance,
            inputs=train_inputs[t],
            observation=train_observations[t],
            config=config,
        )

    predictions = []
    targets = []

    for t in range(len(test_observations)):

        predicted_state = (
            config.A @ state
            + config.B @ test_inputs[t]
        )

        predicted_observation = (
            config.C @ predicted_state
        )

        target = test_observations[t]

        if (
            np.all(np.isfinite(predicted_observation))
            and np.all(np.isfinite(target))
        ):
            predictions.append(
                predicted_observation
            )
            targets.append(target)

        # Only after prediction do we reveal the
        # current test observation and update the state.
        state, covariance, _, _ = filter_step(
            state=state,
            covariance=covariance,
            inputs=test_inputs[t],
            observation=target,
            config=config,
        )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    from .predictive_validation import _calculate_metrics

    return _calculate_metrics(
        predictions,
        targets,
    )


def evaluate_persistence_model(
    train_observations: np.ndarray,
    test_observations: np.ndarray,
) -> PredictionMetrics:
    """
    Evaluate persistence on the held-out period.

    The first test prediction uses the final training observation.
    """

    if len(train_observations) < 1:
        raise ValueError(
            "Training observations cannot be empty"
        )

    if len(test_observations) < 1:
        raise ValueError(
            "Test observations cannot be empty"
        )

    previous = train_observations[-1]

    predictions = []
    targets = []

    for observation in test_observations:

        if (
            np.all(np.isfinite(previous))
            and np.all(np.isfinite(observation))
        ):
            predictions.append(previous)
            targets.append(observation)

        previous = observation

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    from .predictive_validation import _calculate_metrics

    return _calculate_metrics(
        predictions,
        targets,
    )


def fit_and_evaluate_personalized_model(
    initial_config: StateSpaceConfig,
    train_inputs: np.ndarray,
    train_observations: np.ndarray,
    test_inputs: np.ndarray,
    test_observations: np.ndarray,
    maxiter: int = MAXITER,
) -> PredictionMetrics:
    """
    Fit PHYSIO-TWIN parameters using training data only.

    The fitted parameters are then frozen during test evaluation.
    """

    from .estimator import fit_population_core
    from .state_space import filter_step

    fitted = fit_population_core(
        observations=train_observations,
        inputs=train_inputs,
        initial_config=initial_config,
        maxiter=maxiter,
    )

    fitted_config = StateSpaceConfig(
        A=fitted.A,
        B=fitted.B,
        C=initial_config.C,
        Q=fitted.Q,
        R=fitted.R,
        initial_state=initial_config.initial_state,
        initial_covariance=initial_config.initial_covariance,
    )

    state = fitted_config.initial_state.copy()
    covariance = fitted_config.initial_covariance.copy()

    # Establish the filtered state using training data only.
    for t in range(len(train_observations)):

        state, covariance, _, _ = filter_step(
            state=state,
            covariance=covariance,
            inputs=train_inputs[t],
            observation=train_observations[t],
            config=fitted_config,
        )

    predictions = []
    targets = []

    for t in range(len(test_observations)):

        predicted_state = (
            fitted_config.A @ state
            + fitted_config.B @ test_inputs[t]
        )

        predicted_observation = (
            fitted_config.C @ predicted_state
        )

        target = test_observations[t]

        if (
            np.all(np.isfinite(predicted_observation))
            and np.all(np.isfinite(target))
        ):
            predictions.append(
                predicted_observation
            )
            targets.append(target)

        # Update only after prediction.
        state, covariance, _, _ = filter_step(
            state=state,
            covariance=covariance,
            inputs=test_inputs[t],
            observation=target,
            config=fitted_config,
        )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    from .predictive_validation import _calculate_metrics

    metrics = _calculate_metrics(
        predictions,
        targets,
    )

    return metrics


def run_single_experiment(
    config: StateSpaceConfig,
    days: int,
    seed: int,
    train_ratio: float = TRAIN_RATIO,
    maxiter: int = MAXITER,
) -> list[dict]:
    """Run one complete M3 experiment."""

    dataset = make_synthetic_dataset(
        config=config,
        days=days,
        seed=seed,
    )

    train_days = int(
        round(days * train_ratio)
    )

    split = chronological_split(
        inputs=dataset.inputs,
        observations=dataset.observations,
        train_days=train_days,
    )

    rows = []

    persistence = evaluate_persistence_model(
        train_observations=split.train_observations,
        test_observations=split.test_observations,
    )

    rows.append(
        {
            "days": days,
            "train_days": train_days,
            "test_days": len(split.test_observations),
            "seed": seed,
            "model": "persistence",
            "converged": True,
            "fatigue_mae": persistence.fatigue_mae,
            "recovery_mae": persistence.recovery_mae,
            "overall_mae": persistence.overall_mae,
            "fatigue_rmse": persistence.fatigue_rmse,
            "recovery_rmse": persistence.recovery_rmse,
            "overall_rmse": persistence.overall_rmse,
            "n_predictions": persistence.n_predictions,
        }
    )

    population = evaluate_population_model(
        config=config,
        train_inputs=split.train_inputs,
        train_observations=split.train_observations,
        test_inputs=split.test_inputs,
        test_observations=split.test_observations,
    )

    rows.append(
        {
            "days": days,
            "train_days": train_days,
            "test_days": len(split.test_observations),
            "seed": seed,
            "model": "population",
            "converged": True,
            "fatigue_mae": population.fatigue_mae,
            "recovery_mae": population.recovery_mae,
            "overall_mae": population.overall_mae,
            "fatigue_rmse": population.fatigue_rmse,
            "recovery_rmse": population.recovery_rmse,
            "overall_rmse": population.overall_rmse,
            "n_predictions": population.n_predictions,
        }
    )

    personalized = fit_and_evaluate_personalized_model(
        initial_config=config,
        train_inputs=split.train_inputs,
        train_observations=split.train_observations,
        test_inputs=split.test_inputs,
        test_observations=split.test_observations,
        maxiter=maxiter,
    )

    # Check convergence separately so that the result is not confused
    # with the prediction metrics.
    from .estimator import fit_population_core

    fit_result = fit_population_core(
        observations=split.train_observations,
        inputs=split.train_inputs,
        initial_config=config,
        maxiter=maxiter,
    )

    rows.append(
        {
            "days": days,
            "train_days": train_days,
            "test_days": len(split.test_observations),
            "seed": seed,
            "model": "personalized",
            "converged": fit_result.converged,
            "fatigue_mae": personalized.fatigue_mae,
            "recovery_mae": personalized.recovery_mae,
            "overall_mae": personalized.overall_mae,
            "fatigue_rmse": personalized.fatigue_rmse,
            "recovery_rmse": personalized.recovery_rmse,
            "overall_rmse": personalized.overall_rmse,
            "n_predictions": personalized.n_predictions,
        }
    )

    return rows


def run_experiment_matrix(
    config: StateSpaceConfig | None = None,
    history_lengths: tuple[int, ...] = HISTORY_LENGTHS,
    seeds: tuple[int, ...] = SEEDS,
    train_ratio: float = TRAIN_RATIO,
    maxiter: int = MAXITER,
) -> pd.DataFrame:
    """Run the complete M3 experiment matrix."""

    if config is None:
        config = make_reference_config()

    rows = []

    total = len(history_lengths) * len(seeds)
    completed = 0

    for days in history_lengths:

        for seed in seeds:

            completed += 1

            print(
                f"[{completed:02d}/{total:02d}] "
                f"days={days}, seed={seed}"
            )

            experiment_rows = run_single_experiment(
                config=config,
                days=days,
                seed=seed,
                train_ratio=train_ratio,
                maxiter=maxiter,
            )

            rows.extend(experiment_rows)

            for row in experiment_rows:

                print(
                    f"    "
                    f"{row['model']:12s} "
                    f"RMSE={row['overall_rmse']:.6f}"
                )

    return pd.DataFrame(rows)


def summarize_experiments(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate M3 results."""

    summary = (
        results
        .groupby(
            [
                "days",
                "train_days",
                "model",
            ]
        )
        .agg(
            n_runs=("seed", "count"),
            convergence_rate=(
                "converged",
                "mean",
            ),
            mean_overall_mae=(
                "overall_mae",
                "mean",
            ),
            std_overall_mae=(
                "overall_mae",
                "std",
            ),
            mean_overall_rmse=(
                "overall_rmse",
                "mean",
            ),
            std_overall_rmse=(
                "overall_rmse",
                "std",
            ),
            mean_fatigue_rmse=(
                "fatigue_rmse",
                "mean",
            ),
            mean_recovery_rmse=(
                "recovery_rmse",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


def main() -> None:
    """Run and save the M3.1 experiment."""

    project_root = Path(
        __file__
    ).resolve().parents[1]

    data_dir = project_root / "data"

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        data_dir
        / "m3_1_predictive_validation.csv"
    )

    summary_path = (
        data_dir
        / "m3_1_predictive_validation_summary.csv"
    )

    config = make_reference_config()

    print("=" * 72)
    print(
        "PHYSIO-TWIN M3.1 PREDICTIVE VALIDATION"
    )
    print("=" * 72)

    print()
    print(
        f"History lengths: {HISTORY_LENGTHS}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Train ratio: {TRAIN_RATIO}"
    )

    print(
        f"Maximum optimizer iterations: {MAXITER}"
    )

    print()

    results = run_experiment_matrix(
        config=config,
        history_lengths=HISTORY_LENGTHS,
        seeds=SEEDS,
        train_ratio=TRAIN_RATIO,
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
    print("M3.1 COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Raw results: {raw_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print()
    print("SUMMARY")
    print("-" * 72)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()