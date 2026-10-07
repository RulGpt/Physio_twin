from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .estimator import fit_population_core
from .state_space import StateSpaceConfig, filter_step
from .synthetic import make_synthetic_dataset


HISTORY_LENGTHS = (180, 240, 500)
SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (1, 3, 7, 14)
TRAIN_RATIO = 0.75
MAXITER = 600


@dataclass(frozen=True)
class HorizonMetrics:
    days: int
    train_days: int
    seed: int
    horizon: int
    model: str
    n_predictions: int
    fatigue_mae: float
    recovery_mae: float
    overall_mae: float
    fatigue_rmse: float
    recovery_rmse: float
    overall_rmse: float
    converged: bool


def make_reference_config() -> StateSpaceConfig:
    return StateSpaceConfig(
        A=np.array([
            [0.70, 0.05, 0.03],
            [0.03, 0.75, 0.02],
            [0.02, 0.03, 0.60],
        ]),
        B=np.array([
            [0.10, 0.08, 0.05],
            [-0.04, -0.03, -0.02],
            [0.08, 0.12, 0.04],
        ]),
        C=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
        Q=np.diag([0.01, 0.01, 0.02]),
        R=np.diag([0.04, 0.04]),
        initial_state=np.array([0.5, 0.5, 0.2]),
        initial_covariance=np.diag([0.10, 0.10, 0.20]),
    )


def _metrics(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> tuple[float, float, float, float, float, float, int]:
    predictions = np.asarray(predictions, dtype=float)
    observations = np.asarray(observations, dtype=float)

    valid = np.all(
        np.isfinite(predictions) & np.isfinite(observations),
        axis=1,
    )

    predictions = predictions[valid]
    observations = observations[valid]

    if len(predictions) == 0:
        raise ValueError("No valid predictions available.")

    errors = predictions - observations

    fatigue_mae = float(np.mean(np.abs(errors[:, 0])))
    recovery_mae = float(np.mean(np.abs(errors[:, 1])))
    overall_mae = float(np.mean(np.abs(errors)))

    fatigue_rmse = float(np.sqrt(np.mean(errors[:, 0] ** 2)))
    recovery_rmse = float(np.sqrt(np.mean(errors[:, 1] ** 2)))
    overall_rmse = float(np.sqrt(np.mean(errors ** 2)))

    return (
        fatigue_mae,
        recovery_mae,
        overall_mae,
        fatigue_rmse,
        recovery_rmse,
        overall_rmse,
        len(predictions),
    )


def _persistence_forecast(
    observations: np.ndarray,
    start: int,
    horizon: int,
) -> np.ndarray:
    """
    Persistence forecast.

    For every target day t+h, use the most recent known observation
    at the forecast origin t.
    """
    predictions = []
    targets = []

    for origin in range(start, len(observations) - horizon):
        target_index = origin + horizon

        y = observations[origin]

        if np.all(np.isfinite(y)):
            predictions.append(y)
            targets.append(observations[target_index])

    return np.asarray(predictions), np.asarray(targets)


def _state_space_forecast(
    config: StateSpaceConfig,
    inputs: np.ndarray,
    observations: np.ndarray,
    start: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling-origin multi-step state-space forecast.

    At forecast origin t:
        1. filter Y_t using U_t
        2. propagate the state using U_(t+1)...U_(t+h)
        3. compare forecast against Y_(t+h)

    No future observation is used during forecasting.
    """
    state = config.initial_state.copy()
    covariance = config.initial_covariance.copy()

    predictions = []
    targets = []

    # Build filtered state using observations up to forecast origin - 1.
    for t in range(start):
        state, covariance, _, _ = filter_step(
            state,
            covariance,
            inputs[t],
            observations[t],
            config,
        )

    for origin in range(start, len(observations) - horizon):
        # Incorporate current observation before forecasting.
        state, covariance, _, _ = filter_step(
            state,
            covariance,
            inputs[origin],
            observations[origin],
            config,
        )

        future_state = state.copy()
        future_covariance = covariance.copy()

        for step in range(1, horizon + 1):
            future_state, future_covariance = _predict_state(
                future_state,
                future_covariance,
                inputs[origin + step],
                config,
            )

        predicted_observation = config.C @ future_state
        target_observation = observations[origin + horizon]

        if (
            np.all(np.isfinite(predicted_observation))
            and np.all(np.isfinite(target_observation))
        ):
            predictions.append(predicted_observation)
            targets.append(target_observation)

    return np.asarray(predictions), np.asarray(targets)


def _predict_state(
    state: np.ndarray,
    covariance: np.ndarray,
    inputs: np.ndarray,
    config: StateSpaceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    next_state = config.A @ state + config.B @ inputs

    next_covariance = (
        config.A @ covariance @ config.A.T + config.Q
    )

    next_covariance = (
        next_covariance + next_covariance.T
    ) / 2.0

    return next_state, next_covariance


def _evaluate_model(
    model_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    days: int,
    train_days: int,
    seed: int,
    horizon: int,
    converged: bool,
) -> HorizonMetrics:
    (
        fatigue_mae,
        recovery_mae,
        overall_mae,
        fatigue_rmse,
        recovery_rmse,
        overall_rmse,
        n_predictions,
    ) = _metrics(predictions, targets)

    return HorizonMetrics(
        days=days,
        train_days=train_days,
        seed=seed,
        horizon=horizon,
        model=model_name,
        n_predictions=n_predictions,
        fatigue_mae=fatigue_mae,
        recovery_mae=recovery_mae,
        overall_mae=overall_mae,
        fatigue_rmse=fatigue_rmse,
        recovery_rmse=recovery_rmse,
        overall_rmse=overall_rmse,
        converged=converged,
    )


def run_single_experiment(
    config: StateSpaceConfig,
    days: int,
    seed: int,
) -> list[HorizonMetrics]:
    dataset = make_synthetic_dataset(
        config,
        days=days,
        seed=seed,
    )

    train_days = int(days * TRAIN_RATIO)

    # ------------------------------------------------------------
    # Fit personalized model ONLY on training data.
    # ------------------------------------------------------------
    fit = fit_population_core(
        dataset.observations[:train_days],
        dataset.inputs[:train_days],
        initial_config=config,
        maxiter=MAXITER,
    )

    fitted_config = StateSpaceConfig(
        A=fit.A,
        B=fit.B,
        C=config.C,
        Q=fit.Q,
        R=fit.R,
        initial_state=config.initial_state,
        initial_covariance=config.initial_covariance,
    )

    results: list[HorizonMetrics] = []

    for horizon in HORIZONS:

        # --------------------------------------------------------
        # Persistence
        # --------------------------------------------------------
        persistence_predictions = []
        persistence_targets = []

        for origin in range(
            train_days - 1,
            days - horizon,
        ):
            target_index = origin + horizon

            prediction = dataset.observations[origin]
            target = dataset.observations[target_index]

            if (
                np.all(np.isfinite(prediction))
                and np.all(np.isfinite(target))
            ):
                persistence_predictions.append(prediction)
                persistence_targets.append(target)

        results.append(
            _evaluate_model(
                "persistence",
                np.asarray(persistence_predictions),
                np.asarray(persistence_targets),
                days,
                train_days,
                seed,
                horizon,
                True,
            )
        )

        # --------------------------------------------------------
        # Population model
        # --------------------------------------------------------
        population_predictions, population_targets = (
            _state_space_forecast(
                config,
                dataset.inputs,
                dataset.observations,
                train_days - 1,
                horizon,
            )
        )

        results.append(
            _evaluate_model(
                "population",
                population_predictions,
                population_targets,
                days,
                train_days,
                seed,
                horizon,
                True,
            )
        )

        # --------------------------------------------------------
        # Personalized model
        # --------------------------------------------------------
        personalized_predictions, personalized_targets = (
            _state_space_forecast(
                fitted_config,
                dataset.inputs,
                dataset.observations,
                train_days - 1,
                horizon,
            )
        )

        results.append(
            _evaluate_model(
                "personalized",
                personalized_predictions,
                personalized_targets,
                days,
                train_days,
                seed,
                horizon,
                fit.converged,
            )
        )

    return results


def summarize(results: list[HorizonMetrics]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [r.__dict__ for r in results]
    )

    summary = (
        frame
        .groupby(
            ["days", "train_days", "horizon", "model"],
            as_index=False,
        )
        .agg(
            n_runs=("seed", "count"),
            convergence_rate=("converged", "mean"),
            mean_overall_mae=("overall_mae", "mean"),
            std_overall_mae=("overall_mae", "std"),
            mean_overall_rmse=("overall_rmse", "mean"),
            std_overall_rmse=("overall_rmse", "std"),
            mean_fatigue_rmse=("fatigue_rmse", "mean"),
            mean_recovery_rmse=("recovery_rmse", "mean"),
        )
    )

    return summary


def main() -> None:
    config = make_reference_config()

    print("=" * 72)
    print("PHYSIO-TWIN M3.2 MULTI-HORIZON FORECAST VALIDATION")
    print("=" * 72)
    print()
    print(f"History lengths: {HISTORY_LENGTHS}")
    print(f"Seeds: {SEEDS}")
    print(f"Forecast horizons: {HORIZONS}")
    print(f"Train ratio: {TRAIN_RATIO}")
    print(f"Maximum optimizer iterations: {MAXITER}")
    print()

    all_results: list[HorizonMetrics] = []

    total = (
        len(HISTORY_LENGTHS)
        * len(SEEDS)
    )

    completed = 0

    for days in HISTORY_LENGTHS:
        for seed in SEEDS:

            completed += 1

            print(
                f"[{completed:02d}/{total}] "
                f"days={days}, seed={seed}"
            )

            results = run_single_experiment(
                config,
                days,
                seed,
            )

            all_results.extend(results)

            for result in results:
                print(
                    f"    {result.model:<12} "
                    f"h={result.horizon:<2} "
                    f"RMSE={result.overall_rmse:.6f}"
                )

    frame = pd.DataFrame(
        [r.__dict__ for r in all_results]
    )

    summary = summarize(all_results)

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "m3_2_multi_horizon_validation.csv"
    summary_path = (
        output_dir
        / "m3_2_multi_horizon_validation_summary.csv"
    )

    frame.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print()
    print("=" * 72)
    print("M3.2 COMPLETE")
    print("=" * 72)
    print()
    print(f"Raw results: {raw_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print()
    print("SUMMARY")
    print("-" * 72)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()