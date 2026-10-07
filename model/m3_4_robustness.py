from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .estimator import fit_population_core
from .mismatch import simulate_mismatched_process
from .m3_2_experiment import make_reference_config
from .state_space import StateSpaceConfig, filter_step


HISTORY_LENGTHS = (180, 240, 500)
SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (1, 3, 7, 14)

SCENARIOS = (
    "nonlinear_saturation",
    "delayed_sleep",
    "correlated_noise",
    "person_shift",
)

TRAIN_RATIO = 0.75
MAXITER = 600

NOMINAL_COVERAGE = 0.95
Z_95 = 1.959963984540054

OUTPUT_DIR = Path("data")


@dataclass(frozen=True)
class RobustnessMetrics:
    scenario: str
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

    fatigue_coverage: float
    recovery_coverage: float

    fatigue_width: float
    recovery_width: float
    mean_interval_width: float

    fatigue_calibration_error: float
    recovery_calibration_error: float

    converged: bool


def _coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
) -> float:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(actual) == 0:
        return float("nan")

    return float(
        np.mean(
            (actual >= lower)
            & (actual <= upper)
        )
    )


def _interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if len(lower) == 0:
        return float("nan")

    return float(np.mean(upper - lower))


def _calibration_error(coverage: float) -> float:
    return float(abs(coverage - NOMINAL_COVERAGE))


def _predict_state(
    state: np.ndarray,
    covariance: np.ndarray,
    inputs: np.ndarray,
    config: StateSpaceConfig,
):
    predicted_state = config.A @ state + config.B @ inputs

    predicted_covariance = (
        config.A
        @ covariance
        @ config.A.T
        + config.Q
    )

    predicted_covariance = (
        predicted_covariance
        + predicted_covariance.T
    ) / 2.0

    return predicted_state, predicted_covariance


def _filter_to_origin(
    observations: np.ndarray,
    inputs: np.ndarray,
    origin: int,
    config: StateSpaceConfig,
):
    state = np.asarray(
        config.initial_state,
        dtype=float,
    ).copy()

    covariance = np.asarray(
        config.initial_covariance,
        dtype=float,
    ).copy()

    for t in range(origin + 1):
        state, covariance, _, _ = filter_step(
            state,
            covariance,
            inputs[t],
            observations[t],
            config,
        )

    return state, covariance


def _forecast_interval(
    observations: np.ndarray,
    inputs: np.ndarray,
    origin: int,
    horizon: int,
    config: StateSpaceConfig,
):
    state, covariance = _filter_to_origin(
        observations,
        inputs,
        origin,
        config,
    )

    future_states = []
    future_covariances = []
    lower_bounds = []
    upper_bounds = []

    for step in range(1, horizon + 1):
        index = origin + step

        state, covariance = _predict_state(
            state,
            covariance,
            inputs[index],
            config,
        )

        observation_mean = config.C @ state

        observation_covariance = (
            config.C
            @ covariance
            @ config.C.T
            + config.R
        )

        observation_covariance = (
            observation_covariance
            + observation_covariance.T
        ) / 2.0

        observation_std = np.sqrt(
            np.maximum(
                np.diag(observation_covariance),
                0.0,
            )
        )

        lower = (
            observation_mean
            - Z_95 * observation_std
        )

        upper = (
            observation_mean
            + Z_95 * observation_std
        )

        future_states.append(state.copy())
        future_covariances.append(
            covariance.copy()
        )
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    return (
        np.asarray(future_states),
        np.asarray(future_covariances),
        np.asarray(lower_bounds),
        np.asarray(upper_bounds),
    )


def _fit_personalized_config(
    observations: np.ndarray,
    inputs: np.ndarray,
    base_config: StateSpaceConfig,
):
    result = fit_population_core(
        observations,
        inputs,
        initial_config=base_config,
        maxiter=MAXITER,
    )

    config = StateSpaceConfig(
        A=result.A,
        B=result.B,
        C=base_config.C,
        Q=result.Q,
        R=result.R,
        initial_state=base_config.initial_state,
        initial_covariance=base_config.initial_covariance,
    )

    return config, bool(result.converged)


def _evaluate_model(
    scenario: str,
    observations: np.ndarray,
    true_states: np.ndarray,
    inputs: np.ndarray,
    config: StateSpaceConfig,
    days: int,
    train_days: int,
    seed: int,
    horizon: int,
    model: str,
    converged: bool,
):
    origins = range(
        train_days - 1,
        len(observations) - horizon,
    )

    fatigue_actual = []
    recovery_actual = []

    fatigue_pred = []
    recovery_pred = []

    fatigue_lower = []
    fatigue_upper = []

    recovery_lower = []
    recovery_upper = []

    for origin in origins:
        (
            _states,
            _covariances,
            lower,
            upper,
        ) = _forecast_interval(
            observations,
            inputs,
            origin,
            horizon,
            config,
        )

        target_index = origin + horizon

        actual = true_states[target_index]

        prediction = (
            lower[-1] + upper[-1]
        ) / 2.0

        fatigue_actual.append(actual[0])
        recovery_actual.append(actual[1])

        fatigue_pred.append(prediction[0])
        recovery_pred.append(prediction[1])

        fatigue_lower.append(lower[-1, 0])
        fatigue_upper.append(upper[-1, 0])

        recovery_lower.append(lower[-1, 1])
        recovery_upper.append(upper[-1, 1])

    fatigue_actual = np.asarray(fatigue_actual)
    recovery_actual = np.asarray(recovery_actual)

    fatigue_pred = np.asarray(fatigue_pred)
    recovery_pred = np.asarray(recovery_pred)

    fatigue_lower = np.asarray(fatigue_lower)
    fatigue_upper = np.asarray(fatigue_upper)

    recovery_lower = np.asarray(recovery_lower)
    recovery_upper = np.asarray(recovery_upper)

    fatigue_error = (
        fatigue_pred - fatigue_actual
    )

    recovery_error = (
        recovery_pred - recovery_actual
    )

    fatigue_mae = float(
        np.mean(np.abs(fatigue_error))
    )

    recovery_mae = float(
        np.mean(np.abs(recovery_error))
    )

    overall_mae = float(
        np.mean(
            [
                fatigue_mae,
                recovery_mae,
            ]
        )
    )

    fatigue_rmse = float(
        np.sqrt(
            np.mean(
                fatigue_error ** 2
            )
        )
    )

    recovery_rmse = float(
        np.sqrt(
            np.mean(
                recovery_error ** 2
            )
        )
    )

    overall_rmse = float(
        np.mean(
            [
                fatigue_rmse,
                recovery_rmse,
            ]
        )
    )

    fatigue_coverage = _coverage(
        fatigue_lower,
        fatigue_upper,
        fatigue_actual,
    )

    recovery_coverage = _coverage(
        recovery_lower,
        recovery_upper,
        recovery_actual,
    )

    fatigue_width = _interval_width(
        fatigue_lower,
        fatigue_upper,
    )

    recovery_width = _interval_width(
        recovery_lower,
        recovery_upper,
    )

    mean_interval_width = float(
        np.mean(
            [
                fatigue_width,
                recovery_width,
            ]
        )
    )

    return RobustnessMetrics(
        scenario=scenario,
        days=days,
        train_days=train_days,
        seed=seed,
        horizon=horizon,
        model=model,
        n_predictions=len(fatigue_actual),
        fatigue_mae=fatigue_mae,
        recovery_mae=recovery_mae,
        overall_mae=overall_mae,
        fatigue_rmse=fatigue_rmse,
        recovery_rmse=recovery_rmse,
        overall_rmse=overall_rmse,
        fatigue_coverage=fatigue_coverage,
        recovery_coverage=recovery_coverage,
        fatigue_width=fatigue_width,
        recovery_width=recovery_width,
        mean_interval_width=mean_interval_width,
        fatigue_calibration_error=_calibration_error(
            fatigue_coverage
        ),
        recovery_calibration_error=_calibration_error(
            recovery_coverage
        ),
        converged=converged,
    )


def run_single_experiment(
    scenario: str,
    days: int,
    seed: int,
):
    reference_config = make_reference_config()

    dataset = simulate_mismatched_process(
        reference_config,
        days=days,
        seed=seed,
        scenario=scenario,
    )

    observations = dataset.observations
    true_states = dataset.true_states
    inputs = dataset.inputs

    train_days = int(
        np.floor(days * TRAIN_RATIO)
    )

    train_observations = observations[:train_days]
    train_inputs = inputs[:train_days]

    personalized_config, converged = (
        _fit_personalized_config(
            train_observations,
            train_inputs,
            reference_config,
        )
    )

    results = []

    for horizon in HORIZONS:
        population_result = _evaluate_model(
            scenario=scenario,
            observations=observations,
            true_states=true_states,
            inputs=inputs,
            config=reference_config,
            days=days,
            train_days=train_days,
            seed=seed,
            horizon=horizon,
            model="population",
            converged=True,
        )

        personalized_result = _evaluate_model(
            scenario=scenario,
            observations=observations,
            true_states=true_states,
            inputs=inputs,
            config=personalized_config,
            days=days,
            train_days=train_days,
            seed=seed,
            horizon=horizon,
            model="personalized",
            converged=converged,
        )

        results.extend(
            [
                population_result,
                personalized_result,
            ]
        )

        print(
            f"    population    h={horizon:<2d} "
            f"RMSE={population_result.overall_rmse:.4f} "
            f"fat_cov={population_result.fatigue_coverage:.4f} "
            f"rec_cov={population_result.recovery_coverage:.4f}"
        )

        print(
            f"    personalized  h={horizon:<2d} "
            f"RMSE={personalized_result.overall_rmse:.4f} "
            f"fat_cov={personalized_result.fatigue_coverage:.4f} "
            f"rec_cov={personalized_result.recovery_coverage:.4f}"
        )

    return results


def summarize(results: list[RobustnessMetrics]):
    frame = pd.DataFrame(
        [
            result.__dict__
            for result in results
        ]
    )

    summary = (
        frame.groupby(
            [
                "scenario",
                "days",
                "train_days",
                "horizon",
                "model",
            ],
            as_index=False,
        )
        .agg(
            n_runs=("seed", "nunique"),
            convergence_rate=(
                "converged",
                "mean",
            ),
            mean_n_predictions=(
                "n_predictions",
                "mean",
            ),
            mean_fatigue_mae=(
                "fatigue_mae",
                "mean",
            ),
            mean_recovery_mae=(
                "recovery_mae",
                "mean",
            ),
            mean_overall_mae=(
                "overall_mae",
                "mean",
            ),
            mean_fatigue_rmse=(
                "fatigue_rmse",
                "mean",
            ),
            mean_recovery_rmse=(
                "recovery_rmse",
                "mean",
            ),
            mean_overall_rmse=(
                "overall_rmse",
                "mean",
            ),
            mean_fatigue_coverage=(
                "fatigue_coverage",
                "mean",
            ),
            std_fatigue_coverage=(
                "fatigue_coverage",
                "std",
            ),
            mean_recovery_coverage=(
                "recovery_coverage",
                "mean",
            ),
            std_recovery_coverage=(
                "recovery_coverage",
                "std",
            ),
            mean_fatigue_width=(
                "fatigue_width",
                "mean",
            ),
            mean_recovery_width=(
                "recovery_width",
                "mean",
            ),
            mean_interval_width=(
                "mean_interval_width",
                "mean",
            ),
            mean_fatigue_calibration_error=(
                "fatigue_calibration_error",
                "mean",
            ),
            mean_recovery_calibration_error=(
                "recovery_calibration_error",
                "mean",
            ),
        )
    )

    return frame, summary


def main():
    print("=" * 72)
    print("PHYSIO-TWIN M3.4 ROBUSTNESS UNDER MODEL MISMATCH")
    print("=" * 72)

    print()
    print(f"Scenarios: {SCENARIOS}")
    print(f"History lengths: {HISTORY_LENGTHS}")
    print(f"Seeds: {SEEDS}")
    print(f"Forecast horizons: {HORIZONS}")
    print(f"Train ratio: {TRAIN_RATIO}")
    print(f"Nominal coverage: {NOMINAL_COVERAGE}")
    print()

    results = []

    total = (
        len(SCENARIOS)
        * len(HISTORY_LENGTHS)
        * len(SEEDS)
    )

    counter = 0

    for scenario in SCENARIOS:
        print()
        print("=" * 72)
        print(f"SCENARIO: {scenario}")
        print("=" * 72)

        for days in HISTORY_LENGTHS:
            for seed in SEEDS:
                counter += 1

                print(
                    f"\n[{counter}/{total}] "
                    f"scenario={scenario}, "
                    f"days={days}, seed={seed}"
                )

                results.extend(
                    run_single_experiment(
                        scenario=scenario,
                        days=days,
                        seed=seed,
                    )
                )

    frame, summary = summarize(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        OUTPUT_DIR
        / "m3_4_robustness_validation.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "m3_4_robustness_validation_summary.csv"
    )

    frame.to_csv(
        raw_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("M3.4 COMPLETE")
    print("=" * 72)

    print()
    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")

    print()
    print("SUMMARY")
    print("-" * 72)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_rows",
        None,
    ):
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )


if __name__ == "__main__":
    main()