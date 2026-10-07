from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .estimator import fit_population_core
from .state_space import StateSpaceConfig, filter_step
from .synthetic import make_synthetic_dataset
from .m3_2_experiment import make_reference_config


HISTORY_LENGTHS = (180, 240, 500)
SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (1, 3, 7, 14)

TRAIN_RATIO = 0.75
MAXITER = 600

NOMINAL_COVERAGE = 0.95
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class IntervalMetrics:
    days: int
    train_days: int
    seed: int
    horizon: int
    model: str

    n_predictions: int

    fatigue_coverage: float
    recovery_coverage: float

    fatigue_width: float
    recovery_width: float
    mean_interval_width: float

    fatigue_calibration_error: float
    recovery_calibration_error: float

    converged: bool


def _coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    inside = (
        (actual >= lower)
        & (actual <= upper)
    )

    return float(np.mean(inside))


def _interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.asarray(upper)
            - np.asarray(lower)
        )
    )


def _predict_state(
    state: np.ndarray,
    covariance: np.ndarray,
    inputs: np.ndarray,
    config: StateSpaceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step state prediction with covariance propagation.
    """

    next_state = (
        config.A @ state
        + config.B @ inputs
    )

    next_covariance = (
        config.A
        @ covariance
        @ config.A.T
        + config.Q
    )

    next_covariance = (
        next_covariance
        + next_covariance.T
    ) / 2.0

    return (
        next_state,
        next_covariance,
    )


def _filter_to_origin(
    config: StateSpaceConfig,
    inputs: np.ndarray,
    observations: np.ndarray,
    origin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter all observations up to and including the forecast origin.

    No observations after `origin` are used.
    """

    state = config.initial_state.copy()
    covariance = config.initial_covariance.copy()

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
    config: StateSpaceConfig,
    inputs: np.ndarray,
    observations: np.ndarray,
    origin: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an h-step forecast and marginal 95% prediction intervals.

    At forecast origin t:

        1. Filter observations through Y_t.
        2. Propagate X using U_(t+1)...U_(t+h).
        3. Propagate covariance using Q.
        4. Project covariance into observation space.
        5. Construct marginal 95% intervals.

    Future observations are never used.
    """

    state, covariance = _filter_to_origin(
        config=config,
        inputs=inputs,
        observations=observations,
        origin=origin,
    )

    predicted_observations = []
    lower_bounds = []
    upper_bounds = []

    for step in range(
        1,
        horizon + 1,
    ):

        state, covariance = _predict_state(
            state,
            covariance,
            inputs[origin + step],
            config,
        )

        observation_mean = (
            config.C @ state
        )

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

        variances = np.maximum(
            np.diag(
                observation_covariance
            ),
            0.0,
        )

        standard_deviation = np.sqrt(
            variances
        )

        lower = (
            observation_mean
            - Z_95
            * standard_deviation
        )

        upper = (
            observation_mean
            + Z_95
            * standard_deviation
        )

        predicted_observations.append(
            observation_mean
        )

        lower_bounds.append(
            lower
        )

        upper_bounds.append(
            upper
        )

    return (
        np.asarray(
            predicted_observations
        ),
        np.asarray(
            lower_bounds
        ),
        np.asarray(
            upper_bounds
        ),
    )


def _evaluate_model(
    config: StateSpaceConfig,
    inputs: np.ndarray,
    observations: np.ndarray,
    start: int,
    horizon: int,
) -> dict[str, float]:
    """
    Rolling-origin uncertainty evaluation.

    Every forecast uses observations only through its origin.
    """

    actual_values = []
    lower_values = []
    upper_values = []

    for origin in range(
        start,
        len(observations) - horizon,
    ):

        target_index = (
            origin + horizon
        )

        (
            _,
            lower,
            upper,
        ) = _forecast_interval(
            config=config,
            inputs=inputs,
            observations=observations,
            origin=origin,
            horizon=horizon,
        )

        actual = observations[
            target_index
        ]

        if not np.all(
            np.isfinite(actual)
        ):
            continue

        actual_values.append(
            actual
        )

        lower_values.append(
            lower[-1]
        )

        upper_values.append(
            upper[-1]
        )

    if not actual_values:
        raise ValueError(
            "No valid interval predictions available."
        )

    actual_values = np.asarray(
        actual_values,
        dtype=float,
    )

    lower_values = np.asarray(
        lower_values,
        dtype=float,
    )

    upper_values = np.asarray(
        upper_values,
        dtype=float,
    )

    fatigue_coverage = _coverage(
        actual_values[:, 0],
        lower_values[:, 0],
        upper_values[:, 0],
    )

    recovery_coverage = _coverage(
        actual_values[:, 1],
        lower_values[:, 1],
        upper_values[:, 1],
    )

    fatigue_width = _interval_width(
        lower_values[:, 0],
        upper_values[:, 0],
    )

    recovery_width = _interval_width(
        lower_values[:, 1],
        upper_values[:, 1],
    )

    return {
        "n_predictions": len(
            actual_values
        ),
        "fatigue_coverage": (
            fatigue_coverage
        ),
        "recovery_coverage": (
            recovery_coverage
        ),
        "fatigue_width": (
            fatigue_width
        ),
        "recovery_width": (
            recovery_width
        ),
        "mean_interval_width": (
            fatigue_width
            + recovery_width
        ) / 2.0,
        "fatigue_calibration_error": abs(
            fatigue_coverage
            - NOMINAL_COVERAGE
        ),
        "recovery_calibration_error": abs(
            recovery_coverage
            - NOMINAL_COVERAGE
        ),
    }


def _fit_personalized_config(
    config: StateSpaceConfig,
    observations: np.ndarray,
    inputs: np.ndarray,
    train_days: int,
):
    """
    Fit personalized parameters using training data only.
    """

    fit = fit_population_core(
        observations[:train_days],
        inputs[:train_days],
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

    return fitted_config, fit.converged


def run_single_experiment(
    config: StateSpaceConfig,
    days: int,
    seed: int,
) -> list[IntervalMetrics]:
    """
    Run M3.3 for one history length and seed.
    """

    dataset = make_synthetic_dataset(
        config,
        days=days,
        seed=seed,
    )

    train_days = int(
        days * TRAIN_RATIO
    )

    # ------------------------------------------------------------
    # Personalized model.
    #
    # IMPORTANT:
    # Only training observations and inputs are used for fitting.
    # ------------------------------------------------------------

    personalized_config, converged = (
        _fit_personalized_config(
            config=config,
            observations=dataset.observations,
            inputs=dataset.inputs,
            train_days=train_days,
        )
    )

    results = []

    for horizon in HORIZONS:

        # --------------------------------------------------------
        # Population/reference model
        # --------------------------------------------------------

        population_metrics = _evaluate_model(
            config=config,
            inputs=dataset.inputs,
            observations=dataset.observations,
            start=train_days - 1,
            horizon=horizon,
        )

        results.append(
            IntervalMetrics(
                days=days,
                train_days=train_days,
                seed=seed,
                horizon=horizon,
                model="population",
                n_predictions=int(
                    population_metrics[
                        "n_predictions"
                    ]
                ),
                fatigue_coverage=(
                    population_metrics[
                        "fatigue_coverage"
                    ]
                ),
                recovery_coverage=(
                    population_metrics[
                        "recovery_coverage"
                    ]
                ),
                fatigue_width=(
                    population_metrics[
                        "fatigue_width"
                    ]
                ),
                recovery_width=(
                    population_metrics[
                        "recovery_width"
                    ]
                ),
                mean_interval_width=(
                    population_metrics[
                        "mean_interval_width"
                    ]
                ),
                fatigue_calibration_error=(
                    population_metrics[
                        "fatigue_calibration_error"
                    ]
                ),
                recovery_calibration_error=(
                    population_metrics[
                        "recovery_calibration_error"
                    ]
                ),
                converged=True,
            )
        )

        # --------------------------------------------------------
        # Personalized model
        # --------------------------------------------------------

        personalized_metrics = _evaluate_model(
            config=personalized_config,
            inputs=dataset.inputs,
            observations=dataset.observations,
            start=train_days - 1,
            horizon=horizon,
        )

        results.append(
            IntervalMetrics(
                days=days,
                train_days=train_days,
                seed=seed,
                horizon=horizon,
                model="personalized",
                n_predictions=int(
                    personalized_metrics[
                        "n_predictions"
                    ]
                ),
                fatigue_coverage=(
                    personalized_metrics[
                        "fatigue_coverage"
                    ]
                ),
                recovery_coverage=(
                    personalized_metrics[
                        "recovery_coverage"
                    ]
                ),
                fatigue_width=(
                    personalized_metrics[
                        "fatigue_width"
                    ]
                ),
                recovery_width=(
                    personalized_metrics[
                        "recovery_width"
                    ]
                ),
                mean_interval_width=(
                    personalized_metrics[
                        "mean_interval_width"
                    ]
                ),
                fatigue_calibration_error=(
                    personalized_metrics[
                        "fatigue_calibration_error"
                    ]
                ),
                recovery_calibration_error=(
                    personalized_metrics[
                        "recovery_calibration_error"
                    ]
                ),
                converged=converged,
            )
        )

    return results


def summarize(
    results: list[IntervalMetrics],
) -> pd.DataFrame:

    frame = pd.DataFrame(
        [
            result.__dict__
            for result in results
        ]
    )

    summary = (
        frame
        .groupby(
            [
                "days",
                "train_days",
                "horizon",
                "model",
            ],
            as_index=False,
        )
        .agg(
            n_runs=(
                "seed",
                "count",
            ),
            convergence_rate=(
                "converged",
                "mean",
            ),
            mean_n_predictions=(
                "n_predictions",
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

    return summary


def main() -> None:

    config = make_reference_config()

    print("=" * 72)
    print(
        "PHYSIO-TWIN M3.3 "
        "UNCERTAINTY & CALIBRATION"
    )
    print("=" * 72)
    print()

    print(
        f"History lengths: "
        f"{HISTORY_LENGTHS}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Forecast horizons: "
        f"{HORIZONS}"
    )

    print(
        f"Train ratio: "
        f"{TRAIN_RATIO}"
    )

    print(
        f"Nominal coverage: "
        f"{NOMINAL_COVERAGE}"
    )

    print()

    all_results = []

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
                f"days={days}, "
                f"seed={seed}"
            )

            results = run_single_experiment(
                config=config,
                days=days,
                seed=seed,
            )

            all_results.extend(
                results
            )

            for result in results:

                print(
                    f"    "
                    f"{result.model:<12} "
                    f"h={result.horizon:<2} "
                    f"fatigue="
                    f"{result.fatigue_coverage:.4f} "
                    f"recovery="
                    f"{result.recovery_coverage:.4f} "
                    f"width="
                    f"{result.mean_interval_width:.4f}"
                )

    frame = pd.DataFrame(
        [
            result.__dict__
            for result in all_results
        ]
    )

    summary = summarize(
        all_results
    )

    output_dir = Path("data")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "m3_3_uncertainty_validation.csv"
    )

    summary_path = (
        output_dir
        / "m3_3_uncertainty_validation_summary.csv"
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
    print("M3.3 COMPLETE")
    print("=" * 72)
    print()

    print(
        f"Raw results: "
        f"{raw_path.resolve()}"
    )

    print(
        f"Summary: "
        f"{summary_path.resolve()}"
    )

    print()
    print("SUMMARY")
    print("-" * 72)

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()