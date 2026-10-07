from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimator import fit_population_core
from .synthetic import make_synthetic_dataset
from .state_space import StateSpaceConfig, spectral_radius


@dataclass(frozen=True)
class RecoveryMetrics:
    """Metrics produced by one synthetic parameter-recovery experiment."""

    days: int
    seed: int

    converged: bool
    optimizer_iterations: int | None

    # Relative parameter-recovery errors.
    A_rmse: float
    B_rmse: float
    Q_rmse: float
    R_rmse: float

    # Absolute RMSE across the complete 14-parameter estimated vector.
    total_parameter_rmse: float

    # This is an open-loop rollout metric, not a filtered one-step metric.
    open_loop_observation_rmse: float

    spectral_radius: float
    log_likelihood: float


def _relative_rmse(
    estimated: np.ndarray,
    true: np.ndarray,
) -> float:
    """Return RMSE normalized by the RMS magnitude of the true values."""

    estimated = np.asarray(estimated, dtype=float)
    true = np.asarray(true, dtype=float)

    denom = np.sqrt(np.mean(true**2)) + 1e-12

    return float(
        np.sqrt(np.mean((estimated - true) ** 2)) / denom
    )


def _open_loop_observation_rmse(
    config: StateSpaceConfig,
    fitted_A: np.ndarray,
    fitted_B: np.ndarray,
    fitted_Q: np.ndarray,
    fitted_R: np.ndarray,
    dataset,
) -> float:
    """Evaluate open-loop observation rollout error.

    The fitted model is initialized with the known synthetic initial state
    and then rolled forward without correcting the state using observations.

    This is intentionally NOT called a one-step-ahead filtered prediction
    metric. A genuine filtered one-step prediction metric will be introduced
    separately when predictive validation is implemented.
    """

    fitted_config = StateSpaceConfig(
        A=fitted_A,
        B=fitted_B,
        C=config.C,
        Q=fitted_Q,
        R=fitted_R,
        initial_state=config.initial_state,
        initial_covariance=config.initial_covariance,
    )

    state = fitted_config.initial_state.copy()
    errors: list[np.ndarray] = []

    for u, y in zip(
        dataset.inputs,
        dataset.observations,
    ):
        predicted = (
            fitted_config.A @ state
            + fitted_config.B @ u
        )

        predicted_y = fitted_config.C @ predicted

        if np.all(np.isfinite(y)):
            errors.append(
                np.asarray(y, dtype=float) - predicted_y
            )

        state = predicted

    if not errors:
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                np.asarray(errors) ** 2
            )
        )
    )


def _parameter_vectors(
    config: StateSpaceConfig,
    fitted_A: np.ndarray,
    fitted_B: np.ndarray,
    fitted_Q: np.ndarray,
    fitted_R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted and true 14-parameter vectors."""

    estimated = np.concatenate(
        [
            np.asarray(fitted_A[:2, :2], dtype=float).ravel(),
            np.asarray(fitted_B[:2, :], dtype=float).ravel(),
            np.asarray(np.diag(fitted_Q)[:2], dtype=float),
            np.asarray(np.diag(fitted_R), dtype=float),
        ]
    )

    true = np.concatenate(
        [
            np.asarray(config.A[:2, :2], dtype=float).ravel(),
            np.asarray(config.B[:2, :], dtype=float).ravel(),
            np.asarray(np.diag(config.Q)[:2], dtype=float),
            np.asarray(np.diag(config.R), dtype=float),
        ]
    )

    return estimated, true


def run_parameter_recovery(
    config: StateSpaceConfig,
    days: int = 240,
    seed: int = 42,
    maxiter: int = 600,
) -> RecoveryMetrics:
    """Run one controlled synthetic parameter-recovery experiment.

    The estimator receives only:
        - observed outputs
        - input variables

    The latent states used to generate the synthetic data are NOT supplied
    to the estimator.
    """

    dataset = make_synthetic_dataset(
        config,
        days=days,
        seed=seed,
    )

    result = fit_population_core(
        dataset.observations,
        dataset.inputs,
        initial_config=config,
        maxiter=maxiter,
    )

    estimated, true = _parameter_vectors(
        config=config,
        fitted_A=result.A,
        fitted_B=result.B,
        fitted_Q=result.Q,
        fitted_R=result.R,
    )

    total_parameter_rmse = float(
        np.sqrt(
            np.mean(
                (estimated - true) ** 2
            )
        )
    )

    return RecoveryMetrics(
        days=days,
        seed=seed,
        converged=result.converged,
        optimizer_iterations=result.optimizer_iterations,

        A_rmse=_relative_rmse(
            result.A[:2, :2],
            config.A[:2, :2],
        ),

        B_rmse=_relative_rmse(
            result.B[:2],
            config.B[:2],
        ),

        Q_rmse=_relative_rmse(
            np.diag(result.Q)[:2],
            np.diag(config.Q)[:2],
        ),

        R_rmse=_relative_rmse(
            np.diag(result.R),
            np.diag(config.R),
        ),

        total_parameter_rmse=total_parameter_rmse,

        open_loop_observation_rmse=_open_loop_observation_rmse(
            config=config,
            fitted_A=result.A,
            fitted_B=result.B,
            fitted_Q=result.Q,
            fitted_R=result.R,
            dataset=dataset,
        ),

        spectral_radius=spectral_radius(
            result.A
        ),

        log_likelihood=result.llf,
    )