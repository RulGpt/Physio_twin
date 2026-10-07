from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_space import (
    StateSpaceConfig,
    filter_step,
)


@dataclass(frozen=True)
class PredictionMetrics:
    """One-step-ahead predictive performance."""

    n_predictions: int

    fatigue_mae: float
    recovery_mae: float
    overall_mae: float

    fatigue_rmse: float
    recovery_rmse: float
    overall_rmse: float


@dataclass(frozen=True)
class PredictionResult:
    """Predictions and corresponding observations."""

    predictions: np.ndarray
    observations: np.ndarray
    metrics: PredictionMetrics


@dataclass(frozen=True)
class ChronologicalSplit:
    """Chronological train/test split."""

    train_inputs: np.ndarray
    train_observations: np.ndarray

    test_inputs: np.ndarray
    test_observations: np.ndarray


def chronological_split(
    inputs: np.ndarray,
    observations: np.ndarray,
    train_days: int,
) -> ChronologicalSplit:
    """
    Split longitudinal data chronologically.

    No shuffling is performed.
    """

    inputs = np.asarray(inputs, dtype=float)
    observations = np.asarray(observations, dtype=float)

    if inputs.ndim != 2 or inputs.shape[1] != 3:
        raise ValueError(
            "inputs must have shape (n, 3)"
        )

    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(
            "observations must have shape (n, 2)"
        )

    if len(inputs) != len(observations):
        raise ValueError(
            "inputs and observations must have equal length"
        )

    if train_days < 2:
        raise ValueError(
            "train_days must be at least 2"
        )

    if train_days >= len(inputs):
        raise ValueError(
            "train_days must leave at least one test observation"
        )

    return ChronologicalSplit(
        train_inputs=inputs[:train_days].copy(),
        train_observations=observations[:train_days].copy(),
        test_inputs=inputs[train_days:].copy(),
        test_observations=observations[train_days:].copy(),
    )


def _calculate_metrics(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> PredictionMetrics:
    """Calculate MAE and RMSE for two observed outputs."""

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    observations = np.asarray(
        observations,
        dtype=float,
    )

    if predictions.shape != observations.shape:
        raise ValueError(
            "predictions and observations must have equal shape"
        )

    valid = (
        np.all(np.isfinite(predictions), axis=1)
        & np.all(np.isfinite(observations), axis=1)
    )

    predictions = predictions[valid]
    observations = observations[valid]

    if len(predictions) == 0:
        raise ValueError(
            "No valid prediction pairs available"
        )

    errors = predictions - observations

    fatigue_errors = errors[:, 0]
    recovery_errors = errors[:, 1]

    fatigue_mae = float(
        np.mean(np.abs(fatigue_errors))
    )

    recovery_mae = float(
        np.mean(np.abs(recovery_errors))
    )

    overall_mae = float(
        np.mean(np.abs(errors))
    )

    fatigue_rmse = float(
        np.sqrt(
            np.mean(fatigue_errors ** 2)
        )
    )

    recovery_rmse = float(
        np.sqrt(
            np.mean(recovery_errors ** 2)
        )
    )

    overall_rmse = float(
        np.sqrt(
            np.mean(errors ** 2)
        )
    )

    return PredictionMetrics(
        n_predictions=len(predictions),
        fatigue_mae=fatigue_mae,
        recovery_mae=recovery_mae,
        overall_mae=overall_mae,
        fatigue_rmse=fatigue_rmse,
        recovery_rmse=recovery_rmse,
        overall_rmse=overall_rmse,
    )


def persistence_predictions(
    observations: np.ndarray,
) -> PredictionResult:
    """
    One-step persistence baseline.

    Prediction for day t+1:
        Y_hat[t+1] = Y[t]
    """

    observations = np.asarray(
        observations,
        dtype=float,
    )

    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(
            "observations must have shape (n, 2)"
        )

    if len(observations) < 2:
        raise ValueError(
            "At least two observations are required"
        )

    predictions = observations[:-1].copy()
    targets = observations[1:].copy()

    metrics = _calculate_metrics(
        predictions,
        targets,
    )

    return PredictionResult(
        predictions=predictions,
        observations=targets,
        metrics=metrics,
    )


def state_space_one_step_predictions(
    config: StateSpaceConfig,
    inputs: np.ndarray,
    observations: np.ndarray,
) -> PredictionResult:
    """
    Generate chronological one-step-ahead predictions.

    At time t:

        1. Current state estimate is available.
        2. Current observation is incorporated.
        3. Current input drives the transition.
        4. State at t+1 is predicted.
        5. Observation at t+1 is evaluated.

    No future observation is used before its prediction.
    """

    inputs = np.asarray(
        inputs,
        dtype=float,
    )

    observations = np.asarray(
        observations,
        dtype=float,
    )

    if inputs.ndim != 2 or inputs.shape[1] != 3:
        raise ValueError(
            "inputs must have shape (n, 3)"
        )

    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(
            "observations must have shape (n, 2)"
        )

    if len(inputs) != len(observations):
        raise ValueError(
            "inputs and observations must have equal length"
        )

    if len(inputs) < 2:
        raise ValueError(
            "At least two observations are required"
        )

    state = config.initial_state.copy()
    covariance = config.initial_covariance.copy()

    predictions = []
    targets = []

    for t in range(len(observations) - 1):

        current_observation = observations[t]

        state, covariance, _, _ = filter_step(
            state=state,
            covariance=covariance,
            inputs=inputs[t],
            observation=current_observation,
            config=config,
        )

        next_state = (
            config.A @ state
            + config.B @ inputs[t]
        )

        predicted_observation = (
            config.C @ next_state
        )

        next_observation = observations[t + 1]

        if (
            np.all(np.isfinite(predicted_observation))
            and np.all(np.isfinite(next_observation))
        ):
            predictions.append(
                predicted_observation
            )

            targets.append(
                next_observation
            )

    if not predictions:
        raise ValueError(
            "No valid one-step predictions were generated"
        )

    predictions_array = np.asarray(
        predictions,
        dtype=float,
    )

    targets_array = np.asarray(
        targets,
        dtype=float,
    )

    metrics = _calculate_metrics(
        predictions_array,
        targets_array,
    )

    return PredictionResult(
        predictions=predictions_array,
        observations=targets_array,
        metrics=metrics,
    )