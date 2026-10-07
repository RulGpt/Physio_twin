from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_space import StateSpaceConfig, filter_step, forecast
from .synthetic import SyntheticDataset


@dataclass(frozen=True)
class FilteredDataset:
    states: np.ndarray
    covariances: np.ndarray
    predicted_states: np.ndarray
    gains: np.ndarray


def run_filter(
    config: StateSpaceConfig,
    dataset: SyntheticDataset,
) -> FilteredDataset:
    """Run the PHYSIO-TWIN filter over a synthetic longitudinal sequence."""
    state = config.initial_state.copy()
    covariance = config.initial_covariance.copy()

    states = []
    covariances = []
    predicted_states = []
    gains = []

    for inputs, observation in zip(dataset.inputs, dataset.observations):
        state, covariance, predicted, gain = filter_step(
            state,
            covariance,
            inputs,
            observation,
            config,
        )
        states.append(state.copy())
        covariances.append(covariance.copy())
        predicted_states.append(predicted.copy())
        gains.append(gain.copy())

    return FilteredDataset(
        states=np.stack(states),
        covariances=np.stack(covariances),
        predicted_states=np.stack(predicted_states),
        gains=np.stack(gains),
    )


def rmse_by_state(true_states: np.ndarray, estimated_states: np.ndarray) -> np.ndarray:
    """Return RMSE independently for fatigue, recovery, and load."""
    true_states = np.asarray(true_states, dtype=float)
    estimated_states = np.asarray(estimated_states, dtype=float)
    if true_states.shape != estimated_states.shape:
        raise ValueError("true_states and estimated_states must have equal shape")
    return np.sqrt(np.mean((true_states - estimated_states) ** 2, axis=0))


def mae_by_state(true_states: np.ndarray, estimated_states: np.ndarray) -> np.ndarray:
    """Return MAE independently for fatigue, recovery, and load."""
    true_states = np.asarray(true_states, dtype=float)
    estimated_states = np.asarray(estimated_states, dtype=float)
    if true_states.shape != estimated_states.shape:
        raise ValueError("true_states and estimated_states must have equal shape")
    return np.mean(np.abs(true_states - estimated_states), axis=0)


def forecast_rmse(
    config: StateSpaceConfig,
    filtered: FilteredDataset,
    dataset: SyntheticDataset,
    start_index: int,
    horizon: int,
) -> np.ndarray:
    """Forecast future hidden states from a filtered state and measure RMSE.

    Future inputs are assumed known in this controlled experiment.
    """
    if start_index < 0 or start_index >= len(dataset.true_states):
        raise ValueError("start_index out of range")
    if horizon < 1 or start_index + horizon > len(dataset.true_states):
        raise ValueError("horizon exceeds available future data")

    predicted, _ = forecast(
        filtered.states[start_index],
        filtered.covariances[start_index],
        dataset.inputs[start_index + 1 : start_index + horizon + 1],
        config,
    )
    truth = dataset.true_states[start_index + 1 : start_index + horizon + 1]
    return rmse_by_state(truth, predicted)
