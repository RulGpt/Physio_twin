from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_space import STATE_DIM, INPUT_DIM, OBS_DIM, StateSpaceConfig


@dataclass(frozen=True)
class SyntheticDataset:
    """Ground-truth longitudinal sequence plus noisy observations."""

    true_states: np.ndarray
    inputs: np.ndarray
    observations: np.ndarray
    missing_observation_mask: np.ndarray


def simulate_true_states(
    config: StateSpaceConfig,
    inputs: np.ndarray,
    rng: np.random.Generator,
    initial_state: np.ndarray | None = None,
) -> np.ndarray:
    """Generate hidden states using the canonical state-space timing.

    The sequence follows:

        X_0 = initial_state
        X_t = A X_{t-1} + B U_{t-1} + w_{t-1}

    Therefore inputs[t] affect state[t+1], not state[t].
    This matches the observation convention:

        Y_t = C X_t + v_t
    """
    inputs = np.asarray(inputs, dtype=float)

    if inputs.ndim != 2 or inputs.shape[1] != INPUT_DIM:
        raise ValueError("inputs must have shape (days, 3)")

    if not np.all(np.isfinite(inputs)):
        raise ValueError("inputs must contain only finite values")

    state = (
        np.asarray(initial_state, dtype=float).copy()
        if initial_state is not None
        else config.initial_state.copy()
    )

    if state.shape != (STATE_DIM,):
        raise ValueError("initial_state must have shape (3,)")

    states = np.empty((len(inputs), STATE_DIM), dtype=float)

    # First observation corresponds to the known initial state.
    states[0] = state

    # U[t-1] drives the transition from X[t-1] to X[t].
    for day in range(1, len(inputs)):
        process_noise = rng.multivariate_normal(
            mean=np.zeros(STATE_DIM),
            cov=config.Q,
        )

        state = (
            config.A @ state
            + config.B @ inputs[day - 1]
            + process_noise
        )

        states[day] = state

    return states


def simulate_observations(
    config: StateSpaceConfig,
    true_states: np.ndarray,
    rng: np.random.Generator,
    missing_probability: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy fatigue/recovery observations.

    The load state is not directly observed. Missingness is MCAR in this
    controlled experiment and is represented by NaN.
    """
    true_states = np.asarray(true_states, dtype=float)
    if true_states.ndim != 2 or true_states.shape[1] != STATE_DIM:
        raise ValueError("true_states must have shape (days, 3)")
    if not 0.0 <= missing_probability < 1.0:
        raise ValueError("missing_probability must be in [0, 1)")

    observations = true_states @ config.C.T
    measurement_noise = rng.multivariate_normal(
        mean=np.zeros(OBS_DIM),
        cov=config.R,
        size=len(true_states),
    )
    observations = observations + measurement_noise

    missing_mask = rng.random(observations.shape) < missing_probability
    observations[missing_mask] = np.nan

    return observations, missing_mask


def make_synthetic_dataset(
    config: StateSpaceConfig,
    days: int = 120,
    seed: int = 42,
    missing_probability: float = 0.0,
) -> SyntheticDataset:
    """Create a reproducible controlled longitudinal experiment.

    Inputs are generated from bounded, interpretable routine variables:
    sleep deficit, exercise load, and normalized stress.
    """
    if days < 2:
        raise ValueError("days must be at least 2")

    rng = np.random.default_rng(seed)

    sleep_deficit = np.clip(
        rng.normal(loc=0.8, scale=0.45, size=days),
        0.0,
        2.5,
    )
    exercise_load = np.clip(
        rng.gamma(shape=2.0, scale=0.8, size=days),
        0.0,
        6.0,
    )
    stress = np.clip(
        rng.normal(loc=0.45, scale=0.18, size=days),
        0.0,
        1.0,
    )
    inputs = np.column_stack([sleep_deficit, exercise_load, stress])

    true_states = simulate_true_states(config, inputs, rng)
    observations, missing_mask = simulate_observations(
        config,
        true_states,
        rng,
        missing_probability=missing_probability,
    )

    return SyntheticDataset(
        true_states=true_states,
        inputs=inputs,
        observations=observations,
        missing_observation_mask=missing_mask,
    )
