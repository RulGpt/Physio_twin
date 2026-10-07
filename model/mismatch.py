from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_space import INPUT_DIM, OBS_DIM, STATE_DIM, StateSpaceConfig


@dataclass(frozen=True)
class MismatchDataset:
    true_states: np.ndarray
    inputs: np.ndarray
    observations: np.ndarray
    scenario: str


def _base_inputs(days: int, rng: np.random.Generator) -> np.ndarray:
    sleep_deficit = np.clip(rng.normal(0.8, 0.45, days), 0.0, 2.5)
    exercise_load = np.clip(rng.gamma(2.0, 0.8, days), 0.0, 6.0)
    stress = np.clip(rng.normal(0.45, 0.18, days), 0.0, 1.0)
    return np.column_stack([sleep_deficit, exercise_load, stress])


def _draw_noise(
    covariance: np.ndarray,
    rng: np.random.Generator,
    size: int = 1,
) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(len(covariance)), covariance, size=size)


def simulate_mismatched_process(
    config: StateSpaceConfig,
    days: int,
    seed: int,
    scenario: str,
) -> MismatchDataset:
    """Generate data from a process that intentionally differs from PHYSIO-TWIN.

    Scenarios:
    - nonlinear_saturation: exercise response saturates and stress interacts
      with exercise.
    - delayed_sleep: sleep deficit affects the state with a one-day delay.
    - correlated_noise: measurement errors are correlated rather than diagonal.
    - person_shift: the true person has different transition/input parameters.
    """
    if days < 10:
        raise ValueError("days must be at least 10")

    rng = np.random.default_rng(seed)
    inputs = _base_inputs(days, rng)

    A = config.A.copy()
    B = config.B.copy()
    Q = config.Q.copy()

    state = config.initial_state.copy()
    states = np.empty((days, STATE_DIM), dtype=float)

    measurement_cov = config.R.copy()

    if scenario == "correlated_noise":
        measurement_cov = np.array([
            [0.04, 0.025],
            [0.025, 0.04],
        ])

    if scenario == "person_shift":
        A = A.copy()
        B = B.copy()
        A[0, 0] = 0.82
        A[1, 1] = 0.68
        A[2, 2] = 0.68
        B[0, 0] = 0.14
        B[1, 1] = -0.045
        B[2, 1] = 0.16

    for day, u in enumerate(inputs):
        d, e, s = u

        if scenario == "nonlinear_saturation":
            effective_e = 2.0 * (1.0 - np.exp(-e / 2.0))
            effective_u = np.array([d, effective_e, s])
            interaction = np.array([
                0.06 * s * effective_e,
                -0.03 * d * s,
                0.04 * np.tanh(e - 1.0),
            ])
            state = A @ state + B @ effective_u + interaction
        elif scenario == "delayed_sleep":
            previous_d = inputs[day - 1, 0] if day > 0 else d
            effective_u = np.array([previous_d, e, s])
            state = A @ state + B @ effective_u
        elif scenario in {"correlated_noise", "person_shift"}:
            state = A @ state + B @ u
        else:
            raise ValueError(
                "scenario must be one of: nonlinear_saturation, "
                "delayed_sleep, correlated_noise, person_shift"
            )

        process_noise = rng.multivariate_normal(
            np.zeros(STATE_DIM),
            Q,
        )
        state = state + process_noise
        states[day] = state

    observations = states[:, :2] + _draw_noise(
        measurement_cov,
        rng,
        size=days,
    )

    return MismatchDataset(
        true_states=states,
        inputs=inputs,
        observations=observations,
        scenario=scenario,
    )


def compare_scenarios(
    config: StateSpaceConfig,
    scenarios: tuple[str, ...] = (
        "nonlinear_saturation",
        "delayed_sleep",
        "correlated_noise",
        "person_shift",
    ),
    days: int = 180,
    seed: int = 42,
) -> dict[str, MismatchDataset]:
    return {
        name: simulate_mismatched_process(config, days, seed, name)
        for name in scenarios
    }
