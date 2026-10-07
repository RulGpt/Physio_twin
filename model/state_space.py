from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_DIM = 3
INPUT_DIM = 3
OBS_DIM = 2
EPSILON = 1e-10


@dataclass(frozen=True)
class StateSpaceConfig:
    """Validated linear state-space configuration for PHYSIO-TWIN."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    initial_state: np.ndarray
    initial_covariance: np.ndarray

    def __post_init__(self) -> None:
        matrices = {
            "A": (self.A, (STATE_DIM, STATE_DIM)),
            "B": (self.B, (STATE_DIM, INPUT_DIM)),
            "C": (self.C, (OBS_DIM, STATE_DIM)),
            "Q": (self.Q, (STATE_DIM, STATE_DIM)),
            "R": (self.R, (OBS_DIM, OBS_DIM)),
            "initial_state": (self.initial_state, (STATE_DIM,)),
            "initial_covariance": (
                self.initial_covariance,
                (STATE_DIM, STATE_DIM),
            ),
        }

        for name, (value, shape) in matrices.items():
            array = np.asarray(value, dtype=float)
            if array.shape != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")

        if not is_positive_semidefinite(self.Q):
            raise ValueError("Q must be positive semidefinite")
        if not is_positive_definite(self.R):
            raise ValueError("R must be positive definite")
        if not is_positive_semidefinite(self.initial_covariance):
            raise ValueError("initial_covariance must be positive semidefinite")

    @property
    def spectral_radius(self) -> float:
        return spectral_radius(self.A)

    def assert_stable(self, tolerance: float = EPSILON) -> None:
        if self.spectral_radius >= 1.0 - tolerance:
            raise ValueError(
                "Transition matrix A is not strictly stable: "
                f"spectral radius={self.spectral_radius:.6g}"
            )


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def is_positive_semidefinite(matrix: np.ndarray, tolerance: float = 1e-9) -> bool:
    matrix = np.asarray(matrix, dtype=float)
    if not np.allclose(matrix, matrix.T, atol=tolerance):
        return False
    eigenvalues = np.linalg.eigvalsh(_symmetric(matrix))
    return bool(np.min(eigenvalues) >= -tolerance)


def is_positive_definite(matrix: np.ndarray, tolerance: float = 1e-9) -> bool:
    matrix = np.asarray(matrix, dtype=float)
    if not np.allclose(matrix, matrix.T, atol=tolerance):
        return False
    eigenvalues = np.linalg.eigvalsh(_symmetric(matrix))
    return bool(np.min(eigenvalues) > tolerance)


def spectral_radius(A: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(np.asarray(A, dtype=float))
    return float(np.max(np.abs(eigenvalues)))


def predict_state(
    state: np.ndarray,
    covariance: np.ndarray,
    inputs: np.ndarray,
    config: StateSpaceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """One state prediction step: X' = A X + B U + w."""
    state = np.asarray(state, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    inputs = np.asarray(inputs, dtype=float)

    if state.shape != (STATE_DIM,):
        raise ValueError("state must have shape (3,)")
    if covariance.shape != (STATE_DIM, STATE_DIM):
        raise ValueError("covariance must have shape (3, 3)")
    if inputs.shape != (INPUT_DIM,):
        raise ValueError("inputs must have shape (3,)")

    predicted_state = config.A @ state + config.B @ inputs
    predicted_covariance = _symmetric(
        config.A @ covariance @ config.A.T + config.Q
    )
    return predicted_state, predicted_covariance


def update_state(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    observation: np.ndarray,
    config: StateSpaceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measurement update, supporting partially missing observations.

    Missing observations are represented by NaN. Only observed components
    participate in the update; if all components are missing, this is a
    prediction-only step.
    """
    predicted_state = np.asarray(predicted_state, dtype=float)
    predicted_covariance = np.asarray(predicted_covariance, dtype=float)
    observation = np.asarray(observation, dtype=float)

    if predicted_state.shape != (STATE_DIM,):
        raise ValueError("predicted_state must have shape (3,)")
    if predicted_covariance.shape != (STATE_DIM, STATE_DIM):
        raise ValueError("predicted_covariance must have shape (3, 3)")
    if observation.shape != (OBS_DIM,):
        raise ValueError("observation must have shape (2,)")

    observed = np.isfinite(observation)
    if not np.any(observed):
        return predicted_state, _symmetric(predicted_covariance), np.zeros(
            (STATE_DIM, OBS_DIM)
        )

    C = config.C[observed, :]
    R = config.R[np.ix_(observed, observed)]
    y = observation[observed]

    innovation = y - C @ predicted_state
    innovation_covariance = _symmetric(
        C @ predicted_covariance @ C.T + R
    )

    # Solve S K^T = C P instead of explicitly calculating S^{-1}.
    cross_covariance = predicted_covariance @ C.T
    kalman_gain_reduced = np.linalg.solve(
        innovation_covariance, cross_covariance.T
    ).T

    updated_state = predicted_state + kalman_gain_reduced @ innovation

    identity = np.eye(STATE_DIM)
    joseph_covariance = (
        (identity - kalman_gain_reduced @ C)
        @ predicted_covariance
        @ (identity - kalman_gain_reduced @ C).T
        + kalman_gain_reduced @ R @ kalman_gain_reduced.T
    )
    updated_covariance = _symmetric(joseph_covariance)

    full_gain = np.zeros((STATE_DIM, OBS_DIM))
    full_gain[:, observed] = kalman_gain_reduced
    return updated_state, updated_covariance, full_gain


def filter_step(
    state: np.ndarray,
    covariance: np.ndarray,
    inputs: np.ndarray,
    observation: np.ndarray,
    config: StateSpaceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one prediction + measurement update step."""
    predicted_state, predicted_covariance = predict_state(
        state, covariance, inputs, config
    )
    updated_state, updated_covariance, gain = update_state(
        predicted_state, predicted_covariance, observation, config
    )
    return updated_state, updated_covariance, predicted_state, gain


def forecast(
    state: np.ndarray,
    covariance: np.ndarray,
    future_inputs: np.ndarray,
    config: StateSpaceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast states for a sequence of future input vectors."""
    future_inputs = np.asarray(future_inputs, dtype=float)
    if future_inputs.ndim != 2 or future_inputs.shape[1] != INPUT_DIM:
        raise ValueError("future_inputs must have shape (horizon, 3)")
    if not np.all(np.isfinite(future_inputs)):
        raise ValueError("future_inputs must contain only finite values")

    current_state = np.asarray(state, dtype=float)
    current_covariance = np.asarray(covariance, dtype=float)
    states = []
    covariances = []

    for inputs in future_inputs:
        current_state, current_covariance = predict_state(
            current_state, current_covariance, inputs, config
        )
        states.append(current_state.copy())
        covariances.append(current_covariance.copy())

    if not states:
        return np.empty((0, STATE_DIM)), np.empty((0, STATE_DIM, STATE_DIM))

    return np.stack(states), np.stack(covariances)
