from __future__ import annotations

import numpy as np

from model.preprocessing import build_model_features
from model.schemas import DailyRecord
from model.state_space import (
    StateSpaceConfig,
    filter_step,
)


def make_reference_config() -> StateSpaceConfig:
    """
    Reference PHYSIO-TWIN state-space configuration.

    State:
        [fatigue, recovery, load]

    Input:
        [sleep_deficit, exercise_load, stress]

    Observation:
        [observed_fatigue, observed_recovery]
    """

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


def record_to_model_inputs(
    record: DailyRecord,
    reference_sleep_hours: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert a validated DailyRecord into:

        inputs
        observations
        feature metadata
    """

    features = build_model_features(
        record=record,
        reference_sleep_hours=reference_sleep_hours,
    )

    inputs = np.array(
        [
            features["sleep_deficit_hours"],
            features["exercise_load"],
            features["stress_normalized"],
        ],
        dtype=float,
    )

    observation = np.array(
        [
            features["fatigue_observed_normalized"],
            features["recovery_observed_normalized"],
        ],
        dtype=float,
    )

    metadata = {
        "sleep_deficit_hours": float(
            features["sleep_deficit_hours"]
        ),
        "exercise_load_raw": float(
            features["exercise_load_raw"]
        ),
        "exercise_load_model": float(
            features["exercise_load"]
        ),
        "stress_normalized": float(
            features["stress_normalized"]
        ),
    }

    return inputs, observation, metadata


def update_state(
    state: np.ndarray,
    covariance: np.ndarray,
    record: DailyRecord,
    config: StateSpaceConfig | None = None,
    reference_sleep_hours: float = 8.0,
) -> dict:
    """
    Perform one PHYSIO-TWIN Kalman filtering step.

    No parameter fitting happens here.
    """

    if config is None:
        config = make_reference_config()

    inputs, observation, metadata = (
        record_to_model_inputs(
            record=record,
            reference_sleep_hours=reference_sleep_hours,
        )
    )

    (
        updated_state,
        updated_covariance,
        predicted_state,
        gain,
    ) = filter_step(
        state=np.asarray(state, dtype=float),
        covariance=np.asarray(covariance, dtype=float),
        inputs=inputs,
        observation=observation,
        config=config,
    )

    return {
        "state": updated_state,
        "covariance": updated_covariance,
        "predicted_state": predicted_state,
        "kalman_gain": gain,
        "inputs": inputs,
        "observation": observation,
        "metadata": metadata,
    }