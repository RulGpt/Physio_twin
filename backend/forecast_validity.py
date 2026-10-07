from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VALID_MIN = 0.0
VALID_MAX = 1.0


@dataclass(frozen=True)
class StateValidity:
    value: float
    status: str


def classify_state_value(value: float) -> StateValidity:
    """
    Classify a normalized PHYSIO-TWIN latent state.

    The mathematical model is allowed to produce values outside
    the normalized observation scale. We do NOT clip those values.

    Status:
        within_scale
        below_scale
        above_scale
    """

    value = float(value)

    if not np.isfinite(value):
        return StateValidity(
            value=value,
            status="invalid_numeric",
        )

    if value < VALID_MIN:
        return StateValidity(
            value=value,
            status="below_scale",
        )

    if value > VALID_MAX:
        return StateValidity(
            value=value,
            status="above_scale",
        )

    return StateValidity(
        value=value,
        status="within_scale",
    )


def classify_forecast_state(
    fatigue: float,
    recovery: float,
    load: float,
) -> dict[str, StateValidity]:
    """
    Classify all three PHYSIO-TWIN latent states.
    """

    return {
        "fatigue": classify_state_value(fatigue),
        "recovery": classify_state_value(recovery),
        "load": classify_state_value(load),
    }


def forecast_is_within_scale(
    fatigue: float,
    recovery: float,
    load: float,
) -> bool:
    """
    Return True only when all forecast states are within [0, 1].
    """

    states = classify_forecast_state(
        fatigue=fatigue,
        recovery=recovery,
        load=load,
    )

    return all(
        state.status == "within_scale"
        for state in states.values()
    )


def validity_to_dict(
    fatigue: float,
    recovery: float,
    load: float,
) -> dict[str, dict[str, float | str]]:
    """
    Convert forecast validity information into an API-friendly dict.
    """

    states = classify_forecast_state(
        fatigue=fatigue,
        recovery=recovery,
        load=load,
    )

    return {
        name: {
            "value": state.value,
            "status": state.status,
        }
        for name, state in states.items()
    }