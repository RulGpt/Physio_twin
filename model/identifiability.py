from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_space import INPUT_DIM, STATE_DIM, OBS_DIM, StateSpaceConfig


@dataclass(frozen=True)
class ObservabilityReport:
    """Local linear observability diagnostics for (A, C)."""

    rank: int
    state_dimension: int
    singular_values: np.ndarray
    condition_number: float
    observable: bool


@dataclass(frozen=True)
class ExcitationReport:
    """Rank/conditioning diagnostics for the transition regressor [X, U]."""

    rank: int
    parameter_regressor_dimension: int
    singular_values: np.ndarray
    condition_number: float
    full_rank: bool


@dataclass(frozen=True)
class ParameterPlan:
    """Explicit staged parameterization for estimation."""

    name: str
    free_A: tuple[tuple[int, int], ...]
    free_B: tuple[tuple[int, int], ...]
    free_Q_diagonal: tuple[int, ...]
    free_R_diagonal: tuple[int, ...]
    total_parameters: int
    rationale: str


def observability_matrix(config: StateSpaceConfig) -> np.ndarray:
    """Build O = [C; C A; C A^2] for the 3-state model."""
    blocks = []
    power = np.eye(STATE_DIM)
    for _ in range(STATE_DIM):
        blocks.append(config.C @ power)
        power = power @ config.A
    return np.vstack(blocks)


def observability_report(
    config: StateSpaceConfig,
    rank_tolerance: float = 1e-9,
) -> ObservabilityReport:
    matrix = observability_matrix(config)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix, tol=rank_tolerance))
    nonzero = singular_values[singular_values > rank_tolerance]
    condition_number = (
        float(nonzero.max() / nonzero.min())
        if len(nonzero)
        else float("inf")
    )

    return ObservabilityReport(
        rank=rank,
        state_dimension=STATE_DIM,
        singular_values=singular_values,
        condition_number=condition_number,
        observable=(rank == STATE_DIM),
    )


def transition_regressor(
    states: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    """Construct the known-state transition regressor [X_t, U_t].

    This diagnostic is intentionally a best-case experiment: true hidden
    states are available in synthetic data. Real fitting cannot assume this.
    """
    states = np.asarray(states, dtype=float)
    inputs = np.asarray(inputs, dtype=float)

    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError("states must have shape (n, 3)")
    if inputs.ndim != 2 or inputs.shape[1] != INPUT_DIM:
        raise ValueError("inputs must have shape (n, 3)")
    if len(states) != len(inputs):
        raise ValueError("states and inputs must have equal length")

    return np.column_stack([states, inputs])


def excitation_report(
    states: np.ndarray,
    inputs: np.ndarray,
    rank_tolerance: float = 1e-9,
) -> ExcitationReport:
    XU = transition_regressor(states, inputs)
    singular_values = np.linalg.svd(XU, compute_uv=False)
    rank = int(np.linalg.matrix_rank(XU, tol=rank_tolerance))
    nonzero = singular_values[singular_values > rank_tolerance]
    condition_number = (
        float(nonzero.max() / nonzero.min())
        if len(nonzero)
        else float("inf")
    )

    dimension = STATE_DIM + INPUT_DIM
    return ExcitationReport(
        rank=rank,
        parameter_regressor_dimension=dimension,
        singular_values=singular_values,
        condition_number=condition_number,
        full_rank=(rank == dimension),
    )


def default_parameter_plans() -> tuple[ParameterPlan, ...]:
    """Return progressively richer, explicitly constrained parameter sets."""
    observed_A = (
        (0, 0), (0, 1),
        (1, 0), (1, 1),
    )
    observed_B = tuple((row, col) for row in (0, 1) for col in range(3))

    latent_A = (
        (0, 2), (1, 2),
        (2, 0), (2, 1), (2, 2),
    )
    latent_B = tuple((2, col) for col in range(3))

    return (
        ParameterPlan(
            name="population_observed_core",
            free_A=observed_A,
            free_B=observed_B,
            free_Q_diagonal=(0, 1),
            free_R_diagonal=(0, 1),
            total_parameters=4 + 6 + 2 + 2,
            rationale=(
                "Estimate dynamics directly connected to observed fatigue/"
                "recovery first. Keep latent-load dynamics fixed until "
                "observability/excitation evidence supports expansion."
            ),
        ),
        ParameterPlan(
            name="population_full_latent",
            free_A=observed_A + latent_A,
            free_B=observed_B + latent_B,
            free_Q_diagonal=(0, 1, 2),
            free_R_diagonal=(0, 1),
            total_parameters=9 + 9 + 3 + 2,
            rationale=(
                "Add latent-load couplings only after the core model has "
                "passed recovery/forecast validation."
            ),
        ),
        ParameterPlan(
            name="personal_adaptation_restricted",
            free_A=((0, 0), (1, 1)),
            free_B=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
            free_Q_diagonal=(),
            free_R_diagonal=(),
            total_parameters=2 + 6,
            rationale=(
                "Adapt only high-value observable-state dynamics per person; "
                "keep population latent dynamics and noise fixed initially."
            ),
        ),
    )
