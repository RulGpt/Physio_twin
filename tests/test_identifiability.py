import numpy as np
import pytest

from model.identifiability import (
    default_parameter_plans,
    excitation_report,
    observability_matrix,
    observability_report,
    transition_regressor,
)
from model.synthetic import make_synthetic_dataset
from model.state_space import StateSpaceConfig


@pytest.fixture
def config():
    return StateSpaceConfig(
        A=np.array([
            [0.70, 0.05, 0.05],
            [0.03, 0.75, 0.02],
            [0.02, 0.03, 0.60],
        ]),
        B=np.array([
            [0.10, 0.08, 0.05],
            [-0.04, -0.03, -0.02],
            [0.08, 0.12, 0.04],
        ]),
        C=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
        Q=np.diag([0.01, 0.01, 0.02]),
        R=np.diag([0.04, 0.04]),
        initial_state=np.array([0.5, 0.5, 0.2]),
        initial_covariance=np.diag([0.10, 0.10, 0.20]),
    )


def test_observability_matrix_shape(config):
    matrix = observability_matrix(config)
    assert matrix.shape == (6, 3)


def test_current_model_is_locally_observable(config):
    report = observability_report(config)
    assert report.rank == 3
    assert report.observable
    assert np.all(np.isfinite(report.singular_values))
    assert report.condition_number < 1e6


def test_decoupled_load_is_not_observable(config):
    A = config.A.copy()
    A[0, 2] = 0.0
    A[1, 2] = 0.0
    decoupled = StateSpaceConfig(
        A=A,
        B=config.B,
        C=config.C,
        Q=config.Q,
        R=config.R,
        initial_state=config.initial_state,
        initial_covariance=config.initial_covariance,
    )
    report = observability_report(decoupled)
    assert report.rank == 2
    assert not report.observable


def test_transition_regressor_shape():
    states = np.ones((20, 3))
    inputs = np.ones((20, 3))
    XU = transition_regressor(states, inputs)
    assert XU.shape == (20, 6)


def test_synthetic_inputs_can_excitate_known_state_regressor(config):
    dataset = make_synthetic_dataset(config, days=250, seed=42)
    report = excitation_report(dataset.true_states[:-1], dataset.inputs[:-1])

    assert report.rank == 6
    assert report.full_rank
    assert np.isfinite(report.condition_number)


def test_constant_inputs_can_destroy_excitation():
    states = np.random.default_rng(1).normal(size=(100, 3))
    inputs = np.ones((100, 3))
    report = excitation_report(states, inputs)

    assert report.rank < 6
    assert not report.full_rank


def test_parameter_plan_is_explicit_and_progressive():
    plans = default_parameter_plans()

    assert [p.name for p in plans] == [
        "population_observed_core",
        "population_full_latent",
        "personal_adaptation_restricted",
    ]

    assert plans[0].total_parameters == 14
    assert plans[1].total_parameters == 23
    assert plans[2].total_parameters == 8

    for plan in plans:
        assert plan.total_parameters > 0
        assert plan.rationale


def test_invalid_transition_regressor_inputs():
    with pytest.raises(ValueError):
        transition_regressor(np.ones((10, 2)), np.ones((10, 3)))

    with pytest.raises(ValueError):
        transition_regressor(np.ones((10, 3)), np.ones((9, 3)))
