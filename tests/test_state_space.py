import numpy as np
import pytest

from model.state_space import (
    StateSpaceConfig,
    filter_step,
    forecast,
    is_positive_definite,
    is_positive_semidefinite,
    predict_state,
    spectral_radius,
    update_state,
)


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


def test_matrix_validation_and_stability(config):
    assert config.spectral_radius < 1.0
    config.assert_stable()
    assert is_positive_semidefinite(config.Q)
    assert is_positive_definite(config.R)
    assert np.isclose(config.spectral_radius, spectral_radius(config.A))


def test_unstable_transition_is_rejected():
    config = StateSpaceConfig(
        A=np.eye(3),
        B=np.zeros((3, 3)),
        C=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        Q=np.eye(3),
        R=np.eye(2),
        initial_state=np.zeros(3),
        initial_covariance=np.eye(3),
    )
    with pytest.raises(ValueError, match="strictly stable"):
        config.assert_stable()


def test_prediction_equation(config):
    state = np.array([0.4, 0.6, 0.2])
    covariance = np.eye(3) * 0.1
    inputs = np.array([0.2, 0.5, 0.3])

    predicted_state, predicted_covariance = predict_state(
        state, covariance, inputs, config
    )

    np.testing.assert_allclose(
        predicted_state,
        config.A @ state + config.B @ inputs,
    )
    np.testing.assert_allclose(
        predicted_covariance,
        config.A @ covariance @ config.A.T + config.Q,
    )
    assert is_positive_semidefinite(predicted_covariance)


def test_full_observation_update_reduces_uncertainty(config):
    predicted_state = np.array([0.6, 0.4, 0.3])
    predicted_covariance = np.diag([0.20, 0.20, 0.20])
    observation = np.array([0.5, 0.5])

    updated_state, updated_covariance, gain = update_state(
        predicted_state, predicted_covariance, observation, config
    )

    assert updated_state[0] < predicted_state[0]
    assert updated_state[1] > predicted_state[1]
    assert np.trace(updated_covariance) < np.trace(predicted_covariance)
    assert np.allclose(updated_covariance, updated_covariance.T)
    assert is_positive_semidefinite(updated_covariance)
    assert gain.shape == (3, 2)


def test_partial_observation_updates_only_available_component(config):
    predicted_state = np.array([0.6, 0.4, 0.3])
    predicted_covariance = np.diag([0.20, 0.20, 0.20])
    observation = np.array([0.5, np.nan])

    updated_state, updated_covariance, gain = update_state(
        predicted_state, predicted_covariance, observation, config
    )

    assert updated_state[0] < predicted_state[0]
    assert np.isclose(updated_state[1], predicted_state[1])
    assert np.isclose(gain[1, 1], 0.0)
    assert is_positive_semidefinite(updated_covariance)


def test_all_missing_observations_equal_prediction(config):
    predicted_state = np.array([0.6, 0.4, 0.3])
    predicted_covariance = np.diag([0.20, 0.20, 0.20])

    updated_state, updated_covariance, gain = update_state(
        predicted_state, predicted_covariance, np.array([np.nan, np.nan]), config
    )

    np.testing.assert_allclose(updated_state, predicted_state)
    np.testing.assert_allclose(updated_covariance, predicted_covariance)
    assert np.allclose(gain, 0.0)


def test_filter_step_and_forecast(config):
    state = config.initial_state
    covariance = config.initial_covariance
    inputs = np.array([0.2, 0.4, 0.3])
    observation = np.array([0.55, 0.60])

    state, covariance, predicted, gain = filter_step(
        state, covariance, inputs, observation, config
    )

    assert state.shape == (3,)
    assert covariance.shape == (3, 3)
    assert predicted.shape == (3,)
    assert gain.shape == (3, 2)

    states, covariances = forecast(
        state,
        covariance,
        np.array([
            [0.1, 0.3, 0.2],
            [0.2, 0.1, 0.4],
            [0.0, 0.0, 0.1],
        ]),
        config,
    )

    assert states.shape == (3, 3)
    assert covariances.shape == (3, 3, 3)
    for covariance in covariances:
        assert is_positive_semidefinite(covariance)
