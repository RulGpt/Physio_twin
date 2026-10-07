import numpy as np
import pytest

from model.experiment import (
    forecast_rmse,
    mae_by_state,
    rmse_by_state,
    run_filter,
)
from model.state_space import StateSpaceConfig
from model.synthetic import (
    make_synthetic_dataset,
    simulate_observations,
    simulate_true_states,
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


def test_synthetic_shapes_and_reproducibility(config):
    a = make_synthetic_dataset(config, days=100, seed=7)
    b = make_synthetic_dataset(config, days=100, seed=7)

    assert a.true_states.shape == (100, 3)
    assert a.inputs.shape == (100, 3)
    assert a.observations.shape == (100, 2)
    np.testing.assert_allclose(a.true_states, b.true_states)
    np.testing.assert_allclose(a.observations, b.observations)


def test_true_state_simulation_is_independent_of_observation_generation(config):
    rng = np.random.default_rng(1)
    inputs = np.ones((20, 3)) * 0.2
    true_states = simulate_true_states(config, inputs, rng)

    observations, mask = simulate_observations(
        config,
        true_states,
        np.random.default_rng(2),
    )

    assert mask.sum() == 0
    assert not np.allclose(observations, true_states[:, :2])


def test_filter_recovers_hidden_states_reasonably(config):
    dataset = make_synthetic_dataset(config, days=180, seed=42)
    filtered = run_filter(config, dataset)

    # The first days include cold-start effects. Evaluate the mature portion.
    truth = dataset.true_states[30:]
    estimate = filtered.states[30:]
    rmse = rmse_by_state(truth, estimate)
    mae = mae_by_state(truth, estimate)

    assert np.all(np.isfinite(rmse))
    assert np.all(np.isfinite(mae))
    assert rmse[0] < 0.30
    assert rmse[1] < 0.30
    assert rmse[2] < 0.50


def test_latent_load_is_estimable_but_less_directly_observed(config):
    dataset = make_synthetic_dataset(config, days=180, seed=42)
    filtered = run_filter(config, dataset)

    rmse = rmse_by_state(dataset.true_states[30:], filtered.states[30:])

    # Load has no direct observation channel, so we do not demand it match
    # fatigue/recovery accuracy. We only verify finite, bounded reconstruction.
    assert np.isfinite(rmse[2])
    assert rmse[2] < 0.50


def test_missing_observations_do_not_break_filter(config):
    dataset = make_synthetic_dataset(
        config,
        days=150,
        seed=11,
        missing_probability=0.25,
    )
    filtered = run_filter(config, dataset)

    assert np.all(np.isfinite(filtered.states))
    assert np.all(np.isfinite(filtered.covariances))

    missing_days = np.all(dataset.missing_observation_mask, axis=1)
    assert np.any(missing_days)

    for i in np.where(missing_days)[0]:
        # On a fully missing day, update_state is prediction-only.
        np.testing.assert_allclose(
            filtered.states[i],
            filtered.predicted_states[i],
        )


def test_uncertainty_grows_during_long_missing_run(config):
    dataset = make_synthetic_dataset(config, days=80, seed=5)
    dataset.observations[30:45, :] = np.nan

    filtered = run_filter(config, dataset)

    before = np.trace(filtered.covariances[29])
    during = np.trace(filtered.covariances[44])
    after = np.trace(filtered.covariances[45])

    assert during > before
    assert after < during


def test_short_horizon_forecast_is_finite(config):
    dataset = make_synthetic_dataset(config, days=100, seed=9)
    filtered = run_filter(config, dataset)

    rmse = forecast_rmse(
        config,
        filtered,
        dataset,
        start_index=70,
        horizon=7,
    )

    assert rmse.shape == (3,)
    assert np.all(np.isfinite(rmse))


def test_invalid_synthetic_arguments(config):
    with pytest.raises(ValueError):
        make_synthetic_dataset(config, days=1)

    with pytest.raises(ValueError):
        make_synthetic_dataset(config, days=10, missing_probability=1.0)
