import numpy as np

from model.m3_2_experiment import (
    HORIZONS,
    _metrics,
    _predict_state,
    make_reference_config,
    run_single_experiment,
)


def test_horizons_are_defined():
    assert HORIZONS == (1, 3, 7, 14)


def test_predict_state_is_finite():
    config = make_reference_config()

    state = config.initial_state.copy()
    covariance = config.initial_covariance.copy()
    inputs = np.array([0.2, 0.4, 0.3])

    next_state, next_covariance = _predict_state(
        state,
        covariance,
        inputs,
        config,
    )

    assert next_state.shape == (3,)
    assert next_covariance.shape == (3, 3)

    assert np.all(np.isfinite(next_state))
    assert np.all(np.isfinite(next_covariance))


def test_metrics_are_finite():
    predictions = np.array([
        [0.5, 0.6],
        [0.6, 0.7],
        [0.7, 0.8],
    ])

    observations = np.array([
        [0.4, 0.5],
        [0.5, 0.6],
        [0.6, 0.7],
    ])

    metrics = _metrics(
        predictions,
        observations,
    )

    assert all(np.isfinite(x) for x in metrics[:-1])
    assert metrics[-1] == 3


def test_single_m3_2_experiment():
    config = make_reference_config()

    results = run_single_experiment(
        config,
        days=100,
        seed=42,
    )

    assert len(results) == 12

    models = {
        result.model
        for result in results
    }

    horizons = {
        result.horizon
        for result in results
    }

    assert models == {
        "persistence",
        "population",
        "personalized",
    }

    assert horizons == {
        1,
        3,
        7,
        14,
    }

    for result in results:
        assert result.n_predictions > 0
        assert np.isfinite(result.overall_mae)
        assert np.isfinite(result.overall_rmse)