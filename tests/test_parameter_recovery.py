import numpy as np
import pytest

from model.estimator import PopulationCoreMLE, fit_population_core
from model.synthetic import make_synthetic_dataset
from model.parameter_recovery import run_parameter_recovery
from model.state_space import StateSpaceConfig, spectral_radius


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


def test_parameter_model_exposes_14_parameters(config):
    data = make_synthetic_dataset(config, days=30, seed=1)
    model = PopulationCoreMLE(data.observations, data.inputs, config)

    assert len(model.start_params) == 14
    assert len(model.param_names) == 14


def test_variance_transform_is_positive(config):
    data = make_synthetic_dataset(config, days=30, seed=1)
    model = PopulationCoreMLE(data.observations, data.inputs, config)

    transformed = model.transform_params(
        np.array([0.0] * 10 + [-10.0, 0.0, -5.0, 1.0])
    )

    assert np.all(transformed[10:] > 0)


def test_fit_returns_finite_parameters(config):
    data = make_synthetic_dataset(config, days=100, seed=3)
    result = fit_population_core(
        data.observations,
        data.inputs,
        config,
        maxiter=40,
    )

    assert result.A.shape == (3, 3)
    assert result.B.shape == (3, 3)
    assert result.Q.shape == (3, 3)
    assert result.R.shape == (2, 2)
    assert np.all(np.isfinite(result.constrained_params))
    assert np.isfinite(result.llf)
    assert spectral_radius(result.A) < 0.995


def test_parameter_recovery_experiment_is_reproducible(config):
    a = run_parameter_recovery(config, days=100, seed=10, maxiter=40)
    b = run_parameter_recovery(config, days=100, seed=10, maxiter=40)

    assert a.days == b.days
    assert np.isclose(a.A_rmse, b.A_rmse)
    assert np.isclose(a.B_rmse, b.B_rmse)
    assert np.isclose(a.log_likelihood, b.log_likelihood)


def test_parameter_recovery_metrics_are_finite(config):
    metrics = run_parameter_recovery(
        config,
        days=180,
        seed=42,
        maxiter=60,
    )

    assert metrics.A_rmse >= 0
    assert metrics.B_rmse >= 0
    assert metrics.Q_rmse >= 0
    assert metrics.R_rmse >= 0
    assert metrics.total_parameter_rmse >= 0
    assert metrics.open_loop_observation_rmse >= 0
    assert metrics.spectral_radius >= 0
    assert np.isfinite(metrics.log_likelihood)
    assert (
        metrics.optimizer_iterations is None
        or metrics.optimizer_iterations >= 0
    )


def test_shorter_history_is_allowed_but_not_assumed_sufficient(config):
    short = run_parameter_recovery(
        config,
        days=60,
        seed=2,
        maxiter=30,
    )
    long = run_parameter_recovery(
        config,
        days=180,
        seed=2,
        maxiter=30,
    )

    # This test deliberately does not assert that long data must always win.
    # It only establishes that both experiments run and produce measurable
    # quantities; empirical recovery quality is reported, not hard-coded.
    assert short.days == 60
    assert long.days == 180
    assert np.isfinite(short.A_rmse)
    assert np.isfinite(long.A_rmse)
