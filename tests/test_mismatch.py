import numpy as np
import pytest

from model.mismatch import compare_scenarios, simulate_mismatched_process
from model.robustness import evaluate_all_mismatches, evaluate_mismatch
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


def test_each_mismatch_has_expected_shape(config):
    for scenario in (
        "nonlinear_saturation",
        "delayed_sleep",
        "correlated_noise",
        "person_shift",
    ):
        data = simulate_mismatched_process(
            config, days=80, seed=1, scenario=scenario
        )
        assert data.true_states.shape == (80, 3)
        assert data.inputs.shape == (80, 3)
        assert data.observations.shape == (80, 2)
        assert np.all(np.isfinite(data.true_states))
        assert np.all(np.isfinite(data.observations))


def test_mismatch_generator_is_reproducible(config):
    a = simulate_mismatched_process(
        config, days=80, seed=12, scenario="delayed_sleep"
    )
    b = simulate_mismatched_process(
        config, days=80, seed=12, scenario="delayed_sleep"
    )
    np.testing.assert_allclose(a.true_states, b.true_states)
    np.testing.assert_allclose(a.observations, b.observations)


def test_invalid_scenario_is_rejected(config):
    with pytest.raises(ValueError):
        simulate_mismatched_process(
            config, days=20, seed=1, scenario="made_up_scenario"
        )


def test_all_scenarios_are_distinct(config):
    datasets = compare_scenarios(config, days=80, seed=5)
    assert set(datasets) == {
        "nonlinear_saturation",
        "delayed_sleep",
        "correlated_noise",
        "person_shift",
    }

    assert not np.allclose(
        datasets["nonlinear_saturation"].true_states,
        datasets["delayed_sleep"].true_states,
    )


def test_robustness_metrics_are_finite_and_nonnegative(config):
    datasets = compare_scenarios(config, days=120, seed=42)
    results = evaluate_all_mismatches(config, datasets)

    assert len(results) == 4
    for result in results:
        assert result.rmse_fatigue >= 0
        assert result.rmse_recovery >= 0
        assert result.rmse_load >= 0
        assert result.mae_fatigue >= 0
        assert result.mae_recovery >= 0
        assert result.mae_load >= 0
        assert result.mean_normalized_innovation >= 0
        assert np.all(np.isfinite([
            result.rmse_fatigue,
            result.rmse_recovery,
            result.rmse_load,
            result.mae_fatigue,
            result.mae_recovery,
            result.mae_load,
            result.mean_normalized_innovation,
        ]))


def test_model_mismatch_is_detectable_with_innovation_diagnostic(config):
    dataset = simulate_mismatched_process(
        config,
        days=180,
        seed=42,
        scenario="person_shift",
    )
    result = evaluate_mismatch(config, dataset)

    # This is deliberately a diagnostic threshold, not a model-selection
    # threshold. The point is that a materially shifted person can produce
    # residuals larger than the assumed observation-noise scale.
    assert result.mean_normalized_innovation > 0.5


def test_correlated_noise_changes_the_assumption_without_breaking_filter(config):
    dataset = simulate_mismatched_process(
        config,
        days=150,
        seed=7,
        scenario="correlated_noise",
    )
    result = evaluate_mismatch(config, dataset)

    assert result.rmse_fatigue < 0.50
    assert result.rmse_recovery < 0.50
    assert result.rmse_load < 0.80
