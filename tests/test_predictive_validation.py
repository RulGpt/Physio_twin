import numpy as np
import pytest

from model.predictive_validation import (
    chronological_split,
    persistence_predictions,
    state_space_one_step_predictions,
)
from model.state_space import StateSpaceConfig


def make_config() -> StateSpaceConfig:
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


def make_data(days: int = 20):
    rng = np.random.default_rng(42)

    inputs = np.column_stack(
        [
            rng.uniform(0.0, 1.5, days),
            rng.uniform(0.0, 3.0, days),
            rng.uniform(0.0, 1.0, days),
        ]
    )

    observations = np.column_stack(
        [
            np.linspace(0.5, 1.5, days),
            np.linspace(0.4, 1.2, days),
        ]
    )

    return inputs, observations


def test_chronological_split_preserves_order():

    inputs, observations = make_data(20)

    split = chronological_split(
        inputs,
        observations,
        train_days=15,
    )

    assert split.train_inputs.shape == (15, 3)
    assert split.train_observations.shape == (15, 2)

    assert split.test_inputs.shape == (5, 3)
    assert split.test_observations.shape == (5, 2)

    np.testing.assert_array_equal(
        split.train_inputs,
        inputs[:15],
    )

    np.testing.assert_array_equal(
        split.test_inputs,
        inputs[15:],
    )


def test_chronological_split_rejects_invalid_train_size():

    inputs, observations = make_data(10)

    with pytest.raises(ValueError):
        chronological_split(
            inputs,
            observations,
            train_days=1,
        )

    with pytest.raises(ValueError):
        chronological_split(
            inputs,
            observations,
            train_days=10,
        )


def test_persistence_prediction_alignment():

    observations = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ]
    )

    result = persistence_predictions(
        observations
    )

    expected_predictions = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )

    expected_targets = np.array(
        [
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ]
    )

    np.testing.assert_array_equal(
        result.predictions,
        expected_predictions,
    )

    np.testing.assert_array_equal(
        result.observations,
        expected_targets,
    )


def test_persistence_metrics_are_finite():

    observations = np.array(
        [
            [1.0, 2.0],
            [1.5, 2.5],
            [1.2, 2.2],
            [1.8, 2.8],
        ]
    )

    result = persistence_predictions(
        observations
    )

    assert result.metrics.n_predictions == 3

    assert np.isfinite(
        result.metrics.fatigue_mae
    )

    assert np.isfinite(
        result.metrics.recovery_mae
    )

    assert np.isfinite(
        result.metrics.overall_mae
    )

    assert np.isfinite(
        result.metrics.fatigue_rmse
    )

    assert np.isfinite(
        result.metrics.recovery_rmse
    )

    assert np.isfinite(
        result.metrics.overall_rmse
    )


def test_state_space_predictions_are_finite():

    config = make_config()

    inputs, observations = make_data(30)

    result = state_space_one_step_predictions(
        config=config,
        inputs=inputs,
        observations=observations,
    )

    assert result.predictions.shape[1] == 2

    assert result.observations.shape == result.predictions.shape

    assert result.metrics.n_predictions > 0

    assert np.all(
        np.isfinite(result.predictions)
    )

    assert np.all(
        np.isfinite(result.observations)
    )

    assert np.isfinite(
        result.metrics.overall_rmse
    )


def test_state_space_prediction_count():

    config = make_config()

    inputs, observations = make_data(25)

    result = state_space_one_step_predictions(
        config=config,
        inputs=inputs,
        observations=observations,
    )

    assert result.metrics.n_predictions == 24


def test_prediction_shape_validation():

    config = make_config()

    with pytest.raises(ValueError):
        state_space_one_step_predictions(
            config=config,
            inputs=np.zeros((10, 2)),
            observations=np.zeros((10, 2)),
        )

    with pytest.raises(ValueError):
        state_space_one_step_predictions(
            config=config,
            inputs=np.zeros((10, 3)),
            observations=np.zeros((10, 3)),
        )