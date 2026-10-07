import numpy as np

from model.m3_2_experiment import make_reference_config
from model.m3_3_uncertainty import (
    _coverage,
    _interval_width,
    _forecast_interval,
)


def test_coverage_is_correct():
    actual = np.array(
        [1.0, 2.0, 3.0]
    )

    lower = np.array(
        [0.0, 1.5, 3.1]
    )

    upper = np.array(
        [2.0, 2.5, 4.0]
    )

    coverage = _coverage(
        actual,
        lower,
        upper,
    )

    assert coverage == 2 / 3


def test_interval_width_is_correct():
    lower = np.array(
        [0.0, 1.0]
    )

    upper = np.array(
        [2.0, 5.0]
    )

    width = _interval_width(
        lower,
        upper,
    )

    assert width == 3.0


def test_forecast_interval_shapes():

    config = make_reference_config()

    rng = np.random.default_rng(42)

    n_days = 30

    inputs = rng.normal(
        size=(n_days, 3)
    )

    observations = rng.normal(
        size=(n_days, 2)
    )

    mean, lower, upper = (
        _forecast_interval(
            config=config,
            inputs=inputs,
            observations=observations,
            origin=10,
            horizon=3,
        )
    )

    assert mean.shape == (3, 2)
    assert lower.shape == (3, 2)
    assert upper.shape == (3, 2)


def test_intervals_contain_forecast_mean():

    config = make_reference_config()

    rng = np.random.default_rng(42)

    n_days = 30

    inputs = rng.normal(
        size=(n_days, 3)
    )

    observations = rng.normal(
        size=(n_days, 2)
    )

    mean, lower, upper = (
        _forecast_interval(
            config=config,
            inputs=inputs,
            observations=observations,
            origin=10,
            horizon=3,
        )
    )

    assert np.all(
        lower <= mean
    )

    assert np.all(
        mean <= upper
    )


def test_interval_widths_are_positive():

    config = make_reference_config()

    rng = np.random.default_rng(42)

    n_days = 50

    inputs = rng.normal(
        size=(n_days, 3)
    )

    observations = rng.normal(
        size=(n_days, 2)
    )

    _, lower, upper = (
        _forecast_interval(
            config=config,
            inputs=inputs,
            observations=observations,
            origin=10,
            horizon=7,
        )
    )

    widths = (
        upper - lower
    )

    assert np.all(
        widths > 0.0
    )


def test_longer_horizon_produces_prediction_intervals():

    config = make_reference_config()

    rng = np.random.default_rng(42)

    n_days = 50

    inputs = rng.normal(
        size=(n_days, 3)
    )

    observations = rng.normal(
        size=(n_days, 2)
    )

    _, lower_1, upper_1 = (
        _forecast_interval(
            config=config,
            inputs=inputs,
            observations=observations,
            origin=10,
            horizon=1,
        )
    )

    _, lower_7, upper_7 = (
        _forecast_interval(
            config=config,
            inputs=inputs,
            observations=observations,
            origin=10,
            horizon=7,
        )
    )

    width_1 = np.mean(
        upper_1[-1] - lower_1[-1]
    )

    width_7 = np.mean(
        upper_7[-1] - lower_7[-1]
    )

    assert width_1 > 0.0
    assert width_7 > 0.0