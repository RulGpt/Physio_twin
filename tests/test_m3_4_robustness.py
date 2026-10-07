import numpy as np

from model.m3_4_robustness import (
    _calibration_error,
    _coverage,
    _interval_width,
)


def test_coverage_perfect():
    actual = np.array([1.0, 2.0, 3.0])

    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([2.0, 3.0, 4.0])

    assert _coverage(
        lower,
        upper,
        actual,
    ) == 1.0


def test_coverage_zero():
    actual = np.array([10.0, 10.0, 10.0])

    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])

    assert _coverage(
        lower,
        upper,
        actual,
    ) == 0.0


def test_interval_width():
    lower = np.array([1.0, 2.0, 3.0])
    upper = np.array([2.0, 4.0, 7.0])

    assert np.isclose(
        _interval_width(
            lower,
            upper,
        ),
        (1.0 + 2.0 + 4.0) / 3.0,
    )


def test_calibration_error():
    assert np.isclose(
        _calibration_error(0.95),
        0.0,
    )

    assert np.isclose(
        _calibration_error(0.90),
        0.05,
    )


def test_calibration_error_symmetric():
    assert np.isclose(
        _calibration_error(1.0),
        _calibration_error(0.90),
    )
    