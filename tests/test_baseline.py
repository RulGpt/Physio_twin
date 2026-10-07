import pytest

from model.baseline import (
    BaselineStatistics,
    calculate_baseline,
    calculate_z_score,
)


def test_calculate_baseline():

    baseline = calculate_baseline(
        [6.0, 7.0, 8.0, 7.0, 7.5]
    )

    assert baseline.count == 5
    assert baseline.mean == pytest.approx(7.1)
    assert baseline.median == pytest.approx(7.0)
    assert baseline.minimum == 6.0
    assert baseline.maximum == 8.0


def test_baseline_standard_deviation():

    baseline = calculate_baseline(
        [6.0, 7.0, 8.0]
    )

    assert baseline.standard_deviation == pytest.approx(
        1.0
    )


def test_empty_baseline_is_rejected():

    with pytest.raises(ValueError):

        calculate_baseline([])


def test_non_finite_values_are_rejected():

    with pytest.raises(ValueError):

        calculate_baseline(
            [7.0, float("nan")]
        )


def test_z_score():

    baseline = calculate_baseline(
        [6.0, 7.0, 8.0]
    )

    z = calculate_z_score(
        value=9.0,
        baseline=baseline,
    )

    assert z == pytest.approx(2.0)


def test_z_score_below_baseline():

    baseline = calculate_baseline(
        [6.0, 7.0, 8.0]
    )

    z = calculate_z_score(
        value=5.0,
        baseline=baseline,
    )

    assert z == pytest.approx(-2.0)


def test_single_observation_has_no_variability_information():

    baseline = calculate_baseline(
        [7.0]
    )

    z = calculate_z_score(
        value=10.0,
        baseline=baseline,
    )

    assert z == 0.0


def test_zero_variance_baseline():

    baseline = calculate_baseline(
        [7.0, 7.0, 7.0]
    )

    z = calculate_z_score(
        value=10.0,
        baseline=baseline,
    )

    assert z == 0.0