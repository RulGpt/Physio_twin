import numpy as np

from backend.forecast_validity import (
    classify_forecast_state,
    classify_state_value,
    forecast_is_within_scale,
    validity_to_dict,
)


def test_value_within_scale():

    result = classify_state_value(0.5)

    assert result.value == 0.5
    assert result.status == "within_scale"


def test_value_below_scale():

    result = classify_state_value(-0.2)

    assert result.value == -0.2
    assert result.status == "below_scale"


def test_value_above_scale():

    result = classify_state_value(1.2)

    assert result.value == 1.2
    assert result.status == "above_scale"


def test_nan_is_invalid():

    result = classify_state_value(np.nan)

    assert result.status == "invalid_numeric"


def test_all_states_within_scale():

    result = classify_forecast_state(
        fatigue=0.6,
        recovery=0.4,
        load=0.7,
    )

    assert result["fatigue"].status == "within_scale"
    assert result["recovery"].status == "within_scale"
    assert result["load"].status == "within_scale"


def test_forecast_outside_scale():

    result = classify_forecast_state(
        fatigue=1.2,
        recovery=-0.1,
        load=0.8,
    )

    assert result["fatigue"].status == "above_scale"
    assert result["recovery"].status == "below_scale"
    assert result["load"].status == "within_scale"


def test_forecast_is_within_scale():

    assert forecast_is_within_scale(
        fatigue=0.5,
        recovery=0.4,
        load=0.7,
    )


def test_forecast_is_not_within_scale():

    assert not forecast_is_within_scale(
        fatigue=1.2,
        recovery=0.4,
        load=0.7,
    )


def test_validity_to_dict():

    result = validity_to_dict(
        fatigue=1.15,
        recovery=-0.12,
        load=0.8,
    )

    assert result["fatigue"]["status"] == "above_scale"
    assert result["recovery"]["status"] == "below_scale"
    assert result["load"]["status"] == "within_scale"