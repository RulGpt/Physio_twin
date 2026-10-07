from datetime import date

import pytest

from model.preprocessing import (
    build_model_features,
    calculate_exercise_load,
    calculate_session_rpe_load,
    calculate_sleep_deficit,
    normalize_scale_1_to_10,
)
from model.schemas import (
    DailyRecord,
    ExerciseRecord,
    SleepRecord,
    SubjectiveIndices,
)


def make_daily_record():
    return DailyRecord(
        date=date.today(),
        sleep=SleepRecord(
            duration_hours=5.5,
            quality=6,
        ),
        exercise=ExerciseRecord(
            duration_minutes=60,
            intensity=7,
        ),
        subjective=SubjectiveIndices(
            stress=8,
            fatigue=7,
            recovery=4,
        ),
    )


def test_sleep_deficit():

    result = calculate_sleep_deficit(
        sleep_duration_hours=5.5,
        reference_sleep_hours=7.5,
    )

    assert result == 2.0


def test_sleep_deficit_never_negative():

    result = calculate_sleep_deficit(
        sleep_duration_hours=8.0,
        reference_sleep_hours=7.5,
    )

    assert result == 0.0


def test_raw_session_rpe_load():

    result = calculate_session_rpe_load(
        duration_minutes=60,
        intensity=7,
    )

    assert result == 420.0


def test_exercise_load_model_scale():

    result = calculate_exercise_load(
        duration_minutes=60,
        intensity=7,
    )

    # 60 * 7 = 420 raw AU
    # 420 / 60 = 7.0 model units
    # Model input is capped at 6.0
    assert result == 6.0


def test_exercise_load_example_from_api():

    result = calculate_exercise_load(
        duration_minutes=45,
        intensity=7,
    )

    # 45 * 7 = 315 raw AU
    # 315 / 60 = 5.25 model units
    assert result == 5.25


def test_exercise_load_without_intensity():

    raw_result = calculate_session_rpe_load(
        duration_minutes=60,
        intensity=None,
    )

    model_result = calculate_exercise_load(
        duration_minutes=60,
        intensity=None,
    )

    assert raw_result == 0.0
    assert model_result == 0.0


def test_normalize_scale():

    assert normalize_scale_1_to_10(10) == 1.0
    assert normalize_scale_1_to_10(5) == 0.5


def test_normalize_scale_rejects_invalid_value():

    with pytest.raises(ValueError):
        normalize_scale_1_to_10(11)


def test_build_model_features():

    record = make_daily_record()

    features = build_model_features(
        record,
        reference_sleep_hours=7.5,
    )

    assert features["sleep_deficit_hours"] == 2.0

    # Raw exercise load:
    # 60 * 7 = 420 AU
    assert features["exercise_load_raw"] == 420.0

    # Model exercise load:
    # 420 / 60 = 7, capped at 6
    assert features["exercise_load"] == 6.0

    assert features["stress_normalized"] == 0.8
    assert features["fatigue_observed_normalized"] == 0.7
    assert features["recovery_observed_normalized"] == 0.4