from datetime import date

import pytest
from pydantic import ValidationError

from model.schemas import (
    DailyRecord,
    DietRecord,
    ExerciseRecord,
    SleepRecord,
    SubjectiveIndices,
)


def make_valid_daily_record():
    return DailyRecord(
        date=date.today(),
        sleep=SleepRecord(
            duration_hours=7.5,
            quality=8,
            consistency=8,
        ),
        exercise=ExerciseRecord(
            duration_minutes=40,
            intensity=6,
            sessions=1,
        ),
        subjective=SubjectiveIndices(
            stress=4,
            fatigue=3,
            recovery=8,
        ),
        diet=DietRecord(
            calories=2100,
            protein_grams=85,
            fruit_vegetable_servings=4,
            water_liters=2.5,
            caffeine_mg=120,
        ),
    )


def test_valid_daily_record():
    record = make_valid_daily_record()

    assert record.sleep.duration_hours == 7.5
    assert record.subjective.fatigue == 3
    assert record.exercise.duration_minutes == 40


def test_fatigue_must_be_between_1_and_10():
    with pytest.raises(ValidationError):
        SubjectiveIndices(
            stress=4,
            fatigue=11,
            recovery=8,
        )


def test_sleep_duration_cannot_be_negative():
    with pytest.raises(ValidationError):
        SleepRecord(
            duration_hours=-1,
            quality=8,
        )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        SleepRecord(
            duration_hours=7,
            quality=8,
            unknown_field=123,
        )


def test_future_daily_record_is_rejected():
    future_date = date(
        date.today().year + 1,
        1,
        1,
    )

    with pytest.raises(ValidationError):
        DailyRecord(
            date=future_date,
            sleep=SleepRecord(
                duration_hours=7,
                quality=8,
            ),
            exercise=ExerciseRecord(
                duration_minutes=30,
            ),
            subjective=SubjectiveIndices(
                stress=4,
                fatigue=3,
                recovery=8,
            ),
        )