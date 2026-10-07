"""
PHYSIO-TWIN
-----------
Preprocessing and feature derivation.

This module converts validated DailyRecord objects into transparent,
model-ready numerical features.

Important:
    These transformations are deterministic feature engineering steps.
    They are not clinical scores or diagnostic measurements.
"""

from typing import Any, Dict

from model.schemas import DailyRecord


# ---------------------------------------------------------------------
# Model input conventions
# ---------------------------------------------------------------------

MODEL_EXERCISE_LOAD_DIVISOR = 60.0
MODEL_EXERCISE_LOAD_MAX = 6.0


def calculate_sleep_deficit(
    sleep_duration_hours: float,
    reference_sleep_hours: float,
) -> float:
    """
    Calculate sleep deficit relative to a reference duration.

    Formula:
        deficit = max(0, reference - observed)

    Unit:
        hours
    """

    if sleep_duration_hours < 0:
        raise ValueError(
            "Sleep duration cannot be negative."
        )

    if reference_sleep_hours <= 0:
        raise ValueError(
            "Reference sleep duration must be positive."
        )

    return max(
        0.0,
        reference_sleep_hours - sleep_duration_hours,
    )


def calculate_session_rpe_load(
    duration_minutes: float,
    intensity: int | None,
) -> float:
    """
    Calculate raw session-RPE exercise load.

    Formula:
        raw_load = duration_minutes * intensity

    Unit:
        arbitrary units (AU)

    This follows the standard session-RPE formulation in which
    session duration is multiplied by perceived session intensity.
    """

    if duration_minutes < 0:
        raise ValueError(
            "Exercise duration cannot be negative."
        )

    if intensity is None:
        return 0.0

    if not 1 <= intensity <= 10:
        raise ValueError(
            "Exercise intensity must be between 1 and 10."
        )

    return float(duration_minutes * intensity)


def calculate_exercise_load(
    duration_minutes: float,
    intensity: int | None,
) -> float:
    """
    Convert raw session-RPE load into the PHYSIO-TWIN
    model input scale.

    Raw:
        duration_minutes * intensity

    Model scale:
        raw_session_rpe_load / 60

    The model input is capped at 6.0 because the current
    validated synthetic state-space experiments use an
    exercise-input range of 0-6.

    This scaling is a model-input convention, not a
    physiological reference range.
    """

    raw_load = calculate_session_rpe_load(
        duration_minutes=duration_minutes,
        intensity=intensity,
    )

    model_load = raw_load / MODEL_EXERCISE_LOAD_DIVISOR

    return min(
        MODEL_EXERCISE_LOAD_MAX,
        model_load,
    )


def normalize_scale_1_to_10(value: int) -> float:
    """
    Normalize a 1-10 subjective score to 0.1-1.0.
    """

    if not 1 <= value <= 10:
        raise ValueError(
            "Value must be between 1 and 10."
        )

    return value / 10.0


def build_model_features(
    record: DailyRecord,
    reference_sleep_hours: float,
) -> Dict[str, Any]:
    """
    Convert a validated DailyRecord into transparent
    model-ready features.
    """

    subjective = record.subjective

    raw_exercise_load = calculate_session_rpe_load(
        duration_minutes=record.exercise.duration_minutes,
        intensity=record.exercise.intensity,
    )

    model_exercise_load = calculate_exercise_load(
        duration_minutes=record.exercise.duration_minutes,
        intensity=record.exercise.intensity,
    )

    features: Dict[str, Any] = {
        "date": record.date,

        # -------------------------------------------------------------
        # Raw observations
        # -------------------------------------------------------------

        "sleep_duration_hours": record.sleep.duration_hours,
        "sleep_quality": record.sleep.quality,

        "exercise_duration_minutes": (
            record.exercise.duration_minutes
        ),
        "exercise_intensity": (
            record.exercise.intensity
        ),

        # -------------------------------------------------------------
        # Derived model inputs
        # -------------------------------------------------------------

        "sleep_deficit_hours": calculate_sleep_deficit(
            sleep_duration_hours=record.sleep.duration_hours,
            reference_sleep_hours=reference_sleep_hours,
        ),

        # Preserve both raw and model-scaled exercise load.
        "exercise_load_raw": raw_exercise_load,
        "exercise_load": model_exercise_load,

        "stress_normalized": normalize_scale_1_to_10(
            subjective.stress
        ),

        # -------------------------------------------------------------
        # Observed targets
        # -------------------------------------------------------------

        "fatigue_observed_normalized": (
            normalize_scale_1_to_10(
                subjective.fatigue
            )
        ),

        "recovery_observed_normalized": (
            normalize_scale_1_to_10(
                subjective.recovery
            )
        ),
    }

    # Optional activity information
    if record.activity is not None:
        features["steps"] = record.activity.steps
        features["sedentary_hours"] = (
            record.activity.sedentary_hours
        )

    # Optional diet information
    if record.diet is not None:
        features["calories"] = record.diet.calories
        features["protein_grams"] = (
            record.diet.protein_grams
        )
        features["water_liters"] = (
            record.diet.water_liters
        )
        features["caffeine_mg"] = (
            record.diet.caffeine_mg
        )

    # Optional habit information
    if record.habits is not None:
        features["cigarettes"] = (
            record.habits.cigarettes
        )
        features["alcohol_drinks"] = (
            record.habits.alcohol_drinks
        )
        features["screen_time_hours"] = (
            record.habits.screen_time_hours
        )

    return features