"""
PHYSIO-TWIN
-----------
Data contracts for user profile and daily physiological/lifestyle records.

This module defines the validated input layer of the PHYSIO-TWIN system.
The mathematical model should never receive raw, unvalidated user input.
"""

from datetime import date as Date, time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# ENUMERATIONS
# ---------------------------------------------------------------------------

class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class ActivityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ExerciseType(str, Enum):
    NONE = "none"
    WALKING = "walking"
    RUNNING = "running"
    CYCLING = "cycling"
    STRENGTH = "strength"
    SPORTS = "sports"
    YOGA = "yoga"
    OTHER = "other"


# ---------------------------------------------------------------------------
# USER PROFILE
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    """
    Persistent characteristics of a PHYSIO-TWIN user.

    These values are not expected to change every day.
    """

    model_config = ConfigDict(extra="forbid")

    age: int = Field(
        ...,
        ge=13,
        le=100,
        description="User age in years.",
    )

    sex: Sex = Field(
        ...,
        description="Self-reported sex category used as a modelling covariate where appropriate.",
    )

    height_cm: float = Field(
        ...,
        gt=50,
        le=250,
        description="Height in centimetres.",
    )

    weight_kg: float = Field(
        ...,
        gt=20,
        le=300,
        description="Body weight in kilograms.",
    )

    typical_sleep_hours: Optional[float] = Field(
        default=None,
        ge=0,
        le=24,
        description="Typical sleep duration in hours, if known.",
    )

    typical_activity_level: Optional[ActivityLevel] = Field(
        default=None,
        description="Typical overall activity level.",
    )


# ---------------------------------------------------------------------------
# SLEEP
# ---------------------------------------------------------------------------

class SleepRecord(BaseModel):
    """Daily sleep-related observations."""

    model_config = ConfigDict(extra="forbid")

    duration_hours: float = Field(
        ...,
        ge=0,
        le=24,
        description="Total sleep duration in hours.",
    )

    quality: int = Field(
        ...,
        ge=1,
        le=10,
        description="Self-reported sleep quality from 1 to 10.",
    )

    bedtime: Optional[time] = Field(
        default=None,
        description="Approximate bedtime.",
    )

    wake_time: Optional[time] = Field(
        default=None,
        description="Approximate wake-up time.",
    )

    consistency: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Self-reported sleep schedule consistency from 1 to 10.",
    )


# ---------------------------------------------------------------------------
# DIET
# ---------------------------------------------------------------------------

class DietRecord(BaseModel):
    """Daily coarse-grained dietary observations."""

    model_config = ConfigDict(extra="forbid")

    calories: Optional[float] = Field(
        default=None,
        ge=0,
        le=10000,
        description="Approximate daily energy intake in kcal.",
    )

    protein_grams: Optional[float] = Field(
        default=None,
        ge=0,
        le=500,
        description="Approximate daily protein intake in grams.",
    )

    fruit_vegetable_servings: Optional[int] = Field(
        default=None,
        ge=0,
        le=30,
        description="Approximate number of fruit and vegetable servings.",
    )

    water_liters: Optional[float] = Field(
        default=None,
        ge=0,
        le=20,
        description="Approximate daily water/fluid intake in litres.",
    )

    caffeine_mg: Optional[float] = Field(
        default=None,
        ge=0,
        le=2000,
        description="Approximate daily caffeine intake in milligrams.",
    )


# ---------------------------------------------------------------------------
# EXERCISE
# ---------------------------------------------------------------------------

class ExerciseRecord(BaseModel):
    """Daily exercise observations."""

    model_config = ConfigDict(extra="forbid")

    duration_minutes: float = Field(
        ...,
        ge=0,
        le=1440,
        description="Total exercise duration in minutes.",
    )

    intensity: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Self-reported exercise intensity from 1 to 10.",
    )

    sessions: Optional[int] = Field(
        default=None,
        ge=0,
        le=20,
        description="Number of exercise sessions during the day.",
    )

    exercise_type: ExerciseType = Field(
        default=ExerciseType.NONE,
        description="Primary exercise type for the day.",
    )


# ---------------------------------------------------------------------------
# GENERAL ACTIVITY
# ---------------------------------------------------------------------------

class ActivityRecord(BaseModel):
    """General daily movement and sedentary behaviour."""

    model_config = ConfigDict(extra="forbid")

    steps: Optional[int] = Field(
        default=None,
        ge=0,
        le=100000,
        description="Approximate number of steps.",
    )

    sedentary_hours: Optional[float] = Field(
        default=None,
        ge=0,
        le=24,
        description="Approximate sedentary time in hours.",
    )

    activity_level: Optional[ActivityLevel] = Field(
        default=None,
        description="Overall daily activity level.",
    )


# ---------------------------------------------------------------------------
# HABITS
# ---------------------------------------------------------------------------

class HabitRecord(BaseModel):
    """Daily habit and routine exposure variables."""

    model_config = ConfigDict(extra="forbid")

    cigarettes: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Number of cigarettes smoked during the day.",
    )

    alcohol_drinks: Optional[float] = Field(
        default=None,
        ge=0,
        le=30,
        description="Approximate number of standard alcoholic drinks.",
    )

    screen_time_hours: Optional[float] = Field(
        default=None,
        ge=0,
        le=24,
        description="Approximate daily recreational/overall screen time.",
    )

    late_night_screen_hours: Optional[float] = Field(
        default=None,
        ge=0,
        le=12,
        description="Approximate screen exposure during the late-night period.",
    )


# ---------------------------------------------------------------------------
# SUBJECTIVE INDICES
# ---------------------------------------------------------------------------

class SubjectiveIndices(BaseModel):
    """
    Daily self-reported outcome/state indicators.

    These are especially important because they provide observable targets
    against which PHYSIO-TWIN's latent state estimates can be evaluated.
    """

    model_config = ConfigDict(extra="forbid")

    stress: int = Field(
        ...,
        ge=1,
        le=10,
        description="Self-reported perceived stress from 1 to 10.",
    )

    fatigue: int = Field(
        ...,
        ge=1,
        le=10,
        description="Self-reported fatigue from 1 to 10.",
    )

    recovery: int = Field(
        ...,
        ge=1,
        le=10,
        description="Self-reported recovery from 1 to 10.",
    )


# ---------------------------------------------------------------------------
# DAILY RECORD
# ---------------------------------------------------------------------------

class DailyRecord(BaseModel):
    """
    Complete daily PHYSIO-TWIN observation.

    Required:
        - date
        - sleep
        - exercise
        - subjective indices

    Optional:
        - diet
        - activity
        - habits
    """

    model_config = ConfigDict(extra="forbid")

    date: Date = Field(
        ...,
        description="Calendar date represented by this record.",
    )

    sleep: SleepRecord
    exercise: ExerciseRecord
    subjective: SubjectiveIndices

    diet: Optional[DietRecord] = None
    activity: Optional[ActivityRecord] = None
    habits: Optional[HabitRecord] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: Date) -> Date:
        """Reject future daily records."""
        if value > Date.today():
            raise ValueError("Daily record date cannot be in the future.")
        return value