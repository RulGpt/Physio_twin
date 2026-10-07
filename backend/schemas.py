from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from model.schemas import DailyRecord


# ============================================================================
# PROFILE / USER
# ============================================================================


class ProfileCreateRequest(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=100,
    )

    username: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_.-]+$",
    )

    password: str = Field(
        min_length=8,
        max_length=128,
    )


class ProfileLoginRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=50,
    )

    password: str = Field(
        min_length=8,
        max_length=128,
    )


class ProfileResponse(BaseModel):
    user_id: str
    name: str
    username: str
    created_at: datetime


class ProfileCreateResponse(BaseModel):
    user_id: str
    name: str
    username: str
    created: bool


class ProfileSummary(BaseModel):
    user_id: str
    name: str
    username: str


class UserCreateResponse(BaseModel):
    user_id: str
    created: bool


# ============================================================================
# OBSERVATION
# ============================================================================


class ObservationRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=36,
    )

    record: DailyRecord

    reference_sleep_hours: float = Field(
        default=8.0,
        gt=0,
        le=24,
    )


# ============================================================================
# STATE
# ============================================================================


class StateEstimateResponse(BaseModel):
    user_id: str
    date: date

    fatigue: float
    recovery: float
    load: float

    fatigue_uncertainty: float
    recovery_uncertainty: float
    load_uncertainty: float

    sleep_deficit: float

    exercise_load_raw: float
    exercise_load_model: float

    stress: float


# ============================================================================
# USER
# ============================================================================


class UserResponse(BaseModel):
    user_id: str
    created_at: datetime


# ============================================================================
# HISTORY
# ============================================================================


class HistoryState(BaseModel):
    fatigue: float
    recovery: float
    load: float

    fatigue_uncertainty: float
    recovery_uncertainty: float
    load_uncertainty: float


class HistoryInputs(BaseModel):
    sleep_duration_hours: float
    sleep_quality: int

    sleep_deficit: float

    exercise_duration_minutes: float
    exercise_intensity: int | None

    exercise_load_raw: float
    exercise_load_model: float

    stress: float


class HistoryObservation(BaseModel):
    fatigue_observed: int
    recovery_observed: int


class HistoryItem(BaseModel):
    date: date

    inputs: HistoryInputs

    observation: HistoryObservation

    state: HistoryState


class UserHistoryResponse(BaseModel):
    user_id: str
    history: list[HistoryItem]


# ============================================================================
# FORECAST
# ============================================================================


class ForecastDayInput(BaseModel):
    sleep_hours: float = Field(
        ge=0,
        le=24,
    )

    exercise_duration_minutes: float = Field(
        ge=0,
        le=1440,
    )

    exercise_intensity: int | None = Field(
        default=None,
        ge=1,
        le=10,
    )

    stress: int = Field(
        ge=1,
        le=10,
    )


class ForecastRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=36,
    )

    future_days: list[ForecastDayInput] = Field(
        min_length=1,
        max_length=30,
    )

    reference_sleep_hours: float = Field(
        default=8.0,
        gt=0,
        le=24,
    )


class ForecastValidity(BaseModel):
    value: float
    status: str


class ForecastValidityResponse(BaseModel):
    fatigue: ForecastValidity
    recovery: ForecastValidity
    load: ForecastValidity


class ForecastState(BaseModel):
    horizon: int

    fatigue: float
    recovery: float
    load: float

    fatigue_uncertainty: float
    recovery_uncertainty: float
    load_uncertainty: float

    validity: ForecastValidityResponse


class ForecastResponse(BaseModel):
    user_id: str

    last_observed_date: date

    forecast: list[ForecastState]