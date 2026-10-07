from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import DailyRecordDB, StateEstimateDB, User
from backend.schemas import (
    HistoryInputs,
    HistoryItem,
    HistoryObservation,
    HistoryState,
    ProfileCreateRequest,
    ProfileCreateResponse,
    ProfileLoginRequest,
    ProfileResponse,
    ProfileSummary,
    UserHistoryResponse,
    UserResponse,
)
from backend.security import hash_password, verify_password


router = APIRouter(
    prefix="/users",
    tags=["users"],
)


# ============================================================================
# CREATE PROFILE / SIGN UP
# ============================================================================


@router.post(
    "",
    response_model=ProfileCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_profile(
    payload: ProfileCreateRequest,
    db: Session = Depends(get_db),
):
    name = payload.name.strip()
    username = payload.username.strip().lower()

    existing_user = db.scalar(
        select(User).where(User.username == username)
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists.",
        )

    user = User(
        name=name,
        username=username,
        password_hash=hash_password(payload.password),
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return ProfileCreateResponse(
        user_id=user.id,
        name=user.name,
        username=user.username,
        created=True,
    )


# ============================================================================
# LOGIN
# ============================================================================


@router.post(
    "/login",
    response_model=ProfileResponse,
)
def login_profile(
    payload: ProfileLoginRequest,
    db: Session = Depends(get_db),
):
    username = payload.username.strip().lower()

    user = db.scalar(
        select(User).where(User.username == username)
    )

    if user is None or not verify_password(
        payload.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    return ProfileResponse(
        user_id=user.id,
        name=user.name,
        username=user.username,
        created_at=user.created_at,
    )


# ============================================================================
# LIST PROFILES
# ============================================================================


@router.get(
    "/profiles",
    response_model=list[ProfileSummary],
)
def list_profiles(
    db: Session = Depends(get_db),
):
    users = db.scalars(
        select(User).order_by(User.created_at)
    ).all()

    return [
        ProfileSummary(
            user_id=user.id,
            name=user.name,
            username=user.username,
        )
        for user in users
    ]


# ============================================================================
# GET USER
# ============================================================================


@router.get(
    "/{user_id}",
    response_model=UserResponse,
)
def get_user(
    user_id: str,
    db: Session = Depends(get_db),
):
    user = db.get(User, user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return UserResponse(
        user_id=user.id,
        created_at=user.created_at,
    )


# ============================================================================
# USER HISTORY
# ============================================================================


@router.get(
    "/{user_id}/history",
    response_model=UserHistoryResponse,
)
def get_user_history(
    user_id: str,
    db: Session = Depends(get_db),
):
    user = db.get(User, user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    daily_records = db.scalars(
        select(DailyRecordDB)
        .where(DailyRecordDB.user_id == user_id)
        .order_by(DailyRecordDB.date)
    ).all()

    state_estimates = db.scalars(
        select(StateEstimateDB)
        .where(StateEstimateDB.user_id == user_id)
        .order_by(StateEstimateDB.date)
    ).all()

    state_by_date = {
        state.date: state
        for state in state_estimates
    }

    history: list[HistoryItem] = []

    for record in daily_records:
        state = state_by_date.get(record.date)

        if state is None:
            continue

        history.append(
            HistoryItem(
                date=record.date,
                inputs=HistoryInputs(
                    sleep_duration_hours=record.sleep_duration_hours,
                    sleep_quality=record.sleep_quality,
                    sleep_deficit=state.sleep_deficit,
                    exercise_duration_minutes=record.exercise_duration_minutes,
                    exercise_intensity=record.exercise_intensity,
                    exercise_load_raw=state.exercise_load_raw,
                    exercise_load_model=state.exercise_load_model,
                    stress=state.stress,
                ),
                observation=HistoryObservation(
                    fatigue_observed=record.fatigue_observed,
                    recovery_observed=record.recovery_observed,
                ),
                state=HistoryState(
                    fatigue=state.fatigue,
                    recovery=state.recovery,
                    load=state.load,
                    fatigue_uncertainty=state.fatigue_variance,
                    recovery_uncertainty=state.recovery_variance,
                    load_uncertainty=state.load_variance,
                ),
            )
        )

    return UserHistoryResponse(
        user_id=user_id,
        history=history,
    )