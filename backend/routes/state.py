from __future__ import annotations

import numpy as np

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.model_service import (
    make_reference_config,
    update_state,
)
from backend.models import (
    DailyRecordDB,
    StateEstimateDB,
    User,
)
from backend.schemas import (
    ObservationRequest,
    StateEstimateResponse,
)
from model.schemas import (
    DailyRecord,
    ExerciseRecord,
    SleepRecord,
    SubjectiveIndices,
)


router = APIRouter(
    prefix="/state",
    tags=["state"],
)


# ============================================================================
# HELPERS
# ============================================================================


def db_record_to_model_record(
    record: DailyRecordDB,
) -> DailyRecord:
    """
    Convert a persisted DailyRecordDB row back into
    the validated DailyRecord model used by PHYSIO-TWIN.
    """

    return DailyRecord(
        date=record.date,

        sleep=SleepRecord(
            duration_hours=record.sleep_duration_hours,
            quality=record.sleep_quality,
        ),

        exercise=ExerciseRecord(
            duration_minutes=record.exercise_duration_minutes,
            intensity=record.exercise_intensity,
            exercise_type=record.exercise_type,
        ),

        subjective=SubjectiveIndices(
            stress=record.stress,
            fatigue=record.fatigue_observed,
            recovery=record.recovery_observed,
        ),
    )


def create_state_estimate(
    user_id: str,
    record: DailyRecordDB,
    state: np.ndarray,
    covariance: np.ndarray,
    metadata: dict,
) -> StateEstimateDB:
    """
    Create a StateEstimateDB row from one filtered state.
    """

    return StateEstimateDB(
        user_id=user_id,
        date=record.date,

        fatigue=float(state[0]),
        recovery=float(state[1]),
        load=float(state[2]),

        fatigue_variance=float(
            max(covariance[0, 0], 0.0)
        ),

        recovery_variance=float(
            max(covariance[1, 1], 0.0)
        ),

        load_variance=float(
            max(covariance[2, 2], 0.0)
        ),

        sleep_deficit=float(
            metadata["sleep_deficit_hours"]
        ),

        exercise_load_raw=float(
            metadata["exercise_load_raw"]
        ),

        exercise_load_model=float(
            metadata["exercise_load_model"]
        ),

        stress=float(
            metadata["stress_normalized"]
        ),
    )


def state_to_response(
    user_id: str,
    date_value,
    state: np.ndarray,
    covariance: np.ndarray,
    metadata: dict,
) -> StateEstimateResponse:
    """
    Convert a filtered state into the API response format.
    """

    return StateEstimateResponse(
        user_id=user_id,
        date=date_value,

        fatigue=float(state[0]),
        recovery=float(state[1]),
        load=float(state[2]),

        fatigue_uncertainty=float(
            np.sqrt(
                max(
                    covariance[0, 0],
                    0.0,
                )
            )
        ),

        recovery_uncertainty=float(
            np.sqrt(
                max(
                    covariance[1, 1],
                    0.0,
                )
            )
        ),

        load_uncertainty=float(
            np.sqrt(
                max(
                    covariance[2, 2],
                    0.0,
                )
            )
        ),

        sleep_deficit=float(
            metadata["sleep_deficit_hours"]
        ),

        exercise_load_raw=float(
            metadata["exercise_load_raw"]
        ),

        exercise_load_model=float(
            metadata["exercise_load_model"]
        ),

        stress=float(
            metadata["stress_normalized"]
        ),
    )


# ============================================================================
# UPDATE PHYSIOLOGICAL STATE
# ============================================================================


@router.post(
    "/update",
    response_model=StateEstimateResponse,
)
def update_physiological_state(
    request: ObservationRequest,
    db: Session = Depends(get_db),
):
    # ------------------------------------------------------------------------
    # 1. Verify user
    # ------------------------------------------------------------------------

    user = (
        db.query(User)
        .filter(User.id == request.user_id)
        .first()
    )

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    current_date = request.record.date

    # ------------------------------------------------------------------------
    # 2. Prevent duplicate date
    # ------------------------------------------------------------------------

    existing_record = (
        db.query(DailyRecordDB)
        .filter(
            DailyRecordDB.user_id == request.user_id,
            DailyRecordDB.date == current_date,
        )
        .first()
    )

    if existing_record is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "A daily record already exists for "
                f"{current_date}."
            ),
        )

    # ------------------------------------------------------------------------
    # 3. Save the new raw observation
    # ------------------------------------------------------------------------

    daily_record = DailyRecordDB(
        user_id=request.user_id,
        date=current_date,

        sleep_duration_hours=(
            request.record.sleep.duration_hours
        ),

        sleep_quality=(
            request.record.sleep.quality
        ),

        exercise_duration_minutes=(
            request.record.exercise.duration_minutes
        ),

        exercise_intensity=(
            request.record.exercise.intensity
        ),

        exercise_type=(
            request.record.exercise.exercise_type.value
        ),

        stress=(
            request.record.subjective.stress
        ),

        fatigue_observed=(
            request.record.subjective.fatigue
        ),

        recovery_observed=(
            request.record.subjective.recovery
        ),
    )

    db.add(daily_record)

    # Make the newly inserted record visible to subsequent queries
    # within this transaction.
    db.flush()

    # ------------------------------------------------------------------------
    # 4. Get ALL observations chronologically
    # ------------------------------------------------------------------------

    all_records = (
        db.query(DailyRecordDB)
        .filter(
            DailyRecordDB.user_id == request.user_id
        )
        .order_by(
            DailyRecordDB.date.asc()
        )
        .all()
    )

    # ------------------------------------------------------------------------
    # 5. Delete old state trajectory
    # ------------------------------------------------------------------------
    #
    # Because the state-space model is sequential, inserting a historical
    # observation can change every state after that observation.
    #
    # Therefore we rebuild the user's complete latent trajectory.
    # ------------------------------------------------------------------------

    (
        db.query(StateEstimateDB)
        .filter(
            StateEstimateDB.user_id == request.user_id
        )
        .delete(
            synchronize_session=False
        )
    )

    # ------------------------------------------------------------------------
    # 6. Initialize reference model
    # ------------------------------------------------------------------------

    config = make_reference_config()

    state = config.initial_state.copy()
    covariance = config.initial_covariance.copy()

    # ------------------------------------------------------------------------
    # 7. Recompute complete longitudinal trajectory
    # ------------------------------------------------------------------------

    response_state = None
    response_covariance = None
    response_metadata = None

    for record in all_records:

        model_record = db_record_to_model_record(
            record
        )

        result = update_state(
            state=state,
            covariance=covariance,
            record=model_record,
            config=config,
            reference_sleep_hours=request.reference_sleep_hours,
        )

        state = np.asarray(
            result["state"],
            dtype=float,
        )

        covariance = np.asarray(
            result["covariance"],
            dtype=float,
        )

        metadata = result["metadata"]

        # -------------------------------------------------------------
        # Save rebuilt state
        # -------------------------------------------------------------

        state_record = create_state_estimate(
            user_id=request.user_id,
            record=record,
            state=state,
            covariance=covariance,
            metadata=metadata,
        )

        db.add(state_record)

        # -------------------------------------------------------------
        # Keep requested date for API response
        # -------------------------------------------------------------

        if record.date == current_date:
            response_state = state.copy()
            response_covariance = covariance.copy()
            response_metadata = dict(metadata)

    # ------------------------------------------------------------------------
    # 8. Safety check
    # ------------------------------------------------------------------------

    if (
        response_state is None
        or response_covariance is None
        or response_metadata is None
    ):
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Unable to rebuild physiological state.",
        )

    # ------------------------------------------------------------------------
    # 9. Commit complete rebuilt trajectory
    # ------------------------------------------------------------------------

    db.commit()

    # ------------------------------------------------------------------------
    # 10. Return state corresponding to submitted date
    # ------------------------------------------------------------------------

    return state_to_response(
        user_id=request.user_id,
        date_value=current_date,
        state=response_state,
        covariance=response_covariance,
        metadata=response_metadata,
    )