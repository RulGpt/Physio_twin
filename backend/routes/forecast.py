from __future__ import annotations

import numpy as np

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.forecast_validity import validity_to_dict
from backend.models import StateEstimateDB, User
from backend.model_service import make_reference_config
from backend.schemas import (
    ForecastRequest,
    ForecastResponse,
    ForecastState,
    ForecastValidity,
    ForecastValidityResponse,
)
from model.preprocessing import (
    calculate_exercise_load,
    calculate_sleep_deficit,
    normalize_scale_1_to_10,
)
from model.state_space import forecast


router = APIRouter(
    prefix="/users",
    tags=["forecast"],
)


@router.post(
    "/{user_id}/forecast",
    response_model=ForecastResponse,
)
def forecast_user_state(
    user_id: str,
    request: ForecastRequest,
    db: Session = Depends(get_db),
):
    # ---------------------------------------------------------
    # 1. Validate path user_id and body user_id
    # ---------------------------------------------------------

    if user_id != request.user_id:
        raise HTTPException(
            status_code=400,
            detail="Path user_id and body user_id must match.",
        )

    # ---------------------------------------------------------
    # 2. Verify user exists
    # ---------------------------------------------------------

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    # ---------------------------------------------------------
    # 3. Get latest persisted state
    # ---------------------------------------------------------

    latest_state = (
        db.query(StateEstimateDB)
        .filter(
            StateEstimateDB.user_id == user_id
        )
        .order_by(
            StateEstimateDB.date.desc()
        )
        .first()
    )

    if latest_state is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No state estimate exists for this user. "
                "Submit at least one daily observation first."
            ),
        )

    # ---------------------------------------------------------
    # 4. Reconstruct current state
    # ---------------------------------------------------------

    current_state = np.array(
        [
            latest_state.fatigue,
            latest_state.recovery,
            latest_state.load,
        ],
        dtype=float,
    )

    # Current database stores the diagonal variances.
    # Reconstruct a diagonal covariance for forecasting.
    current_covariance = np.diag(
        [
            latest_state.fatigue_variance,
            latest_state.recovery_variance,
            latest_state.load_variance,
        ]
    )

    # ---------------------------------------------------------
    # 5. Convert future user inputs to model inputs
    # ---------------------------------------------------------

    future_inputs = []

    for day in request.future_days:

        sleep_deficit = calculate_sleep_deficit(
            sleep_duration_hours=day.sleep_hours,
            reference_sleep_hours=(
                request.reference_sleep_hours
            ),
        )

        exercise_load = calculate_exercise_load(
            duration_minutes=(
                day.exercise_duration_minutes
            ),
            intensity=day.exercise_intensity,
        )

        stress_normalized = normalize_scale_1_to_10(
            day.stress
        )

        future_inputs.append(
            [
                sleep_deficit,
                exercise_load,
                stress_normalized,
            ]
        )

    future_inputs = np.asarray(
        future_inputs,
        dtype=float,
    )

    # ---------------------------------------------------------
    # 6. Run mathematical state-space forecast
    # ---------------------------------------------------------

    config = make_reference_config()

    forecast_states, forecast_covariances = forecast(
        state=current_state,
        covariance=current_covariance,
        future_inputs=future_inputs,
        config=config,
    )

    # ---------------------------------------------------------
    # 7. Build API response
    # ---------------------------------------------------------

    results: list[ForecastState] = []

    for index, (
        state,
        covariance,
    ) in enumerate(
        zip(
            forecast_states,
            forecast_covariances,
            strict=True,
        ),
        start=1,
    ):

        fatigue = float(state[0])
        recovery = float(state[1])
        load = float(state[2])

        fatigue_variance = max(
            float(covariance[0, 0]),
            0.0,
        )

        recovery_variance = max(
            float(covariance[1, 1]),
            0.0,
        )

        load_variance = max(
            float(covariance[2, 2]),
            0.0,
        )

        # -----------------------------------------------------
        # Forecast validity
        # -----------------------------------------------------

        validity = validity_to_dict(
            fatigue=fatigue,
            recovery=recovery,
            load=load,
        )

        results.append(
            ForecastState(
                horizon=index,

                fatigue=fatigue,
                recovery=recovery,
                load=load,

                fatigue_uncertainty=(
                    fatigue_variance ** 0.5
                ),

                recovery_uncertainty=(
                    recovery_variance ** 0.5
                ),

                load_uncertainty=(
                    load_variance ** 0.5
                ),

                validity=ForecastValidityResponse(
                    fatigue=ForecastValidity(
                        value=float(
                            validity["fatigue"]["value"]
                        ),
                        status=str(
                            validity["fatigue"]["status"]
                        ),
                    ),

                    recovery=ForecastValidity(
                        value=float(
                            validity["recovery"]["value"]
                        ),
                        status=str(
                            validity["recovery"]["status"]
                        ),
                    ),

                    load=ForecastValidity(
                        value=float(
                            validity["load"]["value"]
                        ),
                        status=str(
                            validity["load"]["status"]
                        ),
                    ),
                ),
            )
        )

    return ForecastResponse(
        user_id=user_id,
        last_observed_date=latest_state.date,
        forecast=results,
    )