from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db

from backend.routes.forecast import router as forecast_router
from backend.routes.health import router as health_router
from backend.routes.observations import router as observations_router
from backend.routes.state import router as state_router
from backend.routes.users import router as users_router


# ---------------------------------------------------------------------------
# Database initialization / migration
# ---------------------------------------------------------------------------

init_db()


app = FastAPI(
    title="PHYSIO-TWIN API",
    description="Personal Physiological State Digital Twin API",
    version="0.3.0",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------

app.include_router(health_router)
app.include_router(observations_router)
app.include_router(state_router)
app.include_router(users_router)
app.include_router(forecast_router)