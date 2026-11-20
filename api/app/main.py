from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.app import services
from api.app.artifacts import STORE
from api.app.config import cors_origins
from api.app.errors import ApiError
from api.app.schemas import (
    BatteryRuntimeResponse,
    BatteryScenario,
    DegradationSummaryResponse,
    ForecastModel,
    ForecastResponse,
    HealthResponse,
    ProfileId,
    ProfileResponse,
    RecommendationResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    STORE.load()
    yield


app = FastAPI(
    title="supacell API",
    version="0.1.0",
    description="Read-only API for saved supacell experiment outputs.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "loc": list(error.get("loc", [])),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed.",
                "detail": {"errors": errors},
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    code = "not_found" if exc.status_code == 404 else "http_error"
    message = "Route not found." if exc.status_code == 404 else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "error": {
                "code": code,
                "message": message,
                "detail": {},
            }
        },
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return services.health()


@app.get("/profiles", response_model=ProfileResponse)
def profiles():
    return services.profiles()


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    profile: ProfileId,
    date: date,
    model: ForecastModel = ForecastModel.xgboost,
):
    return services.forecast(profile, date, model)


@app.get("/battery/runtime", response_model=BatteryRuntimeResponse)
def battery_runtime(
    profile: ProfileId,
    state_of_charge: float = Query(..., gt=0, le=1),
    scenario: BatteryScenario = BatteryScenario.correct,
):
    return services.battery_runtime(profile, state_of_charge, scenario)


@app.get("/degradation/summary", response_model=DegradationSummaryResponse)
def degradation_summary(level: Optional[str] = None):
    return services.degradation(level)


@app.get("/recommendations", response_model=RecommendationResponse)
def recommendations(
    profile: ProfileId,
    date: date,
    state_of_charge: float = Query(..., gt=0, le=1),
    model: ForecastModel = ForecastModel.xgboost,
):
    return services.recommendations(profile, date, state_of_charge, model)
