from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class ProfileId(str, Enum):
    real = "real"
    grid_only = "grid_only"
    grid_plus_solar_plus_battery = "grid_plus_solar_plus_battery"
    off_grid_solar_plus_battery_with_outages = "off_grid_solar_plus_battery_with_outages"


class ForecastModel(str, Enum):
    xgboost = "xgboost"
    random_forest = "random_forest"


class BatteryScenario(str, Enum):
    correct = "correct"
    capacity_degraded_20pct = "capacity_degraded_20pct"
    efficiency_overestimated = "efficiency_overestimated"
    uncertainty_too_narrow = "uncertainty_too_narrow"


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    artifacts: dict[str, bool]
    missing_artifacts: list[str]
    paths: dict[str, str]


class ProfileInfo(BaseModel):
    id: str
    label: str
    dataset_label: str
    forecast_available: bool
    battery_available: bool


class ProfileResponse(BaseModel):
    profiles: list[ProfileInfo]
    count: int


class ForecastPoint(BaseModel):
    timestamp: str
    actual: Optional[float] = None
    predicted: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    profile: str
    profile_label: str
    dataset_label: str
    date: str
    model: str
    unit: str
    source: str
    available_range: Optional[dict[str, str]] = None
    points: list[ForecastPoint]
    count: int
    note: str


class BatteryRuntimeResponse(BaseModel):
    profile: str
    profile_label: str
    dataset_label: str
    state_of_charge: float
    scenario: str
    horizon_hours: int
    low_hours: float
    expected_hours: float
    high_hours: float
    load_only_low_hours: float
    load_only_expected_hours: float
    load_only_high_hours: float
    assumptions: list[str]
    validation_assumptions: list[str]
    limitations: list[str]
    method: str
    source: str


class DegradationLevel(BaseModel):
    level: str
    description: str
    train_rows: int
    test_rows: int
    mae_kwh: float
    rmse_kwh: float
    mape_pct: float
    nominal_coverage: float
    empirical_coverage: float
    avg_interval_width_kwh: float
    conformal_radius_kwh: float
    mae_change_pct: float
    width_change_pct: float
    coverage_delta_pp: float
    user_message: str


class DegradationSummaryResponse(BaseModel):
    dataset_label: str
    source: str
    metadata: dict[str, Any]
    levels: list[DegradationLevel]
    count: int
    note: str


class RecommendationResponse(BaseModel):
    status: str
    profile: str
    date: str
    state_of_charge: float
    messages: list[str]
    figures: dict[str, Any]
    needed: list[str]
