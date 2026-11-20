import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


class ArtifactPaths:
    forecast_predictions = REPO_ROOT / "results" / "forecasting" / "forecast_predictions_REAL.csv"
    solar_predictions = (
        REPO_ROOT / "results" / "solar_degradation" / "solar_predictions_SYNTHETIC.csv"
    )
    battery_lookup = REPO_ROOT / "api" / "data" / "battery_runtime_lookup_SYNTHETIC.json"
    degradation_results = REPO_ROOT / "results" / "degradation" / "degradation_results_REAL.csv"
    degradation_metadata = (
        REPO_ROOT / "results" / "degradation" / "degradation_metadata_REAL.json"
    )
    simulation_dir = REPO_ROOT / "results" / "simulations"


def cors_origins():
    configured = os.getenv("SUPACELL_CORS_ORIGINS")
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    return list(DEFAULT_CORS_ORIGINS)
