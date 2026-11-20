from fastapi.testclient import TestClient
import pytest

from api.app import artifacts
from api.app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_all_artifacts(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["missing_artifacts"] == []
    assert all(body["artifacts"].values())


def test_profiles_lists_real_and_synthetic_profiles(client):
    response = client.get("/profiles")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 4
    profiles = {item["id"]: item for item in body["profiles"]}
    assert profiles["real"]["dataset_label"] == "real"
    assert profiles["real"]["forecast_available"] is True
    assert profiles["real"]["battery_available"] is False
    assert profiles["grid_only"]["forecast_available"] is False
    assert profiles["grid_only"]["battery_available"] is True
    assert profiles["grid_plus_solar_plus_battery"]["forecast_available"] is True
    assert profiles["off_grid_solar_plus_battery_with_outages"]["forecast_available"] is False


def test_real_forecast_returns_saved_xgboost_band(client):
    response = client.get(
        "/forecast",
        params={"profile": "real", "date": "2010-04-20", "model": "xgboost"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_label"] == "real"
    assert body["model"] == "xgboost"
    assert body["unit"] == "kWh"
    assert body["count"] == 24
    assert body["available_range"]["start"].startswith("2010-04-20")
    first = body["points"][0]
    assert first["lower"] <= first["predicted"] <= first["upper"]


def test_solar_forecast_returns_saved_solar_band(client):
    response = client.get(
        "/forecast",
        params={
            "profile": "grid_plus_solar_plus_battery",
            "date": "2009-11-07",
            "model": "xgboost",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_label"] == "synthetic"
    assert body["model"] == "saved_solar_prediction"
    assert body["unit"] == "kW"
    assert body["count"] == 24
    assert all(point["lower"] <= point["predicted"] <= point["upper"] for point in body["points"])


def test_forecast_unavailable_profile_returns_clear_error(client):
    response = client.get(
        "/forecast",
        params={"profile": "grid_only", "date": "2025-01-01"},
    )
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "forecast_not_available"
    assert "grid_plus_solar_plus_battery" in " ".join(body["error"]["detail"]["needed"])


def test_forecast_unavailable_date_returns_available_range(client):
    response = client.get(
        "/forecast",
        params={"profile": "real", "date": "2010-01-01"},
    )
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "forecast_date_not_available"
    assert "start" in body["error"]["detail"]["available_range"]


def test_forecast_invalid_model_returns_standard_validation_error(client):
    response = client.get(
        "/forecast",
        params={"profile": "real", "date": "2010-04-20", "model": "linear_regression"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["detail"]["errors"][0]["loc"] == ["query", "model"]


def test_forecast_missing_artifact_returns_503(client, monkeypatch):
    monkeypatch.setattr(artifacts.STORE, "forecast_predictions", None)
    response = client.get(
        "/forecast",
        params={"profile": "real", "date": "2010-04-20"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "forecast_artifact_missing"


def test_battery_runtime_reads_saved_lookup(client):
    response = client.get(
        "/battery/runtime",
        params={"profile": "grid_only", "state_of_charge": 0.8, "scenario": "correct"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_label"] == "synthetic"
    assert body["horizon_hours"] == 72
    assert body["low_hours"] <= body["expected_hours"] <= body["high_hours"]
    assert body["source"].endswith("battery_runtime_lookup_SYNTHETIC.json")
    assert any("worst-case envelope" in item for item in body["assumptions"])


def test_battery_runtime_unsupported_soc_lists_allowed_values(client):
    response = client.get(
        "/battery/runtime",
        params={"profile": "grid_only", "state_of_charge": 0.7},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "unsupported_state_of_charge"
    assert 0.8 in body["error"]["detail"]["allowed_values"]


def test_battery_runtime_real_profile_returns_clear_error(client):
    response = client.get(
        "/battery/runtime",
        params={"profile": "real", "state_of_charge": 0.8},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "battery_not_available"


def test_battery_runtime_invalid_soc_returns_validation_error(client):
    response = client.get(
        "/battery/runtime",
        params={"profile": "grid_only", "state_of_charge": 1.2},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_degradation_summary_returns_all_saved_levels(client):
    response = client.get("/degradation/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_label"] == "real"
    assert body["count"] == 12
    assert {item["level"] for item in body["levels"]} >= {"full", "history_3mo", "no_solar_data"}


def test_degradation_summary_filters_saved_level(client):
    response = client.get("/degradation/summary", params={"level": "no_solar_data"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["levels"][0]["level"] == "no_solar_data"


def test_degradation_summary_unknown_level_returns_available_levels(client):
    response = client.get("/degradation/summary", params={"level": "missing"})
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "degradation_level_not_available"
    assert "full" in body["error"]["detail"]["available_levels"]


def test_recommendations_ready_for_profile_with_saved_forecast_and_battery(client):
    response = client.get(
        "/recommendations",
        params={
            "profile": "grid_plus_solar_plus_battery",
            "date": "2009-11-07",
            "state_of_charge": 0.8,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert "forecast" in body["figures"]
    assert "battery_runtime" in body["figures"]
    assert body["needed"] == []
    assert any("no new model or battery calculation was run" in item for item in body["messages"])


def test_recommendations_data_needed_states_missing_data(client):
    response = client.get(
        "/recommendations",
        params={"profile": "real", "date": "2010-04-20", "state_of_charge": 0.8},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "data_needed"
    assert "battery" in " ".join(body["needed"]).lower()
    assert "forecast" in body["figures"]


def test_cors_allows_local_frontend_preflight(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_unknown_route_uses_standard_error_shape(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
