from datetime import date as Date
from typing import Optional

import pandas as pd

from api.app.artifacts import STORE
from api.app.config import ArtifactPaths
from api.app.errors import ApiError
from api.app.schemas import BatteryScenario, ForecastModel, ProfileId

FORECAST_PROFILES = {ProfileId.real.value, ProfileId.grid_plus_solar_plus_battery.value}
SOLAR_PROFILE = ProfileId.grid_plus_solar_plus_battery.value
SYNTHETIC_PROFILES = [
    ProfileId.grid_only.value,
    ProfileId.grid_plus_solar_plus_battery.value,
    ProfileId.off_grid_solar_plus_battery_with_outages.value,
]


def _value(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _profile_id(profile):
    return profile.value if isinstance(profile, ProfileId) else str(profile)


def _model_id(model):
    return model.value if isinstance(model, ForecastModel) else str(model)


def _scenario_id(scenario):
    return scenario.value if isinstance(scenario, BatteryScenario) else str(scenario)


def _require(name, code, message):
    artifact = getattr(STORE, name)
    if artifact is None:
        raise ApiError(
            503,
            code,
            message,
            {"artifact": STORE.paths.get(name), "reason": STORE.errors.get(name)},
        )
    return artifact


def _date_mask_timestamps(timestamps, requested_date):
    start = pd.Timestamp(requested_date)
    end = start + pd.Timedelta(days=1)
    return (timestamps >= start) & (timestamps < end)


def _available_range(timestamps):
    if len(timestamps) == 0:
        return None
    return {
        "start": pd.Timestamp(timestamps.min()).isoformat(),
        "end": pd.Timestamp(timestamps.max()).isoformat(),
    }


def _profile_labels():
    labels = {"real": "Real household"}
    lookup = STORE.battery_lookup
    if lookup and "profile_labels" in lookup:
        labels.update(lookup["profile_labels"])
    else:
        labels.update({profile: profile for profile in SYNTHETIC_PROFILES})
    return labels


def _profile_ids():
    lookup = STORE.battery_lookup
    if lookup and "entries" in lookup:
        return ["real"] + list(lookup["entries"])
    return ["real"] + SYNTHETIC_PROFILES


def health():
    return {
        "status": STORE.health_status,
        "artifacts": dict(STORE.status),
        "missing_artifacts": STORE.missing_artifacts,
        "paths": dict(STORE.paths),
    }


def profiles():
    labels = _profile_labels()
    lookup_entries = STORE.battery_lookup.get("entries", {}) if STORE.battery_lookup else {}
    rows = []
    for profile in _profile_ids():
        dataset_label = "real" if profile == "real" else "synthetic"
        forecast_available = (
            STORE.forecast_predictions is not None
            if profile == "real"
            else STORE.solar_predictions is not None and profile == SOLAR_PROFILE
        )
        rows.append(
            {
                "id": profile,
                "label": labels.get(profile, profile),
                "dataset_label": dataset_label,
                "forecast_available": forecast_available,
                "battery_available": profile in lookup_entries,
            }
        )
    return {"profiles": rows, "count": len(rows)}


def forecast(profile: ProfileId, requested_date: Date, model: ForecastModel):
    profile_id = _profile_id(profile)
    labels = _profile_labels()
    profile_label = labels.get(profile_id, profile_id)

    if profile_id not in FORECAST_PROFILES:
        raise ApiError(
            404,
            "forecast_not_available",
            f"No saved forecast uncertainty band is available for profile '{profile_id}'.",
            {
                "available_profiles": sorted(FORECAST_PROFILES),
                "needed": [
                    "Use a profile with saved forecast bands: real or grid_plus_solar_plus_battery."
                ],
            },
        )

    if profile_id == "real":
        frame = _require(
            "forecast_predictions",
            "forecast_artifact_missing",
            "Saved REAL forecast predictions are not available.",
        )
        model_id = _model_id(model)
        predicted_col = model_id
        lower_col = f"{model_id}_lower_p90"
        upper_col = f"{model_id}_upper_p90"
        required_columns = {predicted_col, lower_col, upper_col}
        missing_columns = sorted(required_columns.difference(frame.columns))
        if missing_columns:
            raise ApiError(
                503,
                "forecast_artifact_schema_mismatch",
                "Saved REAL forecast predictions do not contain the requested model columns.",
                {"missing_columns": missing_columns},
            )
        mask = _date_mask_timestamps(frame.index, requested_date)
        selected = frame.loc[mask]
        unit = "kWh"
        source = str(ArtifactPaths.forecast_predictions)
        dataset_label = "real"
        note = "Saved test-set forecast rows with the frozen 90% prediction band."
        available_range = _available_range(frame.index)
    else:
        frame = _require(
            "solar_predictions",
            "solar_artifact_missing",
            "Saved SYNTHETIC solar predictions are not available.",
        )
        required_columns = {"timestamp", "variant", "actual_kw", "predicted_kw", "lower_kw", "upper_kw"}
        missing_columns = sorted(required_columns.difference(frame.columns))
        if missing_columns:
            raise ApiError(
                503,
                "solar_artifact_schema_mismatch",
                "Saved SYNTHETIC solar predictions do not contain the expected columns.",
                {"missing_columns": missing_columns},
            )
        timestamps = pd.to_datetime(frame["timestamp"])
        mask = _date_mask_timestamps(timestamps, requested_date) & (
            frame["variant"] == "with_solar"
        )
        selected = frame.loc[mask]
        model_id = "saved_solar_prediction"
        predicted_col = "predicted_kw"
        lower_col = "lower_kw"
        upper_col = "upper_kw"
        unit = "kW"
        source = str(ArtifactPaths.solar_predictions)
        dataset_label = "synthetic"
        note = "Saved synthetic grid-import prediction with solar available at the target hour."
        available_range = _available_range(timestamps[frame["variant"] == "with_solar"])

    if selected.empty:
        raise ApiError(
            404,
            "forecast_date_not_available",
            f"No saved forecast rows are available for profile '{profile_id}' on {requested_date.isoformat()}.",
            {
                "available_range": available_range,
                "needed": [
                    f"Saved forecast rows for {profile_id} on {requested_date.isoformat()}."
                ],
            },
        )

    points = []
    for timestamp, row in selected.iterrows():
        if profile_id == "real":
            timestamp_value = pd.Timestamp(timestamp).isoformat()
        else:
            timestamp_value = pd.Timestamp(row["timestamp"]).isoformat()
        points.append(
            {
                "timestamp": timestamp_value,
                "actual": _value(row["actual_kwh"]) if profile_id == "real" else _value(row["actual_kw"]),
                "predicted": float(row[predicted_col]),
                "lower": float(row[lower_col]),
                "upper": float(row[upper_col]),
            }
        )

    return {
        "profile": profile_id,
        "profile_label": profile_label,
        "dataset_label": dataset_label,
        "date": requested_date.isoformat(),
        "model": model_id,
        "unit": unit,
        "source": source,
        "available_range": available_range,
        "points": points,
        "count": len(points),
        "note": note,
    }


def battery_runtime(
    profile: ProfileId,
    state_of_charge: float,
    scenario: BatteryScenario = BatteryScenario.correct,
):
    profile_id = _profile_id(profile)
    lookup = _require(
        "battery_lookup",
        "battery_artifact_missing",
        "Saved battery runtime lookup is not available.",
    )
    labels = _profile_labels()
    entries = lookup.get("entries", {})
    if profile_id not in entries:
        raise ApiError(
            404,
            "battery_not_available",
            f"No saved battery runtime lookup is available for profile '{profile_id}'.",
            {
                "available_profiles": sorted(entries),
                "needed": ["Use a synthetic profile with a saved battery runtime lookup entry."],
            },
        )

    soc_key = f"{state_of_charge:.2f}"
    profile_entries = entries[profile_id]
    if soc_key not in profile_entries:
        raise ApiError(
            422,
            "unsupported_state_of_charge",
            f"State of charge {state_of_charge} is not present in the saved lookup.",
            {"allowed_values": lookup.get("state_of_charge_values", list(profile_entries))},
        )

    scenario_id = _scenario_id(scenario)
    entry = profile_entries[soc_key].get(scenario_id)
    if entry is None:
        raise ApiError(
            404,
            "battery_scenario_not_available",
            f"Battery scenario '{scenario_id}' is not present in the saved lookup.",
            {"available_scenarios": sorted(profile_entries[soc_key])},
        )

    return {
        "profile": profile_id,
        "profile_label": labels.get(profile_id, profile_id),
        "dataset_label": lookup.get("dataset_label", "synthetic"),
        "state_of_charge": state_of_charge,
        "scenario": scenario_id,
        "horizon_hours": int(lookup.get("forecast_hours", 72)),
        "low_hours": float(entry["low_hours"]),
        "expected_hours": float(entry["expected_hours"]),
        "high_hours": float(entry["high_hours"]),
        "load_only_low_hours": float(entry["load_only_low_hours"]),
        "load_only_expected_hours": float(entry["load_only_expected_hours"]),
        "load_only_high_hours": float(entry["load_only_high_hours"]),
        "assumptions": list(entry["assumptions"]),
        "validation_assumptions": list(lookup.get("global_assumptions", [])),
        "limitations": list(lookup.get("limitations", [])),
        "method": lookup.get("method", ""),
        "source": str(ArtifactPaths.battery_lookup),
    }


def degradation(level: Optional[str] = None):
    frame = _require(
        "degradation_results",
        "degradation_artifact_missing",
        "Saved degradation results are not available.",
    )
    metadata = _require(
        "degradation_metadata",
        "degradation_metadata_missing",
        "Saved degradation metadata is not available.",
    )
    if level is not None:
        selected = frame.loc[frame["level"] == level]
        if selected.empty:
            raise ApiError(
                404,
                "degradation_level_not_available",
                f"Degradation level '{level}' is not present in saved results.",
                {"available_levels": sorted(frame["level"].tolist())},
            )
    else:
        selected = frame

    levels = []
    for record in selected.to_dict("records"):
        levels.append({key: _value(value) for key, value in record.items()})
    return {
        "dataset_label": metadata.get("dataset_label", "real"),
        "source": str(ArtifactPaths.degradation_results),
        "metadata": metadata,
        "levels": levels,
        "count": len(levels),
        "note": "Saved gradual-degradation experiment rows; no model is refit when this endpoint is called.",
    }


def recommendations(
    profile: ProfileId,
    requested_date: Date,
    state_of_charge: float,
    model: ForecastModel = ForecastModel.xgboost,
):
    profile_id = _profile_id(profile)
    messages = []
    needed = []
    figures = {}
    forecast_result = None
    battery_result = None

    try:
        forecast_result = forecast(profile, requested_date, model)
    except ApiError as exc:
        messages.append(exc.message)
        needed.extend(exc.detail.get("needed", []))

    try:
        battery_result = battery_runtime(profile, state_of_charge, BatteryScenario.correct)
    except ApiError as exc:
        messages.append(exc.message)
        needed.extend(exc.detail.get("needed", []))

    if forecast_result:
        point = forecast_result["points"][0]
        unit = forecast_result["unit"]
        messages.append(
            f"Saved {profile_id} forecast for {forecast_result['date']} at "
            f"{point['timestamp']}: {point['predicted']:.3f} {unit}, with a 90% band "
            f"from {point['lower']:.3f} to {point['upper']:.3f} {unit}."
        )
        figures["forecast"] = {
            "date": forecast_result["date"],
            "timestamp": point["timestamp"],
            "predicted": point["predicted"],
            "lower": point["lower"],
            "upper": point["upper"],
            "unit": unit,
        }

    if battery_result:
        messages.append(
            f"Saved battery runtime for {profile_id} at {state_of_charge:.0%} state of charge: "
            f"{battery_result['low_hours']:.2f} to {battery_result['high_hours']:.2f} hours, "
            f"expected {battery_result['expected_hours']:.2f} hours."
        )
        figures["battery_runtime"] = {
            "low_hours": battery_result["low_hours"],
            "expected_hours": battery_result["expected_hours"],
            "high_hours": battery_result["high_hours"],
            "scenario": battery_result["scenario"],
        }

    messages.append("Values are saved experiment outputs; no new model or battery calculation was run.")
    return {
        "status": "ready" if forecast_result and battery_result else "data_needed",
        "profile": profile_id,
        "date": requested_date.isoformat(),
        "state_of_charge": state_of_charge,
        "messages": messages,
        "figures": figures,
        "needed": sorted(set(needed)),
    }
