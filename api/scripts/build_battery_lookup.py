import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from supacell_ml.battery import estimate_runtime_range  # noqa: E402
from supacell_ml.simulation import PROFILES, PROFILE_LABELS  # noqa: E402

RUNTIME_METADATA_PATH = (
    REPO_ROOT / "results" / "battery_runtime" / "runtime_metadata_SYNTHETIC.json"
)
SIMULATION_DIR = REPO_ROOT / "results" / "simulations"
OUTPUT_PATH = REPO_ROOT / "api" / "data" / "battery_runtime_lookup_SYNTHETIC.json"
STATE_OF_CHARGE_VALUES = (0.2, 0.4, 0.6, 0.8, 1.0)
FORECAST_HOURS = 72
CAPACITY_KWH = 10.0


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def clean_hours(value):
    value = float(value)
    return 0.0 if abs(value) < 1e-9 else value


def simulation_path(profile):
    return SIMULATION_DIR / f"{profile}_hourly_SYNTHETIC.csv"


def build_forecast(demand_kw, band_fraction):
    expected = np.asarray(demand_kw, dtype=float)
    if len(expected) != FORECAST_HOURS:
        raise ValueError(f"expected {FORECAST_HOURS} load rows, got {len(expected)}")
    if not np.isfinite(expected).all() or (expected < 0).any():
        raise ValueError("saved simulation demand must be finite and non-negative")
    return pd.DataFrame(
        {
            "expected_kw": expected,
            "p10_kw": expected * (1.0 - band_fraction),
            "p90_kw": expected * (1.0 + band_fraction),
        }
    )


def estimate(capacity_health_range, efficiency_range, forecast, assumed_efficiency, soc_kwh):
    return estimate_runtime_range(
        CAPACITY_KWH,
        soc_kwh,
        usable_dod_fraction=0.80,
        efficiency=assumed_efficiency,
        forecast=forecast,
        nominal_level=0.80,
        capacity_health_range=capacity_health_range,
        efficiency_range=efficiency_range,
    )


def main():
    metadata = json.loads(RUNTIME_METADATA_PATH.read_text(encoding="utf-8"))
    entries = {}
    for profile in PROFILES:
        path = simulation_path(profile)
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        demand = frame["demand_kw"].head(FORECAST_HOURS).to_numpy(dtype=float)
        profile_entries = {}
        for soc_fraction in STATE_OF_CHARGE_VALUES:
            soc_kwh = CAPACITY_KWH * soc_fraction
            scenario_entries = {}
            for scenario_name, scenario in metadata["scenarios"].items():
                forecast = build_forecast(demand, scenario["band_fraction"])
                assumed_efficiency = float(scenario["assumed_efficiency"])
                combined = estimate(
                    tuple(metadata["capacity_health_range"]),
                    tuple(metadata["efficiency_range"]),
                    forecast,
                    assumed_efficiency,
                    soc_kwh,
                )
                load_only = estimate(
                    (1.0, 1.0),
                    (assumed_efficiency, assumed_efficiency),
                    forecast,
                    assumed_efficiency,
                    soc_kwh,
                )
                scenario_entries[scenario_name] = {
                    "low_hours": clean_hours(combined.low_hours),
                    "expected_hours": clean_hours(combined.expected_hours),
                    "high_hours": clean_hours(combined.high_hours),
                    "load_only_low_hours": clean_hours(load_only.low_hours),
                    "load_only_expected_hours": clean_hours(load_only.expected_hours),
                    "load_only_high_hours": clean_hours(load_only.high_hours),
                    "assumptions": combined.assumptions,
                }
            profile_entries[f"{soc_fraction:.2f}"] = scenario_entries
        entries[profile] = profile_entries

    lookup = {
        "dataset_label": "synthetic",
        "forecast_hours": FORECAST_HOURS,
        "generated_by": "api/scripts/build_battery_lookup.py",
        "method": (
            "Offline deterministic estimates using the first 72 hours of saved simulation "
            "demand as expected load, p10/p90 from each frozen scenario band fraction, "
            "and the frozen capacity-health and efficiency ranges."
        ),
        "source_files": {
            "runtime_metadata": str(RUNTIME_METADATA_PATH.relative_to(REPO_ROOT)),
            "simulations": [
                str(simulation_path(profile).relative_to(REPO_ROOT)) for profile in PROFILES
            ],
        },
        "profile_labels": PROFILE_LABELS,
        "capacity_kwh": CAPACITY_KWH,
        "usable_dod": metadata["usable_dod"],
        "nominal_level": metadata["nominal_level"],
        "capacity_health_range": metadata["capacity_health_range"],
        "capacity_range_justification": metadata["capacity_range_justification"],
        "efficiency_range": metadata["efficiency_range"],
        "efficiency_range_justification": metadata["efficiency_range_justification"],
        "state_of_charge_values": list(STATE_OF_CHARGE_VALUES),
        "scenarios": list(metadata["scenarios"]),
        "global_assumptions": metadata["assumptions"],
        "limitations": [
            "These are deterministic synthetic lookup values, not measured battery behavior.",
            "Only the first 72 hours of each saved simulation profile are used.",
            "Unsupported state-of-charge values are not computed at request time.",
            "The combined range is a worst-case envelope, not a calibrated probability interval.",
        ],
        "entries": entries,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(lookup, indent=2, default=json_default), encoding="utf-8"
    )
    print(f"lookup={OUTPUT_PATH}")
    print(
        f"profiles={len(entries)} soc_values={len(STATE_OF_CHARGE_VALUES)} "
        f"scenarios={len(metadata['scenarios'])}"
    )


if __name__ == "__main__":
    main()
