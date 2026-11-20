from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .battery import BatteryConfig, clamp_battery, dispatch_battery
from .data_labels import validate_label

GRID_ONLY = "grid_only"
GRID_PLUS_SOLAR_PLUS_BATTERY = "grid_plus_solar_plus_battery"
OFF_GRID_SOLAR_PLUS_BATTERY_WITH_OUTAGES = (
    "off_grid_solar_plus_battery_with_outages"
)

PROFILES = (
    GRID_ONLY,
    GRID_PLUS_SOLAR_PLUS_BATTERY,
    OFF_GRID_SOLAR_PLUS_BATTERY_WITH_OUTAGES,
)

PROFILE_LABELS = {
    GRID_ONLY: "Grid only",
    GRID_PLUS_SOLAR_PLUS_BATTERY: "Grid plus solar plus battery",
    OFF_GRID_SOLAR_PLUS_BATTERY_WITH_OUTAGES: (
        "Off-grid solar plus battery with outages"
    ),
}

SYNTHETIC = "SYNTHETIC"


@dataclass(frozen=True)
class SimulationConfig:
    start: str = "2025-01-01"
    hours: int = 8760
    seed: int = 42
    demand_base_kw: float = 1.0
    demand_noise_kw: float = 0.12
    min_demand_kw: float = 0.15
    solar_capacity_kw: float = 4.0
    cloud_mean: float = 0.75
    cloud_std: float = 0.18
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    peak_shaving_hours: tuple = (17, 18, 19, 20, 21)

    def __post_init__(self):
        if self.hours <= 0:
            raise ValueError("hours must be positive")
        if self.demand_noise_kw < 0 or self.cloud_std < 0:
            raise ValueError("noise standard deviations must be non-negative")
        if not 0 < self.cloud_mean <= 1:
            raise ValueError("cloud_mean must be in (0, 1]")
        object.__setattr__(self, "hours", int(self.hours))


def validate_profile(profile):
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}; got {profile!r}")
    return profile


def _hour_of_day(index):
    return index.hour.to_numpy() + index.minute.to_numpy() / 60.0


def _demand_profile(index, rng, config):
    hours = _hour_of_day(index)
    day_of_year = index.dayofyear.to_numpy()
    morning = np.exp(-0.5 * ((hours - 8.0) / 2.0) ** 2)
    evening = np.exp(-0.5 * ((hours - 19.0) / 2.5) ** 2)
    midday = np.exp(-0.5 * ((hours - 13.0) / 4.0) ** 2)
    shape = 0.35 + 0.45 * morning + 0.85 * evening + 0.18 * midday
    weekend = np.where(index.dayofweek >= 5, 1.12, 1.0)
    season = 1.0 + 0.15 * np.cos(2.0 * np.pi * (day_of_year - 15.0) / 365.25)
    noise = rng.normal(0.0, config.demand_noise_kw, len(index))
    demand = config.demand_base_kw * shape * weekend * season + noise
    return np.clip(demand, config.min_demand_kw, None)


def _solar_profile(index, rng, config):
    hours = _hour_of_day(index)
    day_of_year = index.dayofyear.to_numpy()
    daylight_hours = 9.0 + 5.0 * (
        1.0 - np.cos(2.0 * np.pi * (day_of_year - 172.0) / 365.25)
    ) / 2.0
    sunrise = 12.0 - daylight_hours / 2.0
    phase = (hours - sunrise) / daylight_hours
    in_daylight = (phase > 0.0) & (phase < 1.0)
    clear_sky = np.sin(np.pi * np.clip(phase, 0.0, 1.0)) ** 1.2
    cloud = rng.normal(config.cloud_mean, config.cloud_std, len(index))
    cloud = np.clip(cloud, 0.05, 1.0)
    solar = config.solar_capacity_kw * clear_sky * cloud
    return solar * in_daylight


def _grid_availability(profile, hours):
    if profile == OFF_GRID_SOLAR_PLUS_BATTERY_WITH_OUTAGES:
        return np.zeros(hours, dtype=bool)
    return np.ones(hours, dtype=bool)


def generate_household_timeseries(
    profile,
    start=None,
    hours=None,
    seed=None,
    config=None,
):
    profile = validate_profile(profile)
    config = config if config is not None else SimulationConfig()
    if start is not None:
        config = replace(config, start=start)
    if hours is not None:
        config = replace(config, hours=hours)
    if seed is not None:
        config = replace(config, seed=seed)

    rng = np.random.default_rng(config.seed)
    index = pd.date_range(
        start=config.start, periods=config.hours, freq="h", name="timestamp"
    )
    demand = _demand_profile(index, rng, config)
    has_solar = profile != GRID_ONLY
    has_battery = profile != GRID_ONLY
    solar = _solar_profile(index, rng, config) if has_solar else np.zeros(len(index))
    grid_available = _grid_availability(profile, len(index))

    state_of_charge = clamp_battery(
        config.battery.initial_soc_kwh, config.battery.capacity_kwh
    )
    rows = []
    for position, timestamp in enumerate(index):
        demand_kw = float(demand[position])
        solar_kw = float(solar[position])
        grid_on = bool(grid_available[position])

        served_by_solar = min(demand_kw, solar_kw)
        surplus_kw = solar_kw - served_by_solar
        deficit_kw = demand_kw - served_by_solar
        charge_kw = 0.0
        discharge_kw = 0.0

        if has_battery and surplus_kw > 0:
            state_of_charge, charge_kw, _ = dispatch_battery(
                state_of_charge, surplus_kw, 0.0, config.battery
            )
            surplus_kw = max(0.0, surplus_kw - charge_kw)

        peak_shaving = timestamp.hour in config.peak_shaving_hours
        if has_battery and deficit_kw > 0 and (not grid_on or peak_shaving):
            state_of_charge, _, discharge_kw = dispatch_battery(
                state_of_charge, 0.0, deficit_kw, config.battery
            )
            deficit_kw = max(
                0.0, deficit_kw - discharge_kw * config.battery.discharge_efficiency
            )

        grid_import_kw = deficit_kw if grid_on else 0.0
        unmet_demand_kw = 0.0 if grid_on else deficit_kw
        solar_curtailment_kw = surplus_kw
        battery_kwh = state_of_charge if has_battery else 0.0

        rows.append(
            {
                "dataset_label": SYNTHETIC,
                "profile": profile,
                "profile_label": PROFILE_LABELS[profile],
                "demand_kw": demand_kw,
                "solar_kw": solar_kw,
                "battery_kwh": battery_kwh,
                "battery_charge_kw": charge_kw,
                "battery_discharge_kw": discharge_kw,
                "grid_available": grid_on,
                "grid_import_kw": grid_import_kw,
                "unmet_demand_kw": unmet_demand_kw,
                "solar_curtailment_kw": solar_curtailment_kw,
                "loss_of_load": unmet_demand_kw > 0.0,
            }
        )

    frame = pd.DataFrame(rows, index=index)
    validate_label("synthetic")
    return frame
