from dataclasses import dataclass

import numpy as np
import pandas as pd

CAPACITY_HEALTH_RANGE = (0.80, 1.00)
EFFICIENCY_RANGE = (0.90, 0.98)


def clamp_battery(soc_kwh, capacity_kwh):
    return float(np.clip(soc_kwh, 0.0, capacity_kwh))


class BatteryConfig:
    def __init__(
        self,
        capacity_kwh=10.0,
        initial_soc_kwh=5.0,
        max_charge_kw=5.0,
        max_discharge_kw=5.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    ):
        if capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if charge_efficiency <= 0 or charge_efficiency > 1:
            raise ValueError("charge_efficiency must be in (0, 1]")
        if discharge_efficiency <= 0 or discharge_efficiency > 1:
            raise ValueError("discharge_efficiency must be in (0, 1]")
        self.capacity_kwh = float(capacity_kwh)
        self.initial_soc_kwh = clamp_battery(initial_soc_kwh, capacity_kwh)
        self.max_charge_kw = float(max_charge_kw)
        self.max_discharge_kw = float(max_discharge_kw)
        self.charge_efficiency = float(charge_efficiency)
        self.discharge_efficiency = float(discharge_efficiency)


def dispatch_battery(soc_kwh, charge_request_kw, discharge_request_kw, config):
    soc_kwh = clamp_battery(soc_kwh, config.capacity_kwh)
    charge_kw = 0.0
    discharge_kw = 0.0

    if charge_request_kw > 0:
        headroom_kwh = config.capacity_kwh - soc_kwh
        max_input_kw = headroom_kwh / config.charge_efficiency
        charge_kw = float(
            min(charge_request_kw, config.max_charge_kw, max_input_kw)
        )
        soc_kwh = clamp_battery(
            soc_kwh + charge_kw * config.charge_efficiency, config.capacity_kwh
        )

    if discharge_request_kw > 0:
        available_output_kw = soc_kwh * config.discharge_efficiency
        discharge_kw = float(
            min(discharge_request_kw, config.max_discharge_kw, available_output_kw)
        )
        soc_kwh = clamp_battery(
            soc_kwh - discharge_kw / config.discharge_efficiency,
            config.capacity_kwh,
        )

    return soc_kwh, charge_kw, discharge_kw


@dataclass(frozen=True)
class RuntimeEstimate:
    low_hours: float
    expected_hours: float
    high_hours: float
    assumptions: list


def _validate_runtime_inputs(capacity_kwh, soc_kwh, usable_dod_fraction, efficiency):
    if capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be positive")
    if not 0 <= soc_kwh <= capacity_kwh:
        raise ValueError("soc_kwh must be between 0 and capacity_kwh")
    if not 0 < usable_dod_fraction <= 1:
        raise ValueError("usable_dod_fraction must be in (0, 1]")
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")


def _validate_fraction_range(name, values, allow_one=True):
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f"{name} must be a two-value tuple")
    low, high = float(values[0]), float(values[1])
    upper_bound = 1.0 if allow_one else 1.0
    if not 0 < low <= high <= upper_bound:
        raise ValueError(f"{name} must satisfy 0 < low <= high <= 1")
    return low, high


def _usable_ac_kwh(capacity_kwh, soc_kwh, usable_dod_fraction, efficiency, health):
    effective_capacity = capacity_kwh * health
    effective_soc = soc_kwh * health
    floor_kwh = effective_capacity * (1.0 - usable_dod_fraction)
    usable_stored_kwh = max(0.0, effective_soc - floor_kwh)
    return usable_stored_kwh * efficiency, floor_kwh, usable_stored_kwh, effective_capacity


def _runtime_for_load(load_kw, usable_ac_kwh):
    load = np.asarray(load_kw, dtype=float)
    if usable_ac_kwh <= 0:
        return 0.0
    remaining = usable_ac_kwh
    for hour, demand in enumerate(load, start=1):
        if demand <= 0:
            continue
        if demand >= remaining:
            return float((hour - 1) + remaining / demand)
        remaining -= demand
    return float(len(load))


def _forecast_scenarios(forecast, nominal_level):
    if not isinstance(forecast, pd.DataFrame):
        raise ValueError("forecast must be a pandas DataFrame")
    required = {"expected_kw", "p10_kw", "p90_kw"}
    missing = required - set(forecast.columns)
    if missing:
        raise ValueError(f"forecast missing columns: {sorted(missing)}")
    values = {}
    for column in ("expected_kw", "p10_kw", "p90_kw"):
        series = pd.to_numeric(forecast[column], errors="coerce").to_numpy(dtype=float)
        if len(series) == 0 or not np.isfinite(series).all() or (series < 0).any():
            raise ValueError(f"forecast column {column} must be finite and non-negative")
        values[column] = series
    if not (values["p10_kw"] <= values["expected_kw"]).all():
        raise ValueError("p10_kw must not exceed expected_kw")
    if not (values["expected_kw"] <= values["p90_kw"]).all():
        raise ValueError("p90_kw must not be below expected_kw")
    if not 0 < nominal_level < 1:
        raise ValueError("nominal_level must be between 0 and 1")
    return values


def estimate_runtime_range(
    capacity_kwh,
    soc_kwh,
    usable_dod_fraction,
    efficiency,
    forecast,
    nominal_level=0.80,
    capacity_health_range=CAPACITY_HEALTH_RANGE,
    efficiency_range=EFFICIENCY_RANGE,
):
    _validate_runtime_inputs(capacity_kwh, soc_kwh, usable_dod_fraction, efficiency)
    scenarios = _forecast_scenarios(forecast, nominal_level)
    health_low, health_high = _validate_fraction_range(
        "capacity_health_range", capacity_health_range
    )
    efficiency_low, efficiency_high = _validate_fraction_range(
        "efficiency_range", efficiency_range
    )

    expected_ac_kwh, floor_kwh, expected_stored_kwh, _ = _usable_ac_kwh(
        capacity_kwh, soc_kwh, usable_dod_fraction, efficiency, 1.0
    )
    low_ac_kwh, _, _, low_capacity_kwh = _usable_ac_kwh(
        capacity_kwh, soc_kwh, usable_dod_fraction, efficiency_low, health_low
    )
    high_ac_kwh, _, _, high_capacity_kwh = _usable_ac_kwh(
        capacity_kwh, soc_kwh, usable_dod_fraction, efficiency_high, health_high
    )

    low_hours = _runtime_for_load(scenarios["p90_kw"], low_ac_kwh)
    expected_hours = _runtime_for_load(scenarios["expected_kw"], expected_ac_kwh)
    high_hours = _runtime_for_load(scenarios["p10_kw"], high_ac_kwh)
    horizon_hours = len(scenarios["expected_kw"])

    assumptions = [
        f"Nominal {nominal_level:.0%} load band supplied as p10/p90; endpoints are treated as simultaneous load paths.",
        f"Usable stored energy at point assumptions is max(0, {soc_kwh:.3f} kWh SoC - {floor_kwh:.3f} kWh DoD floor) = {expected_stored_kwh:.3f} kWh.",
        f"Point one-way conversion efficiency of {efficiency:.0%} gives {expected_ac_kwh:.3f} kWh usable AC energy.",
        f"Capacity-health assumption range is {health_low:.0%}-{health_high:.0%} of supplied capacity; health scales both capacity and SoC.",
        f"One-way efficiency assumption range is {efficiency_low:.0%}-{efficiency_high:.0%}.",
        "Low runtime combines p90 load, low capacity health and low efficiency; high runtime combines p10 load, high capacity health and high efficiency.",
        "The combined range is a simultaneous worst-case envelope, not a calibrated probability interval.",
        f"Low-case effective capacity is {low_capacity_kwh:.3f} kWh; high-case effective capacity is {high_capacity_kwh:.3f} kWh.",
        "Load is applied in one-hour steps with linear interpolation inside the depletion hour.",
        "No solar generation, grid charging, load shifting or reserve requirement during the horizon.",
        "No inverter power cap, temperature derating or aging beyond the supplied capacity-health range.",
        f"Runtime is capped at the {horizon_hours}-hour forecast horizon when the battery is not depleted.",
    ]
    return RuntimeEstimate(
        low_hours=low_hours,
        expected_hours=expected_hours,
        high_hours=high_hours,
        assumptions=assumptions,
    )


def simulate_discharge(
    load_kw,
    capacity_kwh,
    initial_soc_kwh,
    usable_dod_fraction,
    efficiency,
    max_discharge_kw=None,
):
    _validate_runtime_inputs(capacity_kwh, initial_soc_kwh, usable_dod_fraction, efficiency)
    if max_discharge_kw is not None and max_discharge_kw <= 0:
        raise ValueError("max_discharge_kw must be positive when supplied")
    load = np.asarray(load_kw, dtype=float)
    if not np.isfinite(load).all() or (load < 0).any():
        raise ValueError("load_kw must be finite and non-negative")

    floor_kwh = capacity_kwh * (1.0 - usable_dod_fraction)
    soc_kwh = clamp_battery(initial_soc_kwh, capacity_kwh)
    if soc_kwh <= floor_kwh:
        return 0.0

    for hour, demand_kw in enumerate(load, start=1):
        if demand_kw <= 0:
            continue
        if max_discharge_kw is not None and demand_kw > max_discharge_kw:
            return float(hour - 1)
        available_output_kw = max(0.0, soc_kwh - floor_kwh) * efficiency
        if demand_kw >= available_output_kw:
            return float((hour - 1) + available_output_kw / demand_kw)
        soc_kwh = clamp_battery(soc_kwh - demand_kw / efficiency, capacity_kwh)
    return float(len(load))
