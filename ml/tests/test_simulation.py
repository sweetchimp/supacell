import pandas as pd
import pytest

from supacell_ml.battery import BatteryConfig
from supacell_ml.simulation import (
    GRID_ONLY,
    GRID_PLUS_SOLAR_PLUS_BATTERY,
    OFF_GRID_SOLAR_PLUS_BATTERY_WITH_OUTAGES,
    SimulationConfig,
    generate_household_timeseries,
)


def test_same_seed_produces_identical_output():
    first = generate_household_timeseries(
        OFF_GRID_SOLAR_PLUS_BATTERY_WITH_OUTAGES,
        start="2025-03-01",
        hours=168,
        seed=123,
    )
    second = generate_household_timeseries(
        OFF_GRID_SOLAR_PLUS_BATTERY_WITH_OUTAGES,
        start="2025-03-01",
        hours=168,
        seed=123,
    )
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_output():
    first = generate_household_timeseries(
        GRID_PLUS_SOLAR_PLUS_BATTERY, hours=72, seed=1
    )
    second = generate_household_timeseries(
        GRID_PLUS_SOLAR_PLUS_BATTERY, hours=72, seed=2
    )
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(first, second)


def test_battery_state_stays_within_capacity():
    capacity_kwh = 8.0
    config = SimulationConfig(
        hours=24 * 30,
        seed=99,
        battery=BatteryConfig(
            capacity_kwh=capacity_kwh,
            initial_soc_kwh=capacity_kwh,
            max_charge_kw=3.0,
            max_discharge_kw=3.0,
        ),
    )
    for profile in (
        GRID_PLUS_SOLAR_PLUS_BATTERY,
        OFF_GRID_SOLAR_PLUS_BATTERY_WITH_OUTAGES,
    ):
        frame = generate_household_timeseries(profile, config=config)
        assert (frame["battery_kwh"] >= 0.0).all()
        assert (frame["battery_kwh"] <= capacity_kwh).all()
        assert (frame["dataset_label"] == "SYNTHETIC").all()


def test_grid_availability_matches_profiles():
    config = SimulationConfig(hours=48, seed=5)
    grid_only = generate_household_timeseries(GRID_ONLY, config=config)
    off_grid = generate_household_timeseries(
        OFF_GRID_SOLAR_PLUS_BATTERY_WITH_OUTAGES, config=config
    )

    assert grid_only["grid_available"].all()
    assert not off_grid["grid_available"].any()
    assert (grid_only["solar_kw"] == 0.0).all()
