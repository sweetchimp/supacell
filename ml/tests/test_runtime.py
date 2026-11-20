import pandas as pd
import pytest

from supacell_ml.battery import (
    CAPACITY_HEALTH_RANGE,
    EFFICIENCY_RANGE,
    estimate_runtime_range,
    simulate_discharge,
)


def forecast():
    return pd.DataFrame(
        {
            "expected_kw": [1.0, 1.0, 1.0, 1.0],
            "p10_kw": [0.5, 0.5, 0.5, 0.5],
            "p90_kw": [1.5, 1.5, 1.5, 1.5],
        }
    )


def test_runtime_range_is_ordered_and_has_plain_assumptions():
    estimate = estimate_runtime_range(
        capacity_kwh=10.0,
        soc_kwh=8.0,
        usable_dod_fraction=0.8,
        efficiency=0.95,
        forecast=forecast(),
        nominal_level=0.8,
    )

    assert estimate.low_hours <= estimate.expected_hours <= estimate.high_hours
    assert isinstance(estimate.assumptions, list)
    assert all(isinstance(item, str) for item in estimate.assumptions)
    assert len(estimate.assumptions) >= 5


def test_runtime_range_matches_physics_with_correct_assumptions():
    load = forecast()["expected_kw"].to_numpy()
    estimate = estimate_runtime_range(
        capacity_kwh=10.0,
        soc_kwh=8.0,
        usable_dod_fraction=0.8,
        efficiency=0.95,
        forecast=forecast(),
    )
    true_runtime = simulate_discharge(
        load,
        capacity_kwh=10.0,
        initial_soc_kwh=8.0,
        usable_dod_fraction=0.8,
        efficiency=0.95,
    )

    assert true_runtime == pytest.approx(estimate.expected_hours)


def test_physics_simulator_respects_dod_floor():
    load = [1.0] * 4
    full_range = simulate_discharge(load, 2.0, 2.0, 1.0, 1.0)
    limited_range = simulate_discharge(load, 2.0, 2.0, 0.5, 1.0)

    assert full_range == pytest.approx(2.0)
    assert limited_range == pytest.approx(1.0)


def test_invalid_forecast_order_is_rejected():
    bad_forecast = forecast()
    bad_forecast.loc[0, "p10_kw"] = 2.0
    with pytest.raises(ValueError):
        estimate_runtime_range(10.0, 8.0, 0.8, 0.95, bad_forecast)


def test_assumption_ranges_are_frozen():
    assert CAPACITY_HEALTH_RANGE == (0.80, 1.00)
    assert EFFICIENCY_RANGE == (0.90, 0.98)


def test_combined_range_is_wider_than_load_only_range():
    load_only = estimate_runtime_range(
        10.0,
        8.0,
        0.8,
        0.95,
        forecast(),
        capacity_health_range=(1.0, 1.0),
        efficiency_range=(0.95, 0.95),
    )
    combined = estimate_runtime_range(10.0, 8.0, 0.8, 0.95, forecast())

    assert combined.low_hours <= load_only.low_hours
    assert combined.high_hours >= load_only.high_hours
    assert (combined.high_hours - combined.low_hours) >= (
        load_only.high_hours - load_only.low_hours
    )
    assert any("worst-case envelope" in item for item in combined.assumptions)


def test_low_case_uses_low_health_and_low_efficiency():
    combined = estimate_runtime_range(10.0, 8.0, 0.8, 0.95, forecast())
    low_only = estimate_runtime_range(
        10.0,
        8.0,
        0.8,
        0.95,
        forecast(),
        capacity_health_range=(0.8, 0.8),
        efficiency_range=(0.9, 0.9),
    )

    assert combined.low_hours == pytest.approx(low_only.low_hours)


def test_invalid_assumption_ranges_are_rejected():
    with pytest.raises(ValueError):
        estimate_runtime_range(
            10.0,
            8.0,
            0.8,
            0.95,
            forecast(),
            capacity_health_range=(1.0, 0.8),
        )
    with pytest.raises(ValueError):
        estimate_runtime_range(
            10.0,
            8.0,
            0.8,
            0.95,
            forecast(),
            efficiency_range=(0.0, 0.9),
        )
