import numpy as np
import pandas as pd

from supacell_ml.degradation import (
    DEGRADATION_LEVELS,
    apply_degradation,
    apply_feature_missingness,
    fill_degraded_features,
    lag_feature_columns,
)
from supacell_ml.forecasting import build_day_ahead_features


def sample_energy():
    index = pd.date_range("2020-01-01", periods=24 * 60, freq="h", name="timestamp")
    series = pd.Series(np.linspace(1.0, 2.0, len(index)), index=index)
    series.name = "global_active_energy_kwh"
    return series


def test_no_solar_and_no_weather_levels_are_noops():
    energy = sample_energy()
    validation_start = "2020-03-01"
    for level in (
        {"name": "no_solar_data", "kind": "noop"},
        {"name": "no_weather_features", "kind": "noop"},
    ):
        pd.testing.assert_series_equal(
            apply_degradation(energy, level, validation_start), energy
        )


def test_feature_missingness_is_seeded_and_preserves_timestamps():
    energy = sample_energy()
    features, target = build_day_ahead_features(energy)
    first = apply_feature_missingness(features, 0.20, seed=42)
    second = apply_feature_missingness(features, 0.20, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(features)
    assert first[lag_feature_columns(features)].isna().any().any()


def test_degraded_prediction_inputs_differ_from_clean_inputs():
    energy = sample_energy()
    clean_features, target = build_day_ahead_features(energy)
    train_index = clean_features.index[: len(clean_features) // 2]
    degraded = apply_feature_missingness(clean_features, 0.50, seed=180)
    degraded = fill_degraded_features(degraded, train_index)
    lag_columns = lag_feature_columns(clean_features)

    assert list(degraded.index) == list(clean_features.index)
    assert not np.allclose(
        clean_features.loc[degraded.index, lag_columns].to_numpy(dtype=float),
        degraded.loc[degraded.index, lag_columns].to_numpy(dtype=float),
    )


def test_harsh_degradation_levels_and_short_history_are_defined():
    names = {level["name"] for level in DEGRADATION_LEVELS}
    assert {"missing_50pct", "missing_80pct", "history_3mo"} <= names
    missing_levels = {
        level["name"]: level for level in DEGRADATION_LEVELS if level["kind"] == "missing"
    }
    assert missing_levels["missing_50pct"]["seed"] == 150
    assert missing_levels["missing_80pct"]["seed"] == 180


def test_features_can_use_degraded_history_with_clean_targets():
    energy = sample_energy()
    history = energy.copy()
    history.iloc[:200] = np.nan

    features, target = build_day_ahead_features(
        history, target=energy, fill_missing=True
    )

    assert len(features) > 0
    assert target.name == "actual_kwh"
    assert set(features.index) == set(target.index)
