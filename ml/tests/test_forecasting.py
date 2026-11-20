import numpy as np
import pandas as pd
import pytest

from supacell_ml.forecasting import (
    ForecastConfig,
    build_day_ahead_features,
    chronological_split,
    conformal_radius,
    regression_metrics,
)


def sample_energy_series(days=10):
    index = pd.date_range("2020-01-01", periods=24 * days, freq="h", name="timestamp")
    values = pd.Series(np.arange(1.0, len(index) + 1.0), index=index)
    values.name = "global_active_energy_kwh"
    return values


def test_features_use_lags_available_at_forecast_origin():
    energy = sample_energy_series()
    features, target = build_day_ahead_features(energy)
    sample = features.index[0]
    origin = sample.normalize() - pd.Timedelta(hours=1)

    assert features.loc[sample, "lag_24h"] == pytest.approx(
        energy.loc[sample - pd.Timedelta(hours=24)]
    )
    assert features.loc[sample, "lag_168h"] == pytest.approx(
        energy.loc[sample - pd.Timedelta(hours=168)]
    )
    assert features.loc[sample, "origin_lag_0h"] == pytest.approx(energy.loc[origin])
    assert features.loc[sample, "origin_roll_mean_24h"] == pytest.approx(
        energy.loc[origin - pd.Timedelta(hours=23) : origin].mean()
    )
    assert target.name == "actual_kwh"


def test_exogenous_features_are_aligned_to_target_timestamps():
    energy = sample_energy_series()
    exogenous = pd.DataFrame({"solar_kw": np.linspace(0.0, 4.0, len(energy))}, index=energy.index)
    features, target = build_day_ahead_features(energy, exogenous=exogenous)
    sample = features.index[10]

    assert "exog_solar_kw" in features.columns
    assert features.loc[sample, "exog_solar_kw"] == pytest.approx(
        exogenous.loc[sample, "solar_kw"]
    )
    assert set(features.index) == set(target.index)


def test_chronological_split_is_contiguous_and_never_shuffles():
    energy = sample_energy_series(days=30)
    features, target = build_day_ahead_features(energy)
    config = ForecastConfig(train_fraction=0.6, validation_fraction=0.2)
    split = chronological_split(features.index, config)

    assert split.train.equals(features.index[: len(split.train)])
    assert split.validation.equals(features.index[len(split.train) : len(split.train) + len(split.validation)])
    assert split.test.equals(features.index[len(split.train) + len(split.validation) :])
    assert split.train.max() < split.validation.min()
    assert split.validation.max() < split.test.min()
    assert len(split.train) + len(split.validation) + len(split.test) == len(features)


def test_mape_is_omitted_for_nonpositive_actual():
    metrics = regression_metrics([1.0, 0.0], [1.0, 1.0])
    assert np.isnan(metrics["mape"])


def test_conformal_radius_uses_finite_sample_rank():
    actual = np.arange(1.0, 21.0)
    predicted = np.zeros(20)
    radius = conformal_radius(actual, predicted, alpha=0.10)
    assert radius == pytest.approx(19.0)
