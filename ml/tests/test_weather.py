from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from supacell_ml.weather import (
    ARCHIVE_BASE_URL,
    UCI_LOCATION,
    WEATHER_VARIABLES,
    build_archive_url,
    build_weather_features,
    build_weather_frame,
)


def test_archive_url_uses_era5_and_documented_location():
    url = build_archive_url("2006-12-16", "2010-11-27")
    query = parse_qs(urlparse(url).query)

    assert url.startswith(ARCHIVE_BASE_URL)
    assert query["models"] == ["era5"]
    assert query["timeformat"] == ["unixtime"]
    assert query["latitude"] == [str(UCI_LOCATION["latitude"])]
    assert query["longitude"] == [str(UCI_LOCATION["longitude"])]
    assert query["timezone"] == ["Europe/Paris"]
    assert query["hourly"] == [",".join(WEATHER_VARIABLES)]


def test_weather_frame_converts_unix_time_to_local_naive_index():
    utc_index = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
    hourly = {"time": [int(value.timestamp()) for value in utc_index]}
    hourly.update({variable: [1.0, 2.0, 3.0] for variable in WEATHER_VARIABLES})

    frame = build_weather_frame({"hourly": hourly})

    assert frame.columns.tolist() == [f"weather_{item}" for item in WEATHER_VARIABLES]
    assert frame.index[0] == pd.Timestamp("2020-01-01 01:00:00")
    assert frame.index[-1] == pd.Timestamp("2020-01-01 03:00:00")


def test_weather_features_use_previous_day_origin_and_not_target_hour():
    weather_index = pd.date_range("2020-01-01", periods=48, freq="h")
    weather = pd.DataFrame(
        {
            f"weather_{variable}": np.arange(48, dtype=float) + offset * 100.0
            for offset, variable in enumerate(WEATHER_VARIABLES)
        },
        index=weather_index,
    )
    target_index = pd.DatetimeIndex(["2020-01-02 00:00", "2020-01-02 01:00"])

    features = build_weather_features(weather, target_index)
    sample = target_index[0]
    origin = pd.Timestamp("2020-01-01 23:00")
    lag_column = "weather_temperature_2m_origin_lag_0h"
    mean_column = "weather_temperature_2m_origin_roll24_mean"

    assert features.loc[sample, lag_column] == pytest.approx(
        weather.loc[origin, "weather_temperature_2m"]
    )
    assert features.loc[sample, lag_column] != pytest.approx(
        weather.loc[sample, "weather_temperature_2m"]
    )
    assert features.loc[sample, mean_column] == pytest.approx(
        weather.loc[weather_index[0]:origin, "weather_temperature_2m"].mean()
    )
    assert "weather_precipitation_origin_roll24_sum" in features.columns
