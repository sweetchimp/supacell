import json
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
UCI_LOCATION = {
    "name": "Sceaux",
    "country": "France",
    "latitude": 48.77644,
    "longitude": 2.29026,
    "timezone": "Europe/Paris",
}
WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "precipitation",
)
ORIGIN_LAG_HOURS = (0, 3, 6, 12, 24)


def build_archive_url(start_date, end_date):
    params = {
        "latitude": UCI_LOCATION["latitude"],
        "longitude": UCI_LOCATION["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARIABLES),
        "models": "era5",
        "timeformat": "unixtime",
        "timezone": UCI_LOCATION["timezone"],
    }
    return f"{ARCHIVE_BASE_URL}?{urlencode(params)}"


def fetch_weather(start_date, end_date, raw_path=None):
    url = build_archive_url(start_date, end_date)
    with urlopen(url, timeout=120) as response:
        payload = json.load(response)
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason')}")
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, url


def build_weather_frame(payload):
    hourly = payload["hourly"]
    timestamps = pd.to_datetime(hourly["time"], unit="s", utc=True)
    local_index = pd.DatetimeIndex(
        timestamps.tz_convert(UCI_LOCATION["timezone"]).tz_localize(None),
        name="timestamp",
    )
    data = {
        f"weather_{variable}": hourly[variable] for variable in WEATHER_VARIABLES
    }
    frame = pd.DataFrame(data, index=local_index)
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    continuous_index = pd.date_range(
        frame.index.min(), frame.index.max(), freq="h", name="timestamp"
    )
    frame = frame.reindex(continuous_index)
    frame = frame.interpolate(method="linear", limit=1, limit_direction="both")
    frame.index.name = "timestamp"
    return frame


def build_weather_features(weather_frame, target_index, origin_offset_hours=1):
    target_index = pd.DatetimeIndex(target_index)
    origin_index = target_index.normalize() - pd.Timedelta(hours=origin_offset_hours)
    features = pd.DataFrame(index=target_index)

    for column in weather_frame.columns:
        series = weather_frame[column].astype(float)
        for lag in ORIGIN_LAG_HOURS:
            lookup = origin_index - pd.Timedelta(hours=lag)
            features[f"{column}_origin_lag_{lag}h"] = series.reindex(lookup).to_numpy(
                dtype=float
            )
        rolled = series.rolling("24h", min_periods=24)
        features[f"{column}_origin_roll24_mean"] = rolled.mean().reindex(
            origin_index
        ).to_numpy(dtype=float)
        features[f"{column}_origin_roll24_std"] = rolled.std().reindex(
            origin_index
        ).to_numpy(dtype=float)
        if "precipitation" in column:
            features[f"{column}_origin_roll24_sum"] = rolled.sum().reindex(
                origin_index
            ).to_numpy(dtype=float)

    return features


def join_weather_to_household(cleaned_path, weather_frame, output_path):
    frame = pd.read_csv(cleaned_path, parse_dates=["timestamp"])
    frame = frame.set_index("timestamp").sort_index()
    joined = frame.join(weather_frame, how="left")
    joined = joined[~joined.index.duplicated(keep="first")]
    joined.index.name = "timestamp"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index_label="timestamp")
    return joined
