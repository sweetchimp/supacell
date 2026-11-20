from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

RAW_COLUMNS = (
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)

NUMERIC_COLUMNS = (
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)

ENERGY_OUTPUT_COLUMNS = (
    "global_active_energy_kwh",
    "sub_metering_1_energy_kwh",
    "sub_metering_2_energy_kwh",
    "sub_metering_3_energy_kwh",
)


@dataclass(frozen=True)
class HouseholdPowerStats:
    source_path: str
    sha256: str
    raw_rows: int
    observed_start: str
    observed_end: str
    modal_interval_seconds: float
    unique_timestamps: int
    duplicate_timestamps_removed: int
    missing_cell_pct: float
    rows_with_missing_pct: float
    missing_by_column_pct: dict
    cleaned_start: str
    cleaned_end: str
    cleaned_rows: int
    cleaned_energy_missing_pct: float
    min_observations_per_hour: int


@dataclass(frozen=True)
class HouseholdPowerResult:
    hourly: pd.DataFrame
    stats: HouseholdPowerStats


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stringify_timestamp(value):
    return pd.Timestamp(value).isoformat()


def _validate_parameters(min_observations_per_hour, interpolation_limit):
    if not 1 <= min_observations_per_hour <= 60:
        raise ValueError("min_observations_per_hour must be between 1 and 60")
    if interpolation_limit < 0:
        raise ValueError("interpolation_limit must be non-negative")


def load_household_power(
    raw_path,
    output_path=None,
    stats_path=None,
    min_observations_per_hour=50,
    interpolation_limit=5,
):
    _validate_parameters(min_observations_per_hour, interpolation_limit)
    raw_path = Path(raw_path)
    raw = pd.read_csv(
        raw_path,
        sep=";",
        dtype=str,
        na_values=["?", ""],
        keep_default_na=True,
        low_memory=False,
    )
    missing_columns = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"missing expected columns: {missing_columns}")

    for column in raw.columns:
        if raw[column].dtype == object:
            raw[column] = raw[column].str.strip()
    raw = raw.replace({"": np.nan, "?": np.nan})
    raw = raw[list(RAW_COLUMNS)]

    timestamp = pd.to_datetime(
        raw["Date"] + " " + raw["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    if timestamp.isna().any():
        bad_rows = int(timestamp.isna().sum())
        raise ValueError(f"could not parse {bad_rows} timestamps")
    raw = raw.assign(timestamp=timestamp).drop(columns=["Date", "Time"])

    for column in NUMERIC_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw_rows = len(raw)
    raw_missing = raw[list(NUMERIC_COLUMNS)].isna()
    missing_by_column_pct = {
        column: float(100.0 * raw_missing[column].mean()) for column in NUMERIC_COLUMNS
    }
    missing_cell_pct = float(100.0 * raw_missing.to_numpy().mean())
    rows_with_missing_pct = float(100.0 * raw_missing.any(axis=1).mean())

    duplicate_timestamps_removed = int(raw["timestamp"].duplicated().sum())
    raw = (
        raw.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="first")
        .set_index("timestamp")
    )
    sorted_timestamps = raw.index.sort_values()
    intervals = sorted_timestamps.to_series().diff().dropna()
    modal_interval_seconds = (
        float(intervals.mode().iloc[0].total_seconds()) if not intervals.empty else np.nan
    )

    interpolated = raw[list(NUMERIC_COLUMNS)].interpolate(
        method="time",
        limit=interpolation_limit,
        limit_direction="both",
    )
    coverage = (
        raw["Global_active_power"].notna().resample("h").sum().astype(int)
    )
    active_energy = (
        interpolated["Global_active_power"].resample("h").sum(min_count=1) / 60.0
    )
    submeter_energies = {
        f"{column.lower()}_energy_kwh": (
            interpolated[column].resample("h").sum(min_count=1) / 1000.0
        )
        for column in ("Sub_metering_1", "Sub_metering_2", "Sub_metering_3")
    }

    hourly = pd.DataFrame(
        {
            "dataset_label": "REAL",
            "global_active_energy_kwh": active_energy,
            **submeter_energies,
            "global_active_power_mean_kw": interpolated["Global_active_power"]
            .resample("h")
            .mean(),
            "global_reactive_power_mean_kw": interpolated["Global_reactive_power"]
            .resample("h")
            .mean(),
            "voltage_mean_v": interpolated["Voltage"].resample("h").mean(),
            "global_intensity_mean_a": interpolated["Global_intensity"]
            .resample("h")
            .mean(),
            "observed_minutes": coverage,
        }
    )
    hourly.index.name = "timestamp"
    hourly.loc[coverage < min_observations_per_hour, list(ENERGY_OUTPUT_COLUMNS)] = np.nan
    cleaned_energy_missing_pct = float(
        100.0 * hourly[list(ENERGY_OUTPUT_COLUMNS)].isna().to_numpy().mean()
    )

    stats = HouseholdPowerStats(
        source_path=str(raw_path),
        sha256=sha256_file(raw_path),
        raw_rows=int(raw_rows),
        observed_start=_stringify_timestamp(sorted_timestamps.min()),
        observed_end=_stringify_timestamp(sorted_timestamps.max()),
        modal_interval_seconds=modal_interval_seconds,
        unique_timestamps=int(len(raw)),
        duplicate_timestamps_removed=duplicate_timestamps_removed,
        missing_cell_pct=missing_cell_pct,
        rows_with_missing_pct=rows_with_missing_pct,
        missing_by_column_pct=missing_by_column_pct,
        cleaned_start=_stringify_timestamp(hourly.index.min()),
        cleaned_end=_stringify_timestamp(hourly.index.max()),
        cleaned_rows=int(len(hourly)),
        cleaned_energy_missing_pct=cleaned_energy_missing_pct,
        min_observations_per_hour=int(min_observations_per_hour),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(output_path, index_label="timestamp")
    if stats_path is not None:
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")

    return HouseholdPowerResult(hourly=hourly, stats=stats)
