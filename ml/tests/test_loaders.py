from pathlib import Path

import pandas as pd
import pytest

from supacell_ml.loaders import load_household_power

HEADER = (
    "Date;Time;Global_active_power;Global_reactive_power;Voltage;"
    "Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n"
)


def write_raw_file(
    path,
    missing_power=(),
    missing_voltage=(),
    missing_intensity=(),
):
    start = pd.Timestamp("2007-01-01 00:00:00")
    rows = []
    for minute in range(180):
        timestamp = start + pd.Timedelta(minutes=minute)
        power = "?" if minute in missing_power else "1.000"
        voltage = "" if minute in missing_voltage else "230.000"
        intensity = "?" if minute in missing_intensity else "5.000"
        rows.append(
            ";".join(
                [
                    timestamp.strftime("%d/%m/%Y"),
                    timestamp.strftime("%H:%M:%S"),
                    power,
                    "0.100",
                    voltage,
                    intensity,
                    "0.000",
                    "0.000",
                    "0.000",
                ]
            )
        )
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_missing_markers_are_coerced_and_short_gaps_interpolated(tmp_path):
    raw_path = write_raw_file(
        tmp_path / "raw_REAL.txt",
        missing_power=(10,),
        missing_voltage=(11,),
        missing_intensity=(90,),
    )
    result = load_household_power(raw_path)

    assert result.stats.raw_rows == 180
    assert result.stats.missing_cell_pct == pytest.approx(100 * 3 / (180 * 7))
    assert result.stats.rows_with_missing_pct == pytest.approx(100 * 3 / 180)
    assert (result.hourly["dataset_label"] == "REAL").all()
    assert result.hourly["global_active_energy_kwh"].notna().all()
    assert result.hourly["voltage_mean_v"].notna().all()
    assert not result.hourly.astype(str).eq("?").any().any()
    assert result.hourly["global_active_energy_kwh"].iloc[0] == pytest.approx(1.0)


def test_long_gap_is_masked_and_hourly_frequency_is_correct(tmp_path):
    raw_path = write_raw_file(
        tmp_path / "raw_REAL.txt",
        missing_power=tuple(range(60, 120)),
    )
    result = load_household_power(raw_path)
    expected_index = pd.date_range(
        "2007-01-01 00:00:00", periods=3, freq="h", name="timestamp"
    )

    pd.testing.assert_index_equal(result.hourly.index, expected_index)
    assert result.hourly.index.freq == pd.tseries.frequencies.to_offset("h")
    assert result.hourly["global_active_energy_kwh"].iloc[0] == pytest.approx(1.0)
    assert pd.isna(result.hourly["global_active_energy_kwh"].iloc[1])
    assert result.hourly["observed_minutes"].iloc[1] == 0
    assert result.hourly["global_active_energy_kwh"].iloc[2] == pytest.approx(1.0)
