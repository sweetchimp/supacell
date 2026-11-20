# Datasets

## Individual Household Electric Power Consumption (`real`)

| Field | Value |
| --- | --- |
| Source | UCI Machine Learning Repository, dataset 235 |
| Source page | https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption |
| Download URL | https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip |
| DOI | [10.24432/C58K54](https://doi.org/10.24432/C58K54) |
| Creators | Georges Hebrail and Alice Berard |
| Raw file | `data/raw/household_power_consumption_REAL.txt` |
| Raw SHA256 | `4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7` |
| Collection period, computed from file | 2006-12-16T17:24:00 to 2010-11-26T21:02:00 |
| Measurement frequency, computed from file | modal interval 1 minute(s) |
| Raw rows | 2,075,259 |
| Missing-data percentage, computed from file | 1.2518% of measurement cells |
| Rows with at least one missing measurement, computed from file | 1.2518% |
| License | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| Cleaned file | `data/processed/household_power_hourly_REAL.csv` |
| Cleaned rows | 34,589 |
| Cleaned date range | 2006-12-16T17:00:00 to 2010-11-26T21:00:00 |
| Cleaned hourly-energy missing percentage | 1.2779% |

Missing values in the raw file are represented by `?` or empty fields. The loader
coerces both to `NaN`, interpolates gaps of at most five minutes, and masks hourly
energy values for hours with fewer than 50 observed
minutes. Global active power is a minute-averaged power in kW, so hourly energy is
computed as `sum(power_kw) / 60` kWh. Sub-meter readings are watt-hours per minute, so
hourly sub-meter energy is computed as `sum(watt_hours) / 1000` kWh.

Missing measurement cells by raw column, computed from the downloaded file:

| Column | Missing |
| --- | ---: |
| `Global_active_power` | 1.2518% |
| `Global_reactive_power` | 1.2518% |
| `Voltage` | 1.2518% |
| `Global_intensity` | 1.2518% |
| `Sub_metering_1` | 1.2518% |
| `Sub_metering_2` | 1.2518% |
| `Sub_metering_3` | 1.2518% |

All figures in this document are computed by `ml/experiments/prepare_real_data.py`
from the downloaded file; no published summary statistic is used as a computed value.
