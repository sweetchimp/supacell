import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from supacell_ml.loaders import load_household_power
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
SOURCE_PAGE = (
    "https://archive.ics.uci.edu/dataset/235/"
    "individual+household+electric+power+consumption"
)
RAW_ZIP = REPO_ROOT / "data" / "raw" / "individual_household_electric_power_consumption.zip"
RAW_TXT = REPO_ROOT / "data" / "raw" / "household_power_consumption_REAL.txt"
CLEANED_CSV = REPO_ROOT / "data" / "processed" / "household_power_hourly_REAL.csv"
STATS_JSON = REPO_ROOT / "data" / "processed" / "household_power_hourly_REAL.stats.json"
DOCS_PATH = REPO_ROOT / "docs" / "datasets.md"


def download_raw_file():
    if RAW_TXT.exists():
        return RAW_TXT
    RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_ZIP.exists():
        urllib.request.urlretrieve(DATA_URL, RAW_ZIP)
    with zipfile.ZipFile(RAW_ZIP) as archive:
        candidates = [
            name for name in archive.namelist() if name.lower().endswith(".txt")
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected one text file in archive, found {candidates}")
        with archive.open(candidates[0]) as source, RAW_TXT.open("wb") as target:
            shutil.copyfileobj(source, target)
    return RAW_TXT


def percent(value):
    return f"{value:.4f}%"


def build_docs(stats):
    per_column = "\n".join(
        f"| `{column}` | {percent(value)} |"
        for column, value in stats["missing_by_column_pct"].items()
    )
    interval_minutes = stats["modal_interval_seconds"] / 60.0
    return f"""# Datasets

## Individual Household Electric Power Consumption (`real`)

| Field | Value |
| --- | --- |
| Source | UCI Machine Learning Repository, dataset 235 |
| Source page | {SOURCE_PAGE} |
| Download URL | {DATA_URL} |
| DOI | [10.24432/C58K54](https://doi.org/10.24432/C58K54) |
| Creators | Georges Hebrail and Alice Berard |
| Raw file | `data/raw/household_power_consumption_REAL.txt` |
| Raw SHA256 | `{stats['sha256']}` |
| Collection period, computed from file | {stats['observed_start']} to {stats['observed_end']} |
| Measurement frequency, computed from file | modal interval {interval_minutes:g} minute(s) |
| Raw rows | {stats['raw_rows']:,} |
| Missing-data percentage, computed from file | {percent(stats['missing_cell_pct'])} of measurement cells |
| Rows with at least one missing measurement, computed from file | {percent(stats['rows_with_missing_pct'])} |
| License | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| Cleaned file | `data/processed/household_power_hourly_REAL.csv` |
| Cleaned rows | {stats['cleaned_rows']:,} |
| Cleaned date range | {stats['cleaned_start']} to {stats['cleaned_end']} |
| Cleaned hourly-energy missing percentage | {percent(stats['cleaned_energy_missing_pct'])} |

Missing values in the raw file are represented by `?` or empty fields. The loader
coerces both to `NaN`, interpolates gaps of at most five minutes, and masks hourly
energy values for hours with fewer than {stats['min_observations_per_hour']} observed
minutes. Global active power is a minute-averaged power in kW, so hourly energy is
computed as `sum(power_kw) / 60` kWh. Sub-meter readings are watt-hours per minute, so
hourly sub-meter energy is computed as `sum(watt_hours) / 1000` kWh.

Missing measurement cells by raw column, computed from the downloaded file:

| Column | Missing |
| --- | ---: |
{per_column}

All figures in this document are computed by `ml/experiments/prepare_real_data.py`
from the downloaded file; no published summary statistic is used as a computed value.
"""


def main():
    raw_path = download_raw_file()
    result = load_household_power(
        raw_path,
        output_path=CLEANED_CSV,
        stats_path=STATS_JSON,
    )
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text(build_docs(stats), encoding="utf-8")
    print(f"raw_file={raw_path}")
    print(f"cleaned_file={CLEANED_CSV}")
    print(f"docs_file={DOCS_PATH}")
    print(f"raw_rows={result.stats.raw_rows}")
    print(f"cleaned_rows={result.stats.cleaned_rows}")
    print(f"date_range={result.stats.cleaned_start} to {result.stats.cleaned_end}")
    print(f"missing_cell_pct={result.stats.missing_cell_pct:.6f}")
    print(f"rows_with_missing_pct={result.stats.rows_with_missing_pct:.6f}")
    print(f"cleaned_energy_missing_pct={result.stats.cleaned_energy_missing_pct:.6f}")


if __name__ == "__main__":
    main()
