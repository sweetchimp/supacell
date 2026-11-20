import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from supacell_ml.forecasting import (  # noqa: E402
    ForecastConfig,
    XGBOOST,
    build_day_ahead_features,
    build_models,
    chronological_split,
    conformal_radius,
    regression_metrics,
)
from supacell_ml.weather import (  # noqa: E402
    ARCHIVE_BASE_URL,
    UCI_LOCATION,
    WEATHER_VARIABLES,
    build_weather_features,
)

JOINED_PATH = REPO_ROOT / "data" / "processed" / "household_power_hourly_weather_REAL.csv"
WEATHER_METADATA_PATH = REPO_ROOT / "data" / "processed" / "weather_join_metadata_REAL.json"
OUTPUT_DIR = REPO_ROOT / "results" / "weather_forecast"
DOCS_PATH = REPO_ROOT / "docs" / "real_weather_forecast.md"
TARGET_COLUMN = "global_active_energy_kwh"
MODEL_NAME = XGBOOST
WEATHER_COLUMNS = tuple(f"weather_{variable}" for variable in WEATHER_VARIABLES)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def format_value(value):
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return "N/A"
        return f"{value:.6f}"
    return str(value)


def dataframe_to_markdown(frame):
    lines = ["| " + " | ".join(frame.columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(frame.columns)) + "|")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def load_joined_data():
    frame = pd.read_csv(JOINED_PATH, parse_dates=["timestamp"])
    if "dataset_label" not in frame.columns:
        raise ValueError("joined file must contain dataset_label")
    labels = set(frame["dataset_label"].astype(str).str.lower().unique())
    if labels != {"real"}:
        raise ValueError(f"expected dataset label REAL, found {labels}")
    missing = [column for column in WEATHER_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"joined file is missing weather columns: {missing}")
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    energy = frame[TARGET_COLUMN].astype(float)
    weather = frame[list(WEATHER_COLUMNS)].astype(float)
    return frame, energy, weather


def build_variants(energy, weather):
    no_weather_features, no_weather_target = build_day_ahead_features(energy)
    origin_weather = build_weather_features(weather, no_weather_features.index)
    with_weather_features, with_weather_target = build_day_ahead_features(
        energy, exogenous=origin_weather
    )
    common = no_weather_features.index.intersection(with_weather_features.index)
    return {
        "no_weather": (no_weather_features.loc[common], no_weather_target.loc[common]),
        "with_weather": (
            with_weather_features.loc[common],
            with_weather_target.loc[common],
        ),
    }


def run_variant(name, features, target, config):
    split = chronological_split(features.index, config)
    model = build_models(config)[MODEL_NAME]
    model.fit(features.loc[split.train], target.loc[split.train])
    validation_prediction = np.asarray(
        model.predict(features.loc[split.validation]), dtype=float
    )
    test_prediction = np.asarray(model.predict(features.loc[split.test]), dtype=float)
    actual = target.loc[split.test].to_numpy(dtype=float)
    radius = conformal_radius(
        target.loc[split.validation].to_numpy(dtype=float),
        validation_prediction,
        config.alpha,
    )
    lower = test_prediction - radius
    upper = test_prediction + radius
    metrics = regression_metrics(actual, test_prediction)
    importances = pd.Series(model.feature_importances_, index=features.columns)
    weather_mask = importances.index.str.startswith("exog_")
    weather_importance = float(importances.loc[weather_mask].sum())
    row = {
        "variant": name,
        "feature_count": int(features.shape[1]),
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
        "mae_kw": metrics["mae"],
        "rmse_kw": metrics["rmse"],
        "mape_pct": metrics["mape"],
        "nominal_coverage": float(1.0 - config.alpha),
        "empirical_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "avg_interval_width_kw": float(np.mean(upper - lower)),
        "conformal_radius_kw": radius,
        "weather_feature_importance": weather_importance,
    }
    predictions = pd.DataFrame(
        {
            "variant": name,
            "actual_kw": actual,
            "predicted_kw": test_prediction,
            "lower_kw": lower,
            "upper_kw": upper,
        },
        index=split.test,
    )
    importance_rows = pd.DataFrame(
        {"variant": name, "feature": features.columns, "importance": importances.to_numpy()}
    )
    return row, predictions, importance_rows


def build_docs(results, importances, metadata):
    rows = dataframe_to_markdown(results)
    no_weather = results.iloc[0]
    with_weather = results.iloc[1]
    mae_delta = 100.0 * (with_weather["mae_kw"] - no_weather["mae_kw"]) / no_weather["mae_kw"]
    width_delta = (
        100.0
        * (with_weather["avg_interval_width_kw"] - no_weather["avg_interval_width_kw"])
        / no_weather["avg_interval_width_kw"]
    )
    coverage_delta = 100.0 * (
        with_weather["empirical_coverage"] - no_weather["empirical_coverage"]
    )
    weather_importance = with_weather["weather_feature_importance"]
    direction = "reduced" if mae_delta < 0 else "increased" if mae_delta > 0 else "left unchanged"
    top_weather = (
        importances.loc[importances["feature"].str.startswith("exog_")]
        .sort_values("importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_rows = dataframe_to_markdown(top_weather)
    return f"""# Real weather removal test (`real`)

## Frozen settings

These settings were defined before the experiment was run and were not changed after
seeing results.

- Data: `data/processed/household_power_hourly_weather_REAL.csv`.
- Weather source: Open-Meteo Historical Weather API, ERA5, Sceaux/Paris location
  `{UCI_LOCATION['latitude']}, {UCI_LOCATION['longitude']}`, timezone
  `{UCI_LOCATION['timezone']}`; endpoint `{ARCHIVE_BASE_URL}`.
- Weather features: lagged values at 0, 3, 6, 12 and 24 hours before the forecast origin
  plus 24-hour rolling mean, standard deviation and precipitation sum ending at the
  forecast origin.
- Forecast origin: 23:00 on the day before the target day; target-hour weather is never
  used.
- Model: `{MODEL_NAME}` with the shared forecasting hyper-parameters.
- Split fractions: `{metadata['split_fractions']}`; alpha: `{metadata['alpha']}`.
- Variants: `no_weather` and `with_weather`; conformal intervals are recalibrated
  separately for each variant.

Source, licence and citation details are in `docs/weather_data.md`.

## Results

{rows}

## Weather feature importance

{top_rows}

Adding origin-available real weather {direction} MAE by {abs(mae_delta):.6f}%; interval
width changed by {width_delta:+.6f}%; empirical coverage changed by
{coverage_delta:+.6f} percentage points. All weather features together held
{weather_importance:.2%} of XGBoost importance mass in the `with_weather` variant.
Negative deltas mean the `with_weather` variant was lower than `no_weather`; zero deltas
mean no change.

## Limitations

- This is one household, one location, one reanalysis product, one model and one chronologically held-out split.
- ERA5 grid-cell reanalysis is not a measurement at the house; a local station or on-site sensor is needed to claim household-specific weather value.
- Weather is available only up to 23:00 the day before; operational use would require a day-ahead weather forecast rather than reanalysis.
- The energy target contains no solar column, so any weather effect is behavioral or indirect through heating, cooling and lighting demand.
- Conformal intervals are recalibrated separately for each variant; width and coverage changes are not guaranteed by the nominal level.
- This test reports association and predictive value on one test period, not causality.

All values above are generated by `ml/experiments/run_weather_forecast.py` from
`results/weather_forecast/weather_results_REAL.csv` and
`results/weather_forecast/weather_metadata_REAL.json`.
"""


def plot_results(results, output_path):
    positions = np.arange(len(results["variant"]))
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(positions - 0.2, results["mae_kw"], width=0.4, label="MAE")
    axis.bar(positions + 0.2, results["rmse_kw"], width=0.4, label="RMSE")
    axis.set_xticks(positions)
    axis.set_xticklabels(results["variant"])
    axis.set_ylabel("kW")
    axis.set_title("Real weather removal: forecast error - REAL")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, energy, weather = load_joined_data()
    weather_metadata = json.loads(WEATHER_METADATA_PATH.read_text(encoding="utf-8"))
    config = ForecastConfig(
        target_column=TARGET_COLUMN,
        label="real",
        random_state=42,
    )
    variants = build_variants(energy, weather)
    metric_rows = []
    prediction_frames = []
    importance_frames = []
    for name in ("no_weather", "with_weather"):
        features, target = variants[name]
        row, predictions, importances = run_variant(name, features, target, config)
        metric_rows.append(row)
        prediction_frames.append(predictions)
        importance_frames.append(importances)

    results = pd.DataFrame(metric_rows)
    baseline = results.iloc[0]
    results["mae_change_pct"] = (
        100.0 * (results["mae_kw"] - baseline["mae_kw"]) / baseline["mae_kw"]
    )
    results["rmse_change_pct"] = (
        100.0 * (results["rmse_kw"] - baseline["rmse_kw"]) / baseline["rmse_kw"]
    )
    results["width_change_pct"] = (
        100.0
        * (results["avg_interval_width_kw"] - baseline["avg_interval_width_kw"])
        / baseline["avg_interval_width_kw"]
    )
    results["coverage_delta_pp"] = 100.0 * (
        results["empirical_coverage"] - baseline["empirical_coverage"]
    )
    importances = pd.concat(importance_frames, ignore_index=True)
    metadata = {
        "dataset_label": "real",
        "joined_file": str(JOINED_PATH),
        "target_column": TARGET_COLUMN,
        "model": MODEL_NAME,
        "split_fractions": {
            "train": config.train_fraction,
            "validation": config.validation_fraction,
        },
        "alpha": config.alpha,
        "random_state": config.random_state,
        "weather_source": "Open-Meteo Historical Weather API",
        "weather_model": weather_metadata["model"],
        "weather_request_url": weather_metadata["request_url"],
        "weather_raw_sha256": weather_metadata["raw_sha256"],
        "weather_license": weather_metadata["license"],
        "weather_citation": weather_metadata["citation"],
        "weather_feature_policy": "origin-available prior-day weather only; no target-hour weather",
        "settings_frozen_before_results": True,
    }

    results_path = OUTPUT_DIR / "weather_results_REAL.csv"
    predictions_path = OUTPUT_DIR / "weather_predictions_REAL.csv"
    importance_path = OUTPUT_DIR / "weather_feature_importances_REAL.csv"
    metadata_path = OUTPUT_DIR / "weather_metadata_REAL.json"
    results.to_csv(results_path, index=False)
    pd.concat(prediction_frames).to_csv(predictions_path, index_label="timestamp")
    importances.to_csv(importance_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=json_default), encoding="utf-8"
    )
    plot_results(results, OUTPUT_DIR / "weather_error_comparison_REAL.png")
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text(
        build_docs(results, importances, metadata), encoding="utf-8"
    )

    print(f"results={results_path}")
    print(f"predictions={predictions_path}")
    print(f"importances={importance_path}")
    print(f"metadata={metadata_path}")
    print(f"docs={DOCS_PATH}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
