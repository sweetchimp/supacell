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
from supacell_ml.simulation import (  # noqa: E402
    GRID_PLUS_SOLAR_PLUS_BATTERY,
    SimulationConfig,
    generate_household_timeseries,
)

OUTPUT_DIR = REPO_ROOT / "results" / "solar_degradation"
DOCS_PATH = REPO_ROOT / "docs" / "solar_degradation.md"
PROFILE = GRID_PLUS_SOLAR_PLUS_BATTERY
TARGET_COLUMN = "grid_import_kw"
SIM_START = "2009-01-01"
SIM_HOURS = 24 * 365
SIM_SEED = 20250101
MODEL_NAME = XGBOOST


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


def simulate_frame():
    config = SimulationConfig(
        start=SIM_START,
        hours=SIM_HOURS,
        seed=SIM_SEED,
    )
    return generate_household_timeseries(
        PROFILE,
        start=SIM_START,
        hours=SIM_HOURS,
        seed=SIM_SEED,
        config=config,
    )


def build_variants(frame):
    target = frame[TARGET_COLUMN].astype(float)
    solar = frame["solar_kw"].astype(float)
    no_solar_features, no_solar_target = build_day_ahead_features(target)
    with_solar_features, with_solar_target = build_day_ahead_features(
        target, exogenous=pd.DataFrame({"solar_kw": solar})
    )
    common = no_solar_features.index.intersection(with_solar_features.index)
    return {
        "no_solar": (
            no_solar_features.loc[common],
            no_solar_target.loc[common],
        ),
        "with_solar": (
            with_solar_features.loc[common],
            with_solar_target.loc[common],
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
    solar_importance = float(importances.get("exog_solar_kw", 0.0))
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
        "solar_feature_importance": solar_importance,
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
    importance_row = {
        "variant": name,
        "top_feature": str(importances.idxmax()),
        "top_feature_importance": float(importances.max()),
        "solar_feature_importance": solar_importance,
    }
    return row, predictions, importance_row


def build_docs(results, importances, metadata):
    rows = dataframe_to_markdown(results)
    importance_rows = dataframe_to_markdown(importances)
    no_solar = results.iloc[0]
    with_solar = results.iloc[1]
    mae_delta = 100.0 * (with_solar["mae_kw"] - no_solar["mae_kw"]) / no_solar["mae_kw"]
    solar_importance = with_solar["solar_feature_importance"]
    if mae_delta < 0:
        interpretation = (
            f"Adding solar reduced MAE by {abs(mae_delta):.6f}%; solar held "
            f"{solar_importance:.1%} of XGBoost importance mass."
        )
    elif mae_delta > 0:
        interpretation = (
            f"Adding solar increased MAE by {mae_delta:.6f}%; solar held "
            f"{solar_importance:.1%} of XGBoost importance mass."
        )
    else:
        interpretation = (
            f"Adding solar left MAE unchanged; solar held "
            f"{solar_importance:.1%} of XGBoost importance mass."
        )
    return f"""# Synthetic solar removal test (`synthetic`)

## Frozen settings

These settings were defined before the experiment was run and were not changed after
seeing results.

- Profile: `{PROFILE}`.
- Target: `{TARGET_COLUMN}`, because simulated household demand is independent of solar
  by construction while grid import is not.
- Seed: `{SIM_SEED}`; start: `{SIM_START}`; hours: `{SIM_HOURS}`.
- Model: `{MODEL_NAME}` with the shared forecasting hyper-parameters.
- Split fractions: `{metadata['split_fractions']}`; alpha: `{metadata['alpha']}`.
- Solar is assumed available at the target hour as a perfect synthetic input.

## Results

{rows}

## Feature importance

{importance_rows}

{interpretation}

## Limitations

- This is a synthetic simulation, not measured household or PV data.
- Perfect solar availability is unrealistic for operational day-ahead forecasting.
- Grid import depends on solar and battery dispatch, so this tests solar value for net-grid prediction rather than household demand prediction.
- Conformal intervals are re-calibrated separately for each variant.
- One profile, one seed and one model configuration were used.

All values above are generated by `ml/experiments/run_solar_degradation.py` from
`results/solar_degradation/solar_results_SYNTHETIC.csv` and
`results/solar_degradation/solar_metadata_SYNTHETIC.json`.
"""


def plot_results(results, output_path):
    positions = np.arange(len(results["variant"]))
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(positions - 0.2, results["mae_kw"], width=0.4, label="MAE")
    axis.bar(positions + 0.2, results["rmse_kw"], width=0.4, label="RMSE")
    axis.set_xticks(positions)
    axis.set_xticklabels(results["variant"])
    axis.set_ylabel("kW")
    axis.set_title("Synthetic solar removal: forecast error - SYNTHETIC")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = simulate_frame()
    if not (frame["solar_kw"] > 0).any():
        raise ValueError("simulated profile contains no solar generation")

    config = ForecastConfig(
        target_column=TARGET_COLUMN,
        label="synthetic",
        random_state=SIM_SEED,
    )
    variants = build_variants(frame)
    metric_rows = []
    prediction_frames = []
    importance_rows = []
    for name in ("no_solar", "with_solar"):
        features, target = variants[name]
        row, predictions, importance = run_variant(name, features, target, config)
        metric_rows.append(row)
        prediction_frames.append(predictions)
        importance_rows.append(importance)

    results = pd.DataFrame(metric_rows)
    no_solar = results.iloc[0]
    results["mae_change_pct"] = (
        100.0 * (results["mae_kw"] - no_solar["mae_kw"]) / no_solar["mae_kw"]
    )
    results["width_change_pct"] = (
        100.0
        * (results["avg_interval_width_kw"] - no_solar["avg_interval_width_kw"])
        / no_solar["avg_interval_width_kw"]
    )
    results["coverage_delta_pp"] = (
        100.0 * (results["empirical_coverage"] - no_solar["empirical_coverage"])
    )
    importances = pd.DataFrame(importance_rows)
    metadata = {
        "dataset_label": "synthetic",
        "profile": PROFILE,
        "target_column": TARGET_COLUMN,
        "seed": SIM_SEED,
        "start": SIM_START,
        "hours": SIM_HOURS,
        "model": MODEL_NAME,
        "split_fractions": {
            "train": config.train_fraction,
            "validation": config.validation_fraction,
        },
        "alpha": config.alpha,
        "settings_frozen_before_results": True,
        "solar_availability": "perfect target-hour synthetic solar input",
    }

    results_path = OUTPUT_DIR / "solar_results_SYNTHETIC.csv"
    predictions_path = OUTPUT_DIR / "solar_predictions_SYNTHETIC.csv"
    importance_path = OUTPUT_DIR / "solar_feature_importances_SYNTHETIC.csv"
    metadata_path = OUTPUT_DIR / "solar_metadata_SYNTHETIC.json"
    results.to_csv(results_path, index=False)
    pd.concat(prediction_frames).to_csv(predictions_path, index_label="timestamp")
    importances.to_csv(importance_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=json_default), encoding="utf-8"
    )
    plot_results(results, OUTPUT_DIR / "solar_error_comparison_SYNTHETIC.png")
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
