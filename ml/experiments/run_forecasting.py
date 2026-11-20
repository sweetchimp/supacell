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
    HISTORICAL_AVERAGE,
    LINEAR_REGRESSION,
    MODEL_NAMES,
    ForecastConfig,
    run_forecast_comparison,
)

CLEANED_PATH = REPO_ROOT / "data" / "processed" / "household_power_hourly_REAL.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "forecasting"
DOCS_PATH = REPO_ROOT / "docs" / "forecasting.md"
PLOT_WINDOW_HOURS = 336
CONFIG = ForecastConfig()


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def format_value(value):
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return "N/A"
        return f"{value:.6f}"
    return str(value)


def dataframe_to_markdown(frame):
    lines = []
    lines.append("| " + " | ".join(frame.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(frame.columns)) + "|")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def interval_rows(metrics):
    return metrics.loc[metrics["interval_model"].astype(bool)].copy()


def plot_model_comparison(metrics, output_path):
    comparison = metrics.set_index("model")[["mae_kwh", "rmse_kwh"]]
    fig, axis = plt.subplots(figsize=(10, 5))
    comparison.plot(kind="bar", ax=axis)
    axis.set_ylabel("kWh")
    axis.set_xlabel("Model")
    axis.set_title("Next-day hourly forecast error by model - REAL")
    axis.grid(axis="y", alpha=0.3)
    axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(predictions, output_path):
    window = predictions.iloc[:PLOT_WINDOW_HOURS]
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(window.index, window["actual_kwh"], color="black", linewidth=2, label="Actual")
    for model in MODEL_NAMES:
        axis.plot(window.index, window[model], linewidth=1, alpha=0.8, label=model)
    axis.set_title(f"First {PLOT_WINDOW_HOURS} test hours: actual vs predicted - REAL")
    axis.set_ylabel("Hourly energy (kWh)")
    axis.grid(alpha=0.3)
    axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(predictions, output_path):
    residuals = pd.DataFrame(
        {
            model: predictions["actual_kwh"] - predictions[model]
            for model in MODEL_NAMES
        }
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    residuals.boxplot(ax=axis)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_title("Test residuals by model - REAL")
    axis.set_ylabel("Actual minus predicted (kWh)")
    axis.grid(axis="y", alpha=0.3)
    axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_intervals(predictions, metrics, metadata, output_path):
    selected = interval_rows(metrics)
    suffix = f"p{int(round((1.0 - metadata['alpha']) * 100))}"
    window = predictions.iloc[:PLOT_WINDOW_HOURS]
    fig, axes = plt.subplots(
        len(selected), 1, figsize=(14, 5 * len(selected)), sharex=True, squeeze=False
    )
    for axis, (_, row) in zip(axes[:, 0], selected.iterrows()):
        model = row["model"]
        axis.fill_between(
            window.index,
            window[f"{model}_lower_{suffix}"],
            window[f"{model}_upper_{suffix}"],
            alpha=0.25,
            label="90% conformal interval",
        )
        axis.plot(window.index, window["actual_kwh"], color="black", linewidth=1.5, label="Actual")
        axis.plot(window.index, window[model], linewidth=1.2, label="Predicted")
        axis.set_title(
            f"{model}: empirical coverage {row['empirical_coverage']:.4f}, "
            f"mean width {row['avg_interval_width_kwh']:.4f} kWh - REAL"
        )
        axis.set_ylabel("Hourly energy (kWh)")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_docs(metrics, metadata):
    split = metadata["split"]
    interval_frame = interval_rows(metrics)
    best_mae = metrics.loc[metrics["mae_kwh"].idxmin()]
    best_rmse = metrics.loc[metrics["rmse_kwh"].idxmin()]
    baseline = metrics.loc[metrics["model"] == HISTORICAL_AVERAGE].iloc[0]
    baseline_gap_pct = 100.0 * (baseline["mae_kwh"] - best_mae["mae_kwh"]) / best_mae["mae_kwh"]
    baseline_competitive = baseline_gap_pct <= 5.0

    coverage_lines = []
    surprises = []
    for _, row in interval_frame.iterrows():
        coverage_lines.append(
            f"- {row['model']}: empirical coverage {row['empirical_coverage']:.6f}, "
            f"average width {row['avg_interval_width_kwh']:.6f} kWh, "
            f"conformal radius {row['conformal_radius_kwh']:.6f} kWh "
            f"(nominal {row['nominal_coverage']:.2f})."
        )
        delta_points = 100.0 * (
            row["empirical_coverage"] - row["nominal_coverage"]
        )
        if abs(delta_points) > 5.0:
            surprises.append(
                f"- {row['model']} coverage differs from nominal by {delta_points:.2f} "
                "percentage points; chronological drift can break conformal exchangeability."
            )

    if baseline_competitive:
        surprises.append(
            f"- The historical-average baseline is competitive: its MAE is only "
            f"{baseline_gap_pct:.2f}% above the best MAE."
        )
    else:
        surprises.append(
            f"- The historical-average baseline is not competitive on test: its MAE is "
            f"{baseline_gap_pct:.2f}% above the best MAE."
        )

    validation_by_model = {row["model"]: row for row in metadata["validation_metrics"]}
    baseline_validation = validation_by_model[HISTORICAL_AVERAGE]
    linear_validation = validation_by_model[LINEAR_REGRESSION]
    if (
        baseline_validation["mae"] <= linear_validation["mae"]
        and baseline_validation["rmse"] <= linear_validation["rmse"]
    ):
        surprises.append(
            "- The simple historical average beat linear regression on both validation "
            "MAE and validation RMSE, so the linear model was not competitive there."
        )

    mape_note = (
        f"MAPE is computed because all {split['test']['rows']} test actuals are positive "
        f"(minimum {metadata['min_test_actual_kwh']:.6f} kWh)."
        if metadata["mape_eligible"]
        else "MAPE is omitted because at least one test actual is non-positive."
    )
    best_two = ", ".join(metadata["best_two_by_validation_rmse"])
    validation_table = dataframe_to_markdown(
        pd.DataFrame(metadata["validation_metrics"])
    )

    return f"""# Forecasting results (`real`)

## What was measured

Four next-day hourly forecasts of `global_active_energy_kwh` were evaluated on the
cleaned UCI household data (`data/processed/household_power_hourly_REAL.csv`).
Each target day's 24 hours were predicted from a single origin at 23:00 on the prior
day. Features used calendar fields plus lagged/rolling target energy only; no
sub-meter, weather or same-day actual values were used.

- Chronological splits, no shuffling: train {split['train']['start']} to {split['train']['end']} ({split['train']['rows']} rows); validation {split['validation']['start']} to {split['validation']['end']} ({split['validation']['rows']} rows); test {split['test']['start']} to {split['test']['end']} ({split['test']['rows']} rows).
- Models: same-hour/same-weekday historical average, linear regression, random forest and XGBoost.
- Best two by validation RMSE, used for prediction intervals: {best_two}.
- Intervals: split conformal on absolute validation residuals at nominal {1.0 - metadata['alpha']:.0%}.
- {mape_note}

## Validation metrics used to select the interval models

{validation_table}

## Test results

{dataframe_to_markdown(metrics)}

## Interval coverage

{chr(10).join(coverage_lines)}

## What the numbers show

- Lowest test MAE: {best_mae['model']} ({best_mae['mae_kwh']:.6f} kWh).
- Lowest test RMSE: {best_rmse['model']} ({best_rmse['rmse_kwh']:.6f} kWh).
{chr(10).join(surprises)}

All figures above are generated from `results/forecasting/forecast_results_REAL.csv`
and `results/forecasting/forecast_metadata_REAL.json` by
`ml/experiments/run_forecasting.py`.
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_forecast_comparison(CLEANED_PATH, CONFIG)

    results_csv = OUTPUT_DIR / "forecast_results_REAL.csv"
    predictions_csv = OUTPUT_DIR / "forecast_predictions_REAL.csv"
    metadata_json = OUTPUT_DIR / "forecast_metadata_REAL.json"
    result.metrics.to_csv(results_csv, index=False)
    result.predictions.to_csv(predictions_csv, index_label="timestamp")
    metadata_json.write_text(
        json.dumps(result.metadata, indent=2, default=json_default), encoding="utf-8"
    )

    plot_model_comparison(result.metrics, OUTPUT_DIR / "forecast_model_comparison_REAL.png")
    plot_actual_vs_predicted(
        result.predictions, OUTPUT_DIR / "forecast_actual_vs_predicted_REAL.png"
    )
    plot_residuals(result.predictions, OUTPUT_DIR / "forecast_residuals_REAL.png")
    plot_intervals(
        result.predictions,
        result.metrics,
        result.metadata,
        OUTPUT_DIR / "forecast_prediction_intervals_REAL.png",
    )

    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text(build_docs(result.metrics, result.metadata), encoding="utf-8")

    print(f"results={results_csv}")
    print(f"predictions={predictions_csv}")
    print(f"metadata={metadata_json}")
    print(f"docs={DOCS_PATH}")
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
