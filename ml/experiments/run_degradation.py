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

from supacell_ml.degradation import run_degradation_experiment  # noqa: E402

CLEANED_PATH = REPO_ROOT / "data" / "processed" / "household_power_hourly_REAL.csv"
SOURCE_METADATA = REPO_ROOT / "results" / "forecasting" / "forecast_metadata_REAL.json"
OUTPUT_DIR = REPO_ROOT / "results" / "degradation"
DOCS_PATH = REPO_ROOT / "docs" / "graceful_degradation.md"


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


def dataframe_to_markdown(frame, columns=None):
    source = frame if columns is None else frame[columns]
    lines = ["| " + " | ".join(source.columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(source.columns)) + "|")
    for _, row in source.iterrows():
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def plot_errors(metrics, output_path):
    labels = metrics["level"]
    x_positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(13, 6))
    axis.bar(x_positions - 0.2, metrics["mae_kwh"], width=0.4, label="MAE")
    axis.bar(x_positions + 0.2, metrics["rmse_kwh"], width=0.4, label="RMSE")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_ylabel("kWh")
    axis.set_title("Forecast error under progressive data degradation - REAL")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_intervals(metrics, output_path):
    labels = metrics["level"]
    x_positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(13, 6))
    width_line = axis.plot(
        x_positions,
        metrics["avg_interval_width_kwh"],
        marker="o",
        color="#1f77b4",
        label="Mean interval width (kWh)",
    )
    axis.set_ylabel("Mean interval width (kWh)", color="#1f77b4")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.grid(axis="y", alpha=0.3)

    coverage_axis = axis.twinx()
    coverage_line = coverage_axis.plot(
        x_positions,
        metrics["empirical_coverage"],
        marker="s",
        color="#2ca02c",
        label="Empirical coverage",
    )
    coverage_axis.axhline(
        metrics["nominal_coverage"].iloc[0],
        color="red",
        linestyle="--",
        label="Nominal 90%",
    )
    coverage_axis.set_ylabel("Coverage")
    coverage_axis.set_ylim(0.0, 1.05)

    lines = width_line + coverage_line
    axis.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=8)
    axis.set_title("Interval width and coverage under data degradation - REAL")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_docs(result):
    metrics = result.metrics
    split = result.metadata["split"]
    messages = "\n".join(
        f"- `{row['level']}`: {row['user_message']}"
        for _, row in metrics.iterrows()
    )
    notable = []
    worst = metrics.loc[metrics["mae_kwh"].idxmax()]
    best = metrics.loc[metrics["mae_kwh"].idxmin()]
    notable.append(
        f"Lowest MAE occurred at `{best['level']}` ({best['mae_kwh']:.6f} kWh); "
        f"highest MAE occurred at `{worst['level']}` ({worst['mae_kwh']:.6f} kWh)."
    )
    for _, row in metrics.iterrows():
        if row["level"] in ("no_solar_data", "no_weather_features"):
            notable.append(
                f"`{row['level']}` produced MAE change {row['mae_change_pct']:+.6f}% "
                "because the selected model never consumed that input."
            )
    missing_levels = metrics[metrics["level"].str.startswith("missing_")]
    if len(missing_levels) > 0:
        notable.append(
            "Random missing-hour levels changed MAE by at most "
            f"{missing_levels['mae_change_pct'].abs().max():.6f}% versus full data."
        )
    coverage_spread = float(metrics["empirical_coverage"].max() - metrics["empirical_coverage"].min())
    notable.append(
        f"Empirical coverage ranged from {metrics['empirical_coverage'].min():.6f} "
        f"to {metrics['empirical_coverage'].max():.6f} (spread {coverage_spread:.6f})."
    )

    importance_frame = result.feature_importances
    worst_missing = missing_levels.loc[missing_levels["mae_change_pct"].abs().idxmax()]
    worst_importance = importance_frame.loc[
        importance_frame["level"] == worst_missing["level"]
    ].iloc[0]
    if abs(worst_missing["mae_change_pct"]) < 1.0:
        if worst_importance["lag_feature_importance_share"] < 0.5:
            importance_explanation = (
                f"The largest missing-hour MAE change was {worst_missing['mae_change_pct']:+.6f}% "
                f"at `{worst_missing['level']}`. In that model, corrupted lag/rolling columns held "
                f"{worst_importance['lag_feature_importance_share']:.1%} of XGBoost importance mass, "
                f"while calendar columns held {worst_importance['calendar_feature_importance_share']:.1%}. "
                "The small MAE movement is consistent with the model relying mainly on calendar features."
            )
        else:
            importance_explanation = (
                f"The largest missing-hour MAE change was {worst_missing['mae_change_pct']:+.6f}% "
                f"at `{worst_missing['level']}`, even though corrupted lag/rolling columns held "
                f"{worst_importance['lag_feature_importance_share']:.1%} of XGBoost importance mass. "
                "Low importance is not a sufficient explanation; this result is flagged as unexpected."
            )
    else:
        importance_explanation = (
            f"The largest missing-hour MAE change was {worst_missing['mae_change_pct']:+.6f}% "
            f"at `{worst_missing['level']}`; corrupted lag/rolling columns held "
            f"{worst_importance['lag_feature_importance_share']:.1%} of XGBoost importance mass."
        )

    return f"""# Graceful degradation experiment (`real`)

## What was measured

The degradation experiment used the previously selected best forecasting model,
`{result.metadata['model']}`, on the cleaned UCI household data. Chronological splits
were kept fixed: train {split['train']['start']} to {split['train']['end']},
validation {split['validation']['start']} to {split['validation']['end']}, test
{split['test']['start']} to {split['test']['end']}. No rows were shuffled.

For each level, the model was refit on degraded features, conformal intervals were
re-calibrated on that level's validation residuals, and the same clean test targets
were used for scoring. Missing-hour levels set seeded lag/rolling feature cells to NaN
and then filled them by forward fill from the previous feature row; the first affected
row used the training-split median. Scored timestamps were preserved. Short-history
levels truncated the input history before feature construction. No evaluation target
was corrupted.

`no_solar_data` and `no_weather_features` are honest no-ops in this experiment: the
selected model's features are calendar and lagged energy only, so those inputs were
never available to it.

## Results

{dataframe_to_markdown(metrics)}

## What the system should say

{messages}

## What the numbers show

{chr(10).join(f'- {item}' for item in notable)}

## Feature importance evidence

Importance is XGBoost `feature_importances_`, saved for every fitted level. The
`lag_feature_importance_share` column is the importance mass on the same lag/rolling
columns that missing-hour degradation corrupts.

{dataframe_to_markdown(importance_frame)}

{importance_explanation}

## Limitations

- Missing-hour and short-history levels are synthetic corruptions applied to real data, not naturally missing periods.
- Forward fill can smooth over the exact outage pattern; it tests degraded feature quality, not missing-meter behaviour.
- The no-solar/no-weather levels test nothing about model reliance because those columns are absent from the feature set in this experiment.
- Re-calibrating conformal intervals at every level measures adaptive behaviour, not whether a fixed interval would remain valid.
- Results are for one household, one chronological test window and one model configuration.

All figures above are generated by `ml/experiments/run_degradation.py` from
`results/degradation/degradation_results_REAL.csv` and
`results/degradation/degradation_metadata_REAL.json`.
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_degradation_experiment(CLEANED_PATH, SOURCE_METADATA)
    results_path = OUTPUT_DIR / "degradation_results_REAL.csv"
    predictions_path = OUTPUT_DIR / "degradation_predictions_REAL.csv"
    importance_path = OUTPUT_DIR / "degradation_feature_importances_REAL.csv"
    metadata_path = OUTPUT_DIR / "degradation_metadata_REAL.json"
    result.metrics.to_csv(results_path, index=False)
    result.predictions.to_csv(predictions_path, index_label="timestamp")
    result.feature_importances.to_csv(importance_path, index=False)
    metadata_path.write_text(
        json.dumps(result.metadata, indent=2, default=json_default), encoding="utf-8"
    )

    plot_errors(result.metrics, OUTPUT_DIR / "degradation_mae_rmse_REAL.png")
    plot_intervals(
        result.metrics, OUTPUT_DIR / "degradation_interval_width_coverage_REAL.png"
    )
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text(build_docs(result), encoding="utf-8")

    print(f"results={results_path}")
    print(f"predictions={predictions_path}")
    print(f"importances={importance_path}")
    print(f"metadata={metadata_path}")
    print(f"docs={DOCS_PATH}")
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
