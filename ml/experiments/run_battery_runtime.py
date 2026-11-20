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

from supacell_ml.battery import (  # noqa: E402
    CAPACITY_HEALTH_RANGE,
    EFFICIENCY_RANGE,
    estimate_runtime_range,
    simulate_discharge,
)
from supacell_ml.simulation import (  # noqa: E402
    GRID_ONLY,
    SimulationConfig,
    generate_household_timeseries,
)

OUTPUT_DIR = REPO_ROOT / "results" / "battery_runtime"
DOCS_PATH = REPO_ROOT / "docs" / "battery_runtime.md"
N_CASES = 80
FORECAST_HOURS = 72
SOC_FRACTION = 0.80
USABLE_DOD = 0.80
NOMINAL_LEVEL = 0.80
START = "2025-01-01"
CAPACITY_RANGE_JUSTIFICATION = "80% of nameplate is a conservative end-of-life-style lower bound; 100% covers a healthy asset."
EFFICIENCY_RANGE_JUSTIFICATION = "90%-98% is a broad normal inverter/converter envelope; no vendor efficiency curve is available."

SCENARIOS = {
    "correct": {
        "true_capacity_kwh": 10.0,
        "assumed_capacity_kwh": 10.0,
        "true_efficiency": 0.95,
        "assumed_efficiency": 0.95,
        "band_fraction": 0.25,
    },
    "capacity_degraded_20pct": {
        "true_capacity_kwh": 8.0,
        "assumed_capacity_kwh": 10.0,
        "true_efficiency": 0.95,
        "assumed_efficiency": 0.95,
        "band_fraction": 0.25,
    },
    "efficiency_overestimated": {
        "true_capacity_kwh": 10.0,
        "assumed_capacity_kwh": 10.0,
        "true_efficiency": 0.85,
        "assumed_efficiency": 0.95,
        "band_fraction": 0.25,
    },
    "uncertainty_too_narrow": {
        "true_capacity_kwh": 10.0,
        "assumed_capacity_kwh": 10.0,
        "true_efficiency": 0.95,
        "assumed_efficiency": 0.95,
        "band_fraction": 0.05,
    },
}


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
    lines = ["| " + " | ".join(frame.columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(frame.columns)) + "|")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def build_forecast(realized_kw, band_fraction, seed):
    rng = np.random.default_rng(seed + 100000)
    expected = realized_kw * (1.0 + rng.normal(0.0, 0.05, len(realized_kw)))
    expected = np.clip(expected, 0.05, None)
    return pd.DataFrame(
        {
            "expected_kw": expected,
            "p10_kw": expected * (1.0 - band_fraction),
            "p90_kw": expected * (1.0 + band_fraction),
        }
    )


def run_validation():
    detail_rows = []
    example_cases = {}
    for case_id in range(N_CASES):
        seed = 7000 + case_id
        frame = generate_household_timeseries(
            GRID_ONLY,
            start=START,
            hours=FORECAST_HOURS,
            seed=seed,
            config=SimulationConfig(hours=FORECAST_HOURS, start=START),
        )
        realized_kw = frame["demand_kw"].to_numpy(dtype=float)

        for scenario_name, params in SCENARIOS.items():
            forecast = build_forecast(realized_kw, params["band_fraction"], seed)
            true_capacity = params["true_capacity_kwh"]
            assumed_capacity = params["assumed_capacity_kwh"]
            true_soc = true_capacity * SOC_FRACTION
            assumed_soc = assumed_capacity * SOC_FRACTION
            true_hours = simulate_discharge(
                realized_kw,
                true_capacity,
                true_soc,
                USABLE_DOD,
                params["true_efficiency"],
            )
            load_only = estimate_runtime_range(
                assumed_capacity,
                assumed_soc,
                USABLE_DOD,
                params["assumed_efficiency"],
                forecast,
                nominal_level=NOMINAL_LEVEL,
                capacity_health_range=(1.0, 1.0),
                efficiency_range=(params["assumed_efficiency"], params["assumed_efficiency"]),
            )
            combined = estimate_runtime_range(
                assumed_capacity,
                assumed_soc,
                USABLE_DOD,
                params["assumed_efficiency"],
                forecast,
                nominal_level=NOMINAL_LEVEL,
                capacity_health_range=CAPACITY_HEALTH_RANGE,
                efficiency_range=EFFICIENCY_RANGE,
            )
            detail_rows.append(
                {
                    "case_id": case_id,
                    "scenario": scenario_name,
                    "true_runtime_hours": true_hours,
                    "load_only_low_hours": load_only.low_hours,
                    "load_only_expected_hours": load_only.expected_hours,
                    "load_only_high_hours": load_only.high_hours,
                    "load_only_error_expected_hours": true_hours - load_only.expected_hours,
                    "load_only_inside": load_only.low_hours <= true_hours <= load_only.high_hours,
                    "combined_low_hours": combined.low_hours,
                    "combined_expected_hours": combined.expected_hours,
                    "combined_high_hours": combined.high_hours,
                    "error_low_hours": true_hours - combined.low_hours,
                    "error_expected_hours": true_hours - combined.expected_hours,
                    "error_high_hours": true_hours - combined.high_hours,
                    "inside_range": combined.low_hours <= true_hours <= combined.high_hours,
                }
            )
            if scenario_name in ("correct", "capacity_degraded_20pct"):
                example_cases.setdefault(
                    scenario_name,
                    {
                        "forecast": forecast,
                        "realized_kw": realized_kw,
                        "true_hours": true_hours,
                        "estimate": combined,
                    },
                )

    details = pd.DataFrame(detail_rows)
    details["load_only_width_hours"] = (
        details["load_only_high_hours"] - details["load_only_low_hours"]
    )
    details["combined_width_hours"] = (
        details["combined_high_hours"] - details["combined_low_hours"]
    )
    summary_rows = []
    for scenario_name, group in details.groupby("scenario", sort=False):
        load_only_width = float(group["load_only_width_hours"].mean())
        combined_width = float(group["combined_width_hours"].mean())
        load_only_rate = float(group["load_only_inside"].mean())
        combined_rate = float(group["inside_range"].mean())
        summary_rows.append(
            {
                "scenario": scenario_name,
                "cases": int(len(group)),
                "mean_true_hours": float(group["true_runtime_hours"].mean()),
                "load_only_mean_width_hours": load_only_width,
                "load_only_inside_count": int(group["load_only_inside"].sum()),
                "load_only_inside_rate": load_only_rate,
                "combined_mean_low_hours": float(group["combined_low_hours"].mean()),
                "combined_mean_expected_hours": float(group["combined_expected_hours"].mean()),
                "combined_mean_high_hours": float(group["combined_high_hours"].mean()),
                "combined_mean_error_expected_hours": float(group["error_expected_hours"].mean()),
                "combined_mae_expected_hours": float(group["error_expected_hours"].abs().mean()),
                "combined_mean_width_hours": combined_width,
                "combined_inside_count": int(group["inside_range"].sum()),
                "combined_inside_rate": combined_rate,
                "width_gain_hours": combined_width - load_only_width,
                "coverage_gain_pp": 100.0 * (combined_rate - load_only_rate),
            }
        )
    return details, pd.DataFrame(summary_rows), example_cases


def plot_example(example_cases, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, (scenario, case) in zip(axes, example_cases.items()):
        forecast = case["forecast"]
        hours = np.arange(len(forecast))
        axis.plot(hours, forecast["p10_kw"], linestyle="--", color="#1f77b4", label="p10 load")
        axis.plot(hours, forecast["expected_kw"], color="#1f77b4", label="expected load")
        axis.plot(hours, forecast["p90_kw"], linestyle="--", color="#1f77b4", label="p90 load")
        axis.axvline(case["true_hours"], color="black", linewidth=2, label="true runtime")
        axis.axvline(case["estimate"].low_hours, color="#ff7f0e", linestyle=":")
        axis.axvline(case["estimate"].expected_hours, color="#ff7f0e", linestyle="-")
        axis.axvline(case["estimate"].high_hours, color="#ff7f0e", linestyle=":")
        axis.set_title(f"{scenario}\ntrue={case['true_hours']:.2f}h range=[{case['estimate'].low_hours:.2f}, {case['estimate'].high_hours:.2f}]h")
        axis.set_xlabel("Hour")
        axis.set_ylabel("Load (kW)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Battery runtime estimate vs physics-based discharge - SYNTHETIC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_errors(details, output_path):
    columns = [
        "error_low_hours",
        "error_expected_hours",
        "error_high_hours",
    ]
    plot_frame = details.melt(
        id_vars="scenario",
        value_vars=columns,
        var_name="endpoint",
        value_name="error_hours",
    )
    fig, axis = plt.subplots(figsize=(12, 6))
    plot_frame.boxplot(column="error_hours", by=["scenario", "endpoint"], ax=axis, rot=45)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_title("Runtime endpoint errors (true minus estimate) - SYNTHETIC")
    axis.set_ylabel("Error (hours)")
    axis.set_xlabel("")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_inside_rate(summary, output_path):
    positions = np.arange(len(summary["scenario"]))
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(positions - 0.2, summary["load_only_inside_rate"], width=0.4, label="load only")
    axis.bar(positions + 0.2, summary["combined_inside_rate"], width=0.4, label="load + health + efficiency")
    axis.axhline(NOMINAL_LEVEL, color="red", linestyle="--", label=f"nominal {NOMINAL_LEVEL:.0%}")
    axis.set_xticks(positions)
    axis.set_xticklabels(summary["scenario"], rotation=20)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Share of true runtimes inside range")
    axis.set_title("Empirical runtime range coverage - SYNTHETIC")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_width_tradeoff(summary, output_path):
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(
        summary["combined_mean_width_hours"],
        summary["combined_inside_rate"],
        color="#1f77b4",
        s=80,
    )
    for _, row in summary.iterrows():
        axis.annotate(
            row["scenario"],
            (row["combined_mean_width_hours"], row["combined_inside_rate"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    axis.set_xlabel("Mean interval width (hours)")
    axis.set_ylabel("Inside rate")
    axis.set_ylim(-0.05, 1.05)
    axis.set_title("Coverage vs width trade-off after adding assumption uncertainty - SYNTHETIC")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_docs(summary, assumptions):
    assumption_lines = "\n".join(f"- {item}" for item in assumptions)
    return f"""# Battery runtime range (`synthetic`)

## Frozen assumption settings

These settings were defined before the revised validation was run and were not changed
after seeing results.

- Capacity-health range: {CAPACITY_HEALTH_RANGE[0]:.0%}-{CAPACITY_HEALTH_RANGE[1]:.0%} of supplied capacity.
  Justification: {CAPACITY_RANGE_JUSTIFICATION}
- One-way efficiency range: {EFFICIENCY_RANGE[0]:.0%}-{EFFICIENCY_RANGE[1]:.0%}.
  Justification: {EFFICIENCY_RANGE_JUSTIFICATION}
- Combination rule: low runtime uses p90 load, low capacity health and low efficiency;
  high runtime uses p10 load, high capacity health and high efficiency; expected runtime
  uses point capacity, point efficiency and the expected load path.
- The combined range is a simultaneous worst-case envelope, not a calibrated probability
  interval. The load-only range is retained as a comparison so the coverage/width
  trade-off is explicit.

## What was measured

A runtime estimator took a battery capacity, state of charge, usable depth-of-discharge
limit, one-way conversion efficiency and an hourly load forecast with `expected_kw`,
`p10_kw` and `p90_kw`. It returned load-only and combined low/expected/high runtimes and
a plain list of assumptions. The nominal load band was {NOMINAL_LEVEL:.0%}.

Validation used {N_CASES} seeded synthetic load paths of {FORECAST_HOURS} hours from the
`GRID_ONLY` simulator profile. Each path was discharged step-by-step with the same
efficiency and depth-of-discharge physics but with independent "true" parameters.
`SYNTHETIC` labels are used for all artifacts in this section.

## Results

{dataframe_to_markdown(summary)}

## Coverage/width trade-off

The combined range adds capacity-health and efficiency uncertainty to the existing load
band. Compare `load_only_mean_width_hours` with `combined_mean_width_hours` and
`load_only_inside_rate` with `combined_inside_rate`. `width_gain_hours` and
`coverage_gain_pp` report the measured trade-off for each scenario. Higher coverage from
a wider range is not evidence of a calibrated probability interval.

## Assumptions emitted with every estimate

{assumption_lines}

## Limitations

- The p10/p90 paths are treated as simultaneous load trajectories, not independent hourly quantiles.
- Capacity health scales both supplied capacity and supplied SoC; real battery data may not preserve that coupling.
- The efficiency range is an engineering envelope, not a fitted distribution or vendor efficiency curve.
- Wrong assumptions are deliberate here; they show that a numerically precise range can still be badly calibrated if the battery model or uncertainty band is wrong.
- All validation uses simulated discharge only, so these numbers do not measure real battery behaviour.
- To replace simulated validation with real validation, we would need measured capacity/state-of-health tests, battery voltage-current-temperature and SoC traces, inverter efficiency curves, ageing state, derating or fault events, and multiple devices, chemistries and ages.

All values above are generated by `ml/experiments/run_battery_runtime.py` from
`results/battery_runtime/runtime_validation_SYNTHETIC.csv` and
`results/battery_runtime/runtime_summary_SYNTHETIC.csv`.
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    details, summary, examples = run_validation()
    details_path = OUTPUT_DIR / "runtime_validation_SYNTHETIC.csv"
    summary_path = OUTPUT_DIR / "runtime_summary_SYNTHETIC.csv"
    metadata_path = OUTPUT_DIR / "runtime_metadata_SYNTHETIC.json"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "dataset_label": "synthetic",
        "n_cases": N_CASES,
        "forecast_hours": FORECAST_HOURS,
        "soc_fraction": SOC_FRACTION,
        "usable_dod": USABLE_DOD,
        "nominal_level": NOMINAL_LEVEL,
        "capacity_health_range": CAPACITY_HEALTH_RANGE,
        "capacity_range_justification": CAPACITY_RANGE_JUSTIFICATION,
        "efficiency_range": EFFICIENCY_RANGE,
        "efficiency_range_justification": EFFICIENCY_RANGE_JUSTIFICATION,
        "settings_frozen_before_results": True,
        "scenarios": SCENARIOS,
        "assumptions": examples["correct"]["estimate"].assumptions,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=json_default), encoding="utf-8"
    )

    plot_example(examples, OUTPUT_DIR / "runtime_example_case_SYNTHETIC.png")
    plot_errors(details, OUTPUT_DIR / "runtime_errors_SYNTHETIC.png")
    plot_inside_rate(summary, OUTPUT_DIR / "runtime_inside_rate_SYNTHETIC.png")
    plot_coverage_width_tradeoff(
        summary, OUTPUT_DIR / "runtime_coverage_width_tradeoff_SYNTHETIC.png"
    )
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text(build_docs(summary, metadata["assumptions"]), encoding="utf-8")

    print(f"details={details_path}")
    print(f"summary={summary_path}")
    print(f"metadata={metadata_path}")
    print(f"docs={DOCS_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
