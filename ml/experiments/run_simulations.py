import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from supacell_ml.simulation import (
    PROFILES,
    PROFILE_LABELS,
    SimulationConfig,
    generate_household_timeseries,
)

OUTPUT_DIR = REPO_ROOT / "results" / "simulations"
WEEK_HOURS = 168
SEED = 2026
START = "2025-01-01"


def plot_week(frame, profile, output_path):
    week = frame.iloc[:WEEK_HOURS]
    max_power = max(
        week["demand_kw"].max(),
        week["solar_kw"].max(),
        week["grid_import_kw"].max(),
        1.0,
    )
    fig, power_axis = plt.subplots(figsize=(13, 6))
    power_axis.plot(week.index, week["demand_kw"], label="Demand", color="#1f77b4")
    if week["solar_kw"].max() > 0:
        power_axis.plot(week.index, week["solar_kw"], label="Solar", color="#ff7f0e")
    power_axis.step(
        week.index,
        week["grid_available"] * max_power,
        where="post",
        linestyle="--",
        linewidth=1.2,
        color="#7f7f7f",
        label="Grid available (scaled)",
    )
    power_axis.set_ylabel("Power (kW)")
    power_axis.set_xlabel("Timestamp")
    power_axis.grid(alpha=0.3)
    power_axis.legend(loc="upper left", ncol=3)

    battery_axis = power_axis.twinx()
    battery_axis.plot(
        week.index,
        week["battery_kwh"],
        label="Battery SoC",
        color="#2ca02c",
        linewidth=2.0,
    )
    battery_axis.set_ylabel("Battery state of charge (kWh)")
    battery_axis.legend(loc="upper right")

    title = f"{PROFILE_LABELS[profile]} - one week - SYNTHETIC"
    power_axis.set_title(title)
    fig.text(
        0.5,
        0.5,
        "SYNTHETIC",
        ha="center",
        va="center",
        fontsize=54,
        color="red",
        alpha=0.12,
        rotation=25,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(start=START, seed=SEED)
    for profile in PROFILES:
        frame = generate_household_timeseries(profile, config=config)
        csv_path = OUTPUT_DIR / f"{profile}_hourly_SYNTHETIC.csv"
        plot_path = OUTPUT_DIR / f"{profile}_week_SYNTHETIC.png"
        frame.to_csv(csv_path, index_label="timestamp")
        plot_week(frame, profile, plot_path)
        print(f"profile={profile} rows={len(frame)}")
        print(f"csv={csv_path}")
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
