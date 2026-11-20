# supacell research write-up

## Research question

How resilient are household energy forecasts and battery-runtime estimates to imperfect
inputs, and which added signals actually help? The analysis checks three claims:
capacity-health and efficiency uncertainty widen battery-runtime ranges, missing meter
history degrades prediction inputs and accuracy, and solar/weather inputs add forecast
value under explicitly frozen assumptions.

## Data provenance

| Dataset | Label | Source | Notes |
| --- | --- | --- | --- |
| UCI Individual Household Electric Power Consumption | `real` | UCI ML Repository, `data/processed/household_power_hourly_REAL.csv` | Hourly cleaned data; location documented as Sceaux, about 7 km from Paris, France. |
| Open-Meteo Historical Weather API / ERA5 | `real` | Open-Meteo archive for `48.77644, 2.29026`, `Europe/Paris` | Reanalysis joined to household rows in `data/processed/household_power_hourly_weather_REAL.csv`; CC BY 4.0, free tier non-commercial. Details in `docs/weather_data.md`. |
| Solar/battery simulation profiles | `synthetic` | `supacell_ml.simulation` | Controlled profiles with fixed seeds; not measured household, PV or battery data. |
| Battery validation paths | `synthetic` | `run_battery_runtime.py` | 80 seeded 72-hour load paths discharged with known true parameters. |

Label definitions:

- `real`: measured or observed data collected from an actual source.
- `synthetic`: generated data that does not represent a measured household.
- `simulated`: data produced by a model or scenario simulation.

## Methods

- Forecasting: chronological 70/15/15 train/validation/test split; XGBoost; split
  conformal intervals at alpha 0.10; origin is 23:00 on the day before each target day.
- Graceful degradation: seeded missing lag/rolling feature cells are forward-filled from
  the previous feature row, with the first affected row filled by the training median.
  Scored timestamps and clean test targets are preserved.
- Battery: expected load plus point capacity/efficiency gives the point estimate; the
  low case combines p90 load, 80% capacity health and 90% efficiency; the high case
  combines p10 load, 100% capacity health and 98% efficiency. The range is a worst-case
  envelope, not a calibrated probability interval.
- Solar: synthetic `grid_plus_solar_plus_battery` target `grid_import_kw`; target-hour
  solar is assumed available as a perfect synthetic input.
- Weather: ERA5 variables at 0, 3, 6, 12 and 24 hours before the forecast origin plus
  24-hour rolling statistics ending at the origin; target-hour weather is never used.

## Experiments

All settings were frozen before their results were inspected. Saved artifacts are under
`results/`, with detailed docs under `docs/`.

- Battery runtime: `results/battery_runtime/runtime_summary_SYNTHETIC.csv`.
- Graceful degradation: `results/degradation/degradation_results_REAL.csv`.
- Solar removal: `results/solar_degradation/solar_results_SYNTHETIC.csv`.
- Real weather removal: `results/weather_forecast/weather_results_REAL.csv`.

Input alteration for the missing-hours test is proven by
`ml/tests/test_degradation.py::test_degraded_prediction_inputs_differ_from_clean_inputs`,
which compares clean and degraded lag/rolling feature values at matching timestamps.

## Results

### Goal A: battery assumption uncertainty

| scenario | load-only inside | combined inside | load width (h) | combined width (h) | coverage gain (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct | 80/80 | 80/80 | 3.262641 | 4.814177 | 0.000000 |
| capacity_degraded_20pct | 39/80 | 80/80 | 3.262641 | 4.814177 | 51.250000 |
| efficiency_overestimated | 80/80 | 80/80 | 3.262641 | 4.814177 | 0.000000 |
| uncertainty_too_narrow | 79/80 | 80/80 | 0.571563 | 2.099149 | 1.250000 |

The combined range is wider and covers all seeded cases, including severe capacity
degradation, but it is a simultaneous envelope and is not calibrated.

### Goal B: graceful degradation

| level | MAE (kWh) | MAE change (%) | coverage | width (kWh) | coverage delta (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.410507 | 0.000000 | 0.932247 | 2.117985 | 0.000000 |
| missing_5pct | 0.410235 | -0.066221 | 0.933818 | 2.140553 | 0.157109 |
| missing_15pct | 0.413539 | 0.738586 | 0.932443 | 2.130927 | 0.019639 |
| missing_30pct | 0.416725 | 1.514834 | 0.934603 | 2.111014 | 0.235664 |
| missing_50pct | 0.420001 | 2.312742 | 0.934407 | 2.144700 | 0.216025 |
| missing_80pct | 0.423659 | 3.204000 | 0.934211 | 2.157933 | 0.196386 |
| history_2y | 0.411496 | 0.240967 | 0.933425 | 2.113314 | 0.117832 |
| history_1y | 0.408397 | -0.513815 | 0.939906 | 2.176282 | 0.765907 |
| history_6mo | 0.425448 | 3.639749 | 0.943244 | 2.295217 | 1.099764 |
| history_3mo | 0.444133 | 8.191353 | 0.937549 | 2.303710 | 0.530244 |
| no_solar_data | 0.410507 | 0.000000 | 0.932247 | 2.117985 | 0.000000 |
| no_weather_features | 0.410507 | 0.000000 | 0.932247 | 2.117985 | 0.000000 |

Missing-hour corruption now changes MAE monotonically at higher levels, and the lag/rolling
columns corrupted by the test held 60.2% of feature importance in the clean model and
34.8% at `missing_80pct`.

### Goal C: signal removal

| experiment | variant | MAE | RMSE | coverage | width | added-signal importance | MAE change (%) | width change (%) | coverage delta (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic solar | no_solar | 0.054817 | 0.096244 | 0.878030 | 0.300510 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| synthetic solar | with_solar | 0.046859 | 0.086193 | 0.887879 | 0.289111 | 0.064896 | -14.517310 | -3.793144 | 0.984848 |
| real weather | no_weather | 0.410507 | 0.567518 | 0.932247 | 2.117985 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| real weather | with_weather | 0.416044 | 0.573474 | 0.938335 | 2.164640 | 0.495900 | 1.348935 | 2.202786 | 0.608798 |

Negative changes mean the added-signal variant was lower than the baseline; zero means no
change. Positive changes mean the added-signal variant was worse.

## Unexpected findings

- Battery: the `efficiency_overestimated` case has true efficiency 0.85, outside the
  frozen 0.90-0.98 assumption range, yet the combined range still covers it because the
  low capacity-health assumption dominates the low runtime bound.
- Degradation: MAE is no longer flat across missing-hour levels after the frozen gap-fill
  method was applied to lag/rolling feature cells.
- Weather: ERA5 weather held 49.6% of XGBoost importance but worsened MAE by 1.348935%
  and widened intervals by 2.202786%. This is a warning that importance mass is not the
  same as out-of-sample predictive value.

## Limitations and ethics

- Battery validation uses simulated discharge only. These numbers do not measure real
  battery behavior. Real validation would need measured capacity/state-of-health tests,
  voltage-current-temperature and SoC traces, inverter efficiency curves, ageing state,
  derating/fault events, and multiple devices, chemistries and ages.
- Battery p10/p90 paths are treated as simultaneous trajectories, not independent hourly
  quantiles; the combined range is a worst-case envelope, not a calibrated interval.
- Missing-hour and short-history levels are synthetic corruptions of real data, not
  naturally missing periods. Forward fill tests degraded feature quality, not actual
  meter-outage behavior.
- `no_solar_data` and `no_weather_features` in the degradation experiment are honest
  no-ops because the selected model never consumed those inputs.
- Solar results are synthetic and assume perfect target-hour solar availability; they do
  not establish operational day-ahead value.
- Weather is grid-cell reanalysis, not measurement at the house, and is limited to
  origin-available values. A local station, on-site sensor or day-ahead forecast would be
  needed for a stronger household-specific claim.
- All forecast results are for one household, one chronological test window and one model
  configuration. Conformal intervals are recalibrated per variant.
- Open-Meteo free-tier use is non-commercial and requires CC BY attribution; synthetic
  and simulated results must not be presented as measured household data.
