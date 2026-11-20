# Battery runtime range (`synthetic`)

## Frozen assumption settings

These settings were defined before the revised validation was run and were not changed
after seeing results.

- Capacity-health range: 80%-100% of supplied capacity.
  Justification: 80% of nameplate is a conservative end-of-life-style lower bound; 100% covers a healthy asset.
- One-way efficiency range: 90%-98%.
  Justification: 90%-98% is a broad normal inverter/converter envelope; no vendor efficiency curve is available.
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
a plain list of assumptions. The nominal load band was 80%.

Validation used 80 seeded synthetic load paths of 72 hours from the
`GRID_ONLY` simulator profile. Each path was discharged step-by-step with the same
efficiency and depth-of-discharge physics but with independent "true" parameters.
`SYNTHETIC` labels are used for all artifacts in this section.

## Results

| scenario | cases | mean_true_hours | load_only_mean_width_hours | load_only_inside_count | load_only_inside_rate | combined_mean_low_hours | combined_mean_expected_hours | combined_mean_high_hours | combined_mean_error_expected_hours | combined_mae_expected_hours | combined_mean_width_hours | combined_inside_count | combined_inside_rate | width_gain_hours | coverage_gain_pp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| correct | 80 | 9.194165 | 3.262641 | 80 | 1.000000 | 6.846155 | 9.206552 | 11.660331 | -0.012388 | 0.077603 | 4.814177 | 80 | 1.000000 | 1.551536 | 0.000000 |
| capacity_degraded_20pct | 80 | 8.067325 | 3.262641 | 39 | 0.487500 | 6.846155 | 9.206552 | 11.660331 | -1.139227 | 1.139227 | 4.814177 | 80 | 1.000000 | 1.551536 | 51.250000 |
| efficiency_overestimated | 80 | 8.604745 | 3.262641 | 80 | 1.000000 | 6.846155 | 9.206552 | 11.660331 | -0.601807 | 0.601807 | 4.814177 | 80 | 1.000000 | 1.551536 | 0.000000 |
| uncertainty_too_narrow | 80 | 9.194165 | 0.571563 | 79 | 0.987500 | 7.605362 | 9.206552 | 9.704511 | -0.012388 | 0.077603 | 2.099149 | 80 | 1.000000 | 1.527586 | 1.250000 |

## Coverage/width trade-off

The combined range adds capacity-health and efficiency uncertainty to the existing load
band. Compare `load_only_mean_width_hours` with `combined_mean_width_hours` and
`load_only_inside_rate` with `combined_inside_rate`. `width_gain_hours` and
`coverage_gain_pp` report the measured trade-off for each scenario. Higher coverage from
a wider range is not evidence of a calibrated probability interval.

## Assumptions emitted with every estimate

- Nominal 80% load band supplied as p10/p90; endpoints are treated as simultaneous load paths.
- Usable stored energy at point assumptions is max(0, 8.000 kWh SoC - 2.000 kWh DoD floor) = 6.000 kWh.
- Point one-way conversion efficiency of 95% gives 5.700 kWh usable AC energy.
- Capacity-health assumption range is 80%-100% of supplied capacity; health scales both capacity and SoC.
- One-way efficiency assumption range is 90%-98%.
- Low runtime combines p90 load, low capacity health and low efficiency; high runtime combines p10 load, high capacity health and high efficiency.
- The combined range is a simultaneous worst-case envelope, not a calibrated probability interval.
- Low-case effective capacity is 8.000 kWh; high-case effective capacity is 10.000 kWh.
- Load is applied in one-hour steps with linear interpolation inside the depletion hour.
- No solar generation, grid charging, load shifting or reserve requirement during the horizon.
- No inverter power cap, temperature derating or aging beyond the supplied capacity-health range.
- Runtime is capped at the 72-hour forecast horizon when the battery is not depleted.

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
