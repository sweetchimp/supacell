# Forecasting results (`real`)

## What was measured

Four next-day hourly forecasts of `global_active_energy_kwh` were evaluated on the
cleaned UCI household data (`data/processed/household_power_hourly_REAL.csv`).
Each target day's 24 hours were predicted from a single origin at 23:00 on the prior
day. Features used calendar fields plus lagged/rolling target energy only; no
sub-meter, weather or same-day actual values were used.

- Chronological splits, no shuffling: train 2006-12-23T18:00:00 to 2009-09-17T23:00:00 (23837 rows); validation 2009-09-18T00:00:00 to 2010-04-19T23:00:00 (5050 rows); test 2010-04-20T00:00:00 to 2010-11-26T20:00:00 (5092 rows).
- Models: same-hour/same-weekday historical average, linear regression, random forest and XGBoost.
- Best two by validation RMSE, used for prediction intervals: xgboost, random_forest.
- Intervals: split conformal on absolute validation residuals at nominal 90%.
- MAPE is computed because all 5092 test actuals are positive (minimum 0.195100 kWh).

## Validation metrics used to select the interval models

| model | mae | rmse |
|---|---|---|
| historical_average | 0.505811 | 0.707700 |
| linear_regression | 0.531616 | 0.708510 |
| random_forest | 0.477545 | 0.664865 |
| xgboost | 0.475816 | 0.657133 |

## Test results

| model | mae_kwh | rmse_kwh | mape_pct | interval_model | nominal_coverage | empirical_coverage | avg_interval_width_kwh | conformal_radius_kwh |
|---|---|---|---|---|---|---|---|---|
| historical_average | 0.499554 | 0.662938 | 85.878285 | no | N/A | N/A | N/A | N/A |
| linear_regression | 0.446600 | 0.602203 | 64.968668 | no | N/A | N/A | N/A | N/A |
| random_forest | 0.415611 | 0.570417 | 60.590707 | yes | 0.900000 | 0.937353 | 2.115481 | 1.057740 |
| xgboost | 0.410507 | 0.567518 | 59.469378 | yes | 0.900000 | 0.932247 | 2.117985 | 1.058992 |

## Interval coverage

- random_forest: empirical coverage 0.937353, average width 2.115481 kWh, conformal radius 1.057740 kWh (nominal 0.90).
- xgboost: empirical coverage 0.932247, average width 2.117985 kWh, conformal radius 1.058992 kWh (nominal 0.90).

## What the numbers show

- Lowest test MAE: xgboost (0.410507 kWh).
- Lowest test RMSE: xgboost (0.567518 kWh).
- The historical-average baseline is not competitive on test: its MAE is 21.69% above the best MAE.
- The simple historical average beat linear regression on both validation MAE and validation RMSE, so the linear model was not competitive there.

All figures above are generated from `results/forecasting/forecast_results_REAL.csv`
and `results/forecasting/forecast_metadata_REAL.json` by
`ml/experiments/run_forecasting.py`.
