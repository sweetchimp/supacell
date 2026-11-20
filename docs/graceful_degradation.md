# Graceful degradation experiment (`real`)

## What was measured

The degradation experiment used the previously selected best forecasting model,
`xgboost`, on the cleaned UCI household data. Chronological splits
were kept fixed: train 2006-12-23T18:00:00 to 2009-09-17T23:00:00,
validation 2009-09-18T00:00:00 to 2010-04-19T23:00:00, test
2010-04-20T00:00:00 to 2010-11-26T20:00:00. No rows were shuffled.

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

| level | description | train_rows | test_rows | mae_kwh | rmse_kwh | mape_pct | nominal_coverage | empirical_coverage | avg_interval_width_kwh | conformal_radius_kwh | mae_change_pct | width_change_pct | coverage_delta_pp | user_message |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | Clean feature history | 23837 | 5092 | 0.410507 | 0.567518 | 59.469378 | 0.900000 | 0.932247 | 2.117985 | 1.058992 | 0.000000 | 0.000000 | 0.000000 | Full feature history available; operating normally. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.2%; mean width 2.118 kWh. |
| missing_5pct | 5% of history hours randomly unavailable | 23837 | 5092 | 0.410235 | 0.567824 | 59.336581 | 0.900000 | 0.933818 | 2.140553 | 1.070276 | -0.066221 | 1.065536 | 0.157109 | Minor gaps in history were simulated. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.4%; mean width 2.141 kWh. |
| missing_15pct | 15% of history hours randomly unavailable | 23837 | 5092 | 0.413539 | 0.571462 | 59.716110 | 0.900000 | 0.932443 | 2.130927 | 1.065463 | 0.738586 | 0.611039 | 0.019639 | Moderate gaps in history were simulated. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.2%; mean width 2.131 kWh. |
| missing_30pct | 30% of history hours randomly unavailable | 23837 | 5092 | 0.416725 | 0.573098 | 60.011388 | 0.900000 | 0.934603 | 2.111014 | 1.055507 | 1.514834 | -0.329132 | 0.235664 | Substantial gaps in history were simulated. Measured MAE increased 1.5% versus full data on 5092 scored hours; 90% interval coverage 93.5%; mean width 2.111 kWh. |
| missing_50pct | 50% of history hours randomly unavailable | 23837 | 5092 | 0.420001 | 0.576829 | 60.537991 | 0.900000 | 0.934407 | 2.144700 | 1.072350 | 2.312742 | 1.261337 | 0.216025 | Severe gaps in history were simulated. Measured MAE increased 2.3% versus full data on 5092 scored hours; 90% interval coverage 93.4%; mean width 2.145 kWh. |
| missing_80pct | 80% of history hours randomly unavailable | 23837 | 5092 | 0.423659 | 0.580138 | 61.051539 | 0.900000 | 0.934211 | 2.157933 | 1.078966 | 3.204000 | 1.886114 | 0.196386 | Extreme gaps in history were simulated. Measured MAE increased 3.2% versus full data on 5092 scored hours; 90% interval coverage 93.4%; mean width 2.158 kWh. |
| history_2y | History truncated to 2 years before validation | 17275 | 5092 | 0.411496 | 0.570447 | 59.463236 | 0.900000 | 0.933425 | 2.113314 | 1.056657 | 0.240967 | -0.220530 | 0.117832 | Only two years of history are available. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.3%; mean width 2.113 kWh. |
| history_1y | History truncated to 1 year before validation | 8515 | 5092 | 0.408397 | 0.566350 | 58.514255 | 0.900000 | 0.939906 | 2.176282 | 1.088141 | -0.513815 | 2.752494 | 0.765907 | Only one year of history is available. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 94.0%; mean width 2.176 kWh. |
| history_6mo | History truncated to 6 months before validation | 4153 | 5092 | 0.425448 | 0.592885 | 58.677858 | 0.900000 | 0.943244 | 2.295217 | 1.147609 | 3.639749 | 8.367965 | 1.099764 | Only six months of history are available. Measured MAE increased 3.6% versus full data on 5092 scored hours; 90% interval coverage 94.3%; mean width 2.295 kWh. |
| history_3mo | History truncated to 3 months before validation | 1977 | 5092 | 0.444133 | 0.608648 | 62.120385 | 0.900000 | 0.937549 | 2.303709 | 1.151854 | 8.191353 | 8.768897 | 0.530244 | Only three months of history are available. Measured MAE increased 8.2% versus full data on 5092 scored hours; 90% interval coverage 93.8%; mean width 2.304 kWh. |
| no_solar_data | Solar data marked unavailable; model never used solar | 23837 | 5092 | 0.410507 | 0.567518 | 59.469378 | 0.900000 | 0.932247 | 2.117985 | 1.058992 | 0.000000 | 0.000000 | 0.000000 | Solar data is not used by this model. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.2%; mean width 2.118 kWh. |
| no_weather_features | Weather features marked unavailable; model never used weather | 23837 | 5092 | 0.410507 | 0.567518 | 59.469378 | 0.900000 | 0.932247 | 2.117985 | 1.058992 | 0.000000 | 0.000000 | 0.000000 | Weather features are not used by this model. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.2%; mean width 2.118 kWh. |

## What the system should say

- `full`: Full feature history available; operating normally. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.2%; mean width 2.118 kWh.
- `missing_5pct`: Minor gaps in history were simulated. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.4%; mean width 2.141 kWh.
- `missing_15pct`: Moderate gaps in history were simulated. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.2%; mean width 2.131 kWh.
- `missing_30pct`: Substantial gaps in history were simulated. Measured MAE increased 1.5% versus full data on 5092 scored hours; 90% interval coverage 93.5%; mean width 2.111 kWh.
- `missing_50pct`: Severe gaps in history were simulated. Measured MAE increased 2.3% versus full data on 5092 scored hours; 90% interval coverage 93.4%; mean width 2.145 kWh.
- `missing_80pct`: Extreme gaps in history were simulated. Measured MAE increased 3.2% versus full data on 5092 scored hours; 90% interval coverage 93.4%; mean width 2.158 kWh.
- `history_2y`: Only two years of history are available. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.3%; mean width 2.113 kWh.
- `history_1y`: Only one year of history is available. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 94.0%; mean width 2.176 kWh.
- `history_6mo`: Only six months of history are available. Measured MAE increased 3.6% versus full data on 5092 scored hours; 90% interval coverage 94.3%; mean width 2.295 kWh.
- `history_3mo`: Only three months of history are available. Measured MAE increased 8.2% versus full data on 5092 scored hours; 90% interval coverage 93.8%; mean width 2.304 kWh.
- `no_solar_data`: Solar data is not used by this model. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.2%; mean width 2.118 kWh.
- `no_weather_features`: Weather features are not used by this model. Measured MAE changed by less than 1% versus full data on 5092 scored hours; 90% interval coverage 93.2%; mean width 2.118 kWh.

## What the numbers show

- Lowest MAE occurred at `history_1y` (0.408397 kWh); highest MAE occurred at `history_3mo` (0.444133 kWh).
- `no_solar_data` produced MAE change +0.000000% because the selected model never consumed that input.
- `no_weather_features` produced MAE change +0.000000% because the selected model never consumed that input.
- Random missing-hour levels changed MAE by at most 3.204000% versus full data.
- Empirical coverage ranged from 0.932247 to 0.943244 (spread 0.010998).

## Feature importance evidence

Importance is XGBoost `feature_importances_`, saved for every fitted level. The
`lag_feature_importance_share` column is the importance mass on the same lag/rolling
columns that missing-hour degradation corrupts.

| level | lag_feature_importance_share | calendar_feature_importance_share | top_feature | top_feature_importance |
|---|---|---|---|---|
| full | 0.601782 | 0.398218 | lag_168h | 0.184404 |
| missing_5pct | 0.583842 | 0.416158 | lag_168h | 0.179308 |
| missing_15pct | 0.565854 | 0.434146 | lag_168h | 0.158509 |
| missing_30pct | 0.494525 | 0.505475 | lag_168h | 0.122312 |
| missing_50pct | 0.418956 | 0.581044 | year_cos | 0.140895 |
| missing_80pct | 0.348113 | 0.651887 | year_cos | 0.172233 |
| history_2y | 0.597430 | 0.402570 | lag_168h | 0.171351 |
| history_1y | 0.508302 | 0.491698 | lag_168h | 0.104639 |
| history_6mo | 0.476692 | 0.523308 | weekend | 0.103334 |
| history_3mo | 0.504423 | 0.495577 | year_sin | 0.097470 |
| no_solar_data | 0.601782 | 0.398218 | lag_168h | 0.184404 |
| no_weather_features | 0.601782 | 0.398218 | lag_168h | 0.184404 |

The largest missing-hour MAE change was +3.204000% at `missing_80pct`; corrupted lag/rolling columns held 34.8% of XGBoost importance mass.

## Limitations

- Missing-hour and short-history levels are synthetic corruptions applied to real data, not naturally missing periods.
- Forward fill can smooth over the exact outage pattern; it tests degraded feature quality, not missing-meter behaviour.
- The no-solar/no-weather levels test nothing about model reliance because those columns are absent from the feature set in this experiment.
- Re-calibrating conformal intervals at every level measures adaptive behaviour, not whether a fixed interval would remain valid.
- Results are for one household, one chronological test window and one model configuration.

All figures above are generated by `ml/experiments/run_degradation.py` from
`results/degradation/degradation_results_REAL.csv` and
`results/degradation/degradation_metadata_REAL.json`.
