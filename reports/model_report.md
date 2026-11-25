# Model Report — Principal-Finance Operations Analytics

Synthetic loan-portfolio operations data (seed 42): 3 product lines x
730 days (2024-01-01 to 2025-12-30), 600-facility loan book.
Hold-out: last 61 days (2025-11-01 to 2025-12-30).

## Forecasting (deals_closed / avg_cycle_time_days)
- deals_closed / Seasonal Naive: MAPE 24.35%, MAE 3.61, RMSE 4.42
- deals_closed / Holt-Winters: MAPE 41.74%, MAE 4.56, RMSE 5.28
- deals_closed / Linear: MAPE 27.24%, MAE 3.26, RMSE 4.00
- deals_closed / Random Forest: MAPE 22.60%, MAE 3.14, RMSE 3.91
- deals_closed / Gradient Boosting: MAPE 26.00%, MAE 3.78, RMSE 4.51
- avg_cycle_time_days / Seasonal Naive: MAPE 10.53%, MAE 3.93, RMSE 4.95
- avg_cycle_time_days / Holt-Winters: MAPE 20.67%, MAE 7.47, RMSE 7.89
- avg_cycle_time_days / Linear: MAPE 4.05%, MAE 1.45, RMSE 1.75
- avg_cycle_time_days / Random Forest: MAPE 4.41%, MAE 1.58, RMSE 1.96
- avg_cycle_time_days / Gradient Boosting: MAPE 4.98%, MAE 1.78, RMSE 2.16

## Anomaly detection (z-score |z| > 4, hold-out)
- precision 1.000, recall 1.000,
  F1 1.000; 6 of 6 injected incidents
  caught, 0 false alarms, 0 missed.

## Cycle-time drivers (permutation importance, GBM)
- backlog_ratio: 1.982
- staffing_ratio: 0.497
- syndicated_share: 0.135
- approval_rate_pct: 0.132
- exception_rate_pct: 0.132
- system_uptime_pct: 0.117
- avg_deal_size_usd_m: 0.109

## Benchmark (mean cycle time by product line)
- Real Estate: 38.7 days, approval 83.0%, exceptions 5.59%
- Private Equity: 36.4 days, approval 82.9%, exceptions 5.53%
- Private Credit: 33.4 days, approval 83.0%, exceptions 5.59%
