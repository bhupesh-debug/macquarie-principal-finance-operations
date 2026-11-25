"""Forecasting models (Problem 1) + Holt-Winters (additive, hand-rolled).

Models, all on the same features and the same time-based split:
- Seasonal naive (value 7 days ago) — the sanity check.
- Holt-Winters additive (level + trend + weekly seasonality),
  implemented in numpy (statsmodels is not installed here).
- Linear regression, random forest, gradient boosting (scikit-learn).

Two targets, both on the daily portfolio total:
- deals_closed (count) — the pipeline throughput forecast.
- avg_cycle_time_days — the operational turnaround forecast.

The honest multi-step number is the Holt-Winters recursive forecast
from the end of training data (not the 1-step-ahead fits).
"""

import numpy as np
from sklearn.ensemble import (GradientBoostingRegressor,
                               RandomForestRegressor)
from sklearn.linear_model import LinearRegression


def seasonal_naive(train_y, test_len, season=7):
    """Repeat the last full seasonal cycle."""
    base = np.asarray(train_y)[-season:]
    return np.tile(base, int(np.ceil(test_len / season)))[:test_len]


def holt_winters_additive(y, season=7, alpha=0.3, beta=0.1, gamma=0.1,
                          horizon=61):
    """Additive HW: level + trend + seasonal. Returns recursive forecast."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    level = y[:season].mean()
    trend = (y[season:2 * season].mean() - y[:season].mean()) / season
    seas = np.array([y[i] - y[:season].mean() for i in range(season)])
    fitted = np.zeros(n)
    for t in range(n):
        s_idx = t % season
        if t == 0:
            fitted[t] = level + seas[s_idx]
            continue
        prev_level, prev_trend = level, trend
        level = alpha * (y[t] - seas[s_idx]) + (1 - alpha) * (
            prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * prev_trend
        seas[s_idx] = gamma * (y[t] - level) + (1 - gamma) * seas[s_idx]
        fitted[t] = level + trend + seas[s_idx]
    fc = []
    for h in range(1, horizon + 1):
        s_idx = (n + h - 1) % season
        fc.append(level + h * trend + seas[s_idx])
    return np.array(fc), fitted


def fit_sklearn_models(X_train, y_train):
    models = {
        "Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=5, min_samples_leaf=10,
            random_state=42, n_jobs=1),  # n_jobs=1: bit-for-bit
        # reproducible (parallel reduction order varies otherwise)
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}
