"""Feature engineering.

Forecasting features use ONLY information available at forecast time:
lags and rolling means of the target, calendar features, and
operational features known a day ahead (pipeline, staffing). Same-day
operational features (staffing_ratio, backlog_ratio, exception rate)
are EXCLUDED from the forecaster — including them would leak, because
they are measured on the same day as the outcome. They belong to the
driver model (Problem 3), not the forecast.
"""

import numpy as np
import pandas as pd


def add_forecast_features(df, target):
    """Lag / rolling / calendar features for one daily series."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    y = df[target]
    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = y.shift(lag)
    df["roll7"] = y.shift(1).rolling(7).mean()
    df["roll28"] = y.shift(1).rolling(28).mean()
    df["vol_dev_28d"] = (y.shift(1) - df["roll28"]) / df["roll28"]
    df["dow"] = df["date"].dt.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["pipeline_lag1"] = df["deals_in_pipeline"].shift(1)
    df["received_lag1"] = df["deals_received"].shift(1)
    return df


FORECAST_FEATURES = ["lag_1", "lag_7", "lag_14", "lag_28", "roll7",
                     "roll28", "vol_dev_28d", "dow", "weekend",
                     "month", "pipeline_lag1", "received_lag1"]

DRIVER_FEATURES = ["backlog_ratio", "staffing_ratio",
                   "exception_rate_pct", "syndicated_share",
                   "avg_deal_size_usd_m", "approval_rate_pct",
                   "system_uptime_pct"]


def time_split(df, holdout_days=61):
    """Time-based split: last `holdout_days` days are the hold-out."""
    df = df.sort_values("date").reset_index(drop=True)
    cut = df["date"].max() - pd.Timedelta(days=holdout_days - 1)
    train = df[df["date"] < cut].copy()
    test = df[df["date"] >= cut].copy()
    return train, test, cut


if __name__ == "__main__":
    print("features:", FORECAST_FEATURES)
    print("driver features:", DRIVER_FEATURES)
