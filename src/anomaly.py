"""Anomaly detection (Problem 2) + driver analysis (Problem 3).

Problem 2 — z-score rule on daily pipeline-health features. A day is
flagged when ANY health feature is more than 4 sigma from its
training-period mean. The rule is deliberately explainable: a loan-ops
manager must be able to say WHY a day was flagged, which rules out a
black-box detector for this job. Recall is prioritised over precision:
a missed bad feed corrupts the loan book; a false alarm costs one
look.

Problem 3 — gradient boosting on same-day operational features to
explain avg_cycle_time_days, with permutation importance. Same-day
features are legitimate HERE (explanation, not forecasting).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

HEALTH_FEATURES = ["missing_rate_pct", "duplicate_rate_pct",
                   "null_rate_pct", "stale_record_pct",
                   "late_feed_minutes"]


def zscore_flag(train, test, features=HEALTH_FEATURES, k=4.0):
    """Flag test days where any feature exceeds k sigma from train mean."""
    mu = train[features].mean()
    sd = train[features].std(ddof=1).replace(0, np.nan)
    z = (test[features] - mu) / sd
    flagged = (z.abs() > k).any(axis=1)
    return flagged.values, z


def anomaly_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "n_true": int(y_true.sum()), "n_flagged": int(y_pred.sum())}


def driver_model(train, test, features, target="avg_cycle_time_days"):
    """GBM on same-day features + permutation importance (train set)."""
    model = GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(train[features], train[target])
    imp = permutation_importance(model, train[features], train[target],
                                 n_repeats=10, random_state=42,
                                 scoring="neg_mean_absolute_error")
    out = pd.DataFrame({
        "feature": features,
        "permutation_importance": imp.importances_mean,
    }).sort_values("permutation_importance", ascending=False)
    pred = model.predict(test[features])
    return model, out, pred
