"""Orchestrator: generate -> validate -> features -> models -> metrics.

Run from the project root:
    ~/workspace/quant-ipo-engine/.venv/bin/python run_all.py

Deterministic: seed 42 everywhere. Hold-out is the last 61 days
(2025-11-01 to 2025-12-30). Every metric in the PDF and deck comes
from models/*.csv written here.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import anomaly as an  # noqa: E402
import charts as ch  # noqa: E402
import data_generation as dg  # noqa: E402
import data_pipeline as dp  # noqa: E402
import feature_engineering as fe  # noqa: E402
import models as md  # noqa: E402

MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(os.path.join(ROOT, "reports", "charts"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)

# 1. Generate + validate -------------------------------------------------
import subprocess
syn_path = os.path.join(ROOT, "data", "synthetic", "ops_daily.csv")
if not os.path.exists(syn_path):
    subprocess.run([sys.executable, os.path.join(ROOT, "src",
                                                 "data_generation.py")],
                   check=True)
ops, health, fac = dp.load()
findings = dp.validate(ops, health, fac)
print("validation:", findings)
tot = dp.daily_totals(ops)
tot.to_csv(os.path.join(ROOT, "data", "processed",
                        "ops_daily_total.csv"), index=False)

train_all, test_all, cut = fe.time_split(tot, 61)
print(f"train {train_all['date'].min().date()}–"
      f"{train_all['date'].max().date()} ({len(train_all)} days); "
      f"hold-out {test_all['date'].min().date()}–"
      f"{test_all['date'].max().date()} ({len(test_all)} days)")

# 2. Forecasting, both targets -------------------------------------------
metric_rows = []
forecast_store = {}
for target in ("deals_closed", "avg_cycle_time_days"):
    feat = fe.add_forecast_features(tot, target)
    train, test, _ = fe.time_split(feat.dropna().reset_index(drop=True),
                                   61)
    Xtr, ytr = train[fe.FORECAST_FEATURES], train[target]
    Xte, yte = test[fe.FORECAST_FEATURES], test[target]
    preds = {}
    # Seasonal naive on the raw series (no features needed)
    naive = md.seasonal_naive(
        feat[feat["date"] < cut][target].values, len(test))
    preds["Seasonal Naive"] = naive
    # Holt-Winters recursive from end of training data
    hw_fc, _ = md.holt_winters_additive(
        feat[feat["date"] < cut][target].values, horizon=len(test))
    preds["Holt-Winters"] = hw_fc
    # sklearn models on lag/calendar features
    fitted = md.fit_sklearn_models(Xtr, ytr)
    for name, model in fitted.items():
        preds[name] = model.predict(Xte)
    for name, pr in preds.items():
        m = md.metrics(yte, pr)
        metric_rows.append({"target": target, "model": name, **m})
        print(f"{target:22s} {name:18s} MAPE {m['mape']:6.2f}%  "
              f"MAE {m['mae']:6.2f}  RMSE {m['rmse']:6.2f}")
    forecast_store[target] = (test, preds)

pd.DataFrame(metric_rows).to_csv(
    os.path.join(MODELS, "metrics.csv"), index=False)

# 3. Anomaly detection -----------------------------------------------------
htr, hte, _ = fe.time_split(health, 61)
flagged, _ = an.zscore_flag(htr, hte)
am = an.anomaly_metrics(hte["is_anomaly"].values, flagged)
pd.DataFrame([{"metric": k, "value": v}
              for k, v in am.items()]).to_csv(
    os.path.join(MODELS, "anomaly_metrics.csv"), index=False)
print("anomaly:", {k: round(v, 3) if isinstance(v, float) else v
                   for k, v in am.items()})

# 4. Drivers + benchmark ---------------------------------------------------
model, drivers, _ = an.driver_model(
    train_all, test_all, fe.DRIVER_FEATURES)
drivers.to_csv(os.path.join(MODELS, "driver_importance.csv"),
               index=False)
print("drivers:\n", drivers.to_string(index=False))

bench = (ops.groupby("product_line")
         .agg(avg_cycle_time_days=("avg_cycle_time_days", "mean"),
              approval_rate_pct=("approval_rate_pct", "mean"),
              exception_rate_pct=("exception_rate_pct", "mean"),
              deals_received=("deals_received", "sum"),
              avg_deal_size_usd_m=("avg_deal_size_usd_m", "mean"))
         .reset_index()
         .sort_values("avg_cycle_time_days", ascending=False))
bench.to_csv(os.path.join(MODELS, "benchmark.csv"), index=False)
print("benchmark:\n", bench.to_string(index=False))

# 5. Charts ----------------------------------------------------------------
test_c, preds_c = forecast_store["deals_closed"]
hw_fc = preds_c["Holt-Winters"]
ch.forecast_charts(tot, hw_fc, test_c["date"].values,
                   preds_c["Gradient Boosting"], test_c)
ch.model_comparison()
ch.feature_importance()
ch.anomaly_timeline()
ch.benchmark()
ch.correlation_heatmap()

# 6. Model report -----------------------------------------------------------
report = f"""# Model Report — Principal-Finance Operations Analytics

Synthetic loan-portfolio operations data (seed 42): 3 product lines x
730 days (2024-01-01 to 2025-12-30), 600-facility loan book.
Hold-out: last 61 days (2025-11-01 to 2025-12-30).

## Forecasting (deals_closed / avg_cycle_time_days)
"""
for r in metric_rows:
    report += (f"- {r['target']} / {r['model']}: MAPE "
               f"{r['mape']:.2f}%, MAE {r['mae']:.2f}, "
               f"RMSE {r['rmse']:.2f}\n")
report += f"""
## Anomaly detection (z-score |z| > 4, hold-out)
- precision {am['precision']:.3f}, recall {am['recall']:.3f},
  F1 {am['f1']:.3f}; {am['tp']} of {am['n_true']} injected incidents
  caught, {am['fp']} false alarms, {am['fn']} missed.

## Cycle-time drivers (permutation importance, GBM)
"""
for _, row in drivers.iterrows():
    report += f"- {row['feature']}: {row['permutation_importance']:.3f}\n"
report += "\n## Benchmark (mean cycle time by product line)\n"
for _, row in bench.iterrows():
    report += (f"- {row['product_line']}: "
               f"{row['avg_cycle_time_days']:.1f} days, approval "
               f"{row['approval_rate_pct']:.1f}%, exceptions "
               f"{row['exception_rate_pct']:.2f}%\n")
with open(os.path.join(ROOT, "reports", "model_report.md"), "w") as fh:
    fh.write(report)
print("wrote reports/model_report.md")
print("DONE")
