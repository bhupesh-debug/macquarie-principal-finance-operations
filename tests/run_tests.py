"""Test suite for the Macquarie Principal-Finance Operations project.

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python tests/run_tests.py

Every check is a real assertion on the generated artefacts — file
existence, schema, row counts, hold-out design, metric sanity, and
determinism of the headline numbers. Exit code 0 = all pass.
"""

import csv
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

PASSED = 0
FAILED = []


def check(name, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print(f"  PASS {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


print("== files exist ==")
for rel in ["data/synthetic/ops_daily.csv",
            "data/synthetic/pipeline_health_daily.csv",
            "data/synthetic/loan_facilities.csv",
            "data/processed/ops_daily_total.csv",
            "models/metrics.csv", "models/anomaly_metrics.csv",
            "models/driver_importance.csv", "models/benchmark.csv",
            "reports/model_report.md", "README.md",
            "requirements.txt", "run_all.py"]:
    check(f"exists {rel}", os.path.exists(os.path.join(ROOT, rel)))
for chart in ["forecast_vs_actual.png", "model_comparison.png",
              "feature_importance.png", "anomaly_timeline.png",
              "benchmark.png", "correlation_heatmap.png"]:
    check(f"chart {chart}",
          os.path.exists(os.path.join(ROOT, "reports", "charts", chart)))

print("== schema & row counts ==")
ops = pd.read_csv(os.path.join(ROOT, "data/synthetic/ops_daily.csv"),
                  parse_dates=["date"])
check("ops 2190 rows (3 lines x 730 days)", len(ops) == 2190,
      f"got {len(ops)}")
check("ops has 3 product lines", ops["product_line"].nunique() == 3)
check("ops no missing values", int(ops.isna().sum().sum()) == 0)
check("ops no negative deals", (ops["deals_received"] >= 0).all()
      and (ops["deals_closed"] >= 0).all())
check("ops cycle time in range",
      ops["avg_cycle_time_days"].between(9, 68).all())
health = pd.read_csv(os.path.join(
    ROOT, "data/synthetic/pipeline_health_daily.csv"),
    parse_dates=["date"])
check("health 730 rows", len(health) == 730)
check("health has 25 injected incidents",
      int(health["is_anomaly"].sum()) == 25)
check("health 6 incidents in hold-out",
      int(health[health["date"] >= "2025-10-31"]
          ["is_anomaly"].sum()) == 6)
fac = pd.read_csv(os.path.join(ROOT,
                               "data/synthetic/loan_facilities.csv"))
check("facilities 600 rows", len(fac) == 600)
check("facility IDs unique", fac["facility_id"].is_unique)
check("commitments positive", (fac["commitment_usd_m"] > 0).all())

print("== hold-out design ==")
tot = pd.read_csv(os.path.join(ROOT, "data/processed",
                               "ops_daily_total.csv"),
                  parse_dates=["date"])
check("totals end 2025-12-30",
      tot["date"].max() == pd.Timestamp("2025-12-30"))
check("totals 730 days", len(tot) == 730)

print("== metrics sanity ==")
metrics = read_csv(os.path.join(ROOT, "models/metrics.csv"))
check("metrics has 10 rows (2 targets x 5 models)",
      len(metrics) == 10, f"got {len(metrics)}")
for r in metrics:
    check(f"MAPE finite {r['target']}/{r['model']}",
          0 <= float(r["mape"]) < 100)
    check(f"MAE positive {r['target']}/{r['model']}",
          float(r["mae"]) > 0)
closed = [r for r in metrics if r["target"] == "deals_closed"]
best_closed = min(closed, key=lambda r: float(r["mape"]))
check("best deals_closed model is Random Forest",
      best_closed["model"] == "Random Forest",
      f"got {best_closed['model']}")
check("best deals_closed MAPE < 25%",
      float(best_closed["mape"]) < 25,
      f"got {best_closed['mape']}")
cycle = [r for r in metrics if r["target"] == "avg_cycle_time_days"]
best_cycle = min(cycle, key=lambda r: float(r["mape"]))
check("best cycle-time model is Linear",
      best_cycle["model"] == "Linear", f"got {best_cycle['model']}")
check("best cycle-time MAPE < 6%",
      float(best_cycle["mape"]) < 6, f"got {best_cycle['mape']}")

print("== anomaly detection ==")
anom = {r["metric"]: float(r["value"]) for r in
        read_csv(os.path.join(ROOT, "models/anomaly_metrics.csv"))}
check("anomaly recall == 1.0 (all 6 hold-out incidents caught)",
      anom["recall"] == 1.0, f"got {anom['recall']}")
check("anomaly precision == 1.0 (0 false alarms)",
      anom["precision"] == 1.0, f"got {anom['precision']}")
check("anomaly F1 == 1.0", anom["f1"] == 1.0)

print("== drivers & benchmark ==")
drivers = read_csv(os.path.join(ROOT, "models/driver_importance.csv"))
check("top driver is backlog_ratio",
      drivers[0]["feature"] == "backlog_ratio",
      f"got {drivers[0]['feature']}")
check("second driver is staffing_ratio",
      drivers[1]["feature"] == "staffing_ratio")
bench = read_csv(os.path.join(ROOT, "models/benchmark.csv"))
check("benchmark has 3 product lines", len(bench) == 3)
check("Real Estate slowest (longest cycle time)",
      bench[0]["product_line"] == "Real Estate",
      f"got {bench[0]['product_line']}")
check("Private Credit fastest",
      bench[-1]["product_line"] == "Private Credit")

print("== honesty guards ==")
readme = open(os.path.join(ROOT, "README.md")).read()
check("README states data is synthetic", "synthetic" in readme.lower())
check("README states LoanIQ/ACBS unconfirmed",
      "LoanIQ" in readme and "CONFIRM" in readme)
report = open(os.path.join(ROOT, "reports",
                           "model_report.md")).read()
check("report mentions hold-out", "61" in report)

print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", FAILED)
    sys.exit(1)
print("ALL TESTS PASSED")
