"""Generate synthetic loan-portfolio operations data (seed 42).

SYNTHETIC BY CONSTRUCTION — no real Macquarie, client or borrower data.
Three product lines (private credit, private equity, real estate) x 730
days (2024-01-01 to 2025-12-30), plus a facility-level loan book and a
daily pipeline-health file with 25 injected data-quality incidents.

Design: cycle time is driven by backlog pressure and staffing coverage,
which is exactly what the driver model in Problem 3 must recover.
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

LINES = ["Private Credit", "Private Equity", "Real Estate"]
START = pd.Timestamp("2024-01-01")
DAYS = 730
dates = pd.date_range(START, periods=DAYS, freq="D")

# ---------------------------------------------------------------- ops daily
rows = []
for line in LINES:
    base_volume = {"Private Credit": 9.0, "Private Equity": 5.0,
                   "Real Estate": 6.0}[line]
    base_size = {"Private Credit": 42.0, "Private Equity": 68.0,
                 "Real Estate": 55.0}[line]
    # Line-specific cycle-time base: real-estate deals close slowest
    # (title / diligence), private credit fastest.
    base_cycle = {"Private Credit": 19.0, "Private Equity": 22.0,
                  "Real Estate": 24.0}[line]
    for i, d in enumerate(dates):
        dow = d.dayofweek
        weekend = 1 if dow >= 5 else 0
        # Weekly seasonality + slow growth + noise; weekends ~60% down
        seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * i / 7.0)
        trend = 1.0 + 0.0004 * i
        vol = base_volume * seasonal * trend
        vol *= 0.4 if weekend else 1.0
        received = max(0, int(round(vol + rng.normal(0, 1.6))))
        # Staffing ratio dips on some weeks (holidays, leave) -> backlog
        staffing_ratio = np.clip(
            1.0 + rng.normal(0, 0.08) - (0.12 if i % 97 < 5 else 0.0),
            0.70, 1.15)
        backlog_ratio = np.clip(
            0.55 + 0.35 * (1.0 - staffing_ratio)
            + 0.10 * np.sin(2 * np.pi * i / 61.0)
            + rng.normal(0, 0.07), 0.15, 1.35)
        backlog_open = int(round(received * backlog_ratio * 3.2))
        # Cycle time (days to close) driven by backlog + staffing
        cycle = np.clip(
            base_cycle + 26.0 * backlog_ratio
            - 14.0 * (staffing_ratio - 1.0)
            + rng.normal(0, 2.2), 9.0, 68.0)
        closed = max(0, int(round(received * 0.82
                                  + rng.normal(0, 1.4))))
        approval = np.clip(88.0 - 9.0 * backlog_ratio
                           + rng.normal(0, 2.0), 62.0, 98.0)
        exceptions = np.clip(2.0 + 6.5 * backlog_ratio
                             + rng.normal(0, 0.8), 0.5, 18.0)
        size = max(5.0, base_size + rng.normal(0, 9.0))
        syndicated_share = np.clip(
            0.45 + rng.normal(0, 0.08), 0.15, 0.80)
        rows.append({
            "date": d, "product_line": line,
            "deals_received": received, "deals_closed": closed,
            "deals_in_pipeline": backlog_open,
            "avg_cycle_time_days": round(cycle, 1),
            "approval_rate_pct": round(approval, 1),
            "exception_rate_pct": round(exceptions, 2),
            "avg_deal_size_usd_m": round(size, 1),
            "staffing_ratio": round(staffing_ratio, 3),
            "backlog_ratio": round(backlog_ratio, 3),
            "syndicated_share": round(syndicated_share, 3),
            "system_uptime_pct": round(
                np.clip(99.6 + rng.normal(0, 0.25), 97.5, 100.0), 2),
        })
ops = pd.DataFrame(rows)

# ------------------------------------------------------- pipeline health
health_rows = []
# 25 injected incidents: 19 in the training window, 6 in the 61-day
# hold-out (2025-10-31 onward) so the detector is actually tested.
cut_date = pd.Timestamp("2025-10-31")
train_dates = dates[dates < cut_date]
holdout_dates = dates[dates >= cut_date]
incident_dates = set(rng.choice(train_dates, size=19, replace=False))
incident_dates |= set(rng.choice(holdout_dates, size=6, replace=False))
incident_types = ["late_feed", "missing_batch", "duplicate_load",
                  "null_spike"]
for d in dates:
    is_inc = d in incident_dates
    itype = rng.choice(incident_types) if is_inc else "none"
    expected = int(rng.integers(4800, 6200))
    if is_inc and itype == "missing_batch":
        received_r = int(expected * rng.uniform(0.55, 0.75))
    else:
        received_r = int(expected * rng.uniform(0.97, 1.0))
    missing = (rng.uniform(6.0, 14.0) if is_inc and itype
               == "missing_batch" else rng.uniform(0.2, 1.1))
    dup = (rng.uniform(5.0, 11.0) if is_inc and itype
           == "duplicate_load" else rng.uniform(0.05, 0.6))
    nulls = (rng.uniform(4.0, 9.0) if is_inc and itype == "null_spike"
             else rng.uniform(0.1, 0.9))
    late = (int(rng.integers(180, 420)) if is_inc and itype == "late_feed"
            else int(rng.integers(2, 40)))
    stale = (rng.uniform(3.0, 8.0) if is_inc else
             rng.uniform(0.1, 0.8))
    health_rows.append({
        "date": d, "records_expected": expected,
        "records_received": received_r,
        "missing_rate_pct": round(missing, 2),
        "duplicate_rate_pct": round(dup, 2),
        "null_rate_pct": round(nulls, 2),
        "stale_record_pct": round(stale, 2),
        "late_feed_minutes": late,
        "is_anomaly": int(is_inc),
        "anomaly_type": itype,
    })
health = pd.DataFrame(health_rows)

# ------------------------------------------------------------ loan book
fac_rows = []
sectors = ["Real Estate", "Infrastructure", "Telecom & Media",
           "Healthcare", "Technology", "Leisure", "Energy"]
for i in range(600):
    line = rng.choice(LINES, p=[0.45, 0.25, 0.30])
    dtype = rng.choice(["Bilateral", "Syndicated"], p=[0.55, 0.45])
    commitment = round(float(rng.lognormal(np.log(45), 0.7)), 1)
    rate = round(float(rng.uniform(4.5, 11.5)), 2)
    maturity = int(rng.integers(2, 8))
    stage = rng.choice(["Received", "Analysis", "Approval", "Closing",
                        "Admin"], p=[0.12, 0.22, 0.18, 0.16, 0.32])
    days_stage = int(rng.integers(1, 45))
    fac_rows.append({
        "facility_id": f"PF-{1000 + i}",
        "product_line": line, "deal_type": dtype,
        "borrower_sector": rng.choice(sectors),
        "commitment_usd_m": commitment, "rate_pct": rate,
        "maturity_years": maturity, "stage": stage,
        "days_in_stage": days_stage,
        "status": rng.choice(["Active", "Active", "Active", "Closed",
                              "On Hold"], p=[0.7, 0.1, 0.05, 0.1, 0.05]),
    })
facilities = pd.DataFrame(fac_rows)

if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "synthetic")
    os.makedirs(out, exist_ok=True)
    ops.to_csv(os.path.join(out, "ops_daily.csv"), index=False)
    health.to_csv(os.path.join(out, "pipeline_health_daily.csv"),
                  index=False)
    facilities.to_csv(os.path.join(out, "loan_facilities.csv"),
                      index=False)
    print(f"ops_daily: {len(ops)} rows; health: {len(health)} rows; "
          f"facilities: {len(facilities)} rows; "
          f"incidents: {int(health['is_anomaly'].sum())}")
