"""Data pipeline: load, validate, aggregate. No modelling here.

Validation runs BEFORE any modelling: missing values, duplicates,
range checks, date gaps. A pipeline that models dirty loan data is
worse than no pipeline — this is the "ownership of data in loan
management systems" part of the JD, made executable.
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "data", "synthetic")
PROC = os.path.join(ROOT, "data", "processed")


def load():
    ops = pd.read_csv(os.path.join(SYN, "ops_daily.csv"),
                      parse_dates=["date"])
    health = pd.read_csv(
        os.path.join(SYN, "pipeline_health_daily.csv"),
        parse_dates=["date"])
    fac = pd.read_csv(os.path.join(SYN, "loan_facilities.csv"))
    return ops, health, fac


def validate(ops, health, fac):
    """Return a dict of validation findings; raise on fatal issues."""
    findings = {}
    findings["ops_missing"] = int(ops.isna().sum().sum())
    findings["health_missing"] = int(health.isna().sum().sum())
    findings["fac_missing"] = int(fac.isna().sum().sum())
    findings["ops_duplicates"] = int(
        ops.duplicated(subset=["date", "product_line"]).sum())
    findings["fac_duplicate_ids"] = int(
        fac.duplicated(subset=["facility_id"]).sum())
    findings["ops_date_gaps"] = int(
        len(pd.date_range(ops["date"].min(), ops["date"].max()))
        - ops["date"].nunique())
    findings["negative_amounts"] = int((fac["commitment_usd_m"] <= 0).sum())
    findings["rate_out_of_range"] = int(
        ((fac["rate_pct"] < 0) | (fac["rate_pct"] > 25)).sum())
    if findings["ops_missing"] or findings["health_missing"]:
        raise ValueError(f"Missing values found: {findings}")
    if findings["fac_duplicate_ids"]:
        raise ValueError("Duplicate facility IDs found")
    return findings


def daily_totals(ops):
    """Aggregate the three product lines to one daily portfolio series."""
    tot = (ops.groupby("date")
           .agg(deals_received=("deals_received", "sum"),
                deals_closed=("deals_closed", "sum"),
                deals_in_pipeline=("deals_in_pipeline", "sum"),
                avg_cycle_time_days=("avg_cycle_time_days", "mean"),
                approval_rate_pct=("approval_rate_pct", "mean"),
                exception_rate_pct=("exception_rate_pct", "mean"),
                avg_deal_size_usd_m=("avg_deal_size_usd_m", "mean"),
                staffing_ratio=("staffing_ratio", "mean"),
                backlog_ratio=("backlog_ratio", "mean"),
                syndicated_share=("syndicated_share", "mean"),
                system_uptime_pct=("system_uptime_pct", "mean"))
           .reset_index())
    return tot


if __name__ == "__main__":
    ops, health, fac = load()
    findings = validate(ops, health, fac)
    print("validation:", findings)
    os.makedirs(PROC, exist_ok=True)
    daily_totals(ops).to_csv(os.path.join(PROC, "ops_daily_total.csv"),
                             index=False)
    print("wrote data/processed/ops_daily_total.csv")
