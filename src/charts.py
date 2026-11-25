"""Charts (Step 5). All charts read from models/*.csv or the processed
data — nothing is hand-drawn. Figures are described in the PDF with
the business insight, not just the technical finding.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS = os.path.join(ROOT, "reports", "charts")
MODELS = os.path.join(ROOT, "models")
PROC = os.path.join(ROOT, "data", "processed")

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
NAVY = "#1f4e79"
TEAL = "#2e9e6b"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, name), bbox_inches="tight")
    plt.close(fig)
    print("chart:", name)


def forecast_charts(tot, hw_fc, hw_dates, gb_pred, test):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(test["date"], test["deals_closed"], label="Actual",
                 color=NAVY)
    axes[0].plot(test["date"], gb_pred, label="Gradient boosting "
                 "(1-step)", color=TEAL, linestyle="--")
    axes[0].set_title("Deals closed — hold-out actual vs 1-step forecast")
    axes[0].legend()
    axes[1].plot(test["date"], test["deals_closed"], label="Actual",
                 color=NAVY)
    axes[1].plot(hw_dates, hw_fc, label="Holt-Winters recursive "
                 "(14-day-style multi-step)", color="#c0504d",
                 linestyle="--")
    axes[1].set_title("Deals closed — Holt-Winters recursive forecast "
                      "from end of training data")
    axes[1].legend()
    _save(fig, "forecast_vs_actual.png")


def model_comparison():
    m = pd.read_csv(os.path.join(MODELS, "metrics.csv"))
    m = m[m["target"] == "deals_closed"].sort_values("mape")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(m["model"], m["mape"], color=NAVY)
    ax.set_xlabel("MAPE on 61-day hold-out (%)")
    ax.set_title("Forecast error by model — deals closed")
    for i, v in enumerate(m["mape"]):
        ax.text(v + 0.05, i, f"{v:.2f}%", va="center")
    _save(fig, "model_comparison.png")


def feature_importance():
    d = pd.read_csv(os.path.join(MODELS, "driver_importance.csv"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(d["feature"][::-1], d["permutation_importance"][::-1],
            color=TEAL)
    ax.set_xlabel("Permutation importance (MAE increase, days)")
    ax.set_title("What drives cycle time? — permutation importance")
    _save(fig, "feature_importance.png")


def anomaly_timeline():
    h = pd.read_csv(os.path.join(
        ROOT, "data", "synthetic", "pipeline_health_daily.csv"),
        parse_dates=["date"])
    h = h[h["date"] >= "2025-11-01"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(h["date"], h["missing_rate_pct"], label="Missing rate %",
            color=NAVY)
    inc = h[h["is_anomaly"] == 1]
    ax.scatter(inc["date"], inc["missing_rate_pct"], color="#c0504d",
               s=60, zorder=5, label="Injected incident")
    ax.set_title("Pipeline-health timeline (hold-out) — missing rate "
                 "with injected incidents")
    ax.legend()
    _save(fig, "anomaly_timeline.png")


def benchmark():
    b = pd.read_csv(os.path.join(MODELS, "benchmark.csv"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(b))
    ax.bar(x, b["avg_cycle_time_days"], color=NAVY)
    ax.axhline(28, color="#c0504d", linestyle="--",
               label="28-day internal target")
    ax.set_xticks(x)
    ax.set_xticklabels(b["product_line"])
    ax.set_ylabel("Mean cycle time (days)")
    ax.set_title("Cycle-time benchmark by product line")
    ax.legend()
    _save(fig, "benchmark.png")


def correlation_heatmap():
    tot = pd.read_csv(os.path.join(PROC, "ops_daily_total.csv"))
    cols = ["deals_closed", "avg_cycle_time_days", "backlog_ratio",
            "staffing_ratio", "exception_rate_pct",
            "approval_rate_pct", "avg_deal_size_usd_m"]
    corr = tot[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=8)
    ax.set_title("KPI correlations — daily portfolio totals")
    fig.colorbar(im, ax=ax)
    _save(fig, "correlation_heatmap.png")


if __name__ == "__main__":
    os.makedirs(CHARTS, exist_ok=True)
    model_comparison()
    feature_importance()
    anomaly_timeline()
    benchmark()
    correlation_heatmap()
    print("charts done (forecast chart built from run_all.py)")
