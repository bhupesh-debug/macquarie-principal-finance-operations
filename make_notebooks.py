"""Generate the 4 notebooks as valid nbformat-4 JSON.

Run from the project root:
    ~/workspace/quant-ipo-engine/.venv/bin/python make_notebooks.py

The notebooks are code templates that run the project's own src/
modules; they are not pre-executed (nbformat is not installed here),
and the README says so.
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(ROOT, "notebooks")


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [text]}


def code(text):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": [text]}


def nb(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}


NOTEBOOKS = {
    "01_eda.ipynb": [
        md("# 01 — Exploratory Data Analysis\n\nSynthetic loan-portfolio "
           "operations data: 3 product lines x 730 days, a 600-facility "
           "loan book, and a daily pipeline-health file with 25 injected "
           "data-quality incidents."),
        code("import sys\nsys.path.insert(0, '../src')\n"
             "import pandas as pd\n"
             "ops, health, fac = __import__('data_pipeline').load()\n"
             "print(ops.shape, health.shape, fac.shape)\n"
             "print(ops.groupby('product_line')"
             "['deals_received'].describe())"),
        code("import matplotlib.pyplot as plt\n"
             "tot = __import__('data_pipeline').daily_totals(ops)\n"
             "tot.plot(x='date', y=['deals_received', 'deals_closed'], "
             "figsize=(10, 4), title='Daily pipeline volume')\n"
             "plt.show()"),
        md("**What to look for:** the weekly cycle (weekend troughs), "
           "the slow growth trend, and the gap between received and "
           "closed — that gap is the backlog this project manages."),
    ],
    "02_feature_engineering.ipynb": [
        md("# 02 — Feature Engineering\n\nForecasting features use only "
           "information available at forecast time. Same-day operational "
           "features are EXCLUDED from the forecaster (leakage) — they "
           "belong to the driver model."),
        code("import sys\nsys.path.insert(0, '../src')\n"
             "import data_pipeline as dp\n"
             "import feature_engineering as fe\n"
             "ops, _, _ = dp.load()\n"
             "tot = dp.daily_totals(ops)\n"
             "feat = fe.add_forecast_features(tot, 'deals_closed')\n"
             "print(feat[fe.FORECAST_FEATURES].head())\n"
             "train, test, cut = fe.time_split(\n"
             "    feat.dropna().reset_index(drop=True), 61)\n"
             "print('hold-out starts:', cut.date())"),
        md("**Leakage check:** no same-day `staffing_ratio`, "
           "`backlog_ratio` or `exception_rate_pct` in "
           "`FORECAST_FEATURES`. If one appears, the forecast is "
           "invalid — fix it before modelling."),
    ],
    "03_modeling.ipynb": [
        md("# 03 — Modeling\n\nFive forecasters on the same features and "
           "the same 61-day hold-out: seasonal naive, Holt-Winters "
           "(additive, hand-rolled), linear regression, random forest, "
           "gradient boosting."),
        code("import sys\nsys.path.insert(0, '../src')\n"
             "import data_pipeline as dp\n"
             "import feature_engineering as fe\n"
             "import models as md\n"
             "ops, _, _ = dp.load()\n"
             "tot = dp.daily_totals(ops)\n"
             "feat = fe.add_forecast_features(tot, 'deals_closed')\n"
             "train, test, cut = fe.time_split(\n"
             "    feat.dropna().reset_index(drop=True), 61)\n"
             "fitted = md.fit_sklearn_models(\n"
             "    train[fe.FORECAST_FEATURES], train['deals_closed'])\n"
             "for name, m in fitted.items():\n"
             "    print(name, md.metrics(test['deals_closed'],\n"
             "          m.predict(test[fe.FORECAST_FEATURES])))"),
        md("**Expected result:** Random Forest wins on deals closed "
           "(MAPE ~22.6%); Linear wins on cycle time (MAPE ~4.1%). "
           "Holt-Winters trails — MAPE is inflated by weekend troughs "
           "where closings fall to single digits."),
    ],
    "04_evaluation.ipynb": [
        md("# 04 — Evaluation\n\nAnomaly detection (z-score rule, "
           "|z| > 4) against the 6 injected incidents in the hold-out, "
           "and the driver model (gradient boosting + permutation "
           "importance) for cycle time."),
        code("import sys\nsys.path.insert(0, '../src')\n"
             "import data_pipeline as dp\n"
             "import feature_engineering as fe\n"
             "import anomaly as an\n"
             "_, health, _ = dp.load()\n"
             "htr, hte, _ = fe.time_split(health, 61)\n"
             "flagged, _ = an.zscore_flag(htr, hte)\n"
             "print(an.anomaly_metrics(hte['is_anomaly'].values, "
             "flagged))"),
        code("import data_pipeline as dp\n"
             "import feature_engineering as fe\n"
             "import anomaly as an\n"
             "ops, _, _ = dp.load()\n"
             "tot = dp.daily_totals(ops)\n"
             "tr, te, _ = fe.time_split(tot, 61)\n"
             "_, drivers, _ = an.driver_model(tr, te, "
             "fe.DRIVER_FEATURES)\n"
             "print(drivers.to_string(index=False))"),
        md("**What to conclude:** backlog pressure dominates cycle time "
           "(importance ~1.98), staffing coverage is second (~0.50). "
           "The operational fix is staffing and backlog clearance — "
           "not more modelling."),
    ],
}

if __name__ == "__main__":
    os.makedirs(NB, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        with open(os.path.join(NB, name), "w") as fh:
            json.dump(nb(cells), fh, indent=1)
        print("notebook:", name)
