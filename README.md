# Principal-Finance Operations Analytics

**ML/analytics project for Manager, Principal Finance Operations at Macquarie (Macquarie Capital, Principal Finance group)** — loan-portfolio operations analytics: transaction-pipeline forecasting, loan data-quality anomaly detection, and cycle-time driver analysis.

## Design decision, stated openly

This job description asks for middle-office operations, loan administration and transaction management — **not machine learning**. So this is an **analytics + forecasting project**, not a deep-learning demo. The differentiator is operational: data ownership, reconciliation/validation, and KPI tracking for a loan book — the exact language of the JD.

## The three problems

1. **Transaction pipeline forecasting** — forecast daily deals closed and average cycle time (days to close) on a 61-day hold-out. Best models: Random Forest (deals closed, MAPE 22.60%) and Linear regression (cycle time, MAPE 4.05%).
2. **Loan data-quality anomaly detection** — a z-score rule (|z| > 4) on daily pipeline-health features flags bad loan data before it reaches the loan system. Hold-out: precision 1.000, recall 1.000, F1 1.000 — all 6 injected incidents caught, 0 false alarms. (The injected incidents are deliberately large, operationally visible breaks — late feeds, missing batches, duplicate loads — so this measures catching obvious breaks, not subtle drift.)
3. **Cycle-time drivers & benchmarking** — gradient boosting + permutation importance. Backlog pressure dominates (importance 1.98), staffing coverage second (0.50). Real Estate closes slowest (38.7 days), Private Credit fastest (33.4 days).

## Data — synthetic

All data is synthetic (seed 42): 3 product lines × 730 days (2024-01-01 to 2025-12-30), a 600-facility loan book, and a daily pipeline-health file with 25 injected data-quality incidents (19 in training, 6 in the hold-out). **No real Macquarie, client or borrower data is used, and no real portfolio experience is claimed.**

## Honest limits

- **LoanIQ / ACBS: [CONFIRM WITH CANDIDATE]** — the JD requires LoanIQ, ACBS or an equivalent corporate loan management system. Neither appears in the candidate's verified master resume, so no LoanIQ/ACBS experience is claimed anywhere in this project. The pipeline is built on CSVs with the same schema discipline a loan system demands, but it is not LoanIQ.
- **Credit-agreement review: [CONFIRM WITH CANDIDATE]** — likewise not evidenced in the master resume; not claimed.
- **Holt-Winters underperforms here** (deals-closed MAPE 41.74% vs Random Forest 22.60%). The MAPE is inflated by weekend troughs — actual closings fall to single digits on Saturdays, so a small absolute miss becomes a huge percentage error (in-sample fitted MAPE is 17.29%). MAE tells the truer story (HW 4.56 vs RF 3.14 deals/day). Reported as-is.
- **The anomaly detector's perfect score** reflects deliberately large injected incidents. It proves the monitoring design works, not that it would catch subtle drift.

## How to run

```bash
# Python 3.12+, from the project root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py          # generate data, validate, model, charts, report
python tests/run_tests.py  # 30+ checks; exit 0 = all pass
python build_pdf.py        # 10-page report -> ~/workspace/your_files/
python build_deck.py       # 10-slide deck -> ~/workspace/your_files/
```

Deterministic: seed 42 everywhere, single-threaded Random Forest — two fresh runs produce byte-identical `models/*.csv`.

## Structure

```
├── README.md  requirements.txt  .gitignore  run_all.py
├── build_pdf.py  build_deck.py
├── src/            # data_generation, data_pipeline, feature_engineering,
│                   # models, anomaly, charts
├── tests/run_tests.py
├── data/synthetic/ # ops_daily.csv, pipeline_health_daily.csv, loan_facilities.csv
├── data/processed/ # ops_daily_total.csv
├── models/         # metrics.csv, anomaly_metrics.csv, driver_importance.csv, benchmark.csv
├── reports/        # model_report.md, charts/*.png
└── notebooks/      # 01_eda.ipynb, 02_feature_engineering.ipynb,
                    # 03_modeling.ipynb, 04_evaluation.ipynb
```

## Author

Bhupesh Chandra Dimri — Financial Analyst, JPMorgan Chase; M.S. Accounting Analytics & Data Technology, Pace University. R and STATA (working knowledge); SAS and SPSS (learned during M.S.).
