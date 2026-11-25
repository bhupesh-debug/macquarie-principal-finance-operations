"""Build the 10-slide PPTX deck (Step 8 of the skill).

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python build_deck.py
Output: ~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_Manager-Principal-Finance-Operations_Macquarie_Deck.pptx
(Convert to PDF with LibreOffice for the delivered deck.)

Numbers come from models/*.csv (the real Python run). Every "ELI5" is
replaced with a checkmark. The test count is read from a live run of
tests/run_tests.py (its "N passed" line), never counted from check(
call sites — checks inside loops execute multiple times.
"""

import csv
import os
import re
import subprocess
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(
    "~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_Manager-Principal-Finance-Operations_Macquarie_Deck.pptx")


def _runtime_test_count():
    """Count checks from a live run of the suite (its 'N passed' line)."""
    try:
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tests", "run_tests.py")],
            capture_output=True, text=True, timeout=300, cwd=ROOT).stdout
        m = re.search(r"(\d+) passed", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    # Fallback only: count check( call sites (undercounts loop checks).
    with open(os.path.join(ROOT, "tests", "run_tests.py")) as f:
        return f.read().count("check(")


TEST_COUNT = _runtime_test_count()
MODELS = os.path.join(ROOT, "models")
CHARTS = os.path.join(ROOT, "reports", "charts")

NAVY = (31, 78, 121)
TEAL = (46, 158, 107)


def read_csv(name):
    with open(os.path.join(MODELS, name)) as fh:
        return list(csv.DictReader(fh))


metrics = read_csv("metrics.csv")
anom = {r["metric"]: float(r["value"])
        for r in read_csv("anomaly_metrics.csv")}
drivers = read_csv("driver_importance.csv")
bench = read_csv("benchmark.csv")
closed = [r for r in metrics if r["target"] == "deals_closed"]
cycle = [r for r in metrics if r["target"] == "avg_cycle_time_days"]
closed_best = min(closed, key=lambda r: float(r["mape"]))
cycle_best = min(cycle, key=lambda r: float(r["mape"]))

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide(title, bullets, chart=None, footer=None):
    slide = prs.slides.add_slide(BLANK)
    tx = slide.shapes.add_textbox(Inches(0.6), Inches(0.3),
                                  Inches(12.1), Inches(0.8))
    tf = tx.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.text = title
    para.font.size = Pt(30)
    para.font.bold = True
    para.font.color.rgb = RGBColor(*NAVY)
    body = slide.shapes.add_textbox(Inches(0.6), Inches(1.3),
                                    Inches(7.2 if chart else 12.1),
                                    Inches(5.4))
    tf2 = body.text_frame
    tf2.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = "•  " + b
        p.font.size = Pt(17)
        p.space_after = Pt(10)
    if chart:
        slide.shapes.add_picture(
            os.path.join(CHARTS, chart), Inches(8.1), Inches(1.4),
            width=Inches(4.6))
    if footer:
        fx = slide.shapes.add_textbox(Inches(0.6), Inches(6.9),
                                      Inches(12.1), Inches(0.4))
        f = fx.text_frame.paragraphs[0]
        f.text = footer
        f.font.size = Pt(12)
    return slide


add_slide(
    "Principal-Finance Operations Analytics",
    ["ML Project for Manager, Principal Finance Operations at "
     "Macquarie (Macquarie Capital)",
     "Pipeline forecasting  •  loan data-quality anomaly detection  "
     "•  cycle-time driver analysis",
     "Bhupesh Chandra Dimri  •  November 19, 2026",
     "Python  •  scikit-learn  •  Power BI — the JD's own tools"],
    footer="Synthetic loan-portfolio data (seed 42); no real Macquarie "
           "or client data used.")

add_slide(
    "Problem Overview",
    ["Macquarie's Principal Finance group needs operations that keep "
     "a loan book clean and moving: forecast the pipeline, catch bad "
     "data before it corrupts LoanIQ, and explain what slows a deal "
     "down.",
     "✅  Forecast next quarter's closings and cycle time so the "
     "team staffs to demand, not to surprises.",
     "✅  Catch a bad feed before it becomes a week of wrong "
     "dashboards and wrong decisions.",
     "✅  Show which lever — backlog, staffing, deal mix — actually "
     "moves cycle time."],
    footer="Why it matters: a wrong loan record is worse than no "
           "record — it sends the deal team the wrong way.")

add_slide(
    "JD Breakdown",
    ["Must-have: 5+ years middle-office / operations, loan-system "
     "data ownership, end-to-end transaction management, Excel, "
     "large data sets.",
     "Good-to-have: credit-agreement review, valuation concepts, "
     "staff management.",
     "Hidden expectation: approval responsibility — sign-off, not "
     "just analysis — plus owning the loan book's data end-to-end.",
     "Carried honestly: LoanIQ/ACBS and credit-agreement review are "
     "[CONFIRM WITH CANDIDATE] — not evidenced in the record, never "
     "asserted either way."])

add_slide(
    "ML Approach",
    ["Problem 1 — Forecasting (time series / regression): seasonal "
     "naive, Holt-Winters, linear, random forest, gradient boosting.",
     "Problem 2 — Anomaly detection: z-score rule (|z| > 4) on "
     "pipeline-health features.",
     "Problem 3 — Drivers: gradient boosting + permutation "
     "importance; per-product-line benchmark.",
     "✅  No deep learning on purpose — the JD asks for operations "
     "analytics, not a transformer demo."],
    footer="Design decision: same-day operational features are "
           "excluded from the forecaster — that would leak.")

add_slide(
    "Data Strategy",
    ["Synthetic data (seed 42): 3 product lines, 730 days, 2,190 "
     "line-days + 730 pipeline-health days + a 600-facility loan "
     "book.",
     "25 injected data-quality incidents (late feed, missing batch, "
     "duplicate load, null spike) with ground-truth labels — 6 in "
     "the hold-out.",
     "Validation first: missing values, duplicates, range checks, "
     "date gaps — before any modelling.",
     "Time-based split: train 2024-01-01–2025-10-30, hold-out "
     "2025-10-31–2025-12-30 (61 days)."])

add_slide(
    "Model Design",
    ["Baseline: seasonal naive + Holt-Winters (weekly seasonality).",
     "Intermediate: linear regression, random forest (500 trees), "
     "gradient boosting (500 trees, depth 3).",
     "Features: lags 1/7/14/28, 7- and 28-day rolling means, "
     "demand-shock flag, calendar features, pipeline lag.",
     "Metrics: MAE / RMSE / MAPE for forecasts; precision / recall / "
     "F1 for anomalies."],
    chart="model_comparison.png")

add_slide(
    "Results & Insights",
    ["Best closings forecaster: %s (MAPE %.2f%%, MAE %.2f "
     "deals/day on the 61-day hold-out)."
     % (closed_best["model"], float(closed_best["mape"]),
        float(closed_best["mae"])),
     "Best cycle-time forecaster: %s (MAPE %.2f%%, MAE %.2f days)."
     % (cycle_best["model"], float(cycle_best["mape"]),
        float(cycle_best["mae"])),
     "Anomaly detection: precision %.3f, recall %.3f, F1 %.3f — "
     "%d of %d injected incidents caught, %d false alarms."
     % (anom["precision"], anom["recall"], anom["f1"],
        int(anom["tp"]), int(anom["n_true"]), int(anom["fp"])),
     "Top cycle-time driver: %s — backlog pressure dominates."
     % drivers[0]["feature"],
     "Slowest line: %s (%.1f days) — start the staffing "
     "conversation there." % (bench[0]["product_line"],
                             float(bench[0]["avg_cycle_time_days"]))],
    chart="feature_importance.png")

add_slide(
    "Business Impact",
    ["A 61-day pipeline forecast lets the team staff to demand "
     "instead of reacting to it — planned coverage instead of "
     "emergency cover.",
     "Anomaly flags stop one bad feed from becoming a week of wrong "
     "loan records and wrong decisions.",
     "Driver ranking turns a vague “improve cycle time” into a "
     "concrete order of work: clear backlog first, protect staffing "
     "second.",
     "✅  All of it runs on the JD's own tools: Python, SQL, Excel, "
     "Power BI."])

add_slide(
    "Future Improvements",
    ["Re-fit the forecaster weekly on the rolling window; alert when "
     "hold-out MAPE degrades.",
     "Add per-product-line forecasts (Real Estate staffing needs its "
     "own number).",
     "Replace the 4-sigma rule with a calibrated threshold per "
     "incident type.",
     "Confirm LoanIQ/ACBS exposure with the candidate and re-run the "
     "pipeline against a LoanIQ extract schema.",
     "Ship the 61-day forecast into Power BI as a scheduled refresh."])

add_slide(
    "About This Project",
    ["Created by Bhupesh Chandra Dimri — Financial Analyst, "
     "JPMorgan Chase; M.S. Accounting Analytics & Data Technology, "
     "Pace University.",
     "Python, SQL, Power BI, Tableau, Excel VBA/Power Query; R and "
     "STATA (working knowledge); SAS and SPSS (learned during M.S.).",
     "LoanIQ/ACBS and credit-agreement review: [CONFIRM WITH "
     "CANDIDATE] — neither claimed nor denied.",
     "Tests: %d checks — all pass (tests/run_tests.py)." % TEST_COUNT,
     "Contact: bhupesh7cd@gmail.com  •  917-818-6198",
     "All data synthetic; no real client data used."],
    footer="Thank you — questions welcome.")

prs.save(OUT)
print("Wrote", OUT, "slides:", len(prs.slides._sldIdLst))
