# ⚖️ FairLens — Bias & Fairness Auditor for ML Models

A tool that checks whether a trained classifier treats different groups fairly, shows exactly where it doesn't,
and tests two ways of fixing it. Built for the ML Lab project (non-prediction, real-world use: responsible AI).

## What it does
1. **Audit** – splits a model's results by group (e.g. gender, country, employment type) and computes
   demographic parity, disparate impact (four-fifths rule), equal opportunity, false-positive-rate gap,
   equalized odds, predictive parity and accuracy gap, each with a PASS / WARN / FAIL verdict.
2. **Mitigate** – *Reweighing* (pre-processing, retrains the model) and *group-specific thresholds*
   (post-processing), with a before/after comparison that shows the fairness-vs-accuracy trade-off.
3. **Report** – downloads a self-contained HTML report (print to PDF) and CSV tables.

Data sources: synthetic biased hiring demo · **FraudShield (EMSCAD) dataset** · any CSV dataset · predictions exported from *any* model.

## Run it
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q            # optional: run the tests
```

## Project layout
```
app.py                  Streamlit dashboard (UI only)
fairlens/metrics.py     fairness metrics + PASS/WARN/FAIL rules (from scratch)
fairlens/mitigation.py  reweighing + group thresholds
fairlens/models.py      built-in model pipeline (TF-IDF + categorical + numeric)
fairlens/pipeline.py    glue: dataset -> train -> audit -> mitigate
fairlens/data_utils.py  synthetic demo data, EMSCAD loader, sensitive-column prep
fairlens/report.py      HTML report builder
tests/                  15 tests (metrics are cross-checked against Fairlearn)
data/                   put fake_job_postings.csv here for the FraudShield audit
```

## Auditing FraudShield
Copy `fake_job_postings.csv` (EMSCAD) into `data/`, choose **FraudShield (EMSCAD) dataset**, press *Run audit*.
Here a *false positive* = a genuine job posting wrongly flagged as fraud, so the question is:
does the detector wrongly flag genuine postings from some countries / employment types / posting styles more than others?

## Audit your own model
Export predictions and use **Upload predictions (CSV)**:
```python
out = X_test.copy()
out["y_true"] = y_test
out["y_pred"] = model.predict(X_test)
out["y_score"] = model.predict_proba(X_test)[:, 1]   # optional, enables group thresholds
out.to_csv("preds.csv", index=False)
```
A ready-made example is in `data/sample_predictions.csv`.

## Honest limitations
Fairness definitions can conflict; a metric gap shows a disparity, not proof of discrimination; very small groups give noisy numbers.
