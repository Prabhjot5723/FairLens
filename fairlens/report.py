"""Self-contained HTML audit report (download it, print it to PDF, attach it to the lab file)."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from .metrics import verdicts
from .pipeline import audit_attribute, comparison_table

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 16px;color:#1b1f23}
h1{margin-bottom:4px} h2{margin-top:36px;border-bottom:2px solid #e3e6ea;padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid #d6dae0;padding:6px 9px;text-align:right} th{background:#f3f5f8}
td:first-child,th:first-child{text-align:left}
.PASS{color:#0a7d33;font-weight:700}.WARN{color:#b36b00;font-weight:700}.FAIL{color:#c62828;font-weight:700}
.muted{color:#5b636e;font-size:13px}.box{background:#f7f9fb;border-left:4px solid #4a6cf7;padding:8px 12px;margin:10px 0}
"""


def _fmt(x):
    return "n/a" if pd.isna(x) else f"{x:.3f}"


def _table(df: pd.DataFrame) -> str:
    return df.to_html(float_format=_fmt, na_rep="n/a", border=0, escape=True)


def build_html_report(res: dict, tol: float = 0.10, mitigations: list | None = None, title: str = "FairLens - Bias & Fairness Audit") -> str:
    c = res["config"]
    parts = [f"<h1>{html.escape(title)}</h1>", f"<p class='muted'>Generated {datetime.now():%d %b %Y, %H:%M}</p>"]

    if res["mode"] == "train":
        setup = (
            f"Model: <b>{c['model_name']}</b> (class balancing: {c['balance']}), target <b>{html.escape(str(c['target']))}</b>, "
            f"decision threshold {c['decision_threshold']}. Rows: {c['n_rows']} "
            f"(train {c['n_train']} / validation {c['n_val']} / test {c['n_test']}). Audit numbers are on the held-out test set. "
            f"Sensitive attributes removed from model inputs: {c['exclude_sensitive']}."
        )
    else:
        setup = f"Audit of an uploaded predictions file ({c['n_rows']} rows): label <b>{html.escape(c['y_true_col'])}</b>, prediction <b>{html.escape(c['y_pred_col'])}</b>."
    parts.append(f"<div class='box'>{setup}<br>Minimum group size for comparison: {c['min_group_size']}. Tolerance for gaps: {tol:.2f}.</div>")

    for attr in res["audit"]["sens"].columns:
        out = audit_attribute(res, attr)
        v = verdicts(out["summary"], tol)
        parts.append(f"<h2>Attribute: {html.escape(attr)}</h2>")
        if v["status"].eq("N/A").all():
            parts.append("<p>Fewer than two groups have enough samples to compare.</p>")
        else:
            rows = "".join(
                f"<tr><td>{html.escape(r.metric)}</td><td>{_fmt(r.value)}</td><td class='{r.status}'>{r.status}</td>"
                f"<td style='text-align:left'>{html.escape(r.meaning)}</td></tr>"
                for r in v.itertuples()
            )
            parts.append(f"<table><tr><th>Metric</th><th>Value</th><th>Verdict</th><th>Meaning</th></tr>{rows}</table>")
        gm = out["group_metrics"].drop(columns=["fnr"])
        parts.append("<p class='muted'>Per-group results (groups below the minimum size are shown but excluded from the gaps).</p>")
        parts.append(_table(gm))

    if mitigations:
        parts.append("<h2>Mitigation experiments</h2>")
        for m in mitigations:
            parts.append(f"<h3>{html.escape(m['name'])} - attribute: {html.escape(m['attr'])}</h3>")
            parts.append(_table(comparison_table(m).set_index("metric")))

    parts.append("<p class='muted'>Fairness metrics describe disparities in a model's behaviour; they do not prove or rule out discrimination on their own. Different fairness definitions can conflict, so choose the one that fits the real-world harm you care about.</p>")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{''.join(parts)}</body></html>"
