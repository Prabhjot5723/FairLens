"""FairLens - Bias & Fairness Auditor for ML classifiers.   Run:  streamlit run app.py"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from fairlens import data_utils as du
from fairlens.metrics import METRIC_INFO, verdicts
from fairlens.mitigation import OBJECTIVES
from fairlens.models import MODEL_CHOICES, detect_text_columns
from fairlens.pipeline import (
    add_intersection, audit_attribute, comparison_table, mitigate_reweighing,
    mitigate_thresholds, run_prediction_audit, run_training_audit,
)
from fairlens.report import build_html_report

st.set_page_config(page_title="FairLens - ML Bias Auditor", page_icon="⚖️", layout="wide")

SRC_DEMO = "Demo: biased hiring data (synthetic)"
SRC_FRAUD = "FraudShield (EMSCAD) dataset"
SRC_CSV = "Upload my dataset (CSV)"
SRC_PRED = "Upload predictions (CSV)"
EMSCAD_PATH = Path(__file__).parent / "data" / "fake_job_postings.csv"


@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    return du.make_synthetic_hiring()


@st.cache_data(show_spinner="Preparing the EMSCAD dataset...")
def load_fraud(raw: pd.DataFrame) -> pd.DataFrame:
    return du.load_emscad(raw)


def status_style(col):
    palette = {"PASS": "background-color:#d9f2e0;color:#0a5c25", "WARN": "background-color:#fff0cc;color:#7a4b00",
               "FAIL": "background-color:#fddede;color:#8b1a1a"}
    return [palette.get(v, "") for v in col]


def grey_small_groups(row):
    return ["color:#999" if not row["reliable"] else "" for _ in row]


# ============================ SIDEBAR ============================
st.sidebar.title("⚖️ FairLens")
st.sidebar.caption("Audit any classifier for bias across groups.")
source = st.sidebar.radio("1. Data source", [SRC_DEMO, SRC_FRAUD, SRC_CSV, SRC_PRED])

df = None
defaults = {"target": None, "positive": None, "sens": [], "favorable": 1}

if source == SRC_DEMO:
    df = load_demo()
    defaults = {"target": "hired", "positive": "1", "sens": ["gender", "age_group"], "favorable": 1}
    st.sidebar.info("Synthetic hiring data with bias deliberately built in - a safe way to see the tool work.")
elif source == SRC_FRAUD:
    raw = None
    if EMSCAD_PATH.exists():
        raw = pd.read_csv(EMSCAD_PATH)
        st.sidebar.success("Loaded data/fake_job_postings.csv")
    else:
        up = st.sidebar.file_uploader("Upload fake_job_postings.csv (EMSCAD)", type="csv", key="emscad")
        if up is not None:
            raw = read_csv_bytes(up.getvalue())
        st.sidebar.caption("Tip: save the file as data/fake_job_postings.csv to skip uploading next time.")
    if raw is not None:
        try:
            df = load_fraud(raw)
        except ValueError as e:
            st.sidebar.error(str(e))
    defaults = {"target": "fraudulent", "positive": "1", "sens": du.FRAUDSHIELD_SENSITIVE_DEFAULT, "favorable": 0}
elif source in (SRC_CSV, SRC_PRED):
    up = st.sidebar.file_uploader("Upload a CSV", type="csv", key="csv_" + source)
    if up is not None:
        df = read_csv_bytes(up.getvalue())

cfg = None  # collected settings for the Run button
if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**2. Configure**")
    cols = list(df.columns)
    text_like = set(detect_text_columns(df))

    if source != SRC_PRED:
        t_idx = cols.index(defaults["target"]) if defaults["target"] in cols else len(cols) - 1
        target = st.sidebar.selectbox("Target column (what the model predicts)", cols, index=t_idx)
        vals = sorted(df[target].dropna().astype(str).unique())[:50]
        p_idx = vals.index(defaults["positive"]) if defaults["positive"] in vals else (len(vals) - 1 if vals else 0)
        positive = st.sidebar.selectbox("Positive class", vals, index=p_idx)
        label_cols = [target]
    else:
        y_true_col = st.sidebar.selectbox("True label column", cols, index=0)
        y_pred_col = st.sidebar.selectbox("Predicted label column", cols, index=min(1, len(cols) - 1))
        score_opts = ["(none)"] + cols
        score_col = st.sidebar.selectbox("Score / probability column (optional)", score_opts)
        vals = sorted(df[y_true_col].dropna().astype(str).unique())[:50]
        positive = st.sidebar.selectbox("Positive class", vals, index=len(vals) - 1 if vals else 0)
        label_cols = [y_true_col, y_pred_col] + ([] if score_col == "(none)" else [score_col])

    sens_opts = [c for c in cols if c not in label_cols and c not in text_like]
    sens_default = [c for c in defaults["sens"] if c in sens_opts]
    sens_cols = st.sidebar.multiselect("Sensitive attribute(s) to audit", sens_opts, default=sens_default)

    fav_choice = st.sidebar.radio(
        "Which prediction is the *favorable* outcome?",
        ["Predicted 1 (positive class)", "Predicted 0 (not positive)"],
        index=0 if defaults["favorable"] == 1 else 1,
        help="E.g. hiring: 1 = hired = favorable. Fraud detection: 1 = flagged as fraud = unfavorable.",
    )
    favorable = 1 if fav_choice.startswith("Predicted 1") else 0

    with st.sidebar.expander("Advanced settings"):
        min_group = st.slider("Minimum group size to compare", 5, 200, 30)
        tol = st.slider("Tolerance for gaps (PASS if <= this)", 0.02, 0.30, 0.10, 0.01)
        if source != SRC_PRED:
            model_name = st.selectbox("Model", MODEL_CHOICES)
            balance = st.checkbox("Balance classes", value=True)
            exclude_sens = st.checkbox("Remove sensitive columns from model inputs", value=False,
                                       help="Try it: bias usually survives because other columns act as proxies.")
            dec_thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)

    if st.sidebar.button("▶ Run audit", type="primary"):
        try:
            with st.spinner("Training and auditing..." if source != SRC_PRED else "Auditing..."):
                if source != SRC_PRED:
                    res = run_training_audit(df, target, positive, sens_cols, model_name, balance, exclude_sens,
                                             dec_thr, favorable, min_group)
                else:
                    res = run_prediction_audit(df, y_true_col, y_pred_col, positive, sens_cols,
                                               None if score_col == "(none)" else score_col, favorable, min_group)
            res["label"] = source
            st.session_state["res"] = res
            st.session_state["tol"] = tol
            st.session_state["mitigations"] = []
        except ValueError as e:
            st.sidebar.error(str(e))
        except Exception as e:  # unexpected - show it instead of a blank page
            st.sidebar.error(f"Something went wrong: {e}")

# ============================ MAIN ============================
st.title("⚖️ FairLens")
st.caption("Bias & fairness auditor for ML classifiers - measure it, see it, reduce it.")

res = st.session_state.get("res")
tab_audit, tab_fix, tab_report, tab_about = st.tabs(["📊 Audit", "🛠 Mitigate", "📄 Report", "ℹ️ About"])

with tab_about:
    st.subheader("What is this?")
    st.markdown(
        "A model can be accurate **overall** and still treat groups very differently - e.g. flag legitimate job postings "
        "from one country far more often than another. FairLens takes a classifier's results, splits them by group, "
        "measures the gaps with standard fairness metrics, and tests two ways of reducing them.\n\n"
        "**Quick start:** choose *Demo* in the sidebar -> press **Run audit** -> explore the Audit and Mitigate tabs."
    )
    st.subheader("Metrics")
    for key, (label, meaning) in METRIC_INFO.items():
        st.markdown(f"- **{label}** - {meaning}")
    st.subheader("Mitigation methods")
    st.markdown(
        "- **Reweighing (pre-processing):** retrain with sample weights so group membership and label become independent "
        "in the training data (Kamiran & Calders, 2012).\n"
        "- **Group thresholds (post-processing):** keep the model, but pick a separate decision cut-off per group, "
        "fitted on a validation set and judged on untouched test data.")
    st.subheader("Audit a model that is not built in")
    st.markdown("Export its predictions and upload them with the *Upload predictions (CSV)* source:")
    st.code(
        "out = X_test.copy()\nout['y_true']  = y_test\nout['y_pred']  = model.predict(X_test)\n"
        "out['y_score'] = model.predict_proba(X_test)[:, 1]   # optional, enables threshold mitigation\n"
        "out.to_csv('preds.csv', index=False)", language="python")
    st.subheader("Limitations")
    st.markdown(
        "- Fairness definitions can conflict - improving one metric can worsen another.\n"
        "- Small groups give noisy numbers; groups below the minimum size are excluded from gaps.\n"
        "- A metric gap shows a *disparity*, not proof of discrimination; context and domain knowledge decide what is acceptable.")

if res is None:
    with tab_audit:
        st.info("👈 Pick a data source, configure it in the sidebar and press **Run audit**.")
    with tab_fix:
        st.info("Run an audit first.")
    with tab_report:
        st.info("Run an audit first.")
    st.stop()

c, tol = res["config"], st.session_state.get("tol", 0.10)

# ------------------------------ AUDIT ------------------------------
with tab_audit:
    st.caption(f"Source: **{res['label']}** · {c['n_rows']:,} rows" + (f" · audited on the {c['n_test']:,}-row held-out test set" if res["mode"] == "train" else ""))
    perf = audit_attribute(res, res["audit"]["sens"].columns[0])["performance"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{perf['accuracy']:.3f}")
    m2.metric("Precision", f"{perf['precision']:.3f}")
    m3.metric("Recall", f"{perf['recall']:.3f}")
    m4.metric("F1", f"{perf['f1']:.3f}")

    attrs = list(res["audit"]["sens"].columns)
    if len(c["sensitive_cols"]) >= 2:
        with st.expander("➕ Intersectional audit (combine two attributes)"):
            ia, ib = st.columns(2)
            a1 = ia.selectbox("First attribute", c["sensitive_cols"], key="ia")
            a2 = ib.selectbox("Second attribute", [x for x in c["sensitive_cols"] if x != a1], key="ib")
            if st.button("Add combined attribute"):
                add_intersection(res, a1, a2)
                st.rerun()

    attr = st.selectbox("Sensitive attribute", attrs)
    out = audit_attribute(res, attr)
    gm, summ = out["group_metrics"], out["summary"]

    if not summ:
        st.warning("Fewer than two groups have enough samples to compare. Lower the minimum group size or pick another attribute.")
    else:
        v = verdicts(summ, tol)
        n_fail, n_warn = int((v.status == "FAIL").sum()), int((v.status == "WARN").sum())
        if n_fail:
            st.error(f"**{n_fail} metric(s) FAIL** and {n_warn} warn for '{attr}' - the model treats groups noticeably differently.")
        elif n_warn:
            st.warning(f"No failures, but **{n_warn} metric(s) show a warning** for '{attr}'.")
        else:
            st.success(f"All fairness checks pass for '{attr}' at tolerance {tol:.2f}.")
        st.dataframe(
            v[["metric", "value", "status", "meaning"]].style.format({"value": "{:.3f}"}, na_rep="n/a").apply(status_style, subset=["status"]),
            hide_index=True,
        )

        worst = gm[gm["reliable"]]
        if len(worst) >= 2:
            hi, lo = worst["selection_rate"].idxmax(), worst["selection_rate"].idxmin()
            st.markdown(f"**In plain words:** the favorable outcome is given to **{worst.loc[hi, 'selection_rate']:.1%}** of '{hi}' "
                        f"but only **{worst.loc[lo, 'selection_rate']:.1%}** of '{lo}'.")

    st.subheader("Group by group")
    ch1, ch2, ch3 = st.columns(3)
    ch1.caption("Selection rate (favorable outcome)")
    ch1.bar_chart(gm["selection_rate"])
    ch2.caption("True-positive rate (recall)")
    ch2.bar_chart(gm["tpr"])
    ch3.caption("False-positive rate")
    ch3.bar_chart(gm["fpr"])
    fmt = {k: "{:.3f}" for k in ["base_rate", "pred_positive_rate", "selection_rate", "accuracy", "tpr", "fpr", "fnr", "precision"]}
    st.dataframe(gm.style.format(fmt, na_rep="n/a").apply(grey_small_groups, axis=1))
    if (~gm["reliable"]).any():
        st.caption("Grey rows have fewer samples than the minimum group size and are left out of the gap calculations.")

# ------------------------------ MITIGATE ------------------------------
with tab_fix:
    can_thr = res["fit"] is not None and res["fit"].get("y_score") is not None
    can_rw = res["mode"] == "train"
    if not (can_thr or can_rw):
        st.info("This predictions file has no score column, so mitigation is unavailable. Upload predictions with a score/probability column to enable group thresholds.")
    else:
        methods = (["Group thresholds (post-processing)"] if can_thr else []) + (["Reweighing (pre-processing)"] if can_rw else [])
        mc1, mc2, mc3 = st.columns([2, 2, 2])
        method = mc1.selectbox("Method", methods)
        m_attr = mc2.selectbox("Attribute to fix", list(res["audit"]["sens"].columns), key="m_attr")
        objective = None
        if method.startswith("Group"):
            objective = mc3.selectbox("Goal", list(OBJECTIVES), format_func=lambda k: OBJECTIVES[k])
        if st.button("Apply mitigation", type="primary"):
            try:
                with st.spinner("Working..."):
                    m = mitigate_thresholds(res, m_attr, objective) if method.startswith("Group") else mitigate_reweighing(res, m_attr)
                lst = [x for x in st.session_state.get("mitigations", []) if (x["name"], x["attr"]) != (m["name"], m["attr"])]
                st.session_state["mitigations"] = lst + [m]
            except ValueError as e:
                st.error(str(e))

    for m in st.session_state.get("mitigations", []):
        with st.expander(f"{m['name']} - {m['attr']}", expanded=True):
            comp = comparison_table(m)
            st.dataframe(comp.set_index("metric").drop(columns="type").style.format("{:+.3f}", subset=["change"], na_rep="n/a")
                         .format("{:.3f}", subset=["before", "after"], na_rep="n/a"))
            gaps = comp[comp["metric"].isin(["Demographic parity diff", "Equal opportunity diff", "FPR diff", "Equalized odds diff"])]
            st.bar_chart(gaps.set_index("metric")[["before", "after"]])
            if "thresholds" in m:
                st.caption("Learned decision threshold per group")
                st.dataframe(pd.Series(m["thresholds"], name="threshold").to_frame().style.format("{:.3f}"))
            st.caption("Numbers are measured on data the mitigation did not see. Lower gaps with similar accuracy/recall = a good trade-off.")

# ------------------------------ REPORT ------------------------------
with tab_report:
    st.markdown("Download a self-contained report (open in a browser, then *Print -> Save as PDF* for your lab file).")
    html_report = build_html_report(res, tol, st.session_state.get("mitigations", []))
    st.download_button("⬇ Download HTML report", html_report, file_name="fairlens_report.html", mime="text/html")
    r_attr = st.selectbox("Export group table for", list(res["audit"]["sens"].columns), key="r_attr")
    st.download_button("⬇ Download group metrics (CSV)", audit_attribute(res, r_attr)["group_metrics"].to_csv().encode("utf-8"),
                       file_name=f"group_metrics_{r_attr}.csv", mime="text/csv")
