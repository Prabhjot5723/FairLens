"""Group fairness metrics, implemented from scratch with NumPy/Pandas.

Convention: label 1 = "positive" class, prediction 1 = "predicted positive".
`favorable` says which *prediction* is the good outcome for the person/item being
scored (e.g. in fraud detection, predicted 1 = flagged = unfavorable, so favorable=0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------- thresholds used for the PASS / WARN / FAIL verdicts ----------
DI_PASS = 0.80   # "four-fifths rule"
DI_WARN = 0.60


def _safe_div(a: float, b: float) -> float:
    return float("nan") if b == 0 else a / b


def to_group_array(groups) -> np.ndarray:
    """Turn any group column into an array of clean strings (NaN -> 'Missing')."""
    s = pd.Series(np.asarray(groups, dtype=object))
    s = s.where(s.notna(), "Missing")
    return s.astype(str).to_numpy()


def group_metrics(y_true, y_pred, groups, min_group_size: int = 30, favorable: int = 1) -> pd.DataFrame:
    """Per-group confusion-matrix based metrics. One row per group."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = to_group_array(groups)

    rows = []
    for g in pd.unique(groups):
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        n = int(m.sum())
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        pred_pos = float(yp.mean())
        rows.append(
            {
                "group": g,
                "n": n,
                "base_rate": float(yt.mean()),                 # actual share of positives
                "pred_positive_rate": pred_pos,                # share predicted positive
                "selection_rate": pred_pos if favorable == 1 else 1 - pred_pos,
                "accuracy": _safe_div(tp + tn, n),
                "tpr": _safe_div(tp, tp + fn),                 # recall / sensitivity
                "fpr": _safe_div(fp, fp + tn),
                "fnr": _safe_div(fn, fn + tp),
                "precision": _safe_div(tp, tp + fp),
            }
        )
    df = pd.DataFrame(rows).set_index("group").sort_values("n", ascending=False)
    df["reliable"] = df["n"] >= min_group_size
    return df


def _spread(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.max() - s.min()) if len(s) >= 2 else float("nan")


def _nanmax(*vals: float) -> float:
    v = [x for x in vals if not np.isnan(x)]
    return max(v) if v else float("nan")


def disparity_summary(gm: pd.DataFrame) -> dict:
    """Gaps *between groups* (only groups with enough samples are compared)."""
    r = gm[gm["reliable"]]
    if len(r) < 2:
        return {}
    sel = r["selection_rate"]
    di = float(sel.min() / sel.max()) if sel.max() > 0 else float("nan")
    eo = _spread(r["tpr"])
    fpr = _spread(r["fpr"])
    return {
        "demographic_parity_difference": _spread(r["pred_positive_rate"]),
        "disparate_impact_ratio": di,
        "equal_opportunity_difference": eo,
        "fpr_difference": fpr,
        "equalized_odds_difference": _nanmax(eo, fpr),
        "predictive_parity_difference": _spread(r["precision"]),
        "accuracy_gap": _spread(r["accuracy"]),
    }


# human readable names + one-line meaning, used by the UI and the report
METRIC_INFO = {
    "demographic_parity_difference": (
        "Demographic parity difference",
        "Gap in how often groups are predicted positive. 0 = every group treated alike.",
    ),
    "disparate_impact_ratio": (
        "Disparate impact ratio",
        "Lowest group favorable-rate / highest. Should be >= 0.80 (four-fifths rule).",
    ),
    "equal_opportunity_difference": (
        "Equal opportunity difference",
        "Gap in true-positive rate: are real positives caught equally across groups?",
    ),
    "fpr_difference": (
        "False-positive-rate difference",
        "Gap in false alarms: are some groups wrongly flagged more often?",
    ),
    "equalized_odds_difference": (
        "Equalized odds difference",
        "Worst of the TPR gap and the FPR gap.",
    ),
    "predictive_parity_difference": (
        "Predictive parity difference",
        "Gap in precision: is a positive prediction equally trustworthy for each group?",
    ),
    "accuracy_gap": (
        "Accuracy gap",
        "Gap in overall accuracy between the best and worst served group.",
    ),
}


def verdicts(summary: dict, tol: float = 0.10) -> pd.DataFrame:
    """PASS / WARN / FAIL for every metric.

    Differences: PASS <= tol, WARN <= 2*tol, FAIL above.
    Disparate impact ratio: PASS >= 0.8, WARN >= 0.6, FAIL below.
    """
    rows = []
    for key, (label, meaning) in METRIC_INFO.items():
        v = summary.get(key, float("nan"))
        if v is None or np.isnan(v):
            status = "N/A"
        elif key == "disparate_impact_ratio":
            status = "PASS" if v >= DI_PASS else "WARN" if v >= DI_WARN else "FAIL"
        else:
            status = "PASS" if v <= tol else "WARN" if v <= 2 * tol else "FAIL"
        rows.append({"metric": label, "key": key, "value": v, "status": status, "meaning": meaning})
    return pd.DataFrame(rows)


def overall_performance(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate(y_true, y_pred, groups, favorable: int = 1, min_group_size: int = 30) -> dict:
    """Everything in one call: overall performance + per-group table + gaps."""
    gm = group_metrics(y_true, y_pred, groups, min_group_size, favorable)
    return {
        "performance": overall_performance(y_true, y_pred),
        "group_metrics": gm,
        "summary": disparity_summary(gm),
    }
