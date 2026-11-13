"""Orchestration: turn a dataset (or a predictions file) into an audit-ready result bundle,
and run the two mitigations on that bundle. The Streamlit app is a thin layer over this."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_utils import prepare_sensitive
from .metrics import evaluate
from .mitigation import apply_group_thresholds, fit_group_thresholds, reweighing_weights
from .models import build_pipeline, fit_pipeline, predict_scores, three_way_split


def _binary(series: pd.Series, positive) -> np.ndarray:
    return (series.astype(str) == str(positive)).to_numpy().astype(int)


# ---------------------------------------------------------------------------
# Mode A: we get a dataset -> we train the model -> we audit it
# ---------------------------------------------------------------------------
def run_training_audit(
    df: pd.DataFrame,
    target: str,
    positive,
    sensitive_cols: list[str],
    model_name: str = "Logistic Regression",
    balance: bool = True,
    exclude_sensitive: bool = False,
    decision_threshold: float = 0.5,
    favorable: int = 1,
    min_group_size: int = 30,
    seed: int = 42,
) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    if not sensitive_cols:
        raise ValueError("Pick at least one sensitive attribute to audit.")
    if target in sensitive_cols:
        raise ValueError("The target column cannot also be a sensitive attribute.")

    df = df.reset_index(drop=True)
    y = _binary(df[target], positive)
    if y.min() == y.max():
        raise ValueError("The target has only one class after choosing the positive label - nothing to learn.")

    feature_cols = [c for c in df.columns if c != target]
    if exclude_sensitive:
        feature_cols = [c for c in feature_cols if c not in sensitive_cols]
    if not feature_cols:
        raise ValueError("No feature columns left to train on.")

    X = df[feature_cols]
    sens = prepare_sensitive(df, sensitive_cols)
    tr, va, te = three_way_split(y, seed)

    pipe = build_pipeline(X.iloc[tr], model_name, balance, seed)
    fit_pipeline(pipe, X.iloc[tr], y[tr])
    s_va, s_te = predict_scores(pipe, X.iloc[va]), predict_scores(pipe, X.iloc[te])

    def pack(idx, scores, with_X=False):
        d = {
            "y_true": y[idx],
            "y_score": scores,
            "y_pred": (scores >= decision_threshold).astype(int),
            "sens": sens.iloc[idx].reset_index(drop=True),
        }
        if with_X:
            d["X"] = X.iloc[idx]
        return d

    test = pack(te, s_te, with_X=True)
    return {
        "mode": "train",
        "config": {
            "target": target, "positive": positive, "sensitive_cols": list(sensitive_cols),
            "model_name": model_name, "balance": balance, "exclude_sensitive": exclude_sensitive,
            "decision_threshold": decision_threshold, "favorable": favorable,
            "min_group_size": min_group_size, "seed": seed,
            "n_rows": len(df), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        },
        "audit": test,
        "eval": test,
        "fit": pack(va, s_va),
        "train": {"X": X.iloc[tr], "y": y[tr], "sens": sens.iloc[tr].reset_index(drop=True)},
    }


# ---------------------------------------------------------------------------
# Mode B: we get predictions from ANY model -> we only audit (+ thresholds if scores)
# ---------------------------------------------------------------------------
def run_prediction_audit(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    positive,
    sensitive_cols: list[str],
    y_score_col: str | None = None,
    favorable: int = 1,
    min_group_size: int = 30,
    seed: int = 42,
) -> dict:
    if not sensitive_cols:
        raise ValueError("Pick at least one sensitive attribute to audit.")
    needed = [y_true_col, y_pred_col] + ([y_score_col] if y_score_col else [])
    if len(set(needed)) != len(needed):
        raise ValueError("True label, predicted label and score must be three different columns.")
    if set(sensitive_cols) & set(needed):
        raise ValueError("A sensitive attribute cannot be the label/prediction/score column.")

    df = df.dropna(subset=needed).reset_index(drop=True)
    if df[y_pred_col].nunique() > 10:
        raise ValueError(f"'{y_pred_col}' has many distinct values - it looks like a score, not a predicted class.")

    y_true, y_pred = _binary(df[y_true_col], positive), _binary(df[y_pred_col], positive)
    if y_true.min() == y_true.max():
        raise ValueError("The true label has only one class - nothing to audit.")
    score = pd.to_numeric(df[y_score_col], errors="coerce").to_numpy() if y_score_col else None
    if score is not None and np.isnan(score).any():
        raise ValueError(f"'{y_score_col}' must be numeric.")
    sens = prepare_sensitive(df, sensitive_cols)

    full = {"y_true": y_true, "y_pred": y_pred, "y_score": score, "sens": sens}
    res = {
        "mode": "predictions",
        "config": {
            "y_true_col": y_true_col, "y_pred_col": y_pred_col, "y_score_col": y_score_col,
            "positive": positive, "sensitive_cols": list(sensitive_cols), "favorable": favorable,
            "min_group_size": min_group_size, "seed": seed, "n_rows": len(df),
        },
        "audit": full, "fit": None, "eval": full, "train": None,
    }
    if score is not None:  # split in half: fit thresholds on one half, judge them on the other
        idx = np.arange(len(df))
        try:
            a, b = train_test_split(idx, test_size=0.5, stratify=y_true, random_state=seed)
        except ValueError:
            a, b = train_test_split(idx, test_size=0.5, random_state=seed)

        def part(i):
            return {"y_true": y_true[i], "y_pred": y_pred[i], "y_score": score[i], "sens": sens.iloc[i].reset_index(drop=True)}

        res["fit"], res["eval"] = part(a), part(b)
    return res


# ---------------------------------------------------------------------------
# Audit helper used by the UI and the report
# ---------------------------------------------------------------------------
def audit_attribute(res: dict, attr: str) -> dict:
    a, c = res["audit"], res["config"]
    return evaluate(a["y_true"], a["y_pred"], a["sens"][attr], c["favorable"], c["min_group_size"])


def add_intersection(res: dict, attr_a: str, attr_b: str) -> str:
    """Create a combined attribute 'a x b' in every part of the result bundle."""
    name = f"{attr_a} x {attr_b}"
    for part in ("audit", "fit", "eval", "train"):
        d = res.get(part)
        if d is not None and name not in d["sens"].columns:
            d["sens"][name] = d["sens"][attr_a].astype(str) + " | " + d["sens"][attr_b].astype(str)
    return name


# ---------------------------------------------------------------------------
# Mitigations
# ---------------------------------------------------------------------------
def mitigate_thresholds(res: dict, attr: str, objective: str = "demographic_parity") -> dict:
    fit, ev, c = res["fit"], res["eval"], res["config"]
    if fit is None or fit.get("y_score") is None:
        raise ValueError("Threshold mitigation needs prediction scores.")
    thr = fit_group_thresholds(
        fit["y_true"], fit["y_score"], fit["sens"][attr], fit["y_pred"], objective, c["min_group_size"]
    )
    new_pred = apply_group_thresholds(ev["y_score"], ev["sens"][attr], thr)
    before = evaluate(ev["y_true"], ev["y_pred"], ev["sens"][attr], c["favorable"], c["min_group_size"])
    after = evaluate(ev["y_true"], new_pred, ev["sens"][attr], c["favorable"], c["min_group_size"])
    return {"name": f"Group thresholds ({objective.replace('_', ' ')})", "attr": attr,
            "thresholds": thr, "before": before, "after": after}


def mitigate_reweighing(res: dict, attr: str) -> dict:
    if res["mode"] != "train":
        raise ValueError("Reweighing retrains the model, so it needs a dataset (not a predictions file).")
    c, tr, ev = res["config"], res["train"], res["eval"]
    w = reweighing_weights(tr["y"], tr["sens"][attr])
    pipe = build_pipeline(tr["X"], c["model_name"], c["balance"], c["seed"])
    fit_pipeline(pipe, tr["X"], tr["y"], sample_weight=w)
    scores = predict_scores(pipe, ev["X"])
    new_pred = (scores >= c["decision_threshold"]).astype(int)
    before = evaluate(ev["y_true"], ev["y_pred"], ev["sens"][attr], c["favorable"], c["min_group_size"])
    after = evaluate(ev["y_true"], new_pred, ev["sens"][attr], c["favorable"], c["min_group_size"])
    return {"name": "Reweighing (retrained model)", "attr": attr, "weights": w, "before": before, "after": after}


def comparison_table(m: dict) -> pd.DataFrame:
    """Before vs after, performance and fairness side by side."""
    rows = []
    for k, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]:
        rows.append((label, "performance", m["before"]["performance"][k], m["after"]["performance"][k]))
    names = {
        "demographic_parity_difference": "Demographic parity diff",
        "disparate_impact_ratio": "Disparate impact ratio",
        "equal_opportunity_difference": "Equal opportunity diff",
        "fpr_difference": "FPR diff",
        "equalized_odds_difference": "Equalized odds diff",
    }
    for k, label in names.items():
        rows.append((label, "fairness", m["before"]["summary"].get(k, np.nan), m["after"]["summary"].get(k, np.nan)))
    df = pd.DataFrame(rows, columns=["metric", "type", "before", "after"])
    df["change"] = df["after"] - df["before"]
    return df
