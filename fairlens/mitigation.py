"""Bias mitigation: one pre-processing method and one post-processing method."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import to_group_array


# ---------------------------------------------------------------------------
# 1) Pre-processing: Reweighing (Kamiran & Calders, 2012)
#    Give each (group, label) combination a weight so that group and label
#    become statistically independent in the training data.
#    w(g, y) = P(g) * P(y) / P(g, y)
# ---------------------------------------------------------------------------
def reweighing_weights(y, groups) -> np.ndarray:
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "g": to_group_array(groups)})
    n = len(df)
    p_g = df["g"].value_counts(normalize=True)
    p_y = df["y"].value_counts(normalize=True)
    p_gy = df.groupby(["g", "y"])["y"].transform("size") / n
    w = df["g"].map(p_g).to_numpy() * df["y"].map(p_y).to_numpy() / p_gy.to_numpy()
    return w.astype(float)


# ---------------------------------------------------------------------------
# 2) Post-processing: group-specific decision thresholds
#    Keep the trained model as is; only change the cut-off per group.
# ---------------------------------------------------------------------------
OBJECTIVES = {
    "demographic_parity": "Equalize how often each group is predicted positive",
    "equal_opportunity": "Equalize the true-positive rate of each group",
}


def fit_group_thresholds(
    y_true,
    scores,
    groups,
    y_pred_base,
    objective: str = "demographic_parity",
    min_group_size: int = 30,
    default_threshold: float = 0.5,
) -> dict:
    """Find one threshold per group on a *fit* set (never the final test set).

    Targets are taken from the baseline predictions: the overall positive rate
    (demographic parity) or the overall true-positive rate (equal opportunity).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred_base = np.asarray(y_pred_base).astype(int)
    groups = to_group_array(groups)

    target_rate = float(y_pred_base.mean())
    n_pos = int((y_true == 1).sum())
    target_tpr = float(((y_pred_base == 1) & (y_true == 1)).sum() / n_pos) if n_pos else float("nan")

    thresholds = {}
    for g in pd.unique(groups):
        m = groups == g
        thr = default_threshold
        if m.sum() >= min_group_size:
            if objective == "demographic_parity" and 0 < target_rate < 1:
                thr = float(np.quantile(scores[m], 1 - target_rate))
            elif objective == "equal_opportunity" and not np.isnan(target_tpr) and 0 < target_tpr < 1:
                pos_scores = scores[m & (y_true == 1)]
                if len(pos_scores) >= 5:
                    thr = float(np.quantile(pos_scores, 1 - target_tpr))
        thresholds[g] = thr
    return thresholds


def apply_group_thresholds(scores, groups, thresholds: dict, default_threshold: float = 0.5) -> np.ndarray:
    groups = to_group_array(groups)
    thr = np.array([thresholds.get(g, default_threshold) for g in groups], dtype=float)
    return (np.asarray(scores, dtype=float) >= thr).astype(int)
