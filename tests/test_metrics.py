import numpy as np
import pandas as pd
import pytest

from fairlens.metrics import disparity_summary, group_metrics, verdicts


def _toy():
    rng = np.random.default_rng(1)
    n = 600
    g = rng.choice(["A", "B", "C"], n)
    y = rng.integers(0, 2, n)
    p = np.where(g == "A", rng.random(n) < 0.7, rng.random(n) < 0.3).astype(int)
    return y, p, g


def test_hand_computed_group():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([1, 0, 1, 0, 1, 0])
    g = np.array(["a"] * 6)
    r = group_metrics(y, p, g, min_group_size=1).loc["a"]
    assert r["tpr"] == pytest.approx(2 / 3)
    assert r["fpr"] == pytest.approx(1 / 3)
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["precision"] == pytest.approx(2 / 3)


def test_matches_fairlearn():
    fl = pytest.importorskip("fairlearn.metrics")
    y, p, g = _toy()
    s = disparity_summary(group_metrics(y, p, g, min_group_size=1))
    mf = fl.MetricFrame(metrics={"sel": fl.selection_rate, "tpr": fl.true_positive_rate, "fpr": fl.false_positive_rate},
                        y_true=y, y_pred=p, sensitive_features=g)
    d = mf.difference()
    assert s["demographic_parity_difference"] == pytest.approx(d["sel"])
    assert s["equal_opportunity_difference"] == pytest.approx(d["tpr"])
    assert s["fpr_difference"] == pytest.approx(d["fpr"])
    assert s["equalized_odds_difference"] == pytest.approx(fl.equalized_odds_difference(y, p, sensitive_features=g))
    assert s["demographic_parity_difference"] == pytest.approx(fl.demographic_parity_difference(y, p, sensitive_features=g))


def test_small_groups_excluded():
    y = np.array([1, 0] * 50 + [1, 0])
    p = np.array([1, 0] * 50 + [0, 1])
    g = np.array(["big"] * 100 + ["tiny"] * 2)
    gm = group_metrics(y, p, g, min_group_size=30)
    assert not gm.loc["tiny", "reliable"]
    assert disparity_summary(gm) == {}  # only one reliable group


def test_favorable_zero_flips_disparate_impact():
    y = np.array([0] * 100)
    p = np.array([1] * 20 + [0] * 80 + [1] * 40 + [0] * 60)
    g = np.array(["x"] * 100 + ["y"] * 100)
    y = np.zeros(200, dtype=int)
    di1 = disparity_summary(group_metrics(y, p, g, 1, favorable=1))["disparate_impact_ratio"]
    di0 = disparity_summary(group_metrics(y, p, g, 1, favorable=0))["disparate_impact_ratio"]
    assert di1 == pytest.approx(0.5)
    assert di0 == pytest.approx(0.6 / 0.8)


def test_verdict_rules():
    v = verdicts({"disparate_impact_ratio": 0.5, "demographic_parity_difference": 0.05, "equal_opportunity_difference": 0.15,
                  "fpr_difference": float("nan")}, tol=0.10).set_index("key")["status"]
    assert v["disparate_impact_ratio"] == "FAIL"
    assert v["demographic_parity_difference"] == "PASS"
    assert v["equal_opportunity_difference"] == "WARN"
    assert v["fpr_difference"] == "N/A"
