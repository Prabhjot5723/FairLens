import numpy as np
import pandas as pd
import pytest

from fairlens.data_utils import FRAUDSHIELD_SENSITIVE_DEFAULT, load_emscad, make_synthetic_hiring, prepare_sensitive
from fairlens.mitigation import reweighing_weights
from fairlens.pipeline import (add_intersection, audit_attribute, comparison_table, mitigate_reweighing,
                               mitigate_thresholds, run_prediction_audit, run_training_audit)
from fairlens.report import build_html_report
from tests.helpers import make_mock_emscad


@pytest.fixture(scope="module")
def demo_res():
    return run_training_audit(make_synthetic_hiring(4000), "hired", "1", ["gender", "age_group"])


def test_demo_bias_is_detected(demo_res):
    s = audit_attribute(demo_res, "gender")["summary"]
    assert s["disparate_impact_ratio"] < 0.8          # the planted bias is visible
    assert s["demographic_parity_difference"] > 0.10


def test_reweighing_weights_make_groups_independent():
    df = make_synthetic_hiring(4000)
    w = reweighing_weights(df["hired"], df["gender"])
    d = df.assign(w=w)
    rate = d.groupby("gender").apply(lambda x: np.average(x["hired"], weights=x["w"]), include_groups=False)
    assert rate.max() - rate.min() < 1e-9               # weighted hire-rate equal across groups


def test_threshold_mitigation_reduces_gap(demo_res):
    m = mitigate_thresholds(demo_res, "gender", "demographic_parity")
    assert m["after"]["summary"]["demographic_parity_difference"] < m["before"]["summary"]["demographic_parity_difference"]
    assert not comparison_table(m).empty


def test_equal_opportunity_thresholds_reduce_tpr_gap(demo_res):
    m = mitigate_thresholds(demo_res, "gender", "equal_opportunity")
    assert m["after"]["summary"]["equal_opportunity_difference"] <= m["before"]["summary"]["equal_opportunity_difference"] + 0.02


def test_reweighing_runs_and_reduces_gap(demo_res):
    m = mitigate_reweighing(demo_res, "gender")
    assert m["after"]["summary"]["demographic_parity_difference"] < m["before"]["summary"]["demographic_parity_difference"]


def test_intersection_and_report(demo_res):
    name = add_intersection(demo_res, "gender", "age_group")
    assert name in demo_res["audit"]["sens"].columns
    html = build_html_report(demo_res, 0.1, [mitigate_thresholds(demo_res, "gender")])
    assert "Disparate impact ratio" in html and "gender x age_group" in html


def test_prediction_mode():
    rng = np.random.default_rng(3)
    n = 2000
    g = rng.choice(["m", "f"], n)
    y = rng.integers(0, 2, n)
    score = np.clip(0.5 * y + rng.normal(0.25, 0.2, n) - 0.1 * (g == "f"), 0, 1)
    df = pd.DataFrame({"y": y, "pred": (score > 0.5).astype(int), "score": score, "sex": g})
    res = run_prediction_audit(df, "y", "pred", "1", ["sex"], "score")
    assert audit_attribute(res, "sex")["summary"]
    assert mitigate_thresholds(res, "sex")["after"]["summary"]
    no_score = run_prediction_audit(df, "y", "pred", "1", ["sex"])
    assert no_score["fit"] is None
    with pytest.raises(ValueError):
        run_prediction_audit(df, "y", "score", "1", ["sex"])  # score passed as predicted label


def test_fraudshield_preset_runs():
    d = load_emscad(make_mock_emscad())
    res = run_training_audit(d, "fraudulent", "1", FRAUDSHIELD_SENSITIVE_DEFAULT, favorable=0, min_group_size=15)
    out = audit_attribute(res, "country")
    assert not out["group_metrics"].empty
    assert mitigate_thresholds(res, "country")["thresholds"]


def test_emscad_wrong_file_message():
    with pytest.raises(ValueError, match="missing columns"):
        load_emscad(pd.DataFrame({"a": [1]}))


def test_prepare_sensitive_bins_numbers():
    s = prepare_sensitive(pd.DataFrame({"age": np.arange(100)}), ["age"])
    assert s["age"].nunique() == 4
