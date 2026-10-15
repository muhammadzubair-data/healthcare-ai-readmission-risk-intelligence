import numpy as np
import pandas as pd

from readmit.models import fairness, monitoring


def _synthetic(n=30000, seed=0, bias_group="B"):
    rng = np.random.default_rng(seed)
    g = rng.choice(["A", "B", "C"], n)
    true_p = np.clip(rng.beta(2, 9, n), 0.01, 0.9)
    y = (rng.random(n) < np.where(g == bias_group, np.clip(true_p * 1.6, 0, 0.95), true_p)).astype(int)  # model under-predicts group B
    return pd.DataFrame({"g": g, "y": y, "p": true_p, "patient_id": np.arange(n)})


def test_fairness_flags_an_under_predicted_group_only():
    d = _synthetic()
    t = fairness.subgroup_table(d, "g", "y", "p", float(np.quantile(d["p"], 0.9)), n_boot=20)
    flags = t.set_index("group")["review_flags"]
    assert "calibration O/E" in flags["B"]
    assert flags["A"] == "" and flags["C"] == ""


def test_fairness_clean_data_has_no_flags_and_small_groups_skipped():
    d = _synthetic(bias_group="none")
    d = pd.concat([d, pd.DataFrame({"g": ["tiny"] * 10, "y": [0] * 10, "p": [0.1] * 10, "patient_id": range(10)})])
    t = fairness.subgroup_table(d, "g", "y", "p", float(np.quantile(d["p"], 0.9)), n_boot=20)
    assert "tiny" not in set(t["group"]) and (t["review_flags"] == "").all()


def test_equal_rate_recall_separates_base_rate_from_ranking_quality():
    rng = np.random.default_rng(3); n = 40000
    g = rng.choice(["low", "high"], n); p = np.where(g == "low", rng.beta(1, 25, n), rng.beta(3, 8, n))
    y = (rng.random(n) < p).astype(int)  # perfectly calibrated in both groups
    d = pd.DataFrame({"g": g, "y": y, "p": p, "patient_id": np.arange(n)})
    t = fairness.subgroup_table(d, "g", "y", "p", float(np.quantile(p, 0.9)), n_boot=10).set_index("group")
    assert t.loc["low", "recall_at_threshold"] < t.loc["high", "recall_at_threshold"] - 0.2   # same threshold: base-rate gap
    assert "within-group ranking gap" not in t.loc["low", "review_flags"] or abs(t.loc["low", "recall_equal_rate"] - t.loc["low", "recall_eq_overall"]) > 0.05


def test_observed_expected_and_funnel_alert():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.3, 4000)
    ok = monitoring.observed_expected((rng.random(4000) < p).astype(int), p)
    assert abs(ok["oe"] - 1) < 0.12 and abs(ok["z"]) < 3.5
    hi = monitoring.observed_expected((rng.random(4000) < np.clip(p * 1.4, 0, 1)).astype(int), p)
    assert hi["oe"] > 1.25 and hi["z"] > 4 and hi["oe_low"] > 1


def test_facility_oe_detects_planted_shift():
    rng = np.random.default_rng(2); rows = []
    for f, mult in (("F1", 1.0), ("F2", 1.0), ("F3", 1.35)):
        p = rng.uniform(0.05, 0.3, 3000)
        rows.append(pd.DataFrame({"facility_id": f, "risk": p, "readmit_30d": (rng.random(3000) < np.clip(p * mult, 0, 1)).astype(int)}))
    out = monitoring.facility_oe(pd.concat(rows)).set_index("facility_id")
    assert out.loc["F3", "status"] == "Higher than expected" and out.loc["F1", "status"] == "Within expected range"
