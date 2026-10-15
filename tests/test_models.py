import numpy as np
import pandas as pd
import pytest

from readmit.features import build as fb
from readmit.models import evaluate as ev
from readmit.models import explain, train as tr, uncertainty


@pytest.fixture(scope="module")
def cohort(feat):
    return fb.modeling_cohort(feat)


@pytest.fixture(scope="module")
def splits(cohort, cfg):
    return tr.make_splits(cohort, cfg)


@pytest.fixture(scope="module")
def bundle(cohort, splits, cfg):
    return tr.train_all(cohort, splits, cfg, log=lambda *_: None)


def test_split_integrity(cohort, splits, cfg):
    tr_, va, te, un = (cohort[splits[k]] for k in ("train", "val", "test", "test_unseen"))
    # temporal ordering with a gap that exceeds the 30-day label horizon
    assert tr_["discharge_ts"].max() < va["discharge_ts"].min() <= va["discharge_ts"].max() < te["discharge_ts"].min()
    assert te["discharge_ts"].min() - tr_["discharge_ts"].max() > pd.Timedelta(days=cfg.cohort.followup_days)
    # patient-level holdout: unseen patients never appear in train/val
    assert not (set(un["patient_id"]) & (set(tr_["patient_id"]) | set(va["patient_id"])))
    assert not (set(un["patient_id"]) & set(te["patient_id"]))
    # masks are disjoint
    total = sum(splits[k].astype(int) for k in ("train", "val", "test", "test_unseen"))
    assert total.max() == 1
    frac = len(set(cohort.loc[splits["holdout_patient_flag"], "patient_id"])) / cohort["patient_id"].nunique()
    assert abs(frac - cfg.split.patient_holdout_frac) < 0.03


def test_models_beat_chance_and_baseline_is_weaker(bundle, cohort, splits):
    te = cohort[splits["test"]]
    P = bundle.predict(te)
    aucs = {k: ev.core_metrics(te["readmit_30d"], v)["auroc"] for k, v in P.items()}
    assert min(aucs.values()) > 0.58
    assert aucs["logistic_regression"] > aucs["lace_style"] - 0.01
    assert aucs["lightgbm_calibrated"] > 0.62


def test_probabilities_valid_and_calibration_reasonable(bundle, cohort, splits):
    te = cohort[splits["test"]]
    for k, p in bundle.predict(te).items():
        assert np.all((p >= 0) & (p <= 1)), k
    m = ev.core_metrics(te["readmit_30d"], bundle.predict_final(te))
    assert 0.7 < m["observed_over_expected"] < 1.4 and m["ece"] < 0.06


def test_isotonic_monotone_and_tiers_ordered(bundle):
    x = np.linspace(0, 1, 200)
    assert np.all(np.diff(bundle.isotonic.predict(x)) >= -1e-12)
    lo, hi = bundle.tier_cutoffs
    assert 0 < lo < hi < 1
    tiers = uncertainty.assign_tier(np.array([lo - 1e-3, lo, hi - 1e-3, hi]), bundle.tier_cutoffs)
    assert tiers.tolist() == ["Standard", "Elevated", "Elevated", "High"]


def test_shap_exact_and_additive(bundle, cohort, splits):
    te = cohort[splits["test"]].head(150)
    assert explain.explainer_sanity(bundle, te, 150) < 1e-5
    contrib, base = bundle.shap_contributions(te)
    raw = bundle.predict(te)["lightgbm"]
    recon = 1 / (1 + np.exp(-(contrib.sum(1) + base)))
    assert np.allclose(recon, raw, atol=1e-6), "SHAP values must add up to the model's prediction"


def test_uncertainty_ensemble_bands(bundle, cohort, splits):
    te = cohort[splits["test"]].head(300)
    X = tr.prep_frame(te, bundle.levels)
    m = uncertainty.ensemble_predict(bundle.ensemble, X, transform=bundle.isotonic.predict)
    tier = uncertainty.assign_tier(bundle.predict_final(te), bundle.tier_cutoffs)
    s = uncertainty.summarize_ensemble(m, tier, bundle.tier_cutoffs)
    assert (s["risk_p05"] <= s["risk_p95"]).all() and s["tier_stability"].between(0, 1).all()


def test_unseen_category_does_not_crash(bundle, cohort, splits):
    te = cohort[splits["test"]].head(20).copy()
    te["insurance"] = te["insurance"].astype(str).replace({"MEDICARE": "NEW_PAYER"})
    p = bundle.predict_final(te)
    assert np.isfinite(p).all()


# ---------------------------------------------------------------- metric unit tests
def test_recall_at_fraction_exact():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1]); p = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .05])
    assert ev.recall_at_fraction(y, p, 0.3) == pytest.approx(2 / 3)
    assert ev.precision_at_fraction(y, p, 0.3) == pytest.approx(2 / 3)
    assert ev.recall_at_fraction(y, p, 1.0) == 1.0


def test_calibration_metrics_on_perfect_and_miscalibrated_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, 40000); y = (rng.random(40000) < p).astype(int)
    s, i = ev.calibration_slope_intercept(y, p)
    assert abs(s - 1) < 0.08 and abs(i) < 0.08 and ev.expected_calibration_error(y, p) < 0.01
    s2, _ = ev.calibration_slope_intercept(y, np.clip(p * 0.5 + 0.2, 0, 1))  # compressed -> slope > 1
    assert s2 > 1.5
    assert ev.core_metrics(y, p)["observed_over_expected"] == pytest.approx(1, abs=0.03)


def test_wilson_interval_properties():
    lo, hi = ev.wilson_interval(10, 100)
    assert 0 < lo < 0.1 < hi < 1
    assert ev.wilson_interval(0, 50)[0] == pytest.approx(0, abs=1e-9)


def test_cluster_bootstrap_ci_covers_estimate_and_resamples_whole_patients():
    rng = np.random.default_rng(1)
    n_pat = 800; g = np.repeat(np.arange(n_pat), rng.integers(1, 5, n_pat))
    score = rng.normal(size=len(g)); y = (rng.random(len(g)) < 1 / (1 + np.exp(-(score - 1.5)))).astype(int)
    out = ev.cluster_bootstrap_metrics(y, {"a": 1 / (1 + np.exp(-score)), "b": 1 / (1 + np.exp(-rng.normal(size=len(g))))}, g, n_boot=60, seed=0, reference="b")
    lo, hi = out["ci"]["a"]["auroc"]
    assert lo < ev.core_metrics(y, 1 / (1 + np.exp(-score)))["auroc"] < hi
    assert out["paired_diff_vs_reference"]["a"]["auroc"][0] > 0
    order, starts, lens = ev._cluster_index(g)
    idx = ev.cluster_resample_indices(order, starts, lens, np.random.default_rng(0))
    counts = np.bincount(g[idx], minlength=n_pat)
    sizes = np.bincount(g)
    assert np.all((counts % np.maximum(sizes, 1)) == 0), "a patient's encounters must be resampled together"


def test_decision_curve_values():
    y = np.array([1, 0, 1, 0]); p = np.array([.9, .6, .4, .1])
    d = ev.decision_curve(y, p, np.array([0.5]))
    assert d["net_benefit"].iloc[0] == pytest.approx(1 / 4 - 1 / 4 * 1.0)  # tp=1, fp=1 at t=.5
    assert d["treat_all"].iloc[0] == pytest.approx(0.5 - 0.5 * 1.0)
