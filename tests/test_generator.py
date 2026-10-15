import numpy as np
import pandas as pd

from readmit.config import SynthConfig
from readmit.synth.generator import simulate


def test_table_integrity(sim):
    t, truth = sim
    enc = t["encounters"]
    assert enc["encounter_id"].is_unique
    assert (enc["discharge_ts"] > enc["admit_ts"]).all()
    assert set(enc["patient_id"]) <= set(t["patients"]["patient_id"])
    assert set(t["diagnoses"]["encounter_id"]) <= set(enc["encounter_id"])
    assert set(t["labs"]["encounter_id"]) <= set(enc["encounter_id"])
    assert enc["admit_ts"].min() >= pd.Timestamp("2021-01-01")
    assert enc["admit_ts"].max() <= pd.Timestamp("2026-01-01")
    assert len(truth) == len(enc)


def test_every_encounter_has_exactly_one_principal_dx(sim):
    t, _ = sim
    p = t["diagnoses"][t["diagnoses"]["seq"] == 1].groupby("encounter_id").size()
    assert (p == 1).all() and len(p) == len(t["encounters"])


def test_one_encounter_at_a_time_per_patient(sim):
    enc = sim[0]["encounters"].sort_values(["patient_id", "admit_ts"])
    nxt = enc.groupby("patient_id")["admit_ts"].shift(-1)
    assert ((nxt.isna()) | (nxt >= enc["discharge_ts"])).all(), "overlapping stays for a patient"


def test_deterministic_given_seed():
    a = simulate(SynthConfig(n_patients=500, seed=3))
    b = simulate(SynthConfig(n_patients=500, seed=3))
    c = simulate(SynthConfig(n_patients=500, seed=4))
    pd.testing.assert_frame_equal(a["encounters"], b["encounters"])
    assert not a["encounters"].equals(c["encounters"])


def test_plausible_marginals(sim):
    t, truth = sim
    rate = truth["true_event_recorded"].mean()
    assert 0.10 < rate < 0.22
    los = (t["encounters"]["discharge_ts"] - t["encounters"]["admit_ts"]).dt.total_seconds() / 86400
    assert 1.0 < los.mean() < 9.0
    assert t["facilities"]["facility_id"].nunique() == 10


def test_truth_is_separate_from_feature_tables(sim):
    t, truth = sim
    for name, df in t.items():
        assert "true_risk" not in df.columns, name
    assert "true_risk" in truth.columns


def test_hidden_out_of_network_readmissions_exist(sim, cfg):
    _, truth = sim
    assert truth["true_event_any"].sum() > truth["true_event_recorded"].sum()
