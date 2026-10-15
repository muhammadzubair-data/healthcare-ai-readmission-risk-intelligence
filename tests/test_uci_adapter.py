"""The adapter is tested on a small, schema-faithful fixture built here (NOT real patient data)."""
import numpy as np
import pandas as pd
import pytest

from readmit import uci_adapter as ua


def fixture(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "encounter_id": np.arange(n), "patient_nbr": rng.integers(0, n // 2, n), "race": rng.choice(["Caucasian", "AfricanAmerican", "?"], n),
        "gender": rng.choice(["Male", "Female", "Unknown/Invalid"], n, p=[.48, .5, .02]), "age": rng.choice(["[40-50)", "[60-70)", "[70-80)"], n),
        "admission_type_id": rng.integers(1, 4, n), "discharge_disposition_id": rng.choice([1, 3, 6, 11, 13, 19], n, p=[.6, .1, .15, .05, .05, .05]),
        "admission_source_id": rng.integers(1, 8, n), "time_in_hospital": rng.integers(1, 15, n), "payer_code": rng.choice(["MC", "?", "HM"], n),
        "medical_specialty": rng.choice(["Cardiology", "?", "InternalMedicine"], n), "num_lab_procedures": rng.integers(1, 80, n), "num_procedures": rng.integers(0, 6, n),
        "num_medications": rng.integers(1, 30, n), "number_outpatient": rng.integers(0, 5, n), "number_emergency": rng.integers(0, 4, n), "number_inpatient": rng.integers(0, 5, n),
        "diag_1": rng.choice(["250.8", "428", "414", "?"], n), "number_diagnoses": rng.integers(1, 10, n), "max_glu_serum": rng.choice(["None", ">200"], n),
        "A1Cresult": rng.choice(["None", ">8"], n), "change": rng.choice(["No", "Ch"], n), "diabetesMed": rng.choice(["Yes", "No"], n),
        "insulin": rng.choice(["No", "Steady", "Up", "Down"], n), "metformin": rng.choice(["No", "Steady"], n)})
    p = 0.04 + 0.05 * d["number_inpatient"] + 0.008 * d["time_in_hospital"]
    d["readmitted"] = np.where(rng.random(n) < p, "<30", rng.choice([">30", "NO"], n))
    return d


def test_label_and_exclusions():
    raw = fixture()
    d = ua.prepare(raw.replace(["?", "Unknown/Invalid"], np.nan))
    assert set(d["readmit_30d"].unique()) <= {0, 1}
    assert not d["discharge_disposition_id"].isin(ua.EXCLUDED_DISPOSITIONS).any()
    # '>30' and 'NO' are negatives; only '<30' is the 30-day event
    src = raw[~raw["discharge_disposition_id"].isin(ua.EXCLUDED_DISPOSITIONS)]
    assert d["readmit_30d"].sum() == (src["readmitted"] == "<30").sum()
    assert d["age_mid"].isin([45.0, 65.0, 75.0]).all()


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        ua.prepare(pd.DataFrame({"x": [1]}))


def test_grouped_split_has_no_patient_overlap_and_benchmark_runs():
    d = ua.prepare(fixture().replace("?", np.nan))
    tr, va, te = ua.grouped_split(d, seed=1)
    g = d["patient_nbr"]
    assert not (set(g[tr]) & set(g[te])) and not (set(g[tr]) & set(g[va])) and not (set(g[va]) & set(g[te]))
    res = ua.benchmark(d, n_boot=10)
    assert res["split"].startswith("patient-grouped") and 0.5 < res["models"]["logistic_regression"]["auroc"] < 0.9


def test_cli_without_file_returns_error(capsys):
    assert ua.main([]) == 2
