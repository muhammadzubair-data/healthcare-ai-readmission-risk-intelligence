"""Leakage guards: features at discharge may not depend on anything that happens afterwards."""
import numpy as np
import pandas as pd

from readmit.features import build as fb


def _mini():
    pat = pd.DataFrame({"patient_id": ["p"], "birth_year": [1950], "sex": ["F"], "insurance": ["MEDICARE"], "ses_quintile": [3], "death_ts": [pd.NaT]})
    enc = pd.DataFrame({
        "encounter_id": ["e1", "e2"], "patient_id": ["p", "p"], "facility_id": ["F01", "F01"],
        "admit_ts": pd.to_datetime(["2024-01-01", "2024-03-01"]), "discharge_ts": pd.to_datetime(["2024-01-05", "2024-03-04"]),
        "admission_type": ["EMERGENCY"] * 2, "admit_source": ["ED"] * 2, "discharge_disposition": ["HOME"] * 2,
        "principal_dx": ["I50.9", "J44.1"], "icu_stay": [0, 0], "n_procedures": [0, 0], "is_planned_admission": [0, 0]})
    dx = pd.DataFrame({"encounter_id": ["e1", "e1", "e2"], "seq": [1, 2, 1], "icd10": ["I50.9", "E11.9", "J44.1"]})
    meds = pd.DataFrame({"encounter_id": ["e1", "e2"], "drug_class": ["DIURETIC", "OTHER"], "n_drugs": [1, 3]})
    labs = pd.DataFrame({"encounter_id": ["e1", "e2"], "patient_id": ["p", "p"], "facility_id": ["F01"] * 2, "lab_name": ["SODIUM", "SODIUM"],
                         "collected_ts": pd.to_datetime(["2024-01-04", "2024-03-03"]), "value": [133.0, 140.0], "unit": ["mmol/L"] * 2})
    ed = pd.DataFrame({"ed_visit_id": ["v1"], "patient_id": ["p"], "facility_id": ["F01"], "visit_ts": pd.to_datetime(["2023-12-20"])})
    return dict(patients=pat, encounters=enc, diagnoses=dx, medications=meds, labs=labs, ed_visits=ed, facilities=pd.DataFrame())


def _build(t):
    from readmit.config import CohortConfig
    return fb.build_features(t, CohortConfig(), pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")).set_index("encounter_id")


def test_history_features_are_correct_on_handmade_case():
    f = _build(_mini())
    assert f.loc["e1", "prior_admissions_12m"] == 0 and f.loc["e1", "ed_visits_prior_6m"] == 1
    assert f.loc["e2", "prior_admissions_12m"] == 1 and f.loc["e2", "prior_admissions_6m"] == 1
    assert f.loc["e2", "days_since_last_discharge"] == 56 and f.loc["e2", "is_readmission_of_prior"] == 0
    assert f.loc["e1", "cm_diabetes"] == 1 and f.loc["e2", "cm_chf"] == 1  # CHF only known from the *prior* admission's principal dx
    assert f.loc["e1", "cm_chf"] == 0, "principal dx of the index stay itself is not a comorbidity"
    assert f.loc["e1", "lab_sodium_last"] == 133.0 and f.loc["e1", "med_diuretic"] == 1


def test_adding_future_events_changes_no_earlier_feature():
    base = _build(_mini())
    t = _mini()
    t["encounters"] = pd.concat([t["encounters"], pd.DataFrame({
        "encounter_id": ["e3"], "patient_id": ["p"], "facility_id": ["F01"], "admit_ts": pd.to_datetime(["2024-03-20"]), "discharge_ts": pd.to_datetime(["2024-03-25"]),
        "admission_type": ["EMERGENCY"], "admit_source": ["ED"], "discharge_disposition": ["HOME"], "principal_dx": ["N17.9"], "icu_stay": [1], "n_procedures": [2], "is_planned_admission": [0]})])
    t["diagnoses"] = pd.concat([t["diagnoses"], pd.DataFrame({"encounter_id": ["e3", "e3"], "seq": [1, 2], "icd10": ["N17.9", "N18.30"]})])
    t["labs"] = pd.concat([t["labs"], pd.DataFrame({"encounter_id": ["e3", "e2"], "patient_id": ["p", "p"], "facility_id": ["F01"] * 2, "lab_name": ["SODIUM", "SODIUM"],
                                                     "collected_ts": pd.to_datetime(["2024-03-21", "2024-03-10"]), "value": [120.0, 99.0], "unit": ["mmol/L"] * 2})])  # e2 lab AFTER its discharge
    t["ed_visits"] = pd.concat([t["ed_visits"], pd.DataFrame({"ed_visit_id": ["v2"], "patient_id": ["p"], "facility_id": ["F01"], "visit_ts": pd.to_datetime(["2024-03-10"])})])
    t["medications"] = pd.concat([t["medications"], pd.DataFrame({"encounter_id": ["e3"], "drug_class": ["OPIOID"], "n_drugs": [1]})])
    new = _build(t)
    cols = fb.FEATURE_COLUMNS
    pd.testing.assert_frame_equal(base.loc[["e1", "e2"], cols].astype(str), new.loc[["e1", "e2"], cols].astype(str))
    assert new.loc["e2", "readmit_30d"] == 1 and base.loc["e2", "readmit_30d"] == 0  # only the LABEL may see the future


def test_no_forbidden_columns_are_features():
    assert not (set(fb.FEATURE_COLUMNS) & fb.FORBIDDEN_FEATURES)
    assert len(set(fb.FEATURE_COLUMNS)) == len(fb.FEATURE_COLUMNS)
    assert set(fb.FEATURE_COLUMNS) <= set(fb.FEATURE_DICTIONARY)


def test_features_do_not_encode_the_label(feat):
    mc = fb.modeling_cohort(feat)
    corr = mc[fb.NUMERIC_FEATURES].apply(lambda s: np.corrcoef(s.fillna(s.median()), mc["readmit_30d"])[0, 1]).abs()
    assert corr.max() < 0.35, corr.sort_values().tail(3)


def test_label_shuffle_removes_signal(feat):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    mc = fb.modeling_cohort(feat).dropna(subset=fb.NUMERIC_FEATURES, how="all")
    X = mc[fb.NUMERIC_FEATURES].fillna(0).to_numpy(); y = mc["readmit_30d"].to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
    ys = rng.permutation(y)
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    auc = roc_auc_score(ys[idx[cut:]], LogisticRegression(max_iter=500).fit(Xs[idx[:cut]], ys[idx[:cut]]).predict_proba(Xs[idx[cut:]])[:, 1])
    assert abs(auc - 0.5) < 0.03
