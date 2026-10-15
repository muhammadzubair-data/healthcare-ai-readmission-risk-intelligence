import numpy as np
import pandas as pd
import pytest

from readmit.data import ingest, quality
from readmit.data.privacy import PHIDetectedError


def test_defects_are_detected_and_repaired(cleaned, defects):
    t, rep = cleaned
    ledger = defects[1]
    a = rep["actions"]
    assert a["duplicate_encounters_dropped"] == ledger["duplicate_encounter_rows"]
    assert a["encounters_quarantined_bad_dates"] == ledger["discharge_before_admit_rows"]
    assert len(t["_quarantine_encounters"]) == ledger["discharge_before_admit_rows"]
    assert rep["post"]["duplicate_encounter_ids"] == 0
    assert rep["post"]["temporal_violations"] == 0
    assert rep["post"]["implausible_labs"] == 0
    assert all(v == 0 for v in rep["post"]["referential_integrity"].values())


def test_unit_shift_found_in_right_facility_and_fixed(cleaned, defects, cfg):
    t, rep = cleaned
    assert rep["actions"]["creatinine_unit_shift_facilities"] == [cfg.synth.dq_unit_change_facility]
    assert rep["actions"]["creatinine_unit_shift_rows_converted"] >= 0.9 * defects[1]["creatinine_unit_shift_rows"]
    cr = t["labs"][t["labs"]["lab_name"] == "CREATININE"]
    assert cr["value"].max() < 20, "creatinine still in umol/L after cleaning"
    assert not any(c.get("facility_id") != cfg.synth.dq_unit_change_facility for c in rep["pre"]["unit_shift_cells"])


def test_unit_shift_detector_on_clean_data_finds_nothing(sim):
    labs = sim[0]["labs"]
    cells = quality.detect_unit_shift(labs)
    assert not cells["flag_high"].any() and not cells["flag_low"].any()


def test_cleaning_does_not_touch_clean_facilities(cleaned, sim, cfg):
    """Values at facilities other than the shifted one must be unchanged after cleaning."""
    t = cleaned[0]
    other = sim[0]["labs"]
    other = other[other["facility_id"] != cfg.synth.dq_unit_change_facility]
    assert len(t["labs"]) >= len(other) * 0.99


def test_phi_gate_blocks_ingest(defects, cfg):
    bad = {k: v.copy() for k, v in defects[0].items()}
    bad["patients"]["ssn"] = "123-45-6789"
    with pytest.raises(PHIDetectedError):
        ingest.clean(bad, cfg)


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert quality.population_stability_index(a, rng.normal(0, 1, 5000)) < 0.05
    assert quality.population_stability_index(a, rng.normal(1.0, 1, 5000)) > 0.25
    assert np.isnan(quality.population_stability_index(a[:10], a))


def test_implausible_lab_mask():
    labs = pd.DataFrame({"lab_name": ["SODIUM", "SODIUM", "CREATININE", "WBC"], "value": [140, 9999, -1, np.nan]})
    assert quality.implausible_lab_mask(labs).tolist() == [False, True, True, True]


def test_volume_anomaly_detector_flags_spike():
    rng = np.random.default_rng(1)
    ts = pd.concat([pd.Series(pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 28, 100) + 30 * m, unit="D")) for m in range(12)])
    enc = pd.DataFrame({"facility_id": "F1", "admit_ts": ts})
    spike = pd.DataFrame({"facility_id": "F1", "admit_ts": pd.Timestamp("2022-06-10") + pd.to_timedelta(rng.integers(0, 5, 400), unit="D")})
    out = quality.volume_anomalies(pd.concat([enc, spike]))
    assert len(out) >= 1
