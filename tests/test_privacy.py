import pandas as pd
import pytest

from readmit.data import privacy


def test_pseudonyms_deterministic_and_keyed():
    s = pd.Series(["P1", "P2", "P1"])
    a, b = privacy.pseudonymize(s, "PT_", "salt-a"), privacy.pseudonymize(s, "PT_", "salt-a")
    c = privacy.pseudonymize(s, "PT_", "salt-b")
    assert a.equals(b) and a.iloc[0] == a.iloc[2] and a.iloc[0] != a.iloc[1]
    assert not a.equals(c)
    assert a.str.startswith("PT_").all() and not a.str.contains("P1").any()


def test_date_shift_preserves_intervals_but_moves_dates():
    df = pd.DataFrame({"patient_id": ["a", "a", "b"], "t1": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
                       "t2": pd.to_datetime(["2022-01-05", "2022-02-09", "2022-03-04"])})
    out = privacy.shift_dates(df, "patient_id", ["t1", "t2"], salt="x")
    assert ((out["t2"] - out["t1"]) == (df["t2"] - df["t1"])).all()
    assert (out.loc[out.patient_id == "a", "t1"].diff().dropna() == pd.Timedelta(days=31)).all()
    assert (out["t1"] != df["t1"]).any()


def test_phi_screen_catches_columns_and_values():
    bad_cols = pd.DataFrame({"patient_name": ["x"], "ssn": ["1"], "value": [1]})
    assert {f["column"] for f in privacy.scan_for_phi(bad_cols)} >= {"patient_name", "ssn"}
    bad_vals = pd.DataFrame({"note": ["call 555-123-4567", "mail a@b.org", "123-45-6789", "fine"]})
    kinds = {f["detail"] for f in privacy.scan_for_phi(bad_vals)}
    assert kinds == {"phone", "email", "ssn"}
    assert privacy.scan_for_phi(pd.DataFrame({"facility_name": ["Harbor"], "x": [1]})) == []
    with pytest.raises(privacy.PHIDetectedError):
        privacy.enforce_no_phi({"t": bad_cols})


def test_top_coding_and_small_cells():
    assert list(privacy.top_code_age([40, 90, 97])) == [40, 90, 90]
    df = pd.DataFrame({"g": list("abc"), "n": [5, 11, 40], "rate": [0.2, 0.3, 0.1]})
    out = privacy.suppress_small_cells(df, "n", ["rate"])
    assert out["rate"].isna().tolist() == [True, False, False] and out["suppressed"].tolist() == [True, False, False]


def test_clean_tables_contain_no_raw_ids(cleaned, sim):
    t, _ = cleaned[0], None
    raw_ids = set(sim[0]["patients"]["patient_id"])
    assert not (set(t["patients"]["patient_id"]) & raw_ids)
    assert t["patients"]["patient_id"].str.startswith("PT_").all()
    assert t["encounters"]["encounter_id"].str.startswith("EN_").all()
    assert not any(c in t["patients"].columns for c in ("name", "address", "ssn", "zip"))
