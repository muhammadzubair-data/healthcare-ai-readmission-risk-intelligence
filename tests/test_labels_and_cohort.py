import numpy as np
import pandas as pd

from readmit.config import CohortConfig
from readmit.features import build as fb


def _enc(rows):
    df = pd.DataFrame(rows, columns=["patient_id", "admit_ts", "discharge_ts", "is_planned_admission", "discharge_disposition"])
    for c in ("admit_ts", "discharge_ts"):
        df[c] = pd.to_datetime(df[c], format="ISO8601")
    return df


END = pd.Timestamp("2025-12-31")


def test_boundary_30_days_inclusive_31_excluded():
    e = _enc([("a", "2024-01-01", "2024-01-05", 0, "HOME"), ("a", "2024-02-04", "2024-02-06", 0, "HOME"),   # exactly 30.0d later -> event
              ("b", "2024-01-01", "2024-01-05", 0, "HOME"), ("b", "2024-02-05 00:01", "2024-02-07", 0, "HOME")])  # 31d -> no event
    out = fb.build_labels(e, CohortConfig(), END).set_index(["patient_id", "admit_ts"])
    assert out.loc[("a", pd.Timestamp("2024-01-01")), "readmit_30d"] == 1
    assert out.loc[("b", pd.Timestamp("2024-01-01")), "readmit_30d"] == 0


def test_planned_readmissions_are_not_events_but_later_unplanned_are():
    e = _enc([("a", "2024-01-01", "2024-01-05", 0, "HOME"), ("a", "2024-01-12", "2024-01-14", 1, "HOME"),
              ("c", "2024-01-01", "2024-01-05", 0, "HOME"), ("c", "2024-01-12", "2024-01-14", 1, "HOME"), ("c", "2024-01-20", "2024-01-22", 0, "HOME")])
    out = fb.build_labels(e, CohortConfig(), END).set_index(["patient_id", "admit_ts"])
    assert out.loc[("a", pd.Timestamp("2024-01-01")), "readmit_30d"] == 0
    assert out.loc[("c", pd.Timestamp("2024-01-01")), "readmit_30d"] == 1  # unplanned return after the planned one, <=30d of index


def test_unobservable_followup_is_nan_not_zero():
    e = _enc([("a", "2025-12-10", "2025-12-15", 0, "HOME"), ("b", "2025-06-01", "2025-06-03", 0, "HOME")])
    out = fb.build_labels(e, CohortConfig(), END).set_index("patient_id")
    assert np.isnan(out.loc["a", "readmit_30d"]) and not out.loc["a", "label_observable"]
    assert out.loc["b", "readmit_30d"] == 0 and out.loc["b", "label_observable"]


def test_exclusions_applied():
    e = _enc([("a", "2024-01-01", "2024-01-05", 0, "HOSPICE"), ("b", "2024-01-01", "2024-01-05", 0, "EXPIRED"),
              ("c", "2024-01-01", "2024-01-05", 0, "TRANSFER_ACUTE"), ("d", "2024-01-01", "2024-01-05", 0, "HOME")])
    out = fb.build_labels(e, CohortConfig(), END).set_index("patient_id")
    assert out["in_cohort"].to_dict() == {"a": False, "b": False, "c": False, "d": True}


def test_label_matches_simulator_truth(feat_raw_ids, sim):
    """The label re-derived from the encounter table must agree with how the data were generated."""
    _, truth = sim
    f = fb.modeling_cohort(feat_raw_ids).merge(truth[["encounter_id", "true_event_recorded"]], on="encounter_id")
    agree = (f["readmit_30d"] == f["true_event_recorded"]).mean()
    assert agree > 0.995, agree


def test_modeling_cohort_has_only_observable_labelled_rows(feat):
    mc = fb.modeling_cohort(feat)
    assert mc["readmit_30d"].notna().all() and mc["label_observable"].all() and mc["in_cohort"].all()
    live = fb.scoring_cohort(feat)
    assert live["readmit_30d"].isna().all() and len(live) > 0
    assert 0.08 < mc["readmit_30d"].mean() < 0.25
