"""Cohort definition, label construction and point-in-time feature engineering.

Prediction time = index discharge. For an index encounter *e* of patient *p*:

* encounter-level features use only data recorded up to e's discharge timestamp,
* history features use only encounters/ED visits that finished strictly before e's admission,
* the label looks *forward* from e's discharge and is never visible to any feature.

`tests/test_point_in_time.py` appends future events to the raw tables and asserts that the
features of earlier encounters do not change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..clinical import (
    COMORBIDITY_CATEGORIES,
    COMORBIDITY_PREFIXES,
    COMORBIDITY_WEIGHTS,
    HIGH_RISK_MED_CLASSES,
    LAB_NAMES,
    MED_CLASSES,
    comorbidity_category,
    dx_group,
    egfr_ckdepi_2021,
)
from ..config import CohortConfig

MAX_LAGS = 15  # prior-encounter lookback depth (patients with more are truncated, documented)
DAY = pd.Timedelta(days=1)

CATEGORICAL_FEATURES = ["sex", "insurance", "admission_type", "admit_source", "discharge_disposition", "dx_group"]

_LAB_COLS = [f"lab_{n.lower()}_last" for n in LAB_NAMES]
_LAB_CHG = [f"lab_{n.lower()}_change" for n in LAB_NAMES]
_LAB_MISS = [f"lab_{n.lower()}_missing" for n in LAB_NAMES]
_CM = [f"cm_{c}" for c in COMORBIDITY_CATEGORIES]
_MED = [f"med_{m.lower()}" for m in MED_CLASSES if m != "OTHER"]

NUMERIC_FEATURES = (
    ["age", "ses_quintile", "los_days", "icu_stay", "n_procedures", "acute_admission", "weekend_discharge",
     "discharge_month_sin", "discharge_month_cos", "n_secondary_dx", "charlson_score"]
    + _CM
    + _LAB_COLS + _LAB_CHG + _LAB_MISS + ["egfr_last", "n_lab_values"]
    + _MED + ["n_discharge_meds", "polypharmacy", "n_high_risk_med_classes"]
    + ["prior_admissions_6m", "prior_admissions_12m", "prior_los_days_12m", "prior_readmissions_12m",
       "ed_visits_prior_6m", "ed_visits_prior_12m", "days_since_last_discharge", "is_readmission_of_prior",
       "history_window_days", "n_prior_encounters"]
)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that must never be features (identifiers, outcome-derived, simulator truth, facility identity).
FORBIDDEN_FEATURES = {
    "readmit_30d", "days_to_readmit", "label_observable", "in_cohort", "died_within_30d",
    "true_risk", "true_event_any", "true_event_recorded", "patient_id", "encounter_id", "facility_id",
    "admit_ts", "discharge_ts", "principal_dx",
}

META_COLUMNS = [
    "encounter_id", "patient_id", "facility_id", "admit_ts", "discharge_ts", "principal_dx",
    "readmit_30d", "days_to_readmit", "label_observable", "in_cohort", "excluded_reason", "died_within_30d",
]

FEATURE_DICTIONARY: dict[str, tuple[str, str]] = {
    "age": ("demographic", "Age in years at admission, top-coded at 90"),
    "sex": ("demographic", "Recorded sex (F/M)"),
    "insurance": ("demographic", "Primary payer category"),
    "ses_quintile": ("demographic", "Synthetic area-deprivation quintile (1 = least, 5 = most deprived)"),
    "los_days": ("index stay", "Length of stay in days"),
    "icu_stay": ("index stay", "ICU stay during the encounter"),
    "n_procedures": ("index stay", "Number of procedures"),
    "acute_admission": ("index stay", "Emergency or urgent admission"),
    "admission_type": ("index stay", "EMERGENCY / URGENT / ELECTIVE"),
    "admit_source": ("index stay", "ED / clinic referral / transfer in"),
    "discharge_disposition": ("index stay", "Where the patient was discharged to"),
    "dx_group": ("index stay", "Principal-diagnosis group derived from ICD-10"),
    "weekend_discharge": ("index stay", "Discharged on Saturday or Sunday"),
    "discharge_month_sin": ("index stay", "Seasonality (sin of discharge month)"),
    "discharge_month_cos": ("index stay", "Seasonality (cos of discharge month)"),
    "n_secondary_dx": ("diagnoses", "Number of secondary diagnoses coded on the encounter"),
    "charlson_score": ("diagnoses", "Simplified Charlson-style score from current + prior coded diagnoses"),
    "n_lab_values": ("labs", "Number of lab results recorded up to discharge"),
    "egfr_last": ("labs", "CKD-EPI 2021 eGFR from the last creatinine before discharge"),
    "n_discharge_meds": ("medications", "Total distinct drugs on the discharge medication list"),
    "polypharmacy": ("medications", "10 or more discharge drugs"),
    "n_high_risk_med_classes": ("medications", "Count of high-risk drug classes at discharge"),
    "prior_admissions_6m": ("history", "Completed admissions in the 6 months before this admission"),
    "prior_admissions_12m": ("history", "Completed admissions in the 12 months before this admission"),
    "prior_los_days_12m": ("history", "Total inpatient days in the prior 12 months"),
    "prior_readmissions_12m": ("history", "Prior admissions in the past 12 months that were themselves 30-day returns"),
    "ed_visits_prior_6m": ("history", "ED visits without admission in the 6 months before admission"),
    "ed_visits_prior_12m": ("history", "ED visits without admission in the 12 months before admission"),
    "days_since_last_discharge": ("history", "Days since previous discharge (missing = no prior encounter observed)"),
    "is_readmission_of_prior": ("history", "This admission began within 30 days of a prior discharge"),
    "history_window_days": ("history", "Days of look-back actually observable (capped at 365; left-censoring indicator)"),
    "n_prior_encounters": ("history", "Prior encounters observed in the data window"),
}
for _c in COMORBIDITY_CATEGORIES:
    FEATURE_DICTIONARY[f"cm_{_c}"] = ("diagnoses", f"Coded history of {_c.replace('_', ' ')} (current secondary dx or any prior encounter)")
for _n in LAB_NAMES:
    FEATURE_DICTIONARY[f"lab_{_n.lower()}_last"] = ("labs", f"Last {_n.title()} before discharge")
    FEATURE_DICTIONARY[f"lab_{_n.lower()}_change"] = ("labs", f"Last minus first {_n.title()} during the stay")
    FEATURE_DICTIONARY[f"lab_{_n.lower()}_missing"] = ("labs", f"No {_n.title()} measured during the stay")
for _m in MED_CLASSES:
    if _m != "OTHER":
        FEATURE_DICTIONARY[f"med_{_m.lower()}"] = ("medications", f"{_m.replace('_', ' ').title()} on discharge list")

assert set(FEATURE_COLUMNS) <= set(FEATURE_DICTIONARY), set(FEATURE_COLUMNS) - set(FEATURE_DICTIONARY)


# ------------------------------------------------------------------------------ helpers
def _secs(ts: pd.Series) -> np.ndarray:
    return ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)


def _map_unique(s: pd.Series, fn) -> pd.Series:
    u = s.unique()
    return s.map({k: fn(k) for k in u})


# ------------------------------------------------------------------------------ labels
def build_labels(enc: pd.DataFrame, cohort: CohortConfig, data_end: pd.Timestamp) -> pd.DataFrame:
    """Derive the forward-looking 30-day unplanned-readmission label from the encounter table.

    An index discharge is labelled 1 if any *unplanned* admission to any facility in the network
    begins within `followup_days` of discharge. Planned admissions are ignored (not events).
    Discharges without complete follow-up (discharge + 30d > data_end) are marked unobservable.
    """
    enc = enc.sort_values(["patient_id", "admit_ts"]).reset_index(drop=True)
    g = enc.groupby("patient_id", sort=False)
    event = np.zeros(len(enc), dtype=bool)
    best_gap = np.full(len(enc), np.inf)
    for k in (1, 2, 3):
        nxt_admit = g["admit_ts"].shift(-k)
        nxt_planned = g["is_planned_admission"].shift(-k)
        gap = (nxt_admit - enc["discharge_ts"]) / DAY
        ok = nxt_admit.notna() & (gap >= 0) & (gap <= cohort.followup_days) & (nxt_planned == 0)
        event |= ok.to_numpy()
        best_gap = np.where(ok.to_numpy(), np.minimum(best_gap, gap.to_numpy()), best_gap)
    observable = ((enc["discharge_ts"] + pd.Timedelta(days=cohort.followup_days)) <= data_end).to_numpy()
    enc["label_observable"] = observable
    # an outcome that cannot yet be known is NaN, never 0
    enc["readmit_30d"] = np.where(observable, event.astype(float), np.nan)
    enc["days_to_readmit"] = np.where(event & observable, best_gap, np.nan)
    excl = enc["discharge_disposition"].isin(cohort.excluded_dispositions)
    enc["excluded_reason"] = np.where(excl, "disposition:" + enc["discharge_disposition"], "")
    enc["in_cohort"] = ~excl
    return enc


# ------------------------------------------------------------------------------ features
def build_features(tables: dict[str, pd.DataFrame], cohort: CohortConfig, data_start: pd.Timestamp,
                   data_end: pd.Timestamp) -> pd.DataFrame:
    """Return one row per encounter with metadata, label columns and point-in-time features."""
    pat = tables["patients"].set_index("patient_id")
    enc = build_labels(tables["encounters"], cohort, data_end)
    n = len(enc)
    g = enc.groupby("patient_id", sort=False)

    # ---- demographics
    enc = enc.join(pat[["birth_year", "sex", "insurance", "ses_quintile", "death_ts"]], on="patient_id")
    birth = pd.to_datetime(enc["birth_year"].astype(int).astype(str) + "-07-01")
    enc["age"] = np.minimum(((enc["admit_ts"] - birth) / pd.Timedelta(days=365.25)).to_numpy(), 90.0)
    enc["died_within_30d"] = (
        enc["death_ts"].notna()
        & ((enc["death_ts"] - enc["discharge_ts"]) / DAY).between(0, cohort.followup_days)
    )

    # ---- index stay
    enc["los_days"] = ((enc["discharge_ts"] - enc["admit_ts"]) / DAY).to_numpy()
    enc["acute_admission"] = enc["admission_type"].isin(["EMERGENCY", "URGENT"]).astype(int)
    enc["weekend_discharge"] = (enc["discharge_ts"].dt.dayofweek >= 5).astype(int)
    m = enc["discharge_ts"].dt.month
    enc["discharge_month_sin"] = np.sin(2 * np.pi * m / 12)
    enc["discharge_month_cos"] = np.cos(2 * np.pi * m / 12)
    enc["dx_group"] = _map_unique(enc["principal_dx"], dx_group)

    # ---- history from earlier completed encounters (lag loop)
    adm = enc["admit_ts"]
    prior6 = np.zeros(n); prior12 = np.zeros(n); prior_los12 = np.zeros(n); prior_rd12 = np.zeros(n)
    enc["_is_readm"] = 0.0
    last_gap = g["discharge_ts"].shift(1)
    gap1 = (adm - last_gap) / DAY
    enc["days_since_last_discharge"] = np.where(gap1 >= 0, gap1, np.nan)
    enc["is_readmission_of_prior"] = ((enc["days_since_last_discharge"] <= cohort.followup_days)).fillna(False).astype(int)
    enc["_is_readm"] = enc["is_readmission_of_prior"].astype(float)
    g = enc.groupby("patient_id", sort=False)  # re-bind: the groupby must see the new columns
    for k in range(1, MAX_LAGS + 1):
        prev_dis = g["discharge_ts"].shift(k)
        gap = ((adm - prev_dis) / DAY).to_numpy()
        valid = ~np.isnan(gap) & (gap >= 0)
        in6 = valid & (gap <= 182.5)
        in12 = valid & (gap <= 365.0)
        prior6 += in6
        prior12 += in12
        prior_los12 += np.where(in12, g["los_days"].shift(k).to_numpy(), 0.0)
        prior_rd12 += np.where(in12, g["_is_readm"].shift(k).to_numpy(), 0.0)
        if k > 1 and not valid.any():
            break
    enc["prior_admissions_6m"], enc["prior_admissions_12m"] = prior6, prior12
    enc["prior_los_days_12m"] = np.nan_to_num(prior_los12)
    enc["prior_readmissions_12m"] = np.nan_to_num(prior_rd12)
    enc["n_prior_encounters"] = g.cumcount().astype(float)
    enc["history_window_days"] = np.clip(((adm - data_start) / DAY).to_numpy(), 0, 365)

    # ---- ED visits strictly before admission (searchsorted on composite patient-time keys)
    ed = tables["ed_visits"]
    codes = {p: i for i, p in enumerate(pd.unique(pd.concat([enc["patient_id"], ed["patient_id"]])))}
    B = 10_000_000_000
    ed_keys = np.sort(ed["patient_id"].map(codes).to_numpy(dtype=np.int64) * B + _secs(ed["visit_ts"]))
    pk = enc["patient_id"].map(codes).to_numpy(dtype=np.int64) * B
    t_adm = _secs(adm)
    hi = np.searchsorted(ed_keys, pk + t_adm, side="left")
    enc["ed_visits_prior_6m"] = (hi - np.searchsorted(ed_keys, pk + t_adm - int(182.5 * 86400), side="left")).astype(float)
    enc["ed_visits_prior_12m"] = (hi - np.searchsorted(ed_keys, pk + t_adm - 365 * 86400, side="left")).astype(float)

    # ---- diagnoses -> comorbidity flags + secondary dx counts
    dx = tables["diagnoses"]
    cat_of = _map_unique(dx["icd10"], comorbidity_category)
    dxj = dx.assign(cat=cat_of).merge(
        enc[["encounter_id", "patient_id", "discharge_ts"]], on="encounter_id", how="inner"
    )
    sec_n = dxj[dxj["seq"] >= 2].groupby("encounter_id").size()
    enc["n_secondary_dx"] = enc["encounter_id"].map(sec_n).fillna(0).astype(float)
    dxc = dxj[dxj["cat"].notna()]
    cur = dxc[dxc["seq"] >= 2].assign(v=1).pivot_table(index="encounter_id", columns="cat", values="v", aggfunc="max", fill_value=0)
    first_seen = dxc.groupby(["patient_id", "cat"])["discharge_ts"].min().unstack()
    charlson = np.zeros(n)
    for cat in COMORBIDITY_CATEGORIES:
        in_cur = enc["encounter_id"].map(cur[cat]).fillna(0).to_numpy() > 0 if cat in cur.columns else np.zeros(n, bool)
        if cat in first_seen.columns:
            fs = enc["patient_id"].map(first_seen[cat])
            hist = (fs < adm).fillna(False).to_numpy()
        else:
            hist = np.zeros(n, bool)
        flag = (in_cur | hist).astype(int)
        enc[f"cm_{cat}"] = flag
        charlson += flag * COMORBIDITY_WEIGHTS[cat]
    enc["charlson_score"] = charlson

    # ---- labs up to discharge
    labs = tables["labs"].merge(enc[["encounter_id", "discharge_ts"]], on="encounter_id", how="inner")
    labs = labs[labs["collected_ts"] <= labs["discharge_ts"]].sort_values("collected_ts")
    last = labs.groupby(["encounter_id", "lab_name"])["value"].last().unstack()
    first = labs.groupby(["encounter_id", "lab_name"])["value"].first().unstack()
    nl = labs.groupby("encounter_id").size()
    for name in LAB_NAMES:
        col = name.lower()
        lv = enc["encounter_id"].map(last[name]) if name in last.columns else pd.Series(np.nan, index=enc.index)
        fv = enc["encounter_id"].map(first[name]) if name in first.columns else pd.Series(np.nan, index=enc.index)
        enc[f"lab_{col}_last"] = lv
        enc[f"lab_{col}_change"] = lv - fv
        enc[f"lab_{col}_missing"] = lv.isna().astype(int)
    enc["egfr_last"] = egfr_ckdepi_2021(enc["lab_creatinine_last"], enc["age"], (enc["sex"] == "F").to_numpy())
    enc["n_lab_values"] = enc["encounter_id"].map(nl).fillna(0).astype(float)

    # ---- discharge medications
    meds = tables["medications"]
    flags = (meds[meds["drug_class"] != "OTHER"].assign(v=1)
             .pivot_table(index="encounter_id", columns="drug_class", values="v", aggfunc="max", fill_value=0))
    for mc in MED_CLASSES:
        if mc == "OTHER":
            continue
        enc[f"med_{mc.lower()}"] = enc["encounter_id"].map(flags[mc]).fillna(0).astype(int) if mc in flags.columns else 0
    nd = meds.groupby("encounter_id")["n_drugs"].sum()
    enc["n_discharge_meds"] = enc["encounter_id"].map(nd).fillna(0).astype(float)
    enc["polypharmacy"] = (enc["n_discharge_meds"] >= 10).astype(int)
    enc["n_high_risk_med_classes"] = enc[[f"med_{c.lower()}" for c in HIGH_RISK_MED_CLASSES]].sum(axis=1).astype(float)

    for c in CATEGORICAL_FEATURES:
        enc[c] = enc[c].astype("category")
    keep = META_COLUMNS + FEATURE_COLUMNS
    out = enc[keep].sort_values("discharge_ts").reset_index(drop=True)
    return out


def modeling_cohort(feat: pd.DataFrame) -> pd.DataFrame:
    """Index discharges eligible for model development: in cohort and label observable."""
    return feat[feat["in_cohort"] & feat["label_observable"]].reset_index(drop=True)


def scoring_cohort(feat: pd.DataFrame) -> pd.DataFrame:
    """Eligible index discharges whose 30-day outcome is not yet known (the live worklist)."""
    return feat[feat["in_cohort"] & ~feat["label_observable"]].reset_index(drop=True)
