"""Small, transparent clinical helpers (not validated clinical software).

Everything here is deliberately simple and documented so that reviewers can audit it:
* CKD-EPI 2021 eGFR (race-free),
* a *simplified* Charlson-style comorbidity score from ICD-10 prefixes,
* a LACE-style heuristic baseline (Length of stay, Acuity, Comorbidity, ED visits),
* ICD-10 -> principal-diagnosis group mapping used for reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- eGFR


def egfr_ckdepi_2021(scr_mg_dl, age_years, female):
    """CKD-EPI 2021 creatinine equation (race-free). Vectorised; NaN-safe."""
    scr = np.asarray(scr_mg_dl, dtype=float)
    age = np.asarray(age_years, dtype=float)
    fem = np.asarray(female, dtype=bool)
    kappa = np.where(fem, 0.7, 0.9)
    alpha = np.where(fem, -0.241, -0.302)
    ratio = scr / kappa
    with np.errstate(invalid="ignore", divide="ignore"):
        egfr = (
            142.0
            * np.minimum(ratio, 1.0) ** alpha
            * np.maximum(ratio, 1.0) ** -1.200
            * 0.9938**age
            * np.where(fem, 1.012, 1.0)
        )
    return egfr


# ------------------------------------------------------------------ comorbidity map
# (prefix, category, simplified Charlson-style weight). Weights are a simplification of
# Charlson et al. (1987) for demonstration; they are NOT a validated implementation.
COMORBIDITY_PREFIXES: list[tuple[str, str, int]] = [
    ("I50", "chf", 1),
    ("J44", "copd", 1),
    ("E11", "diabetes", 1),
    ("N18", "ckd", 2),
    ("C", "cancer", 2),
    ("F03", "dementia", 1),
    ("G30", "dementia", 1),
    ("K74", "liver", 1),
    ("I25", "cad", 1),
    ("Z86.73", "stroke_hx", 1),
    ("I48", "afib", 0),
    ("I10", "htn", 0),
    ("F32", "depression", 0),
]
COMORBIDITY_CATEGORIES = sorted({c for _, c, _ in COMORBIDITY_PREFIXES})
COMORBIDITY_WEIGHTS = {c: w for _, c, w in COMORBIDITY_PREFIXES}


def comorbidity_category(icd: str) -> str | None:
    # longest-prefix match so "Z86.73" is not shadowed by shorter prefixes
    best = None
    best_len = -1
    for prefix, cat, _ in COMORBIDITY_PREFIXES:
        if icd.startswith(prefix) and len(prefix) > best_len:
            best, best_len = cat, len(prefix)
    return best


# ------------------------------------------------------------------ dx grouping
DX_GROUP_PREFIXES = [
    ("I50", "HF"),
    ("J44", "COPD"),
    ("J18", "PNEUMONIA"),
    ("I21", "AMI"),
    ("A41", "SEPSIS"),
    ("I63", "STROKE"),
    ("E11", "DIABETES"),
    ("N17", "AKI"),
    ("K92", "GI_BLEED"),
    ("N39", "UTI"),
    ("M16", "JOINT_REPLACEMENT"),
    ("K80", "SURGICAL_GI"),
]


def dx_group(icd: str) -> str:
    for prefix, grp in DX_GROUP_PREFIXES:
        if icd.startswith(prefix):
            return grp
    return "OTHER"


# ------------------------------------------------------------------ LACE-style score


def lace_style_score(los_days, acute_admission, charlson, ed_visits_6m) -> np.ndarray:
    """LACE-style heuristic (after van Walraven et al., CMAJ 2010), computed from this
    platform's own features. It is a *baseline*, not a reproduction of the original index
    (our comorbidity score is a simplified Charlson-style count)."""
    los = np.asarray(los_days, dtype=float)
    l_pts = np.select(
        [los < 1, los < 2, los < 3, los < 4, los < 7, los < 14],
        [0, 1, 2, 3, 4, 5],
        default=7,
    )
    a_pts = np.where(np.asarray(acute_admission, dtype=bool), 3, 0)
    c = np.asarray(charlson, dtype=float)
    c_pts = np.select([c <= 0, c == 1, c == 2, c == 3], [0, 1, 2, 3], default=5)
    e_pts = np.minimum(np.asarray(ed_visits_6m, dtype=float), 4)
    return (l_pts + a_pts + c_pts + e_pts).astype(float)


# ------------------------------------------------------------------ lab plausibility
LAB_NAMES = ["SODIUM", "CREATININE", "HEMOGLOBIN", "WBC", "ALBUMIN"]
LAB_UNITS = {
    "SODIUM": "mmol/L",
    "CREATININE": "mg/dL",
    "HEMOGLOBIN": "g/dL",
    "WBC": "10^9/L",
    "ALBUMIN": "g/dL",
}
# Hard physiological plausibility limits used by the data-quality layer.
LAB_PLAUSIBLE = {
    "SODIUM": (100.0, 180.0),
    "CREATININE": (0.1, 20.0),
    "HEMOGLOBIN": (3.0, 22.0),
    "WBC": (0.3, 100.0),
    "ALBUMIN": (1.0, 6.0),
}
# Reporting-only reference ranges.
LAB_ABNORMAL = {
    "SODIUM": ("low", 135.0),
    "HEMOGLOBIN": ("low", 10.0),
    "ALBUMIN": ("low", 3.0),
    "WBC": ("high", 12.0),
}

MED_CLASSES = [
    "ANTICOAGULANT",
    "INSULIN",
    "OPIOID",
    "DIURETIC",
    "ANTIPSYCHOTIC",
    "STEROID",
    "BETA_BLOCKER",
    "STATIN",
    "ANTIBIOTIC",
    "ACE_ARB",
    "OTHER",
]
HIGH_RISK_MED_CLASSES = ["ANTICOAGULANT", "INSULIN", "OPIOID", "DIURETIC", "ANTIPSYCHOTIC", "STEROID"]


def age_band(age) -> pd.Series:
    bins = [0, 34, 49, 64, 74, 84, 200]
    labels = ["18-34", "35-49", "50-64", "65-74", "75-84", "85+"]
    return pd.cut(pd.Series(age), bins=bins, labels=labels)
