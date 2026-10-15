"""Explainability: SHAP summaries and reason codes.

SHAP values describe how the *model* arrives at a risk estimate (associations in the
training data). They are not causal effects: changing a feature in the real world need not
change a patient's risk in the way a SHAP value suggests. All wording in this project uses
"associated with higher/lower estimated risk" for that reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from ..features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS, FEATURE_DICTIONARY, NUMERIC_FEATURES
from .train import Bundle

PRETTY = {
    "los_days": "Length of stay (days)", "age": "Age", "charlson_score": "Comorbidity score (Charlson-style)",
    "prior_admissions_12m": "Admissions in prior 12 months", "prior_admissions_6m": "Admissions in prior 6 months",
    "ed_visits_prior_6m": "ED visits in prior 6 months", "ed_visits_prior_12m": "ED visits in prior 12 months",
    "n_discharge_meds": "Number of discharge medications", "egfr_last": "eGFR at discharge",
    "lab_sodium_last": "Sodium at discharge", "lab_hemoglobin_last": "Hemoglobin at discharge",
    "lab_albumin_last": "Albumin at discharge", "lab_wbc_last": "WBC at discharge", "lab_creatinine_last": "Creatinine at discharge",
    "days_since_last_discharge": "Days since last discharge", "is_readmission_of_prior": "Admitted within 30 days of prior discharge",
    "dx_group": "Principal diagnosis group", "discharge_disposition": "Discharge destination", "insurance": "Payer",
    "ses_quintile": "Area deprivation quintile", "n_secondary_dx": "Secondary diagnoses coded", "n_procedures": "Procedures",
    "weekend_discharge": "Weekend discharge", "icu_stay": "ICU stay", "polypharmacy": "10+ discharge medications",
    "admission_type": "Admission type", "admit_source": "Admission source", "sex": "Sex",
    "history_window_days": "Observable history (days)", "n_prior_encounters": "Prior encounters observed",
    "prior_los_days_12m": "Inpatient days in prior 12 months", "prior_readmissions_12m": "Prior 30-day returns (12m)",
    "n_lab_values": "Lab results recorded", "n_high_risk_med_classes": "High-risk medication classes",
}


def pretty(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    if name.startswith("cm_"):
        return "History of " + name[3:].replace("_", " ")
    if name.startswith("med_"):
        return name[4:].replace("_", " ").title() + " at discharge"
    if name.startswith("lab_") and name.endswith("_missing"):
        return name[4:-8].title() + " not measured"
    if name.startswith("lab_") and name.endswith("_change"):
        return name[4:-7].title() + " change during stay"
    return FEATURE_DICTIONARY.get(name, ("", name))[1] if name in FEATURE_DICTIONARY else name


def explainer_sanity(bundle: Bundle, df: pd.DataFrame, n: int = 200) -> float:
    """Max abs difference between the `shap` library and LightGBM's native TreeSHAP (should be ~1e-6)."""
    sub = df.head(n)
    X = bundle_frame(bundle, sub)
    sv = shap.TreeExplainer(bundle.lgbm).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    native, _ = bundle.shap_contributions(sub)
    return float(np.abs(np.asarray(sv) - native).max())


def bundle_frame(bundle: Bundle, df: pd.DataFrame) -> pd.DataFrame:
    from .train import prep_frame
    return prep_frame(df, bundle.levels)


def global_importance(shap_values: np.ndarray, features: pd.DataFrame) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    rows = []
    for j, f in enumerate(FEATURE_COLUMNS):
        direction = np.nan
        if f in NUMERIC_FEATURES:
            x = features[f].to_numpy(dtype=float)
            m = ~np.isnan(x)
            if m.sum() > 50 and np.nanstd(x[m]) > 0 and np.std(shap_values[m, j]) > 0:
                direction = float(np.corrcoef(x[m], shap_values[m, j])[0, 1])
        rows.append(dict(feature=f, label=pretty(f), group=FEATURE_DICTIONARY[f][0], mean_abs_shap=float(mean_abs[j]), direction_corr=direction))
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def binned_dependence(shap_values: np.ndarray, features: pd.DataFrame, feature: str, bins: int = 12) -> pd.DataFrame:
    j = FEATURE_COLUMNS.index(feature)
    if feature in CATEGORICAL_FEATURES:
        d = pd.DataFrame({"x": features[feature].astype(str).to_numpy(), "shap": shap_values[:, j]})
        return d.groupby("x").agg(mean_shap=("shap", "mean"), n=("shap", "size")).reset_index()
    x = features[feature].to_numpy(dtype=float)
    d = pd.DataFrame({"x": x, "shap": shap_values[:, j]}).dropna()
    if d["x"].nunique() <= bins:
        g = d.groupby("x").agg(mean_shap=("shap", "mean"), n=("shap", "size")).reset_index()
    else:
        d["bin"] = pd.qcut(d["x"], bins, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(x=("x", "mean"), mean_shap=("shap", "mean"), n=("shap", "size")).reset_index(drop=True)
    return g


def reason_codes(shap_row: np.ndarray, feature_row: pd.Series, k: int = 6) -> pd.DataFrame:
    """Top contributors for one patient, in neutral associational language."""
    order = np.argsort(-np.abs(shap_row))[:k]
    rows = []
    for j in order:
        f = FEATURE_COLUMNS[j]
        v = feature_row[f]
        if isinstance(v, (float, np.floating)) and not np.isnan(v):
            val = f"{v:.1f}" if abs(v) < 1000 and not float(v).is_integer() else f"{int(v)}"
        else:
            val = "missing" if pd.isna(v) else str(v)
        rows.append(dict(feature=f, label=pretty(f), value=val, contribution_logodds=float(shap_row[j]),
                         direction="associated with higher estimated risk" if shap_row[j] > 0 else "associated with lower estimated risk"))
    return pd.DataFrame(rows)


def logistic_odds_ratios(bundle: Bundle) -> pd.DataFrame:
    """Odds ratios (per 1 SD for numeric inputs; vs. reference for one-hot categories)."""
    pre, clf = bundle.logreg.named_steps["pre"], bundle.logreg.named_steps["clf"]
    names = pre.get_feature_names_out()
    coefs = clf.coef_[0]
    df = pd.DataFrame({"term": names, "coef": coefs, "odds_ratio": np.exp(coefs)})
    df["term"] = df["term"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    df["abs"] = df["coef"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
