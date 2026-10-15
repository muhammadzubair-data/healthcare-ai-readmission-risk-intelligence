"""Post-deployment-style monitoring: facility observed/expected, drift, temporal stability.

Expected readmissions come from a model that does NOT know the facility, so facility O/E is
a case-mix-adjusted comparison. An elevated O/E is a prompt to investigate (process, coding,
patient mix not captured by the model, data issues) - it is not a ranking of "good" and "bad"
hospitals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..data.quality import population_stability_index, psi_label
from ..features.build import NUMERIC_FEATURES


def observed_expected(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    O, E = float(y.sum()), float(p.sum())
    var = float((p * (1 - p)).sum())
    se = np.sqrt(var)
    z = (O - E) / se if se > 0 else np.nan
    return dict(n=int(len(y)), observed=O, expected=E, oe=O / E if E else np.nan,
                oe_low=(O - 1.96 * se) / E if E else np.nan, oe_high=(O + 1.96 * se) / E if E else np.nan, z=float(z), se_events=float(se))


def facility_oe(df: pd.DataFrame, p_col: str = "risk", alert_z: float = 3.0) -> pd.DataFrame:
    rows = []
    for fid, g in df.groupby("facility_id"):
        r = observed_expected(g["readmit_30d"], g[p_col])
        r["facility_id"] = fid
        r["status"] = ("Higher than expected" if r["z"] >= alert_z else "Lower than expected" if r["z"] <= -alert_z else "Within expected range")
        rows.append(r)
    return pd.DataFrame(rows).sort_values("oe", ascending=False).reset_index(drop=True)


def facility_monthly_oe(df: pd.DataFrame, p_col: str = "risk", min_n: int = 100) -> pd.DataFrame:
    d = df.assign(month=df["discharge_ts"].dt.to_period("M").dt.to_timestamp())
    rows = []
    for (fid, m), g in d.groupby(["facility_id", "month"]):
        if len(g) < min_n:
            continue
        r = observed_expected(g["readmit_30d"], g[p_col])
        rows.append(dict(facility_id=fid, month=m, n=r["n"], oe=r["oe"], observed=r["observed"], expected=r["expected"]))
    return pd.DataFrame(rows)


def facility_period_oe(df: pd.DataFrame, cut: str, p_col: str = "risk") -> pd.DataFrame:
    """O/E before vs after a date per facility (e.g. to localise a change in performance)."""
    d = df.assign(period=np.where(df["discharge_ts"] < pd.Timestamp(cut), "before", "after"))
    rows = []
    for (fid, per), g in d.groupby(["facility_id", "period"]):
        r = observed_expected(g["readmit_30d"], g[p_col])
        rows.append(dict(facility_id=fid, period=per, n=r["n"], oe=r["oe"], z=r["z"]))
    return pd.DataFrame(rows)


def feature_drift(train: pd.DataFrame, later: pd.DataFrame, features: list[str] | None = None, bins: int = 10) -> pd.DataFrame:
    features = features or [f for f in NUMERIC_FEATURES if not f.endswith("_missing")]
    rows = []
    for f in features:
        psi = population_stability_index(train[f].to_numpy(dtype=float), later[f].to_numpy(dtype=float), bins)
        rows.append(dict(feature=f, psi=psi, status=psi_label(psi),
                         train_mean=float(np.nanmean(train[f].to_numpy(dtype=float))), later_mean=float(np.nanmean(later[f].to_numpy(dtype=float)))))
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def temporal_performance(df: pd.DataFrame, p_col: str = "risk", freq: str = "Q") -> pd.DataFrame:
    d = df.assign(period=df["discharge_ts"].dt.to_period(freq).astype(str))
    rows = []
    for per, g in d.groupby("period"):
        y = g["readmit_30d"].to_numpy()
        if y.min() == y.max() or len(g) < 200:
            continue
        rows.append(dict(period=per, n=len(g), prevalence=float(y.mean()), mean_predicted=float(g[p_col].mean()),
                         auroc=float(roc_auc_score(y, g[p_col])), oe=float(y.mean() / g[p_col].mean())))
    return pd.DataFrame(rows)
