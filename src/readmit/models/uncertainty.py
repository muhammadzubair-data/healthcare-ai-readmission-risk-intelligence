"""Uncertainty-aware risk presentation.

Point predictions are shown with:
* a *model-uncertainty band* from a patient-level bootstrap ensemble of LightGBM models
  (the spread reflects how much the estimate depends on the training sample, NOT outcome
  randomness - even a perfectly known risk of 20% means 80% of such patients are not readmitted),
* a *tier stability* score: the share of ensemble members that agree with the assigned tier,
* data-completeness warnings for key inputs that were missing at discharge.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .evaluate import _cluster_index, cluster_resample_indices

TIER_LABELS = ["Standard", "Elevated", "High"]


def assign_tier(p, cutoffs: tuple[float, float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.where(p >= cutoffs[1], "High", np.where(p >= cutoffs[0], "Elevated", "Standard"))


def train_bootstrap_ensemble(X: pd.DataFrame, y: np.ndarray, patient_ids, params: dict, num_rounds: int,
                             size: int, seed: int, categorical: list[str]) -> list[lgb.Booster]:
    order, starts, lens = _cluster_index(patient_ids)
    rng = np.random.default_rng(seed)
    boosters = []
    for b in range(size):
        idx = cluster_resample_indices(order, starts, lens, rng)
        p = dict(params, seed=seed + 101 * (b + 1), bagging_seed=seed + 7 * (b + 1), feature_fraction_seed=seed + 13 * (b + 1))
        ds = lgb.Dataset(X.iloc[idx], label=np.asarray(y)[idx], categorical_feature=categorical, free_raw_data=True)
        boosters.append(lgb.train(p, ds, num_boost_round=num_rounds))
    return boosters


def ensemble_predict(boosters: list[lgb.Booster], X: pd.DataFrame, transform=None) -> np.ndarray:
    """(members, n) matrix of probabilities; `transform` (e.g. isotonic) is applied per member."""
    out = np.vstack([b.predict(X) for b in boosters])
    if transform is not None:
        out = np.vstack([transform(row) for row in out])
    return out


def summarize_ensemble(member_preds: np.ndarray, point_tier: np.ndarray, cutoffs: tuple[float, float]) -> pd.DataFrame:
    lo, hi = np.percentile(member_preds, [5, 95], axis=0)
    member_tiers = np.vstack([assign_tier(r, cutoffs) for r in member_preds])
    stability = (member_tiers == point_tier[None, :]).mean(axis=0)
    return pd.DataFrame({"risk_p05": lo, "risk_p95": hi, "ensemble_mean": member_preds.mean(axis=0), "tier_stability": stability})
