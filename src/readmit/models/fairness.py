"""Subgroup performance and fairness assessment.

We report *descriptive* gaps with uncertainty. Thresholds used to raise flags are
documented heuristics for human review, not legal or clinical standards, and a flag is a
prompt to investigate (data, measurement, access to care), not proof of model bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .evaluate import metric_ci_single, recall_at_fraction, wilson_interval

TPR_GAP_FLAG = 0.05      # absolute difference in recall (at the operating threshold) vs. overall
OE_RANGE_FLAG = (0.80, 1.25)  # observed/expected outside this range
FLAG_RATE_RATIO = 0.80   # group's flag rate / overall flag rate below this (or above 1/this)
AUROC_GAP_FLAG = 0.05
MIN_EVENTS_FOR_FLAG = 100
MIN_N_REPORT = 50


def subgroup_table(df: pd.DataFrame, group_col: str, y_col: str, p_col: str, threshold: float,
                   cluster_col: str = "patient_id", n_boot: int = 100, seed: int = 0, capacity_frac: float = 0.10) -> pd.DataFrame:
    y_all, p_all = df[y_col].to_numpy().astype(int), df[p_col].to_numpy()
    flag_all = p_all >= threshold
    overall_tpr = flag_all[y_all == 1].mean()
    overall_flag = flag_all.mean()
    overall_auc = float(roc_auc_score(y_all, p_all))
    overall_eq = recall_at_fraction(y_all, p_all, capacity_frac)
    rows = []
    for level, g in df.groupby(group_col, observed=True):
        y, p = g[y_col].to_numpy().astype(int), g[p_col].to_numpy()
        n = len(g)
        if n < MIN_N_REPORT:
            continue
        flag = p >= threshold
        ev = int(y.sum())
        tpr = float(flag[y == 1].mean()) if ev else np.nan
        fpr = float(flag[y == 0].mean()) if (y == 0).any() else np.nan
        ppv = float(y[flag].mean()) if flag.any() else np.nan
        tpr_lo, tpr_hi = wilson_interval(flag[y == 1].sum(), ev) if ev else (np.nan, np.nan)
        auc = float(roc_auc_score(y, p)) if 0 < ev < n else np.nan
        auc_ci = metric_ci_single(y, p, g[cluster_col].to_numpy(), roc_auc_score, n_boot, seed) if 0 < ev < n and n_boot else (np.nan, np.nan)
        oe = float(y.mean() / p.mean())
        # recall when the SAME share (capacity_frac) of this group is flagged: isolates ranking quality
        # from base-rate differences that a single network-wide threshold would otherwise mix in
        eq = recall_at_fraction(y, p, capacity_frac) if ev else np.nan
        eq_lo, eq_hi = wilson_interval(round(eq * ev), ev) if ev else (np.nan, np.nan)
        rows.append(dict(
            recall_equal_rate=eq, recall_eq_ci_low=eq_lo, recall_eq_ci_high=eq_hi, recall_eq_overall=float(overall_eq),
            attribute=group_col, group=str(level), n=n, events=ev, prevalence=float(y.mean()), mean_predicted=float(p.mean()),
            observed_over_expected=oe, auroc=auc, auroc_ci_low=auc_ci[0], auroc_ci_high=auc_ci[1],
            flag_rate=float(flag.mean()), recall_at_threshold=tpr, recall_ci_low=tpr_lo, recall_ci_high=tpr_hi,
            fpr_at_threshold=fpr, ppv_at_threshold=ppv,
            recall_overall=float(overall_tpr), auroc_gap_vs_overall=(auc - overall_auc) if not np.isnan(auc) else np.nan,
            recall_gap_vs_overall=(tpr - overall_tpr) if ev else np.nan,
            flag_rate_ratio_vs_overall=float(flag.mean() / overall_flag) if overall_flag else np.nan,
        ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["review_flags"] = out.apply(_flags, axis=1)
    return out


def _flags(r) -> str:
    """Review prompts. Differences in who a single network-wide threshold selects (flag rate, recall at
    the threshold) are reported descriptively but NOT flagged: they mostly mirror base-rate differences.
    Flags instead target calibration, within-group ranking quality at an equal selection rate, and AUROC."""
    f = []
    if r["events"] >= MIN_EVENTS_FOR_FLAG:
        if not (OE_RANGE_FLAG[0] <= r["observed_over_expected"] <= OE_RANGE_FLAG[1]):
            f.append("calibration O/E")
        if pd.notna(r["recall_equal_rate"]) and abs(r["recall_equal_rate"] - r["recall_eq_overall"]) > TPR_GAP_FLAG \
                and not (r["recall_eq_ci_low"] <= r["recall_eq_overall"] <= r["recall_eq_ci_high"]):
            f.append("within-group ranking gap")
    if pd.notna(r["auroc_gap_vs_overall"]) and r["n"] >= 500 and r["auroc_gap_vs_overall"] < -AUROC_GAP_FLAG:
        f.append("lower AUROC")
    return "; ".join(f)


def fairness_report(df: pd.DataFrame, attributes: list[str], y_col: str, p_col: str, threshold: float,
                    n_boot: int = 100, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-group table, per-attribute disparity summary)."""
    parts = [subgroup_table(df, a, y_col, p_col, threshold, n_boot=n_boot, seed=seed) for a in attributes]
    table = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    summ = []
    for a, g in table.groupby("attribute"):
        summ.append(dict(
            attribute=a, groups=len(g),
            recall_range=float(g["recall_at_threshold"].max() - g["recall_at_threshold"].min()),
            recall_equal_rate_range=float(g["recall_equal_rate"].max() - g["recall_equal_rate"].min()),
            auroc_range=float(g["auroc"].max() - g["auroc"].min()),
            oe_min=float(g["observed_over_expected"].min()), oe_max=float(g["observed_over_expected"].max()),
            flag_rate_min=float(g["flag_rate"].min()), flag_rate_max=float(g["flag_rate"].max()),
            groups_flagged=int((g["review_flags"] != "").sum()),
            flagged_groups="; ".join(g.loc[g["review_flags"] != "", "group"].astype(str)),
        ))
    return table, pd.DataFrame(summ)
