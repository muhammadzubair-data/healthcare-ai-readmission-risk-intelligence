"""Evaluation metrics: discrimination, calibration, capacity-based utility, cluster bootstrap."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def top_k_indices(p: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(round(frac * len(p))))
    return np.argsort(-p, kind="stable")[:k]


def recall_at_fraction(y, p, frac: float) -> float:
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(y[top_k_indices(np.asarray(p), frac)].sum() / y.sum())


def precision_at_fraction(y, p, frac: float) -> float:
    y = np.asarray(y)
    return float(y[top_k_indices(np.asarray(p), frac)].mean())


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration of the model's logit: slope 1 / intercept 0 is ideal."""
    lp = _logit(p)[:, None]
    lr = LogisticRegression(C=1e8, solver="lbfgs", max_iter=500).fit(lp, np.asarray(y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def core_metrics(y, p, capacity_fracs=(0.05, 0.10, 0.15, 0.20)) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    slope, intercept = calibration_slope_intercept(y, p)
    out = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS))),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_predicted": float(p.mean()),
        "observed_over_expected": float(y.mean() / p.mean()),
        "ece": expected_calibration_error(y, p),
    }
    for f in capacity_fracs:
        out[f"recall_at_{int(f * 100)}pct"] = recall_at_fraction(y, p, f)
        out[f"precision_at_{int(f * 100)}pct"] = precision_at_fraction(y, p, f)
        out[f"lift_at_{int(f * 100)}pct"] = out[f"precision_at_{int(f * 100)}pct"] / out["prevalence"]
    return out


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        lo, hi = wilson_interval(y[m].sum(), m.sum())
        rows.append(dict(bin=b + 1, n=int(m.sum()), mean_predicted=float(p[m].mean()), observed_rate=float(y[m].mean()), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def gains_curve(y, p, points: int = 50) -> pd.DataFrame:
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p), kind="stable")
    cum = np.cumsum(y[order]) / max(y.sum(), 1)
    fr = np.arange(1, len(y) + 1) / len(y)
    sel = np.unique(np.linspace(0, len(y) - 1, points).astype(int))
    return pd.DataFrame({"fraction_flagged": fr[sel], "recall": cum[sel]})


def decision_curve(y, p, thresholds=None) -> pd.DataFrame:
    """Net benefit of acting on patients with predicted risk >= threshold (Vickers & Elkin 2006).
    Reflects a *screening* decision (who to review), not a treatment-effect claim."""
    y, p = np.asarray(y), np.asarray(p)
    thresholds = np.linspace(0.05, 0.5, 46) if thresholds is None else thresholds
    n, prev = len(y), y.mean()
    rows = []
    for t in thresholds:
        flag = p >= t
        tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum()
        nb = tp / n - fp / n * t / (1 - t)
        nb_all = prev - (1 - prev) * t / (1 - t)
        rows.append(dict(threshold=float(t), net_benefit=float(nb), treat_all=float(nb_all), treat_none=0.0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cluster bootstrap
def _cluster_index(groups) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes, uniq = pd.factorize(pd.Series(np.asarray(groups)))
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    starts = np.searchsorted(sc, np.arange(len(uniq)), side="left")
    ends = np.searchsorted(sc, np.arange(len(uniq)), side="right")
    return order, starts, ends - starts


def cluster_resample_indices(order, starts, lens, rng) -> np.ndarray:
    samp = rng.integers(0, len(starts), len(starts))
    ln = lens[samp]
    total = int(ln.sum())
    base = np.repeat(starts[samp], ln)
    off = np.arange(total) - np.repeat(np.cumsum(ln) - ln, ln)
    return order[base + off]


def cluster_bootstrap_metrics(y, preds: dict[str, np.ndarray], groups, n_boot: int = 200, seed: int = 0,
                              capacity_frac: float = 0.10, reference: str | None = None) -> dict:
    """Patient-clustered bootstrap CIs (encounters of one patient are resampled together).

    Returns {model: {metric: (lo, hi)}} and, if `reference` is given, paired differences of
    every other model against it for AUROC and PR-AUC.
    """
    y = np.asarray(y).astype(int)
    order, starts, lens = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    names = list(preds)
    keys = ["auroc", "pr_auc", "recall_at_cap", "brier"]
    store = {m: {k: [] for k in keys} for m in names}
    diffs = {m: {"auroc": [], "pr_auc": []} for m in names if reference and m != reference}
    for _ in range(n_boot):
        idx = cluster_resample_indices(order, starts, lens, rng)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        res = {}
        for m in names:
            pb = np.asarray(preds[m])[idx]
            res[m] = (roc_auc_score(yb, pb), average_precision_score(yb, pb), recall_at_fraction(yb, pb, capacity_frac), brier_score_loss(yb, pb))
            for k, v in zip(keys, res[m]):
                store[m][k].append(v)
        if reference:
            for m in diffs:
                diffs[m]["auroc"].append(res[m][0] - res[reference][0])
                diffs[m]["pr_auc"].append(res[m][1] - res[reference][1])
    ci = {m: {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in d.items() if v} for m, d in store.items()}
    dci = {m: {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.mean(v))) for k, v in d.items() if v} for m, d in diffs.items()}
    return {"ci": ci, "paired_diff_vs_reference": dci, "n_boot": n_boot, "reference": reference}


def metric_ci_single(y, p, groups, fn: Callable[[np.ndarray, np.ndarray], float], n_boot: int = 100, seed: int = 0) -> tuple[float, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    order, starts, lens = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = cluster_resample_indices(order, starts, lens, rng)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))
