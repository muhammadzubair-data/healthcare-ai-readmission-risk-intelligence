"""Care-management prioritisation under staffing constraints (simulation only).

What this is: a transparent *work-queue ranking* tool. Given each day's discharges, a risk
estimate and a number of care-manager minutes, it orders the follow-up queue and shows what
a policy would capture under explicitly stated assumptions.

What this is NOT: it does not recommend, select or withhold any clinical treatment. Risk is
a predictive association; the effect sizes of follow-up programmes below are user-editable
ASSUMPTIONS (the model has no causal or uplift component), and every output is a scenario,
not a forecast of real savings.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

PRODUCTIVE_MINUTES_PER_FTE_DAY = 360


@dataclass(frozen=True)
class Intervention:
    name: str
    minutes: float            # care-manager minutes per patient
    rrr: float                # ASSUMED relative reduction in readmission probability if delivered


DEFAULT_INTERVENTIONS = {
    "High": Intervention("Transitional-care bundle", 110.0, 0.18),
    "Elevated": Intervention("Nurse follow-up + medication review", 40.0, 0.10),
    "Standard": Intervention("48-hour check-in call", 12.0, 0.04),
}


@dataclass(frozen=True)
class ProgramParams:
    fte: float = 3.0
    staffing: str = "pooled"                 # "pooled" (central team) | "facility" (staff tied to a facility)
    interventions: dict = field(default_factory=lambda: dict(DEFAULT_INTERVENTIONS))
    cost_per_readmission: float = 15_000.0   # ILLUSTRATIVE placeholder, edit for your system
    cost_per_care_minute: float = 1.0        # ~ $60/hour fully loaded; ILLUSTRATIVE placeholder
    rrr_rel_sd: float = 0.35                 # relative uncertainty on each assumed RRR
    equity_col: str | None = None            # e.g. "insurance"; used with equity_floor
    equity_floor: float = 0.0                # share of daily minutes first allocated proportionally by group

    @property
    def minutes_per_day(self) -> float:
        return self.fte * PRODUCTIVE_MINUTES_PER_FTE_DAY


STRATEGIES = {
    "random": "No targeting (random order)",
    "lace": "LACE-style score ranking",
    "risk": "Model risk ranking",
    "benefit_per_minute": "Assumed benefit per care-manager minute",
}


def _priority_score(pool: pd.DataFrame, strategy: str, params: ProgramParams, rng: np.random.Generator) -> np.ndarray:
    if strategy == "random":
        return rng.random(len(pool))
    if strategy == "lace":
        return pool["lace_score"].to_numpy() + rng.random(len(pool)) * 1e-3  # random tie-break
    if strategy == "risk":
        return pool["risk"].to_numpy()
    if strategy == "benefit_per_minute":
        rrr = pool["tier"].map({k: v.rrr for k, v in params.interventions.items()}).to_numpy()
        mins = pool["tier"].map({k: v.minutes for k, v in params.interventions.items()}).to_numpy()
        return pool["risk"].to_numpy() * rrr / mins
    raise ValueError(f"unknown strategy {strategy}")


def _first_fit(order: np.ndarray, cost: np.ndarray, budget: float, served: np.ndarray) -> float:
    """Walk down the ranked list; serve a patient if their intervention fits the remaining minutes."""
    remaining = budget
    for i in order:
        if served[i]:
            continue
        c = cost[i]
        if c <= remaining:
            served[i] = True
            remaining -= c
            if remaining <= 0:
                break
    return budget - remaining


def allocate(pool: pd.DataFrame, strategy: str, params: ProgramParams, seed: int = 0) -> np.ndarray:
    """Boolean mask of served discharges. Pool must have: day, facility_id, tier, risk, lace_score."""
    rng = np.random.default_rng(seed)
    score = _priority_score(pool, strategy, params, rng)
    cost = pool["tier"].map({k: v.minutes for k, v in params.interventions.items()}).to_numpy(dtype=float)
    served = np.zeros(len(pool), dtype=bool)
    day = pool["day"].to_numpy()
    fac = pool["facility_id"].to_numpy()
    grp = pool[params.equity_col].astype(str).to_numpy() if (params.equity_col and params.equity_floor > 0) else None

    fac_share = None
    if params.staffing == "facility":
        vc = pool["facility_id"].value_counts(normalize=True)
        fac_share = vc.to_dict()

    # indices grouped by day (and facility if facility-staffed)
    keys = pd.Series(day) if params.staffing == "pooled" else pd.Series(day).astype(str) + "|" + pd.Series(fac)
    for _, idx in keys.groupby(keys, sort=False).indices.items():
        idx = np.asarray(idx)
        if params.staffing == "pooled":
            budget = params.minutes_per_day
        else:
            budget = params.minutes_per_day * fac_share[fac[idx[0]]]
        order_all = idx[np.argsort(-score[idx], kind="stable")]
        if grp is not None:
            floor_budget = budget * params.equity_floor
            groups, counts = np.unique(grp[idx], return_counts=True)
            for g, cnt in zip(groups, counts):
                members = order_all[grp[order_all] == g]
                _first_fit(members, cost, floor_budget * cnt / len(idx), served)
            used = cost[idx][served[idx]].sum()
            _first_fit(order_all, cost, max(budget - used, 0.0), served)
        else:
            _first_fit(order_all, cost, budget, served)
    return served


# ------------------------------------------------------------------ outcome simulation
def _beta_params(mean: float, rel_sd: float) -> tuple[float, float]:
    sd = max(mean * rel_sd, 1e-6)
    k = mean * (1 - mean) / sd**2 - 1
    k = max(k, 2.0)
    return mean * k, (1 - mean) * k


def simulate_outcomes(pool: pd.DataFrame, served: np.ndarray, params: ProgramParams,
                      n_draws: int = 1000, seed: int = 0) -> dict:
    """Monte-Carlo avoided-readmission interval under the ASSUMED relative risk reductions.

    Events are the (synthetic) observed 30-day readmissions among served patients; each is
    avoided with probability RRR_k, where RRR_k is itself drawn from a Beta distribution around
    the assumption to propagate assumption uncertainty.
    """
    # separate streams: the assumed-effect draws are IDENTICAL across strategies (common random
    # numbers), so differences between strategies have tight paired intervals
    rng = np.random.default_rng(seed)
    rng_bin = np.random.default_rng(seed + 1)
    y = pool["readmit_30d"].to_numpy().astype(int)
    tier = pool["tier"].to_numpy()
    avoided = np.zeros(n_draws)
    expected_draws = np.zeros(n_draws)
    per_tier_events = {}
    for k, iv in params.interventions.items():
        n_k = int(((tier == k) & served & (y == 1)).sum())
        per_tier_events[k] = n_k
        a, b = _beta_params(iv.rrr, params.rrr_rel_sd)
        rrr_draw = rng.beta(a, b, n_draws)
        avoided += rng_bin.binomial(n_k, rrr_draw)
        expected_draws += n_k * rrr_draw
    mins = pool["tier"].map({k: v.minutes for k, v in params.interventions.items()}).to_numpy(dtype=float)
    minutes_used = float(mins[served].sum())
    n_served = int(served.sum())
    events_total = int(y.sum())
    events_served = int(y[served].sum())
    prog_cost = minutes_used * params.cost_per_care_minute
    gross = avoided * params.cost_per_readmission
    net = gross - prog_cost
    days = max(pool["day"].nunique(), 1)
    pct = lambda a_, q: float(np.percentile(a_, q))
    return dict(
        discharges=len(pool), days=int(days), served=n_served, served_share=n_served / len(pool),
        minutes_used=minutes_used, minutes_available=params.minutes_per_day * days,
        utilization=minutes_used / (params.minutes_per_day * days) if params.minutes_per_day else np.nan,
        events_total=events_total, events_in_served=events_served,
        recall_of_events=events_served / events_total if events_total else np.nan,
        precision_in_served=events_served / n_served if n_served else np.nan,
        avoided_mean=float(avoided.mean()), avoided_p05=pct(avoided, 5), avoided_p95=pct(avoided, 95),
        program_cost=prog_cost,
        net_savings_mean=float(net.mean()), net_savings_p05=pct(net, 5), net_savings_p95=pct(net, 95),
        minutes_per_avoided=minutes_used / avoided.mean() if avoided.mean() > 0 else np.nan,
        served_per_avoided=n_served / avoided.mean() if avoided.mean() > 0 else np.nan,
        events_by_tier_served=per_tier_events, _expected_draws=expected_draws,
    )


def run_scenario(pool: pd.DataFrame, strategies: list[str], params: ProgramParams, seed: int = 0,
                 n_draws: int = 1000) -> pd.DataFrame:
    rows = []
    for s in strategies:
        served = allocate(pool, s, params, seed)
        res = simulate_outcomes(pool, served, params, n_draws, seed)
        res.update(strategy=s, strategy_label=STRATEGIES.get(s, s))
        rows.append(res)
    ref = next((r["_expected_draws"] for r in rows if r["strategy"] == "random"), None)
    for r in rows:
        if ref is not None:
            d = r["_expected_draws"] - ref
            r.update(avoided_vs_random_mean=float(d.mean()), avoided_vs_random_p05=float(np.percentile(d, 5)), avoided_vs_random_p95=float(np.percentile(d, 95)))
    out = pd.DataFrame(rows)
    return out.drop(columns=["events_by_tier_served", "_expected_draws"]) if len(out) else out


def coverage_by_group(pool: pd.DataFrame, served: np.ndarray, group_col: str) -> pd.DataFrame:
    """Who gets served vs. who is discharged and who has events (equity view)."""
    d = pool.assign(served=served)
    g = d.groupby(group_col, observed=True).agg(discharges=("served", "size"), served=("served", "sum"),
                                              events=("readmit_30d", "sum"),
                                              events_served=("readmit_30d", lambda s: int(s[d.loc[s.index, "served"]].sum())))
    g["share_of_discharges"] = g["discharges"] / g["discharges"].sum()
    g["share_of_served"] = g["served"] / max(g["served"].sum(), 1)
    g["coverage_ratio"] = g["share_of_served"] / g["share_of_discharges"]
    g["event_recall"] = g["events_served"] / g["events"].replace(0, np.nan)
    return g.reset_index().rename(columns={group_col: "group"})


def sensitivity(pool: pd.DataFrame, strategy: str, params: ProgramParams, seed: int = 0) -> pd.DataFrame:
    """One-at-a-time sensitivity of expected avoided readmissions and net savings."""
    base = simulate_outcomes(pool, allocate(pool, strategy, params, seed), params, 400, seed)
    rows = [dict(variant="Base case", avoided=base["avoided_mean"], net=base["net_savings_mean"])]

    def with_rrr(mult):
        iv = {k: replace(v, rrr=min(v.rrr * mult, 0.95)) for k, v in params.interventions.items()}
        return replace(params, interventions=iv)

    def with_minutes(mult):
        iv = {k: replace(v, minutes=v.minutes * mult) for k, v in params.interventions.items()}
        return replace(params, interventions=iv)

    variants = {
        "Intervention effect x0.5": with_rrr(0.5), "Intervention effect x1.5": with_rrr(1.5),
        "Minutes per patient x0.75": with_minutes(0.75), "Minutes per patient x1.25": with_minutes(1.25),
        "Staffing x0.5": replace(params, fte=params.fte * 0.5), "Staffing x2": replace(params, fte=params.fte * 2),
        "Cost per readmission x0.5": replace(params, cost_per_readmission=params.cost_per_readmission * 0.5),
        "Cost per readmission x1.5": replace(params, cost_per_readmission=params.cost_per_readmission * 1.5),
    }
    for name, p in variants.items():
        r = simulate_outcomes(pool, allocate(pool, strategy, p, seed), p, 400, seed)
        rows.append(dict(variant=name, avoided=r["avoided_mean"], net=r["net_savings_mean"]))
    return pd.DataFrame(rows)


def full_service_minutes_per_day(pool: pd.DataFrame, interventions: dict | None = None) -> float:
    """Daily minutes needed if EVERY discharge received the intervention for its tier."""
    iv = interventions or DEFAULT_INTERVENTIONS
    return float(pool["tier"].map({k: v.minutes for k, v in iv.items()}).sum() / max(pool["day"].nunique(), 1))


def fte_for_coverage(pool: pd.DataFrame, coverage: float, interventions: dict | None = None) -> float:
    """FTE that would cover `coverage` (0-1) of the full-service workload."""
    return round(coverage * full_service_minutes_per_day(pool, interventions) / PRODUCTIVE_MINUTES_PER_FTE_DAY, 1)


def build_pool(scored: pd.DataFrame, splits=("test", "test_unseen")) -> pd.DataFrame:
    """Out-of-sample discharges with a known outcome, in the shape the simulator expects."""
    p = scored[scored["split"].isin(splits)].copy()
    p["day"] = p["discharge_ts"].dt.normalize()
    return p.reset_index(drop=True)
