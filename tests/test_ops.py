from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from readmit.ops import prioritize as ops


def _pool(n_days=60, per_day=40, seed=0):
    rng = np.random.default_rng(seed); n = n_days * per_day
    risk = np.clip(rng.beta(2, 9, n), 0.01, 0.9)
    tier = np.where(risk >= np.quantile(risk, 0.9), "High", np.where(risk >= np.quantile(risk, 0.7), "Elevated", "Standard"))
    return pd.DataFrame({"day": np.repeat(pd.date_range("2025-01-01", periods=n_days), per_day), "facility_id": rng.choice(["F1", "F2", "F3"], n),
                         "risk": risk, "lace_score": risk * 10 + rng.normal(0, 2, n), "tier": tier, "readmit_30d": (rng.random(n) < risk).astype(int),
                         "insurance": rng.choice(["MEDICARE", "MEDICAID", "COMMERCIAL"], n)})


def _minutes(pool, served, params):
    cost = pool["tier"].map({k: v.minutes for k, v in params.interventions.items()}).to_numpy()
    return pd.Series(cost * served).groupby(pool["day"].to_numpy()).sum()


@pytest.mark.parametrize("staffing", ["pooled", "facility"])
@pytest.mark.parametrize("strategy", list(ops.STRATEGIES))
def test_daily_minutes_never_exceed_budget(staffing, strategy):
    pool = _pool(); params = ops.ProgramParams(fte=2.0, staffing=staffing)
    served = ops.allocate(pool, strategy, params, seed=1)
    assert _minutes(pool, served, params).max() <= params.minutes_per_day + 1e-6
    assert served.sum() > 0


def test_zero_capacity_serves_nobody_and_full_capacity_serves_everyone():
    pool = _pool()
    assert ops.allocate(pool, "risk", ops.ProgramParams(fte=0.0), 0).sum() == 0
    assert ops.allocate(pool, "risk", ops.ProgramParams(fte=1000.0), 0).all()


def test_risk_ranking_captures_more_events_than_random_under_scarcity():
    pool = _pool(n_days=200, per_day=50); params = ops.ProgramParams(fte=ops.fte_for_coverage(pool, 0.15))
    ev_risk = pool["readmit_30d"][ops.allocate(pool, "risk", params, 0)].sum() / ops.allocate(pool, "risk", params, 0).sum()
    ev_rand = pool["readmit_30d"][ops.allocate(pool, "random", params, 0)].sum() / ops.allocate(pool, "random", params, 0).sum()
    assert ev_risk > ev_rand * 1.3


def test_avoided_never_exceeds_events_and_grows_with_capacity():
    pool = _pool(); res = []
    for fte in (0.5, 1.5, 4.0):
        r = ops.run_scenario(pool, ["risk"], ops.ProgramParams(fte=fte), 0, 300).iloc[0]
        assert r["avoided_p95"] <= r["events_in_served"] and r["events_in_served"] <= r["events_total"]
        res.append(r["avoided_mean"])
    assert res[0] < res[1] < res[2]


def test_no_effect_means_no_avoided_readmissions():
    pool = _pool()
    iv = {k: replace(v, rrr=0.0001) for k, v in ops.DEFAULT_INTERVENTIONS.items()}
    r = ops.run_scenario(pool, ["risk"], ops.ProgramParams(fte=3.0, interventions=iv, rrr_rel_sd=0.0), 0, 200).iloc[0]
    assert r["avoided_mean"] < 0.5


def test_common_random_numbers_give_tight_paired_difference():
    pool = _pool(n_days=200, per_day=50); params = ops.ProgramParams(fte=ops.fte_for_coverage(pool, 0.2))
    r = ops.run_scenario(pool, ["random", "risk"], params, 3, 600).set_index("strategy")
    width_abs = r.loc["risk", "avoided_p95"] - r.loc["risk", "avoided_p05"]
    width_pair = r.loc["risk", "avoided_vs_random_p95"] - r.loc["risk", "avoided_vs_random_p05"]
    assert width_pair < width_abs and r.loc["risk", "avoided_vs_random_mean"] > 0


def test_equity_floor_increases_reach_for_low_risk_group():
    pool = _pool(n_days=150, per_day=60)
    base = ops.ProgramParams(fte=ops.fte_for_coverage(pool, 0.2))
    s0 = ops.allocate(pool, "risk", base, 0)
    s1 = ops.allocate(pool, "risk", replace(base, equity_col="insurance", equity_floor=0.8), 0)
    c0 = ops.coverage_by_group(pool, s0, "insurance").set_index("group")["coverage_ratio"]
    c1 = ops.coverage_by_group(pool, s1, "insurance").set_index("group")["coverage_ratio"]
    assert (c1.max() - c1.min()) <= (c0.max() - c0.min()) + 1e-9


def test_sensitivity_and_helpers():
    pool = _pool(); params = ops.ProgramParams(fte=2.0)
    s = ops.sensitivity(pool, "risk", params, 0)
    assert len(s) == 9 and s.iloc[0]["variant"] == "Base case"
    assert s.set_index("variant").loc["Intervention effect x1.5", "avoided"] > s.set_index("variant").loc["Intervention effect x0.5", "avoided"]
    assert ops.fte_for_coverage(pool, 1.0) * ops.PRODUCTIVE_MINUTES_PER_FTE_DAY == pytest.approx(ops.full_service_minutes_per_day(pool), rel=0.02)
