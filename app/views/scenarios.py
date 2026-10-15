from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import ui
from readmit.ops import prioritize as ops


@st.cache_data(show_spinner=False)
def _pool():
    return ops.build_pool(data.scored())


@st.cache_data(show_spinner="Simulating follow-up workload...")
def _run(fte, staffing, strategies, iv_key, cpr, cpm, rel_sd, eq_col, eq_floor, draws):
    pool = _pool()
    iv = {k: ops.Intervention(n, mi, r) for k, (n, mi, r) in iv_key}
    params = ops.ProgramParams(fte=fte, staffing=staffing, interventions=iv, cost_per_readmission=cpr, cost_per_care_minute=cpm,
                               rrr_rel_sd=rel_sd, equity_col=eq_col, equity_floor=eq_floor)
    res = ops.run_scenario(pool, list(strategies), params, seed=7, n_draws=draws)
    served = {s: ops.allocate(pool, s, params, 7) for s in strategies}
    return res, served, params


def render():
    st.header("Care-management scenarios")
    ui.disclaimer()
    st.markdown(
        "A **what-if workload simulator**, not a treatment recommender. It shows how many discharges a follow-up team could reach and what that *might* avoid "
        "**if** the assumed programme effects below were true. The model predicts who is at risk; it cannot tell who would *benefit*, so every benefit figure is an "
        "assumption-driven scenario. Nothing here selects or withholds clinical care; human care managers decide.")
    pool = _pool()
    full_min = ops.full_service_minutes_per_day(pool)
    days = pool["day"].nunique()
    st.caption(f"Pool: {len(pool):,} out-of-sample discharges over {days} days (~{len(pool) / days:.0f}/day). Serving everyone with their tier's intervention would take ~{full_min:,.0f} care-manager minutes/day (~{full_min / ops.PRODUCTIVE_MINUTES_PER_FTE_DAY:.1f} FTE). Unserved patients are not carried over (follow-up is time-critical).")

    with st.sidebar:
        st.markdown("### Scenario assumptions")
        cov = st.slider("Staffing as share of full-service workload", 0.05, 1.0, 0.35, 0.05)
        fte = ops.fte_for_coverage(pool, cov)
        st.caption(f"= {fte:.1f} care-manager FTE ({fte * ops.PRODUCTIVE_MINUTES_PER_FTE_DAY:,.0f} min/day, 360 productive min/FTE)")
        staffing = st.radio("Staffing model", ["pooled", "facility"], format_func=lambda x: "Central team (pooled)" if x == "pooled" else "Facility-based (capacity split by volume)")
        st.markdown("**Interventions by risk tier** (minutes per patient, assumed relative risk reduction)")
        iv = {}
        for tier, d in ops.DEFAULT_INTERVENTIONS.items():
            c1, c2 = st.columns(2)
            mi = c1.number_input(f"{tier}: minutes", 5, 300, int(d.minutes), key=f"m{tier}")
            rr = c2.number_input(f"{tier}: RRR %", 0, 60, int(d.rrr * 100), key=f"r{tier}") / 100
            iv[tier] = (d.name, float(mi), float(rr))
        cpr = st.number_input("Cost per readmission ($, illustrative)", 1000, 100000, 15000, 1000)
        cpm = st.number_input("Cost per care-manager minute ($, illustrative)", 0.1, 5.0, 1.0, 0.1)
        rel_sd = st.slider("Uncertainty on each assumed effect (relative sd)", 0.0, 0.8, 0.35, 0.05)
        eq_col = st.selectbox("Equity floor: group by", ["(none)", "insurance", "ses_group", "age_band", "sex"])
        eq_floor = st.slider("Share of minutes allocated proportionally by group first", 0.0, 1.0, 0.0, 0.1) if eq_col != "(none)" else 0.0

    strategies = st.multiselect("Prioritisation rules to compare", list(ops.STRATEGIES), default=list(ops.STRATEGIES), format_func=ops.STRATEGIES.get)
    if not strategies:
        st.info("Pick at least one rule."); return
    res, served, params = _run(fte, staffing, tuple(strategies), tuple(sorted(iv.items())), cpr, cpm, rel_sd, None if eq_col == "(none)" else eq_col, eq_floor, 1000)

    import textwrap
    res = res.assign(wrapped=res["strategy_label"].map(lambda t: "<br>".join(textwrap.wrap(t, 16))))
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Bar(x=res["wrapped"], y=res["avoided_mean"], marker_color=ui.BLUE,
                               error_y=dict(type="data", symmetric=False, array=res["avoided_p95"] - res["avoided_mean"], arrayminus=res["avoided_mean"] - res["avoided_p05"], color="rgba(128,128,128,.9)")))
        fig.update_yaxes(title="Readmissions"); fig.update_layout(title="Readmissions avoided (assumed effects; bars = 90% interval)")
        st.plotly_chart(ui.style_fig(fig, 360, False), width="stretch")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=res["wrapped"], y=res["served_share"], name="Share of discharges reached", marker_color=ui.AQUA))
        fig.add_trace(go.Bar(x=res["wrapped"], y=res["recall_of_events"], name="Share of readmissions in reached patients", marker_color=ui.ORANGE))
        fig.update_yaxes(tickformat=".0%", title=""); fig.update_layout(barmode="group", title="Reach vs. readmissions captured")
        fig = ui.style_fig(fig, 360); fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), margin=dict(l=10, r=10, t=90, b=10))
        st.plotly_chart(fig, width="stretch")

    show = res.assign(**{"Rule": res["strategy_label"], "Reached": res["served"], "Reached %": res["served_share"] * 100, "Readmissions captured": res["events_in_served"],
                         "Avoided (mean)": res["avoided_mean"], "Interval 5-95%": res.apply(lambda r: f"{r.avoided_p05:.0f}-{r.avoided_p95:.0f}", axis=1),
                         "vs. random (5-95%)": res.apply(lambda r: "reference" if r.strategy == "random" else (f"{r.avoided_vs_random_mean:+.0f} ({r.avoided_vs_random_p05:+.0f} to {r.avoided_vs_random_p95:+.0f})" if "avoided_vs_random_mean" in res.columns else ""), axis=1),
                         "Care minutes used": res["minutes_used"], "Reached per avoided": res["served_per_avoided"], "Minutes per avoided": res["minutes_per_avoided"],
                         "Net $ (mean)": res["net_savings_mean"]})[["Rule", "Reached", "Reached %", "Readmissions captured", "Avoided (mean)", "Interval 5-95%", "vs. random (5-95%)", "Care minutes used", "Reached per avoided", "Minutes per avoided", "Net $ (mean)"]]
    st.dataframe(show, hide_index=True, width="stretch", column_config={"Reached %": st.column_config.NumberColumn(format="%.1f%%"),
                 "Avoided (mean)": st.column_config.NumberColumn(format="%.0f"), "Net $ (mean)": st.column_config.NumberColumn(format="$%.0f"),
                 "Care minutes used": st.column_config.NumberColumn(format="%.0f"), "Reached per avoided": st.column_config.NumberColumn(format="%.0f"), "Minutes per avoided": st.column_config.NumberColumn(format="%.0f")})
    ui.note("<b>How to read this.</b> 'Readmissions captured' is how many of the (synthetic) real readmissions were among reached patients. 'Avoided' applies the assumed effect of each tier's intervention to those captured events "
            "and propagates the uncertainty on the assumption. Intervals are wide because the assumed effect itself is uncertain. All rules use the same random draws, so 'vs. random' is a paired comparison; when its range reaches below zero, the advantage over random order is not secure under these assumptions. Net $ = avoided x cost per readmission - care minutes x cost per minute, using placeholder costs. Change the assumptions to see how fragile or robust the ranking is.")

    st.subheader("Who gets reached? (equity view)")
    rule = st.selectbox("Rule", strategies, format_func=ops.STRATEGIES.get)
    attr = st.selectbox("Group by", ["insurance", "ses_group", "age_band", "facility_id"])
    cov_t = ops.coverage_by_group(pool, served[rule], attr)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cov_t["group"], y=cov_t["share_of_discharges"], name="Share of discharges", marker_color="rgba(138,138,133,.8)"))
    fig.add_trace(go.Bar(x=cov_t["group"], y=cov_t["share_of_served"], name="Share of reached patients", marker_color=ui.BLUE))
    fig.update_layout(barmode="group", title="Reach vs. population share"); fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(ui.style_fig(fig, 320), width="stretch")
    st.dataframe(ui.suppress(cov_t, "discharges", ["coverage_ratio", "event_recall"]).round(3), hide_index=True, width="stretch")
    st.caption("Risk-ranking concentrates effort on groups with higher estimated risk (older, more comorbid patients). The equity floor in the sidebar reserves part of the capacity for proportional reach and shows the trade-off in avoided readmissions.")

    st.subheader("How sensitive is the answer?")
    base_rule = "risk" if "risk" in strategies else strategies[0]
    sens = ops.sensitivity(pool, base_rule, params, 7)
    s2 = sens.iloc[1:].copy(); b0 = sens.iloc[0]["avoided"]
    s2["delta"] = s2["avoided"] - b0
    fig = go.Figure(go.Bar(x=s2["delta"], y=s2["variant"], orientation="h", marker_color=[ui.AQUA if d >= 0 else ui.ORANGE for d in s2["delta"]]))
    fig.update_xaxes(title=f"Change in avoided readmissions vs. base case ({b0:.0f}) - {ops.STRATEGIES[base_rule]}"); fig.update_yaxes(title="")
    st.plotly_chart(ui.style_fig(fig, 340, False), width="stretch")

    st.subheader("Capacity curve (precomputed, default assumptions)")
    cc = data.table("ops_capacity_curve.csv")
    fig = go.Figure()
    for s, col in (("random", ui.GREY), ("lace", ui.AQUA), ("risk", ui.BLUE), ("benefit_per_minute", ui.ORANGE)):
        r = cc[cc["strategy"] == s]
        fig.add_trace(go.Scatter(x=r["workload_coverage"], y=r["avoided_mean"], name=ops.STRATEGIES[s], mode="lines+markers", line=dict(color=col, width=2)))
    fig.update_xaxes(title="Staffing as share of full-service workload", tickformat=".0%"); fig.update_yaxes(title="Avoided readmissions (assumption-based)")
    st.plotly_chart(ui.style_fig(fig, 340), width="stretch")
    st.caption("Targeting matters most when staff are scarce; as capacity approaches full service, every rule converges because nearly everyone is reached.")
