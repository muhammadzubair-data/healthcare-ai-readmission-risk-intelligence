import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import ui


def render():
    m = data.metrics()
    sc = data.scored()
    eda = data.eda()
    fac = data.facilities().set_index("facility_id")
    funnel = data.jsonf("cohort_funnel.json")
    test = m["models"]["test"]["lightgbm_calibrated"]
    ops = data.table("ops_scenarios.csv").set_index("strategy")
    foe = data.table("facility_oe.csv")

    st.header("Executive overview")
    st.caption("What the network looks like, how well the model separates risk, and what a capacity-limited follow-up programme *could* achieve under stated assumptions.")
    c = st.columns(5)
    mc = eda[eda["split"].isin(["train", "val", "test", "test_unseen"])]
    ui.kpi(c[0], "Index discharges modelled", f"{len(mc):,}", f"{eda['facility_id'].nunique()} facilities, {mc['month'].min():%Y}-{mc['month'].max():%Y}")
    ui.kpi(c[1], "30-day readmission rate", ui.pct(mc["readmit_30d"].mean()), f"held-out period: {ui.pct(test['prevalence'])}", ui.ORANGE)
    ui.kpi(c[2], "AUROC (held-out future)", f"{test['auroc']:.3f}", "temporal test, calibrated LightGBM", ui.BLUE)
    ui.kpi(c[3], "Recall at 10% capacity", ui.pct(test["recall_at_10pct"], 0), f"lift {test['lift_at_10pct']:.1f}x vs. random", ui.AQUA)
    flagged = foe[foe["status"] != "Within expected range"]
    ui.kpi(c[4], "Facilities outside expected range", f"{len(flagged)} of {len(foe)}", "case-mix adjusted O/E, held-out period", ui.RED if len(flagged) else ui.GREEN)

    st.subheader("Readmission trend: observed vs. model-expected")
    base = mc.groupby("month").agg(observed=("readmit_30d", "mean"), n=("readmit_30d", "size")).reset_index()
    oos = sc[sc["split"].isin(["val", "test", "test_unseen"])].copy()
    oos["month"] = oos["discharge_ts"].dt.to_period("M").dt.to_timestamp()
    exp = oos.groupby("month").agg(expected=("risk", "mean")).reset_index()
    t = base.merge(exp, on="month", how="left")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["month"], y=t["observed"], name="Observed", mode="lines", line=dict(color=ui.BLUE, width=2)))
    fig.add_trace(go.Scatter(x=t["month"], y=t["expected"], name="Model-expected (out-of-sample months)", mode="lines", line=dict(color=ui.ORANGE, width=2, dash="dot")))
    fig.add_vrect(x0=pd.Timestamp(m["splits"]["test"]["first"]), x1=t["month"].max(), fillcolor="rgba(128,128,128,.10)", line_width=0,
                  annotation_text="held-out test period", annotation_position="top left")
    fig.update_yaxes(tickformat=".0%", title="30-day readmission rate")
    st.plotly_chart(ui.style_fig(fig, 340), width="stretch")
    ui.note("Early months look lower because the simulated network starts empty: patients have little observable history, and history-rich patients are higher risk. "
            "This is a documented property of the simulator, not a clinical trend.")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Facility observed / expected (held-out period)")
        f = foe.merge(fac[["facility_name", "facility_type"]], left_on="facility_id", right_index=True)
        f = f.sort_values("oe")
        col = f["status"].map({"Higher than expected": ui.RED, "Lower than expected": ui.GREEN, "Within expected range": ui.BLUE})
        fig = go.Figure(go.Scatter(x=f["oe"], y=f["facility_name"], mode="markers", marker=dict(size=11, color=col),
                                   error_x=dict(type="data", symmetric=False, array=f["oe_high"] - f["oe"], arrayminus=f["oe"] - f["oe_low"], color="rgba(128,128,128,.7)"),
                                   customdata=np.c_[f["observed"], f["expected"], f["n"], f["status"]],
                                   hovertemplate="%{y}<br>O/E %{x:.2f}<br>observed %{customdata[0]:.0f} vs expected %{customdata[1]:.0f} (n=%{customdata[2]})<br>%{customdata[3]}<extra></extra>"))
        fig.add_vline(x=1.0, line_dash="dash", line_color="rgba(128,128,128,.7)")
        fig.update_xaxes(title="Observed / expected readmissions (95% interval)")
        st.plotly_chart(ui.style_fig(fig, 380, False), width="stretch")
        st.caption("Expected counts come from a model that does not know the facility, so this compares case-mix-adjusted performance. A flag is a prompt to investigate (process, coding, patient mix the model cannot see, data issues), not a quality ranking.")
    with right:
        st.subheader("Highest-rate diagnosis groups")
        g = mc[mc["split"].isin(["val", "test", "test_unseen"])].groupby("dx_group").agg(rate=("readmit_30d", "mean"), n=("readmit_30d", "size")).reset_index()
        g = ui.suppress(g, "n", ["rate"]).dropna().sort_values("rate")
        fig = px.bar(g, x="rate", y="dx_group", orientation="h", color_discrete_sequence=[ui.BLUE], hover_data={"n": True, "rate": ":.1%"})
        fig.update_xaxes(tickformat=".0%", title="30-day readmission rate (out-of-sample period)")
        fig.update_yaxes(title="")
        st.plotly_chart(ui.style_fig(fig, 380, False), width="stretch")

    st.subheader("What a capacity-limited programme could look like (scenario, not forecast)")
    p = m["ops"]["params"]
    r, rk = ops.loc["risk"], ops.loc["random"]
    cc = st.columns(3)
    ui.kpi(cc[0], "Staffing assumed", f"{p['fte']:.1f} FTE", f"{p['minutes_per_day']:.0f} care-manager min/day (~{m['ops']['default_workload_coverage']:.0%} of full-service workload)")
    ui.kpi(cc[1], "Avoided readmissions, risk-ranked", f"{r['avoided_mean']:.0f}", f"90% interval {r['avoided_p05']:.0f}-{r['avoided_p95']:.0f} over {m['ops']['pool_days']} days", ui.AQUA)
    ui.kpi(cc[2], "vs. random targeting, same staff", f"{r['avoided_mean'] / rk['avoided_mean']:.2f}x", f"{rk['avoided_mean']:.0f} avoided without targeting", ui.ORANGE)
    ui.note("These numbers rest on <b>assumed</b> intervention effects (relative risk reductions of 4%, 10% and 18% by tier) that the model cannot verify. "
            "Explore and change them on the <i>Care-management scenarios</i> page.")
