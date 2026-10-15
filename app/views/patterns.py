import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import ui


def _rate_table(df, by):
    g = df.groupby(by, observed=True).agg(n=("readmit_30d", "size"), events=("readmit_30d", "sum"), rate=("readmit_30d", "mean")).reset_index()
    return ui.suppress(g, "n", ["rate", "events"])


def render():
    st.header("Readmission patterns")
    st.caption("Descriptive patterns in the synthetic network. Differences between groups are associations in simulated data; they are not evidence of cause. Cells with fewer than 11 discharges are suppressed.")
    eda = data.eda()
    fac = data.facilities().set_index("facility_id")
    d = eda[eda["split"].isin(["train", "val", "test", "test_unseen"])].copy()
    d["facility"] = d["facility_id"].map(fac["facility_name"])

    c1, c2, c3 = st.columns(3)
    years = sorted(d["month"].dt.year.unique())
    yr = c1.multiselect("Discharge year", years, default=years)
    facs = c2.multiselect("Facility", sorted(d["facility"].unique()))
    ins = c3.multiselect("Payer", sorted(d["insurance"].unique()))
    d = d[d["month"].dt.year.isin(yr)]
    if facs: d = d[d["facility"].isin(facs)]
    if ins: d = d[d["insurance"].isin(ins)]
    if d.empty:
        st.warning("No discharges match."); return
    st.markdown(f"**{len(d):,} discharges** - {ui.pct(d['readmit_30d'].mean())} readmitted within 30 days")

    t1, t2, t3, t4 = st.tabs(["Facility & diagnosis", "Patient groups", "Stay & discharge", "Timing"])
    with t1:
        a, b = st.columns(2)
        with a:
            g = _rate_table(d, "facility").dropna().sort_values("rate")
            fig = px.bar(g, x="rate", y="facility", orientation="h", color_discrete_sequence=[ui.BLUE], hover_data={"n": True, "rate": ":.1%"})
            fig.update_xaxes(tickformat=".0%", title="30-day readmission rate"); fig.update_yaxes(title="")
            st.plotly_chart(ui.style_fig(fig, 400, False), width="stretch")
            st.caption("Unadjusted. Facilities differ in case mix; see the executive page for case-mix-adjusted O/E.")
        with b:
            g = _rate_table(d, "dx_group").dropna().sort_values("rate")
            fig = px.bar(g, x="rate", y="dx_group", orientation="h", color_discrete_sequence=[ui.ORANGE], hover_data={"n": True, "rate": ":.1%"})
            fig.update_xaxes(tickformat=".0%", title="30-day readmission rate"); fig.update_yaxes(title="")
            st.plotly_chart(ui.style_fig(fig, 400, False), width="stretch")
        d["quarter"] = d["month"].dt.to_period("Q").astype(str)
        h = d.groupby(["facility", "quarter"]).agg(rate=("readmit_30d", "mean"), n=("readmit_30d", "size")).reset_index()
        h = ui.suppress(h, "n", ["rate"])
        piv = h.pivot(index="facility", columns="quarter", values="rate")
        fig = px.imshow(piv, aspect="auto", color_continuous_scale=["#e8f0fb", "#2a78d6", "#0b2f66"], labels=dict(color="Rate"))
        fig.update_layout(title="Readmission rate by facility and quarter", coloraxis_colorbar=dict(tickformat=".0%"))
        st.plotly_chart(ui.style_fig(fig, 380, False), width="stretch")
        st.caption("The simulated network starts empty, so the earliest quarters have little observable history and lower rates (a simulator property).")
    with t2:
        a, b = st.columns(2)
        with a:
            g = _rate_table(d, "age_band").dropna()
            order = ["18-34", "35-49", "50-64", "65-74", "75-84", "85+"]
            g["age_band"] = pd.Categorical(g["age_band"], order, ordered=True); g = g.sort_values("age_band")
            fig = px.bar(g, x="age_band", y="rate", color_discrete_sequence=[ui.BLUE], hover_data={"n": True, "rate": ":.1%"})
            fig.update_yaxes(tickformat=".0%", title="Rate"); fig.update_xaxes(title="Age band")
            st.plotly_chart(ui.style_fig(fig, 320, False), width="stretch")
        with b:
            g = _rate_table(d, "insurance").dropna()
            fig = px.bar(g, x="insurance", y="rate", color_discrete_sequence=[ui.AQUA], hover_data={"n": True, "rate": ":.1%"})
            fig.update_yaxes(tickformat=".0%", title="Rate"); fig.update_xaxes(title="Payer")
            st.plotly_chart(ui.style_fig(fig, 320, False), width="stretch")
        a, b = st.columns(2)
        with a:
            g = _rate_table(d, "ses_quintile").dropna()
            fig = px.bar(g, x="ses_quintile", y="rate", color_discrete_sequence=[ui.VIOLET], hover_data={"n": True, "rate": ":.1%"})
            fig.update_yaxes(tickformat=".0%", title="Rate"); fig.update_xaxes(title="Area deprivation quintile (1 = least, 5 = most; synthetic)")
            st.plotly_chart(ui.style_fig(fig, 320, False), width="stretch")
        with b:
            g = _rate_table(d, "sex").dropna()
            fig = px.bar(g, x="sex", y="rate", color_discrete_sequence=[ui.ORANGE], hover_data={"n": True, "rate": ":.1%"})
            fig.update_yaxes(tickformat=".0%", title="Rate")
            st.plotly_chart(ui.style_fig(fig, 320, False), width="stretch")
    with t3:
        a, b = st.columns(2)
        with a:
            d["los_band"] = pd.cut(d["los_days"], [0, 2, 4, 7, 14, 100], labels=["<=2d", "3-4d", "5-7d", "8-14d", ">14d"])
            g = _rate_table(d, "los_band").dropna()
            fig = px.bar(g, x="los_band", y="rate", color_discrete_sequence=[ui.BLUE], hover_data={"n": True, "rate": ":.1%"})
            fig.update_yaxes(tickformat=".0%", title="Rate"); fig.update_xaxes(title="Length of stay")
            st.plotly_chart(ui.style_fig(fig, 320, False), width="stretch")
        with b:
            g = _rate_table(d, "discharge_disposition").dropna().sort_values("rate")
            fig = px.bar(g, x="rate", y="discharge_disposition", orientation="h", color_discrete_sequence=[ui.ORANGE], hover_data={"n": True, "rate": ":.1%"})
            fig.update_xaxes(tickformat=".0%", title="Rate"); fig.update_yaxes(title="")
            st.plotly_chart(ui.style_fig(fig, 320, False), width="stretch")
        d["prior_band"] = pd.cut(d["prior_admissions_12m"], [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
        g = _rate_table(d, "prior_band").dropna()
        fig = px.bar(g, x="prior_band", y="rate", color_discrete_sequence=[ui.AQUA], hover_data={"n": True, "rate": ":.1%"})
        fig.update_yaxes(tickformat=".0%", title="Rate"); fig.update_xaxes(title="Admissions in the prior 12 months")
        st.plotly_chart(ui.style_fig(fig, 300, False), width="stretch")
    with t4:
        r = d[d["readmit_30d"] == 1]
        fig = px.histogram(r, x="days_to_readmit", nbins=30, color_discrete_sequence=[ui.BLUE])
        fig.update_xaxes(title="Days from discharge to unplanned readmission"); fig.update_yaxes(title="Readmissions")
        st.plotly_chart(ui.style_fig(fig, 320, False), width="stretch")
        within7 = (r["days_to_readmit"] <= 7).mean()
        ui.note(f"{ui.pct(within7, 0)} of 30-day readmissions in view occur within the first week after discharge. The timing shape is a property of the simulator's truncated-exponential assumption, so do not generalize it to real hospitals.")
