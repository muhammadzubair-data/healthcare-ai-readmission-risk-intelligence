import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import ui
from readmit.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from readmit.models.explain import pretty


def _contrib(row: pd.DataFrame):
    booster, meta = data.booster()
    X = row[FEATURE_COLUMNS].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c].astype(str), categories=meta["levels"][c])
    for c in NUMERIC_FEATURES:
        X[c] = X[c].astype(float)
    c = booster.predict(X, pred_contrib=True)
    return c[0, :-1], c[0, -1]


def render():
    st.header("Patient risk explorer")
    ui.disclaimer()
    sc = data.scored()
    fac = data.facilities().set_index("facility_id")["facility_name"].to_dict()
    m = data.metrics()
    st.caption("A review queue for *human* care managers. Risk is the model's estimate that this discharge is followed by an unplanned readmission within 30 days; "
               "it does not say why a readmission would happen or what would prevent it.")

    c1, c2, c3, c4 = st.columns(4)
    cohort = c1.selectbox("Discharge cohort", ["Live worklist (outcome not yet known)", "Retrospective: held-out test period"], index=0)
    d = sc[sc["split"] == "live"] if cohort.startswith("Live") else sc[sc["split"].isin(["test", "test_unseen"])]
    facs = c2.multiselect("Facility", sorted(d["facility_id"].unique()), format_func=lambda x: f"{x} - {fac.get(x, '')}")
    dxs = c3.multiselect("Principal diagnosis group", sorted(d["dx_group"].unique()))
    tiers = c4.multiselect("Risk tier", ["High", "Elevated", "Standard"], default=["High"])
    if facs: d = d[d["facility_id"].isin(facs)]
    if dxs: d = d[d["dx_group"].isin(dxs)]
    if tiers: d = d[d["tier"].isin(tiers)]
    if d.empty:
        st.warning("No discharges match these filters."); return
    dmin, dmax = d["discharge_ts"].min().date(), d["discharge_ts"].max().date()
    rng = st.slider("Discharge date range", dmin, dmax, (max(dmin, (pd.Timestamp(dmax) - pd.Timedelta(days=13)).date()), dmax))
    d = d[(d["discharge_ts"].dt.date >= rng[0]) & (d["discharge_ts"].dt.date <= rng[1])].sort_values("risk", ascending=False)
    st.markdown(f"**{len(d):,} discharges** in view. Tier cut-offs are fixed from validation risk: Elevated >= {m['tier_cutoffs'][0]:.1%}, High >= {m['tier_cutoffs'][1]:.1%} (top 10%).")

    show = d.head(300).copy()
    tbl = pd.DataFrame({
        "Patient (pseudonym)": show["patient_id"], "Facility": show["facility_id"], "Discharged": show["discharge_ts"].dt.strftime("%Y-%m-%d"),
        "Diagnosis group": show["dx_group"], "Est. risk": show["risk"] * 100, "90% model band": show.apply(lambda r: f"{r.risk_p05:.0%}-{r.risk_p95:.0%}", axis=1),
        "Tier": show["tier"], "Tier stability": show["tier_stability"] * 100,
    })
    if not cohort.startswith("Live"):
        tbl["Outcome (retrospective)"] = np.where(show["readmit_30d"] == 1, "readmitted <=30d", "not readmitted")
    st.dataframe(tbl.reset_index(drop=True), width="stretch", height=300, hide_index=True,
                 column_config={"Est. risk": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                "Tier stability": st.column_config.NumberColumn(format="%.0f%%", help="Share of 10 bootstrap-ensemble models that agree with this tier")})
    st.caption("Showing the 300 highest-risk rows. 'Model band' reflects uncertainty in the model estimate from training-sample variation, not randomness of the outcome.")

    st.subheader("Patient detail")
    pick = st.selectbox("Select a discharge", show["encounter_id"].tolist(),
                        format_func=lambda e: f"{show.set_index('encounter_id').loc[e, 'patient_id']} | {show.set_index('encounter_id').loc[e, 'discharge_ts']:%Y-%m-%d} | risk {show.set_index('encounter_id').loc[e, 'risk']:.0%}")
    row = show[show["encounter_id"] == pick].iloc[[0]]
    r = row.iloc[0]
    a, b = st.columns([1, 1.6])
    with a:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[r.risk], y=[""], orientation="h", marker_color=ui.TIER_COLORS[r.tier], width=0.5, name="Estimated risk",
                             error_x=dict(type="data", symmetric=False, array=[r.risk_p95 - r.risk], arrayminus=[r.risk - r.risk_p05], color="rgba(128,128,128,.9)", thickness=2)))
        fig.update_xaxes(range=[0, max(0.6, r.risk_p95 * 1.2)], tickformat=".0%")
        fig.update_layout(showlegend=False, title=f"Estimated 30-day readmission risk: {r.risk:.0%}")
        st.plotly_chart(ui.style_fig(fig, 170, False), width="stretch")
        st.markdown(f"**Tier: {r.tier}** (stability {r.tier_stability:.0%}). Network base rate in the held-out period: {ui.pct(m['splits']['test']['rate'])}.")
        miss = [n for n in ["sodium", "creatinine", "hemoglobin", "wbc"] if r[f"lab_{n}_missing"] == 1]
        if miss:
            st.warning("Inputs missing at discharge: " + ", ".join(miss) + ". The estimate leans more on other information, so treat it with extra caution.")
        if r.tier_stability < 0.7:
            st.info("This patient sits near a tier boundary: ensemble members disagree about the tier.")
        st.markdown(f"- Facility: **{fac.get(r.facility_id, r.facility_id)}**\n- Age {r.age:.0f}, {r.sex}, {r.insurance.title().replace('_', ' ')}\n- LOS {r.los_days:.1f} days, discharged to {r.discharge_disposition.replace('_', ' ').title()}\n- {int(r.prior_admissions_12m)} admissions and {int(r.ed_visits_prior_12m)} ED visits in prior 12 months")
        if cohort.startswith("Retro"):
            st.success("Observed: readmitted within 30 days" if r.readmit_30d == 1 else "Observed: not readmitted within 30 days")
    with b:
        contrib, base = _contrib(row)
        order = np.argsort(-np.abs(contrib))[:8][::-1]
        vals = []
        for j in order:
            f = FEATURE_COLUMNS[j]; v = row.iloc[0][f]
            vals.append(f"{pretty(f)} = " + ("missing" if pd.isna(v) else (f"{v:.1f}" if isinstance(v, (float, np.floating)) and not float(v).is_integer() else str(int(v) if isinstance(v, (float, np.floating)) else v))))
        fig = go.Figure(go.Bar(x=contrib[order], y=vals, orientation="h", marker_color=[ui.RED if c > 0 else ui.BLUE for c in contrib[order]]))
        fig.update_xaxes(title="Contribution to estimated risk (log-odds, SHAP)")
        fig.update_layout(title="Largest contributors to this estimate")
        st.plotly_chart(ui.style_fig(fig, 330, False), width="stretch")
        st.caption("Red = associated with higher estimated risk; blue = lower. These are associations learned from data, **not causes**: changing an input in practice need not change this patient's risk.")
    hist = data.eda()
    h = hist[hist["patient_id"] == r.patient_id].sort_values("month")
    if len(h) > 1:
        st.markdown("**Encounter history for this pseudonymous patient** (admissions observed in the network)")
        st.dataframe(h[["discharge_ts", "facility_id", "dx_group", "los_days", "discharge_disposition"]].rename(columns={"discharge_ts": "Discharged"}), hide_index=True, width="stretch")
