import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import ui


def render():
    st.header("Data quality, monitoring and governance")
    ui.disclaimer()
    dq = data.jsonf("dq_report.json")
    prov = data.jsonf("provenance.json")
    m = data.metrics()
    funnel = data.jsonf("cohort_funnel.json")
    t1, t2, t3, t4, t5 = st.tabs(["Data quality", "Monitoring", "Provenance & privacy", "Model card", "Limitations & governance"])

    with t1:
        act, led = dq["actions"], dq["injected_defects_ledger"]
        st.markdown("The raw tables contain **documented, injected defects**. The ingestion layer must find and repair them; the table compares what was injected with what was caught.")
        rows = [
            ("Duplicate encounter rows", led["duplicate_encounter_rows"], act["duplicate_encounters_dropped"], "dropped (first kept)"),
            ("Discharge before admission", led["discharge_before_admit_rows"], act["encounters_quarantined_bad_dates"], "quarantined (cannot know which timestamp is wrong)"),
            ("Impossible lab values", led["impossible_lab_values"], act["implausible_lab_values_removed"], "removed (plausibility limits)"),
            ("Creatinine unit shift (umol/L under a mg/dL label)", led["creatinine_unit_shift_rows"], act["creatinine_unit_shift_rows_converted"], "detected by facility-month median ratio, converted"),
            ("Unknown discharge disposition", led["unknown_disposition_rows"], act["unknown_disposition_rows"], "kept as explicit category"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Defect", "Injected", "Handled", "Action"]), hide_index=True, width="stretch")
        st.caption("Counts can differ slightly: a sentinel value overwriting a shifted value is counted once, and a quarantined encounter's child records go with it.")
        cells = pd.DataFrame(dq["pre"]["unit_shift_cells"])
        if len(cells):
            cells["month"] = pd.to_datetime(cells["month"])
            st.markdown(f"**Silent unit change detected:** facility {', '.join(act['creatinine_unit_shift_facilities'])} - monthly median creatinine was ~{cells['ratio_vs_network_median'].median():.0f}x the network median from {cells['month'].min():%b %Y}, although the unit label never changed.")
            fig = px.bar(cells, x="month", y="ratio_vs_network_median", color_discrete_sequence=[ui.RED])
            fig.update_yaxes(title="Ratio to network median", type="log"); fig.update_xaxes(title="")
            st.plotly_chart(ui.style_fig(fig, 280, False), width="stretch")
        a, b = st.columns(2)
        with a:
            comp = pd.DataFrame(dq["post"]["completeness"]["labs"].items(), columns=["column", "missing"])
            enc_c = pd.DataFrame(dq["post"]["completeness"]["encounters"].items(), columns=["column", "missing"])
            st.markdown("**Missingness after cleaning (encounters)**")
            st.dataframe(enc_c[enc_c["missing"] > 0].round(4).rename(columns={"missing": "share missing"}) if (enc_c["missing"] > 0).any() else pd.DataFrame({"result": ["No missing values in encounter columns"]}), hide_index=True, width="stretch")
        with b:
            st.markdown("**Post-clean validation**")
            ri = dq["post"]["referential_integrity"]
            st.dataframe(pd.DataFrame([("Duplicate encounter ids", dq["post"]["duplicate_encounter_ids"]), ("Temporal violations", dq["post"]["temporal_violations"]), ("Implausible labs", dq["post"]["implausible_labs"])] + list(ri.items()), columns=["Check", "Count"]), hide_index=True, width="stretch")
        mo = pd.DataFrame(dq["post"]["monthly_encounters"]); mo["month"] = pd.to_datetime(mo["month"])
        fig = px.line(mo, x="month", y="encounters", color_discrete_sequence=[ui.BLUE]); fig.update_layout(title="Network encounter volume per month"); fig.update_xaxes(title="")
        st.plotly_chart(ui.style_fig(fig, 280, False), width="stretch")
        st.markdown("**Cohort funnel**")
        st.dataframe(pd.DataFrame([("All encounters after cleaning", funnel["all_encounters"]), ("Excluded: death, hospice or transfer to another acute hospital", funnel["excluded_by_disposition"]),
                                   ("Eligible index discharges", funnel["eligible_index_discharges"]), ("Follow-up incomplete (live worklist, no label)", funnel["follow_up_incomplete_live_worklist"]),
                                   ("Modelling cohort (label observable)", funnel["modeling_cohort"])], columns=["Step", "Discharges"]), hide_index=True, width="stretch")

    with t2:
        st.markdown("Post-deployment-style monitoring on out-of-sample data: case-mix-adjusted facility performance, feature drift and score drift.")
        mo = m["monitoring"]
        foe = data.table("facility_oe.csv").merge(data.facilities()[["facility_id", "facility_name"]], on="facility_id")
        st.dataframe(foe[["facility_id", "facility_name", "n", "observed", "expected", "oe", "oe_low", "oe_high", "z", "status"]].round(2), hide_index=True, width="stretch")
        inj = mo["injected_drift_facility"]
        got = inj in mo["facility_flags"]
        ui.note(f"<b>Ground-truth check:</b> the simulator injected a +0.35 log-odds readmission shift at <b>{inj}</b> from {mo['injected_drift_start']}. The monitor {'flagged' if got else 'did NOT flag'} it"
                f" (flagged: {', '.join(mo['facility_flags']) or 'none'}). Facilities flagged lower than expected may reflect genuine facility effects built into the simulator or multiple-comparison chance (10 facilities x 2 tails).")
        mon = data.table("facility_monthly_oe.csv"); mon["month"] = pd.to_datetime(mon["month"])
        sel = st.multiselect("Facilities", sorted(mon["facility_id"].unique()), default=[inj])
        fig = go.Figure()
        for f in sel:
            r = mon[mon["facility_id"] == f]
            fig.add_trace(go.Scatter(x=r["month"], y=r["oe"], mode="lines+markers", name=f, line=dict(width=2)))
        fig.add_hline(y=1, line_dash="dash", line_color="rgba(128,128,128,.7)"); fig.add_vline(x=pd.Timestamp(mo["injected_drift_start"]), line_dash="dot", line_color=ui.RED, annotation_text="injected change")
        fig.update_yaxes(title="Monthly observed / expected"); fig.update_xaxes(title="")
        st.plotly_chart(ui.style_fig(fig, 340), width="stretch")
        fd = data.table("feature_drift.csv")
        st.markdown(f"**Score drift (PSI, train vs. out-of-sample):** {mo['score_psi_train_vs_test']:.3f}. **Feature drift** (PSI < 0.10 stable, 0.10-0.25 moderate, > 0.25 significant):")
        st.dataframe(fd.head(12).round(3), hide_index=True, width="stretch")

    with t3:
        st.markdown("**Lineage** - every stage records input/output SHA-256 hashes, row counts, library versions and the config fingerprint.")
        rows = []
        for s in prov["stages"]:
            rows.append({"Stage": s["stage"], "Time (UTC)": s["timestamp_utc"], "Outputs": ", ".join(f"{o['path'].split('/')[-1]} ({o['sha256'][:10]})" for o in s["outputs"][:3]), "Config": s["config_fingerprint"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.json(prov.get("environment", {}), expanded=False)
        st.markdown("**Privacy controls implemented and tested**")
        st.markdown("- HMAC-SHA256 keyed pseudonyms for patient, encounter and ED-visit identifiers\n- PHI screen on every incoming table (identifier-like column names and SSN/e-mail/phone patterns) that refuses ingest on a hit\n"
                    "- Per-patient date shifting that preserves intervals (switched on for real data; off here because the dates are synthetic)\n- Age top-coded at 90; no free text, names, addresses or ZIP codes in any table\n"
                    "- Small-cell suppression (n < 11) for every subgroup aggregate shown in this app\n- Simulator ground truth is stored in a separate file and is excluded from the app bundle and from every feature set")
        st.caption(dq["actions"]["pseudonymisation"])

    with t4:
        st.markdown(data.read_doc("MODEL_CARD.md"))
    with t5:
        st.markdown(data.read_doc("GOVERNANCE.md"))
