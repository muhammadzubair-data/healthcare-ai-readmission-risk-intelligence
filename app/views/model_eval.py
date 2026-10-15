import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import ui

MN = list(ui.MODEL_LABELS)


def _ci(boot, model, key):
    lo, hi = boot["ci"][model][key]
    return f"{lo:.3f}-{hi:.3f}"


def render():
    m = data.metrics()
    st.header("Model evaluation")
    ui.disclaimer()
    sp = m["splits"]
    st.markdown(
        f"**Design.** Trained on discharges up to {sp['train']['last']} ({sp['train']['n']:,}); model choice, early stopping and calibration on {sp['val']['first']} to {sp['val']['last']} ({sp['val']['n']:,}); "
        f"evaluated on the later **temporal test** ({sp['test']['n']:,} discharges, {sp['test']['first']} to {sp['test']['last']}) and on a **new-patient holdout** "
        f"({sp['test_unseen']['n']:,} discharges of patients never seen in training or validation). CIs are patient-clustered bootstrap intervals."
    )
    t_test, t_unseen, t_oracle = st.tabs(["Temporal test", "New-patient holdout", "Reference points"])

    def comp_table(split, boot):
        rows = []
        for k in MN:
            x = m["models"][split][k]
            rows.append({"Model": ui.MODEL_LABELS[k], "AUROC": f"{x['auroc']:.3f}", "AUROC 95% CI": _ci(boot, k, "auroc"), "PR-AUC": f"{x['pr_auc']:.3f}",
                         "Recall @10%": f"{x['recall_at_10pct']:.1%}", "Recall @20%": f"{x['recall_at_20pct']:.1%}", "Brier": f"{x['brier']:.4f}",
                         "Calib. slope": f"{x['calibration_slope']:.2f}", "O/E": f"{x['observed_over_expected']:.2f}", "ECE": f"{x['ece']:.3f}"})
        return pd.DataFrame(rows)

    with t_test:
        st.dataframe(comp_table("test", m["bootstrap_test"]), hide_index=True, width="stretch")
        d = m["bootstrap_test"]["paired_diff_vs_reference"]
        txt = "; ".join(f"{ui.MODEL_LABELS[k]}: {v['auroc'][2]:+.3f} [{v['auroc'][0]:+.3f}, {v['auroc'][1]:+.3f}]" for k, v in d.items())
        ui.note(f"<b>Paired AUROC difference vs. logistic regression</b> (patient-clustered bootstrap): {txt}.")
    with t_unseen:
        st.dataframe(comp_table("test_unseen", m["bootstrap_test_unseen"]), hide_index=True, width="stretch")
        st.caption("Patients here were excluded from all training and validation data, so this checks generalization to new patients. Smaller sample, so wider intervals.")
    with t_oracle:
        o = m["oracle"]["test"]
        cv = m["cv_auroc"]
        st.markdown(
            f"- **Oracle ceiling (simulator truth, unobservable frailty, no hidden noise):** AUROC {o['auroc']:.3f}, recall@10% {o['recall_at_10pct']:.1%}. No model can reach it, since it uses information not present in any record.\n"
            f"- **Grouped 5-fold CV inside the training period (AUROC):** logistic {np.mean(cv['logistic_regression']):.3f} (sd {np.std(cv['logistic_regression']):.3f}), LightGBM {np.mean(cv['lightgbm']):.3f} (sd {np.std(cv['lightgbm']):.3f}).\n"
            f"- **Prevalence in the test period:** {m['models']['test']['lightgbm']['prevalence']:.1%}; a random list would capture 10% of events at a 10% capacity.")
        st.caption("Because the data are simulated with mostly additive effects, the fact that logistic regression is competitive with gradient boosting says nothing about which wins on real EHR data.")

    st.subheader("Discrimination and capacity")
    roc, gains = data.table("roc.csv"), data.table("gains.csv")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for k in MN:
            r = roc[roc["model"] == k]
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], name=ui.MODEL_LABELS[k], mode="lines", line=dict(color=ui.MODEL_COLORS[k], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="rgba(128,128,128,.6)", dash="dash")))
        fig.update_layout(title="ROC curves (temporal test)"); fig.update_xaxes(title="False-positive rate"); fig.update_yaxes(title="True-positive rate")
        st.plotly_chart(ui.style_fig(fig, 400), width="stretch")
    with b:
        fig = go.Figure()
        for k in MN:
            r = gains[gains["model"] == k]
            fig.add_trace(go.Scatter(x=r["fraction_flagged"], y=r["recall"], name=ui.MODEL_LABELS[k], mode="lines", line=dict(color=ui.MODEL_COLORS[k], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="rgba(128,128,128,.6)", dash="dash")))
        fig.update_layout(title="Recall at capacity: share of readmissions captured"); fig.update_xaxes(title="Share of discharges flagged (capacity)", tickformat=".0%"); fig.update_yaxes(title="Recall", tickformat=".0%")
        st.plotly_chart(ui.style_fig(fig, 400), width="stretch")

    st.subheader("Calibration")
    cal = data.table("calibration.csv")
    pick = st.radio("Model", MN, index=4, format_func=ui.MODEL_LABELS.get, horizontal=True)
    c = cal[cal["model"] == pick]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, c[["mean_predicted", "observed_rate"]].values.max() * 1.1], y=[0, c[["mean_predicted", "observed_rate"]].values.max() * 1.1], mode="lines", name="Perfect calibration", line=dict(color="rgba(128,128,128,.7)", dash="dash")))
    fig.add_trace(go.Scatter(x=c["mean_predicted"], y=c["observed_rate"], mode="lines+markers", name=ui.MODEL_LABELS[pick], line=dict(color=ui.MODEL_COLORS[pick], width=2), marker=dict(size=9),
                             error_y=dict(type="data", symmetric=False, array=c["ci_high"] - c["observed_rate"], arrayminus=c["observed_rate"] - c["ci_low"], color="rgba(128,128,128,.6)")))
    fig.update_xaxes(title="Mean predicted risk (decile)", tickformat=".0%"); fig.update_yaxes(title="Observed readmission rate", tickformat=".0%")
    x = m["models"]["test"][pick]
    fig.update_layout(title=f"Reliability (temporal test): slope {x['calibration_slope']:.2f}, O/E {x['observed_over_expected']:.2f}, ECE {x['ece']:.3f}")
    st.plotly_chart(ui.style_fig(fig, 380), width="stretch")
    st.caption("Slope near 1 and O/E near 1 mean the stated risks can be read as probabilities. O/E above 1 means observed readmissions exceed predictions overall - here partly the injected facility change described under Monitoring.")

    st.subheader("Net benefit of acting on flagged patients (decision curve)")
    dca = data.table("decision_curves.csv")
    fig = go.Figure()
    for k in ["lace_style", "logistic_regression", "lightgbm_calibrated"]:
        r = dca[dca["model"] == k]
        fig.add_trace(go.Scatter(x=r["threshold"], y=r["net_benefit"], name=ui.MODEL_LABELS[k], mode="lines", line=dict(color=ui.MODEL_COLORS[k], width=2)))
    r = dca[dca["model"] == "lightgbm"]
    fig.add_trace(go.Scatter(x=r["threshold"], y=r["treat_all"], name="Review everyone", mode="lines", line=dict(color="rgba(128,128,128,.8)", dash="dot")))
    fig.add_trace(go.Scatter(x=r["threshold"], y=r["treat_none"], name="Review no one", mode="lines", line=dict(color="rgba(128,128,128,.8)", dash="dash")))
    fig.update_xaxes(title="Risk threshold for review", tickformat=".0%"); fig.update_yaxes(title="Net benefit", range=[-0.02, 0.12])
    st.plotly_chart(ui.style_fig(fig, 360), width="stretch")
    st.caption("Net benefit trades true positives against false positives at the stated threshold. It assumes the action is worth taking for a true positive; it says nothing about whether a specific intervention works.")

    st.subheader("Stability over time")
    tp = data.table("temporal_performance.csv")
    a, b = st.columns(2)
    with a:
        fig = px.line(tp, x="period", y="auroc", markers=True, color_discrete_sequence=[ui.BLUE]); fig.update_yaxes(title="AUROC", range=[0.6, 0.8]); fig.update_xaxes(title="")
        fig.update_layout(title="AUROC by quarter (out-of-sample)")
        st.plotly_chart(ui.style_fig(fig, 300, False), width="stretch")
    with b:
        fig = px.line(tp, x="period", y="oe", markers=True, color_discrete_sequence=[ui.ORANGE]); fig.add_hline(y=1, line_dash="dash", line_color="rgba(128,128,128,.7)")
        fig.update_yaxes(title="Observed / expected"); fig.update_xaxes(title=""); fig.update_layout(title="Calibration drift by quarter")
        st.plotly_chart(ui.style_fig(fig, 300, False), width="stretch")

    st.subheader("Explainability (SHAP)")
    gi = data.table("shap_global.csv").head(15).iloc[::-1]
    fig = px.bar(gi, x="mean_abs_shap", y="label", orientation="h", color_discrete_sequence=[ui.BLUE])
    fig.update_xaxes(title="Mean |SHAP| (log-odds), 3,000 test discharges"); fig.update_yaxes(title="")
    a, b = st.columns([1.1, 1])
    with a:
        st.plotly_chart(ui.style_fig(fig, 460, False), width="stretch")
    with b:
        ss = data.shap_sample()
        num = [f for f in gi["feature"] if f in ss.columns and pd.api.types.is_numeric_dtype(ss[f])]
        f = st.selectbox("Association shape for a feature", num[::-1] if num else [], format_func=lambda x: gi.set_index("feature").loc[x, "label"])
        if f:
            x = ss[f].to_numpy(dtype=float); s = ss[f"shap__{f}"].to_numpy(); ok = ~np.isnan(x)
            dfp = pd.DataFrame({"x": x[ok], "s": s[ok]})
            nb = min(12, dfp["x"].nunique())
            dfp["bin"] = pd.qcut(dfp["x"], nb, duplicates="drop") if dfp["x"].nunique() > 12 else dfp["x"]
            g = dfp.groupby("bin", observed=True).agg(x=("x", "mean"), s=("s", "mean")).reset_index(drop=True)
            fig = go.Figure(go.Scatter(x=dfp["x"].sample(min(800, len(dfp)), random_state=0), y=dfp["s"].loc[dfp["x"].sample(min(800, len(dfp)), random_state=0).index], mode="markers", marker=dict(size=4, color="rgba(42,120,214,.25)"), name="Patients"))
            fig.add_trace(go.Scatter(x=g["x"], y=g["s"], mode="lines+markers", line=dict(color=ui.ORANGE, width=2.5), name="Binned mean"))
            fig.update_xaxes(title=gi.set_index("feature").loc[f, "label"]); fig.update_yaxes(title="SHAP (log-odds)")
            st.plotly_chart(ui.style_fig(fig, 400), width="stretch")
    ui.note("SHAP explains the <b>model</b>, not the world. Higher contribution = associated with higher estimated risk in the training data. It must not be read as a cause or as advice that changing a feature would reduce risk.")
    with st.expander("Logistic regression odds ratios (per 1 SD for numeric inputs)"):
        orr = data.table("logreg_odds_ratios.csv").head(20)
        st.dataframe(orr.rename(columns={"term": "Term", "coef": "Coefficient", "odds_ratio": "Odds ratio"}).round(3), hide_index=True, width="stretch")

    st.subheader("Subgroup performance and fairness")
    fr, fs = data.table("fairness_subgroups.csv"), data.table("fairness_summary.csv")
    thr = m["fairness"]["threshold"]
    st.markdown(f"Two views of the same top-10% operating point. **At the network threshold** ({thr:.1%}, fixed on validation data) groups with lower base rates are flagged less often, which is expected and shown descriptively. "
                "**At an equal selection rate** (top 10% *within each group*) recall isolates whether the model ranks that group's patients worse. "
                "Review prompts: calibration O/E outside 0.80-1.25; equal-rate recall more than 5 points from overall with an interval that excludes overall; AUROC more than 0.05 below overall "
                "(needs >=100 events / >=500 discharges). Evaluated on all out-of-sample discharges.")
    st.dataframe(fs.round(3), hide_index=True, width="stretch")
    att = st.selectbox("Attribute", sorted(fr["attribute"].unique()))
    g = fr[fr["attribute"] == att].copy()
    g = ui.suppress(g, "n", ["prevalence", "mean_predicted", "observed_over_expected", "auroc", "flag_rate", "recall_at_threshold", "ppv_at_threshold", "recall_equal_rate"])
    cols = ["group", "n", "events", "prevalence", "mean_predicted", "observed_over_expected", "auroc", "auroc_ci_low", "auroc_ci_high", "flag_rate", "recall_at_threshold", "fpr_at_threshold", "ppv_at_threshold", "recall_equal_rate", "recall_eq_ci_low", "recall_eq_ci_high", "review_flags"]
    st.dataframe(g[cols].round(3), hide_index=True, width="stretch")
    fig = go.Figure(go.Scatter(x=g["recall_equal_rate"], y=g["group"], mode="markers", marker=dict(size=11, color=ui.BLUE),
                               error_x=dict(type="data", symmetric=False, array=g["recall_eq_ci_high"] - g["recall_equal_rate"], arrayminus=g["recall_equal_rate"] - g["recall_eq_ci_low"], color="rgba(128,128,128,.7)")))
    fig.add_vline(x=float(g["recall_eq_overall"].iloc[0]), line_dash="dash", line_color="rgba(128,128,128,.8)", annotation_text="overall")
    fig.update_xaxes(title="Recall when the top 10% of each group is flagged (95% CI)", tickformat=".0%")
    st.plotly_chart(ui.style_fig(fig, 320, False), width="stretch")
    ui.note("A flag is a prompt to investigate data, measurement and access-to-care explanations. It is not proof of model bias, and a clean result is not proof of fairness: only the attributes listed here were checked, in simulated data.")
