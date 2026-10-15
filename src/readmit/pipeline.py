"""End-to-end, reproducible pipeline.

    python -m readmit.pipeline all                 # full run (design-target scale)
    python -m readmit.pipeline all --fast          # small run used in CI
    python -m readmit.pipeline generate|clean|features|models|docs

Every stage is deterministic given the config seed and records hashes in
artifacts/provenance.json.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import provenance as prov
from .clinical import age_band, lace_style_score
from .config import DISCLAIMER, Config, ModelConfig, SynthConfig
from .data import ingest, privacy
from .features import build as fb
from .models import evaluate as ev
from .models import explain, fairness, monitoring, train as tr, uncertainty
from .ops import prioritize as ops
from .synth import generator

warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
FAIRNESS_ATTRS = ["sex", "age_band", "insurance", "ses_quintile", "dx_group", "facility_id"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# =============================================================================== stage 1
def run_generate(cfg: Config) -> None:
    cfg.paths.ensure()
    _log(f"Generating synthetic network: {cfg.synth.n_patients:,} patients, seed {cfg.synth.seed}")
    tables = generator.simulate(cfg.synth)
    truth = tables.pop("_truth")
    raw, ledger = generator.inject_data_quality_issues(tables, cfg.synth)
    paths = ingest.save_tables(raw, cfg.paths.raw)
    tp = cfg.paths.raw / "_synthetic_truth.parquet"
    truth.to_parquet(tp, index=False)
    ledger_pub = {k: v for k, v in ledger.items() if not k.startswith("_")}
    _write_json(ledger_pub, cfg.paths.artifacts / "injected_defects_ledger.json")
    _write_json({"bad_date_encounter_ids": ledger["_bad_date_encounter_ids"]}, cfg.paths.raw / "_bad_date_ids.json")
    pv = prov.Provenance(cfg.paths.artifacts / "provenance.json", cfg.fingerprint(), cfg.paths.root)
    pv.record("1_generate", [], paths + [tp], notes={"synthetic": True, "seed": cfg.synth.seed, "injected_defects": ledger_pub},
              row_counts=prov.table_row_counts({**raw, "_truth": truth}))
    _log(f"  encounters={len(raw['encounters']):,} labs={len(raw['labs']):,} (raw, with injected defects)")


# =============================================================================== stage 2
def run_clean(cfg: Config) -> None:
    _log("Cleaning, validating, pseudonymising")
    raw = ingest.load_tables(cfg.paths.raw)
    clean, report = ingest.clean(raw, cfg)
    ledger = json.loads((cfg.paths.artifacts / "injected_defects_ledger.json").read_text())
    report["injected_defects_ledger"] = ledger
    paths = ingest.save_tables(clean, cfg.paths.interim)
    ingest.write_report(report, cfg.paths.artifacts / "dq_report.json")
    pv = prov.Provenance(cfg.paths.artifacts / "provenance.json", cfg.fingerprint(), cfg.paths.root)
    pv.record("2_clean", [cfg.paths.raw / f"{n}.parquet" for n in ingest.TABLES], paths + [cfg.paths.artifacts / "dq_report.json"],
              notes=report["actions"], row_counts=prov.table_row_counts(clean))
    _log(f"  actions: { {k: v for k, v in report['actions'].items() if isinstance(v, int)} }")


# =============================================================================== stage 3
def run_features(cfg: Config) -> None:
    _log("Building cohort, labels and point-in-time features")
    t = ingest.load_tables(cfg.paths.interim)
    data_start = pd.Timestamp(cfg.synth.window_start)
    data_end = pd.Timestamp(cfg.synth.window_end) + pd.Timedelta(days=1)
    feat = fb.build_features(t, cfg.cohort, data_start, data_end)
    out = cfg.paths.processed / "features.parquet"
    feat.to_parquet(out, index=False)
    pd.DataFrame([dict(feature=k, group=v[0], description=v[1]) for k, v in fb.FEATURE_DICTIONARY.items()]).to_csv(
        cfg.paths.artifacts / "feature_dictionary.csv", index=False)
    enc_per_patient = feat.groupby("patient_id").size()
    funnel = {
        "all_encounters": int(len(feat)),
        "excluded_by_disposition": int((~feat["in_cohort"]).sum()),
        "excluded_disposition_breakdown": feat.loc[~feat["in_cohort"], "excluded_reason"].value_counts().to_dict(),
        "eligible_index_discharges": int(feat["in_cohort"].sum()),
        "follow_up_incomplete_live_worklist": int((feat["in_cohort"] & ~feat["label_observable"]).sum()),
        "modeling_cohort": int((feat["in_cohort"] & feat["label_observable"]).sum()),
        "modeling_readmission_rate": float(feat.loc[feat["in_cohort"] & feat["label_observable"], "readmit_30d"].mean()),
        "patients": int(feat["patient_id"].nunique()),
        "patients_over_lag_depth": int((enc_per_patient > fb.MAX_LAGS).sum()),
        "max_encounters_per_patient": int(enc_per_patient.max()),
    }
    _write_json(funnel, cfg.paths.artifacts / "cohort_funnel.json")
    pv = prov.Provenance(cfg.paths.artifacts / "provenance.json", cfg.fingerprint(), cfg.paths.root)
    pv.record("3_features", [cfg.paths.interim / f"{n}.parquet" for n in ingest.TABLES], [out],
              notes={"prediction_time": "index discharge", "history_rule": "events finished strictly before admission", **funnel},
              row_counts={"features": len(feat)})
    _log(f"  modeling cohort {funnel['modeling_cohort']:,} (readmission rate {funnel['modeling_readmission_rate']:.3f}); live worklist {funnel['follow_up_incomplete_live_worklist']:,}")


# =============================================================================== stage 4
def run_models(cfg: Config) -> None:
    A = cfg.paths.artifacts
    TB = A / "tables"
    TB.mkdir(parents=True, exist_ok=True)
    feat = pd.read_parquet(cfg.paths.processed / "features.parquet")
    mc = fb.modeling_cohort(feat)
    live = fb.scoring_cohort(feat)
    _log(f"Modeling cohort {len(mc):,}; live worklist {len(live):,}")

    splits = tr.make_splits(mc, cfg)
    for k in ("train", "val", "test", "test_unseen"):
        _log(f"  split {k}: {int(splits[k].sum()):,} rows, event rate {mc.loc[splits[k], 'readmit_30d'].mean():.3f}")
    bundle = tr.train_all(mc, splits, cfg, log=_log)
    tr.save_bundle(bundle, cfg.paths.models)

    # ---- predictions on all non-training rows (val/test/test_unseen/live)
    def score(df: pd.DataFrame, split_name) -> pd.DataFrame:
        P = bundle.predict(df)
        out = df.copy()
        out["split"] = split_name
        for name, p in P.items():
            out[f"p_{name}"] = p
        out["risk"] = P["lightgbm_calibrated"]
        out["lace_score"] = lace_style_score(df["los_days"], df["acute_admission"], df["charlson_score"], df["ed_visits_prior_6m"])
        out["tier"] = uncertainty.assign_tier(out["risk"].to_numpy(), bundle.tier_cutoffs)
        return out

    parts = [score(mc[splits[k]], k) for k in ("val", "test", "test_unseen")] + [score(live, "live")]
    scored = pd.concat(parts, ignore_index=True)
    scored["age_band"] = age_band(scored["age"]).astype(str).to_numpy()
    scored["ses_group"] = "Q" + scored["ses_quintile"].astype(int).astype(str)

    # ---- uncertainty ensemble
    X_sc = tr.prep_frame(scored, bundle.levels)
    members = uncertainty.ensemble_predict(bundle.ensemble, X_sc, transform=bundle.isotonic.predict)
    summ = uncertainty.summarize_ensemble(members, scored["tier"].to_numpy(), bundle.tier_cutoffs)
    for c in summ.columns:
        scored[c] = summ[c].to_numpy()
    _log(f"  ensemble: mean 90% band width {float((scored['risk_p95'] - scored['risk_p05']).mean()):.3f}; mean tier stability {scored['tier_stability'].mean():.2f}")

    # ---- oracle ceiling (simulator truth; never a feature)
    truth = pd.read_parquet(cfg.paths.raw / "_synthetic_truth.parquet")
    truth["encounter_id"] = privacy.pseudonymize(truth["encounter_id"], "EN_")
    scored = scored.merge(truth[["encounter_id", "true_risk"]], on="encounter_id", how="left")

    oos = scored[scored["split"].isin(["test", "test_unseen"])]
    metrics: dict = {"disclaimer": DISCLAIMER, "config_fingerprint": cfg.fingerprint(), "tier_cutoffs": list(bundle.tier_cutoffs),
                     "best_iteration": bundle.best_iteration, "logreg_C": bundle.logreg_C, "cv_auroc": bundle.cv,
                     "selection_log": bundle.selection_log, "splits": {}, "models": {}, "oracle": {}}
    for k in ("train", "val", "test", "test_unseen"):
        s = mc[splits[k]]
        metrics["splits"][k] = dict(n=int(len(s)), events=int(s["readmit_30d"].sum()), rate=float(s["readmit_30d"].mean()),
                                    first=str(s["discharge_ts"].min().date()), last=str(s["discharge_ts"].max().date()), patients=int(s["patient_id"].nunique()))
    metrics["splits"]["live"] = dict(n=int(len(live)), first=str(live["discharge_ts"].min().date()), last=str(live["discharge_ts"].max().date()))

    # ---- metrics: val, test, test_unseen for each model
    for split in ("val", "test", "test_unseen"):
        d = scored[scored["split"] == split]
        metrics["models"][split] = {m: ev.core_metrics(d["readmit_30d"], d[f"p_{m}"], cfg.model.capacity_fracs) for m in tr.MODEL_NAMES}
        o = d[d["true_risk"].notna()]
        metrics["oracle"][split] = ev.core_metrics(o["readmit_30d"], o["true_risk"], cfg.model.capacity_fracs)
    _log("  test AUROC: " + ", ".join(f"{m}={metrics['models']['test'][m]['auroc']:.3f}" for m in tr.MODEL_NAMES) + f" | oracle={metrics['oracle']['test']['auroc']:.3f}")

    # ---- patient-clustered bootstrap CIs on the primary test set
    dt = scored[scored["split"] == "test"]
    boot = ev.cluster_bootstrap_metrics(dt["readmit_30d"], {m: dt[f"p_{m}"].to_numpy() for m in tr.MODEL_NAMES}, dt["patient_id"],
                                        n_boot=cfg.model.n_bootstrap, seed=cfg.model.seed, capacity_frac=0.10, reference="logistic_regression")
    metrics["bootstrap_test"] = boot
    du = scored[scored["split"] == "test_unseen"]
    metrics["bootstrap_test_unseen"] = ev.cluster_bootstrap_metrics(
        du["readmit_30d"], {m: du[f"p_{m}"].to_numpy() for m in tr.MODEL_NAMES}, du["patient_id"],
        n_boot=max(cfg.model.n_bootstrap // 2, 20), seed=cfg.model.seed + 1, capacity_frac=0.10, reference="logistic_regression")

    # ---- calibration, gains, decision curves (primary test set)
    cal_rows, gain_rows, dca_rows = [], [], []
    for m in tr.MODEL_NAMES:
        c = ev.calibration_table(dt["readmit_30d"], dt[f"p_{m}"]); c["model"] = m; cal_rows.append(c)
        g = ev.gains_curve(dt["readmit_30d"], dt[f"p_{m}"]); g["model"] = m; gain_rows.append(g)
        dc = ev.decision_curve(dt["readmit_30d"], dt[f"p_{m}"]); dc["model"] = m; dca_rows.append(dc)
    pd.concat(cal_rows).to_csv(TB / "calibration.csv", index=False)
    pd.concat(gain_rows).to_csv(TB / "gains.csv", index=False)
    pd.concat(dca_rows).to_csv(TB / "decision_curves.csv", index=False)

    # ---- temporal stability, ROC points
    vt = scored[scored["split"].isin(["val", "test", "test_unseen"])]
    monitoring.temporal_performance(vt[vt["split"] != "test_unseen"], "risk").to_csv(TB / "temporal_performance.csv", index=False)
    from sklearn.metrics import roc_curve
    roc_rows = []
    for m in tr.MODEL_NAMES:
        fpr, tpr, _ = roc_curve(dt["readmit_30d"], dt[f"p_{m}"])
        idx = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
        roc_rows.append(pd.DataFrame({"fpr": fpr[idx], "tpr": tpr[idx], "model": m}))
    pd.concat(roc_rows).to_csv(TB / "roc.csv", index=False)

    # ---- SHAP
    rng = np.random.default_rng(cfg.model.seed)
    samp = dt.iloc[rng.choice(len(dt), size=min(3000, len(dt)), replace=False)].reset_index(drop=True)
    sv, base = bundle.shap_contributions(samp)
    gi = explain.global_importance(sv, samp)
    gi.to_csv(TB / "shap_global.csv", index=False)
    shap_df = pd.concat([samp[["encounter_id"] + fb.FEATURE_COLUMNS].reset_index(drop=True),
                         pd.DataFrame(sv.astype(np.float32), columns=[f"shap__{f}" for f in fb.FEATURE_COLUMNS])], axis=1)
    for c in fb.CATEGORICAL_FEATURES:
        shap_df[c] = shap_df[c].astype(str)
    shap_df.to_parquet(A / "shap_sample.parquet", index=False)
    metrics["shap"] = {"base_value_logodds": float(base[0]), "sample_n": int(len(samp)),
                       "native_vs_shap_library_max_abs_diff": explain.explainer_sanity(bundle, samp)}
    explain.logistic_odds_ratios(bundle).to_csv(TB / "logreg_odds_ratios.csv", index=False)
    dep = {}
    for f in gi[gi["feature"].isin(fb.NUMERIC_FEATURES)]["feature"].head(8):
        dep[f] = explain.binned_dependence(sv, samp, f).to_dict(orient="records")
    _write_json(dep, TB / "shap_dependence.json")

    # ---- fairness / subgroup performance (operating point = top 10% on validation)
    thr = bundle.tier_cutoffs[1]
    oosf = oos.copy()
    fr_table, fr_sum = fairness.fairness_report(oosf.assign(ses_quintile=oosf["ses_group"]), FAIRNESS_ATTRS, "readmit_30d", "risk", thr,
                                                n_boot=100 if cfg.model.n_bootstrap >= 100 else 20, seed=cfg.model.seed)
    fr_table.to_csv(TB / "fairness_subgroups.csv", index=False)
    fr_sum.to_csv(TB / "fairness_summary.csv", index=False)
    metrics["fairness"] = {"threshold": float(thr), "operating_point": "top 10% of validation risk", "rows": int(len(oosf))}

    # ---- monitoring: facility O/E, drift, scores PSI
    facility_oe = monitoring.facility_oe(oos, "risk")
    facility_oe.to_csv(TB / "facility_oe.csv", index=False)
    monitoring.facility_monthly_oe(scored[scored["split"].isin(["val", "test", "test_unseen"])], "risk").to_csv(TB / "facility_monthly_oe.csv", index=False)
    monitoring.facility_period_oe(scored[scored["split"].isin(["val", "test", "test_unseen"])], cfg.synth.drift_start, "risk").to_csv(TB / "facility_period_oe.csv", index=False)
    train_df = mc[splits["train"]]
    fd = monitoring.feature_drift(train_df, oos)
    fd.to_csv(TB / "feature_drift.csv", index=False)
    psi_score = float(__import__("readmit.data.quality", fromlist=["x"]).population_stability_index(
        bundle.predict(train_df)["lightgbm_calibrated"], oos["risk"].to_numpy()))
    planted = generator.facility_effects(cfg.synth)
    planted.loc[cfg.synth.drift_facility] += cfg.synth.drift_logit_shift  # applies to the whole held-out period
    val = facility_oe.set_index("facility_id")[["oe", "z"]].join(planted.rename("planted_logodds_effect"))
    val["log_oe"] = np.log(val["oe"])
    from scipy.stats import spearmanr
    rho = float(spearmanr(val["planted_logodds_effect"], val["log_oe"]).statistic)
    val.reset_index().to_csv(TB / "monitoring_validation.csv", index=False)
    metrics["monitoring_validation"] = {"spearman_planted_vs_log_oe": rho, "n_facilities": int(len(val))}
    metrics["monitoring"] = {"score_psi_train_vs_test": psi_score, "facility_flags": facility_oe[facility_oe["status"] != "Within expected range"]["facility_id"].tolist(),
                             "injected_drift_facility": cfg.synth.drift_facility, "injected_drift_start": cfg.synth.drift_start}

    # ---- care-management scenarios
    pool = ops.build_pool(scored)
    base_cov = 0.35  # default: staff can deliver ~35% of the full-service workload
    params = ops.ProgramParams(fte=ops.fte_for_coverage(pool, base_cov))
    scen = ops.run_scenario(pool, list(ops.STRATEGIES), params, seed=cfg.model.seed, n_draws=1000)
    scen.to_csv(TB / "ops_scenarios.csv", index=False)
    fte_rows = []
    for cov in (0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
        fte = ops.fte_for_coverage(pool, cov)
        r = ops.run_scenario(pool, ["random", "lace", "risk", "benefit_per_minute"], replace(params, fte=fte), seed=cfg.model.seed, n_draws=400)
        r["fte"] = fte
        r["workload_coverage"] = cov
        fte_rows.append(r)
    pd.concat(fte_rows).to_csv(TB / "ops_capacity_curve.csv", index=False)
    ops.sensitivity(pool, "risk", params, cfg.model.seed).to_csv(TB / "ops_sensitivity.csv", index=False)
    served = ops.allocate(pool, "risk", params, cfg.model.seed)
    cov = pd.concat([ops.coverage_by_group(pool, served, c).assign(attribute=c) for c in ("insurance", "ses_group", "age_band", "facility_id")])
    cov.to_csv(TB / "ops_coverage_risk_ranked.csv", index=False)
    served_eq = ops.allocate(pool, "risk", replace(params, equity_col="insurance", equity_floor=0.5), cfg.model.seed)
    cov_eq = ops.coverage_by_group(pool, served_eq, "insurance").assign(attribute="insurance")
    cov_eq.to_csv(TB / "ops_coverage_equity_floor.csv", index=False)
    r_eq = ops.simulate_outcomes(pool, served_eq, replace(params, equity_col="insurance", equity_floor=0.5), 1000, cfg.model.seed)
    r_eq.pop("events_by_tier_served", None)
    r_eq.pop("_expected_draws", None)
    metrics["ops"] = {"params": {"fte": params.fte, "minutes_per_day": params.minutes_per_day,
                                 "interventions": {k: asdict(v) for k, v in params.interventions.items()},
                                 "cost_per_readmission": params.cost_per_readmission, "cost_per_care_minute": params.cost_per_care_minute},
                      "pool_discharges": int(len(pool)), "pool_days": int(pool["day"].nunique()), "default_workload_coverage": base_cov,
                      "full_service_minutes_per_day": ops.full_service_minutes_per_day(pool),
                      "equity_floor_insurance_0.5": r_eq}

    # ---- artifacts for the application
    keep_cols = ["encounter_id", "patient_id", "facility_id", "discharge_ts", "admit_ts", "split", "readmit_30d", "days_to_readmit",
                 "died_within_30d", "risk", "risk_p05", "risk_p95", "tier", "tier_stability", "lace_score", "age_band", "ses_group", "true_risk"] + [f"p_{m}" for m in tr.MODEL_NAMES]
    keep_cols += [c for c in fb.FEATURE_COLUMNS if c not in keep_cols]
    sc_out = scored[keep_cols].copy()
    for c in fb.CATEGORICAL_FEATURES:
        sc_out[c] = sc_out[c].astype(str)
    sc_out = sc_out.drop(columns=["true_risk"])  # simulator truth stays out of the app bundle
    sc_out.to_parquet(A / "scored.parquet", index=False)

    split_of = pd.Series("excluded", index=feat["encounter_id"])
    split_of.loc[mc.loc[splits["train"], "encounter_id"]] = "train"
    for k in ("val", "test", "test_unseen"):
        split_of.loc[mc.loc[splits[k], "encounter_id"]] = k
    split_of.loc[live["encounter_id"]] = "live"
    eda = feat[["encounter_id", "patient_id", "facility_id", "discharge_ts", "dx_group", "age", "sex", "insurance", "ses_quintile", "los_days",
                "discharge_disposition", "admission_type", "icu_stay", "charlson_score", "prior_admissions_12m", "readmit_30d", "days_to_readmit", "in_cohort"]].copy()
    eda["split"] = split_of.reindex(eda["encounter_id"]).to_numpy()
    eda["age_band"] = age_band(eda["age"]).astype(str).to_numpy()
    eda["month"] = eda["discharge_ts"].dt.to_period("M").dt.to_timestamp()
    for c in ("dx_group", "sex", "insurance", "discharge_disposition", "admission_type", "age_band", "split"):
        eda[c] = eda[c].astype(str)
    eda = eda.drop(columns=["age"])
    eda.to_parquet(A / "eda_encounters.parquet", index=False)
    t = ingest.load_tables(cfg.paths.interim, ["facilities"])
    t["facilities"].to_csv(A / "facilities.csv", index=False)
    train_ref = train_df.sample(min(20000, len(train_df)), random_state=cfg.model.seed)
    train_ref[[c for c in fb.NUMERIC_FEATURES if not c.endswith("_missing")]].astype(np.float32).to_parquet(A / "train_reference_sample.parquet", index=False)

    _write_json(metrics, A / "metrics.json")
    pv = prov.Provenance(A / "provenance.json", cfg.fingerprint(), cfg.paths.root)
    pv.record("4_models", [cfg.paths.processed / "features.parquet"],
              [A / "metrics.json", A / "scored.parquet", A / "eda_encounters.parquet", cfg.paths.models / "lightgbm.txt", cfg.paths.models / "meta.json"],
              notes={"model_selection": "validation period only; test period untouched until final scoring",
                     "tier_cutoffs": list(bundle.tier_cutoffs)}, row_counts={"scored": len(sc_out), "eda": len(eda)})
    _log("Models stage complete")


# =============================================================================== orchestration
def fast_config(base: Config) -> Config:
    return replace(base, synth=replace(base.synth, n_patients=22_000),
                   model=replace(base.model, ensemble_size=4, n_bootstrap=40, max_rounds=400))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["all", "generate", "clean", "features", "models", "docs"])
    ap.add_argument("--fast", action="store_true", help="small configuration for CI/smoke tests")
    ap.add_argument("--patients", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = Config()
    if a.fast:
        cfg = fast_config(cfg)
    if a.patients:
        cfg = replace(cfg, synth=replace(cfg.synth, n_patients=a.patients))
    if a.seed is not None:
        cfg = replace(cfg, synth=replace(cfg.synth, seed=a.seed), model=replace(cfg.model, seed=a.seed))
    cfg.paths.ensure()
    t0 = time.time()
    if a.stage in ("all", "generate"):
        run_generate(cfg)
    if a.stage in ("all", "clean"):
        run_clean(cfg)
    if a.stage in ("all", "features"):
        run_features(cfg)
    if a.stage in ("all", "models"):
        run_models(cfg)
    if a.stage in ("all", "docs"):
        from . import docs_gen
        docs_gen.generate(cfg)
    _log(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
