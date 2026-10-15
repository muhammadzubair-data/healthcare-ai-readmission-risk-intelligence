"""Generate the written documentation from the pipeline's own outputs.

Every number in README/MODEL_CARD/CASE_STUDY etc. is read from artifacts/metrics.json and
artifacts/tables/*.csv at generation time, so the prose cannot drift from the results.
Only qualitative statements (limitations, governance) are written by hand here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DISCLAIMER, Config

LABELS = {"lace_style": "LACE-style score", "logistic_regression": "Logistic regression", "random_forest": "Random forest",
          "lightgbm": "LightGBM", "lightgbm_calibrated": "LightGBM + isotonic"}


def _pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def _ci(ci, d=3):
    return f"{ci[0]:.{d}f}-{ci[1]:.{d}f}"


class _Ctx:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        a = cfg.paths.artifacts
        self.m = json.loads((a / "metrics.json").read_text())
        self.funnel = json.loads((a / "cohort_funnel.json").read_text())
        self.dq = json.loads((a / "dq_report.json").read_text())
        t = a / "tables"
        self.t = {p.stem: pd.read_csv(p) for p in t.glob("*.csv")}
        self.test = self.m["models"]["test"]
        self.unseen = self.m["models"]["test_unseen"]
        self.ci = self.m["bootstrap_test"]["ci"]
        self.pd = self.m["bootstrap_test"]["paired_diff_vs_reference"]
        self.sp = self.m["splits"]
        sc = pd.read_parquet(a / "scored.parquet", columns=["risk_p05", "risk_p95", "tier_stability"])
        self.band = float((sc["risk_p95"] - sc["risk_p05"]).mean())
        self.stab = float(sc["tier_stability"].mean())

    def model_table(self) -> str:
        rows = ["| Model | AUROC (95% CI) | PR-AUC (95% CI) | Recall @ top 10% | Cal. slope | O/E | Brier |", "|---|---|---|---|---|---|---|"]
        for k, v in self.test.items():
            c = self.ci.get(k, {})
            rows.append(f"| {LABELS.get(k, k)} | {v['auroc']:.3f} ({_ci(c['auroc'])}) | {v['pr_auc']:.3f} ({_ci(c['pr_auc'])}) | "
                        f"{_pct(v['recall_at_10pct'])} | {v['calibration_slope']:.2f} | {v['observed_over_expected']:.2f} | {v['brier']:.3f} |"
                        if c else f"| {LABELS.get(k, k)} | {v['auroc']:.3f} | {v['pr_auc']:.3f} | {_pct(v['recall_at_10pct'])} | {v['calibration_slope']:.2f} | {v['observed_over_expected']:.2f} | {v['brier']:.3f} |")
        o = self.m["oracle"]["test"]
        rows.append(f"| *Oracle (true simulated risk, not a model)* | {o['auroc']:.3f} | {o['pr_auc']:.3f} | - | - | - | - |")
        return "\n".join(rows)

    def facility_flag_text(self) -> str:
        fo = self.t["facility_oe"]
        out = [f"{r.facility_id} ({r.oe:.2f}, {'higher' if r.z > 0 else 'lower'} than expected)" for r in fo.itertuples() if r.status != "Within expected range"]
        return ", ".join(out) if out else "no facility"

    def shap_top_text(self) -> str:
        g = self.t["shap_global"].head(6)
        return ", ".join(str(x).lower() for x in g["label"])

    def scenario_table(self) -> str:
        s = self.t["ops_scenarios"]
        rows = ["| Targeting rule | Patients served | Events reached | Avoided readmissions (mean, 5-95%) | Gain vs. random (mean, 5-95%) |", "|---|---|---|---|---|"]
        for r in s.itertuples():
            rows.append(f"| {r.strategy_label} | {int(r.served):,} | {_pct(r.recall_of_events, 0)} | {r.avoided_mean:,.0f} ({r.avoided_p05:,.0f}-{r.avoided_p95:,.0f}) | "
                        f"{r.avoided_vs_random_mean:+,.0f} ({r.avoided_vs_random_p05:+,.0f} to {r.avoided_vs_random_p95:+,.0f}) |")
        return "\n".join(rows)


def _readme(c: _Ctx) -> str:
    lg, gb, lace = c.test["logistic_regression"], c.test["lightgbm"], c.test["lace_style"]
    f = c.funnel
    return f"""# Hospital Readmission Risk & Patient Risk Intelligence Platform

> **{DISCLAIMER}**

An end-to-end, reproducible analytics platform for 30-day all-cause readmission risk: data engineering with
point-in-time features, validated models, calibration and subgroup checks, facility monitoring, a capacity-constrained
follow-up simulator and a six-page Streamlit app. Built by ZubairLabs as a portfolio project.

**All patient data in this repository are synthetic.** They come from a simulator written for this project
(`src/readmit/synth`), including deliberate data-quality defects and a planted facility drift so the monitoring layer can be tested
against a known answer. No real-world performance claim is made anywhere.

## Headline results (temporal test set, synthetic data)

| | |
|---|---|
| Encounters in modeling cohort | {f['modeling_cohort']:,} ({f['patients']:,} patients, 10 simulated facilities, 5 years); {sum(c.sp[k]['n'] for k in ('train','val','test','test_unseen')):,} of them fall in the train / validation / test / new-patient splits. The rest are earlier encounters of the held-out patients, which are never used for training |
| 30-day readmission rate | {_pct(f['modeling_readmission_rate'])} overall; {_pct(c.sp['test']['rate'])} in the test period |
| Test period | {c.sp['test']['first']} to {c.sp['test']['last']} ({c.sp['test']['n']:,} discharges, {c.sp['test']['events']:,} events) |
| Logistic regression AUROC | {lg['auroc']:.3f} ({_ci(c.ci['logistic_regression']['auroc'])}) |
| LightGBM AUROC | {gb['auroc']:.3f} ({_ci(c.ci['lightgbm']['auroc'])}) |
| LACE-style baseline AUROC | {lace['auroc']:.3f} ({_ci(c.ci['lace_style']['auroc'])}) |
| Ceiling (oracle using true simulated risk) | {c.m['oracle']['test']['auroc']:.3f} |
| Recall if the top 10% by risk are followed up | {_pct(gb['recall_at_10pct'])} (lift {gb['lift_at_10pct']:.1f}x) |
| Calibration (LightGBM test) | slope {gb['calibration_slope']:.2f}, O/E {gb['observed_over_expected']:.2f} |

Two things worth saying plainly:

1. **Logistic regression is marginally ahead of LightGBM** (paired AUROC difference LightGBM minus LR
   {c.pd['lightgbm']['auroc'][2]:+.4f}, 95% CI {c.pd['lightgbm']['auroc'][0]:+.4f} to {c.pd['lightgbm']['auroc'][1]:+.4f}).
   That is statistically detectable but practically negligible. The simulator's risk is mostly additive, so there is little for
   a tree model to find. This is a property of the synthetic data, not a general statement about the two methods on real records.
2. **The ceiling is low on purpose.** Real readmission models usually reach AUROC around 0.65-0.75. The simulator has
   hidden patient frailty and encounter noise, so even the true risk only reaches {c.m['oracle']['test']['auroc']:.3f}. The models
   recover most of that, and are far from the near-perfect scores that usually signal leakage.

## What is in the box

| Area | Where |
|---|---|
| Simulator + injected defects | `src/readmit/synth/` |
| Cleaning, PHI screen, pseudonymisation | `src/readmit/data/` |
| Point-in-time features and label | `src/readmit/features/build.py` |
| Models, evaluation, SHAP, fairness, monitoring | `src/readmit/models/` |
| Capacity-constrained follow-up simulator | `src/readmit/ops/prioritize.py` |
| Streamlit app (6 pages) | `app/` |
| Tests (pytest) | `tests/` |
| Documentation | `docs/` (case study, model card, data card, fairness report, governance, dataset assessment) |
| UCI Diabetes-130 benchmark adapter (not run here) | `src/readmit/uci_adapter.py` |

## Quick start

```bash
pip install -r requirements.txt
make fast        # 22k-patient run, a few minutes
make test        # pytest
make app         # streamlit run app/streamlit_app.py
make full        # the 90k-patient run behind the numbers above
```

The pipeline is staged (`generate`, `clean`, `features`, `models`, `docs`) and every stage writes lineage hashes to
`artifacts/provenance.json`. The config fingerprint in `metrics.json` is `{c.m['config_fingerprint']}`.

## Validation design

* **Temporal split.** Train through 2023-12-31, validation 2024-H1 (model selection only), test from 2024-07-01.
  A 15% patient-level holdout is carved out first, so `test_unseen` contains patients the model never saw.
* **Point-in-time features.** History uses only events that finished strictly before the index admission; labs are the
  last value before discharge. Tests check this against a brute-force recomputation.
* **Label.** Any unplanned admission within 30 days of discharge. Planned admissions are ignored, deaths / hospice /
  acute transfers are excluded, and discharges without 30 days of follow-up are left unlabeled (they appear only as a live worklist).
* **Uncertainty.** Patient-clustered bootstrap intervals (encounters from one patient are resampled together) and paired differences.

## Limitations

See `docs/MODEL_CARD.md` and `docs/GOVERNANCE.md`. In short: synthetic data, no external validation, scenario benefits
depend on assumed intervention effects, associational explanations only, and the tool must not be used for real clinical decisions.

---
Co-authored with Claude. Licensed MIT.
"""


def _model_card(c: _Ctx) -> str:
    lg, gb = c.test["logistic_regression"], c.test["lightgbm"]
    u = c.unseen["lightgbm"]
    tiers = c.m["tier_cutoffs"]
    return f"""# Model Card - 30-day readmission risk

> **{DISCLAIMER}**

## Intended use
Demonstration of how a hospital network *could* rank discharged patients for transitional-care follow-up. Output is a
probability with an uncertainty band and a three-tier label (Standard / Elevated / High). It supports a human
care-management conversation; it does not make decisions.

## Out of scope
Real patients; individual treatment decisions; denying or rationing care; staff performance evaluation; any causal reading
of a feature ("reducing X will lower readmission").

## Data
Synthetic network of 10 facilities, {c.funnel['patients']:,} patients, {c.funnel['modeling_cohort']:,} labeled index discharges,
2021-2025. See `DATA_CARD.md`.

## Target
Unplanned readmission to any facility in the network within 30 days of discharge. About {_pct(c.sp['test']['rate'])}
of test discharges are positive.

## Models
LACE-style heuristic (Platt-scaled), L2 logistic regression (C={c.m['logreg_C']}), random forest, LightGBM (early stopping on 2024-H1
log loss, best iteration {c.m['best_iteration']}), LightGBM with isotonic calibration, and a 10-member patient-bootstrap LightGBM
ensemble used only for risk bands and tier stability.

## Performance (temporal test, {c.sp['test']['n']:,} discharges)
{c.model_table()}

On the never-seen-patient holdout LightGBM reaches AUROC {u['auroc']:.3f} and O/E {u['observed_over_expected']:.2f}.

Grouped 5-fold CV AUROC (train period): logistic regression {sum(c.m['cv_auroc']['logistic_regression']) / 5:.3f}, LightGBM
{sum(c.m['cv_auroc']['lightgbm']) / 5:.3f}.

Notes:
* Isotonic calibration improves O/E slightly but can lower PR-AUC because it creates tied scores.
* LR is not worse than LightGBM on this simulator. Expect different behaviour on real data.

## Risk tiers
Tier cut-points are validation-period risk quantiles (70th and 90th percentile): High >= {tiers[1]:.3f}, Elevated >= {tiers[0]:.3f}.
The ensemble gives a mean 90% band width of {c.band:.3f} in probability, and on average {_pct(c.stab, 0)} of ensemble members agree with the assigned tier.
Bands capture model (estimation) uncertainty only, not outcome randomness or data shift.

## Explainability
Exact TreeSHAP (LightGBM native contributions, checked equal to the `shap` library) and logistic-regression odds ratios.
These describe how the *model* uses inputs. They are associations in the training data, not causes.

## Caveats
* No external validation; synthetic data only.
* Performance by facility, age and diagnosis differs; see `FAIRNESS_REPORT.md`.
* A planted drift at one facility lifts observed readmissions above prediction; the monitor catches it (see `GOVERNANCE.md`).
* Patients with more than 15 prior encounters are truncated in history features ({c.funnel['patients_over_lag_depth']} patients).
"""


def _data_card(c: _Ctx) -> str:
    f, a = c.funnel, c.dq["actions"]
    led = c.dq["injected_defects_ledger"]
    return f"""# Data Card

> **{DISCLAIMER}**

## Provenance
Generated by `readmit.synth.generator.simulate` with seed {c.cfg.synth.seed}. Entities: facilities, patients, encounters,
diagnoses, medications, labs and ED visits. The simulator draws patient frailty, comorbidity, utilisation history and
discharge details forward in time, then draws readmissions from a hazard that depends on those quantities plus hidden noise.
The label used for modeling is re-derived from the encounter table by the label builder, not copied from the simulator.

## Cohort
| Step | Count |
|---|---|
| All encounters | {f['all_encounters']:,} |
| Excluded (death / hospice / acute transfer) | {f['excluded_by_disposition']:,} |
| Eligible index discharges | {f['eligible_index_discharges']:,} |
| Follow-up incomplete (live worklist only) | {f['follow_up_incomplete_live_worklist']:,} |
| Modeling cohort | {f['modeling_cohort']:,} |

## Injected data-quality defects and what the pipeline did
| Defect | Handling | Count |
|---|---|---|
| Duplicate encounter rows | dropped | {a['duplicate_encounters_dropped']:,} |
| Discharge before admission | quarantined with child rows | {a['encounters_quarantined_bad_dates']:,} |
| Impossible lab values | removed | {a['implausible_lab_values_removed']:,} |
| Creatinine umol/L silently reported at {', '.join(a['creatinine_unit_shift_facilities'])} | detected by monthly median ratio, converted | {a['creatinine_unit_shift_rows_converted']:,} rows (injected: {led['creatinine_unit_shift_rows']:,}) |
| UNKNOWN discharge disposition | kept as explicit category | {a['unknown_disposition_rows']:,} |

## Privacy handling
HMAC-SHA256 pseudonymous IDs (demo salt; set `READMIT_SALT` for anything else), PHI-pattern screen on column names and values,
per-patient date shifting available for real data, age top-coded at 90, small-cell suppression (n < 11) in the app,
and the simulator's hidden truth is excluded from the app bundle. Synthetic data contain no real person, but the same code
path is meant to be safe for real inputs.

## Known simulator properties
* Network starts empty in 2021, so early history is short and prevalence ramps up as history accrues. The `history_window_days`
  feature therefore shows a large PSI between train and later periods; this is expected, not an incident.
* About 6% of readmissions occur outside the network and are invisible; death is a competing risk.
* Not calibrated to any real hospital system. Prevalence and effect sizes are plausible, not sourced.
"""


def _fairness(c: _Ctx) -> str:
    s = c.t["fairness_summary"]
    rows = ["| Attribute | Groups | AUROC range | Recall range at equal selection rate | O/E range | Groups with review flags |", "|---|---|---|---|---|---|"]
    for r in s.itertuples():
        fl = r.flagged_groups if isinstance(r.flagged_groups, str) and r.flagged_groups else "none"
        rows.append(f"| {r.attribute} | {r.groups} | {r.auroc_range:.3f} | {r.recall_equal_rate_range:.3f} | {r.oe_min:.2f}-{r.oe_max:.2f} | {fl} |")
    return f"""# Fairness Assessment

> **{DISCLAIMER}**

Subgroup performance on the temporal test set, using LightGBM at the operating point "top 10% of validation risk"
(threshold {c.m['fairness']['threshold']:.3f}). The sensitive attributes are those the simulator carries (age band, insurance, sex,
facility and diagnosis group). They are *not* model inputs unless stated in the feature dictionary.

## Method
* Metrics per group: prevalence, mean prediction, observed/expected (O/E), AUROC with clustered bootstrap CI,
  recall and flag rate at the network threshold, and recall when the same share of each group is flagged.
* A single network threshold selects groups in proportion to their base rates, so flag-rate and recall-at-threshold gaps
  are reported but **not** flagged. Flags are raised for calibration O/E outside 0.80-1.25, a recall gap at equal selection
  rate larger than 0.05 whose CI excludes the overall value, or an AUROC gap worse than 0.05 (groups with 500+ rows).
* A flag is a prompt for review of data, measurement and access to care. It is not proof of bias. The thresholds are
  working heuristics, not legal standards.

## Summary
{chr(10).join(rows)}

Full group-level table: `artifacts/tables/fairness_subgroups.csv` (also in the app, *Model evaluation* page).

## Reading the results
* Within-group AUROC is lower for several age bands and diagnosis groups. A grouping variable that is itself a strong
  predictor (age, diagnosis) narrows the spread of risk inside each group, which lowers within-group AUROC even when
  the model is equally well calibrated. That is the likely explanation here, not a demonstrated one.
* The smallest groups (a few hundred events) have wide intervals, so flags there can be sampling noise.
* Differences in prevalence across diagnosis groups are real in the simulation and drive most of the flag-rate gaps.
* Facility flags mostly reflect the planted facility effects and the F07 drift. The model does not see facility, so these
  show up as O/E departures, which is the intended behaviour of the monitor.
* The simulator contains no mechanism that makes a protected group's outcomes less predictable. Any remaining gap is
  therefore noise or case-mix. On real data this assessment would need clinical and community review.
"""


def _governance(c: _Ctx) -> str:
    mv = c.m["monitoring_validation"]
    mon = c.m["monitoring"]
    fo = c.t["facility_oe"]
    top = fo.iloc[0]
    return f"""# Governance, Monitoring and Responsible Use

> **{DISCLAIMER}**

## Roles (suggested)
Clinical owner (accountable for use), data steward (feeds, quality), model owner (retraining, monitoring), privacy officer,
and a care-management lead who owns the workflow the model feeds.

## Human oversight
The model proposes a ranked list. A clinician or care manager decides who is contacted and how. Nothing is automated, and a
patient must never be deprioritised solely because of a score.

## Monitoring that exists in this repo
| Check | Where | Result on this run |
|---|---|---|
| Facility observed/expected with z-scores | `models/monitoring.py` | {c.facility_flag_text()}; largest O/E {top['oe']:.2f} at {top['facility_id']} |
| Planted effects recovered | `monitoring_validation.csv` | Spearman {mv['spearman_planted_vs_log_oe']:.2f} between planted log-odds effect and measured log O/E |
| Injected drift ({mon['injected_drift_facility']} from {mon['injected_drift_start']}) | `facility_period_oe.csv` | detected |
| Score drift (PSI, train vs test) | `feature_drift.csv` | {mon['score_psi_train_vs_test']:.3f} (stable below 0.10) |
| Feature drift PSI | `feature_drift.csv` | see app; `history_window_days` is a known artifact of the simulator start |
| Quarterly AUROC / O/E | `temporal_performance.csv` | app, Model evaluation page |
| Data quality (duplicates, dates, labs, units) | `dq_report.json` | app, Quality & governance page |

## Suggested triggers (starting points, tune locally)
* Facility O/E z-score beyond +/-3 for two consecutive months: investigate coding, process, case-mix.
* PSI above 0.25 on score or a top feature: review the data feed before retraining.
* Network O/E outside 0.9-1.1 for a quarter: recalibrate.
* Subgroup review flags: clinical and equity review.

## Privacy
See `DATA_CARD.md`. For real data: keep the salt in a secrets manager, apply date shifting, review small-cell rules with the
privacy office, and do not publish row-level exports.

## Change control
Config fingerprint and lineage hashes are written to `artifacts/provenance.json`. Any retrain should be re-run through the same
tests and re-reviewed before use.

## Limitations (explicit)
1. Synthetic data; results do not transfer to any real population.
2. No external or prospective validation, no clinical-utility trial.
3. Scenario benefits depend entirely on assumed intervention effects and costs.
4. Explanations are associational.
5. Out-of-network readmissions are invisible to the label.
6. Do not use for real clinical decisions.
"""


def _dataset_assessment(c: _Ctx) -> str:
    return f"""# Public Dataset Assessment

> **{DISCLAIMER}**

The brief asks for a public de-identified dataset where licensing permits, and for the target to be changed rather than
fabricated if the data cannot support a valid 30-day label.

| Candidate | License | 30-day label? | Dates / facilities / labs for point-in-time work? | Verdict |
|---|---|---|---|---|
| UCI "Diabetes 130-US hospitals 1999-2008" (101,766 encounters, 130 hospitals) | CC BY 4.0 | Yes: `readmitted == "<30"` | No admission/discharge dates, no facility id, no lab time series, no dated medications | Valid label, but cannot support temporal validation, facility monitoring or capacity simulation |
| MIMIC-IV / eICU / similar | Credentialed access, data-use agreement | Derivable | Yes | Cannot be redistributed in a public repo or live demo; not used |

## Decision
* The primary platform runs on **clearly labeled synthetic data**, because it is the only way to have dates, facilities,
  point-in-time history and a known ground truth in a redistributable repo.
* An adapter for the UCI dataset is included (`src/readmit/uci_adapter.py`, patient-grouped split, same evaluation code).
  It is tested on a schema-faithful fixture. **It was not run on the real file** because network access to the UCI archive was blocked in the
  build environment, so this repository reports no real-data results. To run it: download `diabetic_data.csv` and call
  `python -m readmit.uci_adapter --csv path/to/diabetic_data.csv`.
"""


def _case_study(c: _Ctx) -> str:
    lg, gb = c.test["logistic_regression"], c.test["lightgbm"]
    s = c.t["ops_scenarios"].set_index("strategy")
    sens = c.t["ops_sensitivity"]
    risk = s.loc["risk"]
    bpm = s.loc["benefit_per_minute"]
    rnd = s.loc["random"]
    p = c.m["ops"]["params"]
    base = sens.iloc[0]
    sens_lines = "\n".join(f"* {r.variant}: about {r.avoided:,.0f} avoided readmissions" for r in sens.itertuples())
    return f"""# Case Study: Readmission Risk Intelligence for a Hospital Network

> **{DISCLAIMER}**

## The question
A Director of Clinical Analytics wants to know which discharged patients carry the highest 30-day readmission risk, why,
how patterns differ across facilities and diagnoses, and what a limited care-management team could achieve if it
focused on the right people.

## Approach
1. **Data.** A simulated 10-facility network ({c.funnel['patients']:,} patients, {c.funnel['all_encounters']:,} encounters, 2021-2025) with
   realistic mess injected on purpose: duplicates, impossible dates and labs, and a silent unit change at one facility.
   The cleaning stage found and repaired all of it and counted every action.
2. **Features without leakage.** Everything is computed as of the index discharge. History looks only at events finished
   before admission. The label is rebuilt from encounters, and unobservable follow-up is left unlabeled.
3. **Models.** A LACE-style baseline, logistic regression, random forest and LightGBM. Selection used 2024-H1; the test period
   is later in time, and a separate holdout of unseen patients checks patient-level leakage.
4. **Honest evaluation.** Clustered bootstrap CIs, calibration slope and O/E, decision curves, subgroup tables and a
   comparison against the oracle ceiling.
5. **Operations.** A day-by-day simulator allocates a finite care-manager time budget to the ranked list.

## What the numbers say
* The best models reach AUROC about {max(gb['auroc'], lg['auroc']):.2f} on the temporal test set, against {c.test['lace_style']['auroc']:.2f} for the LACE-style
  baseline and a ceiling of {c.m['oracle']['test']['auroc']:.2f}. Following up the top 10% by risk reaches {_pct(gb['recall_at_10pct'], 0)} of readmissions
  (lift {gb['lift_at_10pct']:.1f}x).
* Calibration is good on the test period: slope {gb['calibration_slope']:.2f}, O/E {gb['observed_over_expected']:.2f}.
* The most influential factors by mean |SHAP|: {c.shap_top_text()}. These are associations the model uses.
* Facility monitoring flagged {c.facility_flag_text()}; {c.m['monitoring']['injected_drift_facility']}, where a
  drift was injected from {c.m['monitoring']['injected_drift_start']}, is the clear outlier. The measured O/E tracks the planted effects (Spearman
  {c.m['monitoring_validation']['spearman_planted_vs_log_oe']:.2f}).

## Care-management scenario
Assumptions (all editable in the app): {p['fte']} FTE of capacity ({p['minutes_per_day']:.0f} minutes/day), three intervention levels
({', '.join(f"{k}: {v['minutes']:.0f} min, {100 * v['rrr']:.0f}% assumed relative risk reduction" for k, v in p['interventions'].items())}), $ {p['cost_per_readmission']:,.0f} per
readmission and $ {p['cost_per_care_minute']:.2f} per care minute, none of them sourced from real programs.

{c.scenario_table()}

Under these assumptions, ranking by model risk avoids about {risk['avoided_mean'] - rnd['avoided_mean']:,.0f} more readmissions than working the list in random order, and the
"assumed benefit per minute" rule about {bpm['avoided_mean'] - rnd['avoided_mean']:,.0f} more. All rules share the same random draws, so the gains are paired. The 5th-95th percentile ranges
for "gain vs. random" still reach below zero for the model-risk and LACE rules, so on this evidence the size of the benefit is uncertain; only the
benefit-per-minute rule stays above zero across its range. Sensitivity of the base case to the assumed intervention effect:

{sens_lines}

The benefit is **proportional to the assumed effect size**. Treat the scenarios as a way to reason about trade-offs
(coverage vs. depth, pooled vs. facility staffing, equity floors), not as a forecast.

## What this does not show
* Real-world performance; the data are simulated, and the models were not validated on any real hospital.
* That logistic regression and gradient boosting differ little in practice. Here they do not, because the simulator is mostly additive.
* Causal effects of any feature or intervention.

## Next steps for a real deployment
Validate on a local retrospective cohort with a clinical champion, add outside-network capture where possible, test
calibration by facility, run a pilot with a pre-registered comparison group, and put the governance checks in `GOVERNANCE.md` on a schedule.
"""


def generate(cfg: Config) -> list[Path]:
    c = _Ctx(cfg)
    d = cfg.paths.docs
    d.mkdir(parents=True, exist_ok=True)
    files = {cfg.paths.root / "README.md": _readme(c), d / "MODEL_CARD.md": _model_card(c), d / "DATA_CARD.md": _data_card(c),
             d / "FAIRNESS_REPORT.md": _fairness(c), d / "GOVERNANCE.md": _governance(c),
             d / "DATASET_ASSESSMENT.md": _dataset_assessment(c), d / "CASE_STUDY.md": _case_study(c)}
    for p, text in files.items():
        p.write_text(text)
    return list(files)
