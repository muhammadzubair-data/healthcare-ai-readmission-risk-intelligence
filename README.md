# Hospital Readmission Risk & Patient Risk Intelligence Platform

> **DEMONSTRATION ONLY. All patient records in this project are synthetic. Model outputs are predictive associations, not clinical causes, and must never be used for real clinical, coverage, or staffing decisions.**

An end-to-end, reproducible analytics platform for 30-day all-cause readmission risk: data engineering with
point-in-time features, validated models, calibration and subgroup checks, facility monitoring, a capacity-constrained
follow-up simulator and a six-page Streamlit app. Built by ZubairLabs as a portfolio project.

**All patient data in this repository are synthetic.** They come from a simulator written for this project
(`src/readmit/synth`), including deliberate data-quality defects and a planted facility drift so the monitoring layer can be tested
against a known answer. No real-world performance claim is made anywhere.

## Headline results (temporal test set, synthetic data)

| | |
|---|---|
| Encounters in modeling cohort | 168,291 (89,973 patients, 10 simulated facilities, 5 years); 151,324 of them fall in the train / validation / test / new-patient splits. The rest are earlier encounters of the held-out patients, which are never used for training |
| 30-day readmission rate | 14.4% overall; 15.8% in the test period |
| Test period | 2024-07-01 to 2025-12-01 (46,231 discharges, 7,311 events) |
| Logistic regression AUROC | 0.734 (0.726-0.742) |
| LightGBM AUROC | 0.731 (0.723-0.739) |
| LACE-style baseline AUROC | 0.700 (0.692-0.709) |
| Ceiling (oracle using true simulated risk) | 0.747 |
| Recall if the top 10% by risk are followed up | 28.9% (lift 2.9x) |
| Calibration (LightGBM test) | slope 1.01, O/E 1.04 |

Two things worth saying plainly:

1. **Logistic regression is marginally ahead of LightGBM** (paired AUROC difference LightGBM minus LR
   -0.0029, 95% CI -0.0044 to -0.0015).
   That is statistically detectable but practically negligible. The simulator's risk is mostly additive, so there is little for
   a tree model to find. This is a property of the synthetic data, not a general statement about the two methods on real records.
2. **The ceiling is low on purpose.** Real readmission models usually reach AUROC around 0.65-0.75. The simulator has
   hidden patient frailty and encounter noise, so even the true risk only reaches 0.747. The models
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
`artifacts/provenance.json`. The config fingerprint in `metrics.json` is `0e35608da71cc5b8`.

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
