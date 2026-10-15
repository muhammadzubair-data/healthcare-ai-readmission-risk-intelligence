# Model Card - 30-day readmission risk

> **DEMONSTRATION ONLY. All patient records in this project are synthetic. Model outputs are predictive associations, not clinical causes, and must never be used for real clinical, coverage, or staffing decisions.**

## Intended use
Demonstration of how a hospital network *could* rank discharged patients for transitional-care follow-up. Output is a
probability with an uncertainty band and a three-tier label (Standard / Elevated / High). It supports a human
care-management conversation; it does not make decisions.

## Out of scope
Real patients; individual treatment decisions; denying or rationing care; staff performance evaluation; any causal reading
of a feature ("reducing X will lower readmission").

## Data
Synthetic network of 10 facilities, 89,973 patients, 168,291 labeled index discharges,
2021-2025. See `DATA_CARD.md`.

## Target
Unplanned readmission to any facility in the network within 30 days of discharge. About 15.8%
of test discharges are positive.

## Models
LACE-style heuristic (Platt-scaled), L2 logistic regression (C=0.05), random forest, LightGBM (early stopping on 2024-H1
log loss, best iteration 124), LightGBM with isotonic calibration, and a 10-member patient-bootstrap LightGBM
ensemble used only for risk bands and tier stability.

## Performance (temporal test, 46,231 discharges)
| Model | AUROC (95% CI) | PR-AUC (95% CI) | Recall @ top 10% | Cal. slope | O/E | Brier |
|---|---|---|---|---|---|---|
| LACE-style score | 0.700 (0.692-0.709) | 0.297 (0.283-0.309) | 24.1% | 1.03 | 1.09 | 0.123 |
| Logistic regression | 0.734 (0.726-0.742) | 0.373 (0.355-0.387) | 29.0% | 0.98 | 1.03 | 0.118 |
| Random forest | 0.728 (0.720-0.736) | 0.363 (0.346-0.378) | 28.5% | 1.05 | 1.04 | 0.119 |
| LightGBM | 0.731 (0.723-0.739) | 0.371 (0.352-0.385) | 28.9% | 1.01 | 1.04 | 0.118 |
| LightGBM + isotonic | 0.730 (0.722-0.738) | 0.358 (0.341-0.372) | 28.5% | 0.99 | 1.02 | 0.118 |
| *Oracle (true simulated risk, not a model)* | 0.747 | 0.394 | - | - | - | - |

On the never-seen-patient holdout LightGBM reaches AUROC 0.741 and O/E 1.00.

Grouped 5-fold CV AUROC (train period): logistic regression 0.720, LightGBM
0.718.

Notes:
* Isotonic calibration improves O/E slightly but can lower PR-AUC because it creates tied scores.
* LR is not worse than LightGBM on this simulator. Expect different behaviour on real data.

## Risk tiers
Tier cut-points are validation-period risk quantiles (70th and 90th percentile): High >= 0.316, Elevated >= 0.150.
The ensemble gives a mean 90% band width of 0.043 in probability, and on average 93% of ensemble members agree with the assigned tier.
Bands capture model (estimation) uncertainty only, not outcome randomness or data shift.

## Explainability
Exact TreeSHAP (LightGBM native contributions, checked equal to the `shap` library) and logistic-regression odds ratios.
These describe how the *model* uses inputs. They are associations in the training data, not causes.

## Caveats
* No external validation; synthetic data only.
* Performance by facility, age and diagnosis differs; see `FAIRNESS_REPORT.md`.
* A planted drift at one facility lifts observed readmissions above prediction; the monitor catches it (see `GOVERNANCE.md`).
* Patients with more than 15 prior encounters are truncated in history features (104 patients).
