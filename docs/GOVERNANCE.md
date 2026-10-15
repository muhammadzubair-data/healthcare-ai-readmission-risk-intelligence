# Governance, Monitoring and Responsible Use

> **DEMONSTRATION ONLY. All patient records in this project are synthetic. Model outputs are predictive associations, not clinical causes, and must never be used for real clinical, coverage, or staffing decisions.**

## Roles (suggested)
Clinical owner (accountable for use), data steward (feeds, quality), model owner (retraining, monitoring), privacy officer,
and a care-management lead who owns the workflow the model feeds.

## Human oversight
The model proposes a ranked list. A clinician or care manager decides who is contacted and how. Nothing is automated, and a
patient must never be deprioritised solely because of a score.

## Monitoring that exists in this repo
| Check | Where | Result on this run |
|---|---|---|
| Facility observed/expected with z-scores | `models/monitoring.py` | F07 (1.37, higher than expected), F04 (1.14, higher than expected), F03 (1.11, higher than expected), F05 (0.80, lower than expected); largest O/E 1.37 at F07 |
| Planted effects recovered | `monitoring_validation.csv` | Spearman 0.94 between planted log-odds effect and measured log O/E |
| Injected drift (F07 from 2024-07-01) | `facility_period_oe.csv` | detected |
| Score drift (PSI, train vs test) | `feature_drift.csv` | 0.025 (stable below 0.10) |
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
