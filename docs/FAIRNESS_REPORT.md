# Fairness Assessment

> **DEMONSTRATION ONLY. All patient records in this project are synthetic. Model outputs are predictive associations, not clinical causes, and must never be used for real clinical, coverage, or staffing decisions.**

Subgroup performance on the temporal test set, using LightGBM at the operating point "top 10% of validation risk"
(threshold 0.316). The sensitive attributes are those the simulator carries (age band, insurance, sex,
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
| Attribute | Groups | AUROC range | Recall range at equal selection rate | O/E range | Groups with review flags |
|---|---|---|---|---|---|
| age_band | 6 | 0.152 | 0.103 | 0.79-1.04 | 18-34; 35-49; 50-64; 85+ |
| dx_group | 13 | 0.055 | 0.094 | 0.90-1.08 | COPD; DIABETES; SURGICAL_GI |
| facility_id | 10 | 0.029 | 0.099 | 0.80-1.37 | F05; F06; F07 |
| insurance | 4 | 0.048 | 0.046 | 0.91-1.05 | COMMERCIAL; SELF_PAY |
| ses_quintile | 5 | 0.021 | 0.036 | 0.98-1.04 | none |
| sex | 2 | 0.005 | 0.007 | 1.01-1.03 | none |

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
