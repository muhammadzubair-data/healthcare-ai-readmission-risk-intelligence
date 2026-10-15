# Case Study: Readmission Risk Intelligence for a Hospital Network

> **DEMONSTRATION ONLY. All patient records in this project are synthetic. Model outputs are predictive associations, not clinical causes, and must never be used for real clinical, coverage, or staffing decisions.**

## The question
A Director of Clinical Analytics wants to know which discharged patients carry the highest 30-day readmission risk, why,
how patterns differ across facilities and diagnoses, and what a limited care-management team could achieve if it
focused on the right people.

## Approach
1. **Data.** A simulated 10-facility network (89,973 patients, 173,438 encounters, 2021-2025) with
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
* The best models reach AUROC about 0.73 on the temporal test set, against 0.70 for the LACE-style
  baseline and a ceiling of 0.75. Following up the top 10% by risk reaches 29% of readmissions
  (lift 2.9x).
* Calibration is good on the test period: slope 1.01, O/E 1.04.
* The most influential factors by mean |SHAP|: principal diagnosis group, comorbidity score (charlson-style), inpatient days in prior 12 months, length of stay (days), secondary diagnoses coded, discharge destination. These are associations the model uses.
* Facility monitoring flagged F07 (1.37, higher than expected), F04 (1.14, higher than expected), F03 (1.11, higher than expected), F05 (0.80, lower than expected); F07, where a
  drift was injected from 2024-07-01, is the clear outlier. The measured O/E tracks the planted effects (Spearman
  0.94).

## Care-management scenario
Assumptions (all editable in the app): 3.3 FTE of capacity (1188 minutes/day), three intervention levels
(High: 110 min, 18% assumed relative risk reduction, Elevated: 40 min, 10% assumed relative risk reduction, Standard: 12 min, 4% assumed relative risk reduction), $ 15,000 per
readmission and $ 1.00 per care minute, none of them sourced from real programs.

| Targeting rule | Patients served | Events reached | Avoided readmissions (mean, 5-95%) | Gain vs. random (mean, 5-95%) |
|---|---|---|---|---|
| No targeting (random order) | 20,172 | 36% | 318 (209-439) | +0 (+0 to +0) |
| LACE-style score ranking | 8,202 | 31% | 409 (222-636) | +90 (-11 to +204) |
| Model risk ranking | 6,471 | 31% | 441 (222-698) | +123 (-13 to +277) |
| Assumed benefit per care-manager minute | 8,596 | 35% | 450 (249-695) | +132 (+29 to +249) |

Under these assumptions, ranking by model risk avoids about 124 more readmissions than working the list in random order, and the
"assumed benefit per minute" rule about 132 more. All rules share the same random draws, so the gains are paired. The 5th-95th percentile ranges
for "gain vs. random" still reach below zero for the model-risk and LACE rules, so on this evidence the size of the benefit is uncertain; only the
benefit-per-minute rule stays above zero across its range. Sensitivity of the base case to the assumed intervention effect:

* Base case: about 436 avoided readmissions
* Intervention effect x0.5: about 218 avoided readmissions
* Intervention effect x1.5: about 649 avoided readmissions
* Minutes per patient x0.75: about 547 avoided readmissions
* Minutes per patient x1.25: about 365 avoided readmissions
* Staffing x0.5: about 252 avoided readmissions
* Staffing x2: about 743 avoided readmissions
* Cost per readmission x0.5: about 436 avoided readmissions
* Cost per readmission x1.5: about 436 avoided readmissions

The benefit is **proportional to the assumed effect size**. Treat the scenarios as a way to reason about trade-offs
(coverage vs. depth, pooled vs. facility staffing, equity floors), not as a forecast.

## What this does not show
* Real-world performance; the data are simulated, and the models were not validated on any real hospital.
* That logistic regression and gradient boosting differ little in practice. Here they do not, because the simulator is mostly additive.
* Causal effects of any feature or intervention.

## Next steps for a real deployment
Validate on a local retrospective cohort with a clinical champion, add outside-network capture where possible, test
calibration by facility, run a pilot with a pre-registered comparison group, and put the governance checks in `GOVERNANCE.md` on a schedule.
