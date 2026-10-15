# Public Dataset Assessment

> **DEMONSTRATION ONLY. All patient records in this project are synthetic. Model outputs are predictive associations, not clinical causes, and must never be used for real clinical, coverage, or staffing decisions.**

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
