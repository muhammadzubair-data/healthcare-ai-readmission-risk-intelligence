# Project 014 — Technical Audit

## Scope
Audited the supplied `hospital-readmission-risk-platform.zip` as Project 014: AI-Powered Hospital Readmission Prediction & Patient Risk Intelligence. This is a **synthetic-data portfolio demonstration**, not a validated clinical tool.

## Checks performed
- ZIP contents inspected: 116 entries in the source archive; approximately 34.7 MB uncompressed.
- Python source, app, scripts and tests passed `compileall` syntax checks.
- `PYTHONPATH=src python -m pytest -q`: **69 passed, 2 failed due to unavailable optional runtime dependencies** in the audit environment:
  - `test_every_app_page_renders_without_exception`: `streamlit` not installed.
  - `test_app_bundle_has_no_identifiers_or_hidden_frailty`: no `pyarrow` or `fastparquet` installed.
- Both dependencies are specified in `requirements.txt`; the failures do not prove an application defect, but full test success remains unverified.
- CI at `.github/workflows/ci.yml` installs requirements, runs tests, runs the fast pipeline, and smoke-tests the app.
- Streamlit entry point: `app/streamlit_app.py`.

## Reported project outcomes — not independently reproduced
README reports a synthetic-data temporal-test AUROC of 0.734 for logistic regression and 0.731 for LightGBM, 28.9% recall in the top 10% risk-ranked encounters, and 1.04 observed/expected calibration for LightGBM. These results should be independently rerun before representing them as verified by an external auditor.

## Publication and safety requirements
- Keep synthetic data and **not for clinical use** disclosures prominent in README, app, portfolio posts, and screenshots.
- Do not present model associations as clinical causes, nor modeled follow-up benefits as realized patient outcomes.
- Do not use this model for actual patient care, coverage, or staffing decisions.
- Review `docs/MODEL_CARD.md`, `docs/GOVERNANCE.md`, and `docs/DATA_CARD.md` before public release.
- Run GitHub Actions to confirm full tests in a clean Python 3.12 environment; review any failure logs.
- Deploy using `app/streamlit_app.py`; test all six pages and the bundled artifacts after deployment.
- No independent privacy certification, external validation, or full pipeline rerun was performed in this audit.

## GitHub publication suggestion
Repository: `healthcare-ai-readmission-risk-intelligence`.
Description: `Synthetic-data healthcare AI platform for 30-day readmission risk modeling, calibration, fairness monitoring, and care-management scenario analysis.`
