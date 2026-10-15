"""App smoke test and artifact/documentation consistency (needs a completed pipeline run)."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "metrics.json").exists(), reason="run the pipeline first")


def test_every_app_page_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    pages = list(at.sidebar.radio[0].options)
    assert len(pages) == 6
    for p in pages:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, f"{p}: {at.exception}"


def test_disclaimer_present_everywhere_it_matters():
    m = json.loads((ART / "metrics.json").read_text())
    assert "synthetic" in m["disclaimer"].lower() and "not" in m["disclaimer"].lower()
    assert "synthetic" in (ROOT / "README.md").read_text().lower()


def test_docs_numbers_match_metrics():
    m = json.loads((ART / "metrics.json").read_text())
    readme = (ROOT / "README.md").read_text()
    assert f"{m['models']['test']['lightgbm']['auroc']:.3f}" in readme
    assert f"{m['models']['test']['logistic_regression']['auroc']:.3f}" in readme
    for p in (ROOT / "docs").glob("*.md"):
        t = p.read_text()
        assert not re.search(r"\bnan\b|\bNaN\b|\{c\.|\{f\[", t), f"unrendered or NaN value in {p.name}"


def test_app_bundle_has_no_identifiers_or_hidden_frailty():
    import pandas as pd

    forbidden = {"patient_name", "ssn", "mrn", "address", "phone", "dob"}
    cols = set(pd.read_parquet(ART / "scored.parquet").columns)
    assert not (cols & forbidden)
    assert not any(c.startswith("frailty") or c == "latent_frailty" for c in cols)
