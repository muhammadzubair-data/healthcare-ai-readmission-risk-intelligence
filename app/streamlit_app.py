"""Hospital Readmission Risk Intelligence Platform - Streamlit entry point (synthetic demo)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="Readmission Risk Intelligence (synthetic demo)", page_icon="🏥", layout="wide")

import data
import ui
from views import executive, model_eval, patient_risk, patterns, quality_governance, scenarios

ui.inject_css()
PAGES = {
    "Executive overview": executive.render,
    "Patient risk explorer": patient_risk.render,
    "Readmission patterns": patterns.render,
    "Model evaluation": model_eval.render,
    "Care-management scenarios": scenarios.render,
    "Data quality & governance": quality_governance.render,
}

with st.sidebar:
    st.markdown("## 🏥 Readmission Risk Intelligence")
    st.caption("Project 014 - synthetic demonstration")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()

if not data.artifacts_ready():
    st.error("Model artifacts not found. Run `python -m readmit.pipeline all` (or `make all`) first.")
    st.stop()

if page == "Executive overview":
    ui.disclaimer()
PAGES[page]()
