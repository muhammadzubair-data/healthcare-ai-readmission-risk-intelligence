"""Cached artifact loaders for the Streamlit app."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts"
TAB = ART / "tables"


def artifacts_ready() -> bool:
    return (ART / "metrics.json").exists() and (ART / "scored.parquet").exists()


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    return json.loads((ART / "metrics.json").read_text())


@st.cache_data(show_spinner=False)
def jsonf(name: str) -> dict:
    return json.loads((ART / name).read_text())


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / name)


@st.cache_data(show_spinner=False)
def scored() -> pd.DataFrame:
    return pd.read_parquet(ART / "scored.parquet")


@st.cache_data(show_spinner=False)
def eda() -> pd.DataFrame:
    return pd.read_parquet(ART / "eda_encounters.parquet")


@st.cache_data(show_spinner=False)
def shap_sample() -> pd.DataFrame:
    return pd.read_parquet(ART / "shap_sample.parquet")


@st.cache_data(show_spinner=False)
def train_reference() -> pd.DataFrame:
    return pd.read_parquet(ART / "train_reference_sample.parquet")


@st.cache_data(show_spinner=False)
def facilities() -> pd.DataFrame:
    return pd.read_csv(ART / "facilities.csv")


@st.cache_resource(show_spinner=False)
def booster():
    from readmit.models.train import load_lightgbm_for_app
    return load_lightgbm_for_app(ART / "models")


def read_doc(name: str) -> str:
    p = ROOT / "docs" / name
    return p.read_text() if p.exists() else f"*{name} not generated yet - run `python -m readmit.pipeline docs`.*"
