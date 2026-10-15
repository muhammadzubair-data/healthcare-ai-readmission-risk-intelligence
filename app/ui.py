"""Shared UI helpers: palette, CSS, KPI cards, banners, small-cell suppression."""
from __future__ import annotations

import pandas as pd
import streamlit as st

# Validated categorical order (blue, orange, aqua, violet) + neutral grey for baselines.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
GREY = "#8a8a85"
MODEL_COLORS = {
    "lace_style": GREY, "logistic_regression": BLUE, "random_forest": AQUA,
    "lightgbm": ORANGE, "lightgbm_calibrated": VIOLET, "oracle": "#444444",
}
MODEL_LABELS = {
    "lace_style": "LACE-style heuristic", "logistic_regression": "Logistic regression", "random_forest": "Random forest",
    "lightgbm": "LightGBM", "lightgbm_calibrated": "LightGBM + isotonic",
}
TIER_COLORS = {"Standard": "#9bbbe3", "Elevated": YELLOW, "High": RED}
SMALL_CELL = 11

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.kpi {background: var(--secondary-background-color); border-radius: 10px; padding: 14px 16px; border-left: 4px solid #2a78d6;}
.kpi .v {font-size: 1.7rem; font-weight: 650; line-height: 1.15;}
.kpi .l {font-size: 0.78rem; opacity: .75; text-transform: uppercase; letter-spacing: .04em;}
.kpi .s {font-size: 0.78rem; opacity: .7; margin-top: 2px;}
.disclaimer {background: #fff4e5; color: #5a3a00; border: 1px solid #f0c36d; border-radius: 8px; padding: 8px 12px; font-size: .85rem;}
.note {background: var(--secondary-background-color); border-radius: 8px; padding: 10px 14px; font-size: .88rem;}
h1, h2, h3 {letter-spacing: -0.01em;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer"><b>Synthetic demonstration.</b> Every patient record here is simulated. Risk scores are predictive '
        "associations, not clinical causes, and must not be used for real clinical, coverage or staffing decisions.</div>",
        unsafe_allow_html=True,
    )


def kpi(col, label: str, value: str, sub: str = "", color: str = BLUE) -> None:
    col.markdown(f'<div class="kpi" style="border-left-color:{color}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{d}f}%"


def suppress(df: pd.DataFrame, n_col: str, cols: list[str]) -> pd.DataFrame:
    """Small-cell suppression: hide values where the underlying count is below the threshold."""
    out = df.copy()
    m = out[n_col] < SMALL_CELL
    for c in cols:
        out[c] = out[c].astype(float)
        out.loc[m, c] = float("nan")
    return out


def style_fig(fig, height: int = 380, legend_bottom: bool = True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13), title_font_size=15)
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,.4)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig
