# -*- coding: utf-8 -*-
# root/charts/theme.py
"""
Universal Plotly Chart Theme
=============================
Single chart configuration applied to EVERY figure in the platform.
Eliminates the plotly_white/dark-shell mismatch and produces a
consistent institutional dark-theme chart language.

Usage
-----
from root.charts.theme import apply_theme, CHART_COLORS

fig = go.Figure(...)
apply_theme(fig, title="SPREAD Z-SCORE — XLE/XOP")
st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

from typing import Any
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Colour Palettes
# ---------------------------------------------------------------------------

CHART_COLORS: dict[str, str] = {
    "primary":   "#1976D2",   # main series / interactive
    "secondary": "#546E7A",   # comparison / reference
    "positive":  "#1BB152",   # returns, gains, profitable
    "negative":  "#D93025",   # drawdown, losses, blocked
    "caution":   "#F5A623",   # signals, alerts, warning
    "neutral":   "#8FA3B1",   # reference lines, n/a
    "research":  "#00897B",   # research / OOS region
    "advisory":  "#7B1FA2",   # agent outputs
    "spy":       "#FF9800",   # benchmark traces
    "critical":  "#FF3B3B",   # P0 level
}

REGIME_COLORS: dict[str, str] = {
    "MEAN_REVERTING": "#1BB152",
    "TRENDING":       "#F5A623",
    "HIGH_VOL":       "#FF6B35",
    "CRISIS":         "#D93025",
    "BROKEN":         "#7B1FA2",
    "UNKNOWN":        "#546E7A",
}

GRADE_COLORS: dict[str, str] = {
    "A+": "#1BB152", "A": "#1BB152",
    "B+": "#42A5F5", "B": "#42A5F5",
    "C":  "#F5A623",
    "D":  "#D93025",
    "F":  "#FF3B3B",
}

# Ordered colour sequence for multi-series charts
COLOUR_SEQUENCE = [
    "#1976D2", "#1BB152", "#F5A623", "#00897B",
    "#7B1FA2", "#D93025", "#FF9800", "#8FA3B1",
    "#E91E63", "#00BCD4", "#9E9E9E", "#FF5722",
]

# ---------------------------------------------------------------------------
# Standard line widths
# ---------------------------------------------------------------------------

LINE_PRIMARY   = dict(color=CHART_COLORS["primary"],   width=2)
LINE_SECONDARY = dict(color=CHART_COLORS["secondary"], width=1.5, dash="dot")
LINE_POSITIVE  = dict(color=CHART_COLORS["positive"],  width=1.5)
LINE_NEGATIVE  = dict(color=CHART_COLORS["negative"],  width=1.5)
LINE_REFERENCE = dict(color=CHART_COLORS["neutral"],   width=1,   dash="dash")
LINE_SPY       = dict(color=CHART_COLORS["spy"],       width=1.5, dash="dash")

# ---------------------------------------------------------------------------
# Hover templates
# ---------------------------------------------------------------------------

HOVER_VALUE   = "<b>%{y:.3f}</b><br>%{x|%b %d, %Y}<extra></extra>"
HOVER_PNL     = "<b>%{y:$,.0f}</b><br>%{x|%b %d, %Y}<extra></extra>"
HOVER_PCT     = "<b>%{y:+.2f}%</b><br>%{x|%b %d, %Y}<extra></extra>"
HOVER_SHARPE  = "<b>%{y:.2f}</b><br>%{x}<extra></extra>"

# ---------------------------------------------------------------------------
# Base layout dict (applies to every figure)
# ---------------------------------------------------------------------------

_BASE_AXIS = dict(
    gridcolor="rgba(255,255,255,0.04)",
    linecolor="rgba(255,255,255,0.10)",
    tickfont=dict(color="#546E7A", size=11, family="Inter, sans-serif"),
    title_font=dict(color="#8FA3B1", size=12, family="Inter, sans-serif"),
    zeroline=True,
    zerolinecolor="rgba(255,255,255,0.12)",
    zerolinewidth=1,
    showgrid=True,
)

CHART_THEME: dict[str, Any] = dict(
    plot_bgcolor="transparent",
    paper_bgcolor="transparent",
    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        color="#8FA3B1",
        size=12,
    ),
    xaxis=dict(**_BASE_AXIS),
    yaxis=dict(**_BASE_AXIS),
    legend=dict(
        bgcolor="rgba(13,27,42,0.85)",
        bordercolor="rgba(255,255,255,0.10)",
        borderwidth=1,
        font=dict(color="#8FA3B1", size=11),
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
    ),
    margin=dict(l=52, r=16, t=36, b=40),
    hoverlabel=dict(
        bgcolor="#1A2744",
        bordercolor="rgba(255,255,255,0.15)",
        font=dict(color="#E8EDF2", size=12),
    ),
    hovermode="x unified",
    colorway=COLOUR_SEQUENCE,
    title_font=dict(
        family="Inter, sans-serif",
        size=13,
        color="#8FA3B1",
    ),
    title_x=0,
    title_pad=dict(l=0, t=4),
)

# ---------------------------------------------------------------------------
# apply_theme()
# ---------------------------------------------------------------------------

def apply_theme(
    fig: go.Figure,
    title: str = "",
    height: int = 380,
    compact: bool = False,
    show_legend: bool = True,
) -> go.Figure:
    """
    Apply the universal chart theme to a Plotly figure.

    Parameters
    ----------
    fig:          Plotly Figure to style.
    title:        Chart title (format: "METRIC — CONTEXT").
    height:       Chart height in pixels.
    compact:      Use tighter margins (for multi-chart grids).
    show_legend:  Toggle legend visibility.

    Returns the same figure (mutated in place) for chaining.
    """
    theme = dict(CHART_THEME)
    theme["height"] = height
    theme["showlegend"] = show_legend

    if title:
        theme["title"] = dict(
            text=title,
            font=dict(size=13, color="#8FA3B1", family="Inter, sans-serif"),
            x=0,
            pad=dict(l=0, t=4),
        )

    if compact:
        theme["margin"] = dict(l=36, r=8, t=28, b=32)
        theme["font"] = dict(
            family="Inter, -apple-system, sans-serif",
            color="#8FA3B1",
            size=11,
        )

    fig.update_layout(**theme)
    return fig
