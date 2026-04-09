# -*- coding: utf-8 -*-
# root/charts/helpers.py
"""
Institutional Chart Factory Functions
======================================
Pre-built, themed Plotly figures for every recurring chart type
across the platform. All functions return a go.Figure with the
universal CHART_THEME applied.

Usage
-----
from root.charts.helpers import equity_curve_fig, drawdown_fig

fig = equity_curve_fig(equity_series, "PORTFOLIO EQUITY — PAPER")
st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from root.charts.theme import (
    apply_theme,
    CHART_COLORS,
    REGIME_COLORS,
    GRADE_COLORS,
    COLOUR_SEQUENCE,
    LINE_PRIMARY,
    LINE_NEGATIVE,
    LINE_SPY,
    LINE_REFERENCE,
    HOVER_PNL,
    HOVER_VALUE,
    HOVER_PCT,
)


# ---------------------------------------------------------------------------
# 1. Equity Curve
# ---------------------------------------------------------------------------

def equity_curve_fig(
    equity: pd.Series,
    title: str = "PORTFOLIO EQUITY",
    benchmark: Optional[pd.Series] = None,
    is_boundary: Optional[pd.Timestamp] = None,
    height: int = 380,
) -> go.Figure:
    """
    Equity curve with optional benchmark overlay and IS/OOS split.

    Parameters
    ----------
    equity:       Daily portfolio value series (DatetimeIndex).
    title:        Chart title.
    benchmark:    Optional benchmark series (normalised to same base as equity).
    is_boundary:  If provided, shades the IS region differently from OOS.
    height:       Chart height.
    """
    fig = go.Figure()

    # IS / OOS background shading
    if is_boundary is not None and not equity.empty:
        x_start = equity.index[0]
        x_end = equity.index[-1]
        fig.add_vrect(
            x0=x_start, x1=is_boundary,
            fillcolor="rgba(25,118,210,0.06)",
            line_width=0,
            annotation_text="IS", annotation_position="top left",
            annotation_font=dict(color="#546E7A", size=10),
        )
        fig.add_vrect(
            x0=is_boundary, x1=x_end,
            fillcolor="rgba(0,137,123,0.06)",
            line_width=0,
            annotation_text="OOS", annotation_position="top left",
            annotation_font=dict(color="#546E7A", size=10),
        )

    # Main equity trace
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode="lines",
        name="Portfolio",
        line=dict(color=CHART_COLORS["primary"], width=2),
        fill="tozeroy",
        fillcolor="rgba(25,118,210,0.08)",
        hovertemplate=HOVER_PNL,
    ))

    # Benchmark
    if benchmark is not None and not benchmark.empty:
        aligned = benchmark.reindex(equity.index, method="ffill")
        # Normalise to same starting point
        if aligned.iloc[0] != 0:
            aligned = aligned / aligned.iloc[0] * equity.iloc[0]
        fig.add_trace(go.Scatter(
            x=aligned.index,
            y=aligned.values,
            mode="lines",
            name="Benchmark",
            line=LINE_SPY,
            hovertemplate=HOVER_PNL,
        ))

    fig.update_yaxes(tickformat="$,.0f", title_text="NAV")
    fig.update_xaxes(title_text="Date")
    return apply_theme(fig, title=title, height=height)


# ---------------------------------------------------------------------------
# 2. Drawdown
# ---------------------------------------------------------------------------

def drawdown_fig(
    drawdown: pd.Series,
    title: str = "DRAWDOWN",
    height: int = 200,
) -> go.Figure:
    """
    Drawdown chart (values should be in percentage, negative).
    """
    dd = drawdown.copy()
    # Ensure values are expressed as percentages
    if dd.abs().max() <= 1.5:
        dd = dd * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        mode="lines",
        name="Drawdown",
        line=dict(color=CHART_COLORS["negative"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(217,48,37,0.15)",
        hovertemplate=HOVER_PCT,
    ))

    fig.update_yaxes(tickformat=".1f", ticksuffix="%", title_text="Drawdown")
    fig.update_xaxes(title_text="Date")
    return apply_theme(fig, title=title, height=height, compact=True)


# ---------------------------------------------------------------------------
# 3. Z-Score time series
# ---------------------------------------------------------------------------

def z_score_fig(
    z: pd.Series,
    pair_id: str,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    trades: Optional[list[dict]] = None,
    height: int = 300,
) -> go.Figure:
    """
    Spread z-score with entry/exit bands and optional trade markers.

    trades: list of dicts with keys 'date', 'action' ('entry'/'exit'), 'z', 'direction'
    """
    fig = go.Figure()

    # Threshold bands
    for z_level, color, label in [
        (entry_z,  "rgba(27,177,82,0.08)",  f"Entry ±{entry_z}"),
        (-entry_z, "rgba(27,177,82,0.08)",  None),
        (stop_z,   "rgba(217,48,37,0.08)",  f"Stop ±{stop_z}"),
        (-stop_z,  "rgba(217,48,37,0.08)",  None),
    ]:
        fig.add_hline(
            y=z_level,
            line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dash"),
            annotation_text=label or "",
            annotation_font=dict(color="#546E7A", size=10),
        )

    # Exit zone fill
    fig.add_hrect(y0=-exit_z, y1=exit_z,
                  fillcolor="rgba(245,166,35,0.06)", line_width=0)

    # Z-score line
    fig.add_trace(go.Scatter(
        x=z.index,
        y=z.values,
        mode="lines",
        name="Z-Score",
        line=dict(color=CHART_COLORS["primary"], width=1.8),
        hovertemplate=HOVER_VALUE,
    ))

    # Trade markers
    if trades:
        entries = [t for t in trades if t.get("action") == "entry"]
        exits   = [t for t in trades if t.get("action") == "exit"]
        if entries:
            fig.add_trace(go.Scatter(
                x=[t["date"] for t in entries],
                y=[t["z"]    for t in entries],
                mode="markers",
                name="Entry",
                marker=dict(color=CHART_COLORS["positive"], size=8, symbol="triangle-up"),
            ))
        if exits:
            fig.add_trace(go.Scatter(
                x=[t["date"] for t in exits],
                y=[t["z"]    for t in exits],
                mode="markers",
                name="Exit",
                marker=dict(color=CHART_COLORS["caution"], size=8, symbol="triangle-down"),
            ))

    fig.update_yaxes(title_text="Z-Score", zeroline=True)
    fig.update_xaxes(title_text="Date")
    return apply_theme(fig, title=f"SPREAD Z-SCORE — {pair_id}", height=height)


# ---------------------------------------------------------------------------
# 4. Regime bar / background band chart
# ---------------------------------------------------------------------------

def regime_bar_fig(
    regime_series: pd.Series,
    title: str = "REGIME",
    height: int = 100,
) -> go.Figure:
    """
    Colour-coded regime timeline bar.
    regime_series: DatetimeIndex → regime string (MEAN_REVERTING, etc.)
    """
    fig = go.Figure()

    if regime_series.empty:
        return apply_theme(fig, title=title, height=height)

    # Group consecutive same-regime segments and fill vrect
    prev_regime = None
    start_date = regime_series.index[0]

    for date, regime in regime_series.items():
        if regime != prev_regime:
            if prev_regime is not None:
                color = REGIME_COLORS.get(str(prev_regime), "#546E7A")
                fig.add_vrect(
                    x0=start_date, x1=date,
                    fillcolor=color + "33",  # 20% opacity
                    line_width=0,
                )
            start_date = date
            prev_regime = regime

    # Last segment
    if prev_regime is not None:
        color = REGIME_COLORS.get(str(prev_regime), "#546E7A")
        fig.add_vrect(
            x0=start_date, x1=regime_series.index[-1],
            fillcolor=color + "33",
            line_width=0,
        )

    # Legend entries (invisible scatter traces for legend)
    for regime_name, color in REGIME_COLORS.items():
        if regime_name in regime_series.values:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color=color, size=10, symbol="square"),
                name=regime_name.replace("_", " ").title(),
                showlegend=True,
            ))

    fig.update_yaxes(visible=False, showgrid=False)
    fig.update_xaxes(title_text="Date", showgrid=False)
    return apply_theme(fig, title=title, height=height, compact=True)


# ---------------------------------------------------------------------------
# 5. Sensitivity Heatmap (2-D param sensitivity)
# ---------------------------------------------------------------------------

def sensitivity_heatmap_fig(
    x_centers: "np.ndarray",
    y_centers: "np.ndarray",
    sharpe_grid: "np.ndarray",
    x_label: str,
    y_label: str,
    title: str = "PARAMETER SENSITIVITY",
    height: int = 380,
) -> go.Figure:
    """2-D Sharpe heatmap for parameter pair sensitivity analysis."""
    fig = go.Figure(data=go.Heatmap(
        z=sharpe_grid.T,
        x=[f"{v:.3g}" for v in x_centers],
        y=[f"{v:.3g}" for v in y_centers],
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(
            title=dict(text="Sharpe", font=dict(color="#8FA3B1", size=11)),
            tickfont=dict(color="#8FA3B1", size=10),
            bgcolor="rgba(13,27,42,0.8)",
            bordercolor="rgba(255,255,255,0.1)",
        ),
        hovertemplate=(
            f"{x_label}: %{{x}}<br>"
            f"{y_label}: %{{y}}<br>"
            "Sharpe: <b>%{z:.3f}</b>"
            "<extra></extra>"
        ),
    ))
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return apply_theme(fig, title=title, height=height, show_legend=False)


# ---------------------------------------------------------------------------
# 6. Pair Sharpe Bar Chart
# ---------------------------------------------------------------------------

def pair_sharpe_bar_fig(
    pairs: list[str],
    sharpes: list[float],
    title: str = "PAIR SHARPE RATIOS",
    height: int = 320,
    threshold_good: float = 0.7,
    threshold_ok: float = 0.3,
) -> go.Figure:
    """
    Sorted bar chart of pair Sharpe ratios with semantic colouring.
    """
    # Sort descending
    data = sorted(zip(sharpes, pairs), reverse=True)
    sorted_sharpes = [d[0] for d in data]
    sorted_pairs   = [d[1] for d in data]

    colors = [
        CHART_COLORS["positive"] if s >= threshold_good
        else CHART_COLORS["caution"] if s >= threshold_ok
        else CHART_COLORS["negative"]
        for s in sorted_sharpes
    ]

    fig = go.Figure(go.Bar(
        x=sorted_pairs,
        y=sorted_sharpes,
        marker_color=colors,
        text=[f"{s:.2f}" for s in sorted_sharpes],
        textposition="outside",
        textfont=dict(color="#8FA3B1", size=10),
        hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.3f}<extra></extra>",
    ))

    # Threshold reference lines
    fig.add_hline(
        y=threshold_good,
        line=dict(color=CHART_COLORS["positive"], width=1, dash="dash"),
        annotation_text=f"Good ({threshold_good})",
        annotation_font=dict(color="#546E7A", size=10),
    )
    fig.add_hline(
        y=0,
        line=dict(color="rgba(255,255,255,0.12)", width=1),
    )

    fig.update_xaxes(title_text="Pair", tickangle=-45)
    fig.update_yaxes(title_text="Sharpe Ratio")
    return apply_theme(fig, title=title, height=height, show_legend=False)


# ---------------------------------------------------------------------------
# 7. Correlation Heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap_fig(
    corr_matrix: pd.DataFrame,
    title: str = "CORRELATION MATRIX",
    height: int = 420,
) -> go.Figure:
    """Diverging correlation heatmap (RdBu_r, zero-centred)."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        colorscale="RdBu_r",
        zmin=-1, zmax=1, zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=10, color="#E8EDF2"),
        colorbar=dict(
            title=dict(text="ρ", font=dict(color="#8FA3B1", size=12)),
            tickfont=dict(color="#8FA3B1", size=10),
            bgcolor="rgba(13,27,42,0.8)",
            bordercolor="rgba(255,255,255,0.1)",
        ),
        hovertemplate="%{y} × %{x}<br>ρ = <b>%{z:.3f}</b><extra></extra>",
    ))
    fig.update_xaxes(tickangle=-45)
    return apply_theme(fig, title=title, height=height, show_legend=False)


# ---------------------------------------------------------------------------
# 8. Returns Distribution Histogram
# ---------------------------------------------------------------------------

def returns_histogram_fig(
    returns: pd.Series,
    title: str = "RETURN DISTRIBUTION",
    null_distribution: Optional[list[float]] = None,
    observed_sharpe: Optional[float] = None,
    height: int = 300,
) -> go.Figure:
    """
    Return distribution histogram with optional permutation null overlay.
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns.values,
        nbinsx=40,
        name="Strategy Returns",
        marker_color=CHART_COLORS["primary"],
        opacity=0.75,
        hovertemplate="Return: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))

    # Null distribution overlay (permutation test)
    if null_distribution:
        fig.add_trace(go.Histogram(
            x=null_distribution,
            nbinsx=40,
            name="Null (Permutation)",
            marker_color=CHART_COLORS["secondary"],
            opacity=0.45,
            hovertemplate="Null: %{x:.3f}<br>Count: %{y}<extra></extra>",
        ))

    # Observed Sharpe vertical line
    if observed_sharpe is not None:
        fig.add_vline(
            x=observed_sharpe,
            line=dict(color=CHART_COLORS["positive"], width=2),
            annotation_text=f"Observed SR={observed_sharpe:.3f}",
            annotation_font=dict(color="#1BB152", size=11),
        )

    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Sharpe / Return")
    fig.update_yaxes(title_text="Count")
    return apply_theme(fig, title=title, height=height)
