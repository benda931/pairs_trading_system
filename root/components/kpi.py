# -*- coding: utf-8 -*-
# root/components/kpi.py
"""
Institutional KPI Card Components
===================================
Three-tier KPI card hierarchy:

  Tier 1 — Hero metric (large numeral, full accent bar)
  Tier 2 — Standard metric (medium numeral, compact)
  Tier 3 — Inline datum (small label+value pair, table-like)

Limit gauge: optional coloured arc below the value showing
utilisation against a hard limit (e.g. 87% of max drawdown limit).

Usage
-----
from root.components.kpi import render_kpi_card, render_kpi_row

# Single card
render_kpi_card("Sharpe Ratio", "2.41", delta="+0.18", delta_positive=True, tier=1)

# Row of cards inside pre-created st.columns
cols = st.columns(4)
render_kpi_row([
    dict(label="NAV", value="$1.25M", unit="USD", tier=1),
    dict(label="Daily P&L", value="+$4,800", delta_positive=True),
    dict(label="Max DD", value="-8.3%", limit=20.0, current=8.3, semantic="caution"),
    dict(label="Active Pairs", value="8"),
], columns=cols)
"""

from __future__ import annotations

from typing import Optional
import streamlit as st

from root.design_system import DS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _limit_gauge_html(current: float, limit: float, semantic: str = "neutral") -> str:
    """Return a thin linear gauge bar for limit utilisation."""
    pct = min(current / limit, 1.0) if limit else 0.0
    if pct >= 0.90:
        bar_color = DS.CRITICAL
    elif pct >= 0.75:
        bar_color = DS.NEGATIVE
    elif pct >= 0.50:
        bar_color = DS.CAUTION
    else:
        bar_color = DS.POSITIVE

    filled_width = int(pct * 100)
    return f"""
    <div style="margin-top:{DS.SPACE_2};">
      <div style="display:flex;justify-content:space-between;
                  font-size:{DS.FONT_XS};color:{DS.TEXT_MUTED};margin-bottom:2px;">
        <span>0</span>
        <span style="color:{bar_color};">{pct*100:.0f}%</span>
        <span>{limit:.0f}</span>
      </div>
      <div style="height:3px;background:rgba(255,255,255,0.08);border-radius:2px;">
        <div style="height:3px;width:{filled_width}%;background:{bar_color};
                    border-radius:2px;transition:width 0.4s;"></div>
      </div>
    </div>
    """


def _delta_html(delta: str, positive: bool) -> str:
    color = DS.POSITIVE if positive else DS.NEGATIVE
    arrow = "▲" if positive else "▼"
    return (
        f'<span style="font-size:{DS.FONT_XS};color:{color};'
        f'font-weight:600;">{arrow} {delta}</span>'
    )


def _semantic_color(semantic: str) -> str:
    mapping = {
        "positive": DS.POSITIVE,
        "negative": DS.NEGATIVE,
        "caution":  DS.CAUTION,
        "critical": DS.CRITICAL,
        "research": DS.RESEARCH,
        "advisory": DS.ADVISORY,
        "neutral":  DS.TEXT_MUTED,
        "brand":    DS.BRAND,
    }
    return mapping.get(semantic, DS.TEXT_MUTED)


# ---------------------------------------------------------------------------
# render_kpi_card
# ---------------------------------------------------------------------------

def render_kpi_card(
    label: str,
    value: str,
    *,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    unit: Optional[str] = None,
    limit: Optional[float] = None,
    current: Optional[float] = None,
    tier: int = 2,
    semantic: str = "neutral",
    subtitle: Optional[str] = None,
) -> None:
    """
    Render a single KPI card.

    Parameters
    ----------
    label:          Metric label (e.g. "Sharpe Ratio").
    value:          Formatted value string (e.g. "2.41").
    delta:          Change string (e.g. "+0.18 vs last month").
    delta_positive: Whether the delta is favourable.
    unit:           Small unit label appended after value (e.g. "USD").
    limit:          Hard limit for gauge (e.g. 20.0 for 20% max DD limit).
    current:        Current raw value for gauge (e.g. 8.3 for 8.3% DD).
    tier:           1 = hero, 2 = standard, 3 = inline.
    semantic:       Accent colour key (positive/negative/caution/critical/neutral/brand).
    subtitle:       Optional small text below value.
    """
    accent = _semantic_color(semantic)

    if tier == 1:
        val_size   = DS.FONT_2XL
        label_size = DS.FONT_SM
        padding    = f"{DS.SPACE_4} {DS.SPACE_5}"
        border_top = f"3px solid {accent}"
    elif tier == 2:
        val_size   = DS.FONT_XL
        label_size = DS.FONT_XS
        padding    = f"{DS.SPACE_3} {DS.SPACE_4}"
        border_top = f"2px solid {accent}"
    else:  # tier 3
        val_size   = DS.FONT_LG
        label_size = DS.FONT_XS
        padding    = f"{DS.SPACE_2} {DS.SPACE_3}"
        border_top = f"1px solid rgba(255,255,255,0.06)"

    delta_block = _delta_html(delta, delta_positive) if delta else ""
    unit_span = (
        f'<span style="font-size:{DS.FONT_SM};color:{DS.TEXT_MUTED};'
        f'margin-left:4px;font-weight:400;">{unit}</span>'
        if unit else ""
    )
    subtitle_block = (
        f'<div style="font-size:{DS.FONT_XS};color:{DS.TEXT_MUTED};'
        f'margin-top:2px;">{subtitle}</div>'
        if subtitle else ""
    )
    gauge_block = (
        _limit_gauge_html(current, limit, semantic)
        if limit is not None and current is not None else ""
    )

    html = f"""
    <div style="
        background:{DS.BG_SURFACE};
        border-top:{border_top};
        border-radius:0 0 {DS.RADIUS_MD} {DS.RADIUS_MD};
        padding:{padding};
        height:100%;
        box-sizing:border-box;
    ">
      <div style="font-size:{label_size};color:{DS.TEXT_MUTED};
                  text-transform:uppercase;letter-spacing:0.06em;
                  font-weight:600;margin-bottom:{DS.SPACE_1};">{label}</div>
      <div style="font-size:{val_size};color:{DS.TEXT_PRIMARY};
                  font-weight:700;font-variant-numeric:tabular-nums;
                  line-height:1.1;">
        {value}{unit_span}
      </div>
      {subtitle_block}
      {delta_block}
      {gauge_block}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# render_kpi_row
# ---------------------------------------------------------------------------

def render_kpi_row(
    cards: list[dict],
    columns: Optional[list] = None,
) -> None:
    """
    Render a row of KPI cards.

    Parameters
    ----------
    cards:   List of dicts, each containing kwargs for render_kpi_card.
    columns: Pre-created list of st.columns. If None, creates equal columns.
    """
    if columns is None:
        columns = st.columns(len(cards))

    for col, card_kwargs in zip(columns, cards):
        with col:
            render_kpi_card(**card_kwargs)
