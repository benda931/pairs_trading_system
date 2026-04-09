# -*- coding: utf-8 -*-
# root/components/signal_card.py
"""
Signal Card Component
======================
Renders a structured signal card for a pair signal or alpha recommendation.

Two display modes:
  compact  — single-line strip suitable for high-density lists
  expanded — full card with provenance, confidence, expiry, action buttons

Usage
-----
from root.components.signal_card import render_signal_card, render_signal_grid

render_signal_card({
    "pair":       "XLE/XOP",
    "direction":  "LONG",
    "z_score":    2.14,
    "confidence": 0.78,
    "regime":     "MEAN_REVERTING",
    "entry_date": "2026-04-08",
    "expiry":     "2026-04-22",
    "source":     "z_score_engine",
    "grade":      "B+",
}, compact=False)
"""

from __future__ import annotations

from typing import Any, Optional
import streamlit as st

from root.design_system import DS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DIRECTION_COLORS = {
    "LONG":    DS.POSITIVE,
    "SHORT":   DS.NEGATIVE,
    "NEUTRAL": DS.TEXT_MUTED,
    "EXIT":    DS.CAUTION,
}

_REGIME_COLORS = {
    "MEAN_REVERTING": DS.POSITIVE,
    "TRENDING":       DS.CAUTION,
    "HIGH_VOL":       "#FF6B35",
    "CRISIS":         DS.NEGATIVE,
    "BROKEN":         DS.ADVISORY,
    "UNKNOWN":        DS.TEXT_MUTED,
}


def _safe(d: Any, key: str, default: str = "—") -> str:
    if isinstance(d, dict):
        v = d.get(key, default)
    else:
        v = getattr(d, key, default)
    return str(v) if v is not None else default


def _confidence_bar(confidence: float) -> str:
    pct = int(min(max(confidence, 0.0), 1.0) * 100)
    color = DS.POSITIVE if pct >= 70 else DS.CAUTION if pct >= 40 else DS.NEGATIVE
    return f"""
    <div style="display:flex;align-items:center;gap:{DS.SPACE_2};margin-top:{DS.SPACE_1};">
      <div style="flex:1;height:3px;background:rgba(255,255,255,0.07);border-radius:2px;">
        <div style="height:3px;width:{pct}%;background:{color};border-radius:2px;"></div>
      </div>
      <span style="font-size:{DS.FONT_XS};color:{color};font-weight:600;
                   min-width:30px;">{pct}%</span>
    </div>
    """


def _grade_badge(grade: str) -> str:
    color = DS.GRADE_COLORS.get(grade, DS.TEXT_MUTED)
    return (
        f'<span style="font-size:{DS.FONT_XS};font-weight:700;color:{color};'
        f'background:{color}1A;padding:1px 6px;border-radius:{DS.RADIUS_SM};'
        f'border:1px solid {color}44;">{grade}</span>'
    )


def _direction_badge(direction: str) -> str:
    color = _DIRECTION_COLORS.get(direction.upper(), DS.TEXT_MUTED)
    return (
        f'<span style="font-size:{DS.FONT_XS};font-weight:700;color:{color};'
        f'background:{color}1A;padding:2px 8px;border-radius:{DS.RADIUS_SM};'
        f'border:1px solid {color}44;letter-spacing:0.05em;">{direction.upper()}</span>'
    )


def _regime_chip(regime: str) -> str:
    color = _REGIME_COLORS.get(regime.upper(), DS.TEXT_MUTED)
    label = regime.replace("_", " ").title()
    return (
        f'<span style="font-size:10px;color:{color};background:{color}1A;'
        f'padding:1px 5px;border-radius:2px;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# render_signal_card
# ---------------------------------------------------------------------------

def render_signal_card(
    signal: Any,
    *,
    compact: bool = False,
    show_provenance: bool = True,
    show_actions: bool = False,
    key_prefix: str = "",
) -> None:
    """
    Render a single signal card.

    Parameters
    ----------
    signal:          Dict or dataclass with signal fields.
    compact:         If True, renders as a single-row strip.
    show_provenance: Show source, entry date, expiry.
    show_actions:    Show Approve/Reject/Defer buttons (no-op — wires to parent).
    key_prefix:      Unique prefix for Streamlit widget keys.
    """
    pair       = _safe(signal, "pair")
    direction  = _safe(signal, "direction", "NEUTRAL")
    z_score    = _safe(signal, "z_score")
    confidence = float(_safe(signal, "confidence", "0") or 0)
    regime     = _safe(signal, "regime", "UNKNOWN")
    entry_date = _safe(signal, "entry_date")
    expiry     = _safe(signal, "expiry")
    source     = _safe(signal, "source")
    grade      = _safe(signal, "grade", "C")
    notes      = _safe(signal, "notes", "")

    direction_color = _DIRECTION_COLORS.get(direction.upper(), DS.TEXT_MUTED)

    # ── Compact mode ──────────────────────────────────────────────
    if compact:
        html = f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:{DS.SPACE_2} {DS.SPACE_3};
                    background:{DS.BG_SURFACE};
                    border-left:3px solid {direction_color};
                    border-radius:0 {DS.RADIUS_SM} {DS.RADIUS_SM} 0;
                    margin-bottom:4px;">
          <span style="font-size:{DS.FONT_MD};font-weight:700;
                       color:{DS.TEXT_PRIMARY};min-width:90px;">{pair}</span>
          <span>{_direction_badge(direction)}</span>
          <span style="font-size:{DS.FONT_SM};color:{DS.TEXT_SECONDARY};
                       font-variant-numeric:tabular-nums;">z = {z_score}</span>
          <span>{_regime_chip(regime)}</span>
          <span>{_grade_badge(grade)}</span>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        return

    # ── Expanded mode ─────────────────────────────────────────────
    provenance_block = ""
    if show_provenance:
        provenance_block = f"""
        <div style="margin-top:{DS.SPACE_3};padding-top:{DS.SPACE_2};
                    border-top:1px solid rgba(255,255,255,0.06);
                    display:flex;gap:{DS.SPACE_5};flex-wrap:wrap;">
          <div>
            <div style="font-size:10px;color:{DS.TEXT_MUTED};text-transform:uppercase;
                        letter-spacing:0.05em;">Source</div>
            <div style="font-size:{DS.FONT_XS};color:{DS.TEXT_SECONDARY};
                        font-family:monospace;">{source}</div>
          </div>
          <div>
            <div style="font-size:10px;color:{DS.TEXT_MUTED};text-transform:uppercase;
                        letter-spacing:0.05em;">Entry</div>
            <div style="font-size:{DS.FONT_XS};color:{DS.TEXT_SECONDARY};">{entry_date}</div>
          </div>
          <div>
            <div style="font-size:10px;color:{DS.TEXT_MUTED};text-transform:uppercase;
                        letter-spacing:0.05em;">Expiry</div>
            <div style="font-size:{DS.FONT_XS};color:{DS.TEXT_SECONDARY};">{expiry}</div>
          </div>
        </div>
        """

    notes_block = ""
    if notes and notes != "—":
        notes_block = f"""
        <div style="margin-top:{DS.SPACE_2};font-size:{DS.FONT_XS};
                    color:{DS.TEXT_MUTED};font-style:italic;">{notes}</div>
        """

    html = f"""
    <div style="
        background:{DS.BG_SURFACE};
        border:1px solid rgba(255,255,255,0.06);
        border-left:3px solid {direction_color};
        border-radius:{DS.RADIUS_MD};
        padding:{DS.SPACE_4};
        margin-bottom:{DS.SPACE_3};
    ">
      <div style="display:flex;align-items:center;justify-content:space-between;
                  margin-bottom:{DS.SPACE_3};">
        <div style="display:flex;align-items:center;gap:{DS.SPACE_3};">
          <span style="font-size:{DS.FONT_LG};font-weight:800;
                       color:{DS.TEXT_PRIMARY};">{pair}</span>
          {_direction_badge(direction)}
          {_regime_chip(regime)}
        </div>
        <div style="display:flex;align-items:center;gap:{DS.SPACE_2};">
          {_grade_badge(grade)}
        </div>
      </div>

      <div style="display:flex;gap:{DS.SPACE_6};flex-wrap:wrap;
                  margin-bottom:{DS.SPACE_2};">
        <div>
          <div style="font-size:10px;color:{DS.TEXT_MUTED};text-transform:uppercase;
                      letter-spacing:0.05em;">Z-Score</div>
          <div style="font-size:{DS.FONT_MD};font-weight:700;
                      color:{DS.TEXT_PRIMARY};font-variant-numeric:tabular-nums;">{z_score}</div>
        </div>
        <div style="flex:1;min-width:120px;">
          <div style="font-size:10px;color:{DS.TEXT_MUTED};text-transform:uppercase;
                      letter-spacing:0.05em;">Confidence</div>
          {_confidence_bar(confidence)}
        </div>
      </div>

      {notes_block}
      {provenance_block}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    if show_actions:
        key_base = f"{key_prefix}_{pair}_{direction}"
        c1, c2, c3, _ = st.columns([1, 1, 1, 5])
        with c1:
            if st.button("Approve", key=f"{key_base}_approve", type="primary",
                         use_container_width=True):
                st.session_state[f"{key_base}_action"] = "approved"
        with c2:
            if st.button("Reject", key=f"{key_base}_reject",
                         use_container_width=True):
                st.session_state[f"{key_base}_action"] = "rejected"
        with c3:
            if st.button("Defer", key=f"{key_base}_defer",
                         use_container_width=True):
                st.session_state[f"{key_base}_action"] = "deferred"


# ---------------------------------------------------------------------------
# render_signal_grid
# ---------------------------------------------------------------------------

def render_signal_grid(
    signals: list[Any],
    *,
    columns: int = 2,
    compact: bool = False,
    show_provenance: bool = True,
) -> None:
    """
    Render a grid of signal cards.

    Parameters
    ----------
    signals:  List of signal dicts/dataclasses.
    columns:  Number of columns (1, 2, or 3).
    compact:  Use compact strip mode.
    """
    if not signals:
        st.info("No signals available.")
        return

    if compact or columns == 1:
        for sig in signals:
            render_signal_card(sig, compact=compact,
                               show_provenance=show_provenance)
        return

    cols = st.columns(columns)
    for i, sig in enumerate(signals):
        with cols[i % columns]:
            render_signal_card(sig, compact=False,
                               show_provenance=show_provenance)
