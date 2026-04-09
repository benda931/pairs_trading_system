# -*- coding: utf-8 -*-
# root/components/state_banner.py
"""
System State Banner Component
==============================
Persistent banner reflecting the operational mode of the platform.

Modes:
  NOMINAL     — No banner rendered (invisible when healthy).
  DEGRADED    — Yellow banner: specific component(s) unavailable,
                platform operational with reduced capability.
  RISK_BLIND  — Orange banner: risk analytics unavailable,
                NO NEW POSITIONS should be opened.
  HALTED      — Red banner: system halted, no automated actions.
  MAINTENANCE — Blue banner: manual maintenance in progress.

Usage
-----
from root.components.state_banner import render_state_banner

render_state_banner("DEGRADED",
                    details="Correlation service offline — using last-known matrix")
render_state_banner("HALTED",
                    details="Manual halt by PM at 14:32 UTC · Reason: VaR breach")
"""

from __future__ import annotations

from typing import Optional
import streamlit as st

from root.design_system import DS


# ---------------------------------------------------------------------------
# Mode configuration
# ---------------------------------------------------------------------------

_MODE_CONFIG = {
    "NOMINAL": None,  # Silent — no banner
    "DEGRADED": {
        "bg":    "#1F1A00",
        "border": DS.CAUTION,
        "icon":   "⚠️",
        "label":  "SYSTEM DEGRADED",
        "color":  DS.CAUTION,
        "note":   "Some services unavailable. Verify outputs before acting.",
    },
    "RISK_BLIND": {
        "bg":    "#2A1200",
        "border": "#FF6B35",
        "icon":   "🚨",
        "label":  "RISK BLIND",
        "color":  "#FF6B35",
        "note":   "Risk analytics offline. DO NOT open new positions.",
    },
    "HALTED": {
        "bg":    "#2A0000",
        "border": DS.CRITICAL,
        "icon":   "⛔",
        "label":  "SYSTEM HALTED",
        "color":  DS.CRITICAL,
        "note":   "All automated actions suspended. Manual review required.",
    },
    "MAINTENANCE": {
        "bg":    "#001830",
        "border": DS.BRAND,
        "icon":   "🔧",
        "label":  "MAINTENANCE",
        "color":  DS.BRAND,
        "note":   "Scheduled maintenance in progress.",
    },
}


# ---------------------------------------------------------------------------
# render_state_banner
# ---------------------------------------------------------------------------

def render_state_banner(
    mode: str,
    *,
    details: Optional[str] = None,
    since: Optional[str] = None,
    show_dismiss: bool = False,
    banner_key: str = "state_banner",
) -> None:
    """
    Render the system state banner.

    Parameters
    ----------
    mode:         One of NOMINAL / DEGRADED / RISK_BLIND / HALTED / MAINTENANCE.
    details:      Specific detail text about what is impaired.
    since:        Timestamp string for when the state began.
    show_dismiss: Show a dismiss button (for DEGRADED/MAINTENANCE only).
    banner_key:   Streamlit session key for dismiss state.
    """
    mode_upper = mode.upper()
    cfg = _MODE_CONFIG.get(mode_upper)

    if cfg is None:
        return

    if show_dismiss and st.session_state.get(f"{banner_key}_dismissed", False):
        return

    since_block = (
        f'<span style="font-size:10px;color:{DS.TEXT_MUTED};margin-left:{DS.SPACE_3};">'
        f'since {since}</span>'
        if since else ""
    )

    details_block = (
        f'<div style="font-size:{DS.FONT_SM};color:{DS.TEXT_SECONDARY};'
        f'margin-top:{DS.SPACE_1};padding-left:26px;">{details}</div>'
        if details else ""
    )

    html = f"""
    <div style="
        background:{cfg['bg']};
        border:1px solid {cfg['border']};
        border-left:4px solid {cfg['border']};
        border-radius:{DS.RADIUS_SM};
        padding:{DS.SPACE_3} {DS.SPACE_4};
        margin-bottom:{DS.SPACE_3};
    ">
      <div style="display:flex;align-items:center;gap:{DS.SPACE_2};">
        <span style="font-size:{DS.FONT_MD};">{cfg['icon']}</span>
        <span style="font-size:{DS.FONT_SM};font-weight:800;
                     color:{cfg['color']};letter-spacing:0.08em;">{cfg['label']}</span>
        {since_block}
        <span style="margin-left:auto;font-size:{DS.FONT_XS};
                     color:{DS.TEXT_MUTED};font-style:italic;">{cfg['note']}</span>
      </div>
      {details_block}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    if show_dismiss and mode_upper in ("DEGRADED", "MAINTENANCE"):
        if st.button("Acknowledge", key=f"{banner_key}_btn",
                     help="Hide banner for this session"):
            st.session_state[f"{banner_key}_dismissed"] = True
            st.rerun()
