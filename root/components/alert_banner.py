# -*- coding: utf-8 -*-
# root/components/alert_banner.py
"""
Alert Banner Components
========================
Five-tier alert system:

  P0 — Critical / System halt        (red background, persistent)
  P1 — High / Immediate action req.  (orange border, dismissible)
  P2 — Medium / Review required      (yellow border, auto-expire)
  P3 — Low / Informational           (blue border, auto-expire)
  P4 — Debug / Trace                 (muted, dev mode only)

Usage
-----
from root.components.alert_banner import render_alert_banner, render_inline_alert

alerts = [
    {"level": "P0", "title": "RISK LIMIT BREACH",
     "message": "Portfolio DD exceeded 15% hard limit"},
    {"level": "P2", "title": "Regime Change",
     "message": "XLE/XOP regime: MEAN_REVERTING → TRENDING"},
]
render_alert_banner(alerts)

render_inline_alert("P1", "MODEL STALENESS",
                    "Alpha model last updated 36h ago — re-run pipeline")
"""

from __future__ import annotations

from typing import Optional
import streamlit as st

from root.design_system import DS


# ---------------------------------------------------------------------------
# Level configuration
# ---------------------------------------------------------------------------

_LEVEL_CONFIG = {
    "P0": {
        "bg":     "#3D0000",
        "border": DS.CRITICAL,
        "icon":   "⛔",
        "label":  "P0 CRITICAL",
        "text":   DS.CRITICAL,
    },
    "P1": {
        "bg":     "#2A1200",
        "border": DS.NEGATIVE,
        "icon":   "🔴",
        "label":  "P1 HIGH",
        "text":   DS.NEGATIVE,
    },
    "P2": {
        "bg":     "#251A00",
        "border": DS.CAUTION,
        "icon":   "🟡",
        "label":  "P2 MEDIUM",
        "text":   DS.CAUTION,
    },
    "P3": {
        "bg":     "#001225",
        "border": DS.BRAND,
        "icon":   "🔵",
        "label":  "P3 INFO",
        "text":   DS.BRAND,
    },
    "P4": {
        "bg":     DS.BG_SURFACE,
        "border": DS.TEXT_MUTED,
        "icon":   "⚪",
        "label":  "P4 DEBUG",
        "text":   DS.TEXT_MUTED,
    },
}


# ---------------------------------------------------------------------------
# render_inline_alert
# ---------------------------------------------------------------------------

def render_inline_alert(
    level: str,
    title: str,
    message: str,
    *,
    timestamp: Optional[str] = None,
    dismissible: bool = False,
    key: Optional[str] = None,
) -> bool:
    """
    Render a single inline alert box.

    Returns True if the user dismissed the alert (via button click), else False.
    Only meaningful when dismissible=True.
    """
    cfg = _LEVEL_CONFIG.get(level.upper(), _LEVEL_CONFIG["P3"])

    ts_block = (
        f'<span style="font-size:10px;color:{DS.TEXT_MUTED};'
        f'margin-left:auto;">{timestamp}</span>'
        if timestamp else ""
    )

    html = f"""
    <div style="
        background:{cfg['bg']};
        border:1px solid {cfg['border']};
        border-left:4px solid {cfg['border']};
        border-radius:{DS.RADIUS_SM};
        padding:{DS.SPACE_3} {DS.SPACE_4};
        margin-bottom:{DS.SPACE_2};
    ">
      <div style="display:flex;align-items:center;gap:{DS.SPACE_2};margin-bottom:2px;">
        <span style="font-size:{DS.FONT_SM};">{cfg['icon']}</span>
        <span style="font-size:{DS.FONT_XS};font-weight:700;color:{cfg['text']};
                     letter-spacing:0.06em;">{cfg['label']}</span>
        <span style="font-size:{DS.FONT_SM};font-weight:700;
                     color:{DS.TEXT_PRIMARY};">&nbsp;{title}</span>
        {ts_block}
      </div>
      <div style="font-size:{DS.FONT_SM};color:{DS.TEXT_SECONDARY};
                  padding-left:24px;">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    if dismissible and key:
        dismiss_key = f"alert_dismissed_{key}"
        if st.button("Dismiss", key=dismiss_key, help="Remove this alert"):
            return True
    return False


# ---------------------------------------------------------------------------
# render_alert_banner
# ---------------------------------------------------------------------------

def render_alert_banner(
    alerts: list[dict],
    *,
    max_visible: int = 5,
    show_count: bool = True,
    collapsed_threshold: int = 3,
) -> None:
    """
    Render a stack of alert banners, sorted by severity.

    Parameters
    ----------
    alerts:              List of alert dicts with keys:
                         level (P0-P4), title, message, [timestamp], [key]
    max_visible:         Maximum alerts shown before collapse.
    show_count:          Show total count header.
    collapsed_threshold: Auto-collapse list when more than N alerts.
    """
    if not alerts:
        return

    level_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}
    sorted_alerts = sorted(
        alerts,
        key=lambda a: level_order.get(a.get("level", "P4").upper(), 4),
    )

    p0_count = sum(1 for a in alerts if a.get("level", "").upper() == "P0")
    p1_count = sum(1 for a in alerts if a.get("level", "").upper() == "P1")
    total    = len(alerts)

    if show_count:
        header_color = (
            DS.CRITICAL if p0_count > 0
            else DS.NEGATIVE if p1_count > 0
            else DS.CAUTION
        )
        badge_parts = []
        if p0_count:
            badge_parts.append(
                f'<span style="color:{DS.CRITICAL};font-weight:700;">{p0_count} P0</span>'
            )
        if p1_count:
            badge_parts.append(
                f'<span style="color:{DS.NEGATIVE};font-weight:700;">{p1_count} P1</span>'
            )
        others = total - p0_count - p1_count
        if others:
            badge_parts.append(
                f'<span style="color:{DS.TEXT_MUTED};">{others} lower</span>'
            )

        header_html = f"""
        <div style="display:flex;align-items:center;gap:{DS.SPACE_3};
                    margin-bottom:{DS.SPACE_2};">
          <span style="font-size:{DS.FONT_SM};font-weight:700;
                       color:{header_color};">ACTIVE ALERTS</span>
          <span style="font-size:{DS.FONT_XS};color:{DS.TEXT_MUTED};">
            {' · '.join(badge_parts)}
          </span>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

    visible = sorted_alerts[:max_visible]
    hidden  = sorted_alerts[max_visible:]

    for alert in visible:
        render_inline_alert(
            level=alert.get("level", "P3"),
            title=alert.get("title", "Alert"),
            message=alert.get("message", ""),
            timestamp=alert.get("timestamp"),
            dismissible=alert.get("dismissible", False),
            key=alert.get("key"),
        )

    if hidden:
        with st.expander(f"Show {len(hidden)} more alert(s)…"):
            for alert in hidden:
                render_inline_alert(
                    level=alert.get("level", "P3"),
                    title=alert.get("title", "Alert"),
                    message=alert.get("message", ""),
                    timestamp=alert.get("timestamp"),
                )
