# -*- coding: utf-8 -*-
# root/components/risk_ribbon.py
"""
Persistent Risk Ribbon Component
==================================
A fixed 48px horizontal strip showing live risk state at a glance.
Placed at the top of any tab that requires risk awareness.

Fields:
  NAV         — current portfolio net asset value
  Daily P&L   — today's realised + unrealised P&L
  Gross Exp.  — gross exposure as % of NAV
  Net Exp.    — net exposure as % of NAV
  Active Pairs— count of open pair positions
  Blocked     — count of pairs blocked by risk rules
  Mode        — system operational mode (drives ribbon border colour)

Usage
-----
from root.components.risk_ribbon import render_risk_ribbon

render_risk_ribbon(
    nav=1_250_000,
    daily_pnl=4_800,
    gross_exp=0.87,
    net_exp=0.12,
    active_pairs=8,
    blocked_pairs=1,
    mode="NOMINAL",
)
"""

from __future__ import annotations

from typing import Optional
import streamlit as st

from root.design_system import DS


# ---------------------------------------------------------------------------
# Mode → border colour
# ---------------------------------------------------------------------------

_MODE_BORDER = {
    "NOMINAL":     "rgba(255,255,255,0.07)",
    "DEGRADED":    DS.CAUTION,
    "RISK_BLIND":  "#FF6B35",
    "HALTED":      DS.CRITICAL,
    "MAINTENANCE": DS.BRAND,
}


def _pnl_color(pnl: float) -> str:
    if pnl > 0:
        return DS.POSITIVE
    if pnl < 0:
        return DS.NEGATIVE
    return DS.TEXT_MUTED


def _exp_color(exp_pct: float) -> str:
    if exp_pct >= 0.90:
        return DS.NEGATIVE
    if exp_pct >= 0.70:
        return DS.CAUTION
    return DS.POSITIVE


def _stat(label: str, value: str, color: str, min_width: str = "auto") -> str:
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                min-width:{min_width};padding:0 {DS.SPACE_3};">
      <span style="font-size:10px;color:{DS.TEXT_MUTED};text-transform:uppercase;
                   letter-spacing:0.06em;white-space:nowrap;">{label}</span>
      <span style="font-size:{DS.FONT_SM};font-weight:700;color:{color};
                   font-variant-numeric:tabular-nums;white-space:nowrap;">{value}</span>
    </div>
    """


def _separator() -> str:
    return (
        '<div style="width:1px;height:28px;background:rgba(255,255,255,0.08);'
        'align-self:center;"></div>'
    )


def _mode_badge(mode: str) -> str:
    cfg = {
        "NOMINAL":     (DS.POSITIVE,  "NOMINAL"),
        "DEGRADED":    (DS.CAUTION,   "DEGRADED"),
        "RISK_BLIND":  ("#FF6B35",    "RISK BLIND"),
        "HALTED":      (DS.CRITICAL,  "HALTED"),
        "MAINTENANCE": (DS.BRAND,     "MAINT"),
    }
    color, label = cfg.get(mode.upper(), (DS.TEXT_MUTED, mode))
    return (
        f'<span style="font-size:10px;font-weight:700;color:{color};'
        f'background:{color}1A;padding:2px 7px;border-radius:{DS.RADIUS_SM};'
        f'border:1px solid {color}44;letter-spacing:0.06em;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# render_risk_ribbon
# ---------------------------------------------------------------------------

def render_risk_ribbon(
    nav: float,
    daily_pnl: float,
    gross_exp: float,
    net_exp: float,
    active_pairs: int,
    blocked_pairs: int = 0,
    mode: str = "NOMINAL",
    *,
    max_dd_pct: Optional[float] = None,
    var_95: Optional[float] = None,
) -> None:
    """
    Render the persistent risk ribbon.

    Parameters
    ----------
    nav:          Net asset value in dollars.
    daily_pnl:    Today's P&L in dollars (signed).
    gross_exp:    Gross exposure as a fraction of NAV (e.g. 0.87 = 87%).
    net_exp:      Net exposure as a fraction of NAV (e.g. 0.12 = 12%).
    active_pairs: Number of currently open pair positions.
    blocked_pairs:Number of pairs blocked by risk rules.
    mode:         Operational mode string.
    max_dd_pct:   Current drawdown from HWM as positive % (optional).
    var_95:       1-day 95% VaR as positive $ amount (optional).
    """
    border_color = _MODE_BORDER.get(mode.upper(), "rgba(255,255,255,0.07)")

    pnl_sign = "+" if daily_pnl > 0 else ""
    nav_fmt   = f"${nav:,.0f}"
    pnl_fmt   = f"{pnl_sign}${daily_pnl:,.0f}"
    gross_fmt = f"{gross_exp * 100:.1f}%"
    net_fmt   = f"{net_exp * 100:.1f}%"

    stats = [
        _stat("NAV",      nav_fmt,   DS.TEXT_PRIMARY,  "90px"),
        _separator(),
        _stat("Daily P&L", pnl_fmt,  _pnl_color(daily_pnl), "80px"),
        _separator(),
        _stat("Gross Exp", gross_fmt, _exp_color(gross_exp), "70px"),
        _separator(),
        _stat("Net Exp",   net_fmt,   DS.TEXT_SECONDARY,     "60px"),
        _separator(),
        _stat("Pairs",     str(active_pairs), DS.TEXT_PRIMARY, "44px"),
    ]

    if blocked_pairs > 0:
        stats += [
            _separator(),
            _stat("Blocked", str(blocked_pairs), DS.NEGATIVE, "56px"),
        ]

    if max_dd_pct is not None:
        dd_color = (
            DS.NEGATIVE if max_dd_pct > 10
            else DS.CAUTION if max_dd_pct > 5
            else DS.TEXT_MUTED
        )
        stats += [
            _separator(),
            _stat("DD", f"-{max_dd_pct:.1f}%", dd_color, "52px"),
        ]

    if var_95 is not None:
        stats += [
            _separator(),
            _stat("1d VaR 95%", f"${var_95:,.0f}", DS.TEXT_SECONDARY, "80px"),
        ]

    html = f"""
    <div style="
        background:{DS.BG_RAISED};
        border:1px solid {border_color};
        border-radius:{DS.RADIUS_SM};
        padding:0 {DS.SPACE_4};
        margin-bottom:{DS.SPACE_3};
        display:flex;
        align-items:center;
        height:48px;
        overflow-x:auto;
        gap:0;
    ">
      {''.join(stats)}
      <div style="margin-left:auto;padding-left:{DS.SPACE_4};">{_mode_badge(mode)}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
