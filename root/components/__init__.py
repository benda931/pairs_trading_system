# root/components/__init__.py
"""
Institutional UI Component Library
====================================
Pre-built Streamlit components using the DS design tokens.
All components render semantic HTML via st.markdown(unsafe_allow_html=True).

Usage
-----
from root.components import render_kpi_card, render_alert_banner, render_risk_ribbon

render_risk_ribbon(nav=1_250_000, daily_pnl=4_800, gross_exp=0.87, net_exp=0.12,
                   active_pairs=8, blocked_pairs=1, mode="NOMINAL")
"""

from root.components.kpi import render_kpi_card, render_kpi_row
from root.components.signal_card import render_signal_card, render_signal_grid
from root.components.alert_banner import render_alert_banner, render_inline_alert
from root.components.state_banner import render_state_banner
from root.components.risk_ribbon import render_risk_ribbon
from root.components.workflow_timeline import render_workflow_timeline

__all__ = [
    "render_kpi_card",
    "render_kpi_row",
    "render_signal_card",
    "render_signal_grid",
    "render_alert_banner",
    "render_inline_alert",
    "render_state_banner",
    "render_risk_ribbon",
    "render_workflow_timeline",
]
