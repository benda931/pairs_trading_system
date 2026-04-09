# -*- coding: utf-8 -*-
# root/design_system.py
"""
Design System — Institutional Control Plane
============================================
Single source of truth for all design tokens, CSS variables, and
global style injection for the pairs-trading dashboard.

Usage
-----
from root.design_system import DS, inject_global_css, render_page_header

# Once, at app startup (inside dashboard.py main render):
inject_global_css()

# Every tab header:
render_page_header("RISK DASHBOARD", "Live portfolio risk metrics")
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Design Tokens
# ---------------------------------------------------------------------------

class DS:
    """Namespace for all design tokens. Import DS and use DS.BG_BASE, etc."""

    # ── Backgrounds ──────────────────────────────────────────────────────
    BG_BASE      = "#080F17"   # deepest canvas
    BG_SURFACE   = "#0D1B2A"   # card backgrounds, nav
    BG_RAISED    = "#1A2744"   # elevated panels, header
    BG_OVERLAY   = "#243860"   # modals, hover states
    BG_INPUT     = "#151F2E"   # input fields, code blocks
    BG_GRID      = "rgba(255,255,255,0.04)"  # subtle grid

    # ── Semantic Colours ─────────────────────────────────────────────────
    BRAND        = "#1976D2"   # primary blue — one blue only
    BRAND_DIM    = "#0D47A1"

    POSITIVE     = "#1BB152"   # confirmed success, profitable
    POSITIVE_DIM = "rgba(27,177,82,0.15)"

    NEGATIVE     = "#D93025"   # drawdown, loss, block
    NEGATIVE_DIM = "rgba(217,48,37,0.15)"

    CAUTION      = "#F5A623"   # warning, review needed
    CAUTION_DIM  = "rgba(245,166,35,0.15)"

    NEUTRAL      = "#546E7A"   # inactive, n/a, unknown

    CRITICAL     = "#FF3B3B"   # P0 incidents, kill-switch
    CRITICAL_DIM = "rgba(255,59,59,0.15)"

    RESEARCH     = "#00897B"   # research mode, validated
    ADVISORY     = "#7B1FA2"   # agent advisory, AI output

    # ── Text ─────────────────────────────────────────────────────────────
    TEXT_PRIMARY   = "#E8EDF2"   # headlines, values
    TEXT_SECONDARY = "#8FA3B1"   # labels, descriptions
    TEXT_MUTED     = "#546E7A"   # metadata, timestamps
    TEXT_DISABLED  = "#37474F"   # unavailable, placeholder
    TEXT_CODE      = "#A8D8A8"   # monospace values

    # ── Borders ──────────────────────────────────────────────────────────
    BORDER_SUBTLE  = "rgba(255,255,255,0.06)"
    BORDER_DEFAULT = "rgba(255,255,255,0.12)"
    BORDER_STRONG  = "rgba(255,255,255,0.20)"
    BORDER_RADIUS  = "6px"
    BORDER_RADIUS_LG = "10px"
    BORDER_RADIUS_PILL = "16px"

    # ── Shorthand radius aliases (match BORDER_RADIUS_* family) ──────────
    RADIUS_SM  = "4px"
    RADIUS_MD  = "6px"
    RADIUS_LG  = "10px"
    RADIUS_XL  = "14px"

    # ── Shadows ───────────────────────────────────────────────────────────
    SHADOW_SM = "0 1px 3px rgba(0,0,0,0.30)"
    SHADOW_MD = "0 2px 8px rgba(0,0,0,0.40)"
    SHADOW_LG = "0 4px 16px rgba(0,0,0,0.50)"

    # ── Spacing (4px base grid) ───────────────────────────────────────────
    SPACE_1  = "4px"
    SPACE_2  = "8px"
    SPACE_3  = "12px"
    SPACE_4  = "16px"
    SPACE_5  = "20px"
    SPACE_6  = "24px"
    SPACE_8  = "32px"
    SPACE_10 = "40px"

    # ── Typography ────────────────────────────────────────────────────────
    FONT_XS   = "11px"
    FONT_SM   = "12px"
    FONT_BASE = "14px"
    FONT_MD   = "16px"
    FONT_LG   = "20px"
    FONT_XL   = "28px"
    FONT_2XL  = "36px"
    FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

    # ── Environment badge colours ─────────────────────────────────────────
    ENV_COLORS: dict[str, str] = {
        "live":      "#FF3B3B",   # critical red
        "paper":     "#1976D2",   # brand blue
        "dev":       "#546E7A",   # neutral
        "research":  "#00897B",   # research teal
        "backtest":  "#7B1FA2",   # advisory purple
        "staging":   "#F5A623",   # caution amber
    }

    # ── Regime colours ────────────────────────────────────────────────────
    REGIME_COLORS: dict[str, str] = {
        "MEAN_REVERTING": "#1BB152",
        "TRENDING":       "#F5A623",
        "HIGH_VOL":       "#FF6B35",
        "CRISIS":         "#D93025",
        "BROKEN":         "#7B1FA2",
        "UNKNOWN":        "#546E7A",
    }

    # ── Grade colours ─────────────────────────────────────────────────────
    GRADE_COLORS: dict[str, str] = {
        "A":  "#1BB152",
        "A+": "#1BB152",
        "B":  "#42A5F5",
        "B+": "#42A5F5",
        "C":  "#F5A623",
        "D":  "#D93025",
        "F":  "#FF3B3B",
    }

    # ── Chart Colours (semantic, not aesthetic) ───────────────────────────
    CHART_PRIMARY   = "#1976D2"
    CHART_SECONDARY = "#546E7A"
    CHART_POSITIVE  = "#1BB152"
    CHART_NEGATIVE  = "#D93025"
    CHART_CAUTION   = "#F5A623"
    CHART_NEUTRAL   = "#8FA3B1"
    CHART_RESEARCH  = "#00897B"
    CHART_ADVISORY  = "#7B1FA2"

    # ── Pipeline stage state colours ─────────────────────────────────────
    STAGE_COMPLETE = "#1BB152"
    STAGE_RUNNING  = "#1976D2"
    STAGE_FAILED   = "#D93025"
    STAGE_PENDING  = "#546E7A"
    STAGE_WARN     = "#F5A623"


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

_GLOBAL_CSS = f"""
<style>
/* ── Reset & base ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: {DS.FONT_STACK} !important;
}}

/* ── Hide Streamlit chrome ───────────────────────────────────────── */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

/* ── App background ─────────────────────────────────────────────── */
.stApp {{
    background-color: {DS.BG_BASE};
}}

/* ── Remove default padding on main block ───────────────────────── */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1600px !important;
}}

/* ── Tab navigation ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {DS.BG_SURFACE};
    border-bottom: 1px solid {DS.BORDER_DEFAULT};
    gap: 0;
    padding: 0 16px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {DS.TEXT_SECONDARY} !important;
    font-size: {DS.FONT_SM} !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color: {DS.TEXT_PRIMARY} !important;
    border-bottom: 2px solid {DS.BRAND} !important;
    font-weight: 600 !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 16px;
}}

/* ── Metric cards ───────────────────────────────────────────────── */
[data-testid="metric-container"] {{
    background: {DS.BG_SURFACE};
    border: 1px solid {DS.BORDER_SUBTLE};
    border-radius: {DS.BORDER_RADIUS_LG};
    padding: 16px 20px !important;
    box-shadow: {DS.SHADOW_SM};
}}
[data-testid="metric-container"] label {{
    color: {DS.TEXT_SECONDARY} !important;
    font-size: {DS.FONT_SM} !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {DS.TEXT_PRIMARY} !important;
    font-size: {DS.FONT_XL} !important;
    font-weight: 700 !important;
    line-height: 1.2;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: {DS.FONT_SM} !important;
}}

/* ── Dataframes ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {DS.BORDER_SUBTLE} !important;
    border-radius: {DS.BORDER_RADIUS} !important;
}}

/* ── Expanders ──────────────────────────────────────────────────── */
.streamlit-expanderHeader {{
    background: {DS.BG_SURFACE} !important;
    border: 1px solid {DS.BORDER_SUBTLE} !important;
    border-radius: {DS.BORDER_RADIUS} !important;
    color: {DS.TEXT_SECONDARY} !important;
    font-size: {DS.FONT_SM} !important;
    font-weight: 600 !important;
}}
.streamlit-expanderContent {{
    background: {DS.BG_BASE} !important;
    border: 1px solid {DS.BORDER_SUBTLE} !important;
    border-top: none !important;
}}

/* ── Buttons ────────────────────────────────────────────────────── */
.stButton > button {{
    background: {DS.BG_RAISED} !important;
    border: 1px solid {DS.BORDER_DEFAULT} !important;
    color: {DS.TEXT_PRIMARY} !important;
    border-radius: {DS.BORDER_RADIUS} !important;
    font-size: {DS.FONT_SM} !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease;
}}
.stButton > button:hover {{
    background: {DS.BG_OVERLAY} !important;
    border-color: {DS.BRAND} !important;
    color: {DS.TEXT_PRIMARY} !important;
}}
.stButton [kind="primary"] > button,
.stButton > button[kind="primary"] {{
    background: {DS.BRAND} !important;
    border-color: {DS.BRAND} !important;
}}

/* ── Select / multiselect ───────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {DS.BG_INPUT} !important;
    border: 1px solid {DS.BORDER_DEFAULT} !important;
    border-radius: {DS.BORDER_RADIUS} !important;
    color: {DS.TEXT_PRIMARY} !important;
}}

/* ── Number inputs ──────────────────────────────────────────────── */
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {{
    background: {DS.BG_INPUT} !important;
    border: 1px solid {DS.BORDER_DEFAULT} !important;
    border-radius: {DS.BORDER_RADIUS} !important;
    color: {DS.TEXT_PRIMARY} !important;
    font-family: {DS.FONT_STACK} !important;
}}

/* ── Sliders ────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] {{
    padding: 0 4px;
}}

/* ── Captions ───────────────────────────────────────────────────── */
.stCaption {{
    color: {DS.TEXT_MUTED} !important;
    font-size: {DS.FONT_XS} !important;
}}

/* ── Info/Warning/Error/Success ─────────────────────────────────── */
.stAlert {{
    border-radius: {DS.BORDER_RADIUS} !important;
    font-size: {DS.FONT_SM} !important;
}}

/* ── Dividers ───────────────────────────────────────────────────── */
hr {{
    border-color: {DS.BORDER_SUBTLE} !important;
    margin: 12px 0 !important;
}}

/* ── Headings in markdown ───────────────────────────────────────── */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5 {{
    color: {DS.TEXT_PRIMARY} !important;
    font-family: {DS.FONT_STACK} !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}}
.stMarkdown h3 {{
    font-size: {DS.FONT_MD} !important;
    font-weight: 600 !important;
    color: {DS.TEXT_SECONDARY} !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 20px 0 8px 0;
}}

/* ── Custom component classes ───────────────────────────────────── */

/* Panel (replaces expanders as layout primitive) */
.ds-panel {{
    background: {DS.BG_SURFACE};
    border: 1px solid {DS.BORDER_SUBTLE};
    border-radius: {DS.BORDER_RADIUS_LG};
    margin-bottom: 12px;
    overflow: hidden;
    box-shadow: {DS.SHADOW_SM};
}}
.ds-panel-header {{
    background: {DS.BG_RAISED};
    border-bottom: 1px solid {DS.BORDER_SUBTLE};
    padding: 10px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.ds-panel-header-text {{
    font-size: {DS.FONT_SM};
    font-weight: 700;
    color: {DS.TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}
.ds-panel-body {{
    padding: 16px;
}}

/* Badge */
.ds-badge {{
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: {DS.BORDER_RADIUS_PILL};
    font-size: {DS.FONT_XS};
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    line-height: 1.6;
}}
.ds-badge-positive  {{ background: {DS.POSITIVE_DIM};  color: {DS.POSITIVE};  }}
.ds-badge-negative  {{ background: {DS.NEGATIVE_DIM};  color: {DS.NEGATIVE};  }}
.ds-badge-caution   {{ background: {DS.CAUTION_DIM};   color: {DS.CAUTION};   }}
.ds-badge-brand     {{ background: rgba(25,118,210,0.15); color: {DS.BRAND};  }}
.ds-badge-critical  {{ background: {DS.CRITICAL_DIM};  color: {DS.CRITICAL};  }}
.ds-badge-research  {{ background: rgba(0,137,123,0.15); color: {DS.RESEARCH}; }}
.ds-badge-advisory  {{ background: rgba(123,31,162,0.15); color: {DS.ADVISORY}; }}
.ds-badge-neutral   {{ background: rgba(84,110,122,0.15); color: {DS.NEUTRAL};  }}

/* KPI card tier-1 */
.ds-kpi-t1 {{
    background: {DS.BG_SURFACE};
    border: 1px solid {DS.BORDER_SUBTLE};
    border-radius: {DS.BORDER_RADIUS_LG};
    padding: 20px 22px 16px;
    box-shadow: {DS.SHADOW_SM};
    min-height: 110px;
}}
.ds-kpi-label {{
    font-size: {DS.FONT_XS};
    font-weight: 600;
    color: {DS.TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 6px;
}}
.ds-kpi-value {{
    font-size: {DS.FONT_XL};
    font-weight: 700;
    color: {DS.TEXT_PRIMARY};
    line-height: 1.2;
    font-variant-numeric: tabular-nums;
}}
.ds-kpi-delta {{
    font-size: {DS.FONT_XS};
    margin-top: 4px;
    color: {DS.TEXT_MUTED};
}}
.ds-kpi-delta-pos {{ color: {DS.POSITIVE}; }}
.ds-kpi-delta-neg {{ color: {DS.NEGATIVE}; }}
.ds-kpi-sub {{
    font-size: {DS.FONT_XS};
    color: {DS.TEXT_MUTED};
    margin-top: 6px;
    border-top: 1px solid {DS.BORDER_SUBTLE};
    padding-top: 6px;
}}

/* Progress bar (limit gauge) */
.ds-limit-bar-wrap {{
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    height: 4px;
    margin-top: 8px;
    overflow: hidden;
}}
.ds-limit-bar-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}}

/* Signal card */
.ds-signal-card {{
    background: {DS.BG_SURFACE};
    border: 1px solid {DS.BORDER_SUBTLE};
    border-left: 3px solid {DS.BORDER_SUBTLE};
    border-radius: {DS.BORDER_RADIUS};
    padding: 12px 16px;
    margin-bottom: 8px;
    box-shadow: {DS.SHADOW_SM};
}}
.ds-signal-card-positive {{ border-left-color: {DS.POSITIVE} !important; }}
.ds-signal-card-negative {{ border-left-color: {DS.NEGATIVE} !important; }}
.ds-signal-card-caution  {{ border-left-color: {DS.CAUTION}  !important; }}
.ds-signal-card-advisory {{ border-left-color: {DS.ADVISORY} !important; }}
.ds-signal-card-neutral  {{ border-left-color: {DS.NEUTRAL}  !important; }}

/* State banner */
.ds-state-banner {{
    padding: 8px 16px;
    border-radius: {DS.BORDER_RADIUS};
    margin-bottom: 12px;
    font-size: {DS.FONT_SM};
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.ds-state-degraded  {{ background: rgba(245,166,35,0.12); border: 1px solid rgba(245,166,35,0.30); color: {DS.CAUTION}; }}
.ds-state-blind     {{ background: rgba(217,48,37,0.12);  border: 1px solid rgba(217,48,37,0.30);  color: {DS.NEGATIVE}; }}
.ds-state-halted    {{ background: rgba(255,59,59,0.15);  border: 1px solid rgba(255,59,59,0.50);  color: {DS.CRITICAL}; font-weight: 700; }}

/* Risk ribbon */
.ds-risk-ribbon {{
    background: {DS.BG_RAISED};
    border: 1px solid {DS.BORDER_SUBTLE};
    border-radius: {DS.BORDER_RADIUS};
    padding: 8px 16px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 20px;
    font-size: {DS.FONT_XS};
    flex-wrap: wrap;
}}
.ds-risk-ribbon-nominal  {{ border-left: 3px solid {DS.POSITIVE}; }}
.ds-risk-ribbon-degraded {{ border-left: 3px solid {DS.CAUTION}; }}
.ds-risk-ribbon-blind    {{ border-left: 3px solid {DS.NEGATIVE}; }}
.ds-risk-ribbon-halted   {{ border-left: 3px solid {DS.CRITICAL}; }}

/* Stale data watermark */
.ds-stale-cap {{
    font-size: {DS.FONT_XS};
    color: {DS.CAUTION};
    background: rgba(245,166,35,0.08);
    border: 1px solid rgba(245,166,35,0.20);
    border-radius: 4px;
    padding: 2px 8px;
    display: inline-block;
    margin-bottom: 4px;
}}

/* Section heading */
.ds-section-heading {{
    font-size: {DS.FONT_SM};
    font-weight: 700;
    color: {DS.TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid {DS.BORDER_SUBTLE};
}}

/* Page header */
.ds-page-header {{
    background: linear-gradient(90deg, {DS.BG_RAISED} 0%, {DS.BG_SURFACE} 100%);
    border-left: 4px solid {DS.BRAND};
    border-radius: 0 {DS.BORDER_RADIUS_LG} {DS.BORDER_RADIUS_LG} 0;
    padding: 14px 20px;
    margin-bottom: 16px;
    box-shadow: {DS.SHADOW_MD};
}}
.ds-page-header-title {{
    font-size: {DS.FONT_LG};
    font-weight: 800;
    color: {DS.TEXT_PRIMARY};
    letter-spacing: -0.02em;
    line-height: 1.2;
}}
.ds-page-header-sub {{
    font-size: {DS.FONT_SM};
    color: {DS.TEXT_SECONDARY};
    margin-top: 3px;
}}

/* Alert tiers */
.ds-alert-p0 {{
    background: rgba(255,59,59,0.12);
    border: 1px solid rgba(255,59,59,0.40);
    border-left: 4px solid {DS.CRITICAL};
    border-radius: {DS.BORDER_RADIUS};
    padding: 12px 16px;
    margin-bottom: 8px;
    color: {DS.TEXT_PRIMARY};
}}
.ds-alert-p1 {{
    background: rgba(217,48,37,0.10);
    border: 1px solid rgba(217,48,37,0.30);
    border-left: 4px solid {DS.NEGATIVE};
    border-radius: {DS.BORDER_RADIUS};
    padding: 10px 16px;
    margin-bottom: 6px;
    color: {DS.TEXT_PRIMARY};
}}
.ds-alert-p2 {{
    background: rgba(245,166,35,0.08);
    border: 1px solid rgba(245,166,35,0.25);
    border-left: 3px solid {DS.CAUTION};
    border-radius: {DS.BORDER_RADIUS};
    padding: 10px 16px;
    margin-bottom: 6px;
    color: {DS.TEXT_PRIMARY};
}}
.ds-alert-p3 {{
    background: rgba(25,118,210,0.08);
    border: 1px solid rgba(25,118,210,0.20);
    border-left: 2px solid {DS.BRAND};
    border-radius: {DS.BORDER_RADIUS};
    padding: 8px 14px;
    margin-bottom: 4px;
    color: {DS.TEXT_SECONDARY};
}}

.ds-alert-title {{
    font-size: {DS.FONT_BASE};
    font-weight: 700;
    margin-bottom: 2px;
}}
.ds-alert-body {{
    font-size: {DS.FONT_SM};
    color: {DS.TEXT_SECONDARY};
}}
.ds-alert-meta {{
    font-size: {DS.FONT_XS};
    color: {DS.TEXT_MUTED};
    margin-top: 4px;
}}

/* Workflow timeline */
.ds-timeline {{
    display: flex;
    align-items: center;
    gap: 0;
    background: {DS.BG_RAISED};
    border: 1px solid {DS.BORDER_SUBTLE};
    border-radius: {DS.BORDER_RADIUS};
    padding: 8px 16px;
    margin-bottom: 12px;
    overflow-x: auto;
    flex-wrap: wrap;
    gap: 4px;
}}
.ds-stage {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: {DS.FONT_XS};
    padding: 4px 10px;
    border-radius: {DS.BORDER_RADIUS};
    white-space: nowrap;
    font-weight: 500;
}}
.ds-stage-complete {{ color: {DS.POSITIVE}; background: {DS.POSITIVE_DIM}; }}
.ds-stage-running  {{ color: {DS.BRAND};    background: rgba(25,118,210,0.15); }}
.ds-stage-failed   {{ color: {DS.NEGATIVE}; background: {DS.NEGATIVE_DIM}; }}
.ds-stage-pending  {{ color: {DS.NEUTRAL};  background: rgba(84,110,122,0.12); }}
.ds-stage-warn     {{ color: {DS.CAUTION};  background: {DS.CAUTION_DIM}; }}
.ds-stage-arrow    {{ color: {DS.TEXT_MUTED}; font-size: 10px; padding: 0 2px; }}

/* Execution row */
.ds-exec-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border-bottom: 1px solid {DS.BORDER_SUBTLE};
    font-size: {DS.FONT_SM};
    transition: background 0.1s;
}}
.ds-exec-row:hover {{ background: {DS.BG_RAISED}; }}
.ds-exec-row:last-child {{ border-bottom: none; }}

/* Precision table row */
.ds-precision-good {{ color: {DS.POSITIVE}; }}
.ds-precision-warn {{ color: {DS.CAUTION}; }}
.ds-precision-bad  {{ color: {DS.NEGATIVE}; }}

/* Code / monospace */
.ds-mono {{
    font-family: 'SF Mono', 'Fira Code', 'Fira Mono', monospace;
    font-size: {DS.FONT_XS};
    color: {DS.TEXT_CODE};
    background: rgba(168,216,168,0.08);
    padding: 1px 5px;
    border-radius: 3px;
}}

</style>
"""


def inject_global_css() -> None:
    """Inject the global design-system CSS. Call once at app startup."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page header helper
# ---------------------------------------------------------------------------

def render_page_header(
    title: str,
    subtitle: str = "",
    accent_color: str = DS.BRAND,
    badges: list[tuple[str, str]] | None = None,
) -> None:
    """
    Render a consistent institutional page/tab header.

    Parameters
    ----------
    title:        Section title (e.g. "RISK DASHBOARD")
    subtitle:     Optional descriptor line
    accent_color: Left-border colour (default brand blue)
    badges:       Optional list of (label, semantic) tuples for badges.
                  Semantic values: "positive","negative","caution","brand",
                  "critical","research","advisory","neutral"
    """
    badge_html = ""
    if badges:
        for label, semantic in badges:
            badge_html += (
                f'<span class="ds-badge ds-badge-{semantic}" '
                f'style="margin-left:10px;">{label}</span>'
            )

    sub_html = ""
    if subtitle:
        sub_html = (
            f'<div class="ds-page-header-sub">{subtitle}</div>'
        )

    st.markdown(
        f"""
        <div class="ds-page-header"
             style="border-left-color:{accent_color}">
            <div style="display:flex;align-items:center;flex-wrap:wrap;">
                <span class="ds-page-header-title">{title}</span>
                {badge_html}
            </div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(text: str) -> None:
    """Render a uniform section separator/heading."""
    st.markdown(
        f'<div class="ds-section-heading">{text}</div>',
        unsafe_allow_html=True,
    )


def render_badge(label: str, semantic: str = "neutral") -> str:
    """Return badge HTML string (use inside st.markdown unsafe_allow_html)."""
    return f'<span class="ds-badge ds-badge-{semantic}">{label}</span>'


def render_mono(value: str) -> str:
    """Return monospace-styled value HTML string."""
    return f'<span class="ds-mono">{value}</span>'
