# -*- coding: utf-8 -*-
"""
root/alpha_dashboard_tab.py — Alpha Performance Dashboard
==========================================================

Institutional-grade dashboard:
1. Portfolio equity curve + drawdown
2. Alpha pair Sharpe bar chart
3. Pair details table
4. Trade blotter
5. Portfolio composition over time
6. Risk summary

All data comes from logs/alpha_results/ and logs/backtests/.
All charts use the universal dark theme (no plotly_white).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from root.design_system import DS, inject_global_css, render_page_header, render_section_heading
from root.charts import (
    equity_curve_fig,
    drawdown_fig,
    pair_sharpe_bar_fig,
)
from root.components.kpi import render_kpi_row
from root.components.risk_ribbon import render_risk_ribbon

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALPHA_DIR    = PROJECT_ROOT / "logs" / "alpha_results"
BACKTEST_DIR = PROJECT_ROOT / "logs" / "backtests"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_alpha_configs() -> list[dict]:
    path = ALPHA_DIR / "alpha_pairs_latest.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _load_trade_log() -> pd.DataFrame:
    csvs = sorted(BACKTEST_DIR.glob("trade_log_*.csv"))
    if not csvs:
        return pd.DataFrame()
    try:
        return pd.read_csv(csvs[-1])
    except Exception:
        return pd.DataFrame()


def _run_portfolio_backtest() -> Optional[Any]:
    try:
        from core.portfolio_backtester import run_portfolio_backtest, PortfolioConfig
        configs = _load_alpha_configs()
        if not configs:
            return None
        pairs = []
        for ac in configs:
            pair   = ac["pair"].split("/")
            params = ac.get("params", {})
            pairs.append({
                "sym_x":    pair[0],
                "sym_y":    pair[1],
                "z_open":   params.get("z_open", 2.0),
                "z_close":  params.get("z_close", 0.5),
                "stop_z":   params.get("stop_z", 4.0),
                "lookback": int(params.get("lookback", 60)),
            })
        config = PortfolioConfig(
            initial_capital=1_000_000.0,
            vol_target=0.10,
            kelly_fraction=0.25,
            max_position_weight=0.10,
        )
        return run_portfolio_backtest(pairs, config)
    except Exception as e:
        logger.warning("Portfolio backtest failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------

def _build_kpi_cards(result: Any) -> list[dict]:
    """Convert backtest result to render_kpi_row card spec."""
    def _s(attr: str, default: Any = None) -> Any:
        return getattr(result, attr, default)

    sharpe = _s("sharpe", 0.0)
    spy_sharpe = 0.69  # SPY long-run benchmark

    cards = [
        dict(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}",
            delta=f"{sharpe - spy_sharpe:+.2f} vs SPY",
            delta_positive=sharpe > spy_sharpe,
            tier=1,
            semantic="positive" if sharpe > 1.0 else "caution" if sharpe > 0.5 else "negative",
        ),
        dict(
            label="CAGR",
            value=f"{_s('cagr', 0.0):.1f}%",
            tier=2,
            semantic="positive" if _s('cagr', 0) > 8 else "neutral",
        ),
        dict(
            label="Max Drawdown",
            value=f"{_s('max_drawdown', 0.0):.1f}%",
            limit=20.0,
            current=abs(_s('max_drawdown', 0.0)),
            tier=2,
            semantic="positive" if abs(_s('max_drawdown', 0)) < 10 else "caution",
        ),
        dict(
            label="Sortino",
            value=f"{_s('sortino', 0.0):.2f}",
            tier=2,
            semantic="positive" if _s('sortino', 0) > 1.5 else "neutral",
        ),
        dict(
            label="Win Rate",
            value=f"{_s('win_rate', 0.0):.0f}%",
            tier=2,
            semantic="positive" if _s('win_rate', 0) > 55 else "neutral",
        ),
        dict(
            label="Trades",
            value=f"{_s('n_trades', 0):,}",
            tier=2,
        ),
    ]
    return cards


def _build_risk_kpi_cards(result: Any) -> list[dict]:
    return [
        dict(
            label="Annual Vol",
            value=f"{getattr(result, 'annual_vol', 0.0):.1f}%",
            subtitle="Target 8–12%",
            semantic="positive" if 8 < getattr(result, 'annual_vol', 0) < 12 else "caution",
        ),
        dict(
            label="Calmar",
            value=f"{getattr(result, 'calmar', 0.0):.2f}",
            semantic="positive" if getattr(result, 'calmar', 0) > 0.5 else "neutral",
        ),
        dict(
            label="Avg Active Pairs",
            value=f"{getattr(result, 'avg_pairs_active', 0.0):.1f}",
        ),
        dict(
            label="Total Costs",
            value=f"${getattr(result, 'total_costs', 0):,.0f}",
            semantic="caution" if getattr(result, 'total_costs', 0) > 10_000 else "neutral",
        ),
    ]


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_equity_section(result: Any) -> None:
    render_section_heading("Portfolio Equity Curve")

    eq = getattr(result, "equity_curve", None)
    if eq is None or len(eq) == 0:
        st.info("No equity curve data available.")
        return

    # Benchmark (SPY) — best-effort
    benchmark = None
    try:
        from common.data_loader import load_price_data
        spy = load_price_data("SPY")["close"]
        benchmark = spy
    except Exception:
        pass

    fig = equity_curve_fig(
        equity=eq,
        title="PORTFOLIO EQUITY",
        benchmark=benchmark,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_drawdown_section(result: Any) -> None:
    dd = getattr(result, "drawdown_series", None)
    if dd is None or len(dd) == 0:
        return

    render_section_heading("Drawdown")
    fig = drawdown_fig(dd, title="PORTFOLIO DRAWDOWN", height=220)
    st.plotly_chart(fig, use_container_width=True)


def _render_pair_performance_section(alpha_configs: list[dict]) -> None:
    render_section_heading("Alpha Pair Performance")

    df = pd.DataFrame(alpha_configs)
    if df.empty or "sharpe" not in df.columns:
        st.info("No pair performance data available.")
        return

    fig = pair_sharpe_bar_fig(
        pairs=df["pair"].tolist(),
        sharpes=df["sharpe"].tolist(),
        title="PAIR SHARPE RATIOS",
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Details table
    render_section_heading("Pair Details")
    display_cols = ["pair", "sharpe", "return", "max_dd", "win_rate", "n_trades"]
    available    = [c for c in display_cols if c in df.columns]
    if available:
        styled         = df[available].sort_values("sharpe", ascending=False).reset_index(drop=True)
        styled.index  += 1
        styled.columns = [c.replace("_", " ").title() for c in available]
        st.dataframe(styled, use_container_width=True,
                     height=min(420, 36 * len(styled) + 44))


def _render_trade_blotter(result: Any) -> None:
    render_section_heading("Recent Trades")
    trade_log = _load_trade_log()

    if not trade_log.empty:
        recent = trade_log.tail(25).iloc[::-1]
        st.dataframe(recent, use_container_width=True, height=400)
        return

    result_log = getattr(result, "trade_log", None) if result else None
    if result_log:
        recent = pd.DataFrame(result_log[-25:]).iloc[::-1]
        st.dataframe(recent, use_container_width=True, height=400)
        return

    st.info("No trade log available. Run a backtest first.")


def _render_composition_section(result: Any) -> None:
    wh = getattr(result, "weights_history", None)
    if wh is None or len(wh) == 0:
        return

    render_section_heading("Portfolio Composition Over Time")

    active_cols = [c for c in wh.columns if wh[c].abs().sum() > 0]
    if not active_cols:
        return

    import plotly.graph_objects as go
    from root.charts.theme import apply_theme, COLOUR_SEQUENCE

    fig = go.Figure()
    for i, col in enumerate(active_cols[:12]):
        fig.add_trace(go.Scatter(
            x=wh.index,
            y=wh[col] * 100,
            mode="lines",
            name=col,
            stackgroup="one",
            line=dict(color=COLOUR_SEQUENCE[i % len(COLOUR_SEQUENCE)], width=1),
        ))
    fig.update_yaxes(title_text="Weight (%)", ticksuffix="%")
    fig.update_xaxes(title_text="Date")
    fig = apply_theme(fig, title="PORTFOLIO COMPOSITION", height=340)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------

def render_alpha_dashboard(
    app_ctx: Any = None,
    feature_flags: dict = None,
    nav_payload: dict = None,
) -> None:
    """Main render function for the Alpha Dashboard tab."""

    inject_global_css()

    # ── Page header ───────────────────────────────────────────────
    render_page_header(
        title="Alpha Performance Dashboard",
        subtitle="Portfolio analytics · Risk metrics · Trade blotter · Walk-forward validation",
        accent_color=DS.BRAND,
        badges=[("LIVE", "positive"), ("PAPER", "research")],
    )

    # ── Load data ─────────────────────────────────────────────────
    alpha_configs = _load_alpha_configs()

    if not alpha_configs:
        st.warning(
            "No alpha pair configurations found. "
            "Run the alpha pipeline first:\n\n"
            "```\npython scripts/run_full_alpha.py --universe all --trials 30\n```"
        )
        return

    # ── Run or restore backtest ───────────────────────────────────
    result = st.session_state.get("portfolio_result")
    if result is None:
        with st.spinner("Running portfolio backtest…"):
            result = _run_portfolio_backtest()
            if result:
                st.session_state["portfolio_result"] = result

    # ── Risk ribbon ───────────────────────────────────────────────
    if result and hasattr(result, "sharpe"):
        render_risk_ribbon(
            nav=getattr(result, "final_nav", 1_000_000),
            daily_pnl=0.0,  # No intraday P&L in backtest context
            gross_exp=getattr(result, "avg_gross_exposure", 0.5),
            net_exp=getattr(result, "avg_net_exposure", 0.05),
            active_pairs=len(alpha_configs),
            blocked_pairs=0,
            mode="NOMINAL",
            max_dd_pct=abs(getattr(result, "max_drawdown", 0.0)),
        )

    # ── KPI row ───────────────────────────────────────────────────
    if result and hasattr(result, "sharpe"):
        render_section_heading("Key Performance Indicators")
        render_kpi_row(_build_kpi_cards(result))

        st.markdown(f"<div style='height:{DS.SPACE_4}px'></div>", unsafe_allow_html=True)

        # ── Equity + Drawdown ─────────────────────────────────────
        _render_equity_section(result)
        _render_drawdown_section(result)

    # ── Pair performance ──────────────────────────────────────────
    _render_pair_performance_section(alpha_configs)

    # ── Trade blotter ─────────────────────────────────────────────
    _render_trade_blotter(result)

    # ── Portfolio composition ─────────────────────────────────────
    if result:
        _render_composition_section(result)

    # ── Risk summary ─────────────────────────────────────────────
    if result and hasattr(result, "annual_vol"):
        render_section_heading("Risk Summary")
        render_kpi_row(_build_risk_kpi_cards(result))

    # ── Footer ────────────────────────────────────────────────────
    st.markdown(
        f"<div style='height:{DS.SPACE_4}px'></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Data as of {date.today()} · {len(alpha_configs)} alpha pairs configured · "
        f"Run `python scripts/run_full_alpha.py` to refresh"
    )
