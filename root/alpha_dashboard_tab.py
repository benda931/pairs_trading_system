# -*- coding: utf-8 -*-
"""
root/alpha_dashboard_tab.py — Professional Alpha Performance Dashboard
======================================================================

Clean, English-language dashboard showing:
1. Portfolio equity curve + drawdown
2. Alpha pair performance heatmap
3. Trade blotter (recent trades)
4. Key risk metrics
5. Walk-forward validation results
6. Signal status for active pairs

All data comes from logs/alpha_results/ and logs/backtests/.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALPHA_DIR = PROJECT_ROOT / "logs" / "alpha_results"
BACKTEST_DIR = PROJECT_ROOT / "logs" / "backtests"


def _load_alpha_configs() -> list[dict]:
    """Load latest alpha pair configurations."""
    path = ALPHA_DIR / "alpha_pairs_latest.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _load_equity_curve() -> Optional[pd.Series]:
    """Load latest portfolio equity curve."""
    path = BACKTEST_DIR / "portfolio_equity_latest.csv"
    if not path.exists():
        # Try finding any equity curve
        csvs = sorted(BACKTEST_DIR.glob("equity_curves_*.csv"))
        if csvs:
            path = csvs[-1]
        else:
            return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        # Use first column as equity
        return df.iloc[:, 0]
    except Exception:
        return None


def _load_trade_log() -> pd.DataFrame:
    """Load latest trade log."""
    csvs = sorted(BACKTEST_DIR.glob("trade_log_*.csv"))
    if not csvs:
        return pd.DataFrame()
    try:
        return pd.read_csv(csvs[-1])
    except Exception:
        return pd.DataFrame()


def _run_portfolio_backtest() -> Optional[dict]:
    """Run portfolio backtest and return result."""
    try:
        from core.portfolio_backtester import run_portfolio_backtest, PortfolioConfig
        configs = _load_alpha_configs()
        if not configs:
            return None
        pairs = []
        for ac in configs:
            pair = ac["pair"].split("/")
            params = ac.get("params", {})
            pairs.append({
                "sym_x": pair[0], "sym_y": pair[1],
                "z_open": params.get("z_open", 2.0),
                "z_close": params.get("z_close", 0.5),
                "stop_z": params.get("stop_z", 4.0),
                "lookback": int(params.get("lookback", 60)),
            })
        config = PortfolioConfig(
            initial_capital=1_000_000.0,
            vol_target=0.10,
            kelly_fraction=0.25,
            max_position_weight=0.10,
        )
        result = run_portfolio_backtest(pairs, config)
        return result
    except Exception as e:
        logger.warning("Portfolio backtest failed: %s", e)
        return None


def render_alpha_dashboard(app_ctx: Any = None, feature_flags: dict = None, nav_payload: dict = None) -> None:
    """Main render function for the Alpha Dashboard tab."""

    # ── Header ────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0D1B2A 0%,#1B2838 60%,#243B55 100%);
        border-radius:12px;padding:20px 28px;margin-bottom:20px;">
        <div style="font-size:1.4rem;font-weight:800;color:white;">
            📊 Alpha Performance Dashboard
        </div>
        <div style="font-size:0.85rem;color:rgba(255,255,255,0.7);margin-top:4px;">
            Portfolio analytics · Risk metrics · Trade blotter · Walk-forward validation
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────
    alpha_configs = _load_alpha_configs()
    equity = _load_equity_curve()

    if not alpha_configs:
        st.warning("No alpha pair configurations found. Run the alpha pipeline first:")
        st.code("python scripts/run_full_alpha.py --universe all --trials 30")
        return

    # ── KPI Cards ─────────────────────────────────────────────────
    st.markdown("### 📈 Key Performance Indicators")

    # Try to run fresh backtest or use cached
    result = None
    if "portfolio_result" in st.session_state:
        result = st.session_state["portfolio_result"]

    if result is None:
        with st.spinner("Running portfolio backtest..."):
            result = _run_portfolio_backtest()
            if result:
                st.session_state["portfolio_result"] = result

    if result and hasattr(result, "sharpe"):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Sharpe Ratio", f"{result.sharpe:.2f}",
                      delta=f"+{result.sharpe - 0.69:.2f} vs SPY" if result.sharpe > 0.69 else None)
        with c2:
            st.metric("CAGR", f"{result.cagr:.1f}%")
        with c3:
            st.metric("Max Drawdown", f"{result.max_drawdown:.1f}%")
        with c4:
            st.metric("Sortino", f"{result.sortino:.2f}")
        with c5:
            st.metric("Win Rate", f"{result.win_rate:.0f}%")
        with c6:
            st.metric("Trades", f"{result.n_trades:,}")

        # ── Equity Curve ──────────────────────────────────────────
        st.markdown("### 📈 Portfolio Equity Curve")
        if result.equity_curve is not None and len(result.equity_curve) > 0:
            eq = result.equity_curve

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                mode="lines", name="Portfolio",
                line=dict(color="#2196F3", width=2),
                fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
            ))

            # Add SPY benchmark
            try:
                from common.data_loader import load_price_data
                spy = load_price_data("SPY")["close"]
                spy_range = spy[(spy.index >= eq.index[0]) & (spy.index <= eq.index[-1])]
                if len(spy_range) > 0:
                    spy_normalized = spy_range / spy_range.iloc[0] * eq.iloc[0]
                    fig.add_trace(go.Scatter(
                        x=spy_normalized.index, y=spy_normalized.values,
                        mode="lines", name="SPY (benchmark)",
                        line=dict(color="#FF9800", width=1.5, dash="dash"),
                    ))
            except Exception:
                pass

            fig.update_layout(
                height=400,
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                yaxis_tickformat="$,.0f",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Drawdown Chart ────────────────────────────────────────
        if result.drawdown_series is not None and len(result.drawdown_series) > 0:
            st.markdown("### 📉 Drawdown")
            dd = result.drawdown_series * 100

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                mode="lines", name="Drawdown",
                line=dict(color="#F44336", width=1.5),
                fill="tozeroy", fillcolor="rgba(244,67,54,0.15)",
            ))
            fig_dd.update_layout(
                height=250,
                margin=dict(l=40, r=20, t=10, b=40),
                yaxis_title="Drawdown (%)",
                yaxis_tickformat=".1f",
                template="plotly_white",
            )
            st.plotly_chart(fig_dd, use_container_width=True)

    # ── Alpha Pair Heatmap ────────────────────────────────────────
    st.markdown("### 🔥 Alpha Pair Performance")

    df_alpha = pd.DataFrame(alpha_configs)
    if not df_alpha.empty and "sharpe" in df_alpha.columns:
        df_alpha = df_alpha.sort_values("sharpe", ascending=False)

        fig_hm = go.Figure(data=go.Bar(
            x=df_alpha["pair"],
            y=df_alpha["sharpe"],
            marker_color=[
                "#4CAF50" if s > 0.7 else "#FFC107" if s > 0.3 else "#F44336"
                for s in df_alpha["sharpe"]
            ],
            text=[f"{s:.2f}" for s in df_alpha["sharpe"]],
            textposition="auto",
        ))
        fig_hm.update_layout(
            height=350,
            margin=dict(l=40, r=20, t=10, b=60),
            xaxis_title="Pair",
            yaxis_title="Sharpe Ratio",
            xaxis_tickangle=-45,
            template="plotly_white",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Table
        st.markdown("### 📋 Pair Details")
        display_cols = ["pair", "sharpe", "return", "max_dd", "win_rate", "n_trades"]
        available = [c for c in display_cols if c in df_alpha.columns]
        if available:
            styled = df_alpha[available].reset_index(drop=True)
            styled.index = styled.index + 1
            styled.columns = [c.replace("_", " ").title() for c in available]
            st.dataframe(styled, use_container_width=True, height=min(400, 35 * len(styled) + 40))

    # ── Trade Blotter ─────────────────────────────────────────────
    st.markdown("### 📜 Recent Trades")
    trade_log = _load_trade_log()
    if not trade_log.empty:
        recent = trade_log.tail(20).iloc[::-1]  # Most recent first
        st.dataframe(recent, use_container_width=True, height=400)
    elif result and result.trade_log:
        recent = pd.DataFrame(result.trade_log[-20:]).iloc[::-1]
        st.dataframe(recent, use_container_width=True, height=400)
    else:
        st.info("No trade log available. Run a backtest first.")

    # ── Portfolio Composition ─────────────────────────────────────
    if result and result.weights_history is not None and len(result.weights_history) > 0:
        st.markdown("### 🏗️ Portfolio Composition Over Time")
        wh = result.weights_history
        active_cols = [c for c in wh.columns if wh[c].abs().sum() > 0]
        if active_cols:
            fig_comp = go.Figure()
            for col in active_cols[:10]:  # Top 10 pairs
                fig_comp.add_trace(go.Scatter(
                    x=wh.index, y=wh[col] * 100,
                    mode="lines", name=col,
                    stackgroup="one",
                ))
            fig_comp.update_layout(
                height=350,
                margin=dict(l=40, r=20, t=10, b=40),
                yaxis_title="Weight (%)",
                template="plotly_white",
                legend=dict(font=dict(size=9)),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # ── Risk Metrics ──────────────────────────────────────────────
    if result and hasattr(result, "annual_vol"):
        st.markdown("### 🛡️ Risk Summary")
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("Annual Volatility", f"{result.annual_vol:.1f}%",
                      delta="On target" if 8 < result.annual_vol < 12 else "Off target")
        with rc2:
            st.metric("Calmar Ratio", f"{result.calmar:.2f}")
        with rc3:
            st.metric("Avg Pairs Active", f"{result.avg_pairs_active:.1f}")
        with rc4:
            st.metric("Total Costs", f"${result.total_costs:,.0f}")

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(f"Data as of {date.today()} · {len(alpha_configs)} alpha pairs · "
               f"Run `python scripts/run_full_alpha.py` to refresh")
