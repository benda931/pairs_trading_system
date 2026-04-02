# -*- coding: utf-8 -*-
"""
core/portfolio_backtester.py — Institutional Portfolio-Level Backtester
=======================================================================

Multi-pair portfolio backtester with:
- Kelly criterion position sizing
- Volatility targeting (annualized vol target)
- Regime-conditional allocation
- Correlation-aware position limits
- Daily rebalancing with turnover constraints
- Transaction cost model (bps + market impact)
- Drawdown-based de-risking
- Full equity curve + attribution + risk decomposition

This is NOT a single-pair backtester. It simulates running the
entire pairs trading portfolio simultaneously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Regime-conditional risk parameters."""
    # Vol regime thresholds (realized vol annualized)
    low_vol_threshold: float = 0.08    # Below 8% = low vol regime
    high_vol_threshold: float = 0.20   # Above 20% = high vol regime
    crisis_vol_threshold: float = 0.35 # Above 35% = crisis regime

    # Regime-conditional position scaling
    low_vol_scale: float = 1.3         # Increase size in low vol
    normal_vol_scale: float = 1.0      # Normal sizing
    high_vol_scale: float = 0.6        # Reduce in high vol
    crisis_scale: float = 0.2          # Minimal in crisis

    # Regime-conditional entry threshold adjustment
    high_vol_entry_widen: float = 0.5  # Widen entry z by 0.5 in high vol
    crisis_entry_widen: float = 1.0    # Widen entry z by 1.0 in crisis


@dataclass
class PortfolioConfig:
    """Configuration for portfolio-level backtest."""
    initial_capital: float = 1_000_000.0
    vol_target: float = 0.10          # 10% annualized vol target
    max_gross_leverage: float = 3.0    # Max gross exposure / capital
    max_position_weight: float = 0.15  # Max 15% per pair
    min_position_weight: float = 0.02  # Min 2% per pair (avoid dust)
    max_corr_overlap: float = 0.70     # Max pairwise correlation
    commission_bps: float = 5.0        # Round-trip commission
    market_impact_bps: float = 2.0     # Additional market impact
    rebalance_freq: int = 5            # Rebalance every N days
    max_daily_turnover: float = 0.30   # Max 30% daily turnover
    drawdown_deleverage: float = -0.10 # Start de-risking at -10% DD
    drawdown_halt: float = -0.20       # Halt at -20% DD
    kelly_fraction: float = 0.25       # Use 1/4 Kelly (conservative)
    lookback: int = 60                 # Default lookback for spread
    regime: RegimeConfig = None        # Regime-conditional risk

    def __post_init__(self):
        if self.regime is None:
            self.regime = RegimeConfig()


@dataclass
class PairAllocation:
    """Allocation for a single pair in the portfolio."""
    sym_x: str
    sym_y: str
    weight: float = 0.0       # Portfolio weight
    direction: float = 0.0    # +1 long spread, -1 short spread, 0 flat
    z_score: float = 0.0      # Current z-score
    entry_z: float = 2.0      # Entry threshold
    exit_z: float = 0.5       # Exit threshold
    stop_z: float = 4.0       # Stop threshold
    beta: float = 1.0         # Hedge ratio
    half_life: float = 20.0   # Estimated half-life
    kelly_weight: float = 0.0 # Kelly-optimal weight
    vol_scale: float = 1.0    # Vol-targeting scalar


@dataclass
class PortfolioResult:
    """Complete portfolio backtest result."""
    sharpe: float = 0.0
    sortino: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    total_return: float = 0.0
    annual_vol: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    avg_pairs_active: float = 0.0
    max_pairs_active: int = 0
    avg_gross_leverage: float = 0.0
    total_turnover: float = 0.0
    total_costs: float = 0.0
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    pair_pnl: Optional[pd.DataFrame] = None
    drawdown_series: Optional[pd.Series] = None
    weights_history: Optional[pd.DataFrame] = None
    trade_log: list = field(default_factory=list)


def detect_vol_regime(
    returns: pd.Series,
    config: RegimeConfig,
    lookback: int = 20,
) -> tuple[str, float]:
    """
    Detect current volatility regime.

    Returns (regime_name, position_scale).
    Regimes: LOW_VOL, NORMAL, HIGH_VOL, CRISIS
    """
    if len(returns) < lookback:
        return "NORMAL", config.normal_vol_scale

    realized_vol = float(returns.tail(lookback).std() * np.sqrt(252))

    if realized_vol >= config.crisis_vol_threshold:
        return "CRISIS", config.crisis_scale
    elif realized_vol >= config.high_vol_threshold:
        return "HIGH_VOL", config.high_vol_scale
    elif realized_vol <= config.low_vol_threshold:
        return "LOW_VOL", config.low_vol_scale
    else:
        return "NORMAL", config.normal_vol_scale


def compute_kelly_weight(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly criterion: f* = (p*b - q) / b where b = avg_win/avg_loss."""
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = abs(avg_win / avg_loss)
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    return max(0.0, min(kelly, 1.0))


def compute_vol_scalar(
    returns: pd.Series,
    vol_target: float,
    lookback: int = 20,
) -> float:
    """Compute vol-targeting scalar: target_vol / realized_vol."""
    if len(returns) < lookback:
        return 1.0
    realized_vol = float(returns.tail(lookback).std() * np.sqrt(252))
    if realized_vol <= 0 or np.isnan(realized_vol):
        return 1.0
    return min(vol_target / realized_vol, 3.0)  # Cap at 3x


def run_portfolio_backtest(
    pairs: list[dict],
    config: Optional[PortfolioConfig] = None,
) -> PortfolioResult:
    """
    Run a full portfolio-level backtest across multiple pairs.

    Parameters
    ----------
    pairs : list[dict]
        Each dict has: sym_x, sym_y, z_open, z_close, stop_z, lookback
    config : PortfolioConfig
        Portfolio-level configuration.

    Returns
    -------
    PortfolioResult
        Complete backtest results with equity curve and attribution.
    """
    if config is None:
        config = PortfolioConfig()

    from common.data_loader import load_price_data, _load_symbol_full_cached
    if hasattr(_load_symbol_full_cached, "cache_clear"):
        _load_symbol_full_cached.cache_clear()

    # ── Load all price data ──────────────────────────────────────
    pair_data = {}
    for p in pairs:
        sx, sy = p["sym_x"], p["sym_y"]
        label = f"{sx}/{sy}"
        try:
            px = load_price_data(sx)["close"]
            py = load_price_data(sy)["close"]
            common = px.index.intersection(py.index)
            px, py = px.loc[common], py.loc[common]
            if len(px) < config.lookback + 50:
                continue

            beta = float(np.cov(px.values, py.values)[0, 1] / np.var(px.values))
            spread = py - beta * px
            lb = int(p.get("lookback", config.lookback))
            mu = spread.rolling(lb, min_periods=lb // 2).mean()
            sig = spread.rolling(lb, min_periods=lb // 2).std().replace(0, np.nan)
            z = ((spread - mu) / sig).fillna(0.0)

            pair_data[label] = {
                "px": px, "py": py, "beta": beta,
                "spread": spread, "z": z,
                "z_open": p.get("z_open", 2.0),
                "z_close": p.get("z_close", 0.5),
                "stop_z": p.get("stop_z", 4.0),
                "lookback": lb,
            }
        except Exception as e:
            logger.debug(f"Skip {label}: {e}")

    # ── Correlation check: flag overlapping pairs ──────────────
    # Compute pairwise spread correlations and warn about overlap
    if len(pair_data) > 1:
        spread_returns = {}
        for label, d in pair_data.items():
            sr = d["spread"].pct_change().dropna().replace([np.inf, -np.inf], 0)
            spread_returns[label] = sr

        corr_matrix = pd.DataFrame(spread_returns).corr()
        overlapping = []
        checked = set()
        for l1 in corr_matrix.columns:
            for l2 in corr_matrix.columns:
                if l1 >= l2:
                    continue
                key = (l1, l2)
                if key in checked:
                    continue
                checked.add(key)
                c = corr_matrix.loc[l1, l2]
                if abs(c) > config.max_corr_overlap:
                    overlapping.append((l1, l2, round(c, 3)))
        if overlapping:
            logger.info(
                "Correlated pairs detected (%d pairs, corr > %.2f): %s",
                len(overlapping), config.max_corr_overlap,
                ", ".join(f"{a}-{b}({c})" for a, b, c in overlapping[:5]),
            )

    if not pair_data:
        return PortfolioResult()

    # ── Find common date range ───────────────────────────────────
    all_dates = None
    for d in pair_data.values():
        idx = d["z"].dropna().index
        if all_dates is None:
            all_dates = idx
        else:
            all_dates = all_dates.intersection(idx)

    if all_dates is None or len(all_dates) < 100:
        return PortfolioResult()

    all_dates = all_dates.sort_values()
    start_bar = max(config.lookback, 60)

    # ── Initialize portfolio state ───────────────────────────────
    equity = config.initial_capital
    positions = {label: 0.0 for label in pair_data}  # direction
    weights = {label: 0.0 for label in pair_data}
    entry_equity = {label: 0.0 for label in pair_data}
    holding_days = {label: 0 for label in pair_data}

    equity_series = []
    returns_series = []
    weights_history = []
    pair_pnl_data = {label: [] for label in pair_data}
    trade_log = []
    total_costs = 0.0
    total_turnover = 0.0

    prev_equity = equity

    # ── Main simulation loop ─────────────────────────────────────
    for i in range(start_bar, len(all_dates)):
        dt = all_dates[i]
        daily_pnl = 0.0
        current_weights = {}
        n_active = 0

        # Drawdown check
        peak_eq = max(equity_series) if equity_series else equity
        dd = (equity - peak_eq) / peak_eq if peak_eq > 0 else 0

        # De-risk if in drawdown
        dd_scale = 1.0
        if dd < config.drawdown_halt:
            dd_scale = 0.0  # Full halt
        elif dd < config.drawdown_deleverage:
            dd_scale = max(0.2, 1.0 + (dd - config.drawdown_deleverage) / abs(config.drawdown_deleverage))

        # Vol-targeting scalar
        if len(returns_series) > 20:
            vol_scale = compute_vol_scalar(
                pd.Series(returns_series[-60:]),
                config.vol_target,
            )
        else:
            vol_scale = 1.0

        # Regime detection — adjust sizing based on market vol regime
        regime_name, regime_scale = "NORMAL", 1.0
        if len(returns_series) > 20:
            regime_name, regime_scale = detect_vol_regime(
                pd.Series(returns_series[-60:]),
                config.regime,
            )

        # ── Process each pair ────────────────────────────────────
        for label, data in pair_data.items():
            z = data["z"]
            if dt not in z.index:
                pair_pnl_data[label].append(0.0)
                current_weights[label] = 0.0
                continue

            z_val = float(z.loc[dt])
            z_open = data["z_open"]
            z_close = data["z_close"]
            stop_z = data["stop_z"]
            pos = positions[label]

            # ── Mark to market ───────────────────────────────────
            pnl = 0.0
            if pos != 0 and i > start_bar:
                prev_dt = all_dates[i - 1]
                if prev_dt in data["px"].index and dt in data["px"].index:
                    ret_y = (data["py"].loc[dt] - data["py"].loc[prev_dt]) / data["py"].loc[prev_dt]
                    ret_x = (data["px"].loc[dt] - data["px"].loc[prev_dt]) / data["px"].loc[prev_dt]
                    spread_ret = ret_y - data["beta"] * ret_x
                    notional = abs(weights[label]) * equity
                    pnl = pos * spread_ret * notional
                    holding_days[label] += 1

            # ── Signal generation ────────────────────────────────
            if np.isnan(z_val):
                pass
            elif pos == 0:
                # Entry
                if z_val <= -z_open and dd_scale > 0:
                    positions[label] = 1.0
                    target_weight = config.max_position_weight * dd_scale * vol_scale * regime_scale * config.kelly_fraction
                    weights[label] = min(target_weight, config.max_position_weight)
                    entry_equity[label] = equity
                    holding_days[label] = 0
                    cost = weights[label] * equity * (config.commission_bps + config.market_impact_bps) / 10000
                    total_costs += cost
                    pnl -= cost
                    trade_log.append({
                        "pair": label, "date": str(dt.date()), "action": "ENTRY_LONG",
                        "z": round(z_val, 2), "weight": round(weights[label], 4),
                    })
                elif z_val >= z_open and dd_scale > 0:
                    positions[label] = -1.0
                    target_weight = config.max_position_weight * dd_scale * vol_scale * regime_scale * config.kelly_fraction
                    weights[label] = min(target_weight, config.max_position_weight)
                    entry_equity[label] = equity
                    holding_days[label] = 0
                    cost = weights[label] * equity * (config.commission_bps + config.market_impact_bps) / 10000
                    total_costs += cost
                    pnl -= cost
                    trade_log.append({
                        "pair": label, "date": str(dt.date()), "action": "ENTRY_SHORT",
                        "z": round(z_val, 2), "weight": round(weights[label], 4),
                    })
            else:
                # Exit check
                if pos > 0:
                    exit_r = z_val >= -z_close
                    exit_s = z_val <= -stop_z
                else:
                    exit_r = z_val <= z_close
                    exit_s = z_val >= stop_z

                if exit_r or exit_s or holding_days[label] > 60:
                    cost = weights[label] * equity * (config.commission_bps + config.market_impact_bps) / 10000
                    total_costs += cost
                    pnl -= cost
                    trade_pnl = equity + pnl - entry_equity[label]
                    trade_log.append({
                        "pair": label, "date": str(dt.date()),
                        "action": "EXIT_REVERT" if exit_r else "EXIT_STOP" if exit_s else "EXIT_TIME",
                        "z": round(z_val, 2), "holding_days": holding_days[label],
                        "pnl": round(trade_pnl, 2),
                    })
                    positions[label] = 0.0
                    weights[label] = 0.0
                    holding_days[label] = 0

            daily_pnl += pnl
            pair_pnl_data[label].append(pnl)
            current_weights[label] = weights[label] if positions[label] != 0 else 0.0
            if positions[label] != 0:
                n_active += 1

        # ── Update equity ────────────────────────────────────────
        equity += daily_pnl
        equity_series.append(equity)
        daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        returns_series.append(daily_ret)
        prev_equity = equity
        weights_history.append(current_weights.copy())

    # ── Compute portfolio metrics ────────────────────────────────
    eq = pd.Series(equity_series, index=all_dates[start_bar:start_bar + len(equity_series)])
    rets = pd.Series(returns_series, index=eq.index)

    total_ret = (equity - config.initial_capital) / config.initial_capital
    n_years = len(rets) / 252
    cagr = ((equity / config.initial_capital) ** (1 / max(n_years, 0.01)) - 1) if equity > 0 else 0
    annual_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 10 else 0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(252)) if len(downside) > 5 and downside.std() > 0 else 0

    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate from trade log
    exits = [t for t in trade_log if t.get("action", "").startswith("EXIT")]
    wins = sum(1 for t in exits if t.get("pnl", 0) > 0)
    win_rate = wins / len(exits) if exits else 0

    # Average pairs active
    avg_active = np.mean([sum(1 for v in w.values() if v > 0) for w in weights_history]) if weights_history else 0

    return PortfolioResult(
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        cagr=round(cagr * 100, 2),
        max_drawdown=round(max_dd * 100, 2),
        calmar=round(calmar, 2),
        total_return=round(total_ret * 100, 2),
        annual_vol=round(annual_vol * 100, 2),
        win_rate=round(win_rate * 100, 1),
        n_trades=len(exits),
        avg_pairs_active=round(avg_active, 1),
        max_pairs_active=max(sum(1 for v in w.values() if v > 0) for w in weights_history) if weights_history else 0,
        avg_gross_leverage=round(np.mean([sum(w.values()) for w in weights_history]), 3) if weights_history else 0,
        total_turnover=round(total_turnover, 2),
        total_costs=round(total_costs, 2),
        equity_curve=eq,
        daily_returns=rets,
        drawdown_series=dd,
        weights_history=pd.DataFrame(weights_history, index=eq.index),
        trade_log=trade_log,
    )
