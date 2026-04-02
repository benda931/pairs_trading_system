#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_backtest.py — Real PnL Backtester
==============================================

Produces actual equity curves, Sharpe ratios, drawdown metrics,
and comparison vs SPY benchmark for validated pairs.

This is the PROOF that the system generates alpha (or not).

Usage:
    python scripts/run_backtest.py                          # All validated pairs
    python scripts/run_backtest.py --pair GDX-GDXJ          # Single pair
    python scripts/run_backtest.py --capital 100000          # Custom capital
    python scripts/run_backtest.py --from-discovery          # Use alpha pipeline results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def backtest_pair(
    sym_x: str,
    sym_y: str,
    *,
    z_open: float = 2.0,
    z_close: float = 0.5,
    stop_z: float = 4.0,
    lookback: int = 60,
    max_holding: int = 60,
    capital: float = 100_000.0,
    commission_bps: float = 5.0,
    bar_lag: int = 1,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    """
    Run a complete backtest for a single pair and return full metrics.

    Returns dict with:
        sharpe, sortino, cagr, max_dd, win_rate, n_trades,
        profit_factor, avg_trade_pnl, total_pnl, total_return,
        equity_curve (pd.Series), trade_log (list[dict])
    """
    from common.data_loader import load_price_data, _load_symbol_full_cached
    if hasattr(_load_symbol_full_cached, "cache_clear"):
        _load_symbol_full_cached.cache_clear()

    # Load data
    px = load_price_data(sym_x)["close"]
    py = load_price_data(sym_y)["close"]
    common = px.index.intersection(py.index)
    px = px.loc[common]
    py = py.loc[common]

    if start_date:
        px = px[px.index >= pd.Timestamp(start_date)]
        py = py[py.index >= pd.Timestamp(start_date)]
    if end_date:
        px = px[px.index <= pd.Timestamp(end_date)]
        py = py[py.index <= pd.Timestamp(end_date)]

    if len(px) < lookback + 50:
        return {"error": f"Insufficient data: {len(px)} rows", "n_trades": 0}

    # Compute spread and z-score
    beta = float(np.cov(px.values, py.values)[0, 1] / np.var(px.values))
    spread = py - beta * px
    mu = spread.rolling(lookback, min_periods=lookback // 2).mean()
    sigma = spread.rolling(lookback, min_periods=lookback // 2).std().replace(0, np.nan)
    z = ((spread - mu) / sigma).fillna(0.0)

    # ── Simulate trading (realistic dollar-neutral sizing) ──────
    #
    # Position sizing: each leg gets notional = capital * risk_per_trade.
    # For a dollar-neutral pair: long leg_Y notional, short leg_X * beta notional.
    # PnL = notional * (spread_return), where spread_return = d(spread)/spread_entry.
    #
    # This prevents inflated returns from micro-spreads (VOO/SPY, BND/AGG).
    risk_per_trade = 0.20  # Risk 20% of equity per trade (realistic for pairs)

    equity = capital
    position = 0.0  # +1 = long spread, -1 = short spread, 0 = flat
    entry_spread = 0.0
    trade_notional = 0.0
    holding_days = 0
    pending_entry = None  # (target_bar_idx, direction)

    equity_series = []
    trade_log = []
    current_trade = None
    daily_pnl = []

    for i in range(lookback, len(z)):
        z_val = z.iloc[i]
        spread_val = spread.iloc[i]
        px_val = px.iloc[i]
        py_val = py.iloc[i]
        date_val = z.index[i]

        if np.isnan(z_val):
            equity_series.append(equity)
            daily_pnl.append(0.0)
            continue

        # Apply pending entry (bar_lag)
        if pending_entry is not None:
            target_idx, direction = pending_entry
            if i >= target_idx:
                position = direction
                entry_spread = spread_val
                # Notional: risk_per_trade fraction of current equity per leg
                trade_notional = equity * risk_per_trade
                holding_days = 0
                # Commission: bps on notional * 4 legs (2 entry, 2 exit)
                cost = trade_notional * 2 * (commission_bps / 10000)
                equity -= cost
                current_trade = {
                    "entry_date": str(date_val.date()),
                    "entry_z": float(z_val),
                    "entry_spread": float(spread_val),
                    "entry_px": float(px_val),
                    "entry_py": float(py_val),
                    "direction": "LONG" if direction > 0 else "SHORT",
                    "notional": round(trade_notional, 2),
                    "entry_equity": equity,
                }
                pending_entry = None

        # Mark to market PnL (realistic: based on return of spread)
        pnl_today = 0.0
        if position != 0 and i > lookback:
            # Return-based PnL: how much did the spread move as % of price
            prev_px = px.iloc[i - 1]
            prev_py = py.iloc[i - 1]
            ret_y = (py_val - prev_py) / prev_py if prev_py != 0 else 0
            ret_x = (px_val - prev_px) / prev_px if prev_px != 0 else 0
            # Spread return for long spread: long Y, short X*beta
            spread_ret = ret_y - beta * ret_x
            pnl_today = position * spread_ret * trade_notional
            equity += pnl_today
            holding_days += 1

        # Generate signals
        if position == 0 and pending_entry is None:
            if z_val <= -z_open:
                if bar_lag <= 0:
                    position = 1.0
                    entry_spread = spread_val
                    trade_notional = equity * risk_per_trade
                    holding_days = 0
                    cost = trade_notional * 2 * (commission_bps / 10000)
                    equity -= cost
                    current_trade = {
                        "entry_date": str(date_val.date()),
                        "entry_z": float(z_val),
                        "entry_spread": float(spread_val),
                        "entry_px": float(px_val),
                        "entry_py": float(py_val),
                        "direction": "LONG",
                        "notional": round(trade_notional, 2),
                        "entry_equity": equity,
                    }
                else:
                    pending_entry = (i + bar_lag, 1.0)
            elif z_val >= z_open:
                if bar_lag <= 0:
                    position = -1.0
                    entry_spread = spread_val
                    trade_notional = equity * risk_per_trade
                    holding_days = 0
                    cost = trade_notional * 2 * (commission_bps / 10000)
                    equity -= cost
                    current_trade = {
                        "entry_date": str(date_val.date()),
                        "entry_z": float(z_val),
                        "entry_spread": float(spread_val),
                        "entry_px": float(px_val),
                        "entry_py": float(py_val),
                        "direction": "SHORT",
                        "notional": round(trade_notional, 2),
                        "entry_equity": equity,
                    }
                else:
                    pending_entry = (i + bar_lag, -1.0)

        elif position != 0:
            # Exit conditions
            if position > 0:
                exit_revert = z_val >= -z_close
                exit_stop = z_val <= -stop_z
            else:
                exit_revert = z_val <= z_close
                exit_stop = z_val >= stop_z

            exit_time = holding_days >= max_holding

            if exit_revert or exit_stop or exit_time:
                # Close trade — exit commission
                cost = trade_notional * 2 * (commission_bps / 10000)
                equity -= cost

                if current_trade:
                    trade_pnl = equity - current_trade["entry_equity"]
                    current_trade.update({
                        "exit_date": str(date_val.date()),
                        "exit_z": float(z_val),
                        "exit_spread": float(spread_val),
                        "holding_days": holding_days,
                        "pnl": round(trade_pnl, 2),
                        "pnl_pct": round(trade_pnl / capital * 100, 2),
                        "exit_reason": "revert" if exit_revert else "stop" if exit_stop else "time",
                    })
                    trade_log.append(current_trade)
                    current_trade = None

                position = 0.0
                entry_spread = 0.0
                holding_days = 0

        equity_series.append(equity)
        daily_pnl.append(pnl_today)

    # ── Compute metrics ──────────────────────────────────────────
    eq = pd.Series(equity_series, index=z.index[lookback: lookback + len(equity_series)])
    returns = eq.pct_change().dropna()

    total_pnl = equity - capital
    total_return = total_pnl / capital
    n_trades = len(trade_log)

    # Sharpe
    if len(returns) > 10 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 5 and downside.std() > 0:
        sortino = float(returns.mean() / downside.std() * np.sqrt(252))
    else:
        sortino = 0.0

    # CAGR
    n_years = len(returns) / 252
    if n_years > 0 and equity > 0:
        cagr = float((equity / capital) ** (1 / n_years) - 1)
    else:
        cagr = 0.0

    # Max drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    # Win rate
    wins = sum(1 for t in trade_log if t.get("pnl", 0) > 0)
    win_rate = wins / n_trades if n_trades > 0 else 0.0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in trade_log if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in trade_log if t.get("pnl", 0) < 0))
    profit_factor = gross_profit / max(gross_loss, 1.0)

    # Avg trade PnL
    avg_trade_pnl = total_pnl / max(n_trades, 1)

    return {
        "pair": f"{sym_x}/{sym_y}",
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "total_return": round(total_return * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "capital": capital,
        "params": {"z_open": z_open, "z_close": z_close, "stop_z": stop_z,
                   "lookback": lookback, "bar_lag": bar_lag},
        "equity_curve": eq,
        "trade_log": trade_log,
    }


def backtest_benchmark(start_idx, end_idx, capital: float = 100_000.0) -> pd.Series:
    """Buy-and-hold SPY benchmark equity curve."""
    from common.data_loader import load_price_data
    spy = load_price_data("SPY")["close"]
    spy = spy[(spy.index >= start_idx) & (spy.index <= end_idx)]
    if spy.empty:
        return pd.Series(dtype=float)
    return capital * (spy / spy.iloc[0])


def compute_portfolio_equity(results: list[dict], capital: float = 100_000.0) -> dict:
    """Combine all pair equity curves into a portfolio-level result."""
    curves = {}
    for r in results:
        if "equity_curve" in r and r["equity_curve"] is not None and len(r["equity_curve"]) > 0:
            curves[r["pair"]] = r["equity_curve"]

    if not curves:
        return {"error": "No equity curves"}

    # Equal-weight portfolio: allocate capital / n_pairs to each
    n = len(curves)
    weight = 1.0 / n

    # Combine returns
    all_returns = pd.DataFrame({
        pair: eq.pct_change().fillna(0) for pair, eq in curves.items()
    })
    # Fill missing dates with 0 (pair not trading)
    all_returns = all_returns.fillna(0)

    # Portfolio return = equal-weighted average of pair returns
    port_ret = all_returns.mean(axis=1)
    port_eq = capital * (1 + port_ret).cumprod()

    # Metrics
    total_return = float((port_eq.iloc[-1] / capital - 1) * 100) if len(port_eq) > 0 else 0
    sharpe = float(port_ret.mean() / port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0
    dd = (port_eq - port_eq.cummax()) / port_eq.cummax()
    max_dd = float(dd.min() * 100)
    downside = port_ret[port_ret < 0]
    sortino = float(port_ret.mean() / downside.std() * np.sqrt(252)) if len(downside) > 5 and downside.std() > 0 else 0
    n_years = len(port_ret) / 252
    cagr = float((port_eq.iloc[-1] / capital) ** (1 / max(n_years, 0.01)) - 1) * 100 if port_eq.iloc[-1] > 0 else 0

    return {
        "pair": f"PORTFOLIO ({n} pairs)",
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "total_return": round(total_return, 2),
        "n_trades": sum(r.get("n_trades", 0) for r in results),
        "win_rate": round(sum(1 for r in results for t in r.get("trade_log", []) if t.get("pnl", 0) > 0)
                         / max(sum(r.get("n_trades", 0) for r in results), 1) * 100, 1),
        "equity_curve": port_eq,
        "n_pairs": n,
    }


def print_results(results: list[dict], benchmark: pd.Series | None = None) -> None:
    """Print formatted backtest results."""
    print("\n" + "=" * 80)
    print("  📊 BACKTEST RESULTS — INDIVIDUAL PAIRS")
    print("=" * 80)

    if not results:
        print("  No results")
        return

    # Header
    print(f"\n  {'Pair':<14} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} "
          f"{'Return':>8} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'AvgPnL':>8}")
    print("  " + "-" * 90)

    for r in sorted(results, key=lambda x: x.get("sharpe", 0), reverse=True):
        if r.get("error"):
            print(f"  {r.get('pair', '?'):<14} ERROR: {r['error']}")
            continue

        sharpe_color = "🟢" if r["sharpe"] > 0.5 else "🟡" if r["sharpe"] > 0 else "🔴"
        print(f"  {sharpe_color} {r['pair']:<12} {r['sharpe']:>6.2f}  {r['sortino']:>7.2f}  "
              f"{r['cagr']:>6.1f}%  {r['max_dd']:>6.1f}%  {r['total_return']:>6.1f}%  "
              f"{r['n_trades']:>6}  {r['win_rate']:>6.0f}%  {r['profit_factor']:>5.1f}  "
              f"${r['avg_trade_pnl']:>7.0f}")

    # Benchmark comparison
    if benchmark is not None and len(benchmark) > 0:
        spy_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
        spy_sharpe = float(benchmark.pct_change().dropna().mean() / benchmark.pct_change().dropna().std() * np.sqrt(252))
        spy_dd = float(((benchmark - benchmark.cummax()) / benchmark.cummax()).min() * 100)
        print(f"\n  📈 SPY Benchmark: Return={spy_ret:.1f}%  Sharpe={spy_sharpe:.2f}  MaxDD={spy_dd:.1f}%")

        # Alpha
        best = max(results, key=lambda x: x.get("sharpe", -999))
        if not best.get("error") and best["sharpe"] > spy_sharpe:
            alpha = best["total_return"] - spy_ret
            print(f"  🏆 ALPHA: {best['pair']} outperforms SPY by {alpha:+.1f}% return, "
                  f"{best['sharpe'] - spy_sharpe:+.2f} Sharpe")
        else:
            print(f"  ⚠️ No pair beat SPY Sharpe ({spy_sharpe:.2f})")

    # Trade log summary
    all_trades = []
    for r in results:
        all_trades.extend(r.get("trade_log", []))
    if all_trades:
        print(f"\n  📋 Total trades: {len(all_trades)}")
        wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
        print(f"  📋 Overall win rate: {wins}/{len(all_trades)} ({wins/len(all_trades)*100:.0f}%)")
        print(f"  📋 Total PnL: ${sum(t.get('pnl', 0) for t in all_trades):,.0f}")

    print("\n" + "=" * 80)


def save_results(results: list[dict], benchmark: pd.Series | None = None) -> None:
    """Save equity curves and trade logs."""
    out_dir = PROJECT_ROOT / "logs" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save equity curves
    curves = {}
    for r in results:
        if "equity_curve" in r and r["equity_curve"] is not None:
            curves[r["pair"]] = r["equity_curve"]
    if benchmark is not None:
        curves["SPY_benchmark"] = benchmark

    if curves:
        eq_df = pd.DataFrame(curves)
        eq_path = out_dir / f"equity_curves_{ts}.csv"
        eq_df.to_csv(eq_path)
        print(f"  💾 Equity curves: {eq_path.name}")

    # Save trade log
    all_trades = []
    for r in results:
        for t in r.get("trade_log", []):
            t["pair"] = r.get("pair", "?")
            all_trades.append(t)
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_path = out_dir / f"trade_log_{ts}.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"  💾 Trade log: {trades_path.name}")

    # Save summary JSON
    summary = [{k: v for k, v in r.items() if k not in ("equity_curve", "trade_log")}
               for r in results]
    summary_path = out_dir / f"backtest_summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  💾 Summary: {summary_path.name}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    for name in ["httpx", "openai", "urllib3", "yfinance", "agents.registry", "common.data_loader"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Real PnL Backtester")
    parser.add_argument("--pair", type=str, help="Single pair (e.g., GDX-GDXJ)")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--z-open", type=float, default=2.0)
    parser.add_argument("--z-close", type=float, default=0.5)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--from-discovery", action="store_true",
                        help="Use validated pairs from alpha pipeline")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Backtest all 96 pairs in universe")
    args = parser.parse_args()

    print("\n" + "🏦" * 30)
    print("  PAIRS TRADING BACKTESTER")
    print("🏦" * 30)

    if args.pair:
        pairs = [tuple(args.pair.split("-", 1))]
    elif args.from_discovery:
        csv_path = PROJECT_ROOT / "logs" / "alpha_discovery.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0)
            if "validated" in df.columns:
                df = df[df["validated"] == True]
            pairs = [(row["sym_x"], row["sym_y"]) for _, row in df.head(10).iterrows()
                     if "sym_x" in row and "sym_y" in row]
        else:
            print("  ⚠️ No discovery results found. Run alpha pipeline first.")
            pairs = [("GDX", "GDXJ"), ("DIA", "SPY"), ("LQD", "IGIB")]
    elif args.all_pairs:
        from common.data_loader import load_pairs
        raw = load_pairs()
        pairs = [(p["symbols"][0], p["symbols"][1]) for p in raw if len(p.get("symbols", [])) >= 2]
    else:
        # Default: validated pairs + some known good ones
        pairs = [
            ("GDX", "GDXJ"), ("DIA", "SPY"), ("LQD", "IGIB"),
            ("XLY", "XLC"), ("XLY", "XLP"), ("XLI", "XLB"),
            ("XLK", "QQQ"), ("BND", "AGG"), ("HYG", "JNK"),
            ("VOO", "SPY"),
        ]

    print(f"\n  Backtesting {len(pairs)} pairs with ${args.capital:,.0f} capital")
    print(f"  Parameters: z_open={args.z_open}, z_close={args.z_close}, lookback={args.lookback}")

    results = []
    for sym_x, sym_y in pairs:
        try:
            r = backtest_pair(
                sym_x, sym_y,
                z_open=args.z_open,
                z_close=args.z_close,
                lookback=args.lookback,
                capital=args.capital,
                bar_lag=1,
            )
            results.append(r)
        except Exception as e:
            results.append({"pair": f"{sym_x}/{sym_y}", "error": str(e), "n_trades": 0})

    # Get benchmark
    if results and any("equity_curve" in r for r in results):
        first_eq = next(r["equity_curve"] for r in results if "equity_curve" in r and r["equity_curve"] is not None)
        benchmark = backtest_benchmark(first_eq.index[0], first_eq.index[-1], args.capital)
    else:
        benchmark = None

    print_results(results, benchmark)

    # Portfolio-level result
    portfolio = compute_portfolio_equity(results, args.capital)
    if not portfolio.get("error"):
        print("\n" + "=" * 80)
        print("  📊 PORTFOLIO-LEVEL RESULT (equal-weight)")
        print("=" * 80)
        print(f"\n  Sharpe:  {portfolio['sharpe']:.2f}")
        print(f"  Sortino: {portfolio['sortino']:.2f}")
        print(f"  CAGR:    {portfolio['cagr']:.1f}%")
        print(f"  MaxDD:   {portfolio['max_dd']:.1f}%")
        print(f"  Return:  {portfolio['total_return']:.1f}%")
        print(f"  Pairs:   {portfolio['n_pairs']}")
        print(f"  Trades:  {portfolio['n_trades']}")

        if benchmark is not None and len(benchmark) > 0:
            spy_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
            spy_sharpe = float(benchmark.pct_change().dropna().mean() / benchmark.pct_change().dropna().std() * np.sqrt(252))
            alpha_ret = portfolio["total_return"] - spy_ret
            alpha_sharpe = portfolio["sharpe"] - spy_sharpe
            emoji = "🏆" if alpha_sharpe > 0 else "⚠️"
            print(f"\n  {emoji} vs SPY: Return alpha={alpha_ret:+.1f}%  Sharpe alpha={alpha_sharpe:+.2f}")

        results.append(portfolio)  # Include in save

    save_results(results, benchmark)


if __name__ == "__main__":
    main()
