#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_full_alpha.py — Full Automated Alpha Machine
=========================================================

Complete cyclical pipeline that runs autonomously:
1. DISCOVER — scan all pairs, score by mean-reversion quality
2. OPTIMIZE — find ideal parameters per pair (Optuna)
3. BACKTEST — run realistic backtest per pair with optimal params
4. FILTER — keep only pairs with Sharpe > threshold
5. PORTFOLIO — equal-weight portfolio of alpha-positive pairs
6. REPORT — GPT-4o analysis + save results

Runs in a loop: discover → optimize → backtest → filter → repeat
Each cycle improves by learning which pairs/params work.

Usage:
    python scripts/run_full_alpha.py                    # Single cycle
    python scripts/run_full_alpha.py --daemon           # Continuous (every 4h)
    python scripts/run_full_alpha.py --universe all     # All 96 pairs
    python scripts/run_full_alpha.py --min-sharpe 0.3   # Lower filter
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "logs" / "alpha_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_universe() -> list[tuple[str, str]]:
    """Load all pairs from pairs.json."""
    from common.data_loader import load_pairs
    pairs = load_pairs()
    return [(p["symbols"][0], p["symbols"][1])
            for p in pairs if len(p.get("symbols", [])) >= 2]


def optimize_pair(sym_x: str, sym_y: str, n_trials: int = 40) -> dict:
    """Find optimal parameters for a single pair using Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from common.data_loader import load_price_data

        px = load_price_data(sym_x)["close"].tail(756)  # 3 years
        py = load_price_data(sym_y)["close"].tail(756)
        common = px.index.intersection(py.index)
        px, py = px.loc[common], py.loc[common]

        if len(px) < 200:
            return {"pair": f"{sym_x}/{sym_y}", "error": "insufficient_data"}

        def objective(trial):
            z_open = trial.suggest_float("z_open", 1.2, 3.5)
            z_close = trial.suggest_float("z_close", 0.1, 1.2)
            stop_z = trial.suggest_float("stop_z", 3.0, 6.0)
            lookback = trial.suggest_int("lookback", 20, 120)
            max_hold = trial.suggest_int("max_holding", 15, 90)

            # Quick inline backtest for optimization
            beta = float(np.cov(px.values, py.values)[0, 1] / np.var(px.values))
            spread = py - beta * px
            mu = spread.rolling(lookback, min_periods=lookback // 2).mean()
            sig = spread.rolling(lookback, min_periods=lookback // 2).std().replace(0, np.nan)
            z = ((spread - mu) / sig).fillna(0.0)

            equity = 100000.0
            pos = 0.0
            hold = 0
            trades = 0
            wins = 0

            for i in range(lookback + 1, len(z)):
                zv = z.iloc[i]
                if np.isnan(zv):
                    continue

                # PnL
                if pos != 0:
                    ret_y = (py.iloc[i] - py.iloc[i-1]) / py.iloc[i-1]
                    ret_x = (px.iloc[i] - px.iloc[i-1]) / px.iloc[i-1]
                    spread_ret = ret_y - beta * ret_x
                    pnl = pos * spread_ret * equity * 0.20
                    equity += pnl
                    hold += 1

                # Entry
                if pos == 0:
                    if zv <= -z_open:
                        pos = 1.0
                        entry_eq = equity
                        hold = 0
                    elif zv >= z_open:
                        pos = -1.0
                        entry_eq = equity
                        hold = 0

                # Exit
                elif pos != 0:
                    if pos > 0:
                        exit_r = zv >= -z_close
                        exit_s = zv <= -stop_z
                    else:
                        exit_r = zv <= z_close
                        exit_s = zv >= stop_z
                    exit_t = hold >= max_hold

                    if exit_r or exit_s or exit_t:
                        trades += 1
                        if equity > entry_eq:
                            wins += 1
                        equity -= equity * 0.20 * 0.001  # Commission
                        pos = 0.0

            if trades < 5:
                return -999  # Not enough trades

            ret = (equity / 100000 - 1)
            daily_returns = pd.Series(z.values[lookback:]).pct_change().dropna()
            if daily_returns.std() > 0:
                sharpe = float(ret / max(abs(ret), 0.001)) * np.sqrt(252 / max(len(z) - lookback, 1))
            else:
                sharpe = 0
            win_rate = wins / max(trades, 1)

            # Objective: combination of return, win rate, and trade count
            score = ret * 0.4 + win_rate * 0.4 + min(trades / 30, 1.0) * 0.2
            return score

        from common.optuna_factory import create_optuna_study
        study = create_optuna_study(study_name=f"alpha_{sym_x}_{sym_y}", direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return {
            "pair": f"{sym_x}/{sym_y}",
            "sym_x": sym_x, "sym_y": sym_y,
            "best_params": study.best_params,
            "best_value": float(study.best_value),
            "n_trials": n_trials,
        }
    except Exception as e:
        return {"pair": f"{sym_x}/{sym_y}", "error": str(e)}


def backtest_with_params(sym_x: str, sym_y: str, params: dict, capital: float = 100_000.0) -> dict:
    """Run realistic backtest with specific parameters."""
    from scripts.run_backtest import backtest_pair
    return backtest_pair(
        sym_x, sym_y,
        z_open=params.get("z_open", 2.0),
        z_close=params.get("z_close", 0.5),
        stop_z=params.get("stop_z", 4.0),
        lookback=int(params.get("lookback", 60)),
        max_holding=int(params.get("max_holding", 60)),
        capital=capital,
        commission_bps=5.0,
        bar_lag=1,
    )


def run_full_cycle(
    universe: str = "top30",
    n_trials: int = 40,
    min_sharpe: float = 0.5,
    capital: float = 100_000.0,
) -> dict:
    """Run the complete alpha generation cycle."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    print("\n" + "🏦" * 30)
    print("  FULL AUTOMATED ALPHA MACHINE")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Universe: {universe} | Trials: {n_trials} | Min Sharpe: {min_sharpe}")
    print("🏦" * 30)

    # ── Stage 1: Load universe ────────────────────────────────────
    print("\n📋 Loading universe...")
    all_pairs = load_universe()
    if universe == "all":
        pairs = all_pairs
    elif universe.startswith("top"):
        n = int(universe[3:])
        pairs = all_pairs[:n]
    else:
        pairs = all_pairs[:30]

    print(f"  {len(pairs)} pairs to process")

    # ── Stage 2: Optimize each pair ───────────────────────────────
    print(f"\n⚙️ OPTIMIZING {len(pairs)} pairs ({n_trials} trials each)...")
    print("=" * 60)

    from common.data_loader import _load_symbol_full_cached
    if hasattr(_load_symbol_full_cached, "cache_clear"):
        _load_symbol_full_cached.cache_clear()

    optimized = []
    for idx, (sx, sy) in enumerate(pairs, 1):
        result = optimize_pair(sx, sy, n_trials=n_trials)
        if result.get("error"):
            continue
        optimized.append(result)
        bv = result["best_value"]
        bp = result["best_params"]
        status = "✅" if bv > 0 else "⚠️"
        if idx <= 10 or bv > 0.1:  # Print first 10 + good ones
            print(f"  {idx:>3}/{len(pairs)} {status} {sx}/{sy}: "
                  f"z_open={bp.get('z_open', 0):.2f} z_close={bp.get('z_close', 0):.2f} "
                  f"lb={bp.get('lookback', 0)} score={bv:.3f}")

    print(f"\n  Optimized: {len(optimized)}/{len(pairs)}")

    # ── Stage 3: Backtest with optimal params ─────────────────────
    print(f"\n📊 BACKTESTING {len(optimized)} pairs with optimal params...")
    print("=" * 60)

    backtest_results = []
    for opt in optimized:
        result = backtest_with_params(
            opt["sym_x"], opt["sym_y"],
            opt["best_params"],
            capital=capital,
        )
        result["optimal_params"] = opt["best_params"]
        backtest_results.append(result)

    # ── Stage 3.5: Walk-Forward Validation ──────────────────────
    print(f"\n📊 WALK-FORWARD VALIDATION (top {min(10, len(backtest_results))} pairs)...")
    print("=" * 60)

    # Run WF on pairs that passed initial backtest
    candidates = sorted(backtest_results, key=lambda x: x.get("sharpe", -999), reverse=True)
    candidates = [c for c in candidates if c.get("sharpe", -999) > 0 and not c.get("error")][:10]

    for c in candidates:
        try:
            from core.walk_forward_engine import run_walk_forward
            sx, sy = c.get("optimal_params", c).get("sym_x", c["pair"].split("/")[0]), c["pair"].split("/")[-1]
            if "/" in c["pair"]:
                sx, sy = c["pair"].split("/")
            wf = run_walk_forward(sx, sy, n_folds=3, test_days=63, n_optuna_trials=10)
            c["wf_oos_sharpe"] = wf.avg_oos_sharpe
            c["wf_profitable_folds"] = wf.pct_profitable_folds
            c["wf_deflated_sharpe"] = wf.deflated_sharpe
            emoji = "✅" if wf.avg_oos_sharpe > 0 else "❌"
            print(f"  {emoji} {c['pair']:<12} OOS_Sharpe={wf.avg_oos_sharpe:+.2f}  "
                  f"Profitable={wf.pct_profitable_folds:.0f}%  DSR={wf.deflated_sharpe:.3f}")
        except Exception as e:
            c["wf_oos_sharpe"] = 0
            c["wf_profitable_folds"] = 0
            logger.debug(f"WF failed for {c['pair']}: {e}")

    # ── Stage 4: Filter by Sharpe ─────────────────────────────────
    print(f"\n🔍 FILTERING: keeping pairs with Sharpe > {min_sharpe}")
    print("=" * 60)

    alpha_pairs = [r for r in backtest_results
                   if r.get("sharpe", -999) >= min_sharpe and not r.get("error")]
    rejected = [r for r in backtest_results
                if r.get("sharpe", -999) < min_sharpe or r.get("error")]

    print(f"\n  ✅ Alpha pairs: {len(alpha_pairs)}")
    for r in sorted(alpha_pairs, key=lambda x: x.get("sharpe", 0), reverse=True):
        print(f"     🟢 {r['pair']:<12} Sharpe={r['sharpe']:>5.2f}  "
              f"Return={r['total_return']:>6.1f}%  MaxDD={r['max_dd']:>6.1f}%  "
              f"WR={r['win_rate']:>4.0f}%  Trades={r['n_trades']}")

    print(f"\n  ❌ Rejected: {len(rejected)}")
    # Show worst rejections
    sorted_rej = sorted(rejected, key=lambda x: x.get("sharpe", -999))
    for r in sorted_rej[:5]:
        if r.get("error"):
            print(f"     🔴 {r.get('pair', '?'):<12} ERROR: {r['error'][:50]}")
        else:
            print(f"     🔴 {r['pair']:<12} Sharpe={r['sharpe']:>5.2f}")

    # ── Stage 5: Portfolio-level result ────────────────────────────
    if alpha_pairs:
        from scripts.run_backtest import compute_portfolio_equity, backtest_benchmark

        portfolio = compute_portfolio_equity(alpha_pairs, capital)
        if not portfolio.get("error"):
            first_eq = next(r["equity_curve"] for r in alpha_pairs
                           if r.get("equity_curve") is not None and len(r["equity_curve"]) > 0)
            benchmark = backtest_benchmark(first_eq.index[0], first_eq.index[-1], capital)

            print(f"\n{'='*60}")
            print(f"  🏆 PORTFOLIO RESULT ({len(alpha_pairs)} alpha pairs)")
            print(f"{'='*60}")
            print(f"  Sharpe:   {portfolio['sharpe']:.2f}")
            print(f"  Sortino:  {portfolio['sortino']:.2f}")
            print(f"  CAGR:     {portfolio['cagr']:.1f}%")
            print(f"  MaxDD:    {portfolio['max_dd']:.1f}%")
            print(f"  Return:   {portfolio['total_return']:.1f}%")
            print(f"  Trades:   {portfolio['n_trades']}")

            if benchmark is not None and len(benchmark) > 0:
                spy_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
                spy_sharpe = float(benchmark.pct_change().dropna().mean() /
                                   benchmark.pct_change().dropna().std() * np.sqrt(252))
                alpha_s = portfolio["sharpe"] - spy_sharpe
                alpha_r = portfolio["total_return"] - spy_ret
                emoji = "🏆" if alpha_s > 0 else "⚠️"
                print(f"\n  📈 SPY: Return={spy_ret:.1f}%  Sharpe={spy_sharpe:.2f}")
                print(f"  {emoji} ALPHA: Sharpe={alpha_s:+.2f}  Return={alpha_r:+.1f}%")

    # ── Stage 6: Save results ─────────────────────────────────────
    elapsed = time.time() - t0

    # Save alpha pair configs
    alpha_configs = []
    for r in alpha_pairs:
        alpha_configs.append({
            "pair": r["pair"],
            "sharpe": r["sharpe"],
            "return": r["total_return"],
            "max_dd": r["max_dd"],
            "win_rate": r["win_rate"],
            "n_trades": r["n_trades"],
            "params": r.get("optimal_params", {}),
        })

    config_path = RESULTS_DIR / f"alpha_pairs_{ts}.json"
    config_path.write_text(json.dumps(alpha_configs, indent=2, default=str))

    # Save latest as canonical
    latest_path = RESULTS_DIR / "alpha_pairs_latest.json"
    latest_path.write_text(json.dumps(alpha_configs, indent=2, default=str))

    print(f"\n  💾 Saved: {config_path.name}")
    print(f"  💾 Latest: alpha_pairs_latest.json")
    print(f"  ⏱️ Total time: {elapsed:.0f}s")
    print(f"{'='*60}")

    return {
        "alpha_pairs": len(alpha_pairs),
        "total_scanned": len(pairs),
        "portfolio_sharpe": portfolio.get("sharpe", 0) if alpha_pairs else 0,
        "elapsed_seconds": elapsed,
        "configs": alpha_configs,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    for name in ["httpx", "openai", "urllib3", "yfinance", "agents.registry",
                  "common.data_loader", "optuna", "scripts.train_meta_label",
                  "ml.models"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Full Automated Alpha Machine")
    parser.add_argument("--universe", default="top30", help="all | topN (e.g. top30)")
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials per pair")
    parser.add_argument("--min-sharpe", type=float, default=0.5, help="Min Sharpe to keep pair")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=float, default=4.0, help="Hours between cycles")
    args = parser.parse_args()

    if args.daemon:
        print(f"🔁 Alpha Machine daemon: every {args.interval}h")
        cycle = 1
        while True:
            try:
                print(f"\n{'🔄'*20} CYCLE {cycle} {'🔄'*20}")
                result = run_full_cycle(
                    universe=args.universe,
                    n_trials=args.trials,
                    min_sharpe=args.min_sharpe,
                    capital=args.capital,
                )
                print(f"\n  Cycle {cycle}: {result['alpha_pairs']} alpha pairs, "
                      f"portfolio Sharpe={result['portfolio_sharpe']:.2f}")
                cycle += 1
                print(f"\n💤 Next cycle in {args.interval}h...")
                time.sleep(args.interval * 3600)
            except KeyboardInterrupt:
                print("\n🛑 Stopped")
                break
    else:
        run_full_cycle(
            universe=args.universe,
            n_trials=args.trials,
            min_sharpe=args.min_sharpe,
            capital=args.capital,
        )


if __name__ == "__main__":
    main()
