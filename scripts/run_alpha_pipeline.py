#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_alpha_pipeline.py — Full Alpha Generation Pipeline
===============================================================

Complete automated pipeline that discovers, validates, optimizes,
and generates trading signals for pairs trading alpha.

Pipeline stages:
1. DISCOVER — Scan universe for high-quality pair candidates
2. VALIDATE — Statistical validation (cointegration, half-life, stability)
3. OPTIMIZE — Parameter optimization per pair (Optuna)
4. TRAIN — Train ML meta-label models per pair
5. SIGNAL — Generate live signals for validated pairs
6. RANK — Rank signals by expected alpha
7. REPORT — GPT-4o generates comprehensive alpha report

Usage:
    python scripts/run_alpha_pipeline.py                    # Full pipeline
    python scripts/run_alpha_pipeline.py --stage discover   # Discovery only
    python scripts/run_alpha_pipeline.py --stage signals    # Signals only
    python scripts/run_alpha_pipeline.py --top-n 10         # Top 10 pairs
    python scripts/run_alpha_pipeline.py --daemon           # Run every 4 hours
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# STAGE 1: DISCOVER — Scan for pair candidates
# ═══════════════════════════════════════════════════════════════

def stage_discover(top_n: int = 20) -> pd.DataFrame:
    """Scan all pairs and rank by mean-reversion quality."""
    print("\n🔍 STAGE 1: PAIR DISCOVERY")
    print("=" * 60)

    from common.data_loader import load_pairs, load_price_data, _load_symbol_full_cached
    if hasattr(_load_symbol_full_cached, "cache_clear"):
        _load_symbol_full_cached.cache_clear()

    pairs = load_pairs()
    print(f"  Universe: {len(pairs)} pairs")

    results = []
    for p in pairs:
        syms = p.get("symbols", [])
        if len(syms) < 2:
            continue
        try:
            px = load_price_data(syms[0])["close"].tail(504)
            py = load_price_data(syms[1])["close"].tail(504)
            common = px.index.intersection(py.index)
            px, py = px.loc[common], py.loc[common]
            if len(px) < 200:
                continue

            corr = float(px.corr(py))
            beta = float(np.cov(px.values, py.values)[0, 1] / np.var(px.values))
            spread = py - beta * px
            mu = spread.rolling(60, min_periods=30).mean()
            sig = spread.rolling(60, min_periods=30).std().replace(0, np.nan)
            z = ((spread - mu) / sig).fillna(0.0)

            # Half-life via AR(1)
            z_lag = z.shift(1).dropna()
            z_curr = z.iloc[1:]
            if len(z_lag) > 50:
                ar1 = float(z_curr.corr(z_lag))
                half_life = -np.log(2) / np.log(max(abs(ar1), 0.01)) if abs(ar1) < 1 else 999
            else:
                half_life = 999

            # Mean reversion quality
            entries_2 = (z.abs() > 2.0).sum()
            entries_15 = (z.abs() > 1.5).sum()
            reversions = ((z.shift(10).abs() > 1.5) & (z.abs() < 0.5)).sum()
            win_rate = reversions / max(entries_15, 1)

            # Stability: rolling correlation
            rolling_corr = px.rolling(60).corr(py)
            corr_stability = 1.0 - float(rolling_corr.std()) if rolling_corr.std() == rolling_corr.std() else 0

            # Spread volatility ratio (want moderate vol)
            spread_vol = float(spread.std())
            price_vol = float((px.std() + py.std()) / 2)
            vol_ratio = spread_vol / max(price_vol, 0.01)

            # Composite score
            score = (
                0.30 * min(win_rate, 1.0) +
                0.20 * min(corr, 1.0) +
                0.15 * max(0, 1.0 - half_life / 100) +
                0.15 * corr_stability +
                0.10 * min(entries_15 / 50, 1.0) +
                0.10 * min(vol_ratio / 0.05, 1.0)
            )

            results.append({
                "pair": f"{syms[0]}/{syms[1]}",
                "sym_x": syms[0],
                "sym_y": syms[1],
                "score": round(score, 4),
                "corr": round(corr, 3),
                "half_life": round(half_life, 1),
                "win_rate": round(win_rate, 3),
                "entries": int(entries_15),
                "corr_stability": round(corr_stability, 3),
                "z_now": round(float(z.iloc[-1]), 3),
                "vol_ratio": round(vol_ratio, 4),
                "n_obs": len(px),
            })
        except Exception as e:
            logger.debug(f"Skip {syms}: {e}")

    df = pd.DataFrame(results).sort_values("score", ascending=False).head(top_n)
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # 1-based ranking

    print(f"  Analyzed: {len(results)} pairs")
    print(f"  Top {top_n} by composite alpha score:")
    print()
    for _, row in df.iterrows():
        signal = ""
        if abs(row["z_now"]) > 2.0:
            signal = " ← 🔴 SIGNAL!"
        elif abs(row["z_now"]) > 1.5:
            signal = " ← 🟡 WATCH"
        print(f"  {row.name:>3}. {row['pair']:<12} score={row['score']:.3f}  "
              f"corr={row['corr']:.2f}  HL={row['half_life']:>5.0f}d  "
              f"WR={row['win_rate']:.0%}  z={row['z_now']:+.2f}{signal}")

    # Save
    out_path = PROJECT_ROOT / "logs" / "alpha_discovery.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"\n  Saved: {out_path.name}")
    return df


# ═══════════════════════════════════════════════════════════════
# STAGE 2: VALIDATE — Statistical validation
# ═══════════════════════════════════════════════════════════════

def stage_validate(candidates: pd.DataFrame) -> pd.DataFrame:
    """Validate top candidates with rigorous statistical tests."""
    print("\n✅ STAGE 2: STATISTICAL VALIDATION")
    print("=" * 60)

    try:
        from statsmodels.tsa.stattools import adfuller, coint
    except ImportError:
        print("  ⚠️ statsmodels not available — skipping validation")
        candidates["validated"] = True
        return candidates

    from common.data_loader import load_price_data

    validated = []
    for _, row in candidates.iterrows():
        try:
            px = load_price_data(row["sym_x"])["close"].tail(504)
            py = load_price_data(row["sym_y"])["close"].tail(504)
            common = px.index.intersection(py.index)
            px, py = px.loc[common], py.loc[common]

            beta = float(np.cov(px.values, py.values)[0, 1] / np.var(px.values))
            spread = py - beta * px

            # ADF test on spread
            adf_stat, adf_p, *_ = adfuller(spread.dropna(), maxlag=20)

            # Cointegration test
            coint_stat, coint_p, *_ = coint(px, py)

            # Hurst exponent (simplified)
            lags = range(2, min(20, len(spread) // 4))
            tau = [np.sqrt(np.std(np.subtract(spread.values[lag:], spread.values[:-lag])))
                   for lag in lags]
            hurst = float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]) if tau else 0.5

            passed = (adf_p < 0.10) and (coint_p < 0.10) and (hurst < 0.45)
            status = "✅ PASS" if passed else "❌ FAIL"

            print(f"  {row['pair']:<12} ADF_p={adf_p:.3f} COINT_p={coint_p:.3f} "
                  f"Hurst={hurst:.3f} → {status}")

            validated.append({
                **row.to_dict(),
                "adf_p": round(adf_p, 4),
                "coint_p": round(coint_p, 4),
                "hurst": round(hurst, 3),
                "validated": passed,
            })
        except Exception as e:
            logger.debug(f"Validation failed for {row['pair']}: {e}")
            validated.append({**row.to_dict(), "validated": False})

    df = pd.DataFrame(validated)
    passed = df[df["validated"] == True]
    print(f"\n  Passed: {len(passed)}/{len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════
# STAGE 3: OPTIMIZE — Parameter optimization per pair
# ═══════════════════════════════════════════════════════════════

def stage_optimize(validated: pd.DataFrame, n_trials: int = 30) -> pd.DataFrame:
    """Run Optuna optimization for each validated pair."""
    print("\n⚙️ STAGE 3: PARAMETER OPTIMIZATION")
    print("=" * 60)

    from agents.registry import get_default_registry
    from core.contracts import AgentTask, AgentStatus
    registry = get_default_registry()

    optimized = []
    for _, row in validated[validated.get("validated", True) == True].iterrows():
        pair = f"{row['sym_x']}-{row['sym_y']}"
        task = AgentTask(
            task_id=f"opt_{pair}",
            agent_name="auto_parameter_optimizer",
            task_type="auto_optimize_params",
            payload={"pair": pair, "n_trials": n_trials},
        )
        result = registry.dispatch(task)
        if result.status == AgentStatus.COMPLETED and result.output.get("optimized"):
            bp = result.output["best_params"]
            bv = result.output["best_value"]
            print(f"  {pair:<12} z_open={bp.get('z_open', '?'):.2f} "
                  f"z_close={bp.get('z_close', '?'):.2f} "
                  f"lb={bp.get('lookback', '?')} → value={bv:.3f}")
            optimized.append({
                **row.to_dict(),
                "opt_z_open": bp.get("z_open"),
                "opt_z_close": bp.get("z_close"),
                "opt_lookback": bp.get("lookback"),
                "opt_value": bv,
            })
        else:
            optimized.append({**row.to_dict(), "opt_value": 0})

    return pd.DataFrame(optimized)


# ═══════════════════════════════════════════════════════════════
# STAGE 4: SIGNALS — Generate live trading signals
# ═══════════════════════════════════════════════════════════════

def stage_signals(optimized: pd.DataFrame) -> pd.DataFrame:
    """Generate live signals for all validated pairs."""
    print("\n📡 STAGE 4: LIVE SIGNAL GENERATION")
    print("=" * 60)

    from common.data_loader import load_price_data
    from core.signal_pipeline import SignalPipeline
    from core.contracts import PairId

    signals = []
    for _, row in optimized.iterrows():
        try:
            px = load_price_data(row["sym_x"])["close"].tail(252)
            py = load_price_data(row["sym_y"])["close"].tail(252)
            common = px.index.intersection(py.index)
            px, py = px.loc[common], py.loc[common]

            beta = float(np.cov(px.values, py.values)[0, 1] / np.var(px.values))
            spread = py - beta * px
            mu = spread.rolling(60, min_periods=30).mean()
            sig = spread.rolling(60, min_periods=30).std().replace(0, np.nan)
            z_now = float(((spread - mu) / sig).iloc[-1])

            pipeline = SignalPipeline(pair_id=PairId(row["sym_x"], row["sym_y"]))
            decision = pipeline.evaluate_bar(z_score=z_now, current_pos=0.0)

            action_map = {1.0: "LONG_SPREAD", -1.0: "SHORT_SPREAD", 0.0: "FLAT"}
            action = action_map.get(decision.action, "FLAT")
            blocked = decision.blocked

            signal_str = action
            if blocked:
                signal_str = f"BLOCKED ({decision.quality_grade})"

            emoji = {"LONG_SPREAD": "🟢", "SHORT_SPREAD": "🔴", "FLAT": "⚪"}.get(action, "⚪")
            if blocked:
                emoji = "🚫"

            print(f"  {emoji} {row['pair']:<12} z={z_now:+.2f}  action={signal_str}  "
                  f"regime={decision.regime}  quality={decision.quality_grade}")

            signals.append({
                **row.to_dict(),
                "z_live": z_now,
                "action": action,
                "blocked": blocked,
                "regime": decision.regime,
                "quality_grade": decision.quality_grade,
                "entry_z": decision.entry_z,
                "exit_z": decision.exit_z,
            })
        except Exception as e:
            logger.debug(f"Signal gen failed for {row['pair']}: {e}")

    df = pd.DataFrame(signals)
    active = df[df["action"] != "FLAT"]
    print(f"\n  Active signals: {len(active)}/{len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════
# STAGE 5: RANK — Rank by expected alpha
# ═══════════════════════════════════════════════════════════════

def stage_rank(signals: pd.DataFrame) -> pd.DataFrame:
    """Rank active signals by expected alpha."""
    print("\n🏆 STAGE 5: ALPHA RANKING")
    print("=" * 60)

    active = signals[signals["action"] != "FLAT"].copy()
    if active.empty:
        print("  No active signals to rank")
        return signals

    # Alpha score = composite of z-extremity, win rate, score, opt_value
    active["alpha_score"] = (
        0.30 * active["z_live"].abs() / 3.0 +  # Normalized z
        0.25 * active.get("win_rate", 0.0) +
        0.25 * active.get("score", 0.0) +
        0.20 * active.get("opt_value", 0.0)
    )
    active = active.sort_values("alpha_score", ascending=False)

    print(f"  Top signals by expected alpha:")
    for i, (_, row) in enumerate(active.head(5).iterrows(), 1):
        print(f"  {i}. {row['pair']:<12} {row['action']:<14} "
              f"alpha={row['alpha_score']:.3f}  z={row['z_live']:+.2f}  "
              f"WR={row.get('win_rate', 0):.0%}")

    return signals


# ═══════════════════════════════════════════════════════════════
# STAGE 6: REPORT — GPT alpha report
# ═══════════════════════════════════════════════════════════════

def stage_report(signals: pd.DataFrame, all_data: dict) -> str:
    """Generate comprehensive alpha report via GPT-4o."""
    print("\n📝 STAGE 6: GPT ALPHA REPORT")
    print("=" * 60)

    from agents.registry import get_default_registry
    from core.contracts import AgentTask, AgentStatus
    registry = get_default_registry()

    active = signals[signals["action"] != "FLAT"]
    summary = {
        "date": str(date.today()),
        "total_pairs_scanned": len(signals),
        "active_signals": len(active),
        "top_signals": active.head(5)[["pair", "action", "z_live", "score"]].to_dict("records") if len(active) > 0 else [],
        "discovery_stats": all_data.get("discovery_stats", {}),
        "validation_pass_rate": all_data.get("validation_pass_rate", "?"),
    }

    task = AgentTask(
        task_id="alpha_report",
        agent_name="gpt_report_generator",
        task_type="gpt_report_generator.generate",
        payload={"agent_results": [
            {"stage": "alpha_pipeline", "output": json.dumps(summary, default=str)},
        ]},
    )
    result = registry.dispatch(task)
    report = result.output.get("report", "") if result.output else ""

    # Save
    report_dir = PROJECT_ROOT / "logs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"alpha_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_file.write_text(report or "No report generated", encoding="utf-8")
    print(f"  Report: {len(report)} chars → {report_file.name}")

    # Also save signals CSV
    signals_file = PROJECT_ROOT / "logs" / f"alpha_signals_{date.today().isoformat()}.csv"
    signals.to_csv(signals_file, index=False)
    print(f"  Signals: {signals_file.name}")

    return report


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_alpha_pipeline(top_n: int = 15, n_trials: int = 30, stage: str = "full") -> None:
    """Run the complete alpha generation pipeline."""
    print("\n" + "🏦" * 30)
    print("  ALPHA GENERATION PIPELINE")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print("🏦" * 30)

    t0 = time.time()
    all_data = {}

    # Stage 1: Discover
    if stage in ("full", "discover"):
        candidates = stage_discover(top_n=top_n)
        all_data["discovery_stats"] = {
            "n_candidates": len(candidates),
            "avg_score": float(candidates["score"].mean()),
            "top_pair": candidates.iloc[0]["pair"] if len(candidates) > 0 else "?",
        }
    else:
        # Load from saved
        csv_path = PROJECT_ROOT / "logs" / "alpha_discovery.csv"
        if csv_path.exists():
            candidates = pd.read_csv(csv_path, index_col=0)
        else:
            candidates = stage_discover(top_n=top_n)

    # Stage 2: Validate
    if stage in ("full", "validate"):
        validated = stage_validate(candidates)
        passed = validated[validated.get("validated", True) == True]
        all_data["validation_pass_rate"] = f"{len(passed)}/{len(validated)}"
    else:
        validated = candidates
        validated["validated"] = True

    # Stage 3: Optimize
    if stage in ("full", "optimize"):
        optimized = stage_optimize(validated, n_trials=n_trials)
    else:
        optimized = validated

    # Stage 4: Signals
    if stage in ("full", "signals"):
        signals = stage_signals(optimized)
    else:
        signals = optimized

    # Stage 5: Rank
    signals = stage_rank(signals)

    # Stage 6: Report
    if stage in ("full", "report"):
        stage_report(signals, all_data)

    elapsed = time.time() - t0

    # Summary
    active = signals[signals.get("action", "FLAT") != "FLAT"] if "action" in signals.columns else pd.DataFrame()
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete in {elapsed:.0f}s")
    print(f"  📊 Pairs scanned: {len(candidates)}")
    print(f"  ✅ Validated: {len(validated[validated.get('validated', True)==True])}")
    print(f"  📡 Active signals: {len(active)}")
    print(f"{'='*60}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    # Suppress noisy loggers
    for name in ["httpx", "openai", "urllib3", "yfinance", "agents.registry"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Alpha Generation Pipeline")
    parser.add_argument("--top-n", type=int, default=15, help="Top N pairs to analyze")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per pair")
    parser.add_argument("--stage", choices=["full", "discover", "validate", "optimize", "signals", "report"],
                        default="full")
    parser.add_argument("--daemon", action="store_true", help="Run every 4 hours")
    parser.add_argument("--interval", type=float, default=4.0, help="Hours between runs")
    args = parser.parse_args()

    if args.daemon:
        print(f"🔁 Alpha pipeline daemon: every {args.interval} hours")
        while True:
            try:
                run_alpha_pipeline(top_n=args.top_n, n_trials=args.trials, stage=args.stage)
                print(f"\n💤 Next run in {args.interval} hours...")
                time.sleep(args.interval * 3600)
            except KeyboardInterrupt:
                print("\n🛑 Stopped")
                break
    else:
        run_alpha_pipeline(top_n=args.top_n, n_trials=args.trials, stage=args.stage)


if __name__ == "__main__":
    main()
