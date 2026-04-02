#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_auto_improve.py — Autonomous Improvement Cycle
==========================================================

Runs the full autonomous improvement loop:
1. Refresh data for all symbols
2. GPT analyzes signal performance and model metrics
3. Retrain ML model if recommended
4. Optimize parameters if recommended
5. Generate daily report

Usage:
    python scripts/run_auto_improve.py              # Full cycle
    python scripts/run_auto_improve.py --dry-run    # Analysis only, no changes
    python scripts/run_auto_improve.py --cycle data  # Data refresh only
    python scripts/run_auto_improve.py --cycle model # Model retrain only
    python scripts/run_auto_improve.py --cycle optimize # Parameter optimization only
    python scripts/run_auto_improve.py --daemon     # Continuous loop (every 6 hours)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.registry import get_default_registry
from core.contracts import AgentTask, AgentStatus

logger = logging.getLogger(__name__)


def _dispatch(agent_name: str, task_type: str, payload: dict) -> dict:
    """Dispatch a single agent and return its result."""
    registry = get_default_registry()
    task = AgentTask(
        task_id=f"{agent_name}_{datetime.now(timezone.utc).strftime('%H%M%S')}",
        agent_name=agent_name,
        task_type=task_type,
        payload=payload,
    )
    result = registry.dispatch(task)
    status = "OK" if result.status == AgentStatus.COMPLETED else "FAIL"
    logger.info(f"  {agent_name}: {status}")
    return result.output if result.output else {}


def run_data_cycle() -> dict:
    """Refresh price data for all symbols."""
    print("\n📊 Data Refresh Cycle")
    print("=" * 50)
    result = _dispatch("auto_data_refresher", "auto_refresh_data", {})
    print(f"  Refreshed: {result.get('refreshed', 0)} symbols")
    print(f"  Failed: {result.get('failed', 0)}")
    return result


def run_analysis_cycle() -> dict:
    """GPT-powered analysis of signals and models."""
    print("\n🧠 GPT Analysis Cycle")
    print("=" * 50)
    results = {}

    # Signal analysis
    print("  → GPT Signal Advisor...")
    results["signal"] = _dispatch(
        "gpt_signal_advisor", "gpt_signal_advisor.analyze",
        {"pairs_summary": {"note": "Analyze overall pair performance trends"}},
    )
    recs = results["signal"].get("recommendations", [])
    print(f"    {len(recs)} recommendations")

    # Model analysis
    print("  → GPT Model Tuner...")
    try:
        from pathlib import Path as _P
        model_path = _P("models/meta_label_latest.pkl")
        metrics = {"model_exists": model_path.exists()}
        if model_path.exists():
            from ml.models.meta_labeler import MetaLabelModel
            model = MetaLabelModel.load(str(model_path))
            metrics["is_fitted"] = model.is_fitted
    except Exception:
        metrics = {"model_exists": False}

    results["model"] = _dispatch(
        "gpt_model_tuner", "gpt_model_tuner.analyze",
        {"model_metrics": metrics},
    )
    print(f"    Retrain recommended: {results['model'].get('should_retrain', '?')}")

    # Strategy research
    print("  → GPT Strategy Researcher...")
    results["strategy"] = _dispatch(
        "gpt_strategy_researcher", "gpt_strategy_researcher.research",
        {"universe_summary": {}, "market_context": "Current market conditions"},
    )
    ideas = results["strategy"].get("ideas", [])
    print(f"    {len(ideas)} new ideas")

    return results


def run_model_cycle() -> dict:
    """Retrain ML model."""
    print("\n🔄 Model Retrain Cycle")
    print("=" * 50)
    result = _dispatch("auto_model_retrainer", "auto_retrain_model", {
        "pair": "XLY-XLC",
        "horizon": 10,
        "entry_threshold": 1.5,
    })
    if result.get("retrained"):
        print(f"  ✅ Model retrained: AUC={result.get('auc', '?'):.3f}")
    else:
        print(f"  ❌ Retrain failed: {result.get('error', result.get('reason', '?'))}")
    return result


def run_optimize_cycle() -> dict:
    """Optimize trading parameters."""
    print("\n⚙️ Parameter Optimization Cycle")
    print("=" * 50)
    result = _dispatch("auto_parameter_optimizer", "auto_optimize_params", {
        "pair": "XLY-XLC",
        "n_trials": 20,
    })
    if result.get("optimized"):
        print(f"  ✅ Best params: {result.get('best_params')}")
        print(f"  ✅ Best value: {result.get('best_value', '?'):.3f}")
    else:
        print(f"  ❌ Optimization failed: {result.get('error', '?')}")
    return result


def run_report_cycle(all_results: dict) -> str:
    """Generate GPT report of all activity."""
    print("\n📝 Report Generation")
    print("=" * 50)
    result = _dispatch("gpt_report_generator", "gpt_report_generator.generate", {
        "agent_results": [
            {"cycle": k, "output": json.dumps(v, default=str)[:500]}
            for k, v in all_results.items()
        ],
    })
    report = result.get("report", "No report generated")
    print(f"  Report: {len(report)} chars")

    # Save report
    report_dir = PROJECT_ROOT / "logs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"auto_improve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_file.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_file.name}")

    return report


def run_full_cycle(dry_run: bool = False) -> None:
    """Run the complete autonomous improvement cycle."""
    print("\n" + "=" * 60)
    print("🚀 AUTONOMOUS IMPROVEMENT CYCLE")
    print(f"   Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'FULL EXECUTION'}")
    print("=" * 60)

    all_results = {}
    t0 = time.time()

    # 1. Data refresh
    all_results["data"] = run_data_cycle()

    # 2. GPT analysis
    all_results["analysis"] = run_analysis_cycle()

    if not dry_run:
        # 3. Model retrain (if GPT recommends)
        should_retrain = all_results.get("analysis", {}).get("model", {}).get("should_retrain", True)
        if should_retrain:
            all_results["model"] = run_model_cycle()

        # 4. Parameter optimization
        all_results["optimize"] = run_optimize_cycle()

    # 5. Report
    run_report_cycle(all_results)

    elapsed = time.time() - t0

    # Cost summary
    try:
        from common.gpt_client import get_gpt_client
        cost = get_gpt_client().get_cost_summary()
        print(f"\n💰 GPT Cost: ${cost.get('daily_cost_usd', 0):.4f} today")
    except Exception:
        pass

    print(f"\n⏱️ Total time: {elapsed:.1f}s")
    print("=" * 60)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Autonomous improvement cycle")
    parser.add_argument("--dry-run", action="store_true", help="Analysis only, no changes")
    parser.add_argument("--cycle", choices=["full", "data", "model", "optimize", "analysis"],
                        default="full", help="Which cycle to run")
    parser.add_argument("--daemon", action="store_true", help="Continuous loop")
    parser.add_argument("--interval-hours", type=float, default=6.0,
                        help="Hours between cycles in daemon mode")
    args = parser.parse_args()

    if args.daemon:
        print(f"🔁 Daemon mode: running every {args.interval_hours} hours")
        while True:
            try:
                run_full_cycle(dry_run=args.dry_run)
                print(f"\n💤 Sleeping {args.interval_hours} hours...")
                time.sleep(args.interval_hours * 3600)
            except KeyboardInterrupt:
                print("\n🛑 Daemon stopped")
                break
    elif args.cycle == "full":
        run_full_cycle(dry_run=args.dry_run)
    elif args.cycle == "data":
        run_data_cycle()
    elif args.cycle == "model":
        run_model_cycle()
    elif args.cycle == "optimize":
        run_optimize_cycle()
    elif args.cycle == "analysis":
        run_analysis_cycle()


if __name__ == "__main__":
    main()
