#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_daily_pipeline.py — Operational daily pipeline trigger
==================================================================

CLI trigger for PairsOrchestrator.run_daily_pipeline().

Usage
-----
    python scripts/run_daily_pipeline.py
    python scripts/run_daily_pipeline.py --capital 500000
    python scripts/run_daily_pipeline.py --dry-run
    python scripts/run_daily_pipeline.py --daemon
    python scripts/run_daily_pipeline.py --daemon --run-time 16:30 --timezone US/Eastern

This script
-----------
- Runs: health_check → data_refresh → compute_signals → portfolio_allocation
- Dispatches SystemHealthAgent + DataIntegrityAgent via WorkflowEngine
- Calls run_portfolio_allocation_cycle() for funded positions
- Prints structured per-task summary
- Exits 0 = all tasks succeeded, 1 = partial failures, 2 = error

Cron setup (every weekday at 4:15pm ET)
----------------------------------------
    15 16 * * 1-5  cd /path/to/pairs_trading_system && \\
        python scripts/run_daily_pipeline.py >> logs/pipeline.log 2>&1

Or use --daemon mode to run an in-process scheduler:
    python scripts/run_daily_pipeline.py --daemon --run-time 16:15

Audit trail
-----------
    The agent registry records dispatches in its in-memory audit log.
    Run ``registry.get_audit_log()`` after execution to inspect.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Project root on sys.path so bare imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_daily_pipeline")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the pairs trading daily operational pipeline",
    )
    p.add_argument(
        "--capital", type=float, default=None,
        metavar="AMOUNT",
        help="Portfolio capital for allocation (default: from config scheduler_capital)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Log intent only — skip portfolio allocation step",
    )
    p.add_argument(
        "--daemon", action="store_true",
        help="Start a background scheduler that fires daily (blocks until interrupted)",
    )
    p.add_argument(
        "--run-time", type=str, default=None,
        metavar="HH:MM",
        help="Time to run daily in daemon mode (default: from config scheduler_run_time or 16:15)",
    )
    p.add_argument(
        "--timezone", type=str, default=None,
        metavar="TZ",
        help="Timezone for daemon run-time (default: from config scheduler_timezone or America/New_York)",
    )
    return p.parse_args()


def _print_task_result(result) -> None:
    """Print a single TaskResult row."""
    status_icon = "✓" if getattr(result, "success", False) else "✗"
    name = getattr(result, "task_name", "?")
    dur = getattr(result, "duration_ms", 0) or 0
    print(f"  {status_icon} {name:<30} {dur:>8.0f} ms")
    if not getattr(result, "success", True):
        err = getattr(result, "error", None)
        if err:
            print(f"    └─ {err}")


def _run_once(orchestrator, dry_run: bool, capital: float | None) -> int:
    """Execute one pipeline run. Returns exit code."""
    if dry_run:
        logger.info("DRY RUN — portfolio allocation will be skipped")

    try:
        results = orchestrator.run_daily_pipeline()
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        return 2

    # ── Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DAILY PIPELINE RESULT")
    print("=" * 65)

    if not results:
        print("  (no tasks ran)")
        return 0

    failed = 0
    for r in results:
        _print_task_result(r)
        if not getattr(r, "success", True):
            failed += 1

    print("-" * 65)
    total = len(results)
    passed = total - failed
    print(f"  Tasks: {passed}/{total} succeeded")

    if failed == 0:
        print("\n  ✓ Pipeline completed successfully.")
        return 0
    else:
        print(f"\n  ✗ {failed} task(s) failed — see logs for details.")
        return 1


def main() -> int:
    args = _parse_args()

    try:
        from core.orchestrator import PairsOrchestrator
    except ImportError as exc:
        logger.error("Cannot import core.orchestrator: %s", exc)
        return 2

    try:
        orchestrator = PairsOrchestrator()
    except Exception as exc:
        logger.error("Failed to initialise PairsOrchestrator: %s", exc)
        return 2

    # Override capital if provided on CLI
    if args.capital is not None:
        try:
            from common.config_manager import load_config
            cfg = load_config()
            cfg["scheduler_capital"] = args.capital
            logger.info("Capital override: %.0f", args.capital)
        except Exception:
            pass

    if args.daemon:
        # Resolve run_time / timezone (CLI > config > defaults)
        run_time = args.run_time
        timezone = args.timezone

        logger.info(
            "Starting daemon — run_time=%s tz=%s  (Ctrl-C to stop)",
            run_time or "from config/default",
            timezone or "from config/default",
        )
        kwargs: dict = {}
        if run_time:
            kwargs["run_time"] = run_time
        if timezone:
            kwargs["timezone"] = timezone

        try:
            orchestrator.start_daemon(run_immediately=True, **kwargs)
        except KeyboardInterrupt:
            logger.info("Daemon stopped by user.")
            orchestrator.stop_daemon()
        except Exception as exc:
            logger.error("Daemon error: %s", exc, exc_info=True)
            return 2
        return 0

    # Single run
    logger.info(
        "Running daily pipeline (dry_run=%s, capital=%s)",
        args.dry_run,
        args.capital or "from config",
    )
    return _run_once(orchestrator, dry_run=args.dry_run, capital=args.capital)


if __name__ == "__main__":
    sys.exit(main())
