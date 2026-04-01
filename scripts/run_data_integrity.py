#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_data_integrity.py — Standalone data integrity check
===============================================================

CLI trigger for the DataIntegrityAgent workflow.

Usage
-----
    python scripts/run_data_integrity.py
    python scripts/run_data_integrity.py --symbols SPY QQQ GLD
    python scripts/run_data_integrity.py --max-gap 3 --max-move 0.15

This script:
- Dispatches DataIntegrityAgent via WorkflowEngine (P1-AGENTS)
- Prints a structured summary of findings
- Exits 0 if no critical issues, 1 if critical issues found, 2 on error

Audit trail
-----------
    The agent registry records the dispatch in its in-memory audit log.
    Run ``registry.get_audit_log(agent_name='data_integrity')`` to inspect.
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
logger = logging.getLogger("run_data_integrity")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DataIntegrityAgent data-quality check",
    )
    p.add_argument(
        "--symbols", nargs="*", default=None,
        help="Symbols to check (default: all in config)",
    )
    p.add_argument(
        "--max-gap", type=int, default=5,
        metavar="DAYS",
        help="Max consecutive NaN gap allowed (default: 5)",
    )
    p.add_argument(
        "--max-move", type=float, default=0.20,
        metavar="FRAC",
        help="Max single-day price move allowed (default: 0.20 = 20%%)",
    )
    p.add_argument(
        "--expected-days", type=int, default=252,
        metavar="N",
        help="Expected minimum trading day count per symbol (default: 252)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    logger.info(
        "DataIntegrity check: symbols=%s max_gap=%sd max_move=%.0f%%",
        args.symbols or "all",
        args.max_gap,
        args.max_move * 100,
    )

    try:
        from monitoring.workflow import run_data_integrity_workflow
    except ImportError as exc:
        logger.error("Cannot import monitoring.workflow: %s", exc)
        return 2

    try:
        outcome = run_data_integrity_workflow(
            symbols=args.symbols,
            expected_trading_days=args.expected_days,
            max_gap_days=args.max_gap,
            max_daily_move_pct=args.max_move,
            triggered_by="cli:run_data_integrity.py",
            emit_alerts=False,   # no Streamlit session in CLI
        )
    except Exception as exc:
        logger.error("Workflow execution failed: %s", exc)
        return 2

    # ── Print summary ──────────────────────────────────────────────
    from orchestration.contracts import WorkflowStatus

    print("\n" + "=" * 60)
    print("DATA INTEGRITY WORKFLOW RESULT")
    print("=" * 60)
    print(f"  Workflow run ID  : {outcome.run_id}")
    print(f"  Status           : {outcome.status.value}")
    print(f"  Steps completed  : {outcome.steps_completed}")
    print(f"  Steps failed     : {outcome.steps_failed}")
    print(f"  Duration ms      : {outcome.total_duration_ms:.0f}")

    if outcome.status == WorkflowStatus.COMPLETED:
        print("\n  ✓ No critical issues detected.")
        return 0

    # Check audit log for issue details
    try:
        from agents.registry import get_default_registry
        registry = get_default_registry()
        log_entries = registry.get_audit_log(
            agent_name="data_integrity", limit=1,
        )
        if log_entries:
            entry = log_entries[-1]
            print(f"\n  Agent status : {entry.get('status')}")
            print(f"  Duration     : {entry.get('duration_seconds', 0):.2f}s")
            print(f"  Audit entries: {entry.get('n_audit_entries', 0)}")
    except Exception:
        pass

    if outcome.steps_failed > 0:
        print("\n  ✗ One or more integrity steps failed — see logs.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
