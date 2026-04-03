#!/usr/bin/env python
"""
scripts/run_scheduler_daemon.py — Unified Scheduler Daemon
============================================================

Single-process daemon that manages all recurring jobs via APScheduler:
  - Daily pipeline (weekdays 16:15 ET)
  - Auto-improvement cycle (every 6 hours)
  - Data refresh (every 2 hours)
  - Heartbeat (every 30 minutes)

Usage:
    python scripts/run_scheduler_daemon.py              # Start daemon
    python scripts/run_scheduler_daemon.py --run-now     # Run all jobs once, then start daemon
    python scripts/run_scheduler_daemon.py --run-now --once  # Run all jobs once, then exit

The daemon reads schedule config from config.json and sends Telegram
alerts on start/stop/failure (if configured in .env).
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DuckDB segfault guard (Python 3.13)
try:
    import duckdb_engine  # noqa: F401
except ImportError:
    pass

# Load .env early
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logger = logging.getLogger("scheduler_daemon")

# ── Paths ──────────────────────────────────────────────────────────
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "scheduler_daemon.log"
PID_FILE = LOGS_DIR / "scheduler.pid"
HEARTBEAT_FILE = LOGS_DIR / "scheduler_heartbeat.json"

# ── Shutdown event ─────────────────────────────────────────────────
_shutdown = threading.Event()


# =====================================================================
# Logging setup
# =====================================================================

def _setup_logging() -> None:
    """Configure rotating file + console logging."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler (5 MB, 3 backups)
    fh = RotatingFileHandler(
        str(LOG_FILE), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)


# =====================================================================
# Alert helpers
# =====================================================================

def _alert(component: str, status: str, details: str = "") -> None:
    """Send alert via Telegram + console (best-effort)."""
    try:
        from core.alerts import alert_system
        alert_system(component, status, details)
    except Exception:
        logger.info("Alert [%s] %s: %s", component, status, details)


# =====================================================================
# Job wrappers (each catches exceptions and alerts on failure)
# =====================================================================

def _job_daily_pipeline() -> None:
    """Run the full daily trading pipeline."""
    logger.info("=" * 50)
    logger.info("JOB START: daily_pipeline")
    t0 = time.time()
    try:
        from core.orchestrator import PairsOrchestrator
        orch = PairsOrchestrator()
        results = orch.run_daily_pipeline()
        n_ok = sum(1 for r in results if r.status == "OK")
        n_fail = sum(1 for r in results if r.status != "OK")
        elapsed = time.time() - t0
        logger.info("JOB DONE: daily_pipeline — %d OK, %d failed (%.1fs)", n_ok, n_fail, elapsed)
        _alert("daily_pipeline", "OK", f"{n_ok} tasks OK, {n_fail} failed ({elapsed:.0f}s)")
    except Exception as exc:
        logger.error("JOB FAILED: daily_pipeline — %s", exc, exc_info=True)
        _alert("daily_pipeline", "FAILED", str(exc))


def _job_auto_improve() -> None:
    """Run the GPT auto-improvement cycle."""
    logger.info("=" * 50)
    logger.info("JOB START: auto_improve")
    t0 = time.time()
    try:
        from scripts.run_auto_improve import run_full_cycle
        run_full_cycle(dry_run=False)
        elapsed = time.time() - t0
        logger.info("JOB DONE: auto_improve (%.1fs)", elapsed)
        _alert("auto_improve", "OK", f"Completed in {elapsed:.0f}s")
    except Exception as exc:
        logger.error("JOB FAILED: auto_improve — %s", exc, exc_info=True)
        _alert("auto_improve", "FAILED", str(exc))


def _job_data_refresh() -> None:
    """Refresh price data for all symbols."""
    logger.info("JOB START: data_refresh")
    t0 = time.time()
    try:
        from scripts.run_auto_improve import run_data_cycle
        run_data_cycle()
        elapsed = time.time() - t0
        logger.info("JOB DONE: data_refresh (%.1fs)", elapsed)
    except Exception as exc:
        logger.error("JOB FAILED: data_refresh — %s", exc, exc_info=True)
        _alert("data_refresh", "FAILED", str(exc))


def _job_heartbeat() -> None:
    """Write heartbeat timestamp for external monitoring."""
    try:
        HEARTBEAT_FILE.write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "status": "alive",
        }, indent=2), encoding="utf-8")
    except Exception:
        pass


# =====================================================================
# PID file management
# =====================================================================

def _write_pid() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def _remove_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# =====================================================================
# Signal handlers
# =====================================================================

def _signal_handler(signum, frame):
    logger.info("Received signal %s — shutting down...", signum)
    _shutdown.set()


# =====================================================================
# Main daemon
# =====================================================================

def _load_schedule_config() -> dict:
    """Load scheduler-related config from config.json."""
    defaults = {
        "scheduler_run_time": "16:15",
        "scheduler_timezone": "America/New_York",
        "auto_improve_interval_hours": 6.0,
        "data_refresh_interval_hours": 2.0,
        "scheduler_heartbeat_interval_minutes": 30,
    }
    try:
        from common.config_manager import load_config
        cfg = load_config()
        for k, v in defaults.items():
            defaults[k] = cfg.get(k, v)
    except Exception:
        logger.warning("Could not load config — using defaults")
    return defaults


def start_daemon(run_now: bool = False, run_once: bool = False) -> None:
    """Start the scheduler daemon."""
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    _setup_logging()
    _write_pid()
    atexit.register(_remove_pid)

    # Register signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    # Windows-specific
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_handler)

    cfg = _load_schedule_config()
    run_time = cfg["scheduler_run_time"]
    tz = cfg["scheduler_timezone"]
    hour, minute = (int(x) for x in run_time.split(":"))

    logger.info("=" * 60)
    logger.info("PAIRS TRADING SCHEDULER DAEMON")
    logger.info("  PID: %d", os.getpid())
    logger.info("  Daily pipeline: %s %s (weekdays)", run_time, tz)
    logger.info("  Auto-improve: every %.1fh", cfg["auto_improve_interval_hours"])
    logger.info("  Data refresh: every %.1fh", cfg["data_refresh_interval_hours"])
    logger.info("  Heartbeat: every %dm", cfg["scheduler_heartbeat_interval_minutes"])
    logger.info("  Log file: %s", LOG_FILE)
    logger.info("=" * 60)

    _alert("scheduler", "STARTED", f"PID {os.getpid()}, pipeline at {run_time} {tz}")

    # Run immediately if requested
    if run_now:
        logger.info("--run-now: executing all jobs immediately...")
        _job_heartbeat()
        _job_data_refresh()
        _job_daily_pipeline()
        _job_auto_improve()
        if run_once:
            logger.info("--once: exiting after immediate run")
            _alert("scheduler", "STOPPED", "Single run complete")
            return

    # Create scheduler
    scheduler = BackgroundScheduler(timezone=tz)

    # Daily pipeline — weekdays at configured time
    scheduler.add_job(
        _job_daily_pipeline,
        CronTrigger(hour=hour, minute=minute, day_of_week="mon-fri", timezone=tz),
        id="daily_pipeline",
        name="Daily Trading Pipeline",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # Auto-improvement — every N hours
    scheduler.add_job(
        _job_auto_improve,
        IntervalTrigger(hours=cfg["auto_improve_interval_hours"]),
        id="auto_improve",
        name="Auto-Improvement Cycle",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # Data refresh — every N hours
    scheduler.add_job(
        _job_data_refresh,
        IntervalTrigger(hours=cfg["data_refresh_interval_hours"]),
        id="data_refresh",
        name="Data Refresh",
        misfire_grace_time=1800,
        coalesce=True,
    )

    # Heartbeat — every N minutes
    scheduler.add_job(
        _job_heartbeat,
        IntervalTrigger(minutes=cfg["scheduler_heartbeat_interval_minutes"]),
        id="heartbeat",
        name="Heartbeat",
        coalesce=True,
    )

    scheduler.start()
    logger.info("Scheduler started — %d jobs registered", len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        logger.info("  [%s] next run: %s", job.id, job.next_run_time)

    # Write initial heartbeat
    _job_heartbeat()

    # Block until shutdown signal
    try:
        _shutdown.wait()
    except KeyboardInterrupt:
        pass

    logger.info("Shutting down scheduler...")
    scheduler.shutdown(wait=False)
    _alert("scheduler", "STOPPED", "Graceful shutdown")
    logger.info("Scheduler stopped.")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pairs Trading Scheduler Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_scheduler_daemon.py              # Start daemon
  python scripts/run_scheduler_daemon.py --run-now     # Run all jobs immediately, then continue as daemon
  python scripts/run_scheduler_daemon.py --run-now --once  # Run all jobs once, then exit
        """,
    )
    parser.add_argument("--run-now", action="store_true", help="Execute all jobs immediately on start")
    parser.add_argument("--once", action="store_true", help="Exit after --run-now (no daemon loop)")
    args = parser.parse_args()

    if args.once and not args.run_now:
        parser.error("--once requires --run-now")

    start_daemon(run_now=args.run_now, run_once=args.once)


if __name__ == "__main__":
    main()
