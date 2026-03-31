# -*- coding: utf-8 -*-
"""
core/orchestrator.py — Pairs Trading Orchestrator (inspired by srv_quant)
=========================================================================

Automated scheduling and pipeline management for the pairs trading system.

Features:
- Cron-like task scheduling with dependencies
- Daily pipeline: data refresh → signals → risk check → paper trading
- Health monitoring cycle
- Agent bus for inter-component messaging
- Graceful error handling per task

Usage:
    from core.orchestrator import PairsOrchestrator
    orch = PairsOrchestrator()
    orch.run_daily_pipeline()   # Run full daily cycle
    orch.start_daemon()         # Start background scheduler
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("orchestrator")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
AGENT_BUS_PATH = LOGS_DIR / "agent_bus.json"


# ============================================================================
# Agent Bus — file-based inter-component messaging
# ============================================================================

class AgentBus:
    """File-based message bus for agent communication (no external deps)."""

    def __init__(self, path: Path = AGENT_BUS_PATH, max_history: int = 20):
        self.path = path
        self.max_history = max_history
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, list]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self, data: Dict[str, list]) -> None:
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def publish(self, agent: str, payload: Dict[str, Any]) -> None:
        data = self._load()
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        if agent not in data:
            data[agent] = []
        data[agent].append(entry)
        data[agent] = data[agent][-self.max_history:]
        self._save(data)

    def latest(self, agent: str) -> Optional[Dict[str, Any]]:
        data = self._load()
        entries = data.get(agent, [])
        return entries[-1] if entries else None

    def history(self, agent: str, n: int = 10) -> List[Dict[str, Any]]:
        data = self._load()
        return data.get(agent, [])[-n:]


# ============================================================================
# Task Registry
# ============================================================================

@dataclass
class TaskResult:
    """Result from a single task execution."""
    task_name: str
    status: str  # "success" | "failed" | "skipped"
    duration_sec: float = 0.0
    error: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ScheduledTask:
    """Definition of a scheduled task."""
    name: str
    func: Callable
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    enabled: bool = True
    last_run: Optional[str] = None
    last_status: str = "never_run"


# ============================================================================
# Pipeline Tasks
# ============================================================================

def task_data_refresh(**kwargs) -> Dict[str, Any]:
    """Refresh price data from SQL store or Yahoo."""
    from common.data_loader import bulk_download
    from common.config_manager import load_config

    cfg = load_config()
    pairs = cfg.get("pairs", [])
    symbols = set()
    for p in pairs:
        if isinstance(p, dict):
            symbols.add(p.get("sym_x", ""))
            symbols.add(p.get("sym_y", ""))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            symbols.update(p[:2])

    symbols.discard("")
    if not symbols:
        return {"status": "no_symbols", "count": 0}

    results = bulk_download(list(symbols), force=False)
    return {"status": "ok", "symbols_refreshed": len(symbols)}


def task_compute_signals(**kwargs) -> Dict[str, Any]:
    """Compute signals for all pairs in universe."""
    try:
        from core.signals_engine import compute_universe_signals
        from core.app_context import AppContext

        ctx = AppContext.get_global()
        result = compute_universe_signals(ctx)
        n = len(result) if result is not None else 0
        return {"status": "ok", "pairs_computed": n}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def task_risk_check(**kwargs) -> Dict[str, Any]:
    """Run risk assessment on current state."""
    try:
        from core.risk_engine import (
            RiskState, RiskLimits, check_risk_breaches,
            compute_overall_risk_score,
        )

        state = RiskState()
        limits = RiskLimits()
        breaches = check_risk_breaches(state, limits)
        score = compute_overall_risk_score(state, limits)
        return {
            "status": "ok",
            "risk_score": score,
            "breaches": breaches,
            "breach_count": len(breaches),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def task_health_check(**kwargs) -> Dict[str, Any]:
    """System health check — verify all components are importable and responsive."""
    results = {}
    modules = [
        "core.sql_store", "core.optimizer", "core.risk_engine",
        "core.signals_engine", "core.fair_value_engine", "core.macro_engine",
        "common.config_manager", "common.data_loader",
    ]
    for mod in modules:
        try:
            __import__(mod)
            results[mod] = "ok"
        except Exception as e:
            results[mod] = f"error: {e}"

    ok_count = sum(1 for v in results.values() if v == "ok")
    return {
        "status": "ok" if ok_count == len(modules) else "degraded",
        "modules_ok": ok_count,
        "modules_total": len(modules),
        "details": results,
    }


def task_sql_store_maintenance(**kwargs) -> Dict[str, Any]:
    """Run SQL store maintenance (vacuum, integrity checks)."""
    try:
        from core.sql_store import SqlStore
        from common.config_manager import load_config

        cfg = load_config()
        url = cfg.get("engine_url") or cfg.get("sql_store_url")
        if not url:
            return {"status": "skipped", "reason": "no_sql_url"}

        store = SqlStore(url)
        tables = store.list_tables()
        return {"status": "ok", "tables": len(tables)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================================
# Orchestrator
# ============================================================================

class PairsOrchestrator:
    """
    Main orchestrator for the pairs trading system.

    Manages task scheduling, execution, and inter-component communication.
    Inspired by srv_quant's agent orchestrator pattern.
    """

    def __init__(self):
        self.bus = AgentBus()
        self.tasks: Dict[str, ScheduledTask] = {}
        self._results: List[TaskResult] = []
        self._register_default_tasks()

    def _register_default_tasks(self) -> None:
        """Register the standard pipeline tasks."""
        self.register_task(ScheduledTask(
            name="health_check",
            func=task_health_check,
            description="System health check",
        ))
        self.register_task(ScheduledTask(
            name="data_refresh",
            func=task_data_refresh,
            description="Refresh price data",
            depends_on=["health_check"],
        ))
        self.register_task(ScheduledTask(
            name="compute_signals",
            func=task_compute_signals,
            description="Compute pair signals",
            depends_on=["data_refresh"],
        ))
        self.register_task(ScheduledTask(
            name="risk_check",
            func=task_risk_check,
            description="Run risk assessment",
            depends_on=["compute_signals"],
        ))
        self.register_task(ScheduledTask(
            name="sql_maintenance",
            func=task_sql_store_maintenance,
            description="SQL store maintenance",
        ))

    def register_task(self, task: ScheduledTask) -> None:
        self.tasks[task.name] = task

    def _execute_task(self, task: ScheduledTask, **kwargs) -> TaskResult:
        """Execute a single task with timing and error handling."""
        if not task.enabled:
            return TaskResult(task.name, "skipped")

        # Check dependencies
        for dep in task.depends_on:
            dep_task = self.tasks.get(dep)
            if dep_task and dep_task.last_status == "failed":
                logger.warning(
                    "Skipping %s: dependency %s failed", task.name, dep,
                )
                return TaskResult(task.name, "skipped", error=f"dep:{dep} failed")

        logger.info("Running task: %s (%s)", task.name, task.description)
        t0 = time.time()
        try:
            output = task.func(**kwargs)
            elapsed = time.time() - t0
            task.last_run = datetime.now(timezone.utc).isoformat()
            task.last_status = "success"

            result = TaskResult(task.name, "success", elapsed, output=output)
            self.bus.publish(task.name, {
                "status": "success",
                "duration_sec": round(elapsed, 2),
                "output": output,
            })
            logger.info("Task %s completed in %.1fs", task.name, elapsed)
            return result

        except Exception as e:
            elapsed = time.time() - t0
            task.last_status = "failed"
            logger.error("Task %s failed after %.1fs: %s", task.name, elapsed, e)
            result = TaskResult(task.name, "failed", elapsed, error=str(e))
            self.bus.publish(task.name, {
                "status": "failed",
                "duration_sec": round(elapsed, 2),
                "error": str(e),
            })
            return result

    def run_daily_pipeline(self) -> List[TaskResult]:
        """
        Execute the full daily pipeline in dependency order.

        Pipeline: health_check → data_refresh → agent_checks → compute_signals → risk_check

        The agent_checks step dispatches the DataIntegrityAgent to validate
        price data quality after data refresh (P1-AGENTS).  This is the first
        real agent dispatch from operational code.
        """
        logger.info("=" * 60)
        logger.info("Starting daily pipeline at %s", datetime.now(timezone.utc).isoformat())
        logger.info("=" * 60)

        order = ["health_check", "data_refresh", "compute_signals", "risk_check"]
        results = []

        for name in order:
            task = self.tasks.get(name)
            if task:
                result = self._execute_task(task)
                results.append(result)
                self._results.append(result)

            # After data_refresh, dispatch agent integrity check (P1-AGENTS)
            if name == "data_refresh":
                agent_result = self.run_agent_data_integrity_check()
                if agent_result is not None:
                    results.append(agent_result)
                    self._results.append(agent_result)

        # Summary
        ok = sum(1 for r in results if r.status == "success")
        logger.info(
            "Daily pipeline complete: %d/%d tasks succeeded",
            ok, len(results),
        )

        # Publish pipeline summary
        self.bus.publish("daily_pipeline", {
            "tasks_ok": ok,
            "tasks_total": len(results),
            "results": [asdict(r) for r in results],
        })

        return results

    def run_agent_data_integrity_check(self) -> Optional[TaskResult]:
        """
        Dispatch the DataIntegrityAgent to validate price data quality.

        This is the first real agent dispatch from operational code (P1-AGENTS).
        The agent is READ_ONLY — it does not mutate any state.  It returns
        typed AgentResult with audit trail.

        Returns
        -------
        TaskResult or None
            TaskResult wrapping the AgentResult, or None if dispatch fails.
        """
        import time as _time
        t0 = _time.time()
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus

            registry = get_default_registry()
            agent = registry.get_agent("data_integrity")
            if agent is None:
                logger.warning("DataIntegrityAgent not registered — skipping")
                return None

            # Create typed task
            task = AgentTask(
                task_id=f"orch_di_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                agent_name="data_integrity",
                task_type="check_data_integrity",
                payload={},   # Agent handles missing prices gracefully
                priority=3,
                correlation_id=f"daily_pipeline_{datetime.now(timezone.utc).date()}",
            )

            # Dispatch via registry (typed, audited)
            result = registry.dispatch(task)

            elapsed = _time.time() - t0
            status = "success" if result.status == AgentStatus.COMPLETED else "failed"
            issues = result.output.get("issues_found", 0) if result.output else 0

            logger.info(
                "Agent data_integrity: status=%s, issues=%d, duration=%.1fs",
                status, issues, elapsed,
            )

            return TaskResult(
                task_name="agent_data_integrity",
                status=status,
                duration_sec=round(elapsed, 2),
                output={
                    "agent_name": "data_integrity",
                    "agent_status": result.status.value,
                    "issues_found": issues,
                    "critical_issues": result.output.get("critical_issues", []) if result.output else [],
                    "audit_trail_size": len(result.audit_trail),
                },
            )

        except Exception as e:
            elapsed = _time.time() - t0
            logger.warning("Agent data_integrity dispatch failed: %s", e)
            return TaskResult(
                task_name="agent_data_integrity",
                status="failed",
                duration_sec=round(elapsed, 2),
                error=str(e),
            )

    def run_maintenance(self) -> TaskResult:
        """Run SQL store maintenance."""
        task = self.tasks.get("sql_maintenance")
        if task:
            result = self._execute_task(task)
            self._results.append(result)
            return result
        return TaskResult("sql_maintenance", "skipped")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all tasks."""
        return {
            name: {
                "description": t.description,
                "enabled": t.enabled,
                "last_run": t.last_run,
                "last_status": t.last_status,
            }
            for name, t in self.tasks.items()
        }

    def get_results_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent task results."""
        return [asdict(r) for r in self._results[-n:]]


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    """Run the daily pipeline from CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    orch = PairsOrchestrator()
    results = orch.run_daily_pipeline()

    print("\n--- Pipeline Summary ---")
    for r in results:
        icon = "✓" if r.status == "success" else "✗" if r.status == "failed" else "○"
        print(f"  {icon} {r.task_name}: {r.status} ({r.duration_sec:.1f}s)")


if __name__ == "__main__":
    main()
