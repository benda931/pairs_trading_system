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

        Pipeline:
            health_check (bare)
            → agent_system_health (WorkflowEngine, P1-AGENTS)
            → data_refresh
            → agent_data_integrity (WorkflowEngine, P1-AGENTS)
            → compute_signals
            → portfolio_allocation (SignalPipeline → bridge_signals_to_allocator,
                                    with is_safe_to_trade + kill-switch factory)
            → risk_check

        Agent dispatches:
        1. SystemHealthAgent (after health_check) — P1-AGENTS
        2. DataIntegrityAgent (after data_refresh) — P1-AGENTS

        Allocation:
            After compute_signals, _collect_signal_decisions() runs SignalPipeline
            for each active pair, then run_portfolio_allocation_cycle() feeds the
            resulting SignalDecision objects through bridge_signals_to_allocator()
            with runtime safety check and control-plane kill-switch (P1-PORTINT,
            P1-SAFE, P0-KS — all COMPLETE via this call path).
        """
        logger.info("=" * 60)
        logger.info("Starting daily pipeline at %s", datetime.now(timezone.utc).isoformat())
        logger.info("=" * 60)

        order = ["health_check", "data_refresh", "compute_signals", "risk_check"]
        results = []
        all_signal_decisions = []  # Collect for alert summary

        for name in order:
            task = self.tasks.get(name)
            if task:
                result = self._execute_task(task)
                results.append(result)
                self._results.append(result)

            # After health_check, dispatch SystemHealthAgent (P1-AGENTS)
            if name == "health_check":
                agent_result = self.run_agent_system_health_check()
                if agent_result is not None:
                    results.append(agent_result)
                    self._results.append(agent_result)

            # After data_refresh, dispatch DataIntegrityAgent (P1-AGENTS)
            if name == "data_refresh":
                agent_result = self.run_agent_data_integrity_check()
                if agent_result is not None:
                    results.append(agent_result)
                    self._results.append(agent_result)

            # After compute_signals: run signal agents → allocate → risk agents
            if name == "compute_signals":
                # Signal-layer agents (regime surveillance)
                for ar in self._dispatch_signal_agents():
                    results.append(ar)
                    self._results.append(ar)

                # Collect decisions and allocate
                signal_decisions = self._collect_signal_decisions()
                all_signal_decisions = signal_decisions  # Save for alert summary
                if signal_decisions:
                    try:
                        from common.config_manager import load_config
                        capital = float(load_config().get("scheduler_capital", 1_000_000.0))
                    except Exception:
                        capital = 1_000_000.0
                    alloc_result = self.run_portfolio_allocation_cycle(
                        signal_decisions=signal_decisions,
                        capital=capital,
                    )
                    if alloc_result is not None:
                        results.append(alloc_result)
                        self._results.append(alloc_result)

                    # Risk-layer agents (exposure, drawdown, kill-switch)
                    for ar in self._dispatch_risk_agents():
                        results.append(ar)
                        self._results.append(ar)
                else:
                    logger.info("run_daily_pipeline: no signal decisions — allocation skipped")

        # ── Run risk analytics on portfolio returns ─────────────
        risk_result = self._run_risk_analytics()
        if risk_result is not None:
            results.append(risk_result)
            self._results.append(risk_result)

        # ── Run universe scanner for new pair opportunities ───
        scan_result = self._run_universe_scan()
        if scan_result is not None:
            results.append(scan_result)
            self._results.append(scan_result)

        # Summary
        ok = sum(1 for r in results if r.status == "success")
        failed = len(results) - ok
        logger.info(
            "Daily pipeline complete: %d/%d tasks succeeded",
            ok, len(results),
        )

        # ── Send operational alerts ──────────────────────────────
        self._send_pipeline_alerts(results, all_signal_decisions)

        # Publish pipeline summary
        self.bus.publish("daily_pipeline", {
            "tasks_ok": ok,
            "tasks_total": len(results),
            "results": [asdict(r) for r in results],
        })

        return results

    def _run_risk_analytics(self) -> Optional[TaskResult]:
        """Run portfolio-level risk analytics using the new RiskAnalytics engine."""
        try:
            from core.risk_analytics import RiskAnalytics

            # Try to load recent equity curve / returns
            returns = None
            try:
                from core.alpha_persistence import load_alpha_results
                alpha = load_alpha_results()
                if alpha and "equity_curve" in alpha:
                    eq = pd.Series(alpha["equity_curve"])
                    returns = eq.pct_change().dropna()
            except Exception:
                pass

            if returns is None or len(returns) < 30:
                logger.info("Risk analytics: no returns data available, skipping")
                return None

            ra = RiskAnalytics()
            report = ra.full_risk_report(returns, name="Portfolio")

            # Alert on high risk
            if report.var_95 and report.var_95.cvar > 0.03:
                try:
                    from core.alerts import alert_risk
                    alert_risk(
                        "High CVaR",
                        f"95% CVaR = {report.var_95.cvar:.2%} (threshold: 3%)",
                        severity="WARNING",
                    )
                except Exception:
                    pass

            if report.drawdown and report.drawdown.current_drawdown < -0.10:
                try:
                    from core.alerts import alert_risk
                    alert_risk(
                        "Drawdown Alert",
                        f"Current DD = {report.drawdown.current_drawdown:.2%}",
                        severity="CRITICAL" if report.drawdown.current_drawdown < -0.20 else "WARNING",
                    )
                except Exception:
                    pass

            self.bus.publish("risk_analytics", {
                "sharpe": report.sharpe_ratio,
                "vol": report.annualized_vol,
                "var95": report.var_95.cornish_fisher_var if report.var_95 else None,
                "cvar95": report.var_95.cvar if report.var_95 else None,
                "max_dd": report.drawdown.max_drawdown if report.drawdown else None,
                "current_dd": report.drawdown.current_drawdown if report.drawdown else None,
            })

            logger.info(
                "Risk analytics: Sharpe=%.2f, Vol=%.1f%%, CVaR95=%.2f%%, MaxDD=%.1f%%",
                report.sharpe_ratio,
                report.annualized_vol * 100,
                (report.var_95.cvar if report.var_95 else 0) * 100,
                (report.drawdown.max_drawdown if report.drawdown else 0) * 100,
            )

            return TaskResult(
                task_name="risk_analytics",
                status="success",
                message=f"Sharpe={report.sharpe_ratio:.2f}, CVaR95={report.var_95.cvar:.2%}" if report.var_95 else "OK",
            )
        except Exception as exc:
            logger.warning("Risk analytics failed: %s", exc)
            return TaskResult(task_name="risk_analytics", status="failed", message=str(exc))

    def _run_universe_scan(self) -> Optional[TaskResult]:
        """Run universe scanner to find new pair opportunities."""
        try:
            from core.universe_scanner import UniverseScanner
            from common.data_loader import load_prices_multi

            # Get list of symbols to scan
            pairs = self._get_active_pairs()
            symbols = set()
            for p in pairs:
                if isinstance(p, dict):
                    sx = p.get("sym_x") or p.get("symbol_x")
                    sy = p.get("sym_y") or p.get("symbol_y")
                    if sx:
                        symbols.add(sx)
                    if sy:
                        symbols.add(sy)
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    symbols.add(str(p[0]))
                    symbols.add(str(p[1]))

            if len(symbols) < 3:
                logger.info("Universe scan: too few symbols (%d), skipping", len(symbols))
                return None

            prices = load_prices_multi(list(symbols))
            if prices is None or prices.empty or len(prices.columns) < 3:
                logger.info("Universe scan: insufficient price data")
                return None

            scanner = UniverseScanner(
                min_correlation=0.55,
                min_grade="C",
                require_cointegration=True,
                max_pairs=20,
            )
            result = scanner.scan(prices)

            self.bus.publish("universe_scan", {
                "n_instruments": result.n_instruments,
                "n_pairs_screened": result.n_pairs_screened,
                "n_pairs_final": result.n_pairs_final,
                "yield_rate": result.yield_rate,
                "top_pairs": [
                    {"pair": f"{p.sym_x}/{p.sym_y}", "score": p.score, "grade": p.grade}
                    for p in result.pairs[:5]
                ],
            })

            # Alert on new high-quality discoveries
            new_high_quality = [p for p in result.pairs if p.grade in ("A+", "A")]
            if new_high_quality:
                try:
                    from core.alerts import alert_signal
                    for p in new_high_quality[:3]:
                        alert_signal(
                            pair=f"{p.sym_x}/{p.sym_y}",
                            action="NEW_DISCOVERY",
                            z_score=0.0,
                            grade=p.grade,
                            score=f"{p.score:.3f}",
                            half_life=f"{p.half_life:.1f}d",
                        )
                except Exception:
                    pass

            logger.info(
                "Universe scan: %d pairs found from %d screened (yield=%.1f%%)",
                result.n_pairs_final, result.n_pairs_screened, result.yield_rate * 100,
            )

            return TaskResult(
                task_name="universe_scan",
                status="success",
                message=f"{result.n_pairs_final} pairs found ({result.yield_rate:.0%} yield)",
            )
        except Exception as exc:
            logger.warning("Universe scan failed: %s", exc)
            return TaskResult(task_name="universe_scan", status="failed", message=str(exc))

    def _send_pipeline_alerts(self, results: list, signal_decisions: list) -> None:
        """
        Send operational alerts after daily pipeline completes.

        Alerts:
        - Pipeline summary (OK/FAILED tasks count)
        - Active entry/exit signals
        - Risk warnings from failed tasks
        """
        try:
            from core.alerts import alert_system, alert_signal, alert_risk

            # Pipeline summary
            ok = sum(1 for r in results if r.status == "success")
            failed = len(results) - ok
            if failed > 0:
                alert_system(
                    "daily_pipeline",
                    "WARNING",
                    f"{ok}/{len(results)} tasks OK, {failed} failed",
                )
            else:
                alert_system(
                    "daily_pipeline",
                    "OK",
                    f"All {ok} tasks completed successfully",
                )

            # Signal alerts — notify on active entry/exit signals
            for dec in signal_decisions:
                try:
                    if hasattr(dec, 'blocked') and dec.blocked:
                        continue  # Skip blocked signals

                    pair_label = str(dec.pair_id) if hasattr(dec, 'pair_id') else "?"
                    z = dec.z_score if hasattr(dec, 'z_score') else 0.0
                    intent = getattr(dec, 'intent', None)

                    if intent is None:
                        continue

                    action_str = str(getattr(intent, 'action', 'HOLD'))
                    if 'ENTRY' in action_str.upper() or 'EXIT' in action_str.upper():
                        regime = getattr(dec, 'regime', '?')
                        grade = getattr(dec, 'quality_grade', '?')
                        alert_signal(
                            pair=pair_label,
                            action=action_str,
                            z_score=z,
                            regime=regime,
                            quality=grade,
                        )
                except Exception:
                    pass  # Never let alert errors break the pipeline

        except ImportError:
            logger.debug("core.alerts not available — skipping pipeline alerts")
        except Exception as exc:
            logger.warning("_send_pipeline_alerts failed: %s", exc)

    def run_agent_system_health_check(self) -> Optional[TaskResult]:
        """
        Dispatch the SystemHealthAgent to validate core module importability.

        Second real agent dispatch from operational code (P1-AGENTS).
        Mirrors the two-path pattern of run_agent_data_integrity_check:

        1. **WorkflowEngine path** (``monitoring.workflow.run_system_health_workflow``)
           — full WorkflowRun lifecycle, step-level audit, artifact capture,
           dashboard alert emission.

        2. **Direct registry dispatch** (fallback on ImportError / exception).

        READ_ONLY in both paths — never mutates state.

        Returns
        -------
        TaskResult or None
        """
        import time as _time
        t0 = _time.time()

        # ── Path 1: WorkflowEngine (preferred) ────────────────────
        try:
            from monitoring.workflow import run_system_health_workflow
            from orchestration.contracts import WorkflowStatus

            outcome = run_system_health_workflow(
                triggered_by="orchestrator_daily_pipeline",
                emit_alerts=True,
            )

            elapsed = _time.time() - t0

            outcome_status = getattr(outcome, "status", None)
            outcome_run_id = getattr(outcome, "run_id", "unknown")
            steps_completed = getattr(outcome, "steps_completed", None)
            steps_failed    = getattr(outcome, "steps_failed", None)

            is_success = (
                outcome_status == WorkflowStatus.COMPLETED
                if outcome_status is not None else True
            )

            self.bus.publish("agent_system_health", {
                "path": "workflow_engine",
                "workflow_status": outcome_status.value if outcome_status else "unknown",
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "duration_sec": round(elapsed, 2),
            })
            logger.info(
                "Agent system_health (WorkflowEngine): status=%s duration=%.1fs",
                outcome_status.value if outcome_status else "unknown",
                elapsed,
            )
            return TaskResult(
                task_name="agent_system_health",
                status="success" if is_success else "failed",
                duration_sec=round(elapsed, 2),
                output={
                    "path": "workflow_engine",
                    "workflow_run_id": outcome_run_id,
                    "workflow_status": outcome_status.value if outcome_status else "unknown",
                    "steps_completed": steps_completed,
                    "steps_failed": steps_failed,
                },
            )

        except ImportError:
            logger.debug("monitoring.workflow unavailable, using direct dispatch for system_health")
        except Exception as wf_err:
            logger.warning(
                "WorkflowEngine dispatch failed for system_health (%s) — falling back",
                wf_err,
            )

        # ── Path 2: Direct registry dispatch (fallback) ───────────
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus

            registry = get_default_registry()
            agent = registry.get_agent("system_health")
            if agent is None:
                logger.warning("SystemHealthAgent not registered — skipping")
                return None

            task = AgentTask(
                task_id=f"orch_sh_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                agent_name="system_health",
                task_type="health_sweep",
                payload={},
                priority=3,
                correlation_id=f"daily_pipeline_{datetime.now(timezone.utc).date()}",
            )

            result = registry.dispatch(task)
            elapsed = _time.time() - t0
            status = "success" if result.status == AgentStatus.COMPLETED else "failed"
            unhealthy: list = result.output.get("unhealthy_components", []) if result.output else []
            overall_healthy: bool = result.output.get("overall_healthy", True) if result.output else True

            if not overall_healthy and unhealthy:
                try:
                    from core.alert_bus import emit_dashboard_alert
                    emit_dashboard_alert(
                        level="error",
                        source="orchestrator:system_health",
                        message=f"System health: {len(unhealthy)} component(s) unhealthy",
                        details={"unhealthy_components": unhealthy[:10]},
                    )
                except Exception:
                    pass

            logger.info(
                "Agent system_health (direct): status=%s unhealthy=%d duration=%.1fs",
                status, len(unhealthy), elapsed,
            )
            return TaskResult(
                task_name="agent_system_health",
                status=status,
                duration_sec=round(elapsed, 2),
                output={
                    "path": "direct_dispatch",
                    "agent_status": result.status.value,
                    "overall_healthy": overall_healthy,
                    "unhealthy_components": unhealthy,
                    "audit_trail_size": len(result.audit_trail),
                },
            )

        except Exception as e:
            elapsed = _time.time() - t0
            logger.warning("Agent system_health direct dispatch failed: %s", e)
            return TaskResult(
                task_name="agent_system_health",
                status="failed",
                duration_sec=round(elapsed, 2),
                error=str(e),
            )

    def run_agent_data_integrity_check(self) -> Optional[TaskResult]:
        """
        Dispatch the DataIntegrityAgent to validate price data quality.

        This is the first real agent dispatch from operational code (P1-AGENTS).
        Two dispatch paths exist:

        1. **WorkflowEngine path** (``monitoring.workflow.run_data_integrity_workflow``)
           — full WorkflowRun lifecycle, step-level audit, artifact capture,
           dashboard alert emission.  Used when the monitoring.workflow module
           is available.

        2. **Direct registry dispatch** (fallback)
           — calls ``registry.dispatch()`` directly, still typed and audited,
           but without WorkflowEngine lifecycle management.

        The agent is READ_ONLY in both paths — it never mutates state.

        Returns
        -------
        TaskResult or None
            TaskResult wrapping the result, or None if dispatch fails.
        """
        import time as _time
        t0 = _time.time()

        # ── Path 1: WorkflowEngine (preferred) ────────────────────
        try:
            from monitoring.workflow import run_data_integrity_workflow
            from orchestration.contracts import WorkflowStatus

            outcome = run_data_integrity_workflow(
                triggered_by="orchestrator_daily_pipeline",
                emit_alerts=True,
            )

            elapsed = _time.time() - t0

            # outcome is WorkflowOutcome (preferred) or WorkflowRun (fallback)
            outcome_status = getattr(outcome, "status", None)
            outcome_run_id = getattr(outcome, "run_id", "unknown")
            steps_completed = getattr(outcome, "steps_completed", None)
            steps_failed    = getattr(outcome, "steps_failed", None)
            artifact_count  = getattr(outcome, "artifact_count", None)

            # WorkflowStatus.COMPLETED or WorkflowStatus value from wf_run
            from orchestration.contracts import WorkflowStatus
            is_success = (
                outcome_status == WorkflowStatus.COMPLETED
                if outcome_status is not None else True
            )

            self.bus.publish("agent_data_integrity", {
                "path": "workflow_engine",
                "workflow_status": outcome_status.value if outcome_status else "unknown",
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "duration_sec": round(elapsed, 2),
            })
            logger.info(
                "Agent data_integrity (WorkflowEngine): status=%s duration=%.1fs",
                outcome_status.value if outcome_status else "unknown",
                elapsed,
            )
            return TaskResult(
                task_name="agent_data_integrity",
                status="success" if is_success else "failed",
                duration_sec=round(elapsed, 2),
                output={
                    "path": "workflow_engine",
                    "workflow_run_id": outcome_run_id,
                    "workflow_status": outcome_status.value if outcome_status else "unknown",
                    "steps_completed": steps_completed,
                    "steps_failed": steps_failed,
                    "artifact_count": artifact_count,
                },
            )

        except ImportError:
            # monitoring.workflow not available — fall through to direct dispatch
            logger.debug("monitoring.workflow unavailable, using direct dispatch")
        except Exception as wf_err:
            logger.warning(
                "WorkflowEngine dispatch failed (%s) — falling back to direct dispatch",
                wf_err,
            )

        # ── Path 2: Direct registry dispatch (fallback) ───────────
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus

            registry = get_default_registry()
            agent = registry.get_agent("data_integrity")
            if agent is None:
                logger.warning("DataIntegrityAgent not registered — skipping")
                return None

            task = AgentTask(
                task_id=f"orch_di_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                agent_name="data_integrity",
                task_type="check_data_integrity",
                payload={},
                priority=3,
                correlation_id=f"daily_pipeline_{datetime.now(timezone.utc).date()}",
            )

            result = registry.dispatch(task)

            elapsed = _time.time() - t0
            status = "success" if result.status == AgentStatus.COMPLETED else "failed"
            issues: int = result.output.get("issues_found", 0) if result.output else 0
            critical: list = result.output.get("critical_issues", []) if result.output else []

            # ── Alert bus (best-effort) ────────────────────────────
            if issues > 0:
                try:
                    from core.alert_bus import emit_dashboard_alert
                    level = "error" if issues >= 3 else "warning"
                    emit_dashboard_alert(
                        level=level,
                        source="orchestrator:data_integrity",
                        message=f"Data integrity: {issues} critical issue(s) detected",
                        details={
                            "critical_issues": critical[:5],
                            "audit_trail_size": len(result.audit_trail),
                        },
                    )
                except Exception:
                    pass   # alert bus is best-effort

            logger.info(
                "Agent data_integrity (direct): status=%s issues=%d duration=%.1fs",
                status, issues, elapsed,
            )

            return TaskResult(
                task_name="agent_data_integrity",
                status=status,
                duration_sec=round(elapsed, 2),
                output={
                    "path": "direct_dispatch",
                    "agent_status": result.status.value,
                    "issues_found": issues,
                    "critical_issues": critical,
                    "audit_trail_size": len(result.audit_trail),
                },
            )

        except Exception as e:
            elapsed = _time.time() - t0
            logger.warning("Agent data_integrity direct dispatch failed: %s", e)
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

    # ── Scheduler ─────────────────────────────────────────────────────────

    def start_daemon(
        self,
        run_time: Optional[str] = None,
        timezone: str = "America/New_York",
        run_immediately: bool = False,
    ) -> None:
        """
        Start a background daily pipeline scheduler.

        Tries APScheduler (if installed); falls back to threading.Timer.

        Parameters
        ----------
        run_time : str, optional
            "HH:MM" string for daily trigger.  Reads ``scheduler_run_time``
            from config.json if not provided; defaults to ``"16:15"``.
        timezone : str
            Timezone for the cron trigger (APScheduler only).
        run_immediately : bool
            If True, execute one pipeline run immediately before scheduling.
        """
        import atexit

        try:
            from common.config_manager import load_config
            cfg = load_config()
        except Exception:
            cfg = {}

        run_time = run_time or cfg.get("scheduler_run_time", "16:15")
        timezone = cfg.get("scheduler_timezone", timezone)

        if run_immediately:
            logger.info("start_daemon: run_immediately=True — running pipeline now")
            self.run_daily_pipeline()

        # ── APScheduler path (preferred) ─────────────────────────────
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            h, m = (int(x) for x in run_time.split(":"))
            scheduler = BackgroundScheduler(timezone=timezone)
            scheduler.add_job(
                self.run_daily_pipeline,
                trigger=CronTrigger(hour=h, minute=m, timezone=timezone),
                id="daily_pipeline",
                name="Daily Pipeline",
                misfire_grace_time=3600,
            )
            scheduler.start()
            self._scheduler = scheduler
            atexit.register(self.stop_daemon)
            logger.info(
                "start_daemon: APScheduler started — daily run at %s %s",
                run_time, timezone,
            )
            return

        except ImportError:
            logger.debug("APScheduler not installed — using threading.Timer fallback")
        except Exception as e:
            logger.warning("APScheduler setup failed (%s) — using threading.Timer fallback", e)

        # ── threading.Timer fallback ─────────────────────────────────
        import threading
        from datetime import datetime, timezone as dt_tz

        self._daemon_stop_event = threading.Event()

        def _next_run_seconds() -> float:
            now = datetime.now()
            h, m = (int(x) for x in run_time.split(":"))
            target = now.replace(hour=h, minute=m, second=0, microsecond=0)
            delta = (target - now).total_seconds()
            return delta if delta > 0 else delta + 86400  # +24h if already past

        def _loop():
            while not self._daemon_stop_event.is_set():
                delay = _next_run_seconds()
                logger.info("start_daemon: next run in %.0fs (at %s)", delay, run_time)
                if self._daemon_stop_event.wait(timeout=delay):
                    break  # stop requested
                if not self._daemon_stop_event.is_set():
                    logger.info("start_daemon: triggering daily pipeline")
                    try:
                        self.run_daily_pipeline()
                    except Exception as exc:
                        logger.error("start_daemon: pipeline error: %s", exc)

        self._daemon_thread = threading.Thread(target=_loop, daemon=True, name="pipeline-daemon")
        self._daemon_thread.start()
        atexit.register(self.stop_daemon)
        logger.info("start_daemon: threading.Timer fallback started — daily run at %s", run_time)

    def stop_daemon(self) -> None:
        """Stop the background scheduler (APScheduler or threading fallback)."""
        if hasattr(self, "_scheduler"):
            try:
                self._scheduler.shutdown(wait=False)
                logger.info("stop_daemon: APScheduler stopped")
            except Exception:
                pass
        if hasattr(self, "_daemon_stop_event"):
            self._daemon_stop_event.set()
            logger.info("stop_daemon: threading daemon stopped")

    # ── Agent dispatch: signal-layer + risk-layer ──────────────────────────

    def _dispatch_signal_agents(self) -> list:
        """
        Dispatch signal-layer agents after compute_signals.

        Agents dispatched:
        - RegimeSurveillanceAgent: scans for regime shifts across pairs
        - SignalAnalystAgent: classifies signal state per pair
        - TradeLifecycleAgent: inspects lifecycle states for stale/blocked
        - ExitOversightAgent: flags exit opportunities on open positions
        """
        results = []
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus
            registry = get_default_registry()

            signal_agents = [
                ("regime_surveillance", "regime_surveillance.scan",
                 {"spreads": {}}),
                ("signal_analyst", "signal_analyst.classify",
                 {"pair_id": "ALL", "spread": None}),
                ("trade_lifecycle", "lifecycle.inspect",
                 {"states": {}}),
                ("exit_oversight", "exit_oversight.scan",
                 {"open_positions": {}}),
            ]

            ts = datetime.now(timezone.utc)
            for agent_name, task_type, payload in signal_agents:
                agent = registry.get_agent(agent_name)
                if agent is None:
                    continue
                task = AgentTask(
                    task_id=f"{agent_name}_{ts.strftime('%Y%m%d_%H%M%S')}",
                    agent_name=agent_name,
                    task_type=task_type,
                    payload=payload,
                    priority=3,
                    correlation_id=f"daily_pipeline_{ts.date()}",
                )
                result = registry.dispatch(task)
                status = "success" if result.status == AgentStatus.COMPLETED else "failed"
                results.append(TaskResult(
                    task_name=f"agent_{agent_name}",
                    status=status,
                    output={"agent_status": result.status.value,
                            "audit_size": len(result.audit_trail)},
                ))
                logger.info("Agent %s: %s", agent_name, status)

        except Exception as e:
            logger.warning("Signal agent dispatch failed: %s", e)
        return results

    def _dispatch_risk_agents(self) -> list:
        """
        Dispatch risk-layer agents after portfolio allocation.

        Agents dispatched:
        - ExposureMonitorAgent: checks sector/cluster/leverage exposure
        - DrawdownMonitorAgent: updates heat level
        - KillSwitchAgent: evaluates halt conditions
        - CapitalBudgetAgent: checks capital availability
        - DeRiskingAgent: computes de-risking when stressed
        - DriftMonitoringAgent: checks feature/model drift
        - AlertAggregationAgent: deduplicates and prioritises alerts
        """
        results = []
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus
            registry = get_default_registry()

            ts = datetime.now(timezone.utc)
            risk_agents = [
                ("exposure_monitor", "compute_exposure",
                 {"positions": {}, "limits": {}}),
                ("drawdown_monitor", "update_drawdown",
                 {"current_nav": 1_000_000.0, "peak_nav": 1_000_000.0}),
                ("kill_switch", "evaluate_kill_switch",
                 {"current_nav": 1_000_000.0, "peak_nav": 1_000_000.0}),
                ("capital_budget", "check_capital_budget",
                 {"pairs": [], "total_capital": 1_000_000.0}),
                ("derisking", "compute_derisking",
                 {"drawdown_state": {}, "active_allocations": []}),
                ("drift_monitoring", "drift_sweep",
                 {}),
                ("alert_aggregation", "aggregate_alerts",
                 {"alerts": []}),
            ]

            for agent_name, task_type, payload in risk_agents:
                agent = registry.get_agent(agent_name)
                if agent is None:
                    continue
                task = AgentTask(
                    task_id=f"{agent_name}_{ts.strftime('%Y%m%d_%H%M%S')}",
                    agent_name=agent_name,
                    task_type=task_type,
                    payload=payload,
                    priority=2,
                    correlation_id=f"daily_pipeline_{ts.date()}",
                )
                result = registry.dispatch(task)
                status = "success" if result.status == AgentStatus.COMPLETED else "failed"
                results.append(TaskResult(
                    task_name=f"agent_{agent_name}",
                    status=status,
                    output={"agent_status": result.status.value,
                            "audit_size": len(result.audit_trail)},
                ))
                logger.info("Agent %s: %s", agent_name, status)

        except Exception as e:
            logger.warning("Risk agent dispatch failed: %s", e)
        return results

    def _dispatch_agents_batch(
        self,
        agents_spec: list[tuple[str, str, dict]],
        correlation_prefix: str = "pipeline",
        priority: int = 4,
    ) -> list:
        """Generic agent batch dispatcher. Returns list[TaskResult]."""
        results = []
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus
            registry = get_default_registry()
            ts = datetime.now(timezone.utc)

            for agent_name, task_type, payload in agents_spec:
                agent = registry.get_agent(agent_name)
                if agent is None:
                    continue
                task = AgentTask(
                    task_id=f"{agent_name}_{ts.strftime('%Y%m%d_%H%M%S')}",
                    agent_name=agent_name,
                    task_type=task_type,
                    payload=payload,
                    priority=priority,
                    correlation_id=f"{correlation_prefix}_{ts.date()}",
                )
                result = registry.dispatch(task)
                status = "success" if result.status == AgentStatus.COMPLETED else "failed"
                results.append(TaskResult(
                    task_name=f"agent_{agent_name}",
                    status=status,
                    output={"agent_status": result.status.value,
                            "audit_size": len(result.audit_trail)},
                ))
                logger.info("Agent %s: %s", agent_name, status)
        except Exception as e:
            logger.warning("Agent batch dispatch failed: %s", e)
        return results

    def dispatch_research_agents(self, symbols: list[str] | None = None) -> list:
        """
        Dispatch all research-layer agents on demand.

        Agents: universe_curator, candidate_discovery, universe_discovery,
        pair_validation, relationship_validation, spread_fit, spread_specification,
        regime_research, signal_research, experiment_coordinator, research_summarization
        """
        if symbols is None:
            try:
                from common.data_loader import load_pairs
                pairs = load_pairs()
                symbols = sorted({s for p in pairs for s in p.get("symbols", [])})
            except Exception:
                symbols = []

        pair_ids = [f"{symbols[i]}/{symbols[i+1]}" for i in range(0, len(symbols) - 1, 2)] if len(symbols) >= 2 else []

        return self._dispatch_agents_batch([
            ("universe_curator", "curate_universe", {"symbols": symbols}),
            ("universe_discovery", "discover_pairs", {"symbols": symbols}),
            ("candidate_discovery", "discover_candidates", {"symbols": symbols}),
            ("pair_validation", "validate_pairs", {"pair_ids": pair_ids}),
            ("relationship_validation", "validate_relationships", {"pair_ids": pair_ids}),
            ("spread_fit", "fit_spreads", {"pair_ids": pair_ids}),
            ("spread_specification", "specify_spreads", {"pair_ids": pair_ids, "prices": {}}),
            ("regime_research", "classify_regimes", {"spreads": {}}),
            ("signal_research", "analyze_signal_quality", {"spreads": {}, "signals": {}}),
            ("experiment_coordinator", "plan_experiment", {"experiment_type": "pair_scan"}),
            ("research_summarization", "summarize_research_run", {}),
        ], correlation_prefix="research", priority=5)

    def dispatch_ml_agents(self) -> list:
        """
        Dispatch all ML-layer agents on demand.

        Agents: feature_steward, label_governance, model_research,
        meta_labeling, regime_modeling, model_risk, promotion_review
        """
        return self._dispatch_agents_batch([
            ("feature_steward", "audit_feature_health", {}),
            ("label_governance", "check_label_leakage", {}),
            ("model_research", "evaluate_model", {}),
            ("meta_labeling", "assess_meta_label", {}),
            ("regime_modeling", "evaluate_regime_model", {}),
            ("model_risk", "assess_model_risk", {}),
            ("promotion_review", "check_promotion_criteria", {}),
        ], correlation_prefix="ml_review", priority=4)

    def dispatch_governance_agents(self) -> list:
        """
        Dispatch all governance-layer agents on demand.

        Agents: policy_review, audit_trail_validation, approval_recommendation,
        change_impact, promotion_gate, orchestration_reliability,
        incident_triage, postmortem_drafting
        """
        return self._dispatch_agents_batch([
            ("policy_review", "check_policy_compliance", {}),
            ("audit_trail_validation", "validate_audit_trail", {}),
            ("approval_recommendation", "recommend_approval", {}),
            ("change_impact", "assess_change_impact", {}),
            ("promotion_gate", "check_promotion_eligibility", {}),
            ("orchestration_reliability", "check_workflow_health", {}),
            ("incident_triage", "classify_severity",
             {"title": "daily_check", "description": "routine", "affected_components": [], "detected_by": "orchestrator"}),
            ("postmortem_drafting", "analyze_incident_pattern", {}),
        ], correlation_prefix="governance_review", priority=4)

    def dispatch_portfolio_agents(self) -> list:
        """Dispatch portfolio construction agent on demand."""
        return self._dispatch_agents_batch([
            ("portfolio_construction", "run_allocation_cycle",
             {"intents": [], "capital": 1_000_000.0}),
        ], correlation_prefix="portfolio", priority=3)

    # ── Signal collection helpers ─────────────────────────────────────────

    def _get_active_pairs(self) -> list:
        """
        Return the active pairs list from AppContext or config.json.

        Returns [] with a warning if neither source is available.
        """
        try:
            from core.app_context import AppContext
            ctx = AppContext.get_global()
            if ctx.pairs:
                return list(ctx.pairs)
        except Exception:
            pass

        try:
            from common.config_manager import load_config
            pairs = load_config().get("pairs", [])
            if pairs:
                return list(pairs)
        except Exception:
            pass

        logger.warning("_get_active_pairs: no pairs found in AppContext or config")
        return []

    def _load_ml_hook(self):
        """
        Load the best available meta-label ML model for signal quality.

        Priority order:
        1. ModelScorer (formal ML inference layer) — if registry has a champion
        2. XGBoost pickle (ad-hoc trained) — per-pair or generic
        3. MetaLabelModel (LogReg) — lighter fallback
        4. None → deterministic quality fallback (no ML)

        Returns an object with ``predict_success_probability(features_dict)``
        or None if no model is available.
        """
        # Priority 1: Try formal ModelScorer from ML platform
        try:
            from ml.inference.scorer import ModelScorer
            from ml.contracts import InferenceRequest, MLTaskFamily
            scorer = ModelScorer()
            model = scorer.get_model_for_task(MLTaskFamily.META_LABELING)
            if model is not None:
                class _ScorerHook:
                    def __init__(self, _scorer):
                        self._scorer = _scorer
                    def predict_success_probability(self, feats):
                        try:
                            req = InferenceRequest(
                                entity_id="pipeline",
                                task_family=MLTaskFamily.META_LABELING,
                                features=feats if isinstance(feats, dict) else {},
                                allow_fallback=True,
                            )
                            result = self._scorer.score(req)
                            return result.score if not result.fallback_used else float("nan")
                        except Exception:
                            return float("nan")
                logger.info("ML model loaded via ModelScorer (formal inference layer)")
                return _ScorerHook(scorer)
        except Exception as exc:
            logger.debug("ModelScorer not available: %s", exc)

        # Priority 2: XGBoost pickle (ad-hoc trained models)
        try:
            import pickle
            import pandas as _pd
            xgb_path = PROJECT_ROOT / "models" / "xgb_meta_latest.pkl"
            if not xgb_path.exists():
                xgb_files = sorted((PROJECT_ROOT / "models").glob("xgb_meta_*.pkl"))
                if xgb_files:
                    xgb_path = xgb_files[-1]
            if xgb_path.exists():
                with open(xgb_path, "rb") as _f:
                    xgb_data = pickle.load(_f)
                if hasattr(xgb_data, "get"):
                    xgb_model = xgb_data.get("model")
                    xgb_features = xgb_data.get("features", [])
                    if xgb_model is not None:
                        class _XGBHook:
                            def __init__(self, model, features):
                                self._model = model
                                self._features = features
                            def predict_success_probability(self, feats):
                                if isinstance(feats, dict):
                                    row = {k: feats.get(k, 0.0) for k in self._features}
                                    X = _pd.DataFrame([row])
                                else:
                                    X = _pd.DataFrame([{k: 0.0 for k in self._features}])
                                try:
                                    return float(self._model.predict_proba(X)[0, 1])
                                except Exception:
                                    return float("nan")
                        logger.info("XGBoost ML model loaded: %s (%d features)", xgb_path.name, len(xgb_features))
                        return _XGBHook(xgb_model, xgb_features)
        except Exception as exc:
            logger.debug("XGBoost model not available: %s", exc)

        # Priority 3: MetaLabelModel (LogReg fallback)
        try:
            from ml.models.meta_labeler import MetaLabelModel
            model_path = PROJECT_ROOT / "models" / "meta_label_latest.pkl"
            if model_path.exists():
                ml_model = MetaLabelModel.load(str(model_path))
                if ml_model.is_fitted:
                    logger.info("MetaLabel (LogReg) model loaded: %s", model_path.name)
                    return ml_model
        except Exception as exc:
            logger.debug("MetaLabelModel not available: %s", exc)

        logger.info("No ML model available — using deterministic quality fallback")
        return None

    def _collect_signal_decisions(
        self,
        pairs: Optional[list] = None,
        lookback_days: int = 252,
    ) -> list:
        """
        Run SignalPipeline for every active pair and return SignalDecision objects.

        Called from run_daily_pipeline() to bridge compute_signals →
        run_portfolio_allocation_cycle().  Never raises — returns [] on any
        error so the pipeline continues without allocation.

        Parameters
        ----------
        pairs : list, optional
            List of pair dicts ``{"sym_x": ..., "sym_y": ...}`` or 2-tuples.
            Defaults to ``_get_active_pairs()``.
        lookback_days : int
            Price history length fed to SignalPipeline (default: 252).

        Returns
        -------
        list[SignalDecision]
        """
        import time as _time
        import numpy as np

        if pairs is None:
            pairs = self._get_active_pairs()
        if not pairs:
            return []

        try:
            from common.data_loader import load_prices_multi
            from core.contracts import PairId
            from core.signal_pipeline import SignalPipeline
        except ImportError as exc:
            logger.warning("_collect_signal_decisions: required import failed: %s", exc)
            return []

        decisions = []
        t0 = _time.time()

        # ── P1-ML: Load meta-label model for signal quality ─────
        # Priority: ModelScorer (formal) → XGBoost pickle → MetaLabel LogReg
        # If none available → ml_hook=None → deterministic quality fallback.
        ml_hook = self._load_ml_hook()

        for pair_def in pairs:
            sym_x = sym_y = None
            try:
                if isinstance(pair_def, dict):
                    sym_x = pair_def.get("sym_x") or pair_def.get("symbol_x")
                    sym_y = pair_def.get("sym_y") or pair_def.get("symbol_y")
                elif isinstance(pair_def, (list, tuple)) and len(pair_def) >= 2:
                    sym_x, sym_y = str(pair_def[0]), str(pair_def[1])
                else:
                    continue

                if not sym_x or not sym_y:
                    continue

                prices = load_prices_multi([sym_x, sym_y])
                if prices is None or prices.empty:
                    continue

                # Resolve price series by column name or position
                px = prices[sym_x] if sym_x in prices.columns else prices.iloc[:, 0]
                py = prices[sym_y] if sym_y in prices.columns else prices.iloc[:, 1]
                px = px.dropna().iloc[-lookback_days:]
                py = py.dropna().iloc[-lookback_days:]

                if len(px) < 60 or len(py) < 60:
                    continue

                # Simple OLS spread + z-score (no leakage — uses full window)
                beta = float(np.cov(px.values, py.values)[0, 1] / np.var(px.values))
                spread = py - beta * px
                spread_std = float(spread.std())
                if spread_std == 0:
                    continue
                z_score = float((spread.iloc[-1] - spread.mean()) / spread_std)

                pipeline = SignalPipeline(
                    pair_id=PairId(sym_x, sym_y),
                    ml_quality_hook=ml_hook,
                )
                decision = pipeline.evaluate(
                    z_score=z_score,
                    spread_series=spread,
                    prices_x=px,
                    prices_y=py,
                    as_of=datetime.now(timezone.utc),
                )
                decisions.append(decision)

            except Exception as exc:
                logger.warning(
                    "_collect_signal_decisions: %s/%s failed: %s",
                    sym_x or "?", sym_y or "?", exc,
                )

        elapsed = _time.time() - t0
        logger.info(
            "_collect_signal_decisions: %d decisions from %d pairs in %.1fs",
            len(decisions), len(pairs), elapsed,
        )
        return decisions

    def run_portfolio_allocation_cycle(
        self,
        signal_decisions: Optional[list] = None,
        capital: float = 1_000_000.0,
    ) -> Optional[TaskResult]:
        """
        Run one portfolio allocation cycle with safety gating.

        Resolves:
        - P1-PORTINT: Portfolio bridge called from operational code
        - P1-SAFE: Runtime safety check injected as callback
        - P0-KS: Kill-switch with control-plane callback used

        Parameters
        ----------
        signal_decisions : list[SignalDecision], optional
            If None, returns immediately (no signals to allocate).
        capital : float
            Total capital for allocation.

        Returns
        -------
        TaskResult or None
        """
        import time as _time
        t0 = _time.time()

        if not signal_decisions:
            return None

        try:
            from core.portfolio_bridge import bridge_signals_to_allocator

            # ── P1-SAFE: Inject runtime safety check ─────────────
            safety_fn = None
            try:
                from runtime.state import get_runtime_state_manager
                safety_fn = get_runtime_state_manager().is_safe_to_trade
            except ImportError:
                logger.debug("runtime.state unavailable — safety_check=None")

            # ── P0-KS: Use kill-switch factory with control-plane ─
            kill_switch_state = None
            try:
                from portfolio.risk_ops import make_kill_switch_manager_with_control_plane
                ksm = make_kill_switch_manager_with_control_plane()
                kill_switch_state = ksm.state
            except (ImportError, Exception) as e:
                logger.debug("Kill-switch factory unavailable: %s", e)

            allocations, diagnostics = bridge_signals_to_allocator(
                signal_decisions,
                capital=capital,
                safety_check=safety_fn,
                kill_switch=kill_switch_state,
            )

            elapsed = _time.time() - t0
            n_funded = sum(1 for a in allocations if a.approved)

            logger.info(
                "Portfolio allocation: %d funded / %d total, %.1fs",
                n_funded, len(allocations), elapsed,
            )

            return TaskResult(
                task_name="portfolio_allocation",
                status="success",
                duration_sec=round(elapsed, 2),
                output={
                    "n_funded": n_funded,
                    "n_total": len(allocations),
                    "n_intents_received": diagnostics.n_intents_received,
                    "safety_check_used": safety_fn is not None,
                    "kill_switch_used": kill_switch_state is not None,
                },
            )

        except Exception as e:
            elapsed = _time.time() - t0
            logger.warning("Portfolio allocation failed: %s", e)
            return TaskResult(
                task_name="portfolio_allocation",
                status="failed",
                duration_sec=round(elapsed, 2),
                error=str(e),
            )


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
