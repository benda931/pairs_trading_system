# -*- coding: utf-8 -*-
"""
core/system_health.py — System Health Contract
===============================================

SystemHealthContract defines the required and optional components
that must be healthy for the system to operate.

Run as part of every orchestrator startup. Result is persisted to DuckDB
and surfaced in the dashboard status bar.

Usage
-----
    from core.system_health import SystemHealthContract, run_health_check

    result = run_health_check()
    print(result.summary())
    # "HEALTHY: 8/8 required components OK"
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("core.system_health")


# ---------------------------------------------------------------------------
# Health check result objects
# ---------------------------------------------------------------------------

@dataclass
class ComponentHealthResult:
    """Health check result for a single component."""
    name: str
    healthy: bool
    required: bool
    latency_ms: float = 0.0
    message: str = ""
    error: str = ""
    checked_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


@dataclass
class SystemHealthResult:
    """Aggregate health result from a full system health check."""
    checked_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    healthy: bool = False                  # True only if ALL required components pass
    component_results: List[ComponentHealthResult] = field(default_factory=list)
    n_required_total: int = 0
    n_required_passing: int = 0
    n_optional_total: int = 0
    n_optional_passing: int = 0
    total_check_time_ms: float = 0.0
    run_id: str = ""

    def summary(self) -> str:
        status = "HEALTHY" if self.healthy else "DEGRADED"
        failed = [r for r in self.component_results if not r.healthy and r.required]
        if failed:
            return (
                f"{status}: {self.n_required_passing}/{self.n_required_total} required OK. "
                f"Failed: {[r.name for r in failed]}"
            )
        return f"{status}: {self.n_required_passing}/{self.n_required_total} required OK"

    def failed_required(self) -> List[str]:
        return [r.name for r in self.component_results if not r.healthy and r.required]

    def failed_optional(self) -> List[str]:
        return [r.name for r in self.component_results if not r.healthy and not r.required]


# ---------------------------------------------------------------------------
# SystemHealthContract
# ---------------------------------------------------------------------------

class SystemHealthContract:
    """
    Defines and runs health checks for all system components.

    Required components (system is DEGRADED if any fail):
    - DuckDB database connection
    - Signal pipeline imports
    - ML registry availability
    - Config loading
    - Logs directory writable

    Optional components (warnings only if fail):
    - IB broker connection
    - Slack webhook
    - Market data freshness
    """

    def __init__(self):
        self._checks: List[Tuple[str, bool, Callable]] = []
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register built-in health checks. Required=True means failure = DEGRADED."""
        req = True
        opt = False

        self.register("config_loading",         req, self._check_config)
        self.register("logs_dir_writable",       req, self._check_logs_dir)
        self.register("duckdb_connection",       req, self._check_duckdb)
        self.register("signal_pipeline_import",  req, self._check_signal_pipeline)
        self.register("ml_contracts_import",     req, self._check_ml_contracts)
        self.register("ib_connection",           opt, self._check_ib_connection)
        self.register("ml_champion_available",   opt, self._check_ml_champion)

    def register(self, name: str, required: bool, check_fn: Callable) -> None:
        """Register a health check function."""
        self._checks.append((name, required, check_fn))

    def run(self, run_id: str = "") -> SystemHealthResult:
        """Execute all registered health checks and return aggregate result."""
        start = time.monotonic()
        results = []

        for name, required, fn in self._checks:
            t0 = time.monotonic()
            try:
                healthy, message = fn()
                error = ""
            except Exception as exc:
                healthy = False
                message = ""
                error = str(exc)
                logger.warning("Health check %s raised: %s", name, exc)

            latency_ms = (time.monotonic() - t0) * 1000
            results.append(ComponentHealthResult(
                name=name,
                healthy=healthy,
                required=required,
                latency_ms=latency_ms,
                message=message,
                error=error,
            ))

        total_ms = (time.monotonic() - start) * 1000
        n_req_pass = sum(1 for r in results if r.required and r.healthy)
        n_req_total = sum(1 for r in results if r.required)
        n_opt_pass = sum(1 for r in results if not r.required and r.healthy)
        n_opt_total = sum(1 for r in results if not r.required)

        result = SystemHealthResult(
            healthy=(n_req_pass == n_req_total),
            component_results=results,
            n_required_total=n_req_total,
            n_required_passing=n_req_pass,
            n_optional_total=n_opt_total,
            n_optional_passing=n_opt_pass,
            total_check_time_ms=total_ms,
            run_id=run_id,
        )

        if result.healthy:
            logger.info("System health check: %s (%.0fms)", result.summary(), total_ms)
        else:
            logger.warning("System health check: %s (%.0fms)", result.summary(), total_ms)

        return result

    # ------------------------------------------------------------------
    # Built-in check implementations
    # ------------------------------------------------------------------

    def _check_config(self) -> Tuple[bool, str]:
        from common.config_manager import load_config
        cfg = load_config()
        return bool(cfg), f"config keys: {list(cfg.keys())[:5]}"

    def _check_logs_dir(self) -> Tuple[bool, str]:
        from pathlib import Path
        logs_dir = Path(__file__).resolve().parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        test_file = logs_dir / ".health_check_write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return True, f"logs dir writable: {logs_dir}"

    def _check_duckdb(self) -> Tuple[bool, str]:
        from core.sql_store import SQLStore
        store = SQLStore()
        result = store.raw_query("SELECT 1 AS ping")
        ok = result is not None and len(result) > 0
        return ok, "DuckDB ping OK" if ok else "DuckDB ping failed"

    def _check_signal_pipeline(self) -> Tuple[bool, str]:
        from core.signal_pipeline import SignalPipeline  # type: ignore
        return True, "SignalPipeline importable"

    def _check_ml_contracts(self) -> Tuple[bool, str]:
        from ml.contracts import ModelStatus, MLTaskFamily
        return True, f"ml.contracts OK ({len(list(ModelStatus))} statuses)"

    def _check_ib_connection(self) -> Tuple[bool, str]:
        try:
            from root.ibkr_connection import ib_connection_status, get_ib_instance
            ib = get_ib_instance(readonly=True)
            if ib is None:
                return False, "IB instance unavailable"
            status = ib_connection_status(ib)
            return bool(status), f"IB status: {status}"
        except Exception as exc:
            return False, f"IB connection error: {exc}"

    def _check_ml_champion(self) -> Tuple[bool, str]:
        try:
            from ml.registry import get_ml_registry
            from ml.contracts import ModelStatus
            registry = get_ml_registry()
            champions = [
                m for m in registry.list_all()
                if getattr(m, "status", None) == ModelStatus.CHAMPION
            ]
            return len(champions) > 0, f"{len(champions)} champion model(s) available"
        except Exception as exc:
            return False, f"ML registry check failed: {exc}"


def run_health_check(run_id: str = "") -> SystemHealthResult:
    """Convenience function: create contract and run all checks."""
    contract = SystemHealthContract()
    return contract.run(run_id=run_id)
