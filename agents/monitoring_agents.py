# -*- coding: utf-8 -*-
"""
agents/monitoring_agents.py — Monitoring and Incident Agent Implementations
============================================================================

Seven agent classes covering system health monitoring, drift detection, data
integrity validation, orchestration reliability, incident triage, postmortem
drafting, and alert aggregation.

All agents:
  - Subclass BaseAgent (from agents.base)
  - Handle ImportError gracefully with lightweight fallbacks
  - Return a proper dict from _execute() — never None
  - Use uuid.uuid4() for generated IDs
  - Use datetime.utcnow().isoformat() + "Z" for timestamps
  - Are fully type-annotated
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from agents.base import AgentAuditLogger, BaseAgent
from core.contracts import AgentTask


def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Default core module list for health checks ────────────────────
_DEFAULT_CORE_MODULES = [
    "core.contracts",
    "core.signals_engine",
    "core.regime_engine",
    "core.risk_engine",
    "core.optimizer",
    "core.sql_store",
    "common.data_providers",
    "research.pair_validator",
    "research.spread_constructor",
    "ml.inference.scorer",
    "agents.base",
    "agents.registry",
]


# ══════════════════════════════════════════════════════════════════
# 1. SystemHealthAgent
# ══════════════════════════════════════════════════════════════════


class SystemHealthAgent(BaseAgent):
    """
    Checks importability and basic connectivity of all core system modules.

    Task types
    ----------
    check_system_health
        Check a specific list of components.
    health_sweep
        Sweep all default core modules.

    Optional payload keys
    ---------------------
    components : list[str]  (module paths; defaults to ``_DEFAULT_CORE_MODULES``)
    check_imports : bool    (default True)

    Output keys
    -----------
    component_health : dict[str, dict]
    overall_healthy : bool
    unhealthy_components : list[str]
    warnings : list[str]
    """

    NAME = "system_health"
    ALLOWED_TASK_TYPES = {"check_system_health", "health_sweep"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        import importlib
        import time as _time

        components: list[str] = task.payload.get("components") or _DEFAULT_CORE_MODULES
        check_imports: bool = bool(task.payload.get("check_imports", True))

        audit.log(f"Checking health of {len(components)} components")

        component_health: dict[str, dict[str, Any]] = {}
        unhealthy: list[str] = []
        warnings: list[str] = []

        for comp in components:
            entry: dict[str, Any] = {"healthy": False, "import_time_ms": None, "error": None}

            if check_imports:
                t0 = _time.monotonic()
                try:
                    importlib.import_module(comp)
                    elapsed_ms = (_time.monotonic() - t0) * 1000.0
                    entry["healthy"] = True
                    entry["import_time_ms"] = round(elapsed_ms, 2)
                    if elapsed_ms > 2000:
                        warnings.append(f"{comp}: slow import ({elapsed_ms:.0f}ms)")
                except ImportError as exc:
                    elapsed_ms = (_time.monotonic() - t0) * 1000.0
                    entry["import_time_ms"] = round(elapsed_ms, 2)
                    entry["error"] = f"ImportError: {exc}"
                    unhealthy.append(comp)
                    audit.warn(f"{comp}: ImportError — {exc}")
                except Exception as exc:
                    elapsed_ms = (_time.monotonic() - t0) * 1000.0
                    entry["import_time_ms"] = round(elapsed_ms, 2)
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    unhealthy.append(comp)
                    audit.warn(f"{comp}: {type(exc).__name__} — {exc}")
            else:
                entry["healthy"] = True  # cannot verify without importing

            component_health[comp] = entry

        # Optional DB connectivity check
        try:
            from core.sql_store import get_sql_store
            store = get_sql_store()
            store.ping()  # type: ignore[attr-defined]
            component_health["db:sql_store"] = {"healthy": True, "import_time_ms": None, "error": None}
        except AttributeError:
            pass  # ping not implemented — no-op
        except ImportError:
            pass  # sql_store not available — skip
        except Exception as exc:
            unhealthy.append("db:sql_store")
            component_health["db:sql_store"] = {
                "healthy": False, "import_time_ms": None, "error": str(exc)
            }
            warnings.append(f"db:sql_store connectivity check failed: {exc}")

        overall_healthy = len(unhealthy) == 0
        audit.log(
            f"System health sweep complete: "
            f"{len(component_health) - len(unhealthy)}/{len(component_health)} healthy, "
            f"unhealthy={unhealthy}"
        )

        return {
            "component_health": component_health,
            "overall_healthy": overall_healthy,
            "unhealthy_components": unhealthy,
            "warnings": warnings,
        }


# ══════════════════════════════════════════════════════════════════
# 2. DriftMonitoringAgent
# ══════════════════════════════════════════════════════════════════


class DriftMonitoringAgent(BaseAgent):
    """
    Monitors feature drift across the ML feature set.

    Attempts to use ``ml.monitoring.drift.FeatureDriftMonitor``; falls back
    to a simple mean/std shift detector.

    Task types
    ----------
    check_drift
        Check drift for a specific set of features.
    drift_sweep
        Check all provided features.

    Optional payload keys
    ---------------------
    feature_data : dict[str, list]      (current feature distributions)
    reference_data : dict[str, list]    (reference distributions)
    threshold_psi : float               (default 0.2)

    Output keys
    -----------
    drift_results : dict[str, dict]
    drifted_features : list[str]
    requires_retrain : bool
    recommendations : list[str]
    """

    NAME = "drift_monitoring"
    ALLOWED_TASK_TYPES = {"check_drift", "drift_sweep"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    @staticmethod
    def _psi_from_distributions(reference: list, current: list) -> float:
        """Compute approximate PSI from two distributions."""
        try:
            import numpy as np

            ref = np.array(reference, dtype=float)
            cur = np.array(current, dtype=float)
            ref = ref[~np.isnan(ref)]
            cur = cur[~np.isnan(cur)]
            if len(ref) < 5 or len(cur) < 5:
                return 0.0

            # Use 10 equal-width bins across combined range
            lo = min(float(ref.min()), float(cur.min()))
            hi = max(float(ref.max()), float(cur.max()))
            if abs(hi - lo) < 1e-10:
                return 0.0

            bins = np.linspace(lo, hi, 11)
            ref_counts, _ = np.histogram(ref, bins=bins)
            cur_counts, _ = np.histogram(cur, bins=bins)

            ref_pct = (ref_counts + 0.0001) / float(len(ref))
            cur_pct = (cur_counts + 0.0001) / float(len(cur))

            psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
            return max(0.0, psi)
        except Exception:
            return 0.0

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        feature_data: dict[str, list] = task.payload.get("feature_data") or {}
        reference_data: dict[str, list] = task.payload.get("reference_data") or {}
        threshold_psi: float = float(task.payload.get("threshold_psi", 0.2))

        all_features = list(set(list(feature_data.keys()) + list(reference_data.keys())))
        audit.log(
            f"Monitoring drift for {len(all_features)} features, threshold_psi={threshold_psi}"
        )

        # Try FeatureDriftMonitor
        _monitor = None
        try:
            from ml.monitoring.drift import FeatureDriftMonitor
            _monitor = FeatureDriftMonitor()
            audit.log("Using ml.monitoring.drift.FeatureDriftMonitor")
        except ImportError:
            audit.warn("FeatureDriftMonitor unavailable — using PSI fallback")

        drift_results: dict[str, dict[str, Any]] = {}
        drifted: list[str] = []

        for fname in all_features:
            ref = reference_data.get(fname, [])
            cur = feature_data.get(fname, [])

            if not ref or not cur:
                drift_results[fname] = {
                    "psi_score": None,
                    "severity": "UNKNOWN",
                    "drifted": False,
                    "note": "insufficient_data",
                }
                continue

            if _monitor is not None:
                try:
                    report = _monitor.compute_psi(reference=ref, current=cur)
                    psi = float(report.psi) if hasattr(report, "psi") else self._psi_from_distributions(ref, cur)
                except Exception:
                    psi = self._psi_from_distributions(ref, cur)
            else:
                psi = self._psi_from_distributions(ref, cur)

            if psi >= threshold_psi:
                severity = "HIGH"
                has_drifted = True
                drifted.append(fname)
            elif psi >= threshold_psi * 0.5:
                severity = "MEDIUM"
                has_drifted = False
            else:
                severity = "LOW"
                has_drifted = False

            drift_results[fname] = {
                "psi_score": round(psi, 6),
                "severity": severity,
                "drifted": has_drifted,
            }

        requires_retrain = len(drifted) > 0

        recommendations: list[str] = []
        if drifted:
            recommendations.append(
                f"{len(drifted)} features have drifted significantly: {drifted}. "
                "Schedule model retraining."
            )
        medium_drift = [f for f, r in drift_results.items() if r["severity"] == "MEDIUM"]
        if medium_drift:
            recommendations.append(
                f"{len(medium_drift)} features show moderate drift — monitor closely."
            )

        audit.log(
            f"Drift monitoring complete: {len(drifted)} drifted, requires_retrain={requires_retrain}"
        )

        return {
            "drift_results": drift_results,
            "drifted_features": drifted,
            "requires_retrain": requires_retrain,
            "recommendations": recommendations,
        }


# ══════════════════════════════════════════════════════════════════
# 3. DataIntegrityAgent
# ══════════════════════════════════════════════════════════════════


class DataIntegrityAgent(BaseAgent):
    """
    Validates price data quality for all tracked symbols.

    Checks for NaN gaps, staleness, suspicious price jumps, and missing
    symbols relative to an expected list.

    Task types
    ----------
    check_data_integrity
        Full integrity sweep across all symbols in the price DataFrame.
    validate_price_data
        Targeted validation of specific symbols.

    Optional payload keys
    ---------------------
    prices : pd.DataFrame
    symbols : list[str]
    expected_trading_days : int  (default 252)
    max_gap_days : int           (default 5)
    max_daily_move_pct : float   (default 0.20)

    Output keys
    -----------
    integrity_results : dict[str, dict]
    issues_found : int
    critical_issues : list[str]
    warnings : list[str]
    """

    NAME = "data_integrity"
    ALLOWED_TASK_TYPES = {"check_data_integrity", "validate_price_data"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        prices = task.payload.get("prices")
        symbols: list[str] = task.payload.get("symbols") or []
        expected_days: int = int(task.payload.get("expected_trading_days", 252))
        max_gap: int = int(task.payload.get("max_gap_days", 5))
        max_move: float = float(task.payload.get("max_daily_move_pct", 0.20))

        audit.log(
            f"Data integrity check: expected_days={expected_days}, "
            f"max_gap={max_gap}d, max_move={max_move:.0%}"
        )

        integrity_results: dict[str, dict[str, Any]] = {}
        critical_issues: list[str] = []
        warnings: list[str] = []

        if prices is None:
            audit.warn("No prices DataFrame provided — skipping symbol-level checks")
            if symbols:
                for sym in symbols:
                    integrity_results[sym] = {
                        "has_gaps": None,
                        "gap_details": [],
                        "stale": None,
                        "suspicious_jumps": [],
                        "n_observations": 0,
                        "issue": "no_price_data",
                    }
                    critical_issues.append(f"{sym}: no price data provided")
            return {
                "integrity_results": integrity_results,
                "issues_found": len(critical_issues),
                "critical_issues": critical_issues,
                "warnings": ["No prices DataFrame provided"],
            }

        try:
            import numpy as np
            import pandas as pd
            from datetime import date as _date

            # Determine which symbols to check
            check_syms = symbols if symbols else list(prices.columns)
            audit.log(f"Checking {len(check_syms)} symbols")

            today = _date.today()

            for sym in check_syms:
                result: dict[str, Any] = {
                    "has_gaps": False,
                    "gap_details": [],
                    "stale": False,
                    "suspicious_jumps": [],
                    "n_observations": 0,
                    "missing_from_data": False,
                }

                if sym not in prices.columns:
                    result["missing_from_data"] = True
                    result["issue"] = "symbol_not_in_prices"
                    critical_issues.append(f"{sym}: not found in price data")
                    integrity_results[sym] = result
                    continue

                series = prices[sym]
                result["n_observations"] = int(series.notna().sum())

                # Staleness check
                valid_series = series.dropna()
                if len(valid_series) > 0:
                    last_idx = valid_series.index[-1]
                    if hasattr(last_idx, "date"):
                        last_date = last_idx.date()
                    else:
                        try:
                            last_date = pd.Timestamp(last_idx).date()
                        except Exception:
                            last_date = today
                    days_stale = (today - last_date).days
                    if days_stale > 5:
                        result["stale"] = True
                        warnings.append(f"{sym}: last price is {days_stale}d old ({last_date})")

                # Gap detection: find NaN runs > max_gap consecutive periods
                is_nan = series.isna().values
                gap_details: list[str] = []
                run_start = None
                for i, nan in enumerate(is_nan):
                    if nan and run_start is None:
                        run_start = i
                    elif not nan and run_start is not None:
                        run_len = i - run_start
                        if run_len > max_gap:
                            try:
                                idx_start = series.index[run_start]
                                idx_end = series.index[i - 1]
                                gap_details.append(f"{idx_start}→{idx_end} ({run_len} periods)")
                            except Exception:
                                gap_details.append(f"periods {run_start}→{i-1} ({run_len} periods)")
                        run_start = None
                # Handle trailing NaN run
                if run_start is not None:
                    run_len = len(is_nan) - run_start
                    if run_len > max_gap:
                        gap_details.append(f"trailing_gap ({run_len} periods)")

                if gap_details:
                    result["has_gaps"] = True
                    result["gap_details"] = gap_details
                    critical_issues.append(f"{sym}: {len(gap_details)} gap(s) detected")

                # Suspicious jump detection
                rets = series.pct_change().dropna()
                jump_indices = rets[rets.abs() > max_move].index.tolist()
                if jump_indices:
                    result["suspicious_jumps"] = [str(idx) for idx in jump_indices[:10]]
                    warnings.append(
                        f"{sym}: {len(jump_indices)} price jump(s) >{max_move:.0%} detected"
                    )

                integrity_results[sym] = result

        except ImportError:
            audit.warn("pandas/numpy unavailable — cannot run integrity checks")
            for sym in symbols:
                integrity_results[sym] = {
                    "has_gaps": None,
                    "gap_details": [],
                    "stale": None,
                    "suspicious_jumps": [],
                    "n_observations": 0,
                    "issue": "pandas_unavailable",
                }
        except Exception as exc:
            audit.warn(f"Data integrity check failed: {exc}")
            critical_issues.append(f"integrity_check_error:{exc}")

        total_issues = len(critical_issues)
        audit.log(
            f"Data integrity complete: {total_issues} critical issues, "
            f"{len(warnings)} warnings"
        )

        return {
            "integrity_results": integrity_results,
            "issues_found": total_issues,
            "critical_issues": critical_issues,
            "warnings": warnings,
        }


# ══════════════════════════════════════════════════════════════════
# 4. OrchestrationReliabilityAgent
# ══════════════════════════════════════════════════════════════════


class OrchestrationReliabilityAgent(BaseAgent):
    """
    Checks the health and reliability of the orchestration engine.

    Attempts to call ``orchestration.engine.WorkflowEngine.get_health_metrics()``;
    falls back to analysing provided ``workflow_metrics`` directly.

    Task types
    ----------
    check_workflow_health
        Full reliability check including stuck workflow detection.
    audit_stuck_workflows
        Focused check for workflows that have exceeded timeout.

    Optional payload keys
    ---------------------
    workflow_metrics : dict
    timeout_threshold_minutes : float  (default 60)

    Output keys
    -----------
    workflow_health : dict
    stuck_workflows : list[str]
    high_failure_rate_workflows : list[str]
    retry_storm_detected : bool
    recommendations : list[str]
    """

    NAME = "orchestration_reliability"
    ALLOWED_TASK_TYPES = {"check_workflow_health", "audit_stuck_workflows"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        workflow_metrics: dict[str, Any] = task.payload.get("workflow_metrics") or {}
        timeout_minutes: float = float(task.payload.get("timeout_threshold_minutes", 60))

        audit.log(f"Checking orchestration reliability, timeout_threshold={timeout_minutes}min")

        # Try WorkflowEngine health metrics
        _engine_health: dict[str, Any] = {}
        try:
            from orchestration.engine import WorkflowEngine
            engine = WorkflowEngine()
            if hasattr(engine, "get_health_metrics"):
                _engine_health = engine.get_health_metrics()
                audit.log("Retrieved health metrics from WorkflowEngine")
            else:
                audit.warn("WorkflowEngine.get_health_metrics() not available")
        except (ImportError, Exception) as exc:
            audit.warn(f"WorkflowEngine unavailable ({exc}) — using provided metrics")

        # Merge engine health with provided metrics
        metrics = {**_engine_health, **workflow_metrics}

        stuck: list[str] = []
        high_failure: list[str] = []
        retry_storm = False
        recommendations: list[str] = []

        # Analyse per-workflow metrics if structured as {workflow_id: {run_time_min, failure_rate, retry_count}}
        workflow_breakdown = metrics.get("workflows") or metrics.get("per_workflow") or {}
        if isinstance(workflow_breakdown, dict):
            for wf_id, wf_stats in workflow_breakdown.items():
                if isinstance(wf_stats, dict):
                    run_time = float(wf_stats.get("run_time_minutes", 0.0))
                    failure_rate = float(wf_stats.get("failure_rate", 0.0))
                    retry_count = int(wf_stats.get("retry_count", 0))

                    if run_time > timeout_minutes:
                        stuck.append(wf_id)
                    if failure_rate > 0.20:
                        high_failure.append(wf_id)
                    if retry_count > 10:
                        retry_storm = True

        # Check top-level stuck workflows
        stuck_list = metrics.get("stuck_workflows") or []
        for wf in stuck_list:
            wf_id = wf if isinstance(wf, str) else str(wf.get("workflow_id", "unknown"))
            if wf_id not in stuck:
                stuck.append(wf_id)

        # Check global failure rate
        global_failure_rate = float(metrics.get("global_failure_rate", 0.0))
        if global_failure_rate > 0.20:
            recommendations.append(
                f"Global failure rate {global_failure_rate:.1%} exceeds 20% threshold. "
                "Investigate orchestration engine logs."
            )

        if stuck:
            recommendations.append(
                f"{len(stuck)} stuck workflow(s) detected: {stuck}. "
                "Check for deadlocks or resource exhaustion."
            )
        if high_failure:
            recommendations.append(
                f"{len(high_failure)} high-failure-rate workflow(s): {high_failure}."
            )
        if retry_storm:
            recommendations.append(
                "Retry storm detected — one or more workflows retrying excessively. "
                "Check for persistent dependency failures."
            )

        workflow_health = {
            "global_failure_rate": global_failure_rate,
            "n_workflows_checked": len(workflow_breakdown),
            "n_stuck": len(stuck),
            "n_high_failure": len(high_failure),
            "retry_storm": retry_storm,
            "engine_metrics": {k: v for k, v in metrics.items() if k not in ("workflows", "per_workflow", "stuck_workflows")},
        }

        audit.log(
            f"Orchestration reliability check complete: "
            f"stuck={stuck}, high_failure={high_failure}, retry_storm={retry_storm}"
        )

        return {
            "workflow_health": workflow_health,
            "stuck_workflows": stuck,
            "high_failure_rate_workflows": high_failure,
            "retry_storm_detected": retry_storm,
            "recommendations": recommendations,
        }


# ══════════════════════════════════════════════════════════════════
# 5. IncidentTriageAgent
# ══════════════════════════════════════════════════════════════════


class IncidentTriageAgent(BaseAgent):
    """
    Triages incoming incidents and classifies severity.

    Uses keyword matching to assign P0–P3 severity, links runbooks,
    and creates a typed ``IncidentRecord``.

    Task types
    ----------
    triage_incident
        Full triage: create IncidentRecord, link runbooks, recommend actions.
    classify_severity
        Severity classification only (no record creation).

    Required payload keys
    ---------------------
    title : str
    description : str
    affected_components : list[str]
    detected_by : str

    Optional payload keys
    ---------------------
    error_details : dict

    Output keys
    -----------
    incident_id : str
    severity : str
    affected_components : list[str]
    runbook_refs : list[str]
    immediate_actions : list[str]
    escalation_recommended : bool
    """

    NAME = "incident_triage"
    ALLOWED_TASK_TYPES = {"triage_incident", "classify_severity"}
    REQUIRED_PAYLOAD_KEYS = {"title", "description", "affected_components", "detected_by"}

    # Keyword → severity mapping (checked in order)
    _P0_KEYWORDS = {
        "halt", "kill_switch", "data_loss", "system_down", "trading_halt",
        "critical_failure", "p0", "outage", "unrecoverable",
    }
    _P1_KEYWORDS = {
        "degraded", "drift", "model_failure", "model_error", "stale_data",
        "high_latency", "signal_failure", "risk_breach", "p1",
    }
    _P2_KEYWORDS = {
        "warning", "slow", "elevated", "retry", "partial", "p2",
    }

    # Runbook registry (simplified)
    _RUNBOOK_MAP: dict[str, list[str]] = {
        "kill_switch": ["RB-001-KILL-SWITCH"],
        "halt": ["RB-001-KILL-SWITCH"],
        "data_loss": ["RB-002-DATA-RECOVERY"],
        "drift": ["RB-003-MODEL-DRIFT"],
        "model_failure": ["RB-003-MODEL-DRIFT", "RB-004-MODEL-FALLBACK"],
        "risk_breach": ["RB-005-RISK-BREACH"],
        "degraded": ["RB-006-DEGRADED-MODE"],
    }

    def _classify_severity(self, title: str, description: str) -> str:
        text = (title + " " + description).lower()
        tokens = set(text.replace("_", " ").split())
        # Also check underscore tokens
        tokens.update(text.split())
        if self._P0_KEYWORDS.intersection(tokens):
            return "P0"
        if self._P1_KEYWORDS.intersection(tokens):
            return "P1"
        if self._P2_KEYWORDS.intersection(tokens):
            return "P2"
        return "P3"

    def _find_runbooks(self, title: str, description: str) -> list[str]:
        text = (title + " " + description).lower()
        refs: list[str] = []
        for keyword, runbooks in self._RUNBOOK_MAP.items():
            if keyword in text:
                for rb in runbooks:
                    if rb not in refs:
                        refs.append(rb)
        return refs

    def _immediate_actions(self, severity: str, components: list[str]) -> list[str]:
        actions: list[str] = []
        if severity == "P0":
            actions.append("Page on-call engineer immediately")
            actions.append("Engage trading halt procedure if positions are at risk")
            if "kill_switch" in " ".join(components).lower():
                actions.append("Verify kill-switch state and acknowledge")
        elif severity == "P1":
            actions.append("Notify on-call engineer within 15 minutes")
            actions.append("Check system logs for root cause")
            actions.append("Activate degraded-mode fallbacks if available")
        elif severity == "P2":
            actions.append("Log issue for next business-hours review")
            actions.append("Monitor system for escalation indicators")
        else:
            actions.append("Record in incident tracker for SLA review")
        return actions

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        title: str = task.payload["title"]
        description: str = task.payload["description"]
        affected_components: list[str] = list(task.payload["affected_components"])
        detected_by: str = task.payload["detected_by"]
        error_details: dict[str, Any] = task.payload.get("error_details") or {}

        audit.log(
            f"Triaging incident: '{title}', components={affected_components}, "
            f"detected_by={detected_by}"
        )

        severity_str = self._classify_severity(title, description)
        runbook_refs = self._find_runbooks(title, description)
        immediate_actions = self._immediate_actions(severity_str, affected_components)
        escalation_recommended = severity_str in ("P0", "P1")

        incident_id = _new_id()
        detected_at = _utcnow()

        # Create IncidentRecord if available
        try:
            from incidents.contracts import IncidentRecord, IncidentSeverity, IncidentStatus
            severity_enum_map = {
                "P0": IncidentSeverity.P0_CRITICAL,
                "P1": IncidentSeverity.P1_HIGH,
                "P2": IncidentSeverity.P2_MEDIUM,
                "P3": IncidentSeverity.P3_LOW,
            }
            record = IncidentRecord(
                incident_id=incident_id,
                title=title,
                description=description,
                severity=severity_enum_map.get(severity_str, IncidentSeverity.P3_LOW),
                status=IncidentStatus.OPEN,
                detected_at=detected_at,
                detected_by=detected_by,
                affected_components=affected_components,
                affected_agents=[],
                affected_workflows=[],
                evidence_bundle_ids=[],
                related_alert_ids=[],
                runbook_refs=runbook_refs,
                timeline=[
                    {
                        "ts": detected_at,
                        "actor": self.NAME,
                        "action": "triage_completed",
                        "notes": f"Severity classified as {severity_str}",
                    }
                ],
            )
            audit.log(f"Created IncidentRecord id={incident_id}, severity={severity_str}")
        except ImportError:
            audit.warn("incidents.contracts unavailable — returning plain dict only")
        except Exception as exc:
            audit.warn(f"IncidentRecord creation failed: {exc}")

        audit.log(
            f"Triage complete: severity={severity_str}, runbooks={runbook_refs}, "
            f"escalation_recommended={escalation_recommended}"
        )

        return {
            "incident_id": incident_id,
            "severity": severity_str,
            "affected_components": affected_components,
            "runbook_refs": runbook_refs,
            "immediate_actions": immediate_actions,
            "escalation_recommended": escalation_recommended,
            "detected_at": detected_at,
            "error_details": error_details,
        }


# ══════════════════════════════════════════════════════════════════
# 6. PostmortemDraftingAgent
# ══════════════════════════════════════════════════════════════════


class PostmortemDraftingAgent(BaseAgent):
    """
    Drafts structured postmortem documents from incident data.

    Uses incident summaries and timelines to identify root cause, contributing
    factors, phases (detection / diagnosis / mitigation / resolution), and
    produces action items with prevention recommendations.

    Task types
    ----------
    draft_postmortem
        Create a full postmortem draft from incident data.
    analyze_incident_pattern
        Identify patterns across multiple incidents (if batch provided).

    Required payload keys
    ---------------------
    incident_id : str

    Optional payload keys
    ---------------------
    incident_summary : str
    timeline : list[dict]  (each: {ts, actor, action, notes})
    resolution : str

    Output keys
    -----------
    postmortem_draft : dict
    action_items : list[str]
    prevention_recommendations : list[str]
    """

    NAME = "postmortem_drafting"
    ALLOWED_TASK_TYPES = {"draft_postmortem", "analyze_incident_pattern"}
    REQUIRED_PAYLOAD_KEYS = {"incident_id"}

    @staticmethod
    def _identify_phase_duration(
        timeline: list[dict[str, Any]], phase_keywords: list[str]
    ) -> str | None:
        """Find the first timeline event matching any keyword."""
        for event in timeline:
            notes = str(event.get("notes", "")).lower()
            action = str(event.get("action", "")).lower()
            if any(kw in notes or kw in action for kw in phase_keywords):
                return str(event.get("ts", "unknown"))
        return None

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        incident_id: str = task.payload["incident_id"]
        incident_summary: str = task.payload.get("incident_summary") or ""
        timeline: list[dict[str, Any]] = task.payload.get("timeline") or []
        resolution: str = task.payload.get("resolution") or ""

        audit.log(
            f"Drafting postmortem for incident {incident_id}, "
            f"timeline_events={len(timeline)}"
        )

        now = _utcnow()

        # Phase timestamps
        detection_ts = self._identify_phase_duration(
            timeline, ["detect", "alert", "paged", "discovered"]
        )
        diagnosis_ts = self._identify_phase_duration(
            timeline, ["diagnos", "root_cause", "identified", "investigated"]
        )
        mitigation_ts = self._identify_phase_duration(
            timeline, ["mitigat", "workaround", "partial_fix", "reduced_impact"]
        )
        resolution_ts = self._identify_phase_duration(
            timeline, ["resolv", "fixed", "restored", "closed"]
        )

        # Infer root cause from summary keywords
        root_cause = "Unknown — requires further investigation."
        summary_lower = incident_summary.lower()
        if "drift" in summary_lower:
            root_cause = "Feature or data distribution drift caused model degradation."
        elif "data" in summary_lower and "missing" in summary_lower:
            root_cause = "Missing or incomplete data feed caused downstream failures."
        elif "kill_switch" in summary_lower or "halt" in summary_lower:
            root_cause = "Risk limit breach triggered automatic kill-switch."
        elif "timeout" in summary_lower:
            root_cause = "Service timeout caused workflow failure."
        elif "memory" in summary_lower or "oom" in summary_lower:
            root_cause = "Memory exhaustion caused process failure."

        # Contributing factors
        contributing_factors: list[str] = []
        if "no monitoring" in summary_lower or "undetected" in summary_lower:
            contributing_factors.append("Insufficient monitoring coverage")
        if "delayed" in summary_lower or "late" in summary_lower:
            contributing_factors.append("Delayed detection due to alerting gap")
        if "manual" in summary_lower:
            contributing_factors.append("Manual process introduced human error")
        if not contributing_factors:
            contributing_factors.append("Contributing factors require further investigation")

        # What went well
        what_went_well: list[str] = []
        if resolution:
            what_went_well.append("Resolution was successfully applied")
        if detection_ts and mitigation_ts:
            what_went_well.append("Mitigation was initiated promptly after detection")
        if not what_went_well:
            what_went_well.append("On-call procedures were followed")

        # What went wrong
        what_went_wrong: list[str] = []
        if not detection_ts:
            what_went_wrong.append("Detection timeline is unclear — alerting may need improvement")
        if not resolution_ts:
            what_went_wrong.append("Incident not yet marked as resolved")
        if not what_went_wrong:
            what_went_wrong.append("Impact was larger than expected")

        # Action items
        action_items: list[str] = [
            f"[P1] Root-cause analysis follow-up for incident {incident_id}",
            "[P2] Review alerting thresholds to reduce detection time",
            "[P2] Update runbook with lessons learned",
        ]
        if "drift" in root_cause.lower():
            action_items.append("[P1] Implement automated drift alerting with PSI threshold=0.20")
        if "monitoring" in " ".join(contributing_factors).lower():
            action_items.append("[P1] Add monitoring coverage for identified gap")

        # Prevention recommendations
        prevention: list[str] = [
            "Add canary checks before deploying model updates",
            "Implement circuit breaker for critical data pipelines",
            "Schedule quarterly chaos engineering exercises",
        ]

        postmortem_id = _new_id()
        postmortem_draft = {
            "postmortem_id": postmortem_id,
            "incident_id": incident_id,
            "drafted_by": self.NAME,
            "drafted_at": now,
            "status": "DRAFT",
            "incident_summary": incident_summary,
            "root_cause": root_cause,
            "contributing_factors": contributing_factors,
            "timeline": timeline,
            "phases": {
                "detection_ts": detection_ts,
                "diagnosis_ts": diagnosis_ts,
                "mitigation_ts": mitigation_ts,
                "resolution_ts": resolution_ts,
            },
            "what_went_well": what_went_well,
            "what_went_wrong": what_went_wrong,
            "resolution": resolution,
            "action_items": action_items,
            "prevention_recommendations": prevention,
        }

        audit.log(
            f"Postmortem draft complete: id={postmortem_id}, "
            f"{len(action_items)} action items"
        )

        return {
            "postmortem_draft": postmortem_draft,
            "action_items": action_items,
            "prevention_recommendations": prevention,
        }


# ══════════════════════════════════════════════════════════════════
# 7. AlertAggregationAgent
# ══════════════════════════════════════════════════════════════════


class AlertAggregationAgent(BaseAgent):
    """
    Deduplicates, groups, and prioritises incoming alerts.

    Produces a typed ``AlertBundle`` from ``agent_artifacts.contracts``.

    Task types
    ----------
    aggregate_alerts
        Full aggregation + deduplication + prioritisation.
    prioritize_alerts
        Prioritisation only (no deduplication).

    Required payload keys
    ---------------------
    alerts : list[dict]

    Optional payload keys
    ---------------------
    dedup_window_minutes : float  (default 5)
    max_output : int              (default 20)

    Output keys
    -----------
    alert_bundle : dict
    deduplicated_count : int
    requires_immediate_action : bool
    action_items : list[str]
    """

    NAME = "alert_aggregation"
    ALLOWED_TASK_TYPES = {"aggregate_alerts", "prioritize_alerts"}
    REQUIRED_PAYLOAD_KEYS = {"alerts"}

    _SEVERITY_ORDER = ["P0", "P1", "P2", "P3", "P4", "INFO", "WARNING", "UNKNOWN"]

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        alerts: list[dict[str, Any]] = task.payload["alerts"]
        dedup_window_min: float = float(task.payload.get("dedup_window_minutes", 5))
        max_output: int = int(task.payload.get("max_output", 20))

        audit.log(
            f"Aggregating {len(alerts)} alerts, "
            f"dedup_window={dedup_window_min}min, max_output={max_output}"
        )

        # Deduplication by (source, message) within dedup window
        seen: dict[str, str] = {}  # key → first timestamp
        deduped: list[dict[str, Any]] = []
        dedup_count = 0

        for alert in alerts:
            source = str(alert.get("source", "unknown"))
            message = str(alert.get("message", ""))
            ts_str = str(alert.get("timestamp", _utcnow()))
            key = f"{source}::{message}"

            if key in seen:
                # Check if within dedup window
                try:
                    t_first = datetime.fromisoformat(seen[key].replace("Z", ""))
                    t_current = datetime.fromisoformat(ts_str.replace("Z", ""))
                    delta_min = (t_current - t_first).total_seconds() / 60.0
                    if abs(delta_min) <= dedup_window_min:
                        dedup_count += 1
                        continue
                except Exception:
                    dedup_count += 1
                    continue

            seen[key] = ts_str
            # Ensure alert has required fields
            enriched = {
                "alert_id": alert.get("alert_id") or _new_id(),
                "source": source,
                "severity": str(alert.get("severity", "UNKNOWN")),
                "message": message,
                "timestamp": ts_str,
                "component": str(alert.get("component", "unknown")),
            }
            deduped.append(enriched)

        # Sort by severity order then timestamp
        def sort_key(a: dict) -> tuple[int, str]:
            sev = a["severity"].upper()
            try:
                sev_idx = self._SEVERITY_ORDER.index(sev)
            except ValueError:
                sev_idx = len(self._SEVERITY_ORDER)
            return (sev_idx, a["timestamp"])

        deduped.sort(key=sort_key)
        output_alerts = deduped[:max_output]

        # Severity and source breakdowns
        severity_breakdown: dict[str, int] = {}
        source_breakdown: dict[str, int] = {}
        for a in output_alerts:
            sev = a["severity"]
            src = a["source"]
            severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1
            source_breakdown[src] = source_breakdown.get(src, 0) + 1

        requires_immediate = any(
            a["severity"].upper() in ("P0", "P1") for a in output_alerts
        )

        # Action items for immediate alerts
        action_items: list[str] = []
        p0_alerts = [a for a in output_alerts if a["severity"].upper() == "P0"]
        p1_alerts = [a for a in output_alerts if a["severity"].upper() == "P1"]
        if p0_alerts:
            action_items.append(
                f"IMMEDIATE: {len(p0_alerts)} P0 alert(s) require immediate response"
            )
            for a in p0_alerts[:3]:
                action_items.append(f"  - [{a['source']}] {a['message'][:100]}")
        if p1_alerts:
            action_items.append(
                f"URGENT: {len(p1_alerts)} P1 alert(s) require response within 15min"
            )

        bundle_id = _new_id()
        bundle_ts = _utcnow()

        try:
            from agent_artifacts.contracts import AlertBundle
            bundle = AlertBundle(
                bundle_id=bundle_id,
                produced_by=self.NAME,
                timestamp=bundle_ts,
                alert_count=len(output_alerts),
                alerts=tuple(output_alerts),
                severity_breakdown=severity_breakdown,
                source_breakdown=source_breakdown,
                requires_immediate_action=requires_immediate,
                action_items=tuple(action_items),
            )
            bundle_dict = {
                "bundle_id": bundle.bundle_id,
                "produced_by": bundle.produced_by,
                "timestamp": bundle.timestamp,
                "alert_count": bundle.alert_count,
                "alerts": list(bundle.alerts),
                "severity_breakdown": bundle.severity_breakdown,
                "source_breakdown": bundle.source_breakdown,
                "requires_immediate_action": bundle.requires_immediate_action,
                "action_items": list(bundle.action_items),
            }
        except ImportError:
            bundle_dict = {
                "bundle_id": bundle_id,
                "produced_by": self.NAME,
                "timestamp": bundle_ts,
                "alert_count": len(output_alerts),
                "alerts": output_alerts,
                "severity_breakdown": severity_breakdown,
                "source_breakdown": source_breakdown,
                "requires_immediate_action": requires_immediate,
                "action_items": action_items,
            }

        audit.log(
            f"Alert aggregation complete: {len(output_alerts)} output alerts "
            f"(deduped={dedup_count}), requires_immediate={requires_immediate}"
        )

        return {
            "alert_bundle": bundle_dict,
            "deduplicated_count": dedup_count,
            "requires_immediate_action": requires_immediate,
            "action_items": action_items,
        }
