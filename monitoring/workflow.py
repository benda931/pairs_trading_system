# -*- coding: utf-8 -*-
"""
monitoring/workflow.py — Data Integrity Monitoring Workflow
===========================================================

Provides the canonical WorkflowEngine-based definition for the data
integrity monitoring workflow.  This is the *first real WorkflowEngine
workflow* dispatched from operational code (P1-AGENTS).

Workflow: DATA_INTEGRITY_CHECK
------------------------------
One step:
    1. data_integrity_check  →  DataIntegrityAgent.check_data_integrity

Safety class: BOUNDED_SAFE (risk_class)
    - Agent is READ_ONLY: reads price data, emits findings, never mutates
      runtime state, never places orders, never alters risk parameters.
    - No approval gate required.
    - Failures are isolated: WorkflowEngine catches all exceptions, records
      a FailureRecord, and returns FAILED WorkflowOutcome.  The daily
      pipeline continues regardless.

Trigger paths
-------------
    # From orchestrator after data_refresh (operational)
    from monitoring.workflow import run_data_integrity_workflow
    outcome = run_data_integrity_workflow(triggered_by="orchestrator")

    # From CLI / cron (scripts/run_data_integrity.py)
    outcome = run_data_integrity_workflow(triggered_by="cli")

    # From dashboard health panel
    outcome = run_data_integrity_workflow(
        prices=df,
        symbols=["SPY", "QQQ"],
        triggered_by="dashboard",
    )

Audit behaviour
---------------
    Every execution produces:
    - WorkflowRun with per-step WorkflowStepRun (held in WorkflowEngine memory)
    - AgentResult.audit_trail — timestamped log lines from DataIntegrityAgent
    - AgentRegistry._audit_log entry (bounded ring buffer, 10 000 entries)
    - emit_dashboard_alert() when critical issues are found (session state,
      non-persistent — fires only when running under Streamlit)

    Persistent audit chain: NOT YET implemented.
    See docs/remediation/remediation_ledger.md:P1-AUDIT.

Agents still scaffold-only (dispatched from nowhere yet — 27 agents)
--------------------------------------------------------------------
    system_health, drift_monitor, orchestration_reliability, incident_triage,
    postmortem_drafter, alert_aggregator, universe_discovery, pair_validation,
    spread_fit, feature_steward, model_trainer, meta_labeler, model_promoter,
    champion_challenger, label_auditor, policy_review, change_impact,
    portfolio_construction, capital_budget, exposure_monitor,
    drawdown_monitor, kill_switch, derisking, signal_analyst,
    regime_surveillance, trade_lifecycle, exit_oversight
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("monitoring.workflow")

# ── Constants ──────────────────────────────────────────────────────
WORKFLOW_ID = "DATA_INTEGRITY_CHECK"
WORKFLOW_VERSION = "1.0"


# ── WorkflowDefinition factory ─────────────────────────────────────

def build_data_integrity_workflow():
    """
    Return the WorkflowDefinition for the data integrity monitoring workflow.

    Single step, READ_ONLY, BOUNDED_SAFE, no approval gate required.

    Returns
    -------
    WorkflowDefinition
    """
    from orchestration.contracts import (
        EnvironmentClass,
        RiskClass,
        WorkflowDefinition,
        WorkflowStep,
        WorkflowTransition,
    )

    step = WorkflowStep(
        step_id="data_integrity_check",
        name="Data Integrity Check",
        agent_name="data_integrity",
        task_type="check_data_integrity",
        depends_on=(),
        timeout_seconds=120,
        retry_max=1,
        risk_class=RiskClass.BOUNDED_SAFE,
        requires_approval_before=False,
        requires_approval_after=False,
        on_failure="skip",
        notes=(
            "READ_ONLY: validates price data for NaN gaps, staleness, and "
            "suspicious price jumps.  Never mutates any state."
        ),
    )

    terminal = WorkflowTransition(
        from_step_id="data_integrity_check",
        to_step_id=None,   # terminal — workflow ends after this step
        condition="always",
        notes="Single-step workflow; terminate after data_integrity_check.",
    )

    return WorkflowDefinition(
        workflow_id=WORKFLOW_ID,
        name="Data Integrity Monitoring",
        description=(
            "Validates price data quality for all tracked symbols. "
            "Checks NaN gaps, staleness (>5 days), and suspicious price "
            "jumps (>20%).  Emits dashboard alerts on critical findings. "
            "READ_ONLY — does not mutate any state."
        ),
        version=WORKFLOW_VERSION,
        steps=(step,),
        transitions=(terminal,),
        entry_condition=(
            "After data_refresh task completes in daily pipeline, "
            "OR on explicit manual/dashboard trigger."
        ),
        termination_condition=(
            "data_integrity_check step reaches COMPLETED or FAILED."
        ),
        environment_class=EnvironmentClass.RESEARCH,
        risk_class=RiskClass.BOUNDED_SAFE,
        max_duration_seconds=180,
        idempotent=True,
        replayable=True,
        owner="monitoring",
        tags=frozenset({"data_quality", "monitoring", "read_only", "p1_agents"}),
        notes=(
            "First real WorkflowEngine-based dispatch from operational code. "
            "See docs/remediation/remediation_ledger.md:P1-AGENTS."
        ),
    )


# ── Convenience runner ─────────────────────────────────────────────

def run_data_integrity_workflow(
    *,
    prices=None,
    symbols: Optional[list] = None,
    expected_trading_days: int = 252,
    max_gap_days: int = 5,
    max_daily_move_pct: float = 0.20,
    triggered_by: str = "manual",
    emit_alerts: bool = True,
) -> Any:
    """
    Execute the data integrity workflow via WorkflowEngine.

    Parameters
    ----------
    prices : pd.DataFrame, optional
        Price data to validate.  If None the agent returns a graceful
        "no price data" result without failing.
    symbols : list[str], optional
        Subset of symbols to validate.  Defaults to all columns of prices.
    expected_trading_days : int
        Expected minimum trading day count per symbol (default: 252).
    max_gap_days : int
        Maximum allowed consecutive NaN gap length (default: 5).
    max_daily_move_pct : float
        Maximum allowed single-day price move (default: 0.20 = 20%).
    triggered_by : str
        Human-readable label for audit trail.
    emit_alerts : bool
        If True and running under Streamlit, emit dashboard alerts
        when critical issues are found.

    Returns
    -------
    WorkflowOutcome
        Typed outcome from WorkflowEngine (status, step counts, notes).
        The full AgentResult audit trail is accessible via
        ``registry.get_audit_log(agent_name="data_integrity", limit=1)``.
    """
    from agents.registry import get_default_registry
    from orchestration.contracts import EnvironmentClass
    from orchestration.engine import WorkflowEngine

    # ── Closure to capture agent output without engine internals ──
    # We provide an artifact_store callback that records the agent's
    # output dict.  This avoids inspecting non-public engine state.
    _captured_outputs: dict[str, Any] = {}

    def _capture_store(run_id: str, step_id: str, output: dict) -> list:
        """Capture agent output for alert-bus integration. Returns no IDs."""
        _captured_outputs[step_id] = output
        return []

    registry = get_default_registry()
    engine = WorkflowEngine(
        registry=registry,
        artifact_store=_capture_store,
        environment=EnvironmentClass.RESEARCH,
    )

    definition = build_data_integrity_workflow()

    # Build the payload that the engine forwards to the AgentTask
    trigger_payload: dict[str, Any] = {
        "expected_trading_days": expected_trading_days,
        "max_gap_days": max_gap_days,
        "max_daily_move_pct": max_daily_move_pct,
    }
    if symbols:
        trigger_payload["symbols"] = symbols
    if prices is not None:
        trigger_payload["prices"] = prices

    logger.info(
        "Starting data integrity workflow: triggered_by=%s symbols=%s",
        triggered_by,
        symbols or ("all" if prices is not None else "none"),
    )

    wf_run = engine.run(
        definition=definition,
        trigger_payload=trigger_payload,
        triggered_by=triggered_by,
    )

    # Build typed WorkflowOutcome from the completed run
    outcome = engine.get_outcome(wf_run.run_id)

    # ── Alert bus integration (best-effort) ───────────────────────
    if emit_alerts:
        step_output = _captured_outputs.get("data_integrity_check") or {}
        _maybe_emit_alerts(
            step_output=step_output,
            run_id=wf_run.run_id,
        )

    if outcome is not None:
        logger.info(
            "Data integrity workflow finished: status=%s completed=%d failed=%d",
            outcome.status,
            outcome.steps_completed,
            outcome.steps_failed,
        )
        return outcome

    # Fallback: return wf_run directly if get_outcome failed
    logger.debug("get_outcome returned None — returning WorkflowRun as fallback")
    return wf_run


# ── Alert bus helper ───────────────────────────────────────────────

def _maybe_emit_alerts(*, step_output: dict, run_id: str) -> None:
    """
    Emit dashboard alerts when the DataIntegrityAgent reports critical issues.

    This is INFORMATIONAL only: writes to Streamlit session state.
    When running outside Streamlit (CLI, tests) the call is silently skipped.

    Parameters
    ----------
    step_output : dict
        The ``output`` dict from the DataIntegrityAgent AgentResult.
        Expected keys: ``issues_found``, ``critical_issues``, ``warnings``.
    run_id : str
        Workflow run identifier for the alert detail payload.
    """
    if not step_output:
        return

    try:
        from core.alert_bus import emit_dashboard_alert

        issues: int = int(step_output.get("issues_found", 0))
        critical: list = list(step_output.get("critical_issues", []))
        warnings: list = list(step_output.get("warnings", []))

        if issues > 0:
            level = "error" if issues >= 3 else "warning"
            emit_dashboard_alert(
                level=level,
                source="data_integrity_workflow",
                message=f"Data integrity: {issues} critical issue(s) detected",
                details={
                    "critical_issues": critical[:5],
                    "warnings": warnings[:5],
                    "workflow_run_id": run_id,
                },
            )
            logger.warning(
                "DataIntegrityWorkflow run=%s: %d critical issues — alert emitted",
                run_id, issues,
            )
        elif warnings:
            emit_dashboard_alert(
                level="info",
                source="data_integrity_workflow",
                message=f"Data integrity: {len(warnings)} warning(s), no critical issues",
                details={
                    "warnings": warnings[:5],
                    "workflow_run_id": run_id,
                },
            )

    except Exception as exc:
        # Alert bus is best-effort — must never propagate failures
        logger.debug("Alert bus emit failed (non-fatal): %s", exc)
