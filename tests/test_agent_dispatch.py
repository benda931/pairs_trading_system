# -*- coding: utf-8 -*-
"""
tests/test_agent_dispatch.py — Agent Dispatch Integration Tests
===============================================================

Tests for the DataIntegrityAgent and SystemHealthAgent workflows — the first
two real agent dispatches from operational code (P1-AGENTS).

These tests verify:
1.  Workflow definition is structurally sound (WorkflowDefinition validates)
2.  WorkflowEngine dispatches DataIntegrityAgent and returns WorkflowOutcome
3.  Agent produces typed AgentResult with non-empty audit trail
4.  No unauthorized downstream action occurs (READ_ONLY guarantee)
5.  Alert bus emission on critical findings (unit-testable path)
6.  Orchestrator can call the workflow via both paths (WorkflowEngine / direct)
7.  Workflow is safe when prices=None (graceful no-data path)
8.  CLI script entry point is importable
9.  SystemHealthAgent workflow is structurally sound (mirrors §1)
10. WorkflowEngine dispatches SystemHealthAgent and returns outcome (mirrors §2)
11. Orchestrator run_agent_system_health_check() returns TaskResult (mirrors §6)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def registry():
    from agents.registry import get_default_registry
    return get_default_registry()


@pytest.fixture(scope="module")
def workflow_definition():
    from monitoring.workflow import build_data_integrity_workflow
    return build_data_integrity_workflow()


# ══════════════════════════════════════════════════════════════════
# 1. WorkflowDefinition structural soundness
# ══════════════════════════════════════════════════════════════════


class TestWorkflowDefinition:

    def test_definition_is_not_none(self, workflow_definition):
        assert workflow_definition is not None

    def test_definition_has_one_step(self, workflow_definition):
        assert len(workflow_definition.steps) == 1, (
            "DATA_INTEGRITY_CHECK must be a single-step workflow"
        )

    def test_step_agent_is_data_integrity(self, workflow_definition):
        step = workflow_definition.steps[0]
        assert step.agent_name == "data_integrity"
        assert step.task_type == "check_data_integrity"

    def test_step_requires_no_approval(self, workflow_definition):
        step = workflow_definition.steps[0]
        assert not step.requires_approval_before
        assert not step.requires_approval_after

    def test_risk_class_is_bounded_safe(self, workflow_definition):
        from orchestration.contracts import RiskClass
        assert workflow_definition.risk_class == RiskClass.BOUNDED_SAFE
        assert workflow_definition.steps[0].risk_class == RiskClass.BOUNDED_SAFE

    def test_workflow_is_idempotent_and_replayable(self, workflow_definition):
        assert workflow_definition.idempotent
        assert workflow_definition.replayable

    def test_step_on_failure_is_skip(self, workflow_definition):
        # Failures must not block the pipeline — on_failure must be "skip"
        step = workflow_definition.steps[0]
        assert step.on_failure == "skip", (
            "data_integrity step must use on_failure='skip' so pipeline continues"
        )

    def test_dependency_references_valid(self, workflow_definition):
        """All depends_on references must point to valid step_ids."""
        step_ids = {s.step_id for s in workflow_definition.steps}
        for step in workflow_definition.steps:
            for dep in step.depends_on:
                assert dep in step_ids, (
                    f"Step {step.step_id!r} depends on unknown step {dep!r}"
                )

    def test_terminal_transition_exists(self, workflow_definition):
        """At least one transition must have to_step_id=None (terminal)."""
        terminals = [
            t for t in workflow_definition.transitions if t.to_step_id is None
        ]
        assert terminals, (
            "WorkflowDefinition must have at least one terminal transition"
        )


# ══════════════════════════════════════════════════════════════════
# 2. WorkflowEngine dispatch produces typed outcome
# ══════════════════════════════════════════════════════════════════


class TestWorkflowEngineDispatch:

    def test_run_data_integrity_workflow_returns_outcome(self):
        from monitoring.workflow import run_data_integrity_workflow

        outcome = run_data_integrity_workflow(
            triggered_by="pytest",
            emit_alerts=False,
        )
        assert outcome is not None
        # Returns WorkflowOutcome (preferred) or WorkflowRun (fallback)
        assert hasattr(outcome, "run_id"), (
            "Workflow return value must have a run_id attribute"
        )
        assert hasattr(outcome, "status"), (
            "Workflow return value must have a status attribute"
        )

    def test_outcome_has_run_id(self):
        from monitoring.workflow import run_data_integrity_workflow
        outcome = run_data_integrity_workflow(
            triggered_by="pytest",
            emit_alerts=False,
        )
        assert outcome.run_id, "Workflow result must have a non-empty run_id"

    def test_outcome_status_is_terminal(self):
        """Status must be COMPLETED or FAILED — never PENDING/RUNNING."""
        from monitoring.workflow import run_data_integrity_workflow
        from orchestration.contracts import WorkflowStatus
        outcome = run_data_integrity_workflow(
            triggered_by="pytest",
            emit_alerts=False,
        )
        assert outcome.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED), (
            f"Expected terminal status, got {outcome.status}"
        )

    def test_steps_completed_plus_failed_equals_one(self):
        """WorkflowOutcome must account for all steps (completed+failed+skipped=1)."""
        from monitoring.workflow import run_data_integrity_workflow
        from orchestration.contracts import WorkflowOutcome
        outcome = run_data_integrity_workflow(
            triggered_by="pytest",
            emit_alerts=False,
        )
        if not isinstance(outcome, WorkflowOutcome):
            pytest.skip("get_outcome returned WorkflowRun (not terminal yet)")
        total = outcome.steps_completed + outcome.steps_failed + outcome.steps_skipped
        assert total == 1, (
            f"Single-step workflow must have exactly 1 step accounted for, got {total}"
        )

    def test_workflow_safe_with_prices_none(self):
        """Agent must handle no prices gracefully — no exception, typed output."""
        from monitoring.workflow import run_data_integrity_workflow
        from orchestration.contracts import WorkflowStatus

        outcome = run_data_integrity_workflow(
            prices=None,
            symbols=None,
            triggered_by="pytest_no_data",
            emit_alerts=False,
        )
        assert outcome.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED), (
            "Workflow must terminate cleanly even with no price data"
        )

    def test_workflow_safe_with_price_dataframe(self):
        """Agent validates a real price DataFrame without crashing."""
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            pytest.skip("pandas/numpy not available")

        from monitoring.workflow import run_data_integrity_workflow
        from orchestration.contracts import WorkflowStatus

        dates = pd.date_range("2024-01-02", periods=60, freq="B")
        prices = pd.DataFrame(
            {
                "SPY": 400.0 + np.random.randn(60) * 2,
                "QQQ": 300.0 + np.random.randn(60) * 2,
            },
            index=dates,
        )
        prices.iloc[10:16, 0] = float("nan")  # inject a known gap

        outcome = run_data_integrity_workflow(
            prices=prices,
            symbols=["SPY", "QQQ"],
            triggered_by="pytest_with_data",
            emit_alerts=False,
        )
        assert outcome.status == WorkflowStatus.COMPLETED, (
            f"Workflow should COMPLETE with valid price data, got {outcome.status}"
        )


# ══════════════════════════════════════════════════════════════════
# 3. Typed AgentResult and audit trail
# ══════════════════════════════════════════════════════════════════


class TestAgentResultAndAuditTrail:

    def test_direct_dispatch_produces_agent_result(self, registry):
        from core.contracts import AgentResult, AgentStatus, AgentTask

        task = AgentTask(
            task_id="test_dispatch_typed_001",
            agent_name="data_integrity",
            task_type="check_data_integrity",
            payload={},
        )
        result = registry.dispatch(task)

        assert isinstance(result, AgentResult), (
            "Registry.dispatch() must always return AgentResult"
        )
        assert result.status == AgentStatus.COMPLETED, (
            f"Expected COMPLETED, got {result.status} (error={result.error!r})"
        )

    def test_agent_result_output_has_required_keys(self, registry):
        from core.contracts import AgentTask

        task = AgentTask(
            task_id="test_dispatch_keys_001",
            agent_name="data_integrity",
            task_type="check_data_integrity",
            payload={},
        )
        result = registry.dispatch(task)

        assert isinstance(result.output, dict)
        required = {"integrity_results", "issues_found", "critical_issues", "warnings"}
        missing = required - set(result.output.keys())
        assert not missing, (
            f"AgentResult.output missing required keys: {missing}"
        )

    def test_agent_result_audit_trail_non_empty(self, registry):
        from core.contracts import AgentTask

        task = AgentTask(
            task_id="test_audit_trail_001",
            agent_name="data_integrity",
            task_type="check_data_integrity",
            payload={},
        )
        result = registry.dispatch(task)

        assert isinstance(result.audit_trail, list)
        assert len(result.audit_trail) > 0, (
            "DataIntegrityAgent must populate audit_trail with at least one entry"
        )

    def test_registry_audit_log_records_dispatch(self, registry):
        from core.contracts import AgentTask

        task = AgentTask(
            task_id="test_registry_audit_001",
            agent_name="data_integrity",
            task_type="check_data_integrity",
            payload={},
        )
        registry.dispatch(task)

        log = registry.get_audit_log(agent_name="data_integrity", limit=10)
        assert len(log) > 0, (
            "Registry must record dispatch in its audit log"
        )
        latest = log[-1]
        assert latest["agent_name"] == "data_integrity"
        assert latest["task_type"] == "check_data_integrity"
        assert "status" in latest
        assert "duration_seconds" in latest
        assert "n_audit_entries" in latest


# ══════════════════════════════════════════════════════════════════
# 4. No unauthorized downstream action (READ_ONLY guarantee)
# ══════════════════════════════════════════════════════════════════


class TestNoUnauthorizedAction:

    def test_agent_does_not_mutate_registry(self, registry):
        """Agent count must not change after dispatch."""
        before = len(registry.list_agents())
        from core.contracts import AgentTask
        task = AgentTask(
            task_id="test_no_side_effect_001",
            agent_name="data_integrity",
            task_type="check_data_integrity",
            payload={},
        )
        registry.dispatch(task)
        after = len(registry.list_agents())
        assert before == after, (
            "DataIntegrityAgent must not register new agents (no side effects)"
        )

    def test_agent_result_has_no_trade_or_risk_mutation(self, registry):
        """Output dict must not contain trade/order/risk mutation keys."""
        from core.contracts import AgentTask

        FORBIDDEN_OUTPUT_KEYS = {
            "order", "orders", "trade", "trades",
            "kill_switch_trigger", "risk_limit_override",
            "force_exit", "execute",
        }

        task = AgentTask(
            task_id="test_no_exec_keys_001",
            agent_name="data_integrity",
            task_type="check_data_integrity",
            payload={},
        )
        result = registry.dispatch(task)
        actual_keys = set(result.output.keys()) if result.output else set()
        forbidden_present = actual_keys & FORBIDDEN_OUTPUT_KEYS
        assert not forbidden_present, (
            f"DataIntegrityAgent output contains forbidden keys: {forbidden_present}"
        )

    def test_workflow_step_risk_class_not_high_risk(self, workflow_definition):
        from orchestration.contracts import RiskClass
        for step in workflow_definition.steps:
            assert step.risk_class not in (RiskClass.HIGH_RISK, RiskClass.SENSITIVE), (
                f"data_integrity step {step.step_id!r} must not be HIGH_RISK or SENSITIVE"
            )

    def test_workflow_no_approval_gate(self, workflow_definition):
        """READ_ONLY workflow must not require approval gates."""
        for step in workflow_definition.steps:
            assert not step.requires_approval_before, (
                f"Step {step.step_id!r}: READ_ONLY agents must not require pre-approval"
            )
            assert not step.requires_approval_after, (
                f"Step {step.step_id!r}: READ_ONLY agents must not require post-approval"
            )


# ══════════════════════════════════════════════════════════════════
# 5. Alert bus emission path (unit-testable, no Streamlit required)
# ══════════════════════════════════════════════════════════════════


class TestAlertBusIntegration:

    def test_maybe_emit_alerts_does_not_raise_on_empty_output(self):
        from monitoring.workflow import _maybe_emit_alerts
        # Must not raise when output is empty
        _maybe_emit_alerts(step_output={}, run_id="test_run_001")

    def test_maybe_emit_alerts_does_not_raise_outside_streamlit(self):
        """emit_dashboard_alert imports Streamlit session state.
        When Streamlit is not running it raises or silently does nothing.
        _maybe_emit_alerts must absorb the exception and not propagate it."""
        from monitoring.workflow import _maybe_emit_alerts
        _maybe_emit_alerts(
            step_output={
                "issues_found": 5,
                "critical_issues": ["SPY: gap detected", "QQQ: stale"],
                "warnings": ["GLD: jump detected"],
            },
            run_id="test_run_with_issues",
        )
        # If we reach here without exception, the test passes

    def test_maybe_emit_alerts_does_not_raise_on_warnings_only(self):
        from monitoring.workflow import _maybe_emit_alerts
        _maybe_emit_alerts(
            step_output={
                "issues_found": 0,
                "critical_issues": [],
                "warnings": ["SPY: 2 price jumps >20% detected"],
            },
            run_id="test_run_warnings_only",
        )


# ══════════════════════════════════════════════════════════════════
# 6. Orchestrator integration
# ══════════════════════════════════════════════════════════════════


class TestOrchestratorIntegration:

    def test_orchestrator_method_present(self):
        from core.orchestrator import PairsOrchestrator
        orch = PairsOrchestrator()
        assert hasattr(orch, "run_agent_data_integrity_check"), (
            "Orchestrator must expose run_agent_data_integrity_check()"
        )

    def test_orchestrator_dispatch_returns_task_result(self):
        from core.orchestrator import PairsOrchestrator, TaskResult
        orch = PairsOrchestrator()
        result = orch.run_agent_data_integrity_check()

        assert result is not None
        assert isinstance(result, TaskResult)
        assert result.task_name == "agent_data_integrity"

    def test_orchestrator_result_output_has_path_key(self):
        """Output must identify which dispatch path was used."""
        from core.orchestrator import PairsOrchestrator
        orch = PairsOrchestrator()
        result = orch.run_agent_data_integrity_check()

        assert result is not None
        assert result.output is not None
        assert "path" in result.output, (
            "TaskResult.output must have 'path' key identifying dispatch path"
        )
        assert result.output["path"] in ("workflow_engine", "direct_dispatch"), (
            f"Unexpected path value: {result.output.get('path')}"
        )

    def test_run_daily_pipeline_includes_agent_check(self):
        """Source code of run_daily_pipeline must call agent dispatch."""
        import inspect
        from core.orchestrator import PairsOrchestrator
        source = inspect.getsource(PairsOrchestrator.run_daily_pipeline)
        assert "run_agent_data_integrity_check" in source, (
            "run_daily_pipeline must call run_agent_data_integrity_check()"
        )


# ══════════════════════════════════════════════════════════════════
# 7. Scaffold-only agents must NOT be called from operational code
# ══════════════════════════════════════════════════════════════════


class TestScaffoldOnlyAgents:
    """
    Verifies that scaffold-only agents (those not wired to any operational
    trigger) remain accessible from the registry but are not invoked by
    run_daily_pipeline() or run_agent_data_integrity_check().

    This test is a guard against accidental operational dispatch of agents
    that have not been reviewed and approved for production use.
    """

    # Agents that ARE legitimately dispatched from operational code
    OPERATIONAL_AGENTS = frozenset({"data_integrity", "system_health"})

    def test_scaffold_agents_are_registered(self, registry):
        """All expected scaffold agents must be present in the registry."""
        scaffold = [
            "system_health", "drift_monitoring", "portfolio_construction",
            "kill_switch", "policy_review",
        ]
        for name in scaffold:
            agent = registry.get_agent(name)
            assert agent is not None, (
                f"Scaffold agent {name!r} must be registered even if not dispatched"
            )

    def test_daily_pipeline_source_dispatches_only_approved_agents(self):
        """run_daily_pipeline source must reference only approved agent dispatches."""
        import inspect
        from core.orchestrator import PairsOrchestrator

        source = inspect.getsource(PairsOrchestrator.run_daily_pipeline)

        # These calls must be present — the two approved dispatches
        assert "run_agent_data_integrity_check" in source
        assert "run_agent_system_health_check" in source

        # Agents that must NOT appear in the pipeline source (scaffold only)
        disallowed_in_pipeline = [
            "portfolio_construction",
            "kill_switch",
            "drawdown_monitor",
            "signal_analyst",
            "model_promoter",
        ]
        for agent_name in disallowed_in_pipeline:
            assert agent_name not in source, (
                f"Scaffold agent {agent_name!r} must not appear in "
                f"run_daily_pipeline source — it is not approved for operational dispatch"
            )


# ══════════════════════════════════════════════════════════════════
# 8. CLI script importability
# ══════════════════════════════════════════════════════════════════


class TestCLIScript:

    def test_cli_script_exists(self):
        script = PROJECT_ROOT / "scripts" / "run_data_integrity.py"
        assert script.exists(), (
            "scripts/run_data_integrity.py must exist as standalone CLI trigger"
        )

    def test_cli_main_is_callable(self):
        import importlib.util
        script = PROJECT_ROOT / "scripts" / "run_data_integrity.py"
        spec = importlib.util.spec_from_file_location("run_data_integrity", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(getattr(mod, "main", None)), (
            "scripts/run_data_integrity.py must expose a callable main() function"
        )


# ══════════════════════════════════════════════════════════════════
# 9. SystemHealthAgent — WorkflowDefinition structural soundness
# ══════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def system_health_workflow_definition():
    from monitoring.workflow import build_system_health_workflow
    return build_system_health_workflow()


class TestSystemHealthWorkflowDefinition:

    def test_definition_is_not_none(self, system_health_workflow_definition):
        assert system_health_workflow_definition is not None

    def test_definition_has_one_step(self, system_health_workflow_definition):
        assert len(system_health_workflow_definition.steps) == 1

    def test_step_agent_is_system_health(self, system_health_workflow_definition):
        step = system_health_workflow_definition.steps[0]
        assert step.agent_name == "system_health", (
            f"Expected agent_name='system_health', got {step.agent_name!r}"
        )

    def test_step_task_type_is_health_sweep(self, system_health_workflow_definition):
        step = system_health_workflow_definition.steps[0]
        assert step.task_type == "health_sweep"

    def test_step_is_bounded_safe(self, system_health_workflow_definition):
        from orchestration.contracts import RiskClass
        step = system_health_workflow_definition.steps[0]
        assert step.risk_class == RiskClass.BOUNDED_SAFE

    def test_no_approval_gate(self, system_health_workflow_definition):
        for step in system_health_workflow_definition.steps:
            assert not step.requires_approval_before
            assert not step.requires_approval_after

    def test_on_failure_is_skip(self, system_health_workflow_definition):
        step = system_health_workflow_definition.steps[0]
        assert step.on_failure == "skip"

    def test_workflow_is_idempotent(self, system_health_workflow_definition):
        assert system_health_workflow_definition.idempotent is True


# ══════════════════════════════════════════════════════════════════
# 10. SystemHealthAgent — WorkflowEngine dispatch
# ══════════════════════════════════════════════════════════════════


class TestSystemHealthWorkflowDispatch:

    def test_run_returns_outcome_with_run_id(self):
        from monitoring.workflow import run_system_health_workflow
        outcome = run_system_health_workflow(triggered_by="test_suite", emit_alerts=False)
        assert outcome is not None
        assert hasattr(outcome, "run_id"), (
            "Outcome must have run_id attribute (WorkflowOutcome or WorkflowRun)"
        )

    def test_outcome_status_is_terminal(self):
        from monitoring.workflow import run_system_health_workflow
        from orchestration.contracts import WorkflowStatus
        outcome = run_system_health_workflow(triggered_by="test_suite", emit_alerts=False)
        status = getattr(outcome, "status", None)
        if status is not None:
            assert status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED), (
                f"Outcome status must be terminal, got: {status}"
            )

    def test_steps_count_sums_to_one(self):
        from monitoring.workflow import run_system_health_workflow
        outcome = run_system_health_workflow(triggered_by="test_suite", emit_alerts=False)
        completed = getattr(outcome, "steps_completed", None)
        failed = getattr(outcome, "steps_failed", None)
        if completed is not None and failed is not None:
            assert completed + failed == 1, (
                f"Single-step workflow: completed+failed must equal 1, "
                f"got completed={completed} failed={failed}"
            )

    def test_agent_result_output_has_required_keys(self, registry):
        """SystemHealthAgent output must contain all required output keys."""
        from core.contracts import AgentTask
        task = AgentTask(
            task_id="test_sh_output_keys_001",
            agent_name="system_health",
            task_type="health_sweep",
            payload={},
        )
        result = registry.dispatch(task)
        assert isinstance(result.output, dict)
        required = {"component_health", "overall_healthy", "unhealthy_components", "warnings"}
        missing = required - set(result.output.keys())
        assert not missing, f"SystemHealthAgent output missing keys: {missing}"

    def test_agent_result_audit_trail_non_empty(self, registry):
        from core.contracts import AgentTask
        task = AgentTask(
            task_id="test_sh_audit_001",
            agent_name="system_health",
            task_type="health_sweep",
            payload={},
        )
        result = registry.dispatch(task)
        assert isinstance(result.audit_trail, list)
        assert len(result.audit_trail) > 0

    def test_agent_result_output_no_trade_keys(self, registry):
        from core.contracts import AgentTask
        FORBIDDEN = {"order", "orders", "trade", "trades", "kill_switch_trigger",
                     "risk_limit_override", "force_exit", "execute"}
        task = AgentTask(
            task_id="test_sh_no_exec_001",
            agent_name="system_health",
            task_type="health_sweep",
            payload={},
        )
        result = registry.dispatch(task)
        actual = set(result.output.keys()) if result.output else set()
        assert not (actual & FORBIDDEN), f"Forbidden keys in output: {actual & FORBIDDEN}"


# ══════════════════════════════════════════════════════════════════
# 11. SystemHealthAgent — Orchestrator integration
# ══════════════════════════════════════════════════════════════════


class TestSystemHealthOrchestratorIntegration:

    def test_orchestrator_method_present(self):
        from core.orchestrator import PairsOrchestrator
        orch = PairsOrchestrator()
        assert hasattr(orch, "run_agent_system_health_check"), (
            "Orchestrator must expose run_agent_system_health_check()"
        )

    def test_orchestrator_dispatch_returns_task_result(self):
        from core.orchestrator import PairsOrchestrator, TaskResult
        orch = PairsOrchestrator()
        result = orch.run_agent_system_health_check()
        assert result is not None
        assert isinstance(result, TaskResult)
        assert result.task_name == "agent_system_health"

    def test_orchestrator_result_output_has_path_key(self):
        from core.orchestrator import PairsOrchestrator
        orch = PairsOrchestrator()
        result = orch.run_agent_system_health_check()
        assert result is not None
        assert result.output is not None
        assert "path" in result.output
        assert result.output["path"] in ("workflow_engine", "direct_dispatch")

    def test_maybe_emit_health_alerts_does_not_raise_on_empty(self):
        from monitoring.workflow import _maybe_emit_health_alerts
        _maybe_emit_health_alerts(step_output={}, run_id="test_sh_run_001")

    def test_maybe_emit_health_alerts_does_not_raise_on_unhealthy(self):
        from monitoring.workflow import _maybe_emit_health_alerts
        _maybe_emit_health_alerts(
            step_output={
                "overall_healthy": False,
                "unhealthy_components": ["core.signals_engine", "ml.inference.scorer"],
                "warnings": [],
            },
            run_id="test_sh_run_unhealthy",
        )

    def test_maybe_emit_health_alerts_does_not_raise_on_warnings_only(self):
        from monitoring.workflow import _maybe_emit_health_alerts
        _maybe_emit_health_alerts(
            step_output={
                "overall_healthy": True,
                "unhealthy_components": [],
                "warnings": ["core.sql_store: slow import (2500ms)"],
            },
            run_id="test_sh_run_warnings",
        )
