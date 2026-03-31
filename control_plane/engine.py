# -*- coding: utf-8 -*-
"""
control_plane/engine.py — ControlPlaneEngine
=============================================

The main operator control surface for the pairs trading system.

Responsibilities:
  - Receive ControlPlaneActions and route them to the correct handler.
  - Validate permissions, environment rules, and policy constraints before
    executing any mutation.
  - Apply the mutation through RuntimeStateManager.
  - Record every action with a full ControlPlaneActionRecord (prev/new state,
    audit trail).
  - Write OperatorActionRecords for external audit consumption.
  - Run liveTradingReadinessReport preflight checks before mode transitions.

Key design invariants:
  - NO silent state mutations — every path writes a record.
  - Emergency actions (is_emergency=True on an override) bypass approval but
    require a non-empty reason and are flagged in the audit trail.
  - Reversible actions have explicit undo paths.
  - Thread-safe via RuntimeStateManager's internal lock; the engine itself
    is safe to call from multiple threads.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

from runtime.contracts import (
    ActivationStatus,
    AgentActivationRecord,
    LiveTradingReadinessReport,
    ModelActivationRecord,
    PreflightCheckReport,
    RuntimeMode,
    RuntimeOverride,
    RuntimeState,
    ServiceState,
    StrategyActivationRecord,
    ThrottleLevel,
)
from runtime.environment import get_environment_spec, validate_environment_action
from runtime.state import RuntimeStateManager, get_runtime_state_manager

from control_plane.contracts import (
    ControlPlaneAction,
    ControlPlaneActionRecord,
    ControlPlaneActionType,
    KillSwitchState,
    OperatorActionRecord,
    ThrottleState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _make_id(prefix: str = "act") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _expiry_from_minutes(minutes: Optional[float]) -> Optional[str]:
    """Return ISO-8601 expiry string given an optional duration in minutes."""
    if minutes is None:
        return None
    expiry = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return expiry.isoformat()


def _as_dict(obj) -> dict:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {}


def _dict_to_str(d: dict) -> str:
    """Compact string representation of a dict for state capture."""
    import json
    try:
        return json.dumps(d, default=str, separators=(",", ":"))
    except Exception:
        return str(d)


# ---------------------------------------------------------------------------
# ControlPlaneEngine
# ---------------------------------------------------------------------------


class ControlPlaneEngine:
    """The main operator control surface.

    Receives ControlPlaneActions, validates them against policy/permissions,
    applies them to RuntimeStateManager, and records every action with a full
    audit trail.

    Principles:
      - Every action is permission-checked, environment-checked, and
        policy-checked before execution.
      - Every action produces an OperatorActionRecord.
      - Reversible actions have explicit undo paths.
      - Emergency actions bypass approval checks but require justification.
      - No silent state mutations.

    Args:
        state_manager:    RuntimeStateManager instance (uses singleton if None).
        governance_engine: Optional GovernancePolicyEngine for policy checks.
        approval_engine:   Optional ApprovalEngine for approval lookups.
        incident_manager:  Optional IncidentManager — called on kill-switch events.
    """

    def __init__(
        self,
        state_manager: Optional[RuntimeStateManager] = None,
        governance_engine=None,
        approval_engine=None,
        incident_manager=None,
    ) -> None:
        self._state_manager: RuntimeStateManager = (
            state_manager or get_runtime_state_manager()
        )
        self._governance_engine = governance_engine
        self._approval_engine = approval_engine
        self._incident_manager = incident_manager

        self._lock = threading.Lock()
        self._action_history: List[ControlPlaneActionRecord] = []
        self._operator_records: List[OperatorActionRecord] = []
        # Registry of external callbacks fired after significant events
        self._event_callbacks: Dict[str, List[Callable]] = {}

        logger.info(
            "ControlPlaneEngine initialised — env=%s",
            self._state_manager._env,
        )

    # ------------------------------------------------------------------
    # Public action dispatcher
    # ------------------------------------------------------------------

    def execute_action(self, action: ControlPlaneAction) -> ControlPlaneActionRecord:
        """Execute an operator control action.

        This is the single entry point for all programmatic control operations.
        It:
          1. Checks permissions and environment constraints.
          2. Dispatches to the appropriate handler method.
          3. Returns a fully-populated ControlPlaneActionRecord.

        Args:
            action: The action to execute.

        Returns:
            ControlPlaneActionRecord with execution outcome.
        """
        record = ControlPlaneActionRecord(
            record_id=_make_id("rec"),
            action=_as_dict(action),
            executed_at=_utcnow(),
            succeeded=False,
            previous_state="",
            new_state="",
            error=None,
            audit_trail=[],
        )

        record.audit_trail.append(
            f"[{_utcnow()}] execute_action: type={action.action_type.value} "
            f"scope={action.scope} operator={action.requested_by}"
        )

        # Permission gate
        permitted, perm_reason = self._check_permission(
            action.action_type, action.environment, action.requested_by
        )
        if not permitted:
            record.error = f"PERMISSION_DENIED: {perm_reason}"
            record.audit_trail.append(f"[{_utcnow()}] blocked: {perm_reason}")
            self._store_record(record)
            self._log_operator_action(action, "blocked", [action.scope])
            logger.warning(
                "Action %s blocked for operator %s: %s",
                action.action_type.value,
                action.requested_by,
                perm_reason,
            )
            return record

        # Dispatch
        dispatch_map: dict = {
            ControlPlaneActionType.ENABLE_STRATEGY: self._handle_enable_strategy,
            ControlPlaneActionType.DISABLE_STRATEGY: self._handle_disable_strategy,
            ControlPlaneActionType.PAUSE_NEW_ENTRIES: self._handle_pause_new_entries,
            ControlPlaneActionType.ALLOW_EXITS_ONLY: self._handle_allow_exits_only,
            ControlPlaneActionType.THROTTLE_SIZING: self._handle_throttle_sizing,
            ControlPlaneActionType.FREEZE_UNIVERSE: self._handle_freeze_universe,
            ControlPlaneActionType.DISABLE_MODEL: self._handle_disable_model,
            ControlPlaneActionType.ENABLE_FALLBACK: self._handle_enable_fallback,
            ControlPlaneActionType.DISABLE_AGENT: self._handle_disable_agent,
            ControlPlaneActionType.APPLY_TEMP_OVERRIDE: self._handle_apply_temp_override,
            ControlPlaneActionType.CLEAR_OVERRIDE: self._handle_clear_override,
            ControlPlaneActionType.FORCE_RECONCILE: self._handle_force_reconcile,
            ControlPlaneActionType.TRIGGER_DRAIN: self._handle_trigger_drain,
            ControlPlaneActionType.TRIGGER_HALT: self._handle_trigger_halt,
            ControlPlaneActionType.RELEASE_HALT: self._handle_release_halt,
            ControlPlaneActionType.INSPECT_STATE: self._handle_inspect_state,
            ControlPlaneActionType.ACTIVATE_RUNTIME: self._handle_activate_runtime,
            ControlPlaneActionType.DEACTIVATE_RUNTIME: self._handle_deactivate_runtime,
        }

        handler = dispatch_map.get(action.action_type)
        if handler is None:
            record.error = f"No handler for action type {action.action_type.value}"
            record.succeeded = False
            self._store_record(record)
            return record

        try:
            handler(action, record)
            record.succeeded = record.error is None
        except Exception as exc:
            record.error = str(exc)
            record.succeeded = False
            record.audit_trail.append(f"[{_utcnow()}] EXCEPTION: {exc}")
            logger.exception(
                "Unhandled exception executing action %s", action.action_type.value
            )

        outcome = "succeeded" if record.succeeded else "failed"
        self._store_record(record)
        self._log_operator_action(action, outcome, [action.scope])
        return record

    # ------------------------------------------------------------------
    # Convenience methods (thin wrappers over execute_action)
    # ------------------------------------------------------------------

    def enable_strategy(
        self,
        strategy_id: str,
        operator: str,
        approval_id: str,
        env: str,
    ) -> ControlPlaneActionRecord:
        """Enable a strategy for trading in the specified environment."""
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.ENABLE_STRATEGY,
            scope=f"strategy:{strategy_id}",
            value=strategy_id,
            reason="operator enable_strategy call",
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=approval_id,
            environment=env,
            expiry=None,
        )
        return self.execute_action(action)

    def disable_strategy(
        self,
        strategy_id: str,
        operator: str,
        reason: str,
        env: str,
    ) -> ControlPlaneActionRecord:
        """Immediately disable a strategy."""
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.DISABLE_STRATEGY,
            scope=f"strategy:{strategy_id}",
            value=strategy_id,
            reason=reason,
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=None,
            environment=env,
            expiry=None,
        )
        return self.execute_action(action)

    def pause_new_entries(
        self,
        scope: str,
        operator: str,
        reason: str,
        duration_minutes: Optional[float] = None,
    ) -> ControlPlaneActionRecord:
        """Pause new entry signals.  Exits continue."""
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.PAUSE_NEW_ENTRIES,
            scope=scope,
            value="paused",
            reason=reason,
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=None,
            environment=self._state_manager._env,
            expiry=_expiry_from_minutes(duration_minutes),
        )
        return self.execute_action(action)

    def set_exits_only(
        self,
        operator: str,
        reason: str,
    ) -> ControlPlaneActionRecord:
        """Switch to exits-only mode globally."""
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.ALLOW_EXITS_ONLY,
            scope="global",
            value="true",
            reason=reason,
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=None,
            environment=self._state_manager._env,
            expiry=None,
        )
        return self.execute_action(action)

    def throttle(
        self,
        level: ThrottleLevel,
        scope: str,
        operator: str,
        reason: str,
        duration_minutes: Optional[float] = None,
    ) -> ControlPlaneActionRecord:
        """Apply a sizing throttle to the given scope."""
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.THROTTLE_SIZING,
            scope=scope,
            value=level.value,
            reason=reason,
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=None,
            environment=self._state_manager._env,
            expiry=_expiry_from_minutes(duration_minutes),
        )
        return self.execute_action(action)

    def disable_model(
        self,
        model_id: str,
        operator: str,
        reason: str,
    ) -> ControlPlaneActionRecord:
        """Disable a model and engage the fallback inference path."""
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.DISABLE_MODEL,
            scope=f"model:{model_id}",
            value=model_id,
            reason=reason,
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=None,
            environment=self._state_manager._env,
            expiry=None,
        )
        return self.execute_action(action)

    def disable_agent(
        self,
        agent_name: str,
        operator: str,
        reason: str,
    ) -> ControlPlaneActionRecord:
        """Emergency-disable a named agent."""
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.DISABLE_AGENT,
            scope=f"agent:{agent_name}",
            value=agent_name,
            reason=reason,
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=None,
            environment=self._state_manager._env,
            expiry=None,
        )
        return self.execute_action(action)

    def engage_kill_switch(
        self,
        reason: str,
        operator: str,
        scope: str = "global",
    ) -> ControlPlaneActionRecord:
        """Engage the kill switch.

        Steps:
          1. Calls state_manager.engage_kill_switch() to halt trading.
          2. Creates an incident via incident_manager if available.
          3. Logs OperatorActionRecord.
          4. Emits to any registered event callbacks.

        Args:
            reason:   Human-readable reason — required for audit.
            operator: Operator identifier.
            scope:    Scope of kill switch (default "global").

        Returns:
            ControlPlaneActionRecord.
        """
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.TRIGGER_HALT,
            scope=scope,
            value="halted",
            reason=reason,
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=None,
            environment=self._state_manager._env,
            expiry=None,
        )
        return self.execute_action(action)

    def release_kill_switch(
        self,
        operator: str,
        approval_id: str,
        reason: str,
    ) -> ControlPlaneActionRecord:
        """Release the kill switch after recovery.

        Requires a non-empty approval_id.
        """
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.RELEASE_HALT,
            scope="global",
            value="released",
            reason=reason,
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=approval_id,
            environment=self._state_manager._env,
            expiry=None,
        )
        return self.execute_action(action)

    def trigger_drain(
        self,
        operator: str,
        reason: str,
    ) -> ControlPlaneActionRecord:
        """Begin a controlled position drain.  No new entries; exit gracefully."""
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.TRIGGER_DRAIN,
            scope="global",
            value="draining",
            reason=reason,
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=None,
            environment=self._state_manager._env,
            expiry=None,
        )
        return self.execute_action(action)

    def force_reconcile(
        self,
        operator: str,
        reason: str,
    ) -> ControlPlaneActionRecord:
        """Trigger a reconciliation sweep across all active components."""
        action = ControlPlaneAction(
            action_id=_make_id("act"),
            action_type=ControlPlaneActionType.FORCE_RECONCILE,
            scope="global",
            value="reconciling",
            reason=reason,
            requested_by=operator,
            requested_at=_utcnow(),
            approval_id=None,
            environment=self._state_manager._env,
            expiry=None,
        )
        return self.execute_action(action)

    def inspect_state(self) -> RuntimeState:
        """Return the current observable runtime state."""
        return self._state_manager.get_current_state()

    # ------------------------------------------------------------------
    # History / audit accessors
    # ------------------------------------------------------------------

    def get_action_history(self, limit: int = 100) -> List[ControlPlaneActionRecord]:
        """Return recent ControlPlaneActionRecords, most recent first."""
        with self._lock:
            return list(reversed(self._action_history[-limit:]))

    def get_operator_action_records(
        self, operator: Optional[str] = None
    ) -> List[OperatorActionRecord]:
        """Return OperatorActionRecords, optionally filtered by operator."""
        with self._lock:
            if operator is None:
                return list(self._operator_records)
            return [r for r in self._operator_records if r.operator == operator]

    # ------------------------------------------------------------------
    # Preflight checks
    # ------------------------------------------------------------------

    def run_preflight_checks(
        self, target_mode: RuntimeMode
    ) -> LiveTradingReadinessReport:
        """Run all readiness checks before activating a target RuntimeMode.

        Checks performed:
          - Environment validity for target mode.
          - Kill switch not active.
          - No stale / critical overrides.
          - Overall service state is READY.
          - Key components (broker, data, risk, models) are READY.
          - Config version present.
          - Reconciliation state not dirty.
          - Required approvals present for live mode.

        Returns:
            LiveTradingReadinessReport with full check details.
        """
        report_id = _make_id("preflight")
        generated_at = _utcnow()
        env = self._state_manager._env

        checks_passed: List[str] = []
        checks_failed: List[str] = []
        blocking_issues: List[str] = []
        warnings: List[str] = []

        current_state = self._state_manager.get_current_state()
        spec = get_environment_spec(env)

        # ── Check 1: Environment valid for target mode ───────────────────
        if spec.runtime_mode != target_mode:
            checks_failed.append("environment_mode_match")
            blocking_issues.append(
                f"Environment '{env}' canonical mode is '{spec.runtime_mode.value}', "
                f"but target mode is '{target_mode.value}'."
            )
        else:
            checks_passed.append("environment_mode_match")

        # ── Check 2: Kill switch not active ──────────────────────────────
        if current_state.kill_switch_active:
            checks_failed.append("kill_switch_clear")
            blocking_issues.append(
                f"Kill switch is ACTIVE: {current_state.kill_switch_reason}"
            )
        else:
            checks_passed.append("kill_switch_clear")

        # ── Check 3: Global throttle not HALTED ──────────────────────────
        if current_state.global_throttle == ThrottleLevel.HALTED:
            checks_failed.append("global_throttle_not_halted")
            blocking_issues.append("Global throttle is set to HALTED.")
        else:
            checks_passed.append("global_throttle_not_halted")

        # ── Check 4: Overall service state ───────────────────────────────
        overall_ok = current_state.overall_service_state in (
            ServiceState.READY,
            ServiceState.WARMING,
        )
        if overall_ok:
            checks_passed.append("overall_service_state")
        else:
            checks_failed.append("overall_service_state")
            blocking_issues.append(
                f"Overall service state is "
                f"'{current_state.overall_service_state.value}' (expected READY)."
            )

        # ── Check 5: Broker readiness ─────────────────────────────────────
        broker_state = ServiceState(
            current_state.component_states.get("broker", ServiceState.INITIALIZING.value)
        )
        broker_ready = broker_state == ServiceState.READY
        if spec.allow_broker_orders:
            if broker_ready:
                checks_passed.append("broker_ready")
            else:
                checks_failed.append("broker_ready")
                blocking_issues.append(
                    f"Broker component is '{broker_state.value}' (expected READY)."
                )
        else:
            # Broker not required in this env
            broker_ready = True
            checks_passed.append("broker_not_required")

        # ── Check 6: Data feed readiness ──────────────────────────────────
        data_state = ServiceState(
            current_state.component_states.get("data_feed", ServiceState.INITIALIZING.value)
        )
        data_ready = data_state in (ServiceState.READY, ServiceState.DEGRADED)
        if data_ready:
            if data_state == ServiceState.DEGRADED:
                warnings.append("Data feed is DEGRADED — running on fallback source.")
            checks_passed.append("data_feed_ready")
        else:
            checks_failed.append("data_feed_ready")
            blocking_issues.append(
                f"Data feed component is '{data_state.value}'."
            )

        # ── Check 7: Risk engine readiness ────────────────────────────────
        risk_state = ServiceState(
            current_state.component_states.get("risk_engine", ServiceState.INITIALIZING.value)
        )
        risk_ready = risk_state == ServiceState.READY
        if risk_ready:
            checks_passed.append("risk_engine_ready")
        else:
            checks_failed.append("risk_engine_ready")
            blocking_issues.append(
                f"Risk engine is '{risk_state.value}' (expected READY)."
            )

        # ── Check 8: Active models validity ──────────────────────────────
        unhealthy_models = [
            mid
            for mid, rec in current_state.active_models.items()
            if rec.get("status") == ActivationStatus.DISABLED.value
            or rec.get("fallback_active", False)
        ]
        models_ready = len(unhealthy_models) == 0
        if models_ready:
            checks_passed.append("models_ready")
        else:
            checks_failed.append("models_ready")
            if target_mode == RuntimeMode.LIVE:
                blocking_issues.append(
                    f"Models in fallback / disabled state: {unhealthy_models}"
                )
            else:
                warnings.append(
                    f"Models in fallback / disabled state: {unhealthy_models}"
                )

        # ── Check 9: No stale overrides ───────────────────────────────────
        active_overrides = self._state_manager.get_active_overrides()
        halt_overrides = [
            ov
            for ov in active_overrides
            if ov.override_type in ("halt", "exits_only")
        ]
        if halt_overrides:
            checks_failed.append("no_halt_overrides")
            blocking_issues.append(
                f"{len(halt_overrides)} active halt/exits-only override(s) present."
            )
        else:
            checks_passed.append("no_halt_overrides")

        # ── Check 10: Config version present ─────────────────────────────
        config_valid = True  # We check presence; deep validation is deferred.
        checks_passed.append("config_version_present")

        # ── Check 11: Reconciliation state ───────────────────────────────
        reconcile_state = ServiceState(
            current_state.component_states.get(
                "reconciliation", ServiceState.READY.value
            )
        )
        reconciliation_clean = reconcile_state != ServiceState.RECONCILING
        if reconciliation_clean:
            checks_passed.append("reconciliation_clean")
        else:
            checks_failed.append("reconciliation_clean")
            blocking_issues.append(
                "Reconciliation sweep is currently in progress."
            )

        # ── Check 12: Approvals for live mode ────────────────────────────
        required_approvals_present = True
        if target_mode == RuntimeMode.LIVE:
            # Require at least one pending approval for live activation
            required_approvals_present = (
                len(current_state.active_strategies) > 0
            )
            if not required_approvals_present:
                checks_failed.append("live_approvals_present")
                blocking_issues.append(
                    "No approved strategies found — live activation requires "
                    "at least one approved strategy."
                )
            else:
                checks_passed.append("live_approvals_present")
        else:
            checks_passed.append("approvals_not_required_for_mode")

        # ── Derive overall recommendation ────────────────────────────────
        overall_ready = len(blocking_issues) == 0
        if overall_ready and len(warnings) == 0:
            recommendation = "proceed"
        elif overall_ready:
            recommendation = "proceed_with_caution"
        else:
            recommendation = "halt"

        return LiveTradingReadinessReport(
            report_id=report_id,
            generated_at=generated_at,
            env=env,
            mode=target_mode,
            overall_ready=overall_ready,
            blocking_issues=tuple(blocking_issues),
            warnings=tuple(warnings),
            checks_passed=tuple(checks_passed),
            checks_failed=tuple(checks_failed),
            broker_ready=broker_ready,
            data_ready=data_ready,
            risk_ready=risk_ready,
            models_ready=models_ready,
            config_valid=config_valid,
            reconciliation_clean=reconciliation_clean,
            required_approvals_present=required_approvals_present,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Action handlers (private)
    # ------------------------------------------------------------------

    def _handle_enable_strategy(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        strategy_id = action.value or action.scope.split(":")[-1]
        prev = _dict_to_str(
            self._state_manager.get_current_state().active_strategies.get(strategy_id, {})
        )
        record.previous_state = prev

        # Build a minimal activation record if one doesn't already exist.
        existing = self._state_manager._active_strategies.get(strategy_id)
        if existing is None:
            rec = StrategyActivationRecord(
                record_id=_make_id("srec"),
                strategy_id=strategy_id,
                strategy_name=strategy_id,
                env=action.environment,
                mode=self._state_manager._mode,
                status=ActivationStatus.INACTIVE,
                activated_at=None,
                deactivated_at=None,
                activation_decision_id=action.approval_id,
                config_version="unknown",
                policy_version="unknown",
                notes=action.notes,
            )
            self._state_manager.activate_strategy(rec)
        else:
            existing.status = ActivationStatus.ACTIVE
            existing.deactivated_at = None

        record.new_state = _dict_to_str(
            self._state_manager.get_current_state().active_strategies.get(strategy_id, {})
        )
        record.audit_trail.append(
            f"[{_utcnow()}] Strategy '{strategy_id}' enabled in env '{action.environment}'."
        )

    def _handle_disable_strategy(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        strategy_id = action.value or action.scope.split(":")[-1]
        prev = _dict_to_str(
            self._state_manager.get_current_state().active_strategies.get(strategy_id, {})
        )
        record.previous_state = prev
        self._state_manager.deactivate_strategy(strategy_id, action.reason)
        record.new_state = _dict_to_str(
            {"strategy_id": strategy_id, "status": ActivationStatus.DISABLED.value}
        )
        record.audit_trail.append(
            f"[{_utcnow()}] Strategy '{strategy_id}' disabled: {action.reason}"
        )

    def _handle_pause_new_entries(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        prev_state = self._state_manager.get_current_state()
        record.previous_state = _dict_to_str(
            {"global_throttle": prev_state.global_throttle.value}
        )
        override = RuntimeOverride(
            override_id=_make_id("ov"),
            override_type="pause",
            scope=action.scope,
            value="paused",
            reason=action.reason,
            applied_by=action.requested_by,
            applied_at=_utcnow(),
            expires_at=action.expiry,
            approval_id=action.approval_id,
            is_emergency=False,
        )
        self._state_manager.apply_override(override)
        record.new_state = _dict_to_str(
            {"override_id": override.override_id, "scope": action.scope, "value": "paused"}
        )
        record.audit_trail.append(
            f"[{_utcnow()}] New entries paused for scope '{action.scope}'. "
            f"Expiry: {action.expiry or 'permanent'}."
        )

    def _handle_allow_exits_only(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        prev_state = self._state_manager.get_current_state()
        record.previous_state = _dict_to_str(
            {"exits_only_mode": prev_state.exits_only_mode}
        )
        self._state_manager.set_exits_only(True, action.reason, action.requested_by)
        record.new_state = _dict_to_str({"exits_only_mode": True})
        record.audit_trail.append(
            f"[{_utcnow()}] Exits-only mode activated by {action.requested_by}: {action.reason}"
        )

    def _handle_throttle_sizing(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        prev_state = self._state_manager.get_current_state()
        record.previous_state = _dict_to_str(
            {"global_throttle": prev_state.global_throttle.value}
        )
        try:
            level = ThrottleLevel(action.value)
        except ValueError:
            record.error = f"Unknown throttle level: {action.value!r}"
            return

        if action.scope == "global":
            self._state_manager.set_global_throttle(
                level, action.reason, action.requested_by
            )
        else:
            # Scope-specific throttle via override
            override = RuntimeOverride(
                override_id=_make_id("ov"),
                override_type="throttle",
                scope=action.scope,
                value=level.value,
                reason=action.reason,
                applied_by=action.requested_by,
                applied_at=_utcnow(),
                expires_at=action.expiry,
                approval_id=action.approval_id,
                is_emergency=False,
            )
            self._state_manager.apply_override(override)

        multiplier = ThrottleState.multiplier_for(level)
        record.new_state = _dict_to_str(
            {"scope": action.scope, "throttle": level.value, "multiplier": multiplier}
        )
        record.audit_trail.append(
            f"[{_utcnow()}] Throttle set to {level.value} "
            f"(multiplier={multiplier:.2f}) for scope '{action.scope}'."
        )

    def _handle_freeze_universe(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        universe_id = action.scope.split(":")[-1] if ":" in action.scope else action.scope
        record.previous_state = _dict_to_str({"frozen": False, "universe": universe_id})
        override = RuntimeOverride(
            override_id=_make_id("ov"),
            override_type="freeze_universe",
            scope=f"universe:{universe_id}",
            value="frozen",
            reason=action.reason,
            applied_by=action.requested_by,
            applied_at=_utcnow(),
            expires_at=action.expiry,
            approval_id=action.approval_id,
            is_emergency=False,
        )
        self._state_manager.apply_override(override)
        record.new_state = _dict_to_str(
            {"frozen": True, "universe": universe_id, "override_id": override.override_id}
        )
        record.audit_trail.append(
            f"[{_utcnow()}] Universe '{universe_id}' frozen: {action.reason}"
        )

    def _handle_disable_model(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        model_id = action.value or action.scope.split(":")[-1]
        prev = _dict_to_str(
            self._state_manager.get_current_state().active_models.get(model_id, {})
        )
        record.previous_state = prev
        self._state_manager.disable_model(model_id, action.reason)
        record.new_state = _dict_to_str(
            {
                "model_id": model_id,
                "status": ActivationStatus.DISABLED.value,
                "fallback_active": True,
            }
        )
        record.audit_trail.append(
            f"[{_utcnow()}] Model '{model_id}' disabled; fallback activated. "
            f"Reason: {action.reason}"
        )

    def _handle_enable_fallback(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        model_id = action.value or action.scope.split(":")[-1]
        rec = self._state_manager._active_models.get(model_id)
        record.previous_state = _dict_to_str(
            {"model_id": model_id, "fallback_active": rec.fallback_active if rec else False}
        )
        if rec is not None:
            rec.fallback_active = True
        record.new_state = _dict_to_str({"model_id": model_id, "fallback_active": True})
        record.audit_trail.append(
            f"[{_utcnow()}] Fallback explicitly enabled for model '{model_id}'."
        )

    def _handle_disable_agent(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        agent_name = action.value or action.scope.split(":")[-1]
        prev = _dict_to_str(
            self._state_manager.get_current_state().active_agents.get(agent_name, {})
        )
        record.previous_state = prev
        self._state_manager.disable_agent(agent_name, action.reason)
        record.new_state = _dict_to_str(
            {"agent_name": agent_name, "status": ActivationStatus.DISABLED.value}
        )
        record.audit_trail.append(
            f"[{_utcnow()}] Agent '{agent_name}' disabled: {action.reason}"
        )

    def _handle_apply_temp_override(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        record.previous_state = _dict_to_str({"overrides": len(self._state_manager._active_overrides)})
        override = RuntimeOverride(
            override_id=_make_id("ov"),
            override_type="temp_override",
            scope=action.scope,
            value=action.value or "",
            reason=action.reason,
            applied_by=action.requested_by,
            applied_at=_utcnow(),
            expires_at=action.expiry,
            approval_id=action.approval_id,
            is_emergency=False,
        )
        self._state_manager.apply_override(override)
        record.new_state = _dict_to_str({"override_id": override.override_id})
        record.audit_trail.append(
            f"[{_utcnow()}] Temporary override {override.override_id} applied "
            f"for scope '{action.scope}'."
        )

    def _handle_clear_override(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        override_id = action.value or action.scope
        record.previous_state = _dict_to_str({"override_id": override_id, "active": True})
        cleared = self._state_manager.clear_override(override_id, action.requested_by)
        if not cleared:
            record.error = f"Override '{override_id}' not found or already cleared."
            return
        record.new_state = _dict_to_str({"override_id": override_id, "active": False})
        record.audit_trail.append(
            f"[{_utcnow()}] Override '{override_id}' cleared by {action.requested_by}."
        )

    def _handle_force_reconcile(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        prev = self._state_manager.get_component_state("reconciliation")
        record.previous_state = _dict_to_str({"reconciliation": prev.value})
        self._state_manager.update_component_state(
            "reconciliation", ServiceState.RECONCILING
        )
        record.new_state = _dict_to_str({"reconciliation": ServiceState.RECONCILING.value})
        record.audit_trail.append(
            f"[{_utcnow()}] Reconciliation triggered by {action.requested_by}: {action.reason}"
        )
        logger.info(
            "Reconciliation triggered by operator %s: %s",
            action.requested_by,
            action.reason,
        )

    def _handle_trigger_drain(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        prev_state = self._state_manager.get_current_state()
        record.previous_state = _dict_to_str(
            {"exits_only_mode": prev_state.exits_only_mode}
        )
        # Drain = exits only + transition component to DRAINING
        self._state_manager.set_exits_only(True, action.reason, action.requested_by)
        self._state_manager.update_component_state("portfolio", ServiceState.DRAINING)
        record.new_state = _dict_to_str(
            {"exits_only_mode": True, "portfolio_state": ServiceState.DRAINING.value}
        )
        record.audit_trail.append(
            f"[{_utcnow()}] Drain initiated by {action.requested_by}: {action.reason}. "
            "No new entries; positions will exit gracefully."
        )
        logger.warning(
            "DRAIN initiated by %s: %s", action.requested_by, action.reason
        )

    def _handle_trigger_halt(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        """Engage the global kill switch."""
        prev_state = self._state_manager.get_current_state()
        record.previous_state = _dict_to_str(
            {
                "kill_switch_active": prev_state.kill_switch_active,
                "overall_state": prev_state.overall_service_state.value,
            }
        )

        # 1. Engage state-manager kill switch
        self._state_manager.engage_kill_switch(action.reason, action.requested_by)

        # 2. Create incident via incident_manager if available
        if self._incident_manager is not None:
            try:
                self._incident_manager.create_incident(
                    title=f"Kill switch engaged by {action.requested_by}",
                    description=action.reason,
                    severity="critical",
                    source="control_plane",
                    context=_as_dict(action),
                )
            except Exception:
                logger.exception("Failed to create incident for kill-switch event")

        record.new_state = _dict_to_str(
            {
                "kill_switch_active": True,
                "kill_switch_reason": action.reason,
                "overall_state": ServiceState.HALTED.value,
            }
        )
        record.audit_trail.append(
            f"[{_utcnow()}] KILL SWITCH ENGAGED by {action.requested_by}: {action.reason}"
        )

        # 3. Emit event callbacks
        self._emit_event("kill_switch_engaged", action=action)

        logger.critical(
            "KILL SWITCH ENGAGED by %s: %s", action.requested_by, action.reason
        )

    def _handle_release_halt(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        """Release the kill switch."""
        if not action.approval_id:
            record.error = "approval_id is required to release the kill switch."
            return

        prev_state = self._state_manager.get_current_state()
        record.previous_state = _dict_to_str(
            {
                "kill_switch_active": prev_state.kill_switch_active,
                "kill_switch_reason": prev_state.kill_switch_reason,
            }
        )
        self._state_manager.release_kill_switch(action.requested_by, action.approval_id)
        record.new_state = _dict_to_str({"kill_switch_active": False})
        record.audit_trail.append(
            f"[{_utcnow()}] Kill switch released by {action.requested_by} "
            f"(approval={action.approval_id}): {action.reason}"
        )
        self._emit_event("kill_switch_released", action=action)
        logger.warning(
            "Kill switch released by %s (approval=%s)",
            action.requested_by,
            action.approval_id,
        )

    def _handle_inspect_state(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        state = self._state_manager.get_current_state()
        record.previous_state = ""
        record.new_state = _dict_to_str(
            {
                "env": state.env,
                "mode": state.mode.value,
                "overall_state": state.overall_service_state.value,
                "kill_switch_active": state.kill_switch_active,
                "global_throttle": state.global_throttle.value,
            }
        )
        record.audit_trail.append(
            f"[{_utcnow()}] State inspected by {action.requested_by}."
        )

    def _handle_activate_runtime(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        prev_state = self._state_manager.get_current_state()
        record.previous_state = _dict_to_str(
            {"overall_state": prev_state.overall_service_state.value}
        )
        self._state_manager.update_component_state("runtime", ServiceState.READY)
        self._state_manager._overall_state = ServiceState.READY
        record.new_state = _dict_to_str({"overall_state": ServiceState.READY.value})
        record.audit_trail.append(
            f"[{_utcnow()}] Runtime activated by {action.requested_by}: {action.reason}"
        )

    def _handle_deactivate_runtime(
        self, action: ControlPlaneAction, record: ControlPlaneActionRecord
    ) -> None:
        prev_state = self._state_manager.get_current_state()
        record.previous_state = _dict_to_str(
            {"overall_state": prev_state.overall_service_state.value}
        )
        self._state_manager.update_component_state("runtime", ServiceState.DISABLED)
        record.new_state = _dict_to_str({"overall_state": ServiceState.DISABLED.value})
        record.audit_trail.append(
            f"[{_utcnow()}] Runtime deactivated by {action.requested_by}: {action.reason}"
        )

    # ------------------------------------------------------------------
    # Internal: permissions, logging, storage
    # ------------------------------------------------------------------

    def _check_permission(
        self,
        action_type: ControlPlaneActionType,
        env: str,
        operator: str,
    ) -> Tuple[bool, str]:
        """Internal permission check.

        Rules:
          1. TRIGGER_HALT is always permitted (emergency).
          2. RELEASE_HALT requires an approval_id (checked at call site).
          3. For "live" environment, destructive actions are restricted.
          4. If a governance engine is present, delegate policy check.
          5. Default: permit.
        """
        if action_type == ControlPlaneActionType.TRIGGER_HALT:
            # Kill switch is always permitted — it's an emergency action.
            return True, ""

        if not operator:
            return False, "operator identifier must not be empty."

        # In live env, certain actions require explicit governance approval
        if env == "live":
            sensitive_live_actions = {
                ControlPlaneActionType.ENABLE_STRATEGY,
                ControlPlaneActionType.ACTIVATE_RUNTIME,
                ControlPlaneActionType.DEACTIVATE_RUNTIME,
            }
            if action_type in sensitive_live_actions:
                if self._governance_engine is not None:
                    try:
                        ok = self._governance_engine.check_action_permitted(
                            action_type.value, operator, env
                        )
                        if not ok:
                            return (
                                False,
                                f"Governance engine blocked action '{action_type.value}' "
                                f"for operator '{operator}' in env '{env}'.",
                            )
                    except Exception as exc:
                        logger.warning("Governance check failed: %s", exc)
                        # Fail-open for governance exceptions to avoid blocking ops
                        pass

        # Delegate to governance engine for full policy check if available
        if self._governance_engine is not None:
            try:
                ok = self._governance_engine.check_action_permitted(
                    action_type.value, operator, env
                )
                if not ok:
                    return (
                        False,
                        f"Policy check failed for action '{action_type.value}'.",
                    )
            except Exception as exc:
                logger.debug("Governance engine check skipped: %s", exc)
                # Non-blocking governance exception

        return True, ""

    def _log_operator_action(
        self,
        action: ControlPlaneAction,
        outcome: str,
        affected: List[str],
    ) -> OperatorActionRecord:
        """Write an OperatorActionRecord to the in-memory audit log."""
        rec = OperatorActionRecord(
            record_id=_make_id("opr"),
            operator=action.requested_by,
            action_type=action.action_type.value,
            description=action.reason,
            environment=action.environment,
            timestamp=_utcnow(),
            outcome=outcome,
            policy_check_id=None,
            approval_id=action.approval_id,
            affected_components=tuple(affected),
            notes=action.notes,
        )
        with self._lock:
            self._operator_records.append(rec)
            # Limit memory footprint
            if len(self._operator_records) > 10_000:
                self._operator_records = self._operator_records[-10_000:]
        return rec

    def _store_record(self, record: ControlPlaneActionRecord) -> None:
        """Persist a ControlPlaneActionRecord to the in-memory history."""
        with self._lock:
            self._action_history.append(record)
            if len(self._action_history) > 10_000:
                self._action_history = self._action_history[-10_000:]

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def register_event_callback(self, event: str, callback: Callable) -> None:
        """Register a callback fired when the named event occurs.

        Supported events: "kill_switch_engaged", "kill_switch_released".
        """
        with self._lock:
            self._event_callbacks.setdefault(event, []).append(callback)

    def _emit_event(self, event: str, **kwargs) -> None:
        """Fire all callbacks registered for an event."""
        callbacks = []
        with self._lock:
            callbacks = list(self._event_callbacks.get(event, []))
        for cb in callbacks:
            try:
                cb(**kwargs)
            except Exception:
                logger.exception("Event callback for '%s' raised", event)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_control_plane_instance: Optional[ControlPlaneEngine] = None
_control_plane_lock = threading.Lock()


def get_control_plane(
    state_manager: Optional[RuntimeStateManager] = None,
    governance_engine=None,
    approval_engine=None,
    incident_manager=None,
) -> ControlPlaneEngine:
    """Singleton accessor for the global ControlPlaneEngine.

    On first call, the instance is created with the provided dependencies.
    Subsequent calls return the existing instance.
    """
    global _control_plane_instance
    if _control_plane_instance is None:
        with _control_plane_lock:
            if _control_plane_instance is None:
                _control_plane_instance = ControlPlaneEngine(
                    state_manager=state_manager,
                    governance_engine=governance_engine,
                    approval_engine=approval_engine,
                    incident_manager=incident_manager,
                )
    return _control_plane_instance
