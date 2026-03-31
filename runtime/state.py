# -*- coding: utf-8 -*-
"""
runtime/state.py — RuntimeStateManager
========================================

Single source of truth for all observable and desired runtime state.

Responsibilities:
  - Maintain the current RuntimeState (observed world).
  - Maintain the DesiredRuntimeState (operator intent).
  - Track active overrides with expiry-based auto-clearance.
  - Record component-level ServiceState updates.
  - Persist snapshots to SQL store if one is provided.
  - Expose is_safe_to_trade() — conservative gating for the control plane.

Thread safety:
  All mutations are serialised through a single threading.Lock.
  Read-only accessors acquire the lock briefly to capture a consistent view.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from runtime.contracts import (
    ActivationStatus,
    AgentActivationRecord,
    DesiredRuntimeState,
    ModelActivationRecord,
    RuntimeMode,
    RuntimeOverride,
    RuntimeState,
    ServiceState,
    StrategyActivationRecord,
    ThrottleLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _make_id(prefix: str = "snap") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _is_expired(expires_at: Optional[str]) -> bool:
    """Return True if the expires_at timestamp has passed."""
    if expires_at is None:
        return False
    try:
        expiry = datetime.fromisoformat(expires_at)
        # Ensure timezone-aware comparison.
        now = datetime.now(timezone.utc)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return now >= expiry
    except (ValueError, TypeError):
        return False


def _as_dict(obj) -> dict:
    """Safely convert a dataclass instance to a plain dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {}


# ---------------------------------------------------------------------------
# RuntimeStateManager
# ---------------------------------------------------------------------------


class RuntimeStateManager:
    """Single source of truth for all runtime state.

    Maintains:
    - Current RuntimeState (observed: what is actually running).
    - DesiredRuntimeState (operator intent: what should be running).
    - Active overrides with expiry tracking.
    - Pending transitions.
    - Per-component ServiceState.

    Thread-safe via an internal threading.Lock.  All public methods acquire
    the lock before mutating or reading shared state.

    An optional SQL store (any object with a .store_json(table, key, data)
    method) can be injected for persistent snapshots.
    """

    def __init__(
        self,
        env: str = "paper",
        mode: RuntimeMode = RuntimeMode.PAPER,
        sql_store=None,
        *,
        on_kill_switch_engaged: Optional[Callable[[str, str], None]] = None,
        on_override_applied: Optional[Callable[[RuntimeOverride], None]] = None,
    ) -> None:
        self._env = env
        self._mode = mode
        self._sql_store = sql_store

        # Callbacks
        self._on_kill_switch_engaged = on_kill_switch_engaged
        self._on_override_applied = on_override_applied

        # Core mutable state
        self._lock = threading.Lock()
        self._kill_switch_active: bool = False
        self._kill_switch_reason: Optional[str] = None
        self._exits_only_mode: bool = False
        self._global_throttle: ThrottleLevel = ThrottleLevel.NONE
        self._overall_state: ServiceState = ServiceState.INITIALIZING

        self._active_strategies: Dict[str, StrategyActivationRecord] = {}
        self._active_models: Dict[str, ModelActivationRecord] = {}
        self._active_agents: Dict[str, AgentActivationRecord] = {}
        self._active_overrides: Dict[str, RuntimeOverride] = {}
        self._expired_overrides: Dict[str, RuntimeOverride] = {}
        self._component_states: Dict[str, ServiceState] = {}
        self._pending_transitions: list = []

        self._desired_state: Optional[DesiredRuntimeState] = None
        self._audit_log: List[dict] = []   # in-memory ring buffer (last 2 000 entries)
        self._metrics: Dict[str, int] = {
            "overrides_applied": 0,
            "overrides_cleared": 0,
            "overrides_expired": 0,
            "kill_switch_engagements": 0,
            "kill_switch_releases": 0,
            "strategy_activations": 0,
            "strategy_deactivations": 0,
            "model_activations": 0,
            "model_disables": 0,
            "agent_activations": 0,
            "state_mutations": 0,
        }

        self._last_updated: str = _utcnow()
        logger.info(
            "RuntimeStateManager initialised — env=%s mode=%s", env, mode.value
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _touch(self) -> None:
        """Update last_updated timestamp and increment mutation counter."""
        self._last_updated = _utcnow()
        self._metrics["state_mutations"] += 1

    def _audit(self, event: str, **kwargs) -> None:
        """Append an audit entry to the in-memory ring buffer."""
        entry = {"ts": _utcnow(), "event": event, **kwargs}
        self._audit_log.append(entry)
        if len(self._audit_log) > 2_000:
            self._audit_log = self._audit_log[-2_000:]
        logger.debug("audit | %s", entry)

    def _recompute_overall_state(self) -> None:
        """Derive overall_service_state from component states and flags.

        Priority (highest wins):
          HALTED > UNHEALTHY > DEGRADED > RECOVERING > RECONCILING >
          DRAINING > PAUSED > WARMING > READY > INITIALIZING
        """
        if self._kill_switch_active:
            self._overall_state = ServiceState.HALTED
            return

        component_vals = list(self._component_states.values())
        priority: List[ServiceState] = [
            ServiceState.UNHEALTHY,
            ServiceState.DEGRADED,
            ServiceState.RECOVERING,
            ServiceState.RECONCILING,
            ServiceState.DRAINING,
            ServiceState.PAUSED,
            ServiceState.WARMING,
            ServiceState.READY,
            ServiceState.INITIALIZING,
        ]
        if not component_vals:
            # No components registered yet — stay in current state
            return
        for state in priority:
            if state in component_vals:
                self._overall_state = state
                return
        self._overall_state = ServiceState.READY

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_current_state(self) -> RuntimeState:
        """Return a fully populated snapshot of the current observable state."""
        with self._lock:
            return self._build_state_snapshot()

    def get_desired_state(self) -> Optional[DesiredRuntimeState]:
        """Return the current desired state, or None if not yet set."""
        with self._lock:
            return self._desired_state

    def set_desired_state(self, desired: DesiredRuntimeState) -> None:
        """Update the desired state.  Logs drift from the current observed state."""
        with self._lock:
            self._desired_state = desired
            drift = self._compute_diff()
            self._audit(
                "desired_state_updated",
                created_by=desired.created_by,
                drift=drift,
            )
            self._touch()
            if drift.get("drifted_keys") or drift.get("extra_active") or drift.get("missing_active"):
                logger.warning(
                    "Desired state drift detected: drifted_keys=%s extra_active=%s missing_active=%s",
                    drift.get("drifted_keys"),
                    drift.get("extra_active"),
                    drift.get("missing_active"),
                )

    def diff(self) -> dict:
        """Compare current vs desired state.

        Returns a dict with keys:
          drifted_keys:   fields that differ between current and desired
          extra_active:   strategies/models/agents active but not desired
          missing_active: strategies/models/agents desired but not active
        """
        with self._lock:
            return self._compute_diff()

    def _compute_diff(self) -> dict:
        """Internal diff computation — must be called with self._lock held."""
        result: dict = {
            "drifted_keys": [],
            "extra_active": [],
            "missing_active": [],
        }
        if self._desired_state is None:
            return result

        d = self._desired_state
        if d.mode != self._mode:
            result["drifted_keys"].append(f"mode: current={self._mode.value} desired={d.mode.value}")
        if d.global_throttle != self._global_throttle:
            result["drifted_keys"].append(
                f"global_throttle: current={self._global_throttle.value} desired={d.global_throttle.value}"
            )

        current_strategies = set(self._active_strategies.keys())
        desired_strategies = set(d.enabled_strategies)
        result["extra_active"].extend(
            [f"strategy:{s}" for s in current_strategies - desired_strategies]
        )
        result["missing_active"].extend(
            [f"strategy:{s}" for s in desired_strategies - current_strategies]
        )

        current_models = set(self._active_models.keys())
        desired_models = set(d.enabled_models)
        result["extra_active"].extend(
            [f"model:{m}" for m in current_models - desired_models]
        )
        result["missing_active"].extend(
            [f"model:{m}" for m in desired_models - current_models]
        )

        current_agents = set(self._active_agents.keys())
        desired_agents = set(d.enabled_agents)
        result["extra_active"].extend(
            [f"agent:{a}" for a in current_agents - desired_agents]
        )
        result["missing_active"].extend(
            [f"agent:{a}" for a in desired_agents - current_agents]
        )

        return result

    # ------------------------------------------------------------------
    # Component state
    # ------------------------------------------------------------------

    def update_component_state(self, component: str, state: ServiceState) -> None:
        """Update a named component's ServiceState and recompute the overall state."""
        with self._lock:
            prev = self._component_states.get(component)
            self._component_states[component] = state
            self._recompute_overall_state()
            self._audit(
                "component_state_updated",
                component=component,
                prev=prev.value if prev else None,
                new=state.value,
            )
            self._touch()

    def get_component_state(self, component: str) -> ServiceState:
        """Return the ServiceState for a named component.

        Returns INITIALIZING if the component has not yet registered.
        """
        with self._lock:
            return self._component_states.get(component, ServiceState.INITIALIZING)

    # ------------------------------------------------------------------
    # Strategy activation
    # ------------------------------------------------------------------

    def activate_strategy(self, record: StrategyActivationRecord) -> None:
        """Register a strategy as active in the current runtime state."""
        with self._lock:
            record.status = ActivationStatus.ACTIVE
            if record.activated_at is None:
                record.activated_at = _utcnow()
            self._active_strategies[record.strategy_id] = record
            self._metrics["strategy_activations"] += 1
            self._audit("strategy_activated", strategy_id=record.strategy_id)
            self._touch()

    def deactivate_strategy(self, strategy_id: str, reason: str) -> None:
        """Deactivate a strategy.  The record is retained with DISABLED status."""
        with self._lock:
            record = self._active_strategies.get(strategy_id)
            if record is None:
                logger.warning("deactivate_strategy: unknown strategy_id=%s", strategy_id)
                return
            record.status = ActivationStatus.DISABLED
            record.deactivated_at = _utcnow()
            record.notes = f"Deactivated: {reason}"
            del self._active_strategies[strategy_id]
            self._metrics["strategy_deactivations"] += 1
            self._audit("strategy_deactivated", strategy_id=strategy_id, reason=reason)
            self._touch()

    # ------------------------------------------------------------------
    # Model activation
    # ------------------------------------------------------------------

    def activate_model(self, record: ModelActivationRecord) -> None:
        """Register a model as active."""
        with self._lock:
            record.status = ActivationStatus.ACTIVE
            if record.approved_at is None:
                record.approved_at = _utcnow()
            self._active_models[record.model_id] = record
            self._metrics["model_activations"] += 1
            self._audit("model_activated", model_id=record.model_id)
            self._touch()

    def disable_model(self, model_id: str, reason: str) -> None:
        """Disable a model.  Marks fallback_active=True to signal callers."""
        with self._lock:
            record = self._active_models.get(model_id)
            if record is None:
                logger.warning("disable_model: unknown model_id=%s", model_id)
                return
            record.status = ActivationStatus.DISABLED
            record.fallback_active = True
            record.notes = f"Disabled: {reason}"
            self._metrics["model_disables"] += 1
            self._audit("model_disabled", model_id=model_id, reason=reason)
            self._touch()

    # ------------------------------------------------------------------
    # Agent activation
    # ------------------------------------------------------------------

    def activate_agent(self, record: AgentActivationRecord) -> None:
        """Register an agent as active."""
        with self._lock:
            record.status = ActivationStatus.ACTIVE
            if record.activated_at is None:
                record.activated_at = _utcnow()
            self._active_agents[record.agent_name] = record
            self._metrics["agent_activations"] += 1
            self._audit("agent_activated", agent_name=record.agent_name)
            self._touch()

    def disable_agent(self, agent_name: str, reason: str) -> None:
        """Disable a named agent."""
        with self._lock:
            record = self._active_agents.get(agent_name)
            if record is None:
                logger.warning("disable_agent: unknown agent_name=%s", agent_name)
                return
            record.status = ActivationStatus.DISABLED
            record.notes = f"Disabled: {reason}"
            self._audit("agent_disabled", agent_name=agent_name, reason=reason)
            self._touch()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def apply_override(self, override: RuntimeOverride) -> None:
        """Apply a runtime override.  Logs to audit.  Fires optional callback."""
        with self._lock:
            self._active_overrides[override.override_id] = override
            self._metrics["overrides_applied"] += 1
            self._audit(
                "override_applied",
                override_id=override.override_id,
                override_type=override.override_type,
                scope=override.scope,
                value=override.value,
                applied_by=override.applied_by,
                expires_at=override.expires_at,
                is_emergency=override.is_emergency,
            )
            self._touch()
        # Callback outside lock to avoid potential deadlock
        if self._on_override_applied is not None:
            try:
                self._on_override_applied(override)
            except Exception:
                logger.exception("override callback raised")

    def clear_override(self, override_id: str, cleared_by: str) -> bool:
        """Clear a specific override.  Returns True if found and cleared."""
        with self._lock:
            override = self._active_overrides.pop(override_id, None)
            if override is None:
                return False
            self._expired_overrides[override_id] = override
            self._metrics["overrides_cleared"] += 1
            self._audit(
                "override_cleared",
                override_id=override_id,
                cleared_by=cleared_by,
            )
            self._touch()
            return True

    def expire_stale_overrides(self) -> List[str]:
        """Auto-expire overrides whose expiry has passed.

        Returns the list of expired override IDs.
        """
        expired_ids: List[str] = []
        with self._lock:
            to_expire = [
                oid
                for oid, ov in self._active_overrides.items()
                if _is_expired(ov.expires_at)
            ]
            for oid in to_expire:
                ov = self._active_overrides.pop(oid)
                self._expired_overrides[oid] = ov
                self._metrics["overrides_expired"] += 1
                self._audit("override_expired", override_id=oid)
                expired_ids.append(oid)
            if expired_ids:
                self._touch()
        if expired_ids:
            logger.info("Expired %d stale override(s): %s", len(expired_ids), expired_ids)
        return expired_ids

    def get_active_overrides(self, expired: bool = False) -> List[RuntimeOverride]:
        """Return active overrides.  Pass expired=True to include expired ones."""
        with self._lock:
            result = list(self._active_overrides.values())
            if expired:
                result += list(self._expired_overrides.values())
            return result

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def engage_kill_switch(self, reason: str, triggered_by: str) -> None:
        """Engage the global kill switch.  Sets overall state to HALTED."""
        with self._lock:
            if self._kill_switch_active:
                logger.info("Kill switch already active — noop (reason=%s)", reason)
                return
            self._kill_switch_active = True
            self._kill_switch_reason = reason
            self._overall_state = ServiceState.HALTED
            self._metrics["kill_switch_engagements"] += 1
            self._audit(
                "kill_switch_engaged",
                triggered_by=triggered_by,
                reason=reason,
            )
            self._touch()
            logger.critical(
                "KILL SWITCH ENGAGED by %s — reason: %s", triggered_by, reason
            )
        # Callback outside lock
        if self._on_kill_switch_engaged is not None:
            try:
                self._on_kill_switch_engaged(reason, triggered_by)
            except Exception:
                logger.exception("kill_switch callback raised")

    def release_kill_switch(self, released_by: str, approval_id: str) -> None:
        """Release the kill switch.  Requires a non-empty approval_id."""
        if not approval_id:
            raise ValueError("approval_id is required to release the kill switch")
        with self._lock:
            if not self._kill_switch_active:
                logger.info("Kill switch is not active — nothing to release")
                return
            self._kill_switch_active = False
            previous_reason = self._kill_switch_reason
            self._kill_switch_reason = None
            self._metrics["kill_switch_releases"] += 1
            self._recompute_overall_state()
            self._audit(
                "kill_switch_released",
                released_by=released_by,
                approval_id=approval_id,
                previous_reason=previous_reason,
            )
            self._touch()
            logger.warning(
                "Kill switch released by %s (approval=%s)", released_by, approval_id
            )

    # ------------------------------------------------------------------
    # Throttle / exits-only
    # ------------------------------------------------------------------

    def set_global_throttle(
        self, level: ThrottleLevel, reason: str, applied_by: str
    ) -> None:
        """Set the global sizing throttle level."""
        with self._lock:
            prev = self._global_throttle
            self._global_throttle = level
            self._audit(
                "global_throttle_set",
                prev=prev.value,
                new=level.value,
                reason=reason,
                applied_by=applied_by,
            )
            self._touch()
            logger.info(
                "Global throttle changed %s -> %s by %s (%s)",
                prev.value,
                level.value,
                applied_by,
                reason,
            )

    def set_exits_only(self, enabled: bool, reason: str, applied_by: str) -> None:
        """Enable or disable exits-only mode."""
        with self._lock:
            self._exits_only_mode = enabled
            self._audit(
                "exits_only_set",
                enabled=enabled,
                reason=reason,
                applied_by=applied_by,
            )
            self._touch()
            logger.warning(
                "Exits-only mode set to %s by %s (%s)", enabled, applied_by, reason
            )

    # ------------------------------------------------------------------
    # Safety gate
    # ------------------------------------------------------------------

    def is_safe_to_trade(self) -> Tuple[bool, List[str]]:
        """Conservative safety gate for the control plane.

        Returns (True, []) only if all of the following hold:
          - Kill switch is not active.
          - Global throttle is not HALTED.
          - No HALTED or UNHEALTHY component states.
          - Overall service state is not HALTED or UNHEALTHY.

        Any uncertainty returns (False, [reasons]).
        """
        with self._lock:
            blocking: List[str] = []

            if self._kill_switch_active:
                blocking.append(
                    f"kill_switch_active: {self._kill_switch_reason}"
                )

            if self._global_throttle == ThrottleLevel.HALTED:
                blocking.append("global_throttle=HALTED")

            if self._overall_state in (ServiceState.HALTED, ServiceState.UNHEALTHY):
                blocking.append(f"overall_service_state={self._overall_state.value}")

            for comp, state in self._component_states.items():
                if state in (ServiceState.HALTED, ServiceState.UNHEALTHY):
                    blocking.append(f"component {comp!r} is {state.value}")

            return (len(blocking) == 0, blocking)

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def snapshot(self) -> RuntimeState:
        """Capture a new full state snapshot with a fresh unique ID."""
        with self._lock:
            snap = self._build_state_snapshot()
        self._persist_snapshot(snap)
        return snap

    def _build_state_snapshot(self) -> RuntimeState:
        """Build a RuntimeState from current in-memory state.  Must hold lock."""
        return RuntimeState(
            snapshot_id=_make_id("snap"),
            captured_at=_utcnow(),
            env=self._env,
            mode=self._mode,
            overall_service_state=self._overall_state,
            global_throttle=self._global_throttle,
            kill_switch_active=self._kill_switch_active,
            kill_switch_reason=self._kill_switch_reason,
            exits_only_mode=self._exits_only_mode,
            active_strategies={
                sid: _as_dict(rec)
                for sid, rec in self._active_strategies.items()
            },
            active_models={
                mid: _as_dict(rec)
                for mid, rec in self._active_models.items()
            },
            active_agents={
                aname: _as_dict(rec)
                for aname, rec in self._active_agents.items()
            },
            active_overrides=[
                _as_dict(ov) for ov in self._active_overrides.values()
            ],
            component_states={
                k: v.value for k, v in self._component_states.items()
            },
            pending_transitions=list(self._pending_transitions),
            last_updated=self._last_updated,
        )

    def _persist_snapshot(self, snap: RuntimeState) -> None:
        """Persist a snapshot to the SQL store if one is configured."""
        if self._sql_store is None:
            return
        try:
            data = _as_dict(snap)
            self._sql_store.store_json(
                "runtime_snapshots", snap.snapshot_id, data
            )
        except Exception:
            logger.exception("Failed to persist runtime snapshot %s", snap.snapshot_id)

    # ------------------------------------------------------------------
    # Metrics / observability
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return a copy of observability counters."""
        with self._lock:
            return dict(self._metrics)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_state_manager_instance: Optional[RuntimeStateManager] = None
_state_manager_lock = threading.Lock()


def get_runtime_state_manager(
    env: str = "paper",
    mode: RuntimeMode = RuntimeMode.PAPER,
) -> RuntimeStateManager:
    """Singleton accessor for the global RuntimeStateManager.

    On first call the instance is created with the supplied defaults.
    Subsequent calls return the existing instance regardless of arguments.
    """
    global _state_manager_instance
    if _state_manager_instance is None:
        with _state_manager_lock:
            if _state_manager_instance is None:
                _state_manager_instance = RuntimeStateManager(env=env, mode=mode)
    return _state_manager_instance
