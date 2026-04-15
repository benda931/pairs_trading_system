# -*- coding: utf-8 -*-
"""
core/agent_feedback.py — Agent Feedback Loop Engine
=====================================================

The MISSING LINK: connects agent outputs to real system decisions.

Before this module, agents ran in "info-only" mode — results were logged
but never acted upon. This module creates a closed feedback loop:

    Agent Output → Governance Router → Approval/Veto → Execution → Audit

Feedback channels:
1. **Signal Feedback**: regime/quality agents → block/approve entries
2. **Risk Feedback**: drawdown/exposure agents → deleverage/kill-switch
3. **Improvement Feedback**: GPT/optimizer agents → retrain/reconfigure
4. **Health Feedback**: system/data agents → pause/resume pipeline

Governance integration (Phase 1):
    Every action is routed through GovernanceRouter before execution.
    The router enforces the institutional governance matrix:
    - Evidence validation (required fields per action type)
    - Tier routing (ADVISORY / AUTO / POLICY_GATED / HUMAN / EMERGENCY)
    - Duplicate suppression and staleness detection
    - Conflict resolution between concurrent agent actions
    - Approval engine integration (ApprovalEngine)
    - Incident creation on execution (IncidentManager)
    - Precision tracking and tier demotion (PrecisionDemotionEngine)

Usage:
    from core.agent_feedback import AgentFeedbackLoop
    from core.action_governance import TradingEnvironment

    loop = AgentFeedbackLoop(environment=TradingEnvironment.PAPER)
    actions = loop.process_agent_results(pipeline_results)
    loop.execute_actions(actions)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from core.action_governance import GovernedActionRecord, TradingEnvironment

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class FeedbackAction:
    """A concrete action derived from agent output.

    The `parameters` dict is the evidence bundle passed to GovernanceRouter.
    It must contain the fields required by the governance profile for this
    action_type × environment combination (see core/action_governance.py).

    The `environment` field determines which governance profile is applied.
    It defaults to PAPER so that existing callers that do not set it will
    not accidentally operate in LIVE governance mode.
    """
    action_id: str
    source_agent: str
    action_type: str        # BLOCK_ENTRY / FORCE_EXIT / DELEVERAGE / KILL_SWITCH /
                            # RETRAIN_MODEL / OPTIMIZE_PARAMS / UPDATE_CONFIG /
                            # PAUSE_PIPELINE / ALERT / ADJUST_THRESHOLD
    severity: str           # INFO / WARNING / CRITICAL / EMERGENCY
    target: str             # What to act on (pair_id, "portfolio", "system")
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    auto_execute: bool = True    # Retained for backward compat; GovernanceRouter decides
    executed: bool = False
    execution_result: str = ""
    environment: TradingEnvironment = TradingEnvironment.PAPER
    confirming_agent: Optional[str] = None  # For dual-confirmation (EMERGENCY_ONLY)
    governed_record: Optional[GovernedActionRecord] = None  # Populated after routing


@dataclass
class FeedbackSummary:
    """Summary of all feedback actions from one pipeline run."""
    timestamp: str
    n_agent_results_processed: int
    n_actions_generated: int
    n_actions_executed: int
    n_actions_blocked: int
    n_actions_advisory: int = 0
    n_actions_pending_approval: int = 0
    actions: List[FeedbackAction] = field(default_factory=list)
    governed_records: List[GovernedActionRecord] = field(default_factory=list)
    system_state_changes: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Feedback rules
# =====================================================================

class SignalFeedbackRules:
    """Rules for translating signal agent outputs into actions."""

    @staticmethod
    def process(agent_name: str, result: Dict[str, Any]) -> List[FeedbackAction]:
        actions = []
        output = result.get("output", {}) or {}
        status = result.get("status", "")

        if agent_name == "regime_surveillance":
            # Agent returns "regime_map" dict or "current_regime" or "regime"
            regime_map = output.get("regime_map", {})
            regime = (
                output.get("regime")
                or output.get("current_regime")
                or output.get("overall_regime")
                or (list(regime_map.values())[0] if regime_map else "")
                or ""
            )
            if regime.upper() in ("CRISIS", "BROKEN"):
                actions.append(FeedbackAction(
                    action_id=f"regime_{regime}_{_ts()}",
                    source_agent=agent_name,
                    action_type="BLOCK_ENTRY",
                    severity="CRITICAL",
                    target="portfolio",
                    parameters={"regime": regime, "block_new_entries": True},
                    reason=f"Regime={regime}: blocking all new entries",
                ))
                actions.append(FeedbackAction(
                    action_id=f"regime_delever_{_ts()}",
                    source_agent=agent_name,
                    action_type="DELEVERAGE",
                    severity="CRITICAL",
                    target="portfolio",
                    parameters={"target_leverage_multiplier": 0.3 if regime.upper() == "CRISIS" else 0.5},
                    reason=f"Regime={regime}: reducing leverage",
                ))
            elif regime.upper() in ("TENSION", "HIGH_VOL", "TRENDING"):
                actions.append(FeedbackAction(
                    action_id=f"regime_caution_{_ts()}",
                    source_agent=agent_name,
                    action_type="ADJUST_THRESHOLD",
                    severity="WARNING",
                    target="portfolio",
                    parameters={"entry_z_multiplier": 1.3, "reduce_size_by": 0.3},
                    reason=f"Regime={regime}: raising entry bar and reducing size",
                ))

        elif agent_name == "signal_analyst":
            quality = output.get("quality_grade", "")
            if quality in ("D", "F"):
                pair = output.get("pair_id", "")
                actions.append(FeedbackAction(
                    action_id=f"quality_block_{pair}_{_ts()}",
                    source_agent=agent_name,
                    action_type="BLOCK_ENTRY",
                    severity="WARNING",
                    target=str(pair),
                    parameters={"quality_grade": quality},
                    reason=f"Signal quality {quality}: blocking entry for {pair}",
                ))

        elif agent_name == "exit_oversight":
            # Agent may return "exit_signals", "recommended_exits", "exits", etc.
            exits = (
                output.get("recommended_exits")
                or output.get("exit_signals")
                or output.get("exits")
                or []
            )
            for ex in exits:
                pair = ex.get("pair_id", "?")
                reason = ex.get("reason", "agent_recommendation")
                actions.append(FeedbackAction(
                    action_id=f"exit_{pair}_{_ts()}",
                    source_agent=agent_name,
                    action_type="FORCE_EXIT",
                    severity="WARNING",
                    target=str(pair),
                    parameters=ex,
                    reason=f"Exit recommended for {pair}: {reason}",
                ))

        elif agent_name == "trade_lifecycle":
            stale = output.get("stale_positions", [])
            for pos in stale:
                pair = pos.get("pair_id", "?")
                holding_days = pos.get("holding_days", 0)
                if holding_days > 60:
                    actions.append(FeedbackAction(
                        action_id=f"stale_exit_{pair}_{_ts()}",
                        source_agent=agent_name,
                        action_type="FORCE_EXIT",
                        severity="INFO",
                        target=str(pair),
                        parameters={"holding_days": holding_days},
                        reason=f"Stale position {pair}: {holding_days}d holding",
                    ))

        return actions


class RiskFeedbackRules:
    """Rules for translating risk agent outputs into actions."""

    @staticmethod
    def process(agent_name: str, result: Dict[str, Any]) -> List[FeedbackAction]:
        actions = []
        output = result.get("output", {}) or {}

        if agent_name == "drawdown_monitor":
            # Agent may return "current_dd_pct", "current_drawdown", "drawdown_pct", etc.
            current_dd = (
                output.get("current_drawdown")
                or output.get("current_dd_pct")
                or output.get("drawdown_pct")
                or output.get("dd_pct")
                or 0
            )
            if current_dd < -0.20:
                actions.append(FeedbackAction(
                    action_id=f"dd_killswitch_{_ts()}",
                    source_agent=agent_name,
                    action_type="KILL_SWITCH",
                    severity="EMERGENCY",
                    target="system",
                    parameters={"drawdown": current_dd, "mode": "EXITS_ONLY"},
                    reason=f"Drawdown {current_dd:.1%} exceeds -20% threshold",
                    auto_execute=True,
                ))
            elif current_dd < -0.10:
                actions.append(FeedbackAction(
                    action_id=f"dd_delever_{_ts()}",
                    source_agent=agent_name,
                    action_type="DELEVERAGE",
                    severity="CRITICAL",
                    target="portfolio",
                    parameters={"drawdown": current_dd, "target_leverage_multiplier": 0.5},
                    reason=f"Drawdown {current_dd:.1%}: reducing leverage 50%",
                ))

        elif agent_name == "exposure_monitor":
            violations = output.get("violations", [])
            for v in violations:
                if v.get("severity") == "HARD":
                    actions.append(FeedbackAction(
                        action_id=f"exposure_{v.get('type', '?')}_{_ts()}",
                        source_agent=agent_name,
                        action_type="BLOCK_ENTRY",
                        severity="CRITICAL",
                        target="portfolio",
                        parameters=v,
                        reason=f"Exposure violation: {v.get('description', '?')}",
                    ))

        elif agent_name == "kill_switch":
            # Agent may return "triggered", "should_halt", "halt", etc.
            should_halt = (
                output.get("should_halt")
                or output.get("triggered")
                or output.get("halt")
                or False
            )
            if should_halt:
                actions.append(FeedbackAction(
                    action_id=f"killswitch_{_ts()}",
                    source_agent=agent_name,
                    action_type="KILL_SWITCH",
                    severity="EMERGENCY",
                    target="system",
                    parameters={"mode": output.get("mode", "HALT_ALL")},
                    reason=output.get("reason", "Kill-switch triggered"),
                    auto_execute=True,
                ))

        elif agent_name == "derisking":
            actions_list = output.get("derisking_actions", [])
            for a in actions_list:
                actions.append(FeedbackAction(
                    action_id=f"derisk_{a.get('pair', '?')}_{_ts()}",
                    source_agent=agent_name,
                    action_type="DELEVERAGE",
                    severity="WARNING",
                    target=str(a.get("pair", "portfolio")),
                    parameters=a,
                    reason=a.get("reason", "Derisking recommendation"),
                ))

        return actions


class ImprovementFeedbackRules:
    """Rules for translating GPT/optimizer agent outputs into actions."""

    @staticmethod
    def process(agent_name: str, result: Dict[str, Any]) -> List[FeedbackAction]:
        actions = []
        output = result.get("output", {}) or {}

        if agent_name == "gpt_signal_advisor":
            recommendations = output.get("recommendations", [])
            for rec in recommendations:
                action_type = rec.get("action", "").upper()
                if action_type == "RETRAIN":
                    actions.append(FeedbackAction(
                        action_id=f"gpt_retrain_{_ts()}",
                        source_agent=agent_name,
                        action_type="RETRAIN_MODEL",
                        severity="INFO",
                        target=rec.get("pair", "all"),
                        parameters=rec,
                        reason=rec.get("reason", "GPT recommended retraining"),
                    ))
                elif action_type == "OPTIMIZE":
                    actions.append(FeedbackAction(
                        action_id=f"gpt_optimize_{_ts()}",
                        source_agent=agent_name,
                        action_type="OPTIMIZE_PARAMS",
                        severity="INFO",
                        target=rec.get("pair", "all"),
                        parameters=rec,
                        reason=rec.get("reason", "GPT recommended optimization"),
                    ))

        elif agent_name == "gpt_model_tuner":
            should_retrain = output.get("should_retrain", False)
            if should_retrain:
                actions.append(FeedbackAction(
                    action_id=f"tuner_retrain_{_ts()}",
                    source_agent=agent_name,
                    action_type="RETRAIN_MODEL",
                    severity="INFO",
                    target="all",
                    parameters={"reason": output.get("reason", "Model performance degraded")},
                    reason=output.get("reason", "GPT tuner recommended retraining"),
                ))

        elif agent_name == "auto_parameter_optimizer":
            best_params = output.get("best_params")
            if best_params and output.get("optimized", False):
                actions.append(FeedbackAction(
                    action_id=f"apply_params_{_ts()}",
                    source_agent=agent_name,
                    action_type="UPDATE_CONFIG",
                    severity="INFO",
                    target="strategy",
                    parameters={"updates": best_params},
                    reason=f"Optimizer found better params: {best_params}",
                ))

        return actions


# =====================================================================
# Feedback Loop Engine
# =====================================================================

class ActionThrottler:
    """
    Legacy rate limiter — DEPRECATED.

    Retained for backward compatibility only. In the governed pipeline,
    duplicate suppression and cooldown enforcement is handled by
    `DuplicateActionSuppressor` inside `GovernanceRouter`. This class is
    no longer in the hot path when `AgentFeedbackLoop.use_governance=True`.

    For new code, use `GovernanceRouter` directly or via `AgentFeedbackLoop`.
    """

    def __init__(self):
        try:
            from core.contracts import ActionThrottleConfig
            self._config = ActionThrottleConfig()
        except Exception:
            self._config = None
        self._last_action_time: Dict[str, float] = {}
        self._actions_this_cycle: int = 0
        self._emergency_actions_today: int = 0
        self._today: str = ""

    def can_execute(self, action: FeedbackAction) -> tuple:
        """Check if action is allowed. Returns (allowed, reason)."""
        import time
        if self._config is None:
            return True, "OK (throttler config unavailable)"

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if today != self._today:
            self._today = today
            self._emergency_actions_today = 0

        if self._actions_this_cycle >= self._config.max_actions_per_cycle:
            return False, "Max actions per cycle ({}) reached".format(
                self._config.max_actions_per_cycle)

        if action.severity == "EMERGENCY":
            if self._emergency_actions_today >= self._config.max_emergency_actions_per_day:
                return False, "Max emergency actions per day ({}) reached".format(
                    self._config.max_emergency_actions_per_day)

        cool_down = self._config.cool_down_seconds.get(action.action_type, 300)
        key = "{}:{}".format(action.action_type, action.target)
        last_time = self._last_action_time.get(key, 0)
        elapsed = time.time() - last_time
        if elapsed < cool_down:
            remaining = cool_down - elapsed
            return False, "Cool-down: {} for {} ({:.0f}s remaining)".format(
                action.action_type, action.target, remaining)

        return True, "OK"

    def record_execution(self, action: FeedbackAction) -> None:
        import time
        key = "{}:{}".format(action.action_type, action.target)
        self._last_action_time[key] = time.time()
        self._actions_this_cycle += 1
        if action.severity == "EMERGENCY":
            self._emergency_actions_today += 1

    def reset_cycle(self) -> None:
        self._actions_this_cycle = 0


class AgentFeedbackLoop:
    """
    Central engine that converts agent outputs into governed system actions.

    Processes all agent results from a pipeline run, applies rules to generate
    `FeedbackAction` objects, then routes each action through `GovernanceRouter`
    before execution.

    When `use_governance=True` (default), every action passes through the full
    11-step governance pipeline:
        evidence validation → staleness → dedup → conflict → tier routing →
        approval/human review → pre-execution snapshot → execution →
        incident creation → audit write → outcome observation registration.

    When `use_governance=False`, the legacy `ActionThrottler` + direct
    `_execute_single()` path is used. This mode exists for backward compatibility
    in test/research environments and should NOT be used in live trading.

    Parameters
    ----------
    dry_run : bool
        If True, no actions are executed and no approvals are submitted.
    environment : TradingEnvironment
        Deployment context. Controls which governance profiles apply.
    use_governance : bool
        If True, route all actions through GovernanceRouter. Default True.
    executor_registry : dict, optional
        Maps action_type → executor callable for GovernanceRouter.
        If None, the router is initialised without executor validation
        (useful in test environments).
    """

    # Map agent names to rule sets
    SIGNAL_AGENTS = {"regime_surveillance", "signal_analyst", "exit_oversight", "trade_lifecycle"}
    RISK_AGENTS = {"drawdown_monitor", "exposure_monitor", "kill_switch", "derisking", "capital_budget"}
    IMPROVEMENT_AGENTS = {"gpt_signal_advisor", "gpt_model_tuner", "gpt_strategy_researcher",
                          "auto_parameter_optimizer", "auto_model_retrainer"}

    def __init__(
        self,
        dry_run: bool = False,
        environment: TradingEnvironment = TradingEnvironment.PAPER,
        use_governance: bool = True,
        executor_registry: Optional[Dict[str, Any]] = None,
    ):
        self.dry_run = dry_run
        self.environment = environment
        self.use_governance = use_governance
        self._action_history: List[FeedbackAction] = []
        self._governed_records: List[GovernedActionRecord] = []
        self._throttle = ActionThrottler()  # Legacy fallback

        # Lazy-initialise GovernanceRouter on first use to avoid circular imports
        self._router = None
        self._executor_registry = executor_registry or self._build_default_executor_registry()

    def process_agent_results(
        self,
        results: Sequence[Any],
    ) -> List[FeedbackAction]:
        """
        Process all agent results and generate feedback actions.

        Parameters
        ----------
        results : list of TaskResult or similar
            Pipeline results from orchestrator.run_daily_pipeline().

        Returns
        -------
        List of FeedbackAction to execute.
        """
        all_actions: List[FeedbackAction] = []

        for r in results:
            agent_name = getattr(r, 'task_name', '').replace('agent_', '')
            status = getattr(r, 'status', '')
            output = getattr(r, 'output', {}) or {}

            if status != "success":
                continue

            result_dict = {"status": status, "output": output}

            # Route to appropriate rule set
            if agent_name in self.SIGNAL_AGENTS:
                actions = SignalFeedbackRules.process(agent_name, result_dict)
            elif agent_name in self.RISK_AGENTS:
                actions = RiskFeedbackRules.process(agent_name, result_dict)
            elif agent_name in self.IMPROVEMENT_AGENTS:
                actions = ImprovementFeedbackRules.process(agent_name, result_dict)
            else:
                continue

            all_actions.extend(actions)

        # Sort by severity (EMERGENCY first)
        severity_order = {"EMERGENCY": 0, "CRITICAL": 1, "WARNING": 2, "INFO": 3}
        all_actions.sort(key=lambda a: severity_order.get(a.severity, 4))

        logger.info(
            "Feedback loop: %d actions from %d results (E:%d C:%d W:%d I:%d)",
            len(all_actions), len(results),
            sum(1 for a in all_actions if a.severity == "EMERGENCY"),
            sum(1 for a in all_actions if a.severity == "CRITICAL"),
            sum(1 for a in all_actions if a.severity == "WARNING"),
            sum(1 for a in all_actions if a.severity == "INFO"),
        )

        return all_actions

    def _get_router(self):
        """Lazy-initialise and return the GovernanceRouter singleton."""
        if self._router is None:
            try:
                from core.governance_router import get_governance_router
                self._router = get_governance_router(
                    environment=self.environment,
                    executor_registry=self._executor_registry,
                    validate_on_init=False,  # Executors may be partial in test env
                )
            except Exception as exc:
                logger.warning(
                    "AgentFeedbackLoop: GovernanceRouter init failed (%s); "
                    "falling back to legacy throttler.", exc
                )
        return self._router

    def execute_actions(
        self,
        actions: List[FeedbackAction],
    ) -> FeedbackSummary:
        """
        Execute feedback actions through the governance pipeline and return summary.

        When use_governance=True (default), every action is routed through
        GovernanceRouter which enforces the institutional governance matrix.

        When use_governance=False or GovernanceRouter is unavailable, the
        legacy ActionThrottler + direct executor path is used.

        dry_run mode logs intent without executing or submitting approvals.
        """
        n_executed = 0
        n_blocked = 0
        n_advisory = 0
        n_pending = 0
        state_changes: Dict[str, Any] = {}
        governed_records: List[GovernedActionRecord] = []

        router = self._get_router() if self.use_governance else None

        for action in actions:
            # Inject environment + target into parameters for governance evidence
            params = dict(action.parameters)
            params.setdefault("target", action.target)
            params.setdefault("reason", action.reason)

            if self.dry_run:
                logger.info("[DRY RUN] Would route: %s → %s (%s) [env=%s]",
                            action.action_type, action.target,
                            action.reason, self.environment.value)
                n_blocked += 1
                self._action_history.append(action)
                continue

            if router is not None:
                # ── Governed path ──────────────────────────────────────
                try:
                    record = router.route(
                        action_type=action.action_type,
                        action_id=action.action_id,
                        source_agent=action.source_agent,
                        target=action.target,
                        parameters=params,
                        severity=action.severity,
                        created_at=None,
                        confirming_agent=action.confirming_agent,
                    )
                    action.governed_record = record
                    governed_records.append(record)

                    if record.suppressed:
                        logger.info(
                            "[GOVERNED SUPPRESSED] %s: %s",
                            action.action_type, record.suppression_reason,
                        )
                        n_blocked += 1
                    elif record.executed:
                        action.executed = True
                        action.execution_result = record.execution_result
                        n_executed += 1
                        logger.info(
                            "[GOVERNED EXECUTED] %s → %s",
                            action.action_type, record.execution_result,
                        )
                        if action.action_type in ("KILL_SWITCH", "BLOCK_ENTRY", "DELEVERAGE"):
                            state_changes[action.action_type] = {
                                "target": action.target,
                                "parameters": action.parameters,
                                "timestamp": _ts(),
                                "governed": True,
                            }
                    elif record.approval_status in ("PENDING", "ESCALATED"):
                        n_pending += 1
                        logger.info(
                            "[GOVERNED PENDING APPROVAL] %s: approval_request_id=%s",
                            action.action_type, record.approval_request_id,
                        )
                    else:
                        # ADVISORY_ONLY or similar
                        n_advisory += 1
                        logger.info(
                            "[GOVERNED ADVISORY] %s → tier=%s",
                            action.action_type, record.governance_tier.value,
                        )

                except Exception as exc:
                    logger.error(
                        "GovernanceRouter raised for %s: %s — falling back to legacy.",
                        action.action_type, exc,
                    )
                    # Fail-safe: fall through to legacy path
                    self._execute_legacy(action, state_changes)
                    if action.executed:
                        n_executed += 1
                    else:
                        n_blocked += 1

            else:
                # ── Legacy fallback path ───────────────────────────────
                self._execute_legacy(action, state_changes)
                if action.executed:
                    n_executed += 1
                else:
                    n_blocked += 1

            self._action_history.append(action)

        self._governed_records.extend(governed_records)

        # Send alerts for executed actions
        self._send_feedback_alerts(actions)

        return FeedbackSummary(
            timestamp=_ts(),
            n_agent_results_processed=0,  # Set by caller
            n_actions_generated=len(actions),
            n_actions_executed=n_executed,
            n_actions_blocked=n_blocked,
            n_actions_advisory=n_advisory,
            n_actions_pending_approval=n_pending,
            actions=actions,
            governed_records=governed_records,
            system_state_changes=state_changes,
        )

    def _execute_legacy(
        self,
        action: FeedbackAction,
        state_changes: Dict[str, Any],
    ) -> None:
        """Legacy throttler + direct executor path (research/test environments)."""
        self._throttle.reset_cycle()

        if not action.auto_execute and action.severity not in ("EMERGENCY", "CRITICAL"):
            logger.info("[LEGACY BLOCKED] Needs approval: %s → %s",
                        action.action_type, action.reason)
            return

        allowed, throttle_reason = self._throttle.can_execute(action)
        if not allowed:
            logger.info("[LEGACY THROTTLED] %s → %s: %s",
                        action.action_type, action.target, throttle_reason)
            return

        try:
            result = self._execute_single(action)
            action.executed = True
            action.execution_result = result
            self._throttle.record_execution(action)
            if action.action_type in ("KILL_SWITCH", "BLOCK_ENTRY", "DELEVERAGE"):
                state_changes[action.action_type] = {
                    "target": action.target,
                    "parameters": action.parameters,
                    "timestamp": _ts(),
                    "governed": False,
                }
            logger.info("[LEGACY EXECUTED] %s → %s: %s",
                        action.action_type, action.target, result)
        except Exception as exc:
            action.execution_result = "ERROR: {}".format(exc)
            logger.error("Legacy execution failed for %s: %s", action.action_type, exc)

    def _execute_single(self, action: FeedbackAction) -> str:
        """Execute a single feedback action."""

        if action.action_type == "KILL_SWITCH":
            return self._exec_kill_switch(action)
        elif action.action_type == "BLOCK_ENTRY":
            return self._exec_block_entry(action)
        elif action.action_type == "FORCE_EXIT":
            return self._exec_force_exit(action)
        elif action.action_type == "DELEVERAGE":
            return self._exec_deleverage(action)
        elif action.action_type == "ADJUST_THRESHOLD":
            return self._exec_adjust_threshold(action)
        elif action.action_type == "RETRAIN_MODEL":
            return self._exec_retrain(action)
        elif action.action_type == "OPTIMIZE_PARAMS":
            return self._exec_optimize(action)
        elif action.action_type == "UPDATE_CONFIG":
            return self._exec_update_config(action)
        elif action.action_type == "PAUSE_PIPELINE":
            return self._exec_pause(action)
        else:
            return f"Unknown action type: {action.action_type}"

    # ── Action executors ──────────────────────────────────────

    def _exec_kill_switch(self, action: FeedbackAction) -> str:
        try:
            from portfolio.risk_ops import make_kill_switch_manager_with_control_plane
            ksm = make_kill_switch_manager_with_control_plane()
            mode = action.parameters.get("mode", "HALT_ALL")
            ksm.activate(mode=mode, reason=action.reason)
            return f"Kill-switch activated: {mode}"
        except Exception as exc:
            # Fallback: write state to bus
            try:
                from core.agent_bus import AgentBus
                _bus = AgentBus()
                _bus.publish("kill_switch", {"active": True, "mode": action.parameters.get("mode"), "reason": action.reason})
            except Exception:
                pass
            return f"Kill-switch published to bus (direct activation failed: {exc})"

    def _exec_block_entry(self, action: FeedbackAction) -> str:
        try:
            from core.orchestrator import PairsOrchestrator
            orch = PairsOrchestrator()
            orch.bus.publish("entry_block", {
                "target": action.target,
                "reason": action.reason,
                "parameters": action.parameters,
                "timestamp": _ts(),
            })
            return f"Entry blocked for {action.target}"
        except Exception as exc:
            return f"Block published (bus error: {exc})"

    def _exec_force_exit(self, action: FeedbackAction) -> str:
        try:
            from core.orchestrator import PairsOrchestrator
            orch = PairsOrchestrator()
            orch.bus.publish("force_exit", {
                "pair": action.target,
                "reason": action.reason,
                "parameters": action.parameters,
                "timestamp": _ts(),
            })
            return f"Exit signal published for {action.target}"
        except Exception as exc:
            return f"Exit published (bus error: {exc})"

    def _exec_deleverage(self, action: FeedbackAction) -> str:
        multiplier = action.parameters.get("target_leverage_multiplier", 0.5)
        try:
            from core.orchestrator import PairsOrchestrator
            orch = PairsOrchestrator()
            orch.bus.publish("deleverage", {
                "target": action.target,
                "multiplier": multiplier,
                "reason": action.reason,
                "timestamp": _ts(),
            })
            return f"Deleverage to {multiplier:.0%} published"
        except Exception as exc:
            return f"Deleverage signal error: {exc}"

    def _exec_adjust_threshold(self, action: FeedbackAction) -> str:
        try:
            from core.orchestrator import PairsOrchestrator
            orch = PairsOrchestrator()
            orch.bus.publish("threshold_adjustment", {
                "parameters": action.parameters,
                "reason": action.reason,
                "timestamp": _ts(),
            })
            return f"Thresholds adjusted: {action.parameters}"
        except Exception as exc:
            return f"Threshold adjustment error: {exc}"

    def _exec_retrain(self, action: FeedbackAction) -> str:
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask
            registry = get_default_registry()
            agent = registry.get_agent("auto_model_retrainer")
            if agent is None:
                return "auto_model_retrainer not found in registry"

            task = AgentTask(
                task_id=f"feedback_retrain_{_ts()}",
                agent_name="auto_model_retrainer",
                task_type="auto_retrain_model",
                payload={"pair": action.target, "trigger": "feedback_loop"},
                created_at=datetime.now(timezone.utc),
            )
            result = agent.execute(task)
            return f"Retrain dispatched: status={result.status}"
        except Exception as exc:
            return f"Retrain failed: {exc}"

    def _exec_optimize(self, action: FeedbackAction) -> str:
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask
            registry = get_default_registry()
            agent = registry.get_agent("auto_parameter_optimizer")
            if agent is None:
                return "auto_parameter_optimizer not found"

            task = AgentTask(
                task_id=f"feedback_optimize_{_ts()}",
                agent_name="auto_parameter_optimizer",
                task_type="auto_optimize_params",
                payload={
                    "pair": action.target,
                    "n_trials": action.parameters.get("n_trials", 20),
                    "trigger": "feedback_loop",
                },
                created_at=datetime.now(timezone.utc),
            )
            result = agent.execute(task)
            return f"Optimization dispatched: status={result.status}"
        except Exception as exc:
            return f"Optimization failed: {exc}"

    def _exec_update_config(self, action: FeedbackAction) -> str:
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask
            registry = get_default_registry()
            agent = registry.get_agent("auto_config_updater")
            if agent is None:
                return "auto_config_updater not found"

            task = AgentTask(
                task_id=f"feedback_config_{_ts()}",
                agent_name="auto_config_updater",
                task_type="auto_update_config",
                payload={"updates": action.parameters.get("updates", {})},
                created_at=datetime.now(timezone.utc),
            )
            result = agent.execute(task)
            return f"Config update dispatched: status={result.status}"
        except Exception as exc:
            return f"Config update failed: {exc}"

    def _exec_pause(self, action: FeedbackAction) -> str:
        try:
            from core.orchestrator import PairsOrchestrator
            orch = PairsOrchestrator()
            orch.bus.publish("pipeline_pause", {
                "reason": action.reason,
                "timestamp": _ts(),
            })
            return "Pipeline pause signal published"
        except Exception as exc:
            return f"Pause error: {exc}"

    # ── Alerts ────────────────────────────────────────────────

    @staticmethod
    def _send_feedback_alerts(actions: List[FeedbackAction]) -> None:
        """Send Telegram alerts for important feedback actions."""
        try:
            from core.alerts import alert_risk, alert_system
            for action in actions:
                if action.severity in ("EMERGENCY", "CRITICAL"):
                    alert_risk(
                        f"Agent Feedback: {action.action_type}",
                        f"{action.reason}\nTarget: {action.target}\nAgent: {action.source_agent}",
                        severity="CRITICAL" if action.severity == "EMERGENCY" else "WARNING",
                    )
                elif action.severity == "WARNING" and action.action_type in ("FORCE_EXIT", "DELEVERAGE"):
                    alert_risk(
                        f"Agent Action: {action.action_type}",
                        f"{action.reason}",
                        severity="WARNING",
                    )
        except Exception:
            pass

    def _build_default_executor_registry(self) -> Dict[str, Any]:
        """Build the default executor registry mapping action_type → method.

        Each executor receives the action parameters dict and returns a str result.
        These are thin wrappers around the existing _exec_* methods, adapted to
        accept a parameters dict rather than a FeedbackAction object.
        """
        def _make_executor(exec_method_name: str):
            def executor(params: dict) -> str:
                # Reconstruct a minimal FeedbackAction for legacy executor compatibility
                action = FeedbackAction(
                    action_id=params.get("action_id", _ts()),
                    source_agent=params.get("source_agent", "governance_router"),
                    action_type=params.get("action_type", ""),
                    severity=params.get("severity", "INFO"),
                    target=params.get("target", "portfolio"),
                    parameters=params,
                    reason=params.get("reason", ""),
                )
                method = getattr(self, exec_method_name)
                return method(action)
            return executor

        return {
            "KILL_SWITCH":      _make_executor("_exec_kill_switch"),
            "BLOCK_ENTRY":      _make_executor("_exec_block_entry"),
            "FORCE_EXIT":       _make_executor("_exec_force_exit"),
            "DELEVERAGE":       _make_executor("_exec_deleverage"),
            "ADJUST_THRESHOLD": _make_executor("_exec_adjust_threshold"),
            "RETRAIN_MODEL":    _make_executor("_exec_retrain"),
            "OPTIMIZE_PARAMS":  _make_executor("_exec_optimize"),
            "UPDATE_CONFIG":    _make_executor("_exec_update_config"),
            "PAUSE_PIPELINE":   _make_executor("_exec_pause"),
        }

    def run_due_observations(self) -> None:
        """Evaluate pending outcome observations and update precision metrics.

        Call this from a periodic health-check task (e.g., every hour in live
        trading) to keep the PrecisionDemotionEngine current.
        """
        router = self._get_router()
        if router is not None:
            try:
                router.evaluate_due_observations()
            except Exception as exc:
                logger.warning("run_due_observations: %s", exc)

    def get_governance_metrics(self) -> Dict[str, Any]:
        """Return current precision and demotion metrics from the GovernanceRouter."""
        router = self._get_router()
        if router is None:
            return {"error": "GovernanceRouter not available"}
        metrics = router.get_precision_metrics()
        return {
            m.action_type: {
                "environment": m.environment.value,
                "current_tier": m.current_tier.value,
                "30d_precision": round(m.rolling_precision_30d, 3),
                "90d_precision": round(m.rolling_precision_90d, 3),
                "total_executed": m.total_executed,
                "false_positive_rate": round(m.false_positive_rate, 3),
                "tier_demoted": m.tier_demoted_at is not None,
                "demotion_reason": m.tier_demotion_reason,
            }
            for m in metrics
        }

    @property
    def action_history(self) -> List[FeedbackAction]:
        return list(self._action_history)

    @property
    def governed_records(self) -> List[GovernedActionRecord]:
        return list(self._governed_records)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
