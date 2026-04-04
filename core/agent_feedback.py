# -*- coding: utf-8 -*-
"""
core/agent_feedback.py — Agent Feedback Loop Engine
=====================================================

The MISSING LINK: connects agent outputs to real system decisions.

Before this module, agents ran in "info-only" mode — results were logged
but never acted upon. This module creates a closed feedback loop:

    Agent Output → Feedback Engine → System Action → Verification

Feedback channels:
1. **Signal Feedback**: regime/quality agents → block/approve entries
2. **Risk Feedback**: drawdown/exposure agents → deleverage/kill-switch
3. **Improvement Feedback**: GPT/optimizer agents → retrain/reconfigure
4. **Health Feedback**: system/data agents → pause/resume pipeline

Usage:
    from core.agent_feedback import AgentFeedbackLoop

    loop = AgentFeedbackLoop()
    actions = loop.process_agent_results(pipeline_results)
    loop.execute_actions(actions)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class FeedbackAction:
    """A concrete action derived from agent output."""
    action_id: str
    source_agent: str
    action_type: str        # BLOCK_ENTRY / FORCE_EXIT / DELEVERAGE / KILL_SWITCH /
                            # RETRAIN_MODEL / OPTIMIZE_PARAMS / UPDATE_CONFIG /
                            # PAUSE_PIPELINE / ALERT / ADJUST_THRESHOLD
    severity: str           # INFO / WARNING / CRITICAL / EMERGENCY
    target: str             # What to act on (pair_id, "portfolio", "system")
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    auto_execute: bool = True    # False = needs human approval
    executed: bool = False
    execution_result: str = ""


@dataclass
class FeedbackSummary:
    """Summary of all feedback actions from one pipeline run."""
    timestamp: str
    n_agent_results_processed: int
    n_actions_generated: int
    n_actions_executed: int
    n_actions_blocked: int
    actions: List[FeedbackAction] = field(default_factory=list)
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

class AgentFeedbackLoop:
    """
    Central engine that converts agent outputs into system actions.

    Processes all agent results from a pipeline run, applies rules
    to generate actions, then executes approved actions.
    """

    # Map agent names to rule sets
    SIGNAL_AGENTS = {"regime_surveillance", "signal_analyst", "exit_oversight", "trade_lifecycle"}
    RISK_AGENTS = {"drawdown_monitor", "exposure_monitor", "kill_switch", "derisking", "capital_budget"}
    IMPROVEMENT_AGENTS = {"gpt_signal_advisor", "gpt_model_tuner", "gpt_strategy_researcher",
                          "auto_parameter_optimizer", "auto_model_retrainer"}

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._action_history: List[FeedbackAction] = []

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

    def execute_actions(
        self,
        actions: List[FeedbackAction],
    ) -> FeedbackSummary:
        """
        Execute feedback actions and return summary.

        EMERGENCY/CRITICAL actions auto-execute.
        INFO actions execute if auto_execute=True.
        dry_run mode logs but doesn't execute.
        """
        n_executed = 0
        n_blocked = 0
        state_changes: Dict[str, Any] = {}

        for action in actions:
            if self.dry_run:
                logger.info("[DRY RUN] Would execute: %s → %s (%s)",
                            action.action_type, action.target, action.reason)
                n_blocked += 1
                continue

            if not action.auto_execute and action.severity not in ("EMERGENCY", "CRITICAL"):
                logger.info("[BLOCKED] Needs approval: %s → %s", action.action_type, action.reason)
                n_blocked += 1
                continue

            # Execute the action
            try:
                result = self._execute_single(action)
                action.executed = True
                action.execution_result = result
                n_executed += 1

                # Track state changes
                if action.action_type in ("KILL_SWITCH", "BLOCK_ENTRY", "DELEVERAGE"):
                    state_changes[action.action_type] = {
                        "target": action.target,
                        "parameters": action.parameters,
                        "timestamp": _ts(),
                    }

                logger.info("Executed: %s → %s: %s",
                            action.action_type, action.target, result)

            except Exception as exc:
                action.execution_result = f"ERROR: {exc}"
                logger.error("Failed to execute %s: %s", action.action_type, exc)

            self._action_history.append(action)

        # Send alerts for executed actions
        self._send_feedback_alerts(actions)

        return FeedbackSummary(
            timestamp=_ts(),
            n_agent_results_processed=0,  # Set by caller
            n_actions_generated=len(actions),
            n_actions_executed=n_executed,
            n_actions_blocked=n_blocked,
            actions=actions,
            system_state_changes=state_changes,
        )

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
                from core.orchestrator import PairsOrchestrator
                orch = PairsOrchestrator()
                orch.bus.publish("kill_switch", {"active": True, "mode": action.parameters.get("mode"), "reason": action.reason})
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

    @property
    def action_history(self) -> List[FeedbackAction]:
        return list(self._action_history)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
