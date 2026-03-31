# -*- coding: utf-8 -*-
"""
agents/portfolio_agents.py — Portfolio & Risk Agents
=====================================================

6 agents wrapping the portfolio construction and risk operating model:

  1. PortfolioConstructionAgent   — runs full allocation cycle
  2. CapitalBudgetAgent           — answers "can we fund this?" queries
  3. ExposureMonitorAgent         — computes current exposure, flags violations
  4. DrawdownMonitorAgent         — updates drawdown state, emits heat level
  5. KillSwitchAgent              — evaluates and manages kill-switch state
  6. DeRiskingAgent               — computes de-risking decisions when stressed

Each agent follows the BaseAgent protocol:
  - Narrow mandate: one job
  - No shared state mutation: returns results, caller decides what to do
  - Full audit trail via audit.log()

Task types and required payload keys are documented per agent.
"""
from __future__ import annotations

import logging
from typing import Any

from agents.base import AgentAuditLogger, BaseAgent
from core.contracts import AgentTask, PairId
from portfolio.allocator import AllocatorConfig, PortfolioAllocator
from portfolio.analytics import PortfolioAnalytics
from portfolio.capital import CapitalManager
from portfolio.contracts import (
    AllocationDecision,
    DrawdownState,
    ExposureSummary,
    KillSwitchMode,
    KillSwitchState,
    PortfolioHeatLevel,
    PortfolioSnapshot,
    SleeveDef,
)
from portfolio.exposures import ExposureAnalyzer, ExposureConfig
from portfolio.risk_ops import (
    DrawdownConfig,
    DrawdownManager,
    KillSwitchConfig,
    KillSwitchManager,
    RiskOperationsManager,
)

logger = logging.getLogger("agents.portfolio")


# ── Agent 1: Portfolio Construction ───────────────────────────────

class PortfolioConstructionAgent(BaseAgent):
    """
    Runs one full portfolio construction cycle.

    Task type: "run_allocation_cycle"
    Required payload:
      - intents: list[EntryIntent]
      - total_capital: float
    Optional payload:
      - active_allocations: list[AllocationDecision]
      - drawdown_state: DrawdownState
      - kill_switch: KillSwitchState
      - portfolio_vol: float
      - current_drawdown: float
      - vix_level: float
      - sector_map: dict[str, str]
      - cluster_map: dict[str, str]
      - spread_vols: dict[str, float]
      - hedge_ratios: dict[str, float]
      - allocator_config: AllocatorConfig

    Output:
      - decisions: list[AllocationDecision] (serialised as dicts)
      - diagnostics: PortfolioDiagnostics dict
      - n_funded: int
      - n_blocked: int
      - capital_deployed: float
    """
    NAME = "portfolio_construction"
    ALLOWED_TASK_TYPES = {"run_allocation_cycle"}
    REQUIRED_PAYLOAD_KEYS = {"intents", "total_capital"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        intents: list = task.payload["intents"]
        total_capital: float = float(task.payload["total_capital"])
        active_allocations: list = task.payload.get("active_allocations", [])
        drawdown_state: DrawdownState = task.payload.get("drawdown_state") or DrawdownState()
        kill_switch: KillSwitchState = task.payload.get("kill_switch") or KillSwitchState()
        allocator_cfg = task.payload.get("allocator_config") or AllocatorConfig()

        audit.log(f"Cycle: {len(intents)} intents, capital={total_capital:.0f}")
        audit.log(f"Heat: {drawdown_state.heat_level.value}, KS: {kill_switch.mode.value}")

        # Build capital manager
        capital_mgr = CapitalManager(total_capital=total_capital)

        # Register sleeves if provided
        sleeves = task.payload.get("sleeves", [])
        for s in sleeves:
            if isinstance(s, SleeveDef):
                capital_mgr.add_sleeve(s)
            elif isinstance(s, dict):
                capital_mgr.add_sleeve(SleeveDef(**s))

        # Restore existing allocations into capital manager
        for d in active_allocations:
            if isinstance(d, AllocationDecision) and d.approved:
                sleeve = d.rationale.sleeve or "default"
                if capital_mgr.get_sleeve(sleeve) is None:
                    capital_mgr.add_sleeve(SleeveDef(name=sleeve))
                try:
                    capital_mgr.allocate(sleeve, d.pair_id, d.approved_capital)
                except Exception as exc:
                    audit.warn(f"Could not restore {d.pair_id.label}: {exc}")

        allocator = PortfolioAllocator(capital_mgr, config=allocator_cfg)

        decisions, diagnostics = allocator.run_cycle(
            intents,
            active_allocations=active_allocations,
            drawdown_state=drawdown_state,
            kill_switch=kill_switch,
            portfolio_vol=task.payload.get("portfolio_vol"),
            current_drawdown=float(task.payload.get("current_drawdown", 0.0)),
            vix_level=float(task.payload.get("vix_level", 20.0)),
            sector_map=task.payload.get("sector_map", {}),
            cluster_map=task.payload.get("cluster_map", {}),
            spread_vols=task.payload.get("spread_vols", {}),
            hedge_ratios=task.payload.get("hedge_ratios", {}),
        )

        funded = [d for d in decisions if d.approved]
        capital_deployed = sum(d.approved_capital for d in funded)

        audit.log(
            f"Cycle complete: {len(funded)} funded, "
            f"capital_deployed={capital_deployed:.0f}, "
            f"n_hard_violations={diagnostics.n_hard_violations}"
        )

        return {
            "decisions": [d.to_dict() for d in decisions],
            "diagnostics": diagnostics.to_dict(),
            "n_funded": len(funded),
            "n_blocked": diagnostics.n_blocked_signal + diagnostics.n_blocked_risk + diagnostics.n_blocked_capital,
            "capital_deployed": capital_deployed,
        }


# ── Agent 2: Capital Budget ────────────────────────────────────────

class CapitalBudgetAgent(BaseAgent):
    """
    Answers capital availability queries without executing allocations.

    Task type: "check_capital_budget"
    Required payload:
      - total_capital: float
      - pairs: list[dict] — [{"pair_label": "A/B", "proposed_notional": 50000}]
    Optional payload:
      - active_allocations: list[AllocationDecision]
      - sleeves: list[SleeveDef] or list[dict]

    Output:
      - results: list[{"pair": ..., "can_fund": bool, "reason": str, "max_fundable": float}]
      - n_fundable: int
      - free_capital: float
      - utilisation_pct: float
    """
    NAME = "capital_budget"
    ALLOWED_TASK_TYPES = {"check_capital_budget"}
    REQUIRED_PAYLOAD_KEYS = {"total_capital", "pairs"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        total_capital = float(task.payload["total_capital"])
        pairs: list[dict] = task.payload["pairs"]
        active_allocations: list = task.payload.get("active_allocations", [])
        sleeves: list = task.payload.get("sleeves", [])

        capital_mgr = CapitalManager(total_capital=total_capital)
        for s in sleeves:
            if isinstance(s, SleeveDef):
                capital_mgr.add_sleeve(s)
            elif isinstance(s, dict):
                capital_mgr.add_sleeve(SleeveDef(**s))

        # Restore existing allocations
        for d in active_allocations:
            if isinstance(d, AllocationDecision) and d.approved:
                sleeve = d.rationale.sleeve or "default"
                if capital_mgr.get_sleeve(sleeve) is None:
                    capital_mgr.add_sleeve(SleeveDef(name=sleeve))
                try:
                    capital_mgr.allocate(sleeve, d.pair_id, d.approved_capital)
                except Exception:
                    pass

        results = []
        for p in pairs:
            label = p.get("pair_label", "")
            proposed = float(p.get("proposed_notional", 0.0))
            sleeve_name = p.get("sleeve", "default")

            if capital_mgr.get_sleeve(sleeve_name) is None:
                capital_mgr.add_sleeve(SleeveDef(name=sleeve_name))

            ok, reason = capital_mgr.can_allocate(sleeve_name, proposed)
            free_in_sleeve = capital_mgr.free_capital_in_sleeve(sleeve_name)

            results.append({
                "pair": label,
                "can_fund": ok,
                "reason": reason,
                "proposed_notional": proposed,
                "max_fundable": free_in_sleeve,
                "sleeve": sleeve_name,
            })
            audit.log(f"{label}: can_fund={ok} (proposed={proposed:.0f}, free={free_in_sleeve:.0f})")

        pool = capital_mgr.pool_snapshot()
        n_fundable = sum(1 for r in results if r["can_fund"])

        return {
            "results": results,
            "n_fundable": n_fundable,
            "free_capital": pool.free_capital,
            "utilisation_pct": pool.utilisation_pct,
        }


# ── Agent 3: Exposure Monitor ──────────────────────────────────────

class ExposureMonitorAgent(BaseAgent):
    """
    Computes current portfolio exposure and flags violations.

    Task type: "compute_exposure"
    Required payload:
      - active_allocations: list[AllocationDecision]
      - total_capital: float
    Optional payload:
      - sector_map: dict[str, str]
      - cluster_map: dict[str, str]
      - factor_betas: dict[str, dict[str, float]]
      - exposure_config: ExposureConfig

    Output:
      - exposure: ExposureSummary dict
      - violations: list[str] — exposure limit breaches
      - n_dominant_legs: int
      - n_overcrowded_clusters: int
      - gross_leverage: float
      - is_compliant: bool
    """
    NAME = "exposure_monitor"
    ALLOWED_TASK_TYPES = {"compute_exposure"}
    REQUIRED_PAYLOAD_KEYS = {"active_allocations", "total_capital"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        active_allocations: list = task.payload["active_allocations"]
        total_capital = float(task.payload["total_capital"])
        sector_map = task.payload.get("sector_map", {})
        cluster_map = task.payload.get("cluster_map", {})
        factor_betas = task.payload.get("factor_betas", {})
        exp_cfg = task.payload.get("exposure_config") or ExposureConfig()

        audit.log(f"Computing exposure for {len(active_allocations)} positions, capital={total_capital:.0f}")

        # Filter to AllocationDecision objects
        decisions = [a for a in active_allocations if isinstance(a, AllocationDecision) and a.approved]

        analyzer = ExposureAnalyzer(config=exp_cfg)
        summary = analyzer.compute(
            decisions, total_capital,
            sector_map=sector_map,
            cluster_map=cluster_map,
            factor_betas=factor_betas,
        )

        # Flag violations
        violations: list[str] = []
        cfg = exp_cfg

        if summary.gross_leverage > cfg.max_gross_leverage:
            violations.append(f"gross_leverage:{summary.gross_leverage:.2f}>{cfg.max_gross_leverage}")

        if summary.net_leverage > cfg.max_net_leverage:
            violations.append(f"net_leverage:{summary.net_leverage:.2f}>{cfg.max_net_leverage}")

        if summary.max_sector_concentration > cfg.max_sector_fraction:
            violations.append(f"sector_concentration:{summary.max_sector_concentration:.1%}>{cfg.max_sector_fraction:.1%}")

        if summary.max_cluster_concentration > cfg.max_cluster_fraction:
            violations.append(f"cluster_concentration:{summary.max_cluster_concentration:.1%}>{cfg.max_cluster_fraction:.1%}")

        n_dominant = sum(1 for s in summary.shared_legs if s.is_dominant)
        n_overcrowded = sum(1 for c in summary.cluster_exposures if c.is_overcrowded)

        for s in summary.shared_legs:
            if s.is_dominant:
                violations.append(f"dominant_leg:{s.instrument}(n={s.n_pairs_using})")
                audit.warn(f"Dominant leg: {s.instrument} used in {s.n_pairs_using} pairs")

        is_compliant = len(violations) == 0

        audit.log(
            f"Exposure: gross_lev={summary.gross_leverage:.2f}, "
            f"n_dominant_legs={n_dominant}, violations={len(violations)}"
        )

        return {
            "exposure": summary.to_dict(),
            "violations": violations,
            "n_dominant_legs": n_dominant,
            "n_overcrowded_clusters": n_overcrowded,
            "gross_leverage": summary.gross_leverage,
            "net_leverage": summary.net_leverage,
            "is_compliant": is_compliant,
        }


# ── Agent 4: Drawdown Monitor ──────────────────────────────────────

class DrawdownMonitorAgent(BaseAgent):
    """
    Updates drawdown state and returns heat level classification.

    Task type: "update_drawdown"
    Required payload:
      - current_value: float  (normalised: 1.0 = start)
      - peak_value: float
    Optional payload:
      - rolling_dd_7d: float
      - rolling_dd_30d: float
      - n_crisis_pairs: int
      - drawdown_config: DrawdownConfig
      - current_state: DrawdownState (prior state for continuity)

    Output:
      - drawdown_state: DrawdownState dict
      - heat_level: str
      - throttle_factor: float
      - heat_changed: bool
      - is_stressed: bool
    """
    NAME = "drawdown_monitor"
    ALLOWED_TASK_TYPES = {"update_drawdown"}
    REQUIRED_PAYLOAD_KEYS = {"current_value", "peak_value"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        current_value = float(task.payload["current_value"])
        peak_value = float(task.payload["peak_value"])
        rolling_dd_7d = float(task.payload.get("rolling_dd_7d", 0.0))
        rolling_dd_30d = float(task.payload.get("rolling_dd_30d", 0.0))
        n_crisis_pairs = int(task.payload.get("n_crisis_pairs", 0))
        dd_cfg = task.payload.get("drawdown_config") or DrawdownConfig()
        prior_state: DrawdownState = task.payload.get("current_state") or DrawdownState()

        current_dd = max(0.0, (peak_value - current_value) / max(1e-6, peak_value))
        audit.log(
            f"Drawdown update: current={current_value:.4f}, peak={peak_value:.4f}, "
            f"dd={current_dd:.2%}, rolling_30d={rolling_dd_30d:.2%}, crisis_pairs={n_crisis_pairs}"
        )

        mgr = DrawdownManager(dd_cfg)
        # Inject prior state
        mgr._state = prior_state

        new_state = mgr.update(
            current_value,
            peak_value=peak_value,
            rolling_dd_7d=rolling_dd_7d,
            rolling_dd_30d=rolling_dd_30d,
            n_crisis_pairs=n_crisis_pairs,
        )

        prior_heat = prior_state.heat_level.value
        new_heat = new_state.heat_level.value
        heat_changed = prior_heat != new_heat

        if heat_changed:
            audit.warn(f"Heat level changed: {prior_heat} → {new_heat}")
        else:
            audit.log(f"Heat level unchanged: {new_heat}")

        return {
            "drawdown_state": new_state.to_dict(),
            "heat_level": new_heat,
            "throttle_factor": new_state.throttle_factor,
            "current_dd_pct": new_state.current_dd_pct,
            "heat_changed": heat_changed,
            "is_stressed": new_state.is_stressed,
        }


# ── Agent 5: Kill-Switch ───────────────────────────────────────────

class KillSwitchAgent(BaseAgent):
    """
    Evaluates kill-switch conditions and manages state.

    Task types:
      "evaluate_kill_switch"  — automated evaluation
      "manual_trigger"        — operator manual override
      "reset_kill_switch"     — reset to OFF

    Required payload ("evaluate_kill_switch"):
      - current_value: float
      - peak_value: float
    Optional payload:
      - single_day_return: float
      - consecutive_losses: int
      - ks_config: KillSwitchConfig
      - current_state: KillSwitchState (prior state)

    Required payload ("manual_trigger"):
      - mode: str  — "SOFT" | "REDUCE" | "HARD"
      - reason: str

    Output:
      - kill_switch_state: KillSwitchState dict
      - mode: str
      - is_blocking: bool
      - triggered: bool
    """
    NAME = "kill_switch"
    ALLOWED_TASK_TYPES = {"evaluate_kill_switch", "manual_trigger", "reset_kill_switch"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        if task.task_type == "evaluate_kill_switch":
            return self._evaluate(task, audit)
        elif task.task_type == "manual_trigger":
            return self._manual_trigger(task, audit)
        else:
            return self._reset(task, audit)

    def _evaluate(self, task: AgentTask, audit: AgentAuditLogger) -> dict:
        current_value = float(task.payload["current_value"])
        peak_value = float(task.payload["peak_value"])
        single_day = float(task.payload.get("single_day_return", 0.0))
        consecutive = int(task.payload.get("consecutive_losses", 0))
        ks_cfg = task.payload.get("ks_config") or KillSwitchConfig()
        prior: KillSwitchState = task.payload.get("current_state") or KillSwitchState()

        current_dd = max(0.0, (peak_value - current_value) / max(1e-6, peak_value))
        audit.log(
            f"Evaluating KS: dd={current_dd:.2%}, "
            f"single_day={single_day:.2%}, consecutive={consecutive}"
        )

        mgr = KillSwitchManager(ks_cfg)
        mgr._state = prior

        new_state = mgr.check(
            current_value, peak_value,
            single_day_return=single_day,
            consecutive_losses=consecutive,
        )

        if new_state.triggered:
            audit.error(f"Kill-switch TRIGGERED: mode={new_state.mode.value}, reason={new_state.reason}")
        else:
            audit.log(f"Kill-switch OK: mode={new_state.mode.value}")

        return {
            "kill_switch_state": new_state.to_dict(),
            "mode": new_state.mode.value,
            "is_blocking": new_state.is_blocking_new_entries(),
            "triggered": new_state.triggered,
            "scaling_factor": new_state.scaling_factor,
        }

    def _manual_trigger(self, task: AgentTask, audit: AgentAuditLogger) -> dict:
        mode_str = task.payload.get("mode", "SOFT").upper()
        reason = task.payload.get("reason", "manual_operator_trigger")
        ks_cfg = task.payload.get("ks_config") or KillSwitchConfig()

        try:
            mode = KillSwitchMode(mode_str)
        except ValueError:
            raise ValueError(f"Invalid KillSwitchMode: {mode_str}")

        audit.warn(f"Manual kill-switch trigger: mode={mode_str}, reason={reason}")
        mgr = KillSwitchManager(ks_cfg)
        new_state = mgr.trigger_manual(mode, reason)

        return {
            "kill_switch_state": new_state.to_dict(),
            "mode": new_state.mode.value,
            "is_blocking": new_state.is_blocking_new_entries(),
            "triggered": new_state.triggered,
            "scaling_factor": new_state.scaling_factor,
        }

    def _reset(self, task: AgentTask, audit: AgentAuditLogger) -> dict:
        ks_cfg = task.payload.get("ks_config") or KillSwitchConfig()
        force = bool(task.payload.get("force", False))
        prior: KillSwitchState = task.payload.get("current_state") or KillSwitchState()

        mgr = KillSwitchManager(ks_cfg)
        mgr._state = prior

        success = mgr.reset(force=force)
        if not success:
            audit.error("Kill-switch reset BLOCKED: acknowledgment required")
        else:
            audit.log("Kill-switch reset to OFF")

        return {
            "kill_switch_state": mgr.state.to_dict(),
            "mode": mgr.state.mode.value,
            "reset_successful": success,
            "is_blocking": mgr.state.is_blocking_new_entries(),
        }


# ── Agent 6: De-Risking ────────────────────────────────────────────

class DeRiskingAgent(BaseAgent):
    """
    Computes de-risking decisions when portfolio is stressed.

    Task type: "compute_derisking"
    Required payload:
      - active_allocations: list[AllocationDecision]
      - drawdown_state: DrawdownState
    Optional payload:
      - allocator_config: AllocatorConfig

    Output:
      - derisking_decision: DeRiskingDecision dict
      - n_exits: int
      - n_reduces: int
      - pairs_to_exit: list[str]
      - pairs_to_reduce: list[str]
      - is_urgent: bool
    """
    NAME = "derisking"
    ALLOWED_TASK_TYPES = {"compute_derisking"}
    REQUIRED_PAYLOAD_KEYS = {"active_allocations", "drawdown_state"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        active_allocations: list = task.payload["active_allocations"]
        drawdown_state: DrawdownState = task.payload["drawdown_state"]
        allocator_cfg = task.payload.get("allocator_config") or AllocatorConfig()

        # Filter to approved AllocationDecisions
        decisions = [
            a for a in active_allocations
            if isinstance(a, AllocationDecision) and a.approved
        ]

        heat = drawdown_state.heat_level
        audit.log(
            f"De-risking evaluation: heat={heat.value}, "
            f"n_positions={len(decisions)}, dd={drawdown_state.current_dd_pct:.2%}"
        )

        if heat == PortfolioHeatLevel.NORMAL:
            audit.log("Heat=NORMAL — no de-risking required")
            from portfolio.contracts import DeRiskingDecision
            return {
                "derisking_decision": DeRiskingDecision(heat_level=heat).to_dict(),
                "n_exits": 0,
                "n_reduces": 0,
                "pairs_to_exit": [],
                "pairs_to_reduce": [],
                "is_urgent": False,
            }

        # Build a minimal allocator just for the de-risking logic
        capital_mgr = CapitalManager(total_capital=1_000_000.0)
        allocator = PortfolioAllocator(capital_mgr, config=allocator_cfg)
        dr_decision = allocator.compute_derisking(decisions, drawdown_state)

        exits = [p.label for p in dr_decision.pairs_to_exit]
        reduces = [p.label for p in dr_decision.pairs_to_reduce]

        if exits:
            audit.warn(f"De-risk: EXIT {len(exits)} pairs: {exits}")
        if reduces:
            audit.log(f"De-risk: REDUCE {len(reduces)} pairs")

        return {
            "derisking_decision": dr_decision.to_dict(),
            "n_exits": len(exits),
            "n_reduces": len(reduces),
            "pairs_to_exit": exits,
            "pairs_to_reduce": reduces,
            "reduction_fractions": dr_decision.reduction_fractions,
            "is_urgent": dr_decision.urgency == "URGENT",
            "reason": dr_decision.reason,
        }
