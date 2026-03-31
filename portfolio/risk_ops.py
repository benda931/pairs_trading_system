# -*- coding: utf-8 -*-
"""
portfolio/risk_ops.py — Risk Operating Model
=============================================

4-layer risk operating model:

  Layer 1: Instrument/Spread — per-pair stop-losses, vol caps
  Layer 2: Portfolio — drawdown tracking, heat-level state machine
  Layer 3: Drawdown / Degradation — progressive throttling
  Layer 4: Kill-switch / Governance — emergency circuit breakers

Heat-level state machine transitions:
  NORMAL → CAUTIOUS → THROTTLED → DEFENSIVE → RECOVERY_ONLY → HALTED

Transitions triggered by:
  - Drawdown thresholds (current_dd, rolling_dd_30d)
  - Signal degradation (regime CRISIS/BROKEN prevalence)
  - Consecutive losses
  - Manual override

Recovery (upgrading heat level) requires:
  - Drawdown improvement >= recovery threshold
  - No active CRISIS/BROKEN signals
  - Minimum holding period at current level

KillSwitchManager operates independently from DrawdownManager.
Kill-switch can be triggered by:
  - Extreme drawdown (exceeds hard_stop_dd threshold)
  - External governance signal
  - Manual operator trigger

Usage:
    dd_mgr = DrawdownManager(config)
    new_state = dd_mgr.update(current_dd, rolling_dd_30d, n_crisis_pairs)
    ks_mgr = KillSwitchManager(config)
    ks_state = ks_mgr.check(portfolio_value, peak_value)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from portfolio.contracts import (
    DrawdownState,
    KillSwitchMode,
    KillSwitchState,
    PortfolioHeatLevel,
    ThrottleState,
)

logger = logging.getLogger("portfolio.risk_ops")


# ── Drawdown / Heat configuration ─────────────────────────────────

@dataclass
class DrawdownConfig:
    """Thresholds driving the heat-level state machine."""

    # NORMAL → CAUTIOUS
    cautious_dd_threshold: float = 0.03       # 3% drawdown

    # CAUTIOUS → THROTTLED
    throttled_dd_threshold: float = 0.06      # 6% drawdown
    throttled_consecutive_losses: int = 5     # 5 consecutive losing days

    # THROTTLED → DEFENSIVE
    defensive_dd_threshold: float = 0.10     # 10% drawdown
    defensive_rolling_30d: float = 0.08      # 8% 30-day rolling

    # DEFENSIVE → RECOVERY_ONLY
    recovery_only_dd_threshold: float = 0.15  # 15% drawdown

    # RECOVERY_ONLY → HALTED
    halted_dd_threshold: float = 0.20        # 20% drawdown (hard stop)

    # Recovery thresholds (must IMPROVE by this much to step up)
    recovery_step_up_threshold: float = 0.025  # Must recover 2.5% to step up
    recovery_min_holding_days: int = 5         # Days before stepping up

    # Throttle size multipliers per heat level
    throttle_multipliers: dict[str, float] = field(default_factory=lambda: {
        "NORMAL":        1.0,
        "CAUTIOUS":      0.80,
        "THROTTLED":     0.55,
        "DEFENSIVE":     0.25,
        "RECOVERY_ONLY": 0.0,
        "HALTED":        0.0,
    })

    # Max new positions per heat level
    max_new_positions: dict[str, int] = field(default_factory=lambda: {
        "NORMAL":        20,
        "CAUTIOUS":      10,
        "THROTTLED":     5,
        "DEFENSIVE":     2,
        "RECOVERY_ONLY": 0,
        "HALTED":        0,
    })


@dataclass
class KillSwitchConfig:
    """Kill-switch triggering thresholds."""

    # SOFT kill-switch
    soft_dd_threshold: float = 0.12          # 12% drawdown → SOFT

    # REDUCE kill-switch
    reduce_dd_threshold: float = 0.16        # 16% → REDUCE

    # HARD kill-switch
    hard_dd_threshold: float = 0.20          # 20% → HARD
    hard_consecutive_losses: int = 10        # 10 consecutive losses → HARD
    hard_single_day_loss: float = 0.05       # 5% single-day loss → HARD

    # Scaling factors per mode
    scaling_factors: dict[str, float] = field(default_factory=lambda: {
        "OFF":    1.0,
        "SOFT":   0.75,
        "REDUCE": 0.40,
        "HARD":   0.0,
    })

    # Acknowledgment requirement before HARD kill-switch clears
    require_acknowledgment: bool = True


# ── Drawdown Manager ──────────────────────────────────────────────

class DrawdownManager:
    """
    Tracks portfolio drawdown and manages the heat-level state machine.

    State transitions are one-way downward automatically;
    upgrades require explicit recovery criteria.
    """

    def __init__(self, config: Optional[DrawdownConfig] = None):
        self._cfg = config or DrawdownConfig()
        self._state = DrawdownState()
        self._entered_level_at: datetime = datetime.utcnow()
        self._consecutive_losses: int = 0
        self._last_value: float = 1.0

    @property
    def state(self) -> DrawdownState:
        return self._state

    def update(
        self,
        current_value: float,
        *,
        peak_value: Optional[float] = None,
        rolling_dd_7d: float = 0.0,
        rolling_dd_30d: float = 0.0,
        n_crisis_pairs: int = 0,
        force_level: Optional[PortfolioHeatLevel] = None,
    ) -> DrawdownState:
        """
        Update drawdown state with current portfolio value.

        Parameters
        ----------
        current_value : float — current normalised portfolio value (1.0 = starting)
        peak_value : float — running peak (computed if not provided)
        rolling_dd_7d / 30d : float — rolling drawdown fractions
        n_crisis_pairs : int — number of CRISIS/BROKEN regime positions
        force_level : PortfolioHeatLevel — manual override

        Returns
        -------
        Updated DrawdownState
        """
        cfg = self._cfg

        # Update peak
        if peak_value is None:
            peak_value = max(self._state.peak_value, current_value)
        self._state.peak_value = peak_value
        self._state.current_value = current_value

        # Compute current drawdown
        if peak_value > 0:
            current_dd = max(0.0, (peak_value - current_value) / peak_value)
        else:
            current_dd = 0.0

        self._state.current_dd_pct = current_dd
        self._state.rolling_dd_7d = rolling_dd_7d
        self._state.rolling_dd_30d = rolling_dd_30d

        # Track consecutive losses
        if current_value < self._last_value:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = max(0, self._consecutive_losses - 1)
        self._last_value = current_value

        # ── Determine new heat level ──────────────────────────────
        if force_level is not None:
            new_level = force_level
        else:
            new_level = self._classify_heat_level(
                current_dd, rolling_dd_30d, self._consecutive_losses, n_crisis_pairs
            )

        # ── Apply one-way transitions (can only worsen, not recover automatically) ──
        current_level = self._state.heat_level
        new_level = self._apply_transition_rules(current_level, new_level)

        if new_level != current_level:
            logger.warning(
                "Heat level: %s → %s (dd=%.2f%%, consecutive_losses=%d)",
                current_level.value, new_level.value,
                current_dd * 100, self._consecutive_losses,
            )
            self._state.heat_level = new_level
            self._entered_level_at = datetime.utcnow()

        # ── Set throttle factor ───────────────────────────────────
        self._state.throttle_factor = cfg.throttle_multipliers.get(new_level.value, 0.0)
        self._state.max_new_positions = cfg.max_new_positions.get(new_level.value, 0)
        self._state.updated_at = datetime.utcnow()

        return self._state

    def attempt_recovery(self) -> bool:
        """
        Attempt to step up heat level (recover).

        Returns True if a step-up occurred.
        """
        cfg = self._cfg
        state = self._state

        # Must have held current level for min_holding_days
        days_at_level = (datetime.utcnow() - self._entered_level_at).days
        if days_at_level < cfg.recovery_min_holding_days:
            return False

        # Must have improved drawdown by recovery_step_up_threshold
        # (peak must have moved up relative to current)
        if state.current_dd_pct > (state.rolling_dd_30d - cfg.recovery_step_up_threshold):
            return False

        # Step up one level
        current = state.heat_level
        levels = [
            PortfolioHeatLevel.NORMAL,
            PortfolioHeatLevel.CAUTIOUS,
            PortfolioHeatLevel.THROTTLED,
            PortfolioHeatLevel.DEFENSIVE,
            PortfolioHeatLevel.RECOVERY_ONLY,
            PortfolioHeatLevel.HALTED,
        ]
        idx = levels.index(current)
        if idx == 0:
            return False  # Already NORMAL

        new_level = levels[idx - 1]
        logger.info("Heat level recovery: %s → %s", current.value, new_level.value)
        state.heat_level = new_level
        self._entered_level_at = datetime.utcnow()
        state.throttle_factor = cfg.throttle_multipliers.get(new_level.value, 1.0)
        state.max_new_positions = cfg.max_new_positions.get(new_level.value, 20)
        return True

    def build_throttle_state(self) -> ThrottleState:
        """Build a ThrottleState from current drawdown state."""
        level = self._state.heat_level
        cfg = self._cfg
        return ThrottleState(
            heat_level=level,
            size_multiplier=cfg.throttle_multipliers.get(level.value, 1.0),
            max_positions=50,  # total max — allocator uses max_new_positions
            max_new_entries_per_cycle=cfg.max_new_positions.get(level.value, 20),
            min_ranking_score_to_fund=self._min_score_for_level(level),
            available_risk_budget_fraction=cfg.throttle_multipliers.get(level.value, 1.0),
        )

    # ── Private helpers ───────────────────────────────────────────

    def _classify_heat_level(
        self,
        current_dd: float,
        rolling_30d: float,
        consecutive_losses: int,
        n_crisis: int,
    ) -> PortfolioHeatLevel:
        cfg = self._cfg

        if current_dd >= cfg.halted_dd_threshold:
            return PortfolioHeatLevel.HALTED
        if current_dd >= cfg.recovery_only_dd_threshold:
            return PortfolioHeatLevel.RECOVERY_ONLY
        if current_dd >= cfg.defensive_dd_threshold or rolling_30d >= cfg.defensive_rolling_30d:
            return PortfolioHeatLevel.DEFENSIVE
        if (current_dd >= cfg.throttled_dd_threshold
                or consecutive_losses >= cfg.throttled_consecutive_losses):
            return PortfolioHeatLevel.THROTTLED
        if current_dd >= cfg.cautious_dd_threshold or n_crisis > 0:
            return PortfolioHeatLevel.CAUTIOUS
        return PortfolioHeatLevel.NORMAL

    @staticmethod
    def _apply_transition_rules(
        current: PortfolioHeatLevel,
        target: PortfolioHeatLevel,
    ) -> PortfolioHeatLevel:
        """
        Enforce one-way worsening rule.

        Heat level can only worsen automatically; recovery requires
        explicit attempt_recovery() call.
        """
        levels = [
            PortfolioHeatLevel.NORMAL,
            PortfolioHeatLevel.CAUTIOUS,
            PortfolioHeatLevel.THROTTLED,
            PortfolioHeatLevel.DEFENSIVE,
            PortfolioHeatLevel.RECOVERY_ONLY,
            PortfolioHeatLevel.HALTED,
        ]
        current_idx = levels.index(current)
        target_idx = levels.index(target)
        # Only allow worsening (higher index) automatically
        return levels[max(current_idx, target_idx)]

    @staticmethod
    def _min_score_for_level(level: PortfolioHeatLevel) -> float:
        return {
            PortfolioHeatLevel.NORMAL: 0.10,
            PortfolioHeatLevel.CAUTIOUS: 0.25,
            PortfolioHeatLevel.THROTTLED: 0.40,
            PortfolioHeatLevel.DEFENSIVE: 0.60,
            PortfolioHeatLevel.RECOVERY_ONLY: 1.0,  # Effectively no new entries
            PortfolioHeatLevel.HALTED: 1.0,
        }.get(level, 0.10)


# ── Kill-Switch Manager ───────────────────────────────────────────

class KillSwitchManager:
    """
    Manages the portfolio kill-switch — independent of heat level.

    Kill-switch is a separate, faster emergency circuit breaker.
    It can be triggered by extreme drawdown, single-day loss,
    or external governance signal.
    """

    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        control_plane_callback: Optional[Callable[[str], None]] = None,
    ):
        self._cfg = config or KillSwitchConfig()
        self._state = KillSwitchState()
        self._control_plane_callback = control_plane_callback

    @property
    def state(self) -> KillSwitchState:
        return self._state

    def check(
        self,
        current_value: float,
        peak_value: float,
        *,
        single_day_return: float = 0.0,
        consecutive_losses: int = 0,
    ) -> KillSwitchState:
        """
        Evaluate kill-switch conditions.

        Parameters
        ----------
        current_value : float — normalised portfolio value
        peak_value : float — peak value
        single_day_return : float — today's return (negative = loss)
        consecutive_losses : int — days of consecutive losses

        Returns
        -------
        Updated KillSwitchState
        """
        cfg = self._cfg
        state = self._state

        current_dd = max(0.0, (peak_value - current_value) / max(1e-6, peak_value))
        triggered_rules: list[str] = []
        new_mode = KillSwitchMode.OFF

        # Escalate kill-switch mode based on severity
        if current_dd >= cfg.hard_dd_threshold:
            new_mode = KillSwitchMode.HARD
            triggered_rules.append(f"dd_hard:{current_dd:.1%}")

        elif abs(single_day_return) >= cfg.hard_single_day_loss and single_day_return < 0:
            new_mode = KillSwitchMode.HARD
            triggered_rules.append(f"single_day_loss:{single_day_return:.1%}")

        elif consecutive_losses >= cfg.hard_consecutive_losses:
            new_mode = KillSwitchMode.HARD
            triggered_rules.append(f"consecutive_losses:{consecutive_losses}")

        elif current_dd >= cfg.reduce_dd_threshold:
            new_mode = KillSwitchMode.REDUCE
            triggered_rules.append(f"dd_reduce:{current_dd:.1%}")

        elif current_dd >= cfg.soft_dd_threshold:
            new_mode = KillSwitchMode.SOFT
            triggered_rules.append(f"dd_soft:{current_dd:.1%}")

        # Kill-switch can only escalate, not de-escalate automatically
        modes_order = [KillSwitchMode.OFF, KillSwitchMode.SOFT, KillSwitchMode.REDUCE, KillSwitchMode.HARD]
        current_idx = modes_order.index(state.mode)
        new_idx = modes_order.index(new_mode)
        final_mode = modes_order[max(current_idx, new_idx)]

        was_triggered = len(triggered_rules) > 0 and final_mode != KillSwitchMode.OFF

        newly_triggered = was_triggered and not state.triggered

        if newly_triggered:
            logger.error(
                "KILL SWITCH ACTIVATED: mode=%s, dd=%.2f%%, rules=%s",
                final_mode.value, current_dd * 100, triggered_rules,
            )

        state.mode = final_mode
        state.triggered = was_triggered
        if was_triggered and state.triggered_at is None:
            state.triggered_at = datetime.utcnow()
        state.triggered_rules = triggered_rules
        state.severity_score = min(1.0, current_dd / max(0.01, cfg.hard_dd_threshold))
        state.scaling_factor = cfg.scaling_factors.get(final_mode.value, 1.0)
        reason = ", ".join(triggered_rules) if triggered_rules else ""
        state.reason = reason

        # ADR-005: Notify control_plane on new auto-trigger so both kill-switch
        # systems stay in sync (P0-KS remediation).
        if newly_triggered and self._control_plane_callback is not None:
            try:
                self._control_plane_callback(f"Auto-triggered: {reason}")
            except Exception:  # never let callback break trading
                pass

        return state

    def trigger_manual(self, mode: KillSwitchMode, reason: str) -> KillSwitchState:
        """Manually trigger kill-switch to a given mode."""
        self._state.mode = mode
        self._state.triggered = mode != KillSwitchMode.OFF
        self._state.triggered_at = datetime.utcnow() if self._state.triggered else None
        self._state.reason = reason
        self._state.triggered_rules = [f"manual:{reason}"]
        self._state.scaling_factor = self._cfg.scaling_factors.get(mode.value, 1.0)
        logger.warning("Manual kill-switch: mode=%s, reason=%s", mode.value, reason)

        # ADR-005: Notify control_plane on manual trigger (P0-KS remediation).
        if self._state.triggered and self._control_plane_callback is not None:
            try:
                self._control_plane_callback(f"Manual trigger: {reason}")
            except Exception:  # never let callback break trading
                pass

        return self._state

    def acknowledge(self) -> bool:
        """
        Acknowledge the kill-switch (operator confirms awareness).

        After acknowledgment, kill-switch can be reset if conditions clear.
        Returns True if acknowledgment was accepted.
        """
        if not self._state.triggered:
            return False
        self._state.acknowledged = True
        logger.info("Kill-switch acknowledged")
        return True

    def reset(self, *, force: bool = False) -> bool:
        """
        Reset kill-switch to OFF.

        Returns True if reset was accepted.
        """
        if self._cfg.require_acknowledgment and not self._state.acknowledged and not force:
            logger.warning("Kill-switch reset blocked: acknowledgment required")
            return False
        self._state = KillSwitchState()
        logger.info("Kill-switch reset to OFF")
        return True


# ── Risk Operations Facade ────────────────────────────────────────

class RiskOperationsManager:
    """
    Top-level façade combining DrawdownManager and KillSwitchManager.

    Use this as the single interface for risk state management.
    """

    def __init__(
        self,
        drawdown_config: Optional[DrawdownConfig] = None,
        kill_switch_config: Optional[KillSwitchConfig] = None,
    ):
        self._dd = DrawdownManager(drawdown_config)
        self._ks = KillSwitchManager(kill_switch_config)

    @property
    def drawdown_manager(self) -> DrawdownManager:
        return self._dd

    @property
    def kill_switch_manager(self) -> KillSwitchManager:
        return self._ks

    def update(
        self,
        current_value: float,
        peak_value: float,
        *,
        rolling_dd_7d: float = 0.0,
        rolling_dd_30d: float = 0.0,
        single_day_return: float = 0.0,
        consecutive_losses: int = 0,
        n_crisis_pairs: int = 0,
    ) -> tuple[DrawdownState, KillSwitchState]:
        """
        Update both managers with current portfolio state.

        Returns (DrawdownState, KillSwitchState).
        """
        dd_state = self._dd.update(
            current_value,
            peak_value=peak_value,
            rolling_dd_7d=rolling_dd_7d,
            rolling_dd_30d=rolling_dd_30d,
            n_crisis_pairs=n_crisis_pairs,
        )
        ks_state = self._ks.check(
            current_value, peak_value,
            single_day_return=single_day_return,
            consecutive_losses=consecutive_losses,
        )
        return dd_state, ks_state

    def is_any_restriction_active(self) -> bool:
        """True if any restriction (heat or kill-switch) is active."""
        heat = self._dd.state.heat_level
        ks = self._ks.state.mode
        return heat != PortfolioHeatLevel.NORMAL or ks != KillSwitchMode.OFF

    def summary(self) -> dict:
        return {
            "heat_level": self._dd.state.heat_level.value,
            "drawdown_pct": round(self._dd.state.current_dd_pct, 4),
            "throttle_factor": round(self._dd.state.throttle_factor, 4),
            "kill_switch_mode": self._ks.state.mode.value,
            "kill_switch_triggered": self._ks.state.triggered,
            "any_restriction_active": self.is_any_restriction_active(),
        }


# ── Factory ───────────────────────────────────────────────────────

def make_kill_switch_manager_with_control_plane(
    cfg: Optional[KillSwitchConfig] = None,
) -> KillSwitchManager:
    """
    Factory that creates a KillSwitchManager pre-wired to the ControlPlaneEngine.

    Use this in any live or paper trading context to ensure kill-switch events
    propagate to the canonical runtime state (ADR-005).

    Falls back to a plain KillSwitchManager (without callback) if the control
    plane package is unavailable — backward-compatible with all existing tests.
    """
    try:
        from control_plane.engine import get_control_plane
        cp = get_control_plane()

        def _callback(reason: str) -> None:
            cp.engage_kill_switch(reason=reason, operator="system:auto_kill_switch")

    except Exception:
        _callback = None  # type: ignore[assignment]

    return KillSwitchManager(
        cfg=cfg or KillSwitchConfig(),
        control_plane_callback=_callback,
    )
