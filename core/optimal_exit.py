# -*- coding: utf-8 -*-
"""
core/optimal_exit.py — Optimal Stopping & Exit Strategy Engine
================================================================

Mathematical framework for optimal trade exits:

1. **Ornstein-Uhlenbeck Optimal Stopping**
   - Analytical solution for OU exit boundary
   - Time-dependent exit thresholds (decaying with holding time)
   - Half-life aware target computation

2. **Dynamic Programming Exit**
   - Bellman equation discretization
   - State: (z-score, holding_days, regime)
   - Optimal policy: exit/hold at each state

3. **Trailing Stop Optimization**
   - Volatility-adjusted trailing stop
   - ATR-based stop with optimal multiplier
   - Asymmetric stops (tighter on losing side)

4. **Exit Signal Composite**
   - Multi-signal exit scoring
   - Time decay penalty
   - Regime-conditional exit thresholds

Usage:
    from core.optimal_exit import OptimalExitEngine

    engine = OptimalExitEngine(half_life=15, entry_z=2.0)
    signal = engine.compute_exit_signal(
        current_z=0.8, holding_days=10, regime="NORMAL"
    )
    if signal.should_exit:
        print(f"EXIT: {signal.reason} (score={signal.exit_score:.2f})")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """Composite exit signal."""
    should_exit: bool
    exit_score: float                    # [0, 1], higher = more urgent to exit
    reason: str                          # Primary exit reason
    holding_days: int
    current_z: float
    exit_threshold: float                # Dynamic exit threshold at this point
    time_decay_factor: float             # How much time decay penalizes holding
    regime_factor: float                 # Regime adjustment
    pnl_at_exit: Optional[float] = None
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimalBoundary:
    """Optimal exit boundary for OU process."""
    half_life: float
    entry_z: float
    exit_z_static: float                 # Time-independent exit threshold
    exit_z_dynamic: List[float] = field(default_factory=list)  # Time-dependent thresholds
    holding_days: List[int] = field(default_factory=list)
    expected_profit_at_entry: float = 0.0
    max_expected_holding: int = 0        # Optimal max holding period
    breakeven_day: int = 0               # Day where expected profit = 0


@dataclass
class TrailingStopConfig:
    """Optimized trailing stop parameters."""
    initial_stop_z: float                # Initial stop in z-score units
    trailing_atr_multiplier: float       # ATR multiplier for trailing
    tighten_after_days: int              # Start tightening after N days
    tighten_rate: float                  # How fast stop tightens (per day)
    asymmetric_ratio: float              # Loss side / gain side multiplier


class OptimalExitEngine:
    """
    Optimal exit strategy engine based on OU process theory.

    Uses the half-life and entry z-score to compute time-dependent
    optimal exit boundaries, then combines with trailing stops and
    regime adjustments for a composite exit signal.
    """

    def __init__(
        self,
        half_life: float = 15.0,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        max_holding_days: int = 60,
        risk_free_daily: float = 0.04 / 252,
        transaction_cost_z: float = 0.05,    # Cost in z-score units
    ):
        self.half_life = half_life
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_holding = max_holding_days
        self.rf_daily = risk_free_daily
        self.txn_cost_z = transaction_cost_z

        # OU parameters
        self.theta = np.log(2) / max(half_life, 0.1)  # Mean-reversion speed
        self._boundary = None

    # ── Optimal Boundary ──────────────────────────────────────

    def compute_optimal_boundary(self) -> OptimalBoundary:
        """
        Compute the time-dependent optimal exit boundary.

        For an OU process entered at z = entry_z, the expected z-score
        after t days is: E[z_t] = z_0 * exp(-θt)

        The optimal exit maximizes expected profit net of time cost:
            Profit(t) = (z_0 - E[z_t]) - txn_cost - rf * t

        The boundary is where expected marginal profit from waiting = 0.
        """
        days = list(range(1, self.max_holding + 1))
        dynamic_thresholds = []

        z0 = self.entry_z
        best_profit = -np.inf
        best_day = 1
        breakeven = self.max_holding

        for t in days:
            # Expected z at time t (OU mean reversion)
            expected_z = z0 * np.exp(-self.theta * t)

            # Expected profit from exiting at time t
            profit = (z0 - expected_z) - self.txn_cost_z - self.rf_daily * t

            # Time decay: exit threshold rises with holding time
            # (we become more eager to exit as expected profit diminishes)
            time_penalty = 1.0 - np.exp(-self.theta * t * 0.5)  # Slower than full MR
            dynamic_exit = self.exit_z + (z0 - self.exit_z) * time_penalty

            dynamic_thresholds.append(round(float(dynamic_exit), 4))

            if profit > best_profit:
                best_profit = profit
                best_day = t

            if profit <= 0 and breakeven == self.max_holding:
                breakeven = t

        self._boundary = OptimalBoundary(
            half_life=self.half_life,
            entry_z=self.entry_z,
            exit_z_static=self.exit_z,
            exit_z_dynamic=dynamic_thresholds,
            holding_days=days,
            expected_profit_at_entry=round(float(best_profit), 6),
            max_expected_holding=best_day,
            breakeven_day=breakeven,
        )
        return self._boundary

    # ── Exit Signal ───────────────────────────────────────────

    def compute_exit_signal(
        self,
        current_z: float,
        holding_days: int,
        regime: str = "NORMAL",
        spread_vol: float = 1.0,
        entry_pnl_pct: float = 0.0,
    ) -> ExitSignal:
        """
        Compute composite exit signal.

        Combines:
        1. OU optimal boundary (time-dependent)
        2. Static thresholds (exit_z, stop_z)
        3. Time decay penalty
        4. Regime adjustment
        5. PnL-based profit taking
        """
        if self._boundary is None:
            self.compute_optimal_boundary()

        # 1. Dynamic exit threshold
        if holding_days <= len(self._boundary.exit_z_dynamic):
            dynamic_exit = self._boundary.exit_z_dynamic[holding_days - 1]
        else:
            dynamic_exit = self.entry_z * 0.9  # Almost at entry = definitely exit

        # 2. Time decay factor [0, 1] — 1 = full time pressure to exit
        time_decay = 1.0 - np.exp(-holding_days / max(self.half_life * 2, 1))

        # 3. Regime factor — more aggressive exits in bad regimes
        regime_factors = {
            "CALM": 0.8,      # Patient
            "LOW_VOL": 0.8,
            "NORMAL": 1.0,
            "TENSION": 1.3,   # Exit faster
            "HIGH_VOL": 1.3,
            "CRISIS": 2.0,    # Very aggressive exit
            "MEAN_REVERTING": 0.7,  # Most patient
            "TRENDING": 1.5,  # Exit quickly — MR failing
            "BROKEN": 2.5,    # Emergency exit
        }
        regime_factor = regime_factors.get(regime.upper(), 1.0)

        # 4. Adjusted exit threshold
        adjusted_exit = dynamic_exit * regime_factor

        # 5. Exit scoring components
        components = {}

        # Z-score proximity to exit
        z_signal = max(0, 1.0 - abs(current_z) / max(adjusted_exit, 0.1))
        components["z_proximity"] = round(z_signal, 4)

        # Time pressure
        time_signal = time_decay * 0.3  # Max 30% weight from time alone
        components["time_pressure"] = round(time_signal, 4)

        # Stop loss hit
        stop_signal = 1.0 if abs(current_z) > self.stop_z else 0.0
        components["stop_loss"] = stop_signal

        # Max holding exceeded
        max_hold_signal = 1.0 if holding_days > self.max_holding else 0.0
        components["max_holding"] = max_hold_signal

        # Profit taking (if PnL > 2x expected)
        profit_signal = 0.0
        if entry_pnl_pct > 0.02:  # > 2% profit
            profit_signal = min(1.0, entry_pnl_pct / 0.05)  # Full at 5%
        components["profit_taking"] = round(profit_signal, 4)

        # Regime stress
        regime_signal = max(0, (regime_factor - 1.0) / 1.5)
        components["regime_stress"] = round(regime_signal, 4)

        # Composite exit score
        exit_score = (
            z_signal * 0.35
            + time_signal * 0.15
            + stop_signal * 0.20
            + max_hold_signal * 0.10
            + profit_signal * 0.10
            + regime_signal * 0.10
        )

        # Determine if we should exit
        should_exit = (
            abs(current_z) <= adjusted_exit  # Mean reverted enough
            or stop_signal > 0               # Stop loss hit
            or max_hold_signal > 0           # Max holding exceeded
            or (exit_score > 0.70 and regime_factor > 1.5)  # High stress + high score
        )

        # Reason
        if stop_signal > 0:
            reason = f"STOP_LOSS (z={current_z:.2f} > {self.stop_z:.1f})"
        elif max_hold_signal > 0:
            reason = f"MAX_HOLDING ({holding_days}d > {self.max_holding}d)"
        elif abs(current_z) <= adjusted_exit:
            reason = f"MEAN_REVERTED (z={current_z:.2f}, threshold={adjusted_exit:.2f})"
        elif profit_signal > 0.8:
            reason = f"PROFIT_TAKE (pnl={entry_pnl_pct:.1%})"
        elif regime_signal > 0.5:
            reason = f"REGIME_EXIT ({regime})"
        else:
            reason = "HOLD"

        return ExitSignal(
            should_exit=should_exit,
            exit_score=round(exit_score, 4),
            reason=reason,
            holding_days=holding_days,
            current_z=round(current_z, 4),
            exit_threshold=round(adjusted_exit, 4),
            time_decay_factor=round(time_decay, 4),
            regime_factor=round(regime_factor, 4),
            pnl_at_exit=round(entry_pnl_pct, 6) if entry_pnl_pct else None,
            components=components,
        )

    # ── Trailing Stop Optimization ────────────────────────────

    def optimize_trailing_stop(
        self,
        spread_series: pd.Series,
        n_simulations: int = 500,
    ) -> TrailingStopConfig:
        """
        Find optimal trailing stop parameters via simulation.

        Tests multiple ATR multipliers and tightening schedules,
        selects the one that maximizes risk-adjusted return.
        """
        s = spread_series.dropna().values
        n = len(s)
        if n < 100:
            return TrailingStopConfig(
                initial_stop_z=self.stop_z,
                trailing_atr_multiplier=2.0,
                tighten_after_days=int(self.half_life),
                tighten_rate=0.02,
                asymmetric_ratio=1.3,
            )

        vol = float(np.std(s))
        best_sharpe = -np.inf
        best_config = None

        rng = np.random.default_rng(42)

        for atr_mult in [1.5, 2.0, 2.5, 3.0]:
            for tighten_days in [int(self.half_life * 0.5), int(self.half_life), int(self.half_life * 2)]:
                for tighten_rate in [0.01, 0.02, 0.05]:
                    # Simulate trades
                    pnls = []
                    for _ in range(min(n_simulations, n // 10)):
                        start = rng.integers(0, max(1, n - self.max_holding))
                        entry_z = s[start] / max(vol, 1e-6)

                        if abs(entry_z) < 1.0:
                            continue

                        stop = atr_mult * vol
                        best_price = s[start]
                        exit_pnl = 0.0

                        for t in range(1, min(self.max_holding, n - start)):
                            current = s[start + t]

                            # Tighten after threshold
                            if t > tighten_days:
                                stop *= (1 - tighten_rate)

                            # Trailing: adjust stop relative to best
                            if entry_z > 0:  # Short spread
                                best_price = min(best_price, current)
                                if current - best_price > stop:
                                    exit_pnl = (s[start] - current) / max(vol, 1e-6)
                                    break
                            else:  # Long spread
                                best_price = max(best_price, current)
                                if best_price - current > stop:
                                    exit_pnl = (current - s[start]) / max(vol, 1e-6)
                                    break

                            # Mean reversion exit
                            current_z = current / max(vol, 1e-6)
                            if abs(current_z) < self.exit_z:
                                exit_pnl = abs(entry_z) - abs(current_z)
                                break
                        else:
                            exit_pnl = (abs(entry_z) - abs(s[min(start + self.max_holding, n - 1)] / max(vol, 1e-6)))

                        pnls.append(exit_pnl)

                    if len(pnls) > 10:
                        sharpe = float(np.mean(pnls) / max(np.std(pnls), 1e-6))
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_config = TrailingStopConfig(
                                initial_stop_z=round(atr_mult, 2),
                                trailing_atr_multiplier=round(atr_mult, 2),
                                tighten_after_days=tighten_days,
                                tighten_rate=round(tighten_rate, 4),
                                asymmetric_ratio=1.3,
                            )

        return best_config or TrailingStopConfig(
            initial_stop_z=self.stop_z,
            trailing_atr_multiplier=2.0,
            tighten_after_days=int(self.half_life),
            tighten_rate=0.02,
            asymmetric_ratio=1.3,
        )


# ── Standalone OU-based optimal exit z ───────────────────────────────────────

def optimal_exit_z_ou(
    half_life: float,
    holding_days_so_far: int,
    entry_z: float,
    current_z: float,
    spread_vol: float,
    transaction_cost_pct: float = 0.001,
    target_sharpe: float = 0.5,
) -> float:
    """
    Compute the optimal exit z-score using OU process dynamics.

    Rather than exiting at a fixed z=0.5, this computes the expected future
    P&L as a function of current z and remaining expected holding time.

    Logic: Exit when expected_remaining_pnl < transaction_cost_of_holding.

    The OU speed of mean reversion: theta = ln(2) / half_life
    Expected z at time t: E[z(t)] = current_z * exp(-theta * t)
    Expected P&L from here: integral of E[z] from 0 to T

    Returns: optimal exit z (lower = tighter, higher = hold longer)
    """
    if half_life <= 0 or np.isnan(half_life) or np.isinf(half_life):
        return 0.5  # fallback to standard

    theta = np.log(2) / half_life   # OU mean-reversion speed

    # Expected time to exit: remaining half-lives until current z decays to noise
    # z decays exponentially: z(t) = current_z * exp(-theta * t)
    # Exit when z(t) reaches noise level (≈ 0.3-0.5σ)
    noise_floor = max(0.3, transaction_cost_pct / max(spread_vol, 1e-6))

    if abs(current_z) <= noise_floor:
        return float(noise_floor)  # Already at exit territory

    # Time for z to decay to noise_floor from current_z
    time_to_noise = -np.log(noise_floor / abs(current_z)) / theta  # in days

    # Expected P&L from current z to exit:
    # = integral(current_z * exp(-theta*t), 0, time_to_noise) * spread_vol
    # = current_z / theta * (1 - exp(-theta * time_to_noise)) * spread_vol
    expected_pnl_pct = abs(current_z) / theta * (1 - np.exp(-theta * time_to_noise)) * spread_vol

    # Daily holding cost (borrow + opportunity cost)
    daily_holding_cost = transaction_cost_pct / max(holding_days_so_far + 1, 1)

    # If expected remaining P&L doesn't justify holding: exit now
    if expected_pnl_pct < daily_holding_cost * 5:  # need 5x cost coverage
        return float(min(abs(current_z) + 0.1, abs(entry_z) - 0.1))  # exit imminently

    # Standard case: exit at noise floor (tighter than fixed 0.5 for fast-reverting pairs)
    # For slow pairs (HL > 30d), widen exit slightly to avoid premature exit
    if half_life < 10:
        exit_z = noise_floor * 0.8  # Very fast: exit early
    elif half_life < 20:
        exit_z = noise_floor
    elif half_life < 40:
        exit_z = noise_floor * 1.2
    else:
        exit_z = noise_floor * 1.5  # Slow: let it run more

    return float(np.clip(exit_z, 0.1, abs(entry_z) * 0.8))
