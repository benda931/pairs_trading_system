# -*- coding: utf-8 -*-
"""
core/trade_monitor.py — Trade Health Monitoring & Exit Signals
==============================================================

Ported from srv_quant_system. Monitors active pair trades and generates
structured exit signals based on 6 health dimensions.

Features:
- Z-score compression/extension monitoring
- Time decay relative to half-life
- Regime deterioration detection
- P&L proxy tracking
- Composite health scoring (0-1)
- Exit signals with urgency levels
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("core.trade_monitor")


# ── enums & dataclasses ──────────────────────────────────────────

class ExitType(str, Enum):
    PROFIT_TAKE = "PROFIT_TAKE"
    STOP_LOSS = "STOP_LOSS"
    TIME_EXIT = "TIME_EXIT"
    REGIME_EXIT = "REGIME_EXIT"
    HEDGE_REBAL = "HEDGE_REBAL"
    MANUAL = "MANUAL"


class Urgency(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    END_OF_DAY = "END_OF_DAY"
    NEXT_SESSION = "NEXT_SESSION"
    MONITOR = "MONITOR"


@dataclass
class ExitSignal:
    """Structured exit recommendation."""
    signal_type: ExitType
    urgency: Urgency
    strength: float = 0.0  # 0-1
    action: str = ""  # human-readable action
    reason: str = ""


@dataclass
class TradeHealthReport:
    """Per-trade health assessment."""
    pair_label: str
    entry_date: datetime | None = None
    days_held: int = 0

    # z-score dimension
    entry_z: float = np.nan
    current_z: float = np.nan
    z_compression_pct: float = 0.0  # how much z moved toward 0
    z_extension: bool = False  # z moved further away

    # time dimension
    half_life: float = np.nan
    half_lives_elapsed: float = 0.0
    time_decay_score: float = 1.0  # 1=fresh, 0=expired

    # regime dimension
    entry_regime: str = ""
    current_regime: str = ""
    regime_deteriorated: bool = False

    # P&L dimension
    unrealized_pnl_pct: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    pnl_score: float = 0.5  # 0=max loss, 1=max profit

    # composite
    health_score: float = 0.5  # 0-1
    exit_signals: list[ExitSignal] = field(default_factory=list)
    status: str = "ACTIVE"  # ACTIVE, EXIT_RECOMMENDED, FORCE_EXIT


@dataclass
class PortfolioMonitorSummary:
    """Portfolio-level monitoring summary."""
    timestamp: datetime = field(default_factory=datetime.now)
    total_pairs: int = 0
    healthy_pairs: int = 0
    warning_pairs: int = 0
    critical_pairs: int = 0
    exit_recommended: int = 0
    avg_health: float = 0.0
    reports: list[TradeHealthReport] = field(default_factory=list)


# ── engine ────────────────────────────────────────────────────────

class TradeMonitorEngine:
    """Monitors active pair trades and generates exit signals.

    Health score = 30% z_score + 25% time + 25% regime + 20% pnl
    """

    # weights for composite health score
    W_Z = 0.30
    W_TIME = 0.25
    W_REGIME = 0.25
    W_PNL = 0.20

    def __init__(
        self,
        z_target: float = 0.5,
        z_stop: float = 4.0,
        max_half_lives: float = 3.0,
        profit_take_pct: float = 0.05,
        stop_loss_pct: float = -0.03,
    ):
        self.z_target = z_target
        self.z_stop = z_stop
        self.max_half_lives = max_half_lives
        self.profit_take_pct = profit_take_pct
        self.stop_loss_pct = stop_loss_pct

    def assess_trade(
        self,
        pair_label: str,
        entry_z: float,
        current_z: float,
        half_life: float,
        entry_date: datetime,
        current_regime: str = "NORMAL",
        entry_regime: str = "NORMAL",
        unrealized_pnl_pct: float = 0.0,
        max_favorable: float = 0.0,
        max_adverse: float = 0.0,
    ) -> TradeHealthReport:
        """Assess health of a single trade."""
        now = datetime.now()
        days_held = (now - entry_date).days

        report = TradeHealthReport(
            pair_label=pair_label,
            entry_date=entry_date,
            days_held=days_held,
            entry_z=entry_z,
            current_z=current_z,
            half_life=half_life,
            entry_regime=entry_regime,
            current_regime=current_regime,
            unrealized_pnl_pct=unrealized_pnl_pct,
            max_favorable=max_favorable,
            max_adverse=max_adverse,
        )

        # ── 1. Z-score dimension ─────────────────────────────────
        if abs(entry_z) > 0.01:
            compression = 1.0 - abs(current_z) / abs(entry_z)
            report.z_compression_pct = max(0.0, compression)
            report.z_extension = abs(current_z) > abs(entry_z) * 1.1
        z_score_health = report.z_compression_pct

        # exit if z crossed target
        if abs(current_z) <= self.z_target:
            report.exit_signals.append(ExitSignal(
                signal_type=ExitType.PROFIT_TAKE,
                urgency=Urgency.END_OF_DAY,
                strength=0.9,
                action=f"Close {pair_label}: z={current_z:.2f} reached target",
                reason="Z-score mean-reverted to target",
            ))

        # exit if z blew through stop
        if abs(current_z) >= self.z_stop:
            report.exit_signals.append(ExitSignal(
                signal_type=ExitType.STOP_LOSS,
                urgency=Urgency.IMMEDIATE,
                strength=1.0,
                action=f"STOP {pair_label}: z={current_z:.2f} exceeded stop={self.z_stop}",
                reason="Z-score stop loss triggered",
            ))

        # ── 2. Time dimension ────────────────────────────────────
        if half_life > 0:
            report.half_lives_elapsed = days_held / half_life
            # decay: starts at 1.0, reaches 0 at max_half_lives
            report.time_decay_score = max(
                0.0, 1.0 - report.half_lives_elapsed / self.max_half_lives
            )
        time_health = report.time_decay_score

        if report.half_lives_elapsed > 2.5 and report.z_compression_pct < 0.5:
            report.exit_signals.append(ExitSignal(
                signal_type=ExitType.TIME_EXIT,
                urgency=Urgency.NEXT_SESSION,
                strength=0.7,
                action=f"Time exit {pair_label}: {report.half_lives_elapsed:.1f} HL, {report.z_compression_pct:.0%} compression",
                reason="Exceeded 2.5 half-lives with insufficient convergence",
            ))

        # ── 3. Regime dimension ──────────────────────────────────
        regime_order = {"CALM": 0, "NORMAL": 1, "TENSION": 2, "CRISIS": 3}
        entry_level = regime_order.get(entry_regime.upper(), 1)
        current_level = regime_order.get(current_regime.upper(), 1)
        report.regime_deteriorated = current_level > entry_level

        regime_health = 1.0
        if current_regime.upper() == "CRISIS":
            regime_health = 0.1
            report.exit_signals.append(ExitSignal(
                signal_type=ExitType.REGIME_EXIT,
                urgency=Urgency.IMMEDIATE,
                strength=1.0,
                action=f"REGIME EXIT {pair_label}: crisis detected",
                reason="Market regime shifted to CRISIS",
            ))
        elif current_regime.upper() == "TENSION":
            regime_health = 0.4
            if report.regime_deteriorated:
                report.exit_signals.append(ExitSignal(
                    signal_type=ExitType.REGIME_EXIT,
                    urgency=Urgency.END_OF_DAY,
                    strength=0.6,
                    action=f"Regime deterioration {pair_label}: {entry_regime} → {current_regime}",
                    reason="Regime deteriorated since entry",
                ))
        elif report.regime_deteriorated:
            regime_health = 0.7

        # ── 4. P&L dimension ─────────────────────────────────────
        pnl = unrealized_pnl_pct
        report.pnl_score = max(0.0, min(1.0, (pnl - self.stop_loss_pct) / (self.profit_take_pct - self.stop_loss_pct)))

        if pnl >= self.profit_take_pct:
            report.exit_signals.append(ExitSignal(
                signal_type=ExitType.PROFIT_TAKE,
                urgency=Urgency.END_OF_DAY,
                strength=0.8,
                action=f"Profit take {pair_label}: {pnl:.1%}",
                reason="Profit target reached",
            ))

        if pnl <= self.stop_loss_pct:
            report.exit_signals.append(ExitSignal(
                signal_type=ExitType.STOP_LOSS,
                urgency=Urgency.IMMEDIATE,
                strength=1.0,
                action=f"STOP LOSS {pair_label}: {pnl:.1%}",
                reason="Stop loss triggered",
            ))

        pnl_health = report.pnl_score

        # ── composite health score ────────────────────────────────
        report.health_score = (
            self.W_Z * z_score_health
            + self.W_TIME * time_health
            + self.W_REGIME * regime_health
            + self.W_PNL * pnl_health
        )

        # status
        if any(s.urgency == Urgency.IMMEDIATE for s in report.exit_signals):
            report.status = "FORCE_EXIT"
        elif report.exit_signals:
            report.status = "EXIT_RECOMMENDED"
        elif report.health_score < 0.3:
            report.status = "EXIT_RECOMMENDED"
        else:
            report.status = "ACTIVE"

        return report

    def monitor_portfolio(
        self,
        trades: list[dict],
    ) -> PortfolioMonitorSummary:
        """Monitor all active trades in the portfolio.

        Each trade dict should have keys:
        pair_label, entry_z, current_z, half_life, entry_date,
        current_regime, entry_regime, unrealized_pnl_pct,
        max_favorable, max_adverse
        """
        summary = PortfolioMonitorSummary()
        summary.total_pairs = len(trades)

        for t in trades:
            try:
                report = self.assess_trade(**t)
                summary.reports.append(report)

                if report.health_score >= 0.7:
                    summary.healthy_pairs += 1
                elif report.health_score >= 0.4:
                    summary.warning_pairs += 1
                else:
                    summary.critical_pairs += 1

                if report.status in ("EXIT_RECOMMENDED", "FORCE_EXIT"):
                    summary.exit_recommended += 1

            except Exception as e:
                logger.warning("Failed to monitor trade %s: %s", t.get("pair_label"), e)

        if summary.reports:
            summary.avg_health = np.mean([r.health_score for r in summary.reports])

        return summary
