# -*- coding: utf-8 -*-
"""
core/signal_analytics.py — Signal and Lifecycle Analytics
==========================================================

Structured analytics for the signal engine, lifecycle state machine,
and trade outcomes. These analytics answer the hard operational questions:

  - Which signals work in which regimes?
  - Which exits protect capital best?
  - Which thresholds are too tight or too loose?
  - Where is the system overtrading?
  - Where are trades being entered too late?
  - Which lifecycle states are becoming traps?

Three analytics classes:
  SignalAnalytics   — Signal count, block rate, quality distribution
  LifecycleAnalytics — State occupancy, transition rates, timeout rates
  ExitAnalytics     — Exit reason breakdown, P&L by exit type, MAE/MFE

All analytics are computed from lists of typed objects (SignalDecision,
LifecycleDiagnostics, ExitDiagnostics). No raw data dependency.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from core.contracts import (
    ExitReason,
    IntentAction,
    RegimeLabel,
    SignalQualityGrade,
    TradeLifecycleState,
)
from core.diagnostics import ExitDiagnostics, LifecycleDiagnostics
from core.intents import SignalDecision


# ══════════════════════════════════════════════════════════════════
# SIGNAL ANALYTICS
# ══════════════════════════════════════════════════════════════════

@dataclass
class SignalAnalyticsReport:
    """Summary of signal engine performance over a batch of decisions."""
    n_signals_total: int = 0
    n_entry_proposals: int = 0
    n_blocked: int = 0
    n_skipped_by_quality: int = 0
    block_rate: float = 0.0

    # Quality distribution
    grade_counts: dict[str, int] = field(default_factory=dict)
    avg_conviction: float = np.nan
    avg_quality_score: float = np.nan

    # By regime
    proposals_by_regime: dict[str, int] = field(default_factory=dict)
    block_rate_by_regime: dict[str, float] = field(default_factory=dict)

    # Block reason breakdown
    block_reasons: dict[str, int] = field(default_factory=dict)

    # Action breakdown
    action_counts: dict[str, int] = field(default_factory=dict)

    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "n_signals_total": self.n_signals_total,
            "n_entry_proposals": self.n_entry_proposals,
            "n_blocked": self.n_blocked,
            "block_rate": round(self.block_rate, 4),
            "avg_conviction": _safe(self.avg_conviction),
            "avg_quality_score": _safe(self.avg_quality_score),
            "grade_counts": self.grade_counts,
            "action_counts": self.action_counts,
            "proposals_by_regime": self.proposals_by_regime,
            "block_reasons": self.block_reasons,
            "computed_at": self.computed_at.isoformat(),
        }


class SignalAnalytics:
    """Compute signal analytics from a batch of SignalDecision objects."""

    @staticmethod
    def compute(decisions: list[SignalDecision]) -> SignalAnalyticsReport:
        if not decisions:
            return SignalAnalyticsReport()

        report = SignalAnalyticsReport(n_signals_total=len(decisions))

        convictions: list[float] = []
        qualities: list[float] = []
        grade_counts: Counter = Counter()
        action_counts: Counter = Counter()
        block_reasons: Counter = Counter()
        proposals_by_regime: Counter = Counter()
        blocked_by_regime: Counter = Counter()
        total_by_regime: Counter = Counter()

        for d in decisions:
            action_counts[d.action.value] += 1
            total_by_regime[d.regime_label.value] += 1

            if d.is_entry_proposal:
                report.n_entry_proposals += 1
                proposals_by_regime[d.regime_label.value] += 1

            if d.is_blocked:
                report.n_blocked += 1
                blocked_by_regime[d.regime_label.value] += 1
                for br in d.intent.block_reasons:
                    block_reasons[br.value] += 1

            if d.quality_grade == SignalQualityGrade.F:
                report.n_skipped_by_quality += 1

            grade_counts[d.quality_grade.value] += 1

            if not math.isnan(d.conviction):
                convictions.append(d.conviction)
            if not math.isnan(d.quality_score):
                qualities.append(d.quality_score)

        report.block_rate = report.n_blocked / max(report.n_signals_total, 1)
        report.grade_counts = dict(grade_counts)
        report.action_counts = dict(action_counts)
        report.block_reasons = dict(block_reasons)
        report.proposals_by_regime = dict(proposals_by_regime)
        report.block_rate_by_regime = {
            reg: round(blocked_by_regime.get(reg, 0) / max(total, 1), 4)
            for reg, total in total_by_regime.items()
        }

        if convictions:
            report.avg_conviction = float(np.mean(convictions))
        if qualities:
            report.avg_quality_score = float(np.mean(qualities))

        return report


# ══════════════════════════════════════════════════════════════════
# LIFECYCLE ANALYTICS
# ══════════════════════════════════════════════════════════════════

@dataclass
class LifecycleAnalyticsReport:
    """Summary of lifecycle state machine performance."""
    state_occupancy: dict[str, int] = field(default_factory=dict)
    avg_time_in_state: dict[str, float] = field(default_factory=dict)

    n_total_transitions: int = 0
    timeout_transitions: int = 0
    timeout_rate: float = 0.0

    n_in_cooldown: int = 0
    avg_cooldown_days: float = np.nan

    n_suspended: int = 0
    n_retired: int = 0
    avg_suspension_count: float = np.nan

    n_entry_attempts: int = 0
    n_entry_failures: int = 0
    entry_failure_rate: float = 0.0

    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "state_occupancy": self.state_occupancy,
            "avg_time_in_state_days": {k: round(v, 2) for k, v in self.avg_time_in_state.items()},
            "n_total_transitions": self.n_total_transitions,
            "timeout_rate": round(self.timeout_rate, 4),
            "n_in_cooldown": self.n_in_cooldown,
            "n_suspended": self.n_suspended,
            "n_retired": self.n_retired,
            "entry_failure_rate": round(self.entry_failure_rate, 4),
            "computed_at": self.computed_at.isoformat(),
        }


class LifecycleAnalytics:
    """Compute lifecycle analytics from LifecycleDiagnostics objects."""

    @staticmethod
    def compute(diagnostics: list[LifecycleDiagnostics]) -> LifecycleAnalyticsReport:
        if not diagnostics:
            return LifecycleAnalyticsReport()

        report = LifecycleAnalyticsReport()
        state_counts: Counter = Counter()
        state_times: dict[str, list[float]] = defaultdict(list)
        suspension_counts: list[int] = []
        cooldown_days: list[float] = []
        entry_attempts = entry_failures = total_transitions = timeout_count = 0

        for diag in diagnostics:
            state_counts[diag.current_state.value] += 1
            state_times[diag.current_state.value].append(diag.time_in_current_state_days)
            total_transitions += len(diag.transitions)
            timeout_count += sum(1 for t in diag.transitions if t.trigger == "timeout")

            if diag.is_in_cooldown:
                report.n_in_cooldown += 1
                cooldown_days.append(diag.cooldown_days_remaining)
            if diag.current_state == TradeLifecycleState.SUSPENDED:
                report.n_suspended += 1
            if diag.current_state == TradeLifecycleState.RETIRED:
                report.n_retired += 1

            suspension_counts.append(diag.suspension_count)
            entry_attempts += diag.total_trades + diag.failed_entry_attempts
            entry_failures += diag.failed_entry_attempts

        report.state_occupancy = dict(state_counts)
        report.avg_time_in_state = {
            s: round(float(np.mean(ts)), 2) for s, ts in state_times.items()
        }
        report.n_total_transitions = total_transitions
        report.timeout_transitions = timeout_count
        report.timeout_rate = timeout_count / max(total_transitions, 1)
        report.n_entry_attempts = entry_attempts
        report.n_entry_failures = entry_failures
        report.entry_failure_rate = entry_failures / max(entry_attempts, 1)
        if suspension_counts:
            report.avg_suspension_count = float(np.mean(suspension_counts))
        if cooldown_days:
            report.avg_cooldown_days = float(np.mean(cooldown_days))

        return report


# ══════════════════════════════════════════════════════════════════
# EXIT ANALYTICS
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExitAnalyticsReport:
    """Summary of exit quality and patterns."""
    n_exits_total: int = 0
    n_converged: int = 0
    n_stopped: int = 0
    convergence_rate: float = 0.0

    avg_pnl_pct_by_reason: dict[str, float] = field(default_factory=dict)
    win_rate_by_reason: dict[str, float] = field(default_factory=dict)
    exit_reason_counts: dict[str, int] = field(default_factory=dict)

    avg_holding_days: float = np.nan
    avg_holding_days_by_reason: dict[str, float] = field(default_factory=dict)

    avg_mae: float = np.nan
    avg_mfe: float = np.nan

    convergence_rate_by_entry_regime: dict[str, float] = field(default_factory=dict)
    convergence_rate_by_quality: dict[str, float] = field(default_factory=dict)
    avg_pnl_by_holding_bucket: dict[str, float] = field(default_factory=dict)

    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "n_exits_total": self.n_exits_total,
            "convergence_rate": round(self.convergence_rate, 4),
            "n_converged": self.n_converged,
            "n_stopped": self.n_stopped,
            "avg_holding_days": _safe(self.avg_holding_days),
            "avg_mae": _safe(self.avg_mae),
            "avg_mfe": _safe(self.avg_mfe),
            "exit_reason_counts": self.exit_reason_counts,
            "avg_pnl_pct_by_reason": {k: round(v, 4) for k, v in self.avg_pnl_pct_by_reason.items()},
            "convergence_by_regime": self.convergence_rate_by_entry_regime,
            "convergence_by_quality": self.convergence_rate_by_quality,
            "computed_at": self.computed_at.isoformat(),
        }


class ExitAnalytics:
    """Compute exit analytics from ExitDiagnostics objects."""

    @staticmethod
    def compute(exits: list[ExitDiagnostics]) -> ExitAnalyticsReport:
        if not exits:
            return ExitAnalyticsReport()

        report = ExitAnalyticsReport(n_exits_total=len(exits))
        reason_pnls: dict[str, list[float]] = defaultdict(list)
        reason_holding: dict[str, list[float]] = defaultdict(list)
        reason_counts: Counter = Counter()
        mae_vals, mfe_vals, holding_vals = [], [], []
        regime_convgd: dict[str, list[bool]] = defaultdict(list)
        quality_convgd: dict[str, list[bool]] = defaultdict(list)

        for ex in exits:
            if ex.converged:
                report.n_converged += 1
            if ex.was_stopped:
                report.n_stopped += 1

            for reason in ex.exit_reasons:
                reason_counts[reason.value] += 1
                if not math.isnan(ex.realized_pnl_pct):
                    reason_pnls[reason.value].append(ex.realized_pnl_pct)
                reason_holding[reason.value].append(float(ex.holding_days))

            if not math.isnan(ex.max_adverse_excursion):
                mae_vals.append(ex.max_adverse_excursion)
            if not math.isnan(ex.max_favorable_excursion):
                mfe_vals.append(ex.max_favorable_excursion)
            if ex.holding_days > 0:
                holding_vals.append(float(ex.holding_days))

            regime_convgd[ex.regime_at_entry.value].append(ex.converged)
            quality_convgd[ex.entry_quality_grade.value].append(ex.converged)

        report.convergence_rate = report.n_converged / max(report.n_exits_total, 1)
        report.exit_reason_counts = dict(reason_counts)
        report.avg_pnl_pct_by_reason = {
            r: round(float(np.mean(v)), 4) for r, v in reason_pnls.items() if v
        }
        report.win_rate_by_reason = {
            r: round(sum(1 for x in v if x > 0) / len(v), 4)
            for r, v in reason_pnls.items() if v
        }
        report.avg_holding_days_by_reason = {
            r: round(float(np.mean(v)), 2) for r, v in reason_holding.items() if v
        }
        if mae_vals:
            report.avg_mae = float(np.mean(mae_vals))
        if mfe_vals:
            report.avg_mfe = float(np.mean(mfe_vals))
        if holding_vals:
            report.avg_holding_days = float(np.mean(holding_vals))

        report.convergence_rate_by_entry_regime = {
            r: round(sum(v) / len(v), 4) for r, v in regime_convgd.items()
        }
        report.convergence_rate_by_quality = {
            g: round(sum(v) / len(v), 4) for g, v in quality_convgd.items()
        }

        # Holding-time buckets
        for label, lo, hi in [("0-5d", 0, 5), ("5-15d", 5, 15), ("15-30d", 15, 30), (">30d", 30, 9999)]:
            bucket = [
                ex.realized_pnl_pct for ex in exits
                if lo <= ex.holding_days < hi and not math.isnan(ex.realized_pnl_pct)
            ]
            if bucket:
                report.avg_pnl_by_holding_bucket[label] = round(float(np.mean(bucket)), 4)

        return report


# ══════════════════════════════════════════════════════════════════
# COMBINED PLATFORM ANALYTICS
# ══════════════════════════════════════════════════════════════════

@dataclass
class PlatformAnalyticsReport:
    """Combined analytics across signal, lifecycle, and exit layers."""
    signal: Optional[SignalAnalyticsReport] = None
    lifecycle: Optional[LifecycleAnalyticsReport] = None
    exits: Optional[ExitAnalyticsReport] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "signal": self.signal.to_dict() if self.signal else None,
            "lifecycle": self.lifecycle.to_dict() if self.lifecycle else None,
            "exits": self.exits.to_dict() if self.exits else None,
            "generated_at": self.generated_at.isoformat(),
        }

    def print_summary(self) -> None:
        print("=" * 60)
        print("PLATFORM ANALYTICS SUMMARY")
        print("=" * 60)
        if self.signal:
            s = self.signal
            print(f"\nSIGNALS:")
            print(f"  Total: {s.n_signals_total} | Entries: {s.n_entry_proposals} | "
                  f"Blocked: {s.n_blocked} ({s.block_rate:.0%})")
            if s.grade_counts:
                grades = " | ".join(f"{g}:{n}" for g, n in sorted(s.grade_counts.items()))
                print(f"  Grades: {grades}")
        if self.lifecycle:
            lc = self.lifecycle
            print(f"\nLIFECYCLE:")
            occ = " | ".join(f"{s}:{n}" for s, n in sorted(lc.state_occupancy.items()))
            print(f"  Occupancy: {occ}")
            print(f"  Entry fail rate: {lc.entry_failure_rate:.0%}  "
                  f"Timeout rate: {lc.timeout_rate:.0%}")
        if self.exits:
            ex = self.exits
            print(f"\nEXITS:")
            print(f"  Total: {ex.n_exits_total} | Converged: {ex.n_converged} "
                  f"({ex.convergence_rate:.0%}) | Stopped: {ex.n_stopped}")
        print("=" * 60)


def _safe(v: float) -> Optional[float]:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), 4)


__all__ = [
    "SignalAnalyticsReport", "SignalAnalytics",
    "LifecycleAnalyticsReport", "LifecycleAnalytics",
    "ExitAnalyticsReport", "ExitAnalytics",
    "PlatformAnalyticsReport",
]
