# -*- coding: utf-8 -*-
"""
reconciliation/contracts.py — Reconciliation Domain Contracts
==============================================================

All typed domain objects for the position, order, fill, and cash
reconciliation subsystem.

Design principles:
  - ReconcileDiffRecord is frozen (immutable after detection).
  - ReconciliationReport is mutable so the engine can add diffs
    incrementally before returning the final report.
  - EndOfDayReport is frozen (published once; never mutated).
  - Enums inherit from str for JSON round-trip compatibility.
  - stdlib only — no external dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS
# ══════════════════════════════════════════════════════════════════


class ReconciliationStatus(str, enum.Enum):
    """Overall status of a reconciliation run."""

    CLEAN = "clean"
    MISMATCH = "mismatch"
    PENDING = "pending"
    FAILED = "failed"
    SKIPPED = "skipped"


class DiffType(str, enum.Enum):
    """Category of a detected discrepancy."""

    POSITION_MISMATCH = "position_mismatch"
    CASH_MISMATCH = "cash_mismatch"
    ORPHAN_POSITION = "orphan_position"
    MISSING_FILL = "missing_fill"
    DUPLICATE_ORDER = "duplicate_order"
    UNEXPECTED_OPEN = "unexpected_open"
    LEG_IMBALANCE = "leg_imbalance"
    HEDGE_RATIO_DRIFT = "hedge_ratio_drift"
    RESIDUAL_EXPOSURE = "residual_exposure"
    EXPOSURE_MISMATCH = "exposure_mismatch"


# ══════════════════════════════════════════════════════════════════
# 2. DIFF RECORDS (IMMUTABLE)
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReconcileDiffRecord:
    """A single detected discrepancy between internal and external state.

    Fields
    ------
    diff_id : str
        UUID for this diff record.
    diff_type : DiffType
        Category of the discrepancy.
    scope : str
        Affected scope: "global" | "strategy:{id}" | "symbol:{sym}".
    internal_value : str
        What the system believes (stringified for portability).
    external_value : str
        What the broker/clearing house reports.
    discrepancy : str
        Human-readable description of the discrepancy magnitude/nature.
    severity : str
        "info" | "warning" | "critical".
    detected_at : str
        ISO-8601 timestamp when the diff was detected.
    resolved : bool
        True if the diff has been resolved since detection.
    resolution_notes : str
        Free-text notes on how the diff was resolved.
    """

    diff_id: str
    diff_type: DiffType
    scope: str
    internal_value: str
    external_value: str
    discrepancy: str
    severity: str
    detected_at: str
    resolved: bool = False
    resolution_notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 3. RECONCILIATION REPORT (MUTABLE)
# ══════════════════════════════════════════════════════════════════


@dataclass
class ReconciliationReport:
    """Result of a full reconciliation sweep.

    Mutable so the engine can append diffs incrementally before
    returning the final report.

    Fields
    ------
    report_id : str
        UUID for this report.
    env : str
        Deployment environment.
    generated_at : str
        ISO-8601 timestamp when the report was generated.
    status : ReconciliationStatus
        Overall reconciliation outcome.
    diffs : list[ReconcileDiffRecord]
        All detected discrepancies.
    positions_checked : int
        Number of position records compared.
    orders_checked : int
        Number of order records compared.
    fills_checked : int
        Number of fill records compared.
    critical_diffs : int
        Number of diffs with severity == "critical".
    warning_diffs : int
        Number of diffs with severity == "warning".
    blocking_live_resume : bool
        True if any critical unresolved diff prevents live trading resumption.
    auto_resolved : int
        Number of diffs automatically resolved during this run.
    notes : str
        Free-text operator notes.
    """

    report_id: str
    env: str
    generated_at: str
    status: ReconciliationStatus
    diffs: List[ReconcileDiffRecord]
    positions_checked: int
    orders_checked: int
    fills_checked: int
    critical_diffs: int
    warning_diffs: int
    blocking_live_resume: bool
    auto_resolved: int
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 4. END-OF-DAY REPORT (IMMUTABLE)
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EndOfDayReport:
    """End-of-day operational and reconciliation summary.

    Produced once per trading session and never mutated after creation.

    Fields
    ------
    report_id : str
        UUID for this report.
    date : str
        Trading date in YYYY-MM-DD format.
    env : str
        Deployment environment.
    generated_at : str
        ISO-8601 timestamp.
    total_trades : int
        Number of completed round-trip trades.
    open_positions : int
        Number of open positions at end of day.
    gross_pnl : float
        Gross realised + unrealised PnL (USD).
    net_pnl : float
        Net PnL after commissions and fees.
    gross_exposure : float
        Sum of absolute position notional values.
    net_exposure : float
        Net directional notional exposure.
    max_drawdown_pct : float
        Maximum intra-day drawdown as a fraction [0, 1].
    signals_generated : int
        Total signals produced during the session.
    orders_submitted : int
        Total orders sent to the broker.
    orders_filled : int
        Total orders confirmed filled.
    orders_rejected : int
        Total orders rejected.
    reconciliation_status : ReconciliationStatus
        Outcome of end-of-day reconciliation.
    reconciliation_diffs : int
        Total diff records detected.
    incidents_today : int
        Incidents opened during this session.
    alerts_today : int
        Total alert events fired.
    model_versions_in_use : dict
        Mapping of model_name -> version string.
    config_version : str
        Active configuration version.
    notes : str
        Free-text operator notes.
    """

    report_id: str
    date: str
    env: str
    generated_at: str
    total_trades: int
    open_positions: int
    gross_pnl: float
    net_pnl: float
    gross_exposure: float
    net_exposure: float
    max_drawdown_pct: float
    signals_generated: int
    orders_submitted: int
    orders_filled: int
    orders_rejected: int
    reconciliation_status: ReconciliationStatus
    reconciliation_diffs: int
    incidents_today: int
    alerts_today: int
    model_versions_in_use: Dict[str, str]
    config_version: str
    notes: str = ""
