# -*- coding: utf-8 -*-
"""
core/position_reconciliation.py — Startup Position Reconciliation
=================================================================

PositionReconciliationEngine reconciles positions in the local database
against positions reported by the broker on every system startup.

This MUST run before any new trading activity is accepted.

Any discrepancy between local state and broker state must be investigated
and resolved before the system is trusted with new order submissions.

Design
------
- Runs once at startup (called from orchestrator.__init__ or startup hook)
- Compares broker positions with last known positions from sql_store
- Produces a ReconciliationResult with any discrepancies
- On critical discrepancy: raises StartupReconciliationError (pauses new trading)
- On minor discrepancy: logs WARNING and continues
- On clean reconciliation: logs INFO and proceeds

Usage
-----
    from core.position_reconciliation import PositionReconciliationEngine

    engine = PositionReconciliationEngine(router=ib_router, store=sql_store)
    result = engine.reconcile()  # Run on every startup

    if not result.clean:
        # Handle discrepancies before accepting new intents
        logger.error("Reconciliation failed: %s", result.discrepancies)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("core.position_reconciliation")


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class PositionRecord:
    """A position as known by one source (broker or local database)."""
    symbol: str
    quantity: float           # Positive = long, negative = short
    avg_cost: float = 0.0
    currency: str = "USD"
    sec_type: str = "STK"
    source: str = "unknown"   # "broker" | "local_db"


@dataclass
class PositionDiscrepancy:
    """A mismatch between broker and local-db position for one symbol."""
    symbol: str
    broker_qty: float
    local_qty: float
    delta: float              # broker_qty - local_qty
    severity: str             # "critical" | "warning" | "info"
    description: str = ""

    def __post_init__(self):
        if abs(self.delta) > 10:
            self.severity = "critical"
        elif abs(self.delta) > 1:
            self.severity = "warning"
        else:
            self.severity = "info"


@dataclass
class ReconciliationResult:
    """Output of a single reconciliation run."""
    reconciled_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    clean: bool = False                          # True = no discrepancies
    broker_positions: List[PositionRecord] = field(default_factory=list)
    local_positions: List[PositionRecord] = field(default_factory=list)
    discrepancies: List[PositionDiscrepancy] = field(default_factory=list)
    critical_count: int = 0
    warning_count: int = 0
    error_message: str = ""                      # Set if reconciliation itself failed

    def has_critical(self) -> bool:
        return self.critical_count > 0

    def summary(self) -> str:
        if self.error_message:
            return f"RECONCILIATION ERROR: {self.error_message}"
        if self.clean:
            return (
                f"Reconciliation clean: {len(self.broker_positions)} broker positions, "
                f"{len(self.local_positions)} local positions, 0 discrepancies"
            )
        return (
            f"Reconciliation FAILED: {len(self.discrepancies)} discrepancies "
            f"({self.critical_count} critical, {self.warning_count} warnings)"
        )


class StartupReconciliationError(Exception):
    """
    Raised when startup reconciliation finds critical discrepancies.

    When this error is raised, the system must NOT accept new order submissions
    until a human has reviewed and resolved the position discrepancies.
    """
    pass


# ---------------------------------------------------------------------------
# Reconciliation Engine
# ---------------------------------------------------------------------------

class PositionReconciliationEngine:
    """
    Reconciles broker positions against local database on startup.

    Parameters
    ----------
    router : IBOrderRouter, optional
        Active order router for broker position queries.
        If None, broker reconciliation is skipped (local-only mode).
    store : SQLStore, optional
        Local database for last-known positions.
        If None, local position lookup is skipped.
    raise_on_critical : bool
        If True (default), raises StartupReconciliationError when critical
        discrepancies are found. If False, only logs and returns result.
    tolerance_shares : float
        Discrepancies smaller than this (absolute shares) are treated as INFO
        rather than WARNING (rounding tolerance). Default: 0.0.
    """

    def __init__(
        self,
        router=None,
        store=None,
        raise_on_critical: bool = True,
        tolerance_shares: float = 0.0,
    ):
        self._router = router
        self._store = store
        self._raise_on_critical = raise_on_critical
        self._tolerance = tolerance_shares

    def reconcile(self) -> ReconciliationResult:
        """
        Run startup reconciliation. Call once before accepting new intents.

        Returns a ReconciliationResult. If raise_on_critical=True and critical
        discrepancies are found, raises StartupReconciliationError.
        """
        result = ReconciliationResult()

        try:
            broker_positions = self._get_broker_positions()
            local_positions = self._get_local_positions()

            result.broker_positions = broker_positions
            result.local_positions = local_positions

            discrepancies = self._compare(broker_positions, local_positions)
            result.discrepancies = discrepancies
            result.critical_count = sum(1 for d in discrepancies if d.severity == "critical")
            result.warning_count = sum(1 for d in discrepancies if d.severity == "warning")
            result.clean = len(discrepancies) == 0

            if result.clean:
                logger.info(result.summary())
            elif result.has_critical():
                logger.error(result.summary())
                for d in discrepancies:
                    logger.error(
                        "POSITION DISCREPANCY: symbol=%s broker=%.2f local=%.2f delta=%.2f [%s]",
                        d.symbol, d.broker_qty, d.local_qty, d.delta, d.severity,
                    )
                if self._raise_on_critical:
                    raise StartupReconciliationError(
                        f"Critical position discrepancies found on startup. "
                        f"Manual review required before accepting new orders. "
                        f"Discrepancies: {[d.symbol for d in discrepancies if d.severity == 'critical']}"
                    )
            else:
                logger.warning(result.summary())

        except StartupReconciliationError:
            raise
        except Exception as exc:
            logger.error("Position reconciliation failed: %s", exc)
            result.error_message = str(exc)
            result.clean = False

        return result

    def _get_broker_positions(self) -> List[PositionRecord]:
        """Fetch positions from the broker."""
        if self._router is None:
            logger.info("No IB router available — skipping broker position check")
            return []
        try:
            raw = self._router.sync_positions_from_ib()
            positions = []
            for row in raw:
                qty = float(row.get("position") or 0.0)
                if abs(qty) < 1e-8:
                    continue
                positions.append(PositionRecord(
                    symbol=str(row.get("symbol") or ""),
                    quantity=qty,
                    avg_cost=float(row.get("avgCost") or 0.0),
                    currency=str(row.get("currency") or "USD"),
                    sec_type=str(row.get("secType") or "STK"),
                    source="broker",
                ))
            return positions
        except Exception as exc:
            logger.warning("_get_broker_positions failed: %s", exc)
            return []

    def _get_local_positions(self) -> List[PositionRecord]:
        """Fetch last-known positions from the local database."""
        if self._store is None:
            logger.info("No sql_store available — skipping local position check")
            return []
        try:
            # Try to load most recent position snapshot
            if hasattr(self._store, "load_position_snapshot"):
                df = self._store.load_position_snapshot()
            elif hasattr(self._store, "raw_query"):
                import pandas as pd
                df = self._store.raw_query(
                    "SELECT symbol, quantity, avg_cost, currency, sec_type "
                    "FROM position_snapshots "
                    "WHERE ts_utc = (SELECT MAX(ts_utc) FROM position_snapshots)"
                )
            else:
                return []

            if df is None or (hasattr(df, "empty") and df.empty):
                return []

            positions = []
            for _, row in df.iterrows():
                qty = float(row.get("quantity", 0) or 0)
                if abs(qty) < 1e-8:
                    continue
                positions.append(PositionRecord(
                    symbol=str(row.get("symbol", "")),
                    quantity=qty,
                    avg_cost=float(row.get("avg_cost", 0) or 0),
                    currency=str(row.get("currency", "USD")),
                    sec_type=str(row.get("sec_type", "STK")),
                    source="local_db",
                ))
            return positions
        except Exception as exc:
            logger.warning("_get_local_positions failed: %s", exc)
            return []

    def _compare(
        self,
        broker: List[PositionRecord],
        local: List[PositionRecord],
    ) -> List[PositionDiscrepancy]:
        """Compare broker and local positions, return discrepancies."""
        broker_map: Dict[str, float] = {p.symbol.upper(): p.quantity for p in broker}
        local_map: Dict[str, float] = {p.symbol.upper(): p.quantity for p in local}

        all_symbols = set(broker_map) | set(local_map)
        discrepancies = []

        for symbol in sorted(all_symbols):
            b_qty = broker_map.get(symbol, 0.0)
            l_qty = local_map.get(symbol, 0.0)
            delta = b_qty - l_qty

            if abs(delta) <= self._tolerance:
                continue

            desc_parts = []
            if symbol not in broker_map:
                desc_parts.append("position in local_db not found at broker")
            elif symbol not in local_map:
                desc_parts.append("position at broker not in local_db")
            else:
                desc_parts.append(f"quantity mismatch: broker={b_qty:.2f}, local={l_qty:.2f}")

            discrepancies.append(PositionDiscrepancy(
                symbol=symbol,
                broker_qty=b_qty,
                local_qty=l_qty,
                delta=delta,
                severity="warning",  # __post_init__ will override based on magnitude
                description="; ".join(desc_parts),
            ))

        return discrepancies
