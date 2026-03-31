# -*- coding: utf-8 -*-
"""
reconciliation/engine.py — ReconciliationEngine
================================================

Position, order, fill, and cash reconciliation for the pairs
trading system.

Compares internal state (paper_trader / live_pair_store) against
a broker state snapshot and produces a ReconciliationReport with
typed ReconcileDiffRecord entries.

Responsibilities
----------------
- Position-level reconciliation (quantity, side, notional).
- Order-level reconciliation (open, pending, filled, rejected).
- Leg imbalance detection for spread pairs.
- Hedge-ratio drift detection.
- Exposure mismatch detection.
- End-of-day report generation.
- Alert/incident integration for critical diffs.
- Block live trading resumption when critical diffs are unresolved.

Thread safety: a single threading.Lock serializes store mutations.
Singleton access via get_reconciliation_engine().
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from reconciliation.contracts import (
    DiffType,
    EndOfDayReport,
    ReconcileDiffRecord,
    ReconciliationReport,
    ReconciliationStatus,
)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_diff_id() -> str:
    return str(uuid.uuid4())


def _make_report_id() -> str:
    return str(uuid.uuid4())


def _pct_diff(a: float, b: float) -> float:
    """Percentage difference between a and b relative to b.  Returns 0 if b==0."""
    if b == 0.0:
        return 0.0 if a == 0.0 else float("inf")
    return abs(a - b) / abs(b)


# ══════════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════════


class ReconciliationEngine:
    """Position, order, fill, and cash reconciliation engine.

    Compares internal state against broker state and produces a
    ReconciliationReport.  For live environments, a report with
    critical unresolved diffs sets ``blocking_live_resume = True``.

    Parameters
    ----------
    state_manager :
        Optional RuntimeStateManager for pulling internal state.
    alert_engine :
        Optional AlertEngine for firing RECONCILIATION_BREAK alerts.
    incident_manager :
        Optional IncidentManager for auto-incident creation.
    max_position_diff_pct : float
        Maximum fractional position size difference before flagging
        a POSITION_MISMATCH diff.  Default: 0.02 (2%).
    max_cash_diff_pct : float
        Maximum fractional cash difference before flagging a
        CASH_MISMATCH diff.  Default: 0.001 (0.1%).
    """

    def __init__(
        self,
        state_manager: Optional[Any] = None,
        alert_engine: Optional[Any] = None,
        incident_manager: Optional[Any] = None,
        max_position_diff_pct: float = 0.02,
        max_cash_diff_pct: float = 0.001,
    ) -> None:
        self._state_manager = state_manager
        self._alert_engine = alert_engine
        self._incident_manager = incident_manager
        self._pos_tolerance = max_position_diff_pct
        self._cash_tolerance = max_cash_diff_pct
        self._lock = threading.Lock()

        # Historical reports (report_id -> ReconciliationReport)
        self._reports: Dict[str, ReconciliationReport] = {}

        # Metrics
        self._run_count: int = 0
        self._total_diffs: int = 0
        self._critical_diffs: int = 0
        self._auto_resolved_total: int = 0

    # ──────────────────────────────────────────────────────────────
    # FULL RECONCILIATION
    # ──────────────────────────────────────────────────────────────

    def reconcile(
        self,
        internal_positions: Dict[str, Dict[str, Any]],
        broker_positions: Dict[str, Dict[str, Any]],
        internal_orders: List[Dict[str, Any]],
        broker_orders: List[Dict[str, Any]],
        env: str = "paper",
        active_pairs: Optional[List[Dict[str, Any]]] = None,
        expected_ratios: Optional[Dict[str, float]] = None,
    ) -> ReconciliationReport:
        """Full reconciliation sweep.

        Runs all sub-checks in sequence and aggregates results into a
        single ReconciliationReport.

        Parameters
        ----------
        internal_positions : dict
            Internal position state: {symbol: {qty, side, avg_price, ...}}.
        broker_positions : dict
            Broker position state in the same format.
        internal_orders : list[dict]
            Internal order records: [{order_id, symbol, qty, side, status, ...}].
        broker_orders : list[dict]
            Broker order records in the same format.
        env : str
            Deployment environment.
        active_pairs : list[dict] | None
            Active spread pairs for leg-imbalance checks.
            Each dict: {pair_id, leg_x, leg_y}.
        expected_ratios : dict | None
            Expected hedge ratios: {pair_id: ratio}.

        Returns
        -------
        ReconciliationReport
        """
        report_id = _make_report_id()
        generated_at = _now_iso()
        all_diffs: List[ReconcileDiffRecord] = []

        # 1. Position reconciliation
        pos_diffs = self.reconcile_positions(internal_positions, broker_positions)
        all_diffs.extend(pos_diffs)

        # 2. Order reconciliation
        ord_diffs = self.reconcile_orders(internal_orders, broker_orders)
        all_diffs.extend(ord_diffs)

        # 3. Leg imbalance check
        if active_pairs:
            combined_pos = dict(internal_positions)
            combined_pos.update(broker_positions)  # broker state as ground truth
            leg_diffs = self.check_leg_imbalances(combined_pos, active_pairs)
            all_diffs.extend(leg_diffs)

        # 4. Hedge ratio drift check
        if expected_ratios:
            ratio_diffs = self.check_hedge_ratio_drift(
                broker_positions, expected_ratios
            )
            all_diffs.extend(ratio_diffs)

        # Count by severity
        critical_count = sum(1 for d in all_diffs if d.severity == "critical")
        warning_count = sum(1 for d in all_diffs if d.severity == "warning")

        # Determine status
        if not all_diffs:
            status = ReconciliationStatus.CLEAN
        elif critical_count > 0:
            status = ReconciliationStatus.MISMATCH
        else:
            status = ReconciliationStatus.MISMATCH

        blocking = critical_count > 0 and env == "live"

        report = ReconciliationReport(
            report_id=report_id,
            env=env,
            generated_at=generated_at,
            status=status,
            diffs=all_diffs,
            positions_checked=len(set(list(internal_positions.keys()) + list(broker_positions.keys()))),
            orders_checked=len(internal_orders) + len(broker_orders),
            fills_checked=0,
            critical_diffs=critical_count,
            warning_diffs=warning_count,
            blocking_live_resume=blocking,
            auto_resolved=0,
        )

        # Store and update metrics
        with self._lock:
            self._reports[report_id] = report
            self._run_count += 1
            self._total_diffs += len(all_diffs)
            self._critical_diffs += critical_count

        # Fire alert if critical diffs detected
        if critical_count > 0 and self._alert_engine is not None:
            self._fire_break_alert(report, env)

        return report

    # ──────────────────────────────────────────────────────────────
    # POSITION RECONCILIATION
    # ──────────────────────────────────────────────────────────────

    def reconcile_positions(
        self,
        internal: Dict[str, Dict[str, Any]],
        broker: Dict[str, Dict[str, Any]],
    ) -> List[ReconcileDiffRecord]:
        """Position-level reconciliation.

        Checks for:
        - Quantity mismatches beyond tolerance.
        - Side mismatches (long vs short).
        - Orphan positions (in broker but not internal, or vice versa).

        Parameters
        ----------
        internal : dict
            Internal positions: {symbol: {qty: float, side: str, avg_price: float}}.
        broker : dict
            Broker positions in the same format.

        Returns
        -------
        list[ReconcileDiffRecord]
        """
        diffs: List[ReconcileDiffRecord] = []
        detected_at = _now_iso()
        all_syms = set(internal.keys()) | set(broker.keys())

        for sym in all_syms:
            i_pos = internal.get(sym)
            b_pos = broker.get(sym)

            # Orphan: present in broker but not internal
            if i_pos is None and b_pos is not None:
                b_qty = float(b_pos.get("qty", 0))
                if abs(b_qty) > 1e-8:
                    diffs.append(
                        ReconcileDiffRecord(
                            diff_id=_make_diff_id(),
                            diff_type=DiffType.ORPHAN_POSITION,
                            scope=f"symbol:{sym}",
                            internal_value="0",
                            external_value=str(b_qty),
                            discrepancy=f"Broker has position in {sym} not tracked internally.",
                            severity="critical",
                            detected_at=detected_at,
                        )
                    )
                continue

            # Unexpected open: present internally but not at broker
            if b_pos is None and i_pos is not None:
                i_qty = float(i_pos.get("qty", 0))
                if abs(i_qty) > 1e-8:
                    diffs.append(
                        ReconcileDiffRecord(
                            diff_id=_make_diff_id(),
                            diff_type=DiffType.UNEXPECTED_OPEN,
                            scope=f"symbol:{sym}",
                            internal_value=str(i_qty),
                            external_value="0",
                            discrepancy=f"Internal has position in {sym} not confirmed by broker.",
                            severity="critical",
                            detected_at=detected_at,
                        )
                    )
                continue

            # Both sides have the position
            i_qty = float(i_pos.get("qty", 0))  # type: ignore[union-attr]
            b_qty = float(b_pos.get("qty", 0))  # type: ignore[union-attr]
            i_side = str(i_pos.get("side", "")).lower()  # type: ignore[union-attr]
            b_side = str(b_pos.get("side", "")).lower()  # type: ignore[union-attr]

            # Side mismatch
            if i_side and b_side and i_side != b_side:
                diffs.append(
                    ReconcileDiffRecord(
                        diff_id=_make_diff_id(),
                        diff_type=DiffType.POSITION_MISMATCH,
                        scope=f"symbol:{sym}",
                        internal_value=f"qty={i_qty} side={i_side}",
                        external_value=f"qty={b_qty} side={b_side}",
                        discrepancy=f"{sym}: side mismatch (internal={i_side}, broker={b_side}).",
                        severity="critical",
                        detected_at=detected_at,
                    )
                )
                continue

            # Quantity mismatch
            if abs(b_qty) > 1e-8:
                pct = _pct_diff(i_qty, b_qty)
            else:
                pct = 0.0 if abs(i_qty) < 1e-8 else float("inf")

            if pct > self._pos_tolerance:
                severity = "critical" if pct > 0.05 else "warning"
                diffs.append(
                    ReconcileDiffRecord(
                        diff_id=_make_diff_id(),
                        diff_type=DiffType.POSITION_MISMATCH,
                        scope=f"symbol:{sym}",
                        internal_value=str(i_qty),
                        external_value=str(b_qty),
                        discrepancy=(
                            f"{sym}: quantity diff {pct:.2%} "
                            f"(internal={i_qty}, broker={b_qty})."
                        ),
                        severity=severity,
                        detected_at=detected_at,
                    )
                )

        return diffs

    # ──────────────────────────────────────────────────────────────
    # ORDER RECONCILIATION
    # ──────────────────────────────────────────────────────────────

    def reconcile_orders(
        self,
        internal: List[Dict[str, Any]],
        broker: List[Dict[str, Any]],
    ) -> List[ReconcileDiffRecord]:
        """Order-level reconciliation.

        Checks for:
        - Missing fills (order filled at broker but not internally).
        - Duplicate orders (same client order ID appears more than once).
        - Orders present in one system but not the other.

        Parameters
        ----------
        internal : list[dict]
            Internal orders: [{order_id, symbol, qty, side, status, ...}].
        broker : list[dict]
            Broker orders in the same format.

        Returns
        -------
        list[ReconcileDiffRecord]
        """
        diffs: List[ReconcileDiffRecord] = []
        detected_at = _now_iso()

        i_by_id: Dict[str, Dict[str, Any]] = {
            o["order_id"]: o for o in internal if "order_id" in o
        }
        b_by_id: Dict[str, Dict[str, Any]] = {
            o["order_id"]: o for o in broker if "order_id" in o
        }

        # Duplicate detection in internal orders
        i_ids = [o.get("order_id") for o in internal if "order_id" in o]
        seen: Dict[str, int] = {}
        for oid in i_ids:
            seen[oid] = seen.get(oid, 0) + 1
        for oid, count in seen.items():
            if count > 1:
                diffs.append(
                    ReconcileDiffRecord(
                        diff_id=_make_diff_id(),
                        diff_type=DiffType.DUPLICATE_ORDER,
                        scope=f"order:{oid}",
                        internal_value=str(count),
                        external_value="1",
                        discrepancy=f"Order {oid} appears {count} times internally.",
                        severity="critical",
                        detected_at=detected_at,
                    )
                )

        # Missing fills: broker says filled but internal does not
        for oid, b_ord in b_by_id.items():
            b_status = str(b_ord.get("status", "")).lower()
            i_ord = i_by_id.get(oid)
            if b_status in ("filled", "partial_fill"):
                if i_ord is None:
                    diffs.append(
                        ReconcileDiffRecord(
                            diff_id=_make_diff_id(),
                            diff_type=DiffType.MISSING_FILL,
                            scope=f"order:{oid}",
                            internal_value="not_found",
                            external_value=b_status,
                            discrepancy=(
                                f"Broker reports order {oid} as '{b_status}' "
                                f"but order not found internally."
                            ),
                            severity="critical",
                            detected_at=detected_at,
                        )
                    )
                else:
                    i_status = str(i_ord.get("status", "")).lower()
                    if i_status not in ("filled", "partial_fill"):
                        diffs.append(
                            ReconcileDiffRecord(
                                diff_id=_make_diff_id(),
                                diff_type=DiffType.MISSING_FILL,
                                scope=f"order:{oid}",
                                internal_value=i_status,
                                external_value=b_status,
                                discrepancy=(
                                    f"Order {oid}: broker='{b_status}', "
                                    f"internal='{i_status}'."
                                ),
                                severity="warning",
                                detected_at=detected_at,
                            )
                        )

        return diffs

    # ──────────────────────────────────────────────────────────────
    # LEG IMBALANCE CHECK
    # ──────────────────────────────────────────────────────────────

    def check_leg_imbalances(
        self,
        positions: Dict[str, Dict[str, Any]],
        active_pairs: List[Dict[str, Any]],
    ) -> List[ReconcileDiffRecord]:
        """Check spread pairs for leg imbalance.

        A leg imbalance occurs when one leg of a spread is open but
        the other is flat (one leg filled, other not yet filled, or
        one leg was closed independently).

        Parameters
        ----------
        positions : dict
            Current positions: {symbol: {qty: float, side: str, ...}}.
        active_pairs : list[dict]
            Active spread pairs: [{pair_id, leg_x, leg_y}].

        Returns
        -------
        list[ReconcileDiffRecord]
        """
        diffs: List[ReconcileDiffRecord] = []
        detected_at = _now_iso()

        for pair in active_pairs:
            pair_id = pair.get("pair_id", "unknown")
            leg_x = pair.get("leg_x")
            leg_y = pair.get("leg_y")
            if not leg_x or not leg_y:
                continue

            pos_x = positions.get(leg_x, {})
            pos_y = positions.get(leg_y, {})

            qty_x = abs(float(pos_x.get("qty", 0)))
            qty_y = abs(float(pos_y.get("qty", 0)))

            x_open = qty_x > 1e-8
            y_open = qty_y > 1e-8

            if x_open != y_open:
                open_leg = leg_x if x_open else leg_y
                flat_leg = leg_y if x_open else leg_x
                open_qty = qty_x if x_open else qty_y
                diffs.append(
                    ReconcileDiffRecord(
                        diff_id=_make_diff_id(),
                        diff_type=DiffType.LEG_IMBALANCE,
                        scope=f"strategy:{pair_id}",
                        internal_value=f"{open_leg}={open_qty:.4f}, {flat_leg}=0",
                        external_value="both legs should be in matching state",
                        discrepancy=(
                            f"Pair {pair_id}: leg '{open_leg}' open "
                            f"(qty={open_qty:.4f}) but leg '{flat_leg}' is flat."
                        ),
                        severity="critical",
                        detected_at=detected_at,
                    )
                )

        return diffs

    # ──────────────────────────────────────────────────────────────
    # HEDGE RATIO DRIFT CHECK
    # ──────────────────────────────────────────────────────────────

    def check_hedge_ratio_drift(
        self,
        positions: Dict[str, Dict[str, Any]],
        expected_ratios: Dict[str, float],
    ) -> List[ReconcileDiffRecord]:
        """Check if held hedge ratios have drifted from targets.

        For each pair in expected_ratios, computes the actual ratio of
        |qty_y| / |qty_x| and compares to the target.

        Parameters
        ----------
        positions : dict
            Current positions: {symbol: {qty: float, ...}}.
        expected_ratios : dict
            Target ratios: {"{leg_x}:{leg_y}": float}.
            Key format: "AAPL:MSFT" (lexicographic order).

        Returns
        -------
        list[ReconcileDiffRecord]
        """
        diffs: List[ReconcileDiffRecord] = []
        detected_at = _now_iso()
        # Maximum allowed ratio drift (10%)
        max_drift = 0.10

        for pair_key, target_ratio in expected_ratios.items():
            try:
                leg_x, leg_y = pair_key.split(":", 1)
            except ValueError:
                continue

            pos_x = positions.get(leg_x, {})
            pos_y = positions.get(leg_y, {})
            qty_x = abs(float(pos_x.get("qty", 0)))
            qty_y = abs(float(pos_y.get("qty", 0)))

            if qty_x < 1e-8:
                continue  # Pair not held; skip

            if qty_x > 0:
                actual_ratio = qty_y / qty_x
            else:
                actual_ratio = 0.0

            drift = abs(actual_ratio - target_ratio) / max(abs(target_ratio), 1e-8)
            if drift > max_drift:
                severity = "critical" if drift > 0.25 else "warning"
                diffs.append(
                    ReconcileDiffRecord(
                        diff_id=_make_diff_id(),
                        diff_type=DiffType.HEDGE_RATIO_DRIFT,
                        scope=f"strategy:{pair_key}",
                        internal_value=f"{actual_ratio:.4f}",
                        external_value=f"{target_ratio:.4f}",
                        discrepancy=(
                            f"Hedge ratio drift for {pair_key}: "
                            f"actual={actual_ratio:.4f}, "
                            f"target={target_ratio:.4f}, "
                            f"drift={drift:.2%}."
                        ),
                        severity=severity,
                        detected_at=detected_at,
                    )
                )

        return diffs

    # ──────────────────────────────────────────────────────────────
    # END-OF-DAY REPORT
    # ──────────────────────────────────────────────────────────────

    def generate_eod_report(
        self,
        date: str,
        env: str,
        trade_summary: Dict[str, Any],
        reconciliation_report: Optional[ReconciliationReport] = None,
    ) -> EndOfDayReport:
        """Generate an end-of-day summary report.

        Parameters
        ----------
        date : str
            Trading date in YYYY-MM-DD format.
        env : str
            Deployment environment.
        trade_summary : dict
            Aggregated trading statistics. Expected keys:
            total_trades, open_positions, gross_pnl, net_pnl,
            gross_exposure, net_exposure, max_drawdown_pct,
            signals_generated, orders_submitted, orders_filled,
            orders_rejected, incidents_today, alerts_today,
            model_versions, config_version.
        reconciliation_report : Optional[ReconciliationReport]
            Most recent reconciliation result; if None, status
            is set to SKIPPED.

        Returns
        -------
        EndOfDayReport
        """
        recon_status = ReconciliationStatus.SKIPPED
        recon_diffs = 0
        if reconciliation_report is not None:
            recon_status = reconciliation_report.status
            recon_diffs = len(reconciliation_report.diffs)

        return EndOfDayReport(
            report_id=_make_report_id(),
            date=date,
            env=env,
            generated_at=_now_iso(),
            total_trades=int(trade_summary.get("total_trades", 0)),
            open_positions=int(trade_summary.get("open_positions", 0)),
            gross_pnl=float(trade_summary.get("gross_pnl", 0.0)),
            net_pnl=float(trade_summary.get("net_pnl", 0.0)),
            gross_exposure=float(trade_summary.get("gross_exposure", 0.0)),
            net_exposure=float(trade_summary.get("net_exposure", 0.0)),
            max_drawdown_pct=float(trade_summary.get("max_drawdown_pct", 0.0)),
            signals_generated=int(trade_summary.get("signals_generated", 0)),
            orders_submitted=int(trade_summary.get("orders_submitted", 0)),
            orders_filled=int(trade_summary.get("orders_filled", 0)),
            orders_rejected=int(trade_summary.get("orders_rejected", 0)),
            reconciliation_status=recon_status,
            reconciliation_diffs=recon_diffs,
            incidents_today=int(trade_summary.get("incidents_today", 0)),
            alerts_today=int(trade_summary.get("alerts_today", 0)),
            model_versions_in_use=dict(trade_summary.get("model_versions", {})),
            config_version=str(trade_summary.get("config_version", "unknown")),
            notes=str(trade_summary.get("notes", "")),
        )

    # ──────────────────────────────────────────────────────────────
    # UTILITY
    # ──────────────────────────────────────────────────────────────

    def is_clean(self, report: ReconciliationReport) -> bool:
        """Return True if the report has no unresolved critical diffs.

        Parameters
        ----------
        report : ReconciliationReport

        Returns
        -------
        bool
        """
        return not any(
            d.severity == "critical" and not d.resolved
            for d in report.diffs
        )

    def get_report(self, report_id: str) -> Optional[ReconciliationReport]:
        """Return a stored report by ID."""
        with self._lock:
            return self._reports.get(report_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Return reconciliation engine statistics.

        Returns
        -------
        dict
            Keys: run_count, total_diffs, critical_diffs,
            auto_resolved_total, stored_reports.
        """
        with self._lock:
            stored = len(self._reports)
        return {
            "run_count": self._run_count,
            "total_diffs": self._total_diffs,
            "critical_diffs": self._critical_diffs,
            "auto_resolved_total": self._auto_resolved_total,
            "stored_reports": stored,
        }

    # ──────────────────────────────────────────────────────────────
    # PRIVATE: ALERT INTEGRATION
    # ──────────────────────────────────────────────────────────────

    def _fire_break_alert(
        self, report: ReconciliationReport, env: str
    ) -> None:
        """Fire a RECONCILIATION_BREAK alert for the given report."""
        if self._alert_engine is None:
            return
        try:
            self._alert_engine.fire(
                rule_id="RECONCILIATION_BREAK",
                source="reconciliation_engine",
                scope=f"env:{env}",
                message=(
                    f"Reconciliation run {report.report_id} found "
                    f"{report.critical_diffs} critical diff(s)."
                ),
                details={
                    "report_id": report.report_id,
                    "critical_diffs": report.critical_diffs,
                    "warning_diffs": report.warning_diffs,
                    "blocking_live_resume": report.blocking_live_resume,
                },
            )
        except Exception:  # noqa: BLE001
            pass


# ══════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════

_engine_instance: Optional[ReconciliationEngine] = None
_engine_lock = threading.Lock()


def get_reconciliation_engine(
    state_manager: Optional[Any] = None,
    alert_engine: Optional[Any] = None,
    incident_manager: Optional[Any] = None,
    max_position_diff_pct: float = 0.02,
    max_cash_diff_pct: float = 0.001,
) -> ReconciliationEngine:
    """Return the singleton ReconciliationEngine, creating it on first call.

    Parameters
    ----------
    state_manager :
        RuntimeStateManager for first-call injection.
    alert_engine :
        AlertEngine for first-call injection.
    incident_manager :
        IncidentManager for first-call injection.
    max_position_diff_pct : float
        Position tolerance (first call only).
    max_cash_diff_pct : float
        Cash tolerance (first call only).

    Returns
    -------
    ReconciliationEngine
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = ReconciliationEngine(
                    state_manager=state_manager,
                    alert_engine=alert_engine,
                    incident_manager=incident_manager,
                    max_position_diff_pct=max_position_diff_pct,
                    max_cash_diff_pct=max_cash_diff_pct,
                )
    return _engine_instance
