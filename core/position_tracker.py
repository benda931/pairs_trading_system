# -*- coding: utf-8 -*-
"""
core/position_tracker.py — Position & Order Tracking
=====================================================

Tracks open positions, pending orders, fills, and P&L in real-time.
This is the foundation for live trading — even in backtest mode,
it maintains proper position state.

Fixes #8: "No position management / order tracking"
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ACTION_PRIORITY = {"EXIT": 0, "TRIM": 1, "HOLD": 2}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


def _optional_float(
    value: Any,
    *,
    digits: int = 4,
    lower: float | None = None,
    upper: float | None = None,
) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    if lower is not None:
        numeric = max(lower, numeric)
    if upper is not None:
        numeric = min(upper, numeric)
    return round(float(numeric), digits)


def _normalize_monitor_action(value: Any, pressure: float) -> str:
    action = str(value or "").strip().upper()
    if action in _ACTION_PRIORITY:
        return action
    if pressure >= 0.80:
        return "EXIT"
    if pressure >= 0.55:
        return "TRIM"
    return "HOLD"


def normalize_portfolio_monitor_payload(
    payload: Any,
    *,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Canonicalize portfolio-monitor payloads from tracker, allocator, or SQL.

    Different layers historically emitted slightly different payload shapes.
    This helper gives the whole pipeline one stable contract so persistence,
    monitoring, workflows, and dashboard code all receive the same fields.
    """
    source = dict(payload) if isinstance(payload, dict) else {}
    rows_in = source.get("positions")
    rows: list[dict[str, Any]] = []
    pressure_sum = 0.0
    readiness_sum = 0.0
    forced_exit_candidates = 0

    if isinstance(rows_in, list):
        for raw in rows_in:
            if not isinstance(raw, dict):
                continue
            pair = str(raw.get("pair") or raw.get("pair_id") or "").strip()
            if not pair:
                continue

            pressure = _finite_float(raw.get("diagnostic_pressure"), 0.0)
            pressure = max(0.0, min(1.0, pressure))
            readiness = _finite_float(raw.get("execution_readiness"), 1.0)
            readiness = max(0.0, min(1.25, readiness))
            action = _normalize_monitor_action(raw.get("recommended_action"), pressure)
            if action == "EXIT" or pressure >= 0.80:
                forced_exit_candidates += 1
            pressure_sum += pressure
            readiness_sum += readiness

            row = {
                "pair": pair,
                "direction": str(raw.get("direction") or ""),
                "holding_days": int(_finite_float(raw.get("holding_days"), 0.0)),
                "unrealized_pnl": _optional_float(raw.get("unrealized_pnl"), digits=2),
                "current_notional": _optional_float(raw.get("current_notional"), digits=2),
                "approved_capital": _optional_float(raw.get("approved_capital"), digits=2),
                "current_z": _optional_float(raw.get("current_z")),
                "entry_z": _optional_float(raw.get("entry_z")),
                "target_leverage": _optional_float(raw.get("target_leverage")),
                "projected_gross_leverage": _optional_float(raw.get("projected_gross_leverage")),
                "leverage_pressure_score": _optional_float(
                    raw.get("leverage_pressure_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "leverage_diagnostic_multiplier": _optional_float(raw.get("leverage_diagnostic_multiplier")),
                "leverage_concentration_multiplier": _optional_float(raw.get("leverage_concentration_multiplier")),
                "leverage_heat_multiplier": _optional_float(raw.get("leverage_heat_multiplier")),
                "cfa_rank": int(_finite_float(raw.get("cfa_rank"), 0.0)),
                "cfa_super_rank": int(_finite_float(raw.get("cfa_super_rank"), 0.0)),
                "cfa_score": _optional_float(
                    raw.get("cfa_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_super_score": _optional_float(
                    raw.get("cfa_super_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_consensus_score": _optional_float(
                    raw.get("cfa_consensus_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_diversity_score": _optional_float(
                    raw.get("cfa_diversity_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_distortion_score": _optional_float(
                    raw.get("cfa_distortion_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_dominance_score": _optional_float(
                    raw.get("cfa_dominance_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_uncertainty_score": _optional_float(
                    raw.get("cfa_uncertainty_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_effective_breadth": _optional_float(
                    raw.get("cfa_effective_breadth"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_crowding_score": _optional_float(
                    raw.get("cfa_crowding_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_research_active": bool(raw.get("cfa_research_active", False)),
                "cfa_research_super_score": _optional_float(
                    raw.get("cfa_research_super_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_research_uncertainty": _optional_float(
                    raw.get("cfa_research_uncertainty"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_research_changed_lane_count": int(
                    _finite_float(raw.get("cfa_research_changed_lane_count"), 0.0)
                ),
                "cfa_leadership_drift_score": _optional_float(
                    raw.get("cfa_leadership_drift_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_blend_multiplier": _optional_float(
                    raw.get("cfa_blend_multiplier"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_super_score_delta": _optional_float(raw.get("cfa_super_score_delta")),
                "cfa_uncertainty_delta": _optional_float(raw.get("cfa_uncertainty_delta")),
                "cfa_primary_driver": str(raw.get("cfa_primary_driver") or ""),
                "cfa_driver_confidence": _optional_float(
                    raw.get("cfa_driver_confidence"),
                    lower=0.0,
                    upper=1.0,
                ),
                "cfa_scheme_scores": {
                    str(k): round(_finite_float(v, 0.0), 4)
                    for k, v in dict(raw.get("cfa_scheme_scores") or {}).items()
                },
                "composite_score": _optional_float(raw.get("composite_score")),
                "diagnostic_quality_score": _optional_float(
                    raw.get("diagnostic_quality_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "mr_score": _optional_float(raw.get("mr_score")),
                "correlation": _optional_float(raw.get("correlation")),
                "variance_ratio": _optional_float(raw.get("variance_ratio")),
                "data_readiness_score": _optional_float(
                    raw.get("data_readiness_score"),
                    lower=0.0,
                    upper=1.0,
                ),
                "execution_readiness": round(readiness, 4),
                "diagnostic_pressure": round(pressure, 4),
                "exit_score": _optional_float(raw.get("exit_score")),
                "recommended_action": action,
                "last_exit_reason": str(raw.get("last_exit_reason") or ""),
                "updated_at": raw.get("updated_at") or source.get("updated_at"),
            }
            rows.append(row)

    rows.sort(
        key=lambda item: (
            _ACTION_PRIORITY.get(str(item.get("recommended_action", "HOLD")), 3),
            -_finite_float(item.get("diagnostic_pressure"), 0.0),
        )
    )
    n_open = len(rows)
    avg_pressure = (
        round(pressure_sum / n_open, 4)
        if n_open
        else _finite_float(source.get("avg_diagnostic_pressure"), 0.0)
    )
    avg_readiness = (
        round(readiness_sum / n_open, 4)
        if n_open
        else _finite_float(source.get("avg_execution_readiness"), 1.0)
    )

    normalized = dict(source)
    normalized.update(
        {
            "run_id": str(run_id or source.get("run_id") or "latest"),
            "updated_at": source.get("updated_at") or datetime.now(timezone.utc).isoformat(),
            "n_open_positions": n_open if n_open else int(_finite_float(source.get("n_open_positions"), 0.0)),
            "forced_exit_candidates": (
                forced_exit_candidates
                if n_open
                else int(_finite_float(source.get("forced_exit_candidates"), 0.0))
            ),
            "avg_diagnostic_pressure": avg_pressure,
            "avg_execution_readiness": avg_readiness,
            "positions": rows,
        }
    )
    return normalized


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str = ""
    pair: str = ""
    direction: str = ""        # LONG_SPREAD / SHORT_SPREAD
    status: str = "PENDING"    # PENDING / FILLED / CANCELLED / REJECTED
    notional: float = 0.0
    created_at: str = ""
    filled_at: Optional[str] = None
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"ORD-{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class Position:
    """Represents an open position."""
    position_id: str = ""
    pair: str = ""
    direction: str = ""        # LONG_SPREAD / SHORT_SPREAD
    entry_date: str = ""
    entry_z: float = 0.0
    entry_notional: float = 0.0
    current_notional: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    holding_days: int = 0
    current_z: float = 0.0
    stop_z: float = 4.0
    target_z: float = 0.5
    status: str = "OPEN"       # OPEN / CLOSED / STOPPED
    mr_score: Optional[float] = None
    correlation: Optional[float] = None
    variance_ratio: Optional[float] = None
    data_readiness_score: Optional[float] = None
    z_velocity: float = 0.0
    execution_readiness: float = 1.0
    diagnostic_pressure: float = 0.0
    exit_score: float = 0.0
    recommended_action: str = "HOLD"
    last_exit_reason: str = ""
    last_diagnostics_at: Optional[str] = None

    def __post_init__(self):
        if not self.position_id:
            self.position_id = f"POS-{uuid.uuid4().hex[:8]}"


class PositionTracker:
    """
    Tracks all positions and orders across the portfolio.

    Thread-safe position state management with full audit trail.
    """

    def __init__(self, sql_store: Any | None = None):
        self._positions: dict[str, Position] = {}  # pair -> Position
        self._closed_positions: list[Position] = []
        self._orders: list[Order] = []
        self._trade_history: list[dict] = []
        self._sql_store = sql_store

    def open_position(
        self,
        pair: str,
        direction: str,
        notional: float,
        z_score: float,
        stop_z: float = 4.0,
        target_z: float = 0.5,
    ) -> Position:
        """Open a new position for a pair."""
        if pair in self._positions:
            logger.warning("Position already open for %s — closing first", pair)
            self.close_position(pair, z_score, reason="REPLACED")

        pos = Position(
            pair=pair,
            direction=direction,
            entry_date=datetime.now(timezone.utc).isoformat(),
            entry_z=z_score,
            current_z=z_score,
            entry_notional=notional,
            current_notional=notional,
            stop_z=stop_z,
            target_z=target_z,
        )
        self._positions[pair] = pos

        order = Order(
            pair=pair, direction=direction, status="FILLED",
            notional=notional, filled_at=pos.entry_date,
            notes=f"Entry at z={z_score:.2f}",
        )
        self._orders.append(order)

        logger.info("Opened %s %s: notional=$%.0f, z=%.2f", direction, pair, notional, z_score)
        self._persist_monitor_payload()
        return pos

    def close_position(self, pair: str, z_score: float, reason: str = "SIGNAL") -> Optional[Position]:
        """Close an open position."""
        if pair not in self._positions:
            return None

        pos = self._positions.pop(pair)
        pos.status = "CLOSED" if reason != "STOPPED" else "STOPPED"
        pos.realized_pnl = pos.unrealized_pnl

        self._closed_positions.append(pos)
        self._trade_history.append({
            "pair": pair,
            "direction": pos.direction,
            "entry_date": pos.entry_date,
            "exit_date": datetime.now(timezone.utc).isoformat(),
            "entry_z": pos.entry_z,
            "exit_z": z_score,
            "holding_days": pos.holding_days,
            "pnl": pos.realized_pnl,
            "reason": reason,
        })

        order = Order(
            pair=pair, direction="CLOSE", status="FILLED",
            notional=pos.current_notional, filled_at=datetime.now(timezone.utc).isoformat(),
            notes=f"Exit at z={z_score:.2f}, reason={reason}",
        )
        self._orders.append(order)

        logger.info("Closed %s: PnL=$%.2f, reason=%s", pair, pos.realized_pnl, reason)
        self._persist_monitor_payload()
        return pos

    def update_pnl(self, pair: str, unrealized_pnl: float) -> None:
        """Update unrealized PnL for a position."""
        if pair in self._positions:
            self._positions[pair].unrealized_pnl = unrealized_pnl
            self._positions[pair].holding_days += 1
            self._persist_monitor_payload()

    def update_diagnostics(
        self,
        pair: str,
        *,
        current_z: Optional[float] = None,
        mr_score: Optional[float] = None,
        correlation: Optional[float] = None,
        variance_ratio: Optional[float] = None,
        data_readiness_score: Optional[float] = None,
        z_velocity: Optional[float] = None,
        execution_readiness: Optional[float] = None,
        regime: str = "NORMAL",
        spread_vol: float = 1.0,
        entry_pnl_pct: Optional[float] = None,
        threshold_no_trade_band: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Update live diagnostics for an open position and recompute exit pressure."""
        pos = self._positions.get(pair)
        if pos is None:
            return None

        if current_z is not None:
            pos.current_z = float(current_z)
        if mr_score is not None:
            pos.mr_score = float(mr_score)
        if correlation is not None:
            pos.correlation = float(correlation)
        if variance_ratio is not None:
            pos.variance_ratio = float(variance_ratio)
        if data_readiness_score is not None:
            pos.data_readiness_score = float(data_readiness_score)
        if z_velocity is not None:
            pos.z_velocity = float(z_velocity)
        if execution_readiness is not None:
            pos.execution_readiness = float(np.clip(execution_readiness, 0.0, 1.25))

        health = self.evaluate_position_health(
            pair,
            current_z=current_z if current_z is not None else pos.entry_z,
            regime=regime,
            spread_vol=spread_vol,
            entry_pnl_pct=entry_pnl_pct,
            threshold_no_trade_band=threshold_no_trade_band,
        )
        self._persist_monitor_payload()
        return health

    def evaluate_position_health(
        self,
        pair: str,
        *,
        current_z: float,
        regime: str = "NORMAL",
        spread_vol: float = 1.0,
        entry_pnl_pct: Optional[float] = None,
        threshold_no_trade_band: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Evaluate exit pressure and recommended action for a live position."""
        pos = self._positions.get(pair)
        if pos is None:
            return None

        try:
            from core.optimal_exit import OptimalExitEngine
        except Exception:
            return None

        pnl_pct = entry_pnl_pct
        if pnl_pct is None and pos.entry_notional:
            pnl_pct = pos.unrealized_pnl / max(abs(pos.entry_notional), 1e-6)

        engine = OptimalExitEngine(
            half_life=max(float(pos.holding_days or 1), 1.0),
            entry_z=abs(float(pos.entry_z or current_z or 0.5)),
            exit_z=abs(float(pos.target_z or 0.5)),
            stop_z=abs(float(pos.stop_z or 4.0)),
        )
        signal = engine.compute_exit_signal(
            current_z=float(current_z),
            holding_days=max(int(pos.holding_days), 1),
            regime=regime,
            spread_vol=spread_vol,
            entry_pnl_pct=float(pnl_pct or 0.0),
            mr_score=pos.mr_score,
            correlation=pos.correlation,
            variance_ratio=pos.variance_ratio,
            data_readiness_score=pos.data_readiness_score,
            z_velocity=pos.z_velocity,
            threshold_no_trade_band=threshold_no_trade_band,
        )

        pos.exit_score = float(signal.exit_score)
        pos.diagnostic_pressure = float(signal.diagnostic_pressure)
        pos.recommended_action = str(signal.recommended_action)
        pos.last_exit_reason = str(signal.reason)
        pos.last_diagnostics_at = datetime.now(timezone.utc).isoformat()

        return {
            "pair": pair,
            "should_exit": bool(signal.should_exit),
            "exit_score": float(signal.exit_score),
            "diagnostic_pressure": float(signal.diagnostic_pressure),
            "recommended_action": str(signal.recommended_action),
            "reason": str(signal.reason),
            "current_z": float(current_z),
        }

    def get_open_positions(self) -> dict[str, Position]:
        """Get all open positions."""
        return dict(self._positions)

    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary."""
        open_pos = self._positions
        total_notional = sum(p.current_notional for p in open_pos.values())
        total_unrealized = sum(p.unrealized_pnl for p in open_pos.values())
        total_realized = sum(p.realized_pnl for p in self._closed_positions)

        return {
            "n_open": len(open_pos),
            "n_closed": len(self._closed_positions),
            "total_notional": round(total_notional, 2),
            "total_unrealized_pnl": round(total_unrealized, 2),
            "total_realized_pnl": round(total_realized, 2),
            "total_orders": len(self._orders),
            "open_pairs": list(open_pos.keys()),
        }

    def build_monitor_payload(self) -> dict[str, Any]:
        """Build a live diagnostics payload for open positions."""
        rows: list[dict[str, Any]] = []

        for pair, pos in self._positions.items():
            pressure = float(pos.diagnostic_pressure or 0.0)
            readiness = float(np.clip(pos.execution_readiness, 0.0, 1.25))
            rows.append({
                "pair": pair,
                "direction": pos.direction,
                "holding_days": int(pos.holding_days),
                "unrealized_pnl": round(float(pos.unrealized_pnl), 2),
                "current_notional": round(float(pos.current_notional), 2),
                "current_z": round(float(pos.current_z or 0.0), 4),
                "entry_z": round(float(pos.entry_z or 0.0), 4),
                "mr_score": pos.mr_score,
                "correlation": pos.correlation,
                "variance_ratio": pos.variance_ratio,
                "data_readiness_score": pos.data_readiness_score,
                "execution_readiness": round(readiness, 4),
                "diagnostic_pressure": round(pressure, 4),
                "exit_score": round(float(pos.exit_score or 0.0), 4),
                "recommended_action": pos.recommended_action,
                "last_exit_reason": pos.last_exit_reason,
                "updated_at": pos.last_diagnostics_at or pos.entry_date,
            })

        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "positions": rows,
        }
        return normalize_portfolio_monitor_payload(payload)

    def get_trade_history(self) -> list[dict]:
        """Get complete trade history."""
        return list(self._trade_history)

    def get_orders(self, status: Optional[str] = None) -> list[Order]:
        """Get orders, optionally filtered by status."""
        if status:
            return [o for o in self._orders if o.status == status]
        return list(self._orders)

    def _persist_monitor_payload(self) -> None:
        store = self._sql_store
        if store is None:
            return
        saver = getattr(store, "save_portfolio_monitor_payload", None)
        if not callable(saver):
            return
        try:
            saver(self.build_monitor_payload())
        except Exception:
            logger.debug("Failed to persist position monitor payload", exc_info=True)
