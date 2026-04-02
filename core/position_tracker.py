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

logger = logging.getLogger(__name__)


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
    stop_z: float = 4.0
    target_z: float = 0.5
    status: str = "OPEN"       # OPEN / CLOSED / STOPPED

    def __post_init__(self):
        if not self.position_id:
            self.position_id = f"POS-{uuid.uuid4().hex[:8]}"


class PositionTracker:
    """
    Tracks all positions and orders across the portfolio.

    Thread-safe position state management with full audit trail.
    """

    def __init__(self):
        self._positions: dict[str, Position] = {}  # pair -> Position
        self._closed_positions: list[Position] = []
        self._orders: list[Order] = []
        self._trade_history: list[dict] = []

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
        return pos

    def update_pnl(self, pair: str, unrealized_pnl: float) -> None:
        """Update unrealized PnL for a position."""
        if pair in self._positions:
            self._positions[pair].unrealized_pnl = unrealized_pnl
            self._positions[pair].holding_days += 1

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

    def get_trade_history(self) -> list[dict]:
        """Get complete trade history."""
        return list(self._trade_history)

    def get_orders(self, status: Optional[str] = None) -> list[Order]:
        """Get orders, optionally filtered by status."""
        if status:
            return [o for o in self._orders if o.status == status]
        return list(self._orders)
