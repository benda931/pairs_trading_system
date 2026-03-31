# -*- coding: utf-8 -*-
"""
core/paper_trader.py — Virtual Pair Portfolio Tracker (inspired by srv_quant)
=============================================================================

Manages a virtual portfolio of pair trades with:
- Position tracking (long/short legs)
- Realistic slippage & commission simulation
- Daily P&L updates
- Trade lifecycle management (entry, exit, time stops)
- JSON persistence for state continuity

Usage:
    from core.paper_trader import PaperTrader
    trader = PaperTrader()
    trader.open_position("AAPL-MSFT", direction="long_spread", entry_z=2.5, notional=10000)
    trader.update_prices({"AAPL": 180.0, "MSFT": 410.0})
    trader.check_exits()
    trader.save_state()
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("paper_trader")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = PROJECT_ROOT / "data" / "paper_portfolio.json"

# Execution cost profiles (bps)
SLIPPAGE_BPS = {
    "aggressive": 8.0,
    "passive": 3.0,
    "twap": 5.0,
    "emergency_exit": 15.0,
}
COMMISSION_BPS = 1.0  # per leg


# ============================================================================
# Data models
# ============================================================================

@dataclass
class PairPosition:
    """An open pair position."""
    trade_id: str
    pair_label: str  # e.g. "AAPL-MSFT"
    sym_x: str
    sym_y: str
    direction: str  # "long_spread" (long Y, short X) or "short_spread"
    entry_date: str
    entry_z: float
    entry_price_x: float
    entry_price_y: float
    notional: float
    hedge_ratio: float = 1.0

    # Live state
    current_price_x: float = 0.0
    current_price_y: float = 0.0
    current_z: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    days_held: int = 0
    entry_cost_bps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PairPosition:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PairTrade:
    """A closed (completed) pair trade."""
    trade_id: str
    pair_label: str
    sym_x: str
    sym_y: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price_x: float
    entry_price_y: float
    exit_price_x: float
    exit_price_y: float
    entry_z: float
    exit_z: float
    notional: float
    realized_pnl: float
    realized_pnl_pct: float
    holding_days: int
    exit_reason: str  # "profit_target", "stop_loss", "time_stop", "regime_exit", "manual"
    total_cost_bps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Paper Trader
# ============================================================================

class PaperTrader:
    """Virtual portfolio manager for pair trading strategies."""

    def __init__(
        self,
        state_path: Path = STATE_PATH,
        max_positions: int = 20,
        default_slippage_style: str = "twap",
        z_profit_target: float = 0.5,
        z_stop_loss: float = 4.0,
        max_hold_days: int = 30,
    ):
        self.state_path = state_path
        self.max_positions = max_positions
        self.default_slippage_style = default_slippage_style
        self.z_profit_target = z_profit_target
        self.z_stop_loss = z_stop_loss
        self.max_hold_days = max_hold_days

        self.positions: List[PairPosition] = []
        self.closed_trades: List[PairTrade] = []
        self.nav_history: List[Dict[str, Any]] = []
        self.initial_capital: float = 1_000_000.0
        self.cash: float = self.initial_capital

        self.load_state()

    # ----------------------------------------------------------------
    # Position management
    # ----------------------------------------------------------------

    def open_position(
        self,
        pair_label: str,
        sym_x: str,
        sym_y: str,
        price_x: float,
        price_y: float,
        entry_z: float,
        notional: float,
        hedge_ratio: float = 1.0,
        direction: str = "long_spread",
        slippage_style: Optional[str] = None,
    ) -> Optional[PairPosition]:
        """Open a new pair position."""
        if len(self.positions) >= self.max_positions:
            logger.warning("Max positions (%d) reached, skipping", self.max_positions)
            return None

        # Check if already in this pair
        existing = [p for p in self.positions if p.pair_label == pair_label]
        if existing:
            logger.info("Already in %s, skipping", pair_label)
            return None

        style = slippage_style or self.default_slippage_style
        cost_bps = SLIPPAGE_BPS.get(style, 5.0) + COMMISSION_BPS * 2  # 2 legs

        pos = PairPosition(
            trade_id=str(uuid.uuid4())[:8],
            pair_label=pair_label,
            sym_x=sym_x,
            sym_y=sym_y,
            direction=direction,
            entry_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            entry_z=entry_z,
            entry_price_x=price_x,
            entry_price_y=price_y,
            notional=notional,
            hedge_ratio=hedge_ratio,
            current_price_x=price_x,
            current_price_y=price_y,
            current_z=entry_z,
            entry_cost_bps=cost_bps,
        )

        # Deduct entry cost from cash
        entry_cost = notional * cost_bps / 10_000
        self.cash -= entry_cost

        self.positions.append(pos)
        logger.info(
            "Opened %s %s: Z=%.2f, notional=$%.0f, cost=%.1fbps",
            direction, pair_label, entry_z, notional, cost_bps,
        )
        return pos

    def close_position(
        self,
        trade_id: str,
        exit_reason: str = "manual",
        exit_z: float = 0.0,
    ) -> Optional[PairTrade]:
        """Close an existing position."""
        pos = next((p for p in self.positions if p.trade_id == trade_id), None)
        if not pos:
            logger.warning("Position %s not found", trade_id)
            return None

        # Calculate exit cost
        style = "emergency_exit" if exit_reason == "stop_loss" else self.default_slippage_style
        exit_cost_bps = SLIPPAGE_BPS.get(style, 5.0) + COMMISSION_BPS * 2

        # Calculate P&L
        spread_entry = pos.entry_price_y - pos.hedge_ratio * pos.entry_price_x
        spread_exit = pos.current_price_y - pos.hedge_ratio * pos.current_price_x
        spread_return = (spread_exit - spread_entry) / abs(spread_entry) if spread_entry != 0 else 0

        if pos.direction == "short_spread":
            spread_return = -spread_return

        total_cost_bps = pos.entry_cost_bps + exit_cost_bps
        gross_pnl = pos.notional * spread_return
        cost = pos.notional * total_cost_bps / 10_000
        realized_pnl = gross_pnl - cost

        trade = PairTrade(
            trade_id=pos.trade_id,
            pair_label=pos.pair_label,
            sym_x=pos.sym_x,
            sym_y=pos.sym_y,
            direction=pos.direction,
            entry_date=pos.entry_date,
            exit_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            entry_price_x=pos.entry_price_x,
            entry_price_y=pos.entry_price_y,
            exit_price_x=pos.current_price_x,
            exit_price_y=pos.current_price_y,
            entry_z=pos.entry_z,
            exit_z=exit_z,
            notional=pos.notional,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl / pos.notional * 100 if pos.notional else 0,
            holding_days=pos.days_held,
            exit_reason=exit_reason,
            total_cost_bps=total_cost_bps,
        )

        self.cash += pos.notional + realized_pnl
        self.positions.remove(pos)
        self.closed_trades.append(trade)

        logger.info(
            "Closed %s %s: reason=%s, PnL=$%.2f (%.2f%%), held %dd",
            pos.direction, pos.pair_label, exit_reason,
            realized_pnl, trade.realized_pnl_pct, pos.days_held,
        )
        return trade

    # ----------------------------------------------------------------
    # Daily updates
    # ----------------------------------------------------------------

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all positions and recalculate P&L."""
        for pos in self.positions:
            if pos.sym_x in prices:
                pos.current_price_x = prices[pos.sym_x]
            if pos.sym_y in prices:
                pos.current_price_y = prices[pos.sym_y]

            # Recalculate unrealized P&L
            spread_entry = pos.entry_price_y - pos.hedge_ratio * pos.entry_price_x
            spread_current = pos.current_price_y - pos.hedge_ratio * pos.current_price_x
            spread_return = (spread_current - spread_entry) / abs(spread_entry) if spread_entry != 0 else 0

            if pos.direction == "short_spread":
                spread_return = -spread_return

            pos.unrealized_pnl = pos.notional * spread_return
            pos.unrealized_pnl_pct = spread_return * 100
            pos.days_held += 1

    def check_exits(self) -> List[PairTrade]:
        """Check all positions for exit conditions. Returns closed trades."""
        exits = []
        for pos in list(self.positions):
            reason = None

            # Profit target: z reverted
            if abs(pos.current_z) <= self.z_profit_target:
                reason = "profit_target"
            # Stop loss: z exploded
            elif abs(pos.current_z) >= self.z_stop_loss:
                reason = "stop_loss"
            # Time stop
            elif pos.days_held >= self.max_hold_days:
                reason = "time_stop"

            if reason:
                trade = self.close_position(pos.trade_id, reason, pos.current_z)
                if trade:
                    exits.append(trade)

        return exits

    # ----------------------------------------------------------------
    # Portfolio analytics
    # ----------------------------------------------------------------

    def nav(self) -> float:
        """Current Net Asset Value."""
        unrealized = sum(p.unrealized_pnl for p in self.positions)
        invested = sum(p.notional for p in self.positions)
        return self.cash + invested + unrealized

    def gross_exposure(self) -> float:
        return sum(p.notional for p in self.positions)

    def summary(self) -> Dict[str, Any]:
        """Portfolio summary snapshot."""
        nav = self.nav()
        n_closed = len(self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t.realized_pnl > 0)
        total_pnl = sum(t.realized_pnl for t in self.closed_trades)

        return {
            "nav": round(nav, 2),
            "cash": round(self.cash, 2),
            "open_positions": len(self.positions),
            "closed_trades": n_closed,
            "win_rate": round(wins / n_closed * 100, 1) if n_closed else 0,
            "total_realized_pnl": round(total_pnl, 2),
            "total_unrealized_pnl": round(sum(p.unrealized_pnl for p in self.positions), 2),
            "gross_exposure": round(self.gross_exposure(), 2),
            "gross_exposure_pct": round(self.gross_exposure() / nav * 100, 1) if nav else 0,
        }

    def record_nav_snapshot(self) -> None:
        """Record current NAV for equity curve tracking."""
        self.nav_history.append({
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "nav": self.nav(),
            "positions": len(self.positions),
        })

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save_state(self) -> None:
        """Save portfolio state to JSON."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": [p.to_dict() for p in self.positions],
            "closed_trades": [t.to_dict() for t in self.closed_trades[-100:]],
            "nav_history": self.nav_history[-365:],
        }
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved paper portfolio: %d positions, NAV=$%.0f", len(self.positions), self.nav())

    def load_state(self) -> None:
        """Load portfolio state from JSON."""
        if not self.state_path.exists():
            return

        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.initial_capital = state.get("initial_capital", 1_000_000.0)
            self.cash = state.get("cash", self.initial_capital)
            self.positions = [PairPosition.from_dict(p) for p in state.get("positions", [])]
            self.closed_trades = [PairTrade(**t) for t in state.get("closed_trades", [])]
            self.nav_history = state.get("nav_history", [])
            logger.info(
                "Loaded paper portfolio: %d positions, %d closed trades",
                len(self.positions), len(self.closed_trades),
            )
        except Exception as e:
            logger.warning("Failed to load paper portfolio: %s", e)
