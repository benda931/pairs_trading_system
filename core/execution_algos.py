# -*- coding: utf-8 -*-
"""
core/execution_algos.py — Execution Algorithm Framework
=========================================================

Professional execution algorithms for pairs trading:

1. **TWAP** — Time-Weighted Average Price
2. **VWAP** — Volume-Weighted Average Price (estimated)
3. **Iceberg** — Hidden order with small visible clips
4. **Pairs Execution** — Coordinated leg execution with ratio lock
5. **Smart Router** — Urgency-based algorithm selection

Usage:
    from core.execution_algos import ExecutionEngine

    engine = ExecutionEngine()
    plan = engine.create_pairs_execution(
        sym_x="XLI", sym_y="XLB",
        notional_x=50000, notional_y=75000,
        side_x="SHORT", side_y="LONG",
        urgency="NORMAL",
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSlice:
    """A single time slice of an execution algorithm."""
    slice_id: int
    scheduled_time: str                  # ISO UTC
    symbol: str
    side: str                            # BUY / SELL
    target_shares: float
    target_notional: float
    pct_of_total: float                  # What fraction of total order
    algo_type: str                       # TWAP / VWAP / ICEBERG / MARKET
    limit_offset_bps: float = 0.0       # Limit price offset from mid
    max_participation_rate: float = 0.10  # Max % of volume


@dataclass
class PairsExecutionPlan:
    """Coordinated execution plan for a pairs trade (both legs)."""
    plan_id: str
    created_at: str
    pair: str                            # "XLI/XLB"
    algo_type: str                       # TWAP / VWAP / SMART
    urgency: str                         # LOW / NORMAL / HIGH / IMMEDIATE

    # Leg details
    sym_x: str
    sym_y: str
    side_x: str                          # BUY / SELL
    side_y: str                          # BUY / SELL
    notional_x: float
    notional_y: float
    hedge_ratio: float

    # Execution parameters
    n_slices: int
    duration_minutes: int
    start_time: str
    end_time: str

    # Slices
    slices_x: List[ExecutionSlice] = field(default_factory=list)
    slices_y: List[ExecutionSlice] = field(default_factory=list)

    # Risk controls
    max_spread_deviation_pct: float = 2.0   # Cancel if spread deviates > X%
    max_leg_imbalance_pct: float = 10.0     # Max fill imbalance between legs
    cancel_if_halted: bool = True

    # Cost estimates
    estimated_commission: float = 0.0
    estimated_spread_cost: float = 0.0
    estimated_impact: float = 0.0
    estimated_total_cost: float = 0.0


@dataclass
class AlgoConfig:
    """Configuration for an execution algorithm."""
    algo_type: str
    n_slices: int
    duration_minutes: int
    participation_rate: float = 0.10     # Max % of expected volume
    limit_offset_bps: float = 5.0       # Limit offset from mid
    urgency_multiplier: float = 1.0      # Adjusts aggressiveness


class ExecutionEngine:
    """
    Execution algorithm engine for pairs trading.

    Generates coordinated execution plans that maintain hedge ratio
    balance between legs while minimizing market impact.
    """

    ALGO_CONFIGS = {
        "LOW": AlgoConfig("TWAP", n_slices=20, duration_minutes=120, participation_rate=0.05, limit_offset_bps=2),
        "NORMAL": AlgoConfig("TWAP", n_slices=10, duration_minutes=60, participation_rate=0.10, limit_offset_bps=5),
        "HIGH": AlgoConfig("VWAP", n_slices=5, duration_minutes=30, participation_rate=0.20, limit_offset_bps=10),
        "IMMEDIATE": AlgoConfig("MARKET", n_slices=1, duration_minutes=1, participation_rate=1.0, limit_offset_bps=0),
    }

    def __init__(
        self,
        commission_per_trade: float = 1.0,
        spread_bps: float = 5.0,
        impact_coefficient: float = 0.1,
    ):
        self.commission = commission_per_trade
        self.spread_bps = spread_bps
        self.impact_coeff = impact_coefficient

    def create_pairs_execution(
        self,
        sym_x: str,
        sym_y: str,
        notional_x: float,
        notional_y: float,
        side_x: str = "SHORT",
        side_y: str = "LONG",
        hedge_ratio: float = 1.0,
        urgency: str = "NORMAL",
        price_x: float = 100.0,
        price_y: float = 100.0,
        adv_x: float = 5_000_000,
        adv_y: float = 5_000_000,
    ) -> PairsExecutionPlan:
        """
        Create a coordinated pairs execution plan.

        Both legs are sliced in parallel to maintain hedge ratio balance
        throughout execution.
        """
        import uuid

        config = self.ALGO_CONFIGS.get(urgency, self.ALGO_CONFIGS["NORMAL"])

        now = datetime.now(timezone.utc)
        start = now.isoformat(timespec="seconds")
        end_dt = now.replace(
            minute=now.minute + config.duration_minutes
        ) if config.duration_minutes < 60 else now.replace(
            hour=now.hour + config.duration_minutes // 60,
            minute=now.minute + config.duration_minutes % 60,
        )

        try:
            end = end_dt.isoformat(timespec="seconds")
        except (ValueError, OverflowError):
            end = now.isoformat(timespec="seconds")

        # Generate slices
        slices_x = self._generate_slices(
            sym_x, side_x, notional_x, price_x,
            config, n_slices=config.n_slices,
            start_time=now,
        )
        slices_y = self._generate_slices(
            sym_y, side_y, notional_y, price_y,
            config, n_slices=config.n_slices,
            start_time=now,
        )

        # Cost estimates
        shares_x = abs(notional_x / price_x) if price_x > 0 else 0
        shares_y = abs(notional_y / price_y) if price_y > 0 else 0
        est_commission = self.commission * 4  # 2 legs × 2 (entry+exit later)
        est_spread = (abs(notional_x) + abs(notional_y)) * self.spread_bps / 10_000
        est_impact = self.impact_coeff * (
            abs(notional_x) * np.sqrt(shares_x / max(adv_x / price_x, 1))
            + abs(notional_y) * np.sqrt(shares_y / max(adv_y / price_y, 1))
        )

        return PairsExecutionPlan(
            plan_id=uuid.uuid4().hex[:12],
            created_at=start,
            pair=f"{sym_x}/{sym_y}",
            algo_type=config.algo_type,
            urgency=urgency,
            sym_x=sym_x,
            sym_y=sym_y,
            side_x=side_x,
            side_y=side_y,
            notional_x=round(notional_x, 2),
            notional_y=round(notional_y, 2),
            hedge_ratio=round(hedge_ratio, 6),
            n_slices=config.n_slices,
            duration_minutes=config.duration_minutes,
            start_time=start,
            end_time=end,
            slices_x=slices_x,
            slices_y=slices_y,
            estimated_commission=round(est_commission, 2),
            estimated_spread_cost=round(est_spread, 2),
            estimated_impact=round(est_impact, 2),
            estimated_total_cost=round(est_commission + est_spread + est_impact, 2),
        )

    def select_algorithm(
        self,
        notional: float,
        adv: float = 5_000_000,
        volatility: float = 0.15,
        urgency: str = "NORMAL",
    ) -> str:
        """
        Smart algorithm selection based on order characteristics.

        Returns recommended algo type: TWAP / VWAP / ICEBERG / MARKET
        """
        participation = notional / max(adv, 1)

        # Immediate for small orders
        if notional < 10_000 or urgency == "IMMEDIATE":
            return "MARKET"

        # High participation → Iceberg
        if participation > 0.05:
            return "ICEBERG"

        # High vol → VWAP (concentrate in high-volume periods)
        if volatility > 0.25 or urgency == "HIGH":
            return "VWAP"

        # Default
        return "TWAP"

    def _generate_slices(
        self,
        symbol: str,
        side: str,
        total_notional: float,
        price: float,
        config: AlgoConfig,
        n_slices: int,
        start_time: datetime,
    ) -> List[ExecutionSlice]:
        """Generate execution slices for one leg."""
        slices = []
        interval_seconds = max(1, config.duration_minutes * 60 // max(n_slices, 1))

        if config.algo_type == "VWAP":
            # Front-loaded: more volume in first slices (U-shaped approximation)
            raw_weights = np.array([
                1.5 if i < n_slices * 0.2 or i > n_slices * 0.8 else 0.8
                for i in range(n_slices)
            ])
        elif config.algo_type == "ICEBERG":
            # Equal slices with random noise
            raw_weights = np.ones(n_slices) + np.random.uniform(-0.1, 0.1, n_slices)
        else:
            # TWAP: equal
            raw_weights = np.ones(n_slices)

        weights = raw_weights / raw_weights.sum()

        for i in range(n_slices):
            slice_notional = total_notional * weights[i]
            slice_shares = slice_notional / price if price > 0 else 0

            try:
                t = start_time.replace(
                    second=start_time.second + int(i * interval_seconds)
                )
                scheduled = t.isoformat(timespec="seconds")
            except (ValueError, OverflowError):
                scheduled = start_time.isoformat(timespec="seconds")

            slices.append(ExecutionSlice(
                slice_id=i,
                scheduled_time=scheduled,
                symbol=symbol,
                side=side,
                target_shares=round(abs(slice_shares), 4),
                target_notional=round(abs(slice_notional), 2),
                pct_of_total=round(float(weights[i]), 6),
                algo_type=config.algo_type,
                limit_offset_bps=config.limit_offset_bps,
                max_participation_rate=config.participation_rate,
            ))

        return slices
