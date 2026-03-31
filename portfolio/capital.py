# -*- coding: utf-8 -*-
"""
portfolio/capital.py — Capital Pool & Sleeve Manager
=====================================================

Tracks capital across the portfolio using an explicit pool model:

  Total Capital = Allocated + Reserved + Free
  Sleeves = sub-buckets with max_capital_fraction constraints

CapitalManager is the single source of truth for all capital arithmetic.
It never makes allocation decisions — that is the allocator's job.
It answers: "how much is available here?" and enforces the budget model.

Usage:
    mgr = CapitalManager(total_capital=1_000_000)
    mgr.add_sleeve(SleeveDef("mean_reversion", max_capital_fraction=0.60))
    budget = mgr.sleeve_budget("mean_reversion")   # CapitalBudget
    mgr.allocate("mean_reversion", pair_id, 50_000)
    mgr.release("mean_reversion", pair_id)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from portfolio.contracts import (
    CapitalBudget,
    CapitalPool,
    SleeveDef,
)
from core.contracts import PairId

logger = logging.getLogger("portfolio.capital")


# ── Internal allocation record ────────────────────────────────────

@dataclass
class _Allocation:
    pair_id: PairId
    sleeve: str
    allocated: float
    reserved: float
    is_active: bool = True
    opened_at: datetime = field(default_factory=datetime.utcnow)


# ── Capital Manager ───────────────────────────────────────────────

class CapitalManager:
    """
    Manages capital pool and sleeve budgets for the portfolio.

    Sleeves are sub-buckets (e.g. "mean_reversion", "momentum_pairs").
    Each sleeve has a max_capital_fraction of total capital.
    Capital is tracked at both the portfolio level and per-sleeve level.

    Thread-safety: Not thread-safe. Callers must synchronise.
    """

    def __init__(
        self,
        total_capital: float = 1_000_000.0,
        default_sleeve_name: str = "default",
        max_leverage: float = 2.0,
        margin_buffer_fraction: float = 0.10,
    ):
        if total_capital <= 0:
            raise ValueError("total_capital must be positive")

        self._total_capital = total_capital
        self._default_sleeve_name = default_sleeve_name
        self._max_leverage = max_leverage
        self._margin_buffer = margin_buffer_fraction

        # Sleeve registry
        self._sleeves: dict[str, SleeveDef] = {}
        # Per-pair allocations
        self._allocations: dict[str, _Allocation] = {}  # pair_label → record

        # Add a default sleeve
        self._sleeves[default_sleeve_name] = SleeveDef(
            name=default_sleeve_name,
            description="Default catch-all sleeve",
            max_capital_fraction=1.0,
        )

    # ── Sleeve management ──────────────────────────────────────────

    def add_sleeve(self, sleeve: SleeveDef) -> None:
        """Register a new sleeve."""
        if sleeve.name in self._sleeves:
            logger.warning("Sleeve '%s' already registered; overwriting", sleeve.name)
        self._sleeves[sleeve.name] = sleeve
        logger.debug("Registered sleeve '%s' (max_fraction=%.2f)", sleeve.name, sleeve.max_capital_fraction)

    def remove_sleeve(self, name: str) -> None:
        """Remove a sleeve (only if empty)."""
        active = [a for a in self._allocations.values() if a.sleeve == name and a.is_active]
        if active:
            raise ValueError(f"Cannot remove sleeve '{name}': {len(active)} active allocations")
        self._sleeves.pop(name, None)

    def get_sleeve(self, name: str) -> Optional[SleeveDef]:
        return self._sleeves.get(name)

    def list_sleeves(self) -> list[SleeveDef]:
        return list(self._sleeves.values())

    # ── Capital queries ────────────────────────────────────────────

    def pool_snapshot(self) -> CapitalPool:
        """Current capital pool state."""
        allocated = sum(a.allocated for a in self._allocations.values() if a.is_active)
        reserved = sum(a.reserved for a in self._allocations.values() if a.is_active)
        return CapitalPool(
            total_capital=self._total_capital,
            allocated_capital=allocated,
            reserved_capital=reserved,
        )

    def sleeve_budget(self, sleeve_name: str) -> CapitalBudget:
        """Return current capital budget for one sleeve."""
        sleeve = self._sleeves.get(sleeve_name)
        if sleeve is None:
            raise KeyError(f"Unknown sleeve: '{sleeve_name}'")

        max_cap = sleeve.max_capital_fraction * self._total_capital
        allocated = sum(
            a.allocated for a in self._allocations.values()
            if a.sleeve == sleeve_name and a.is_active
        )
        reserved = sum(
            a.reserved for a in self._allocations.values()
            if a.sleeve == sleeve_name and a.is_active
        )
        n_positions = sum(
            1 for a in self._allocations.values()
            if a.sleeve == sleeve_name and a.is_active
        )

        budget = CapitalBudget(
            sleeve=sleeve,
            allocated=allocated,
            reserved=reserved,
            n_positions=n_positions,
        )
        # Attach max_capital via private attr (CapitalBudget.max_capital uses getattr)
        object.__setattr__(budget, "_max_capital", max_cap)
        return budget

    def all_sleeve_budgets(self) -> dict[str, CapitalBudget]:
        return {name: self.sleeve_budget(name) for name in self._sleeves}

    def free_capital(self) -> float:
        """Total free capital across the portfolio."""
        return self.pool_snapshot().free_capital

    def free_capital_in_sleeve(self, sleeve_name: str) -> float:
        """Free capital available in a specific sleeve."""
        budget = self.sleeve_budget(sleeve_name)
        pool_free = self.free_capital()
        return min(budget.free, pool_free)

    def can_allocate(self, sleeve_name: str, amount: float) -> tuple[bool, str]:
        """
        Check if `amount` can be allocated in `sleeve_name`.

        Returns (ok, reason). reason is empty when ok=True.
        """
        sleeve = self._sleeves.get(sleeve_name)
        if sleeve is None:
            return False, f"Unknown sleeve: '{sleeve_name}'"

        if not sleeve.enabled:
            return False, f"Sleeve '{sleeve_name}' is disabled"

        pool = self.pool_snapshot()
        if not pool.can_allocate(amount):
            return False, (
                f"Insufficient free capital: need {amount:.0f}, "
                f"have {pool.free_capital:.0f}"
            )

        budget = self.sleeve_budget(sleeve_name)
        if budget.free < amount:
            return False, (
                f"Sleeve '{sleeve_name}' budget exceeded: "
                f"need {amount:.0f}, sleeve free {budget.free:.0f}"
            )

        if budget.n_positions >= sleeve.max_positions:
            return False, (
                f"Sleeve '{sleeve_name}' position limit reached: "
                f"{budget.n_positions}/{sleeve.max_positions}"
            )

        return True, ""

    # ── Allocation lifecycle ───────────────────────────────────────

    def allocate(
        self,
        sleeve_name: str,
        pair_id: PairId,
        amount: float,
        *,
        reserve_only: bool = False,
    ) -> None:
        """
        Record a capital allocation for a pair.

        Parameters
        ----------
        sleeve_name : str
        pair_id : PairId
        amount : float — notional capital to commit
        reserve_only : bool — if True, mark as reserved (pending fill) not active allocation
        """
        ok, reason = self.can_allocate(sleeve_name, amount)
        if not ok:
            raise ValueError(f"Cannot allocate {amount:.0f} to {pair_id.label}: {reason}")

        key = pair_id.label
        if key in self._allocations and self._allocations[key].is_active:
            logger.warning("Pair %s already has an active allocation; updating", key)
            rec = self._allocations[key]
            if reserve_only:
                rec.reserved += amount
            else:
                rec.allocated += amount
        else:
            self._allocations[key] = _Allocation(
                pair_id=pair_id,
                sleeve=sleeve_name,
                allocated=0.0 if reserve_only else amount,
                reserved=amount if reserve_only else 0.0,
            )

        logger.debug(
            "Allocated %.0f to %s in sleeve '%s' (reserve=%s)",
            amount, pair_id.label, sleeve_name, reserve_only,
        )

    def confirm_fill(self, pair_id: PairId) -> None:
        """Convert reserved → allocated after order fill confirmation."""
        key = pair_id.label
        rec = self._allocations.get(key)
        if rec is None or not rec.is_active:
            logger.warning("No active reservation found for %s", key)
            return
        rec.allocated += rec.reserved
        rec.reserved = 0.0

    def release(self, pair_id: PairId) -> float:
        """
        Release all capital for a pair (position closed).

        Returns the amount released.
        """
        key = pair_id.label
        rec = self._allocations.get(key)
        if rec is None or not rec.is_active:
            logger.warning("No active allocation found for %s", key)
            return 0.0

        released = rec.allocated + rec.reserved
        rec.is_active = False
        rec.allocated = 0.0
        rec.reserved = 0.0

        logger.debug("Released %.0f from %s", released, key)
        return released

    def adjust(self, pair_id: PairId, new_amount: float) -> None:
        """Adjust the allocated amount for an existing position."""
        key = pair_id.label
        rec = self._allocations.get(key)
        if rec is None or not rec.is_active:
            raise KeyError(f"No active allocation for {key}")

        old = rec.allocated
        delta = new_amount - old

        if delta > 0:
            # Increasing — check there's room
            ok, reason = self.can_allocate(rec.sleeve, delta)
            if not ok:
                raise ValueError(f"Cannot increase allocation for {key}: {reason}")

        rec.allocated = new_amount
        logger.debug(
            "Adjusted %s: %.0f → %.0f (delta=%.0f)",
            key, old, new_amount, delta,
        )

    # ── Position queries ───────────────────────────────────────────

    def active_pairs(self, sleeve_name: Optional[str] = None) -> list[PairId]:
        """List active pair positions, optionally filtered by sleeve."""
        return [
            a.pair_id
            for a in self._allocations.values()
            if a.is_active and (sleeve_name is None or a.sleeve == sleeve_name)
        ]

    def n_active(self, sleeve_name: Optional[str] = None) -> int:
        return len(self.active_pairs(sleeve_name))

    def allocated_for(self, pair_id: PairId) -> float:
        """Total capital allocated to a specific pair."""
        rec = self._allocations.get(pair_id.label)
        if rec is None or not rec.is_active:
            return 0.0
        return rec.allocated + rec.reserved

    def sleeve_of(self, pair_id: PairId) -> Optional[str]:
        """Which sleeve a pair is allocated to."""
        rec = self._allocations.get(pair_id.label)
        return rec.sleeve if rec and rec.is_active else None

    # ── Diagnostics ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        pool = self.pool_snapshot()
        budgets = {
            name: self.sleeve_budget(name).to_dict()
            for name in self._sleeves
        }
        return {
            "pool": pool.to_dict(),
            "sleeves": budgets,
            "n_active_pairs": self.n_active(),
        }

    def reset(self) -> None:
        """Clear all allocations (use for testing / new session)."""
        self._allocations.clear()
        logger.info("CapitalManager reset: all allocations cleared")
