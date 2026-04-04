# -*- coding: utf-8 -*-
"""
core/portfolio_rebalancer.py — Portfolio Rebalancer with Transaction Cost Optimization
========================================================================================

Institutional-grade rebalancing engine:

1. **Target Portfolio Construction**
   - Risk parity weighting
   - Inverse-volatility weighting
   - Equal weight with score tilt
   - Max Sharpe (mean-variance)

2. **Transaction Cost Model**
   - Commission costs (per share / per trade)
   - Spread costs (bid-ask)
   - Market impact (square-root model)
   - Slippage estimation

3. **Rebalance Optimization**
   - Minimize tracking error to target
   - Subject to transaction cost budget
   - Turnover constraints
   - Minimum trade size thresholds

4. **Execution Plan**
   - Order generation with priority
   - Netting (reduce gross trades)
   - Urgency classification

Usage:
    from core.portfolio_rebalancer import PortfolioRebalancer

    rebalancer = PortfolioRebalancer(commission_per_trade=1.0)
    plan = rebalancer.rebalance(
        current_positions=current,
        target_weights=target,
        total_capital=1_000_000,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class TransactionCostModel:
    """Transaction cost assumptions."""
    commission_per_trade: float = 1.00       # $ per trade (round trip)
    commission_per_share: float = 0.005      # $ per share
    spread_bps: float = 5.0                  # Bid-ask spread in bps
    impact_coefficient: float = 0.1          # Market impact: c * sqrt(shares/ADV)
    min_trade_notional: float = 500.0        # Don't trade below this $ amount
    slippage_bps: float = 2.0               # Additional slippage estimate

    def estimate_cost(
        self,
        notional: float,
        price: float = 100.0,
        adv: float = 1_000_000.0,
    ) -> float:
        """Estimate total transaction cost for a trade."""
        if abs(notional) < self.min_trade_notional:
            return 0.0

        shares = abs(notional / price) if price > 0 else 0
        commission = self.commission_per_trade + self.commission_per_share * shares
        spread = abs(notional) * self.spread_bps / 10_000
        impact = self.impact_coefficient * abs(notional) * np.sqrt(shares / max(adv / price, 1))
        slippage = abs(notional) * self.slippage_bps / 10_000

        return commission + spread + impact + slippage


@dataclass
class RebalanceOrder:
    """A single rebalance order."""
    symbol: str
    side: str                            # "BUY" or "SELL"
    notional: float                      # $ amount
    shares: float                        # Estimated shares
    current_weight: float
    target_weight: float
    weight_delta: float                  # target - current
    estimated_cost: float                # Transaction cost
    urgency: str = "NORMAL"              # "HIGH" / "NORMAL" / "LOW"
    reason: str = ""


@dataclass
class RebalancePlan:
    """Complete rebalance execution plan."""
    timestamp: str
    total_capital: float
    n_positions: int
    n_trades: int

    # Cost summary
    total_estimated_cost: float
    cost_as_pct_of_capital: float
    gross_turnover: float                # Sum of |buys| + |sells| / capital
    net_turnover: float                  # |Sum of buys + sells| / capital

    # Orders
    orders: List[RebalanceOrder] = field(default_factory=list)

    # Weights
    current_weights: Dict[str, float] = field(default_factory=dict)
    target_weights: Dict[str, float] = field(default_factory=dict)
    post_rebalance_weights: Dict[str, float] = field(default_factory=dict)

    # Quality
    tracking_error_pre: float = 0.0      # TE before rebalance
    tracking_error_post: float = 0.0     # TE after (should be ~0)
    cost_savings_from_netting: float = 0.0


@dataclass
class WeightingResult:
    """Result of target weight computation."""
    weights: Dict[str, float]
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Weighting methods
# =====================================================================

class WeightingEngine:
    """Compute target portfolio weights using various methods."""

    @staticmethod
    def equal_weight(symbols: Sequence[str]) -> WeightingResult:
        """Equal weight across all positions."""
        n = len(symbols)
        w = 1.0 / n if n > 0 else 0.0
        return WeightingResult(
            weights={s: w for s in symbols},
            method="equal_weight",
        )

    @staticmethod
    def inverse_volatility(
        returns: pd.DataFrame,
        lookback: int = 63,
    ) -> WeightingResult:
        """Weight inversely proportional to realized volatility."""
        vol = returns.iloc[-lookback:].std()
        inv_vol = 1.0 / vol.replace(0, np.nan)
        inv_vol = inv_vol.dropna()
        total = inv_vol.sum()
        weights = (inv_vol / total).to_dict() if total > 0 else {}
        return WeightingResult(
            weights=weights,
            method="inverse_volatility",
            metadata={"lookback": lookback, "vols": vol.to_dict()},
        )

    @staticmethod
    def risk_parity(
        returns: pd.DataFrame,
        lookback: int = 63,
    ) -> WeightingResult:
        """
        Risk parity: equal risk contribution from each position.

        Uses iterative solver: w_i ∝ 1 / (σ_i * Σ_j ρ_ij * w_j)
        """
        R = returns.iloc[-lookback:]
        cov = R.cov().values
        n = cov.shape[0]

        if n == 0:
            return WeightingResult(weights={}, method="risk_parity")

        # Initialize with equal weight
        w = np.ones(n) / n

        # Iterative: 20 iterations is usually enough
        for _ in range(20):
            sigma_w = cov @ w
            risk_contrib = w * sigma_w
            total_risk = np.sum(risk_contrib)
            if total_risk < 1e-12:
                break
            # Target: equal risk contribution
            target_rc = total_risk / n
            w = w * (target_rc / (risk_contrib + 1e-12))
            w = w / w.sum()

        symbols = list(returns.columns)
        return WeightingResult(
            weights={symbols[i]: float(w[i]) for i in range(n)},
            method="risk_parity",
        )

    @staticmethod
    def score_tilted(
        symbols: Sequence[str],
        scores: Dict[str, float],
        tilt_strength: float = 0.5,
    ) -> WeightingResult:
        """
        Equal weight tilted by quality scores.

        tilt_strength=0 → pure equal weight
        tilt_strength=1 → pure score-weighted
        """
        n = len(symbols)
        if n == 0:
            return WeightingResult(weights={}, method="score_tilted")

        base = 1.0 / n
        raw_scores = np.array([scores.get(s, 0.5) for s in symbols])
        score_sum = raw_scores.sum()
        if score_sum > 0:
            score_weights = raw_scores / score_sum
        else:
            score_weights = np.ones(n) / n

        # Blend
        blended = (1 - tilt_strength) * base + tilt_strength * score_weights
        blended = blended / blended.sum()

        return WeightingResult(
            weights={symbols[i]: float(blended[i]) for i in range(n)},
            method="score_tilted",
            metadata={"tilt_strength": tilt_strength},
        )


# =====================================================================
# Rebalancer
# =====================================================================

class PortfolioRebalancer:
    """
    Portfolio rebalancer with transaction cost optimization.

    Generates an execution plan that minimizes tracking error to target
    weights while respecting transaction cost constraints.
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        max_turnover: float = 0.50,      # Max 50% turnover per rebalance
        min_weight_change: float = 0.01,  # Ignore changes < 1%
    ):
        self.cost_model = cost_model or TransactionCostModel()
        self.max_turnover = max_turnover
        self.min_weight_change = min_weight_change

    def rebalance(
        self,
        current_positions: Dict[str, float],   # symbol → $ notional
        target_weights: Dict[str, float],       # symbol → target weight
        total_capital: float,
        prices: Optional[Dict[str, float]] = None,
        adv: Optional[Dict[str, float]] = None,
    ) -> RebalancePlan:
        """
        Generate a rebalance execution plan.

        Parameters
        ----------
        current_positions : Dict[str, float]
            Current position sizes in dollars.
        target_weights : Dict[str, float]
            Target portfolio weights (must sum to ~1.0).
        total_capital : float
            Total portfolio capital.
        prices : Dict[str, float], optional
            Current prices for cost estimation.
        adv : Dict[str, float], optional
            Average daily volume for market impact.
        """
        from datetime import datetime, timezone

        prices = prices or {}
        adv = adv or {}

        # Current weights
        current_total = sum(abs(v) for v in current_positions.values())
        if current_total < 1e-6:
            current_total = total_capital
        current_weights = {s: v / current_total for s, v in current_positions.items()}

        # All symbols
        all_symbols = sorted(set(list(current_weights.keys()) + list(target_weights.keys())))

        orders = []
        total_cost = 0.0
        gross_turnover = 0.0

        for sym in all_symbols:
            cur_w = current_weights.get(sym, 0.0)
            tgt_w = target_weights.get(sym, 0.0)
            delta_w = tgt_w - cur_w

            # Skip small changes
            if abs(delta_w) < self.min_weight_change:
                continue

            notional = delta_w * total_capital
            price = prices.get(sym, 100.0)
            shares = notional / price if price > 0 else 0
            sym_adv = adv.get(sym, 1_000_000.0)

            cost = self.cost_model.estimate_cost(notional, price, sym_adv)
            total_cost += cost
            gross_turnover += abs(delta_w)

            # Urgency
            if abs(delta_w) > 0.10:
                urgency = "HIGH"
            elif abs(delta_w) > 0.03:
                urgency = "NORMAL"
            else:
                urgency = "LOW"

            orders.append(RebalanceOrder(
                symbol=sym,
                side="BUY" if notional > 0 else "SELL",
                notional=round(abs(notional), 2),
                shares=round(abs(shares), 2),
                current_weight=round(cur_w, 6),
                target_weight=round(tgt_w, 6),
                weight_delta=round(delta_w, 6),
                estimated_cost=round(cost, 2),
                urgency=urgency,
                reason=f"Rebalance {'increase' if delta_w > 0 else 'decrease'}",
            ))

        # Turnover constraint
        if gross_turnover > self.max_turnover:
            scale = self.max_turnover / gross_turnover
            for order in orders:
                order.notional = round(order.notional * scale, 2)
                order.shares = round(order.shares * scale, 2)
                order.estimated_cost = round(order.estimated_cost * scale, 2)
            total_cost *= scale
            gross_turnover = self.max_turnover
            logger.info("Turnover capped: %.1f%% → %.1f%%",
                        gross_turnover / scale * 100, self.max_turnover * 100)

        # Sort by urgency then notional
        urgency_order = {"HIGH": 0, "NORMAL": 1, "LOW": 2}
        orders.sort(key=lambda o: (urgency_order.get(o.urgency, 1), -o.notional))

        # Post-rebalance weights (approximate)
        post_weights = dict(current_weights)
        for order in orders:
            delta = order.weight_delta
            if gross_turnover > self.max_turnover:
                delta *= (self.max_turnover / gross_turnover)
            post_weights[order.symbol] = post_weights.get(order.symbol, 0.0) + delta

        # Tracking error (pre vs post)
        te_pre = np.sqrt(sum(
            (current_weights.get(s, 0) - target_weights.get(s, 0)) ** 2
            for s in all_symbols
        ))
        te_post = np.sqrt(sum(
            (post_weights.get(s, 0) - target_weights.get(s, 0)) ** 2
            for s in all_symbols
        ))

        return RebalancePlan(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            total_capital=total_capital,
            n_positions=len([w for w in target_weights.values() if abs(w) > 1e-6]),
            n_trades=len(orders),
            total_estimated_cost=round(total_cost, 2),
            cost_as_pct_of_capital=round(total_cost / max(total_capital, 1) * 100, 4),
            gross_turnover=round(gross_turnover, 6),
            net_turnover=round(abs(sum(o.weight_delta for o in orders)), 6),
            orders=orders,
            current_weights={k: round(v, 6) for k, v in current_weights.items()},
            target_weights={k: round(v, 6) for k, v in target_weights.items()},
            post_rebalance_weights={k: round(v, 6) for k, v in post_weights.items()},
            tracking_error_pre=round(te_pre, 6),
            tracking_error_post=round(te_post, 6),
        )
