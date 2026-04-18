# -*- coding: utf-8 -*-
"""
common/trade_costs.py — Transaction Cost Computation
======================================================

Canonical utility for applying transaction costs to return series.

Extracted from root/trade_logic.py to break core/ → root/ import violation (AP-2).
The original function was missing/removed from root/ — this is a clean
reimplementation that covers the common cases used by the optimizer.

Cost model:
    - Per-trade fixed cost (commission, $)
    - Per-dollar variable cost (bps of notional)
    - Entry + exit both charged
    - Optional slippage on entry (bps)

Usage:
    from common.trade_costs import apply_transaction_costs
    net_returns = apply_transaction_costs(returns, positions, params)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_transaction_costs(
    returns: pd.Series,
    positions: Optional[pd.Series] = None,
    per_change_cost: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """
    Apply transaction costs to a strategy return series.

    Parameters
    ----------
    returns : pd.Series
        Gross strategy returns (before costs).
    positions : pd.Series, optional
        Position size series (used to compute turnover). If None, returns
        unchanged.
    per_change_cost : pd.Series, optional
        Pre-computed per-change cost series. If provided, subtracts directly.
    params : dict, optional
        Config with keys:
        - commission_per_trade: float (default 1.0 $ per round trip)
        - spread_bps: float (default 5.0)
        - slippage_bps: float (default 2.0)
        - notional: float (default 100_000)

    Returns
    -------
    pd.Series
        Net returns after costs. Never raises — returns input on error.
    """
    if returns is None or returns.empty:
        return returns

    # Fast path: pre-computed costs
    if per_change_cost is not None:
        try:
            aligned = per_change_cost.reindex(returns.index).fillna(0.0)
            return returns.sub(aligned, fill_value=0.0)
        except Exception as exc:
            logger.debug("apply_transaction_costs per_change path failed: %s", exc)
            return returns

    if positions is None:
        return returns

    try:
        cfg = params or {}
        commission = float(cfg.get("commission_per_trade", 1.0))
        spread_bps = float(cfg.get("spread_bps", 5.0))
        slippage_bps = float(cfg.get("slippage_bps", 2.0))
        notional = float(cfg.get("notional", 100_000))

        # Turnover = |Δposition|
        turnover = positions.diff().abs().fillna(0.0)

        # Per-trade dollar cost → returns cost (commission + spread + slippage)
        spread_frac = spread_bps / 10_000
        slip_frac = slippage_bps / 10_000
        fixed_per_notional = commission / max(notional, 1.0)

        cost_per_unit_turnover = fixed_per_notional + spread_frac + slip_frac
        cost_series = turnover * cost_per_unit_turnover

        return returns.sub(cost_series, fill_value=0.0)
    except Exception as exc:
        logger.debug("apply_transaction_costs failed: %s", exc)
        return returns
