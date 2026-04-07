# -*- coding: utf-8 -*-
"""
core/factor_exposure.py — Factor Exposure Tracker
===================================================

Tracks and neutralises residual factor exposures in pairs portfolios.

A pairs trade that looks "market-neutral" may still carry:
  - Beta-to-SPY (if hedge ratio drifts)
  - Sector beta (if the two legs are in different sub-sectors)
  - Size factor (if one leg is large-cap, other mid-cap)
  - Momentum factor

This module computes marginal and portfolio-level factor exposures
and flags when a pair's net exposure exceeds acceptable thresholds.

Usage:
    from core.factor_exposure import FactorExposureEngine, PairFactorExposure

    engine = FactorExposureEngine()
    exposure = engine.compute_pair_exposure(
        returns_x, returns_y, beta_xy, spy_returns
    )
    print(exposure.net_spy_beta)   # Should be near 0 for market-neutral pair
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairFactorExposure:
    """Factor exposure profile for a single pairs position."""
    pair_id: str = ""

    # Market beta (SPY)
    beta_x: float = 0.0          # Beta of leg X to SPY
    beta_y: float = 0.0          # Beta of leg Y to SPY
    net_spy_beta: float = 0.0    # Net beta of the pair: beta_x - hedge_ratio * beta_y

    # Residual vol (idiosyncratic component)
    residual_vol_x: float = 0.0
    residual_vol_y: float = 0.0
    spread_factor_vol: float = 0.0  # Fraction of spread vol explained by factors

    # Exposure flags
    spy_beta_breach: bool = False   # |net_spy_beta| > threshold
    sector_mismatch: bool = False   # Legs in different GICS sectors

    # Fit quality
    r2_x: float = 0.0
    r2_y: float = 0.0
    n_obs: int = 0

    def is_clean(self, max_net_beta: float = 0.15) -> bool:
        """Returns True if the pair is factor-clean (market-neutral)."""
        return abs(self.net_spy_beta) <= max_net_beta and not self.sector_mismatch


@dataclass
class PortfolioFactorExposure:
    """Aggregate factor exposure across all open positions."""
    total_gross_notional: float = 0.0
    gross_spy_beta_exposure: float = 0.0   # $ gross beta exposure
    net_spy_beta_exposure: float = 0.0     # $ net beta exposure

    pair_exposures: Dict[str, PairFactorExposure] = field(default_factory=dict)

    # Concentration
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    largest_net_beta: float = 0.0

    def portfolio_beta(self) -> float:
        """Portfolio-level beta-to-SPY."""
        if self.total_gross_notional < 1.0:
            return 0.0
        return self.net_spy_beta_exposure / self.total_gross_notional

    def is_market_neutral(self, threshold: float = 0.10) -> bool:
        """True if portfolio net beta is within threshold."""
        return abs(self.portfolio_beta()) <= threshold


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FactorExposureEngine:
    """
    Computes factor exposures for pairs positions.
    Uses OLS regression against SPY (and optionally sector ETFs).
    """

    def __init__(
        self,
        max_net_beta: float = 0.15,
        lookback_days: int = 126,
        min_observations: int = 60,
    ) -> None:
        self.max_net_beta = max_net_beta
        self.lookback_days = lookback_days
        self.min_observations = min_observations

    def compute_pair_exposure(
        self,
        returns_x: pd.Series,
        returns_y: pd.Series,
        hedge_ratio: float,
        spy_returns: pd.Series,
        pair_id: str = "",
    ) -> PairFactorExposure:
        """
        Compute factor exposure for a single pair.

        Uses OLS: r_leg = alpha + beta_spy * r_spy + residual
        Net pair beta = beta_x - hedge_ratio * beta_y
        """
        result = PairFactorExposure(pair_id=pair_id)

        # Align series
        data = pd.concat(
            [returns_x.rename("rx"), returns_y.rename("ry"), spy_returns.rename("spy")],
            axis=1
        ).dropna().iloc[-self.lookback_days:]

        n = len(data)
        result.n_obs = n

        if n < self.min_observations:
            logger.debug("Insufficient data for factor exposure: %d obs", n)
            return result

        spy = data["spy"].values

        # Beta regression for leg X
        beta_x, r2_x, resid_vol_x = self._ols_beta(data["rx"].values, spy)
        # Beta regression for leg Y
        beta_y, r2_y, resid_vol_y = self._ols_beta(data["ry"].values, spy)

        result.beta_x = beta_x
        result.beta_y = beta_y
        result.r2_x = r2_x
        result.r2_y = r2_y
        result.residual_vol_x = resid_vol_x
        result.residual_vol_y = resid_vol_y

        # Net spread beta: long X, short (hedge_ratio * Y)
        result.net_spy_beta = beta_x - hedge_ratio * beta_y
        result.spy_beta_breach = abs(result.net_spy_beta) > self.max_net_beta

        # Spread factor vol: how much of spread variance is from SPY factor?
        spread_returns = data["rx"].values - hedge_ratio * data["ry"].values
        spread_total_var = np.var(spread_returns)
        factor_component_var = (result.net_spy_beta ** 2) * np.var(spy)
        if spread_total_var > 1e-12:
            result.spread_factor_vol = min(1.0, factor_component_var / spread_total_var)

        return result

    def compute_portfolio_exposure(
        self,
        positions: List[dict],
        spy_returns: pd.Series,
        returns_map: Dict[str, pd.Series],
    ) -> PortfolioFactorExposure:
        """
        Compute portfolio-level factor exposure.

        positions: list of dicts with keys: pair_id, notional, hedge_ratio, ticker_x, ticker_y
        returns_map: dict mapping ticker → daily returns series
        """
        portfolio = PortfolioFactorExposure()

        for pos in positions:
            pair_id    = pos.get("pair_id", "")
            notional   = float(pos.get("notional", 0.0))
            hedge_ratio = float(pos.get("hedge_ratio", 1.0))
            ticker_x   = pos.get("ticker_x", "")
            ticker_y   = pos.get("ticker_y", "")

            rx = returns_map.get(ticker_x, pd.Series(dtype=float))
            ry = returns_map.get(ticker_y, pd.Series(dtype=float))

            if len(rx) < self.min_observations or len(ry) < self.min_observations:
                continue

            exposure = self.compute_pair_exposure(rx, ry, hedge_ratio, spy_returns, pair_id)
            portfolio.pair_exposures[pair_id] = exposure

            # Accumulate portfolio exposures
            portfolio.total_gross_notional += notional
            portfolio.gross_spy_beta_exposure += notional * (abs(exposure.beta_x) + abs(hedge_ratio * exposure.beta_y))
            portfolio.net_spy_beta_exposure  += notional * exposure.net_spy_beta

        if portfolio.pair_exposures:
            portfolio.largest_net_beta = max(
                abs(e.net_spy_beta) for e in portfolio.pair_exposures.values()
            )

        return portfolio

    def neutralisation_trade(
        self,
        portfolio: PortfolioFactorExposure,
        spy_price: float = 500.0,
    ) -> Optional[dict]:
        """
        Suggest SPY hedge notional to neutralise portfolio beta.
        Returns None if portfolio is already neutral.
        """
        net_beta_exposure = portfolio.net_spy_beta_exposure
        if abs(net_beta_exposure) < portfolio.total_gross_notional * 0.02:
            return None  # Within 2% → no hedge needed

        # Hedge: short (or long) SPY to offset net beta exposure
        hedge_notional = -net_beta_exposure  # opposite sign to neutralise
        n_spy_shares   = int(hedge_notional / spy_price)

        return {
            "action": "SELL_SPY" if hedge_notional < 0 else "BUY_SPY",
            "hedge_notional": hedge_notional,
            "n_spy_shares": n_spy_shares,
            "portfolio_beta_before": portfolio.portfolio_beta(),
            "portfolio_beta_after": 0.0,   # approximate
        }

    @staticmethod
    def _ols_beta(
        y: np.ndarray, x: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        OLS regression y = alpha + beta*x.
        Returns (beta, R², residual_vol).
        """
        n = len(y)
        if n < 10:
            return 0.0, 0.0, float(np.std(y))

        X = np.column_stack([np.ones(n), x])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = coeffs
            y_hat  = alpha + beta * x
            resid  = y - y_hat
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            resid_vol = float(np.std(resid))
            return float(beta), float(np.clip(r2, 0.0, 1.0)), resid_vol
        except Exception:
            return 0.0, 0.0, float(np.std(y))
