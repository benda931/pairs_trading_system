# -*- coding: utf-8 -*-
"""
research/spread_constructor.py — Spread Construction Methods
=============================================================

Implements multiple spread construction strategies:
  1. Static OLS (full-window hedge ratio)
  2. Rolling OLS (time-varying, simple)
  3. Kalman filter (optimal time-varying, Bayesian)
  4. Total Return (price ratio)
  5. Factor-neutral residual (removes market/sector beta)

Each method produces a SpreadDefinition that fully captures how the
spread was constructed and can be applied out-of-sample.

IMPORTANT: All methods receive a train/test boundary to prevent
look-ahead bias. Spreading in the test window always uses
parameters estimated from training data only.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.contracts import (
    HedgeRatioMethod,
    PairId,
    SpreadDefinition,
    SpreadModel,
)

logger = logging.getLogger("research.spread_constructor")


# ── Static OLS Spread ─────────────────────────────────────────────

class StaticOLSConstructor:
    """
    Full-window OLS hedge ratio.

    Fits on training data only. Out-of-sample spread uses fixed (beta, alpha).
    Best for pairs with stable structural relationship.
    """

    def __init__(self, use_log_prices: bool = True, window: Optional[int] = None):
        self.use_log_prices = use_log_prices
        self.window = window  # if set, use last `window` days for estimation

    def fit(
        self,
        price_x: pd.Series,
        price_y: pd.Series,
        *,
        train_end: Optional[datetime] = None,
        pair_id: Optional[PairId] = None,
    ) -> SpreadDefinition:
        """Estimate hedge ratio from training data."""
        # Align
        common = price_x.index.intersection(price_y.index)
        px = price_x.loc[common]
        py = price_y.loc[common]

        # Apply train_end cutoff
        if train_end is not None:
            mask = px.index <= pd.Timestamp(train_end)
            px, py = px[mask], py[mask]

        # Apply window
        if self.window and len(px) > self.window:
            px = px.iloc[-self.window:]
            py = py.iloc[-self.window:]

        if self.use_log_prices:
            lx = np.log(px.clip(lower=1e-8))
            ly = np.log(py.clip(lower=1e-8))
        else:
            lx, ly = px, py

        # OLS
        X = np.column_stack([np.ones(len(ly)), ly.values])
        beta_vec, _, _, _ = np.linalg.lstsq(X, lx.values, rcond=None)
        intercept, beta = float(beta_vec[0]), float(beta_vec[1])

        # Compute spread and normalize
        spread = lx - beta * ly - intercept
        spread_mean = float(spread.mean())
        spread_std = float(spread.std())

        if pair_id is None:
            pair_id = PairId(str(price_x.name), str(price_y.name))

        return SpreadDefinition(
            pair_id=pair_id,
            model=SpreadModel.STATIC_OLS,
            hedge_ratio=beta,
            hedge_ratio_method=HedgeRatioMethod.OLS,
            intercept=intercept,
            mean=spread_mean,
            std=max(spread_std, 1e-8),
            estimated_at=datetime.utcnow(),
            train_start=px.index[0].to_pydatetime(),
            train_end=px.index[-1].to_pydatetime(),
        )

    def transform(
        self,
        price_x: pd.Series,
        price_y: pd.Series,
        defn: SpreadDefinition,
        *,
        rolling_window: Optional[int] = None,
    ) -> pd.Series:
        """Apply a fitted SpreadDefinition to produce z-scores."""
        if self.use_log_prices:
            lx = np.log(price_x.clip(lower=1e-8))
            ly = np.log(price_y.clip(lower=1e-8))
        else:
            lx, ly = price_x, price_y

        raw = lx - defn.hedge_ratio * ly - defn.intercept

        if rolling_window and rolling_window > 0:
            mu = raw.rolling(rolling_window).mean()
            sigma = raw.rolling(rolling_window).std().clip(lower=1e-8)
            z = (raw - mu) / sigma
        else:
            z = (raw - defn.mean) / max(defn.std, 1e-8)

        z.name = f"z_{defn.pair_id.label}"
        return z


# ── Rolling OLS Spread ────────────────────────────────────────────

class RollingOLSConstructor:
    """
    Rolling window OLS hedge ratio.

    Re-estimates beta every `recalc_freq` bars using the last `window` bars.
    Handles slow drift in the structural relationship.
    """

    def __init__(
        self,
        window: int = 60,
        recalc_freq: int = 5,
        use_log_prices: bool = True,
    ):
        self.window = window
        self.recalc_freq = recalc_freq
        self.use_log_prices = use_log_prices

    def fit(
        self,
        price_x: pd.Series,
        price_y: pd.Series,
        *,
        train_end: Optional[datetime] = None,
        pair_id: Optional[PairId] = None,
    ) -> SpreadDefinition:
        """Fit rolling OLS. Returns a SpreadDefinition using last-available beta."""
        common = price_x.index.intersection(price_y.index)
        px = price_x.loc[common]
        py = price_y.loc[common]

        if train_end is not None:
            mask = px.index <= pd.Timestamp(train_end)
            px, py = px[mask], py[mask]

        lx = np.log(px.clip(lower=1e-8)) if self.use_log_prices else px
        ly = np.log(py.clip(lower=1e-8)) if self.use_log_prices else py

        # Compute rolling betas
        betas, intercepts = self._rolling_ols(lx, ly)

        # Use last estimated values for definition
        last_beta = float(betas.iloc[-1]) if not betas.empty else 1.0
        last_intercept = float(intercepts.iloc[-1]) if not intercepts.empty else 0.0

        spread = lx - betas.reindex(lx.index).ffill() * ly - intercepts.reindex(lx.index).ffill()
        spread_mean = float(spread.mean())
        spread_std = float(spread.std())

        if pair_id is None:
            pair_id = PairId(str(price_x.name), str(price_y.name))

        return SpreadDefinition(
            pair_id=pair_id,
            model=SpreadModel.ROLLING_OLS,
            hedge_ratio=last_beta,
            hedge_ratio_method=HedgeRatioMethod.ROLLING,
            intercept=last_intercept,
            mean=spread_mean,
            std=max(spread_std, 1e-8),
            window=self.window,
            estimated_at=datetime.utcnow(),
            train_start=px.index[0].to_pydatetime(),
            train_end=px.index[-1].to_pydatetime(),
        )

    def _rolling_ols(
        self,
        log_x: pd.Series,
        log_y: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """Compute rolling OLS betas and intercepts."""
        n = len(log_x)
        betas = pd.Series(np.nan, index=log_x.index)
        intercepts = pd.Series(np.nan, index=log_x.index)

        recalc_points = range(self.window, n, self.recalc_freq)
        for i in recalc_points:
            start = max(0, i - self.window)
            end = i
            lx_sub = log_x.iloc[start:end].values
            ly_sub = log_y.iloc[start:end].values

            X = np.column_stack([np.ones(len(ly_sub)), ly_sub])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, lx_sub, rcond=None)
                intercepts.iloc[i - 1] = coeffs[0]
                betas.iloc[i - 1] = coeffs[1]
            except Exception:
                pass

        return betas.ffill().bfill(), intercepts.ffill().bfill()

    def transform(
        self,
        price_x: pd.Series,
        price_y: pd.Series,
        defn: SpreadDefinition,
        *,
        rolling_window: Optional[int] = None,
    ) -> pd.Series:
        """Apply using fixed beta from defn (simplification for production)."""
        lx = np.log(price_x.clip(lower=1e-8)) if self.use_log_prices else price_x
        ly = np.log(price_y.clip(lower=1e-8)) if self.use_log_prices else price_y
        raw = lx - defn.hedge_ratio * ly - defn.intercept

        win = rolling_window or defn.window
        if win > 0:
            mu = raw.rolling(win).mean()
            sigma = raw.rolling(win).std().clip(lower=1e-8)
            z = (raw - mu) / sigma
        else:
            z = (raw - defn.mean) / max(defn.std, 1e-8)

        z.name = f"z_{defn.pair_id.label}"
        return z


# ── Kalman Filter Spread ──────────────────────────────────────────

class KalmanConstructor:
    """
    Kalman filter for time-varying hedge ratio.

    State: [beta, intercept]
    Observation: price_x = beta * price_y + intercept + noise

    This is the most adaptive spread model. Best when the relationship
    drifts slowly over time (e.g., growth vs value pairs, ETF vs basket).
    """

    def __init__(
        self,
        observation_cov: float = 0.001,
        transition_cov: float = 0.0001,
        use_log_prices: bool = True,
        initial_beta: Optional[float] = None,
    ):
        self.observation_cov = observation_cov
        self.transition_cov = transition_cov
        self.use_log_prices = use_log_prices
        self.initial_beta = initial_beta

    def fit(
        self,
        price_x: pd.Series,
        price_y: pd.Series,
        *,
        train_end: Optional[datetime] = None,
        pair_id: Optional[PairId] = None,
    ) -> SpreadDefinition:
        """Run Kalman filter on training data, return final state as SpreadDefinition."""
        common = price_x.index.intersection(price_y.index)
        px = price_x.loc[common]
        py = price_y.loc[common]

        if train_end is not None:
            mask = px.index <= pd.Timestamp(train_end)
            px, py = px[mask], py[mask]

        lx = np.log(px.clip(lower=1e-8)) if self.use_log_prices else px
        ly = np.log(py.clip(lower=1e-8)) if self.use_log_prices else py

        betas, intercepts = self._kalman_filter(lx.values, ly.values)

        # Last estimated state = current hedge ratio
        last_beta = float(betas[-1])
        last_intercept = float(intercepts[-1])

        # Compute in-sample spread and normalize
        spread = lx.values - betas * ly.values - intercepts
        spread_mean = float(np.nanmean(spread[-60:]))  # Use recent 60 days for norm
        spread_std = float(np.nanstd(spread[-60:]))

        if pair_id is None:
            pair_id = PairId(str(price_x.name), str(price_y.name))

        return SpreadDefinition(
            pair_id=pair_id,
            model=SpreadModel.KALMAN,
            hedge_ratio=last_beta,
            hedge_ratio_method=HedgeRatioMethod.KALMAN,
            intercept=last_intercept,
            mean=spread_mean,
            std=max(spread_std, 1e-8),
            kalman_observation_cov=self.observation_cov,
            kalman_transition_cov=self.transition_cov,
            estimated_at=datetime.utcnow(),
            train_start=px.index[0].to_pydatetime(),
            train_end=px.index[-1].to_pydatetime(),
        )

    def _kalman_filter(
        self,
        log_x: np.ndarray,
        log_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter, return (betas, intercepts) arrays."""
        n = len(log_x)
        # State: [beta, intercept]
        theta = np.array([
            self.initial_beta if self.initial_beta else 1.0,
            0.0
        ])
        P = np.eye(2) * 1.0  # Initial state covariance
        Q = np.eye(2) * self.transition_cov  # Process noise
        R = self.observation_cov              # Observation noise

        betas = np.zeros(n)
        intercepts = np.zeros(n)

        for t in range(n):
            # Measurement matrix: H = [y_t, 1]
            H = np.array([log_y[t], 1.0])

            # Prediction
            # (theta, P unchanged - state transition is identity)
            P_pred = P + Q

            # Innovation
            y_hat = H @ theta
            innovation = log_x[t] - y_hat
            S = H @ P_pred @ H + R  # Innovation covariance

            # Kalman gain
            K = P_pred @ H / max(S, 1e-12)

            # Update
            theta = theta + K * innovation
            P = (np.eye(2) - np.outer(K, H)) @ P_pred

            betas[t] = theta[0]
            intercepts[t] = theta[1]

        return betas, intercepts

    def transform(
        self,
        price_x: pd.Series,
        price_y: pd.Series,
        defn: SpreadDefinition,
        *,
        rolling_window: Optional[int] = None,
    ) -> pd.Series:
        """Apply Kalman with last-known beta (no update without new data)."""
        lx = np.log(price_x.clip(lower=1e-8)) if self.use_log_prices else price_x
        ly = np.log(price_y.clip(lower=1e-8)) if self.use_log_prices else price_y

        raw = lx - defn.hedge_ratio * ly - defn.intercept
        win = rolling_window or 30
        if win > 0:
            mu = raw.rolling(win).mean()
            sigma = raw.rolling(win).std().clip(lower=1e-8)
            z = (raw - mu) / sigma
        else:
            z = (raw - defn.mean) / max(defn.std, 1e-8)

        z.name = f"z_{defn.pair_id.label}"
        return z


# ── Factory ───────────────────────────────────────────────────────

def build_spread(
    pair_id: PairId,
    prices: pd.DataFrame,
    *,
    model: SpreadModel = SpreadModel.STATIC_OLS,
    train_end: Optional[datetime] = None,
    window: int = 60,
    use_log_prices: bool = True,
    kalman_obs_cov: float = 0.001,
    kalman_trans_cov: float = 0.0001,
) -> tuple[SpreadDefinition, pd.Series]:
    """
    Factory function: build a spread definition + z-score series.

    Returns (SpreadDefinition, z_score_series)
    The z-score series covers the full price history (train + test).
    The SpreadDefinition uses parameters estimated from training data only.

    This is the primary function to call from the research pipeline.
    """
    sym_x, sym_y = pair_id.sym_x, pair_id.sym_y

    if sym_x not in prices.columns or sym_y not in prices.columns:
        raise ValueError(f"Prices DataFrame must contain columns {sym_x} and {sym_y}")

    price_x = prices[sym_x].dropna()
    price_y = prices[sym_y].dropna()

    common = price_x.index.intersection(price_y.index)
    price_x = price_x.loc[common]
    price_y = price_y.loc[common]

    if model == SpreadModel.STATIC_OLS:
        constructor = StaticOLSConstructor(use_log_prices=use_log_prices)
    elif model == SpreadModel.ROLLING_OLS:
        constructor = RollingOLSConstructor(window=window, use_log_prices=use_log_prices)
    elif model == SpreadModel.KALMAN:
        constructor = KalmanConstructor(
            observation_cov=kalman_obs_cov,
            transition_cov=kalman_trans_cov,
            use_log_prices=use_log_prices,
        )
    else:
        raise ValueError(f"Unsupported spread model: {model}")

    defn = constructor.fit(price_x, price_y, train_end=train_end, pair_id=pair_id)

    # Rolling window for z-normalization: use window parameter
    rolling_z_window = window if model != SpreadModel.STATIC_OLS else None

    z = constructor.transform(price_x, price_y, defn, rolling_window=rolling_z_window)

    return defn, z
