# -*- coding: utf-8 -*-
"""
core/garch_engine.py — GARCH Volatility Forecasting Engine
=============================================================

Conditional volatility modeling for pairs trading:

1. **GARCH(1,1)** — Standard volatility clustering
2. **EWMA** — RiskMetrics exponential smoothing
3. **Realized Volatility** — High-frequency estimator
4. **Volatility Regime Detection** — Low/Normal/High/Crisis
5. **Forecast Horizon** — 1d, 5d, 21d ahead vol forecasts
6. **Vol-of-Vol** — Volatility of volatility for sizing

Usage:
    from core.garch_engine import GarchEngine

    ge = GarchEngine()
    forecast = ge.forecast(returns, horizon=5)
    print(f"5-day vol forecast: {forecast.vol_forecast:.2%}")
    print(f"Regime: {forecast.vol_regime}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VolForecast:
    """Volatility forecast result."""
    as_of: str
    horizon_days: int
    vol_forecast: float                  # Annualized vol forecast
    vol_daily: float                     # Daily vol forecast
    vol_current: float                   # Current realized vol
    vol_ratio: float                     # Forecast / current
    vol_regime: str                      # LOW / NORMAL / HIGH / CRISIS
    vol_percentile: float                # Current vol percentile (0-100)
    vol_of_vol: float                    # Volatility of volatility
    method: str                          # "garch" / "ewma" / "realized"
    confidence_band_lower: float         # Lower vol estimate
    confidence_band_upper: float         # Upper vol estimate
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VolRegime:
    """Volatility regime classification."""
    regime: str                          # LOW / NORMAL / HIGH / CRISIS
    vol_level: float
    percentile: float
    duration_days: int                   # Days in current regime
    transition_prob: Dict[str, float] = field(default_factory=dict)


@dataclass
class GarchParams:
    """Fitted GARCH(1,1) parameters: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}."""
    omega: float                         # Long-run variance intercept
    alpha: float                         # ARCH term (news impact)
    beta: float                          # GARCH term (persistence)
    long_run_var: float                  # ω / (1 - α - β)
    persistence: float                   # α + β
    half_life: float                     # ln(2) / ln(persistence)
    log_likelihood: float = 0.0


class GarchEngine:
    """
    GARCH-based volatility forecasting engine.

    Fits GARCH(1,1) via maximum likelihood, with EWMA and realized vol fallbacks.
    Classifies volatility regimes and produces horizon-ahead forecasts.
    """

    def __init__(
        self,
        ewma_lambda: float = 0.94,
        trading_days: int = 252,
        regime_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.ewma_lambda = ewma_lambda
        self.tdy = trading_days
        self.regime_thresholds = regime_thresholds or {
            "LOW": 0.25,        # Below 25th percentile
            "NORMAL": 0.75,     # 25th-75th
            "HIGH": 0.95,       # 75th-95th
            "CRISIS": 1.01,     # Above 95th
        }

    # ── Forecast ──────────────────────────────────────────────

    def forecast(
        self,
        returns: pd.Series,
        horizon: int = 1,
        method: str = "auto",
    ) -> VolForecast:
        """
        Produce a volatility forecast.

        Parameters
        ----------
        returns : pd.Series
            Daily return series.
        horizon : int
            Forecast horizon in trading days.
        method : str
            "garch", "ewma", "realized", or "auto" (tries GARCH first).
        """
        from datetime import datetime, timezone

        r = returns.dropna()
        n = len(r)

        if n < 30:
            return self._empty_forecast(horizon, method)

        r_vals = r.values

        # Current realized vol
        vol_current = float(np.std(r_vals[-63:], ddof=1) * np.sqrt(self.tdy)) if n >= 63 else float(np.std(r_vals, ddof=1) * np.sqrt(self.tdy))

        # Choose method
        used_method = method
        garch_params = None

        if method in ("auto", "garch"):
            garch_params = self._fit_garch(r_vals)
            if garch_params is not None:
                used_method = "garch"
                daily_var = self._garch_forecast(garch_params, r_vals, horizon)
            else:
                used_method = "ewma"
                daily_var = self._ewma_forecast(r_vals, horizon)
        elif method == "ewma":
            used_method = "ewma"
            daily_var = self._ewma_forecast(r_vals, horizon)
        else:
            used_method = "realized"
            daily_var = float(np.var(r_vals[-min(63, n):], ddof=1))

        vol_daily = float(np.sqrt(max(daily_var, 1e-12)))
        vol_forecast = vol_daily * np.sqrt(self.tdy)

        # Confidence bands (±1 standard error of vol estimate)
        vol_of_vol = self._compute_vol_of_vol(r_vals)
        ci_lower = max(0, vol_forecast - 1.96 * vol_of_vol)
        ci_upper = vol_forecast + 1.96 * vol_of_vol

        # Regime
        regime_info = self._classify_regime(r_vals, vol_current)

        return VolForecast(
            as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            horizon_days=horizon,
            vol_forecast=round(vol_forecast, 6),
            vol_daily=round(vol_daily, 6),
            vol_current=round(vol_current, 6),
            vol_ratio=round(vol_forecast / max(vol_current, 1e-10), 4),
            vol_regime=regime_info.regime,
            vol_percentile=round(regime_info.percentile, 2),
            vol_of_vol=round(vol_of_vol, 6),
            method=used_method,
            confidence_band_lower=round(ci_lower, 6),
            confidence_band_upper=round(ci_upper, 6),
            details={
                "garch_params": {
                    "omega": round(garch_params.omega, 8),
                    "alpha": round(garch_params.alpha, 6),
                    "beta": round(garch_params.beta, 6),
                    "persistence": round(garch_params.persistence, 6),
                    "half_life": round(garch_params.half_life, 2),
                } if garch_params else None,
                "regime_duration_days": regime_info.duration_days,
            },
        )

    # ── GARCH(1,1) ────────────────────────────────────────────

    def _fit_garch(self, r: np.ndarray) -> Optional[GarchParams]:
        """
        Fit GARCH(1,1) via quasi-maximum likelihood.

        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

        Constraints: ω > 0, α ≥ 0, β ≥ 0, α + β < 1
        """
        n = len(r)
        if n < 100:
            return None

        try:
            from scipy.optimize import minimize

            # Demean
            mu = np.mean(r)
            eps = r - mu

            var_sample = float(np.var(eps, ddof=1))

            def neg_log_likelihood(params):
                omega, alpha, beta = params
                if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                    return 1e10

                T = len(eps)
                sigma2 = np.zeros(T)
                sigma2[0] = var_sample

                for t in range(1, T):
                    sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
                    if sigma2[t] <= 0:
                        return 1e10

                # Gaussian log-likelihood
                ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps ** 2 / sigma2)
                return -ll

            # Initial guess
            x0 = [var_sample * 0.05, 0.08, 0.88]
            bounds = [(1e-10, var_sample), (0.001, 0.5), (0.3, 0.999)]

            result = minimize(
                neg_log_likelihood, x0, method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 500},
            )

            if not result.success:
                return None

            omega, alpha, beta = result.x
            persistence = alpha + beta
            if persistence >= 0.9999:
                return None

            long_run_var = omega / (1 - persistence)
            half_life = np.log(2) / (-np.log(persistence)) if persistence > 0 and persistence < 1 else 999.0

            return GarchParams(
                omega=float(omega),
                alpha=float(alpha),
                beta=float(beta),
                long_run_var=float(long_run_var),
                persistence=float(persistence),
                half_life=float(half_life),
                log_likelihood=float(-result.fun),
            )
        except Exception as exc:
            logger.debug("GARCH fit failed: %s", exc)
            return None

    def _garch_forecast(self, params: GarchParams, r: np.ndarray, horizon: int) -> float:
        """Multi-step GARCH variance forecast."""
        mu = np.mean(r)
        eps = r - mu
        n = len(eps)

        # Compute current conditional variance
        sigma2 = params.omega / (1 - params.persistence)  # Start at unconditional
        for t in range(n):
            sigma2 = params.omega + params.alpha * eps[t] ** 2 + params.beta * sigma2

        # h-step ahead forecast
        # σ²_{t+h} = V_L + (α + β)^{h-1} × (σ²_{t+1} - V_L)
        sigma2_h = params.long_run_var + (params.persistence ** (horizon - 1)) * (sigma2 - params.long_run_var)
        return float(max(sigma2_h, 1e-12))

    # ── EWMA ──────────────────────────────────────────────────

    def _ewma_forecast(self, r: np.ndarray, horizon: int) -> float:
        """EWMA (RiskMetrics) variance forecast."""
        lam = self.ewma_lambda
        var = float(r[0] ** 2)

        for t in range(1, len(r)):
            var = lam * var + (1 - lam) * r[t] ** 2

        # EWMA forecast is flat (no mean reversion)
        return float(max(var, 1e-12))

    # ── Vol-of-Vol ────────────────────────────────────────────

    @staticmethod
    def _compute_vol_of_vol(r: np.ndarray, window: int = 21) -> float:
        """Compute volatility of volatility (annualized)."""
        n = len(r)
        if n < window * 3:
            return float(np.std(r) * np.sqrt(252) * 0.2)

        rolling_vol = []
        for i in range(window, n):
            v = np.std(r[i - window:i], ddof=1) * np.sqrt(252)
            rolling_vol.append(v)

        return float(np.std(rolling_vol))

    # ── Regime Classification ─────────────────────────────────

    def _classify_regime(self, r: np.ndarray, current_vol: float) -> VolRegime:
        """Classify current volatility regime."""
        # Historical vol distribution
        n = len(r)
        window = min(21, n // 3)
        hist_vols = []
        for i in range(window, n):
            v = np.std(r[i - window:i], ddof=1) * np.sqrt(self.tdy)
            hist_vols.append(v)

        if not hist_vols:
            return VolRegime(regime="NORMAL", vol_level=current_vol, percentile=50.0, duration_days=0)

        percentile = float(np.mean(np.array(hist_vols) <= current_vol) * 100)

        # Regime classification
        if percentile < self.regime_thresholds["LOW"] * 100:
            regime = "LOW"
        elif percentile < self.regime_thresholds["NORMAL"] * 100:
            regime = "NORMAL"
        elif percentile < self.regime_thresholds["HIGH"] * 100:
            regime = "HIGH"
        else:
            regime = "CRISIS"

        # Duration: how many days in current regime?
        duration = 0
        for i in range(len(hist_vols) - 1, -1, -1):
            v_pct = float(np.mean(np.array(hist_vols) <= hist_vols[i]) * 100)
            if percentile < self.regime_thresholds["LOW"] * 100:
                if v_pct < self.regime_thresholds["LOW"] * 100:
                    duration += 1
                else:
                    break
            elif percentile < self.regime_thresholds["NORMAL"] * 100:
                if self.regime_thresholds["LOW"] * 100 <= v_pct < self.regime_thresholds["NORMAL"] * 100:
                    duration += 1
                else:
                    break
            else:
                if v_pct >= self.regime_thresholds["NORMAL"] * 100:
                    duration += 1
                else:
                    break

        return VolRegime(
            regime=regime,
            vol_level=current_vol,
            percentile=percentile,
            duration_days=duration,
        )

    def _empty_forecast(self, horizon: int, method: str) -> VolForecast:
        from datetime import datetime, timezone
        return VolForecast(
            as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            horizon_days=horizon, vol_forecast=0, vol_daily=0,
            vol_current=0, vol_ratio=1.0, vol_regime="NORMAL",
            vol_percentile=50, vol_of_vol=0, method=method,
            confidence_band_lower=0, confidence_band_upper=0,
        )
