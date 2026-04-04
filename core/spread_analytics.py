# -*- coding: utf-8 -*-
"""
core/spread_analytics.py — Advanced Spread Analytics Engine
=============================================================

Institutional-grade statistical tools for pairs trading:

1. **Cointegration Testing**
   - Engle-Granger 2-step (ADF on residuals)
   - Johansen trace & eigenvalue tests
   - KPSS stationarity test

2. **Mean-Reversion Dynamics**
   - Ornstein-Uhlenbeck half-life (OLS + OU process)
   - Hurst exponent (rescaled range + variance ratio)
   - Variance ratio test (Lo-MacKinlay)

3. **Hedge Ratio Estimation**
   - OLS (static)
   - Rolling OLS (adaptive)
   - Total Least Squares (Deming regression)
   - Error Correction Model (ECM) beta

4. **Spread Quality Scoring**
   - Composite stationarity score
   - Mean-reversion quality index
   - Trading opportunity score

Usage:
    from core.spread_analytics import SpreadAnalytics

    sa = SpreadAnalytics()
    report = sa.full_analysis(prices_x, prices_y)
    print(report.composite_score)
    print(report.half_life_days)
    print(report.is_cointegrated)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class CointegrationResult:
    """Result of cointegration testing."""
    method: str                          # "engle_granger" | "johansen"
    is_cointegrated: bool
    test_statistic: float
    p_value: float                       # For EG; NaN for Johansen
    critical_values: Dict[str, float] = field(default_factory=dict)
    hedge_ratio: float = 0.0
    residual_adf_stat: float = 0.0
    n_cointegrating_vectors: int = 0     # Johansen only
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StationarityResult:
    """Result of stationarity testing (ADF / KPSS)."""
    test_name: str
    is_stationary: bool
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float] = field(default_factory=dict)
    n_lags: int = 0


@dataclass
class MeanReversionMetrics:
    """Mean-reversion quality metrics."""
    half_life_days: float               # OU process half-life
    hurst_exponent: float               # H < 0.5 = mean-reverting
    variance_ratio: float               # VR < 1 = mean-reverting
    variance_ratio_z: float             # Z-stat for VR test
    ar1_coefficient: float              # AR(1) for OU estimation
    ou_mu: float                        # OU long-run mean
    ou_sigma: float                     # OU diffusion coefficient
    ou_theta: float                     # OU mean-reversion speed


@dataclass
class SpreadQuality:
    """Composite spread quality assessment."""
    composite_score: float              # [0, 1] overall quality
    stationarity_score: float           # ADF + KPSS agreement
    mean_reversion_score: float         # Half-life + Hurst
    stability_score: float              # Hedge ratio stability
    trading_score: float                # Tradability (crosses, vol)
    grade: str                          # A+ to F
    warnings: List[str] = field(default_factory=list)


@dataclass
class SpreadAnalysisReport:
    """Complete spread analysis report."""
    sym_x: str
    sym_y: str
    n_observations: int

    # Cointegration
    engle_granger: Optional[CointegrationResult] = None
    johansen: Optional[CointegrationResult] = None
    is_cointegrated: bool = False

    # Stationarity
    adf_spread: Optional[StationarityResult] = None
    kpss_spread: Optional[StationarityResult] = None

    # Mean-reversion
    mean_reversion: Optional[MeanReversionMetrics] = None
    half_life_days: float = float("nan")

    # Hedge ratio
    hedge_ratio_ols: float = 0.0
    hedge_ratio_tls: float = 0.0
    hedge_ratio_rolling_std: float = float("nan")

    # Quality
    quality: Optional[SpreadQuality] = None
    composite_score: float = 0.0

    # Raw data
    spread: Optional[pd.Series] = None
    z_score: Optional[pd.Series] = None


# =====================================================================
# Core analytics
# =====================================================================

class SpreadAnalytics:
    """
    Institutional-grade spread analytics engine.

    Runs a comprehensive battery of statistical tests on a price pair
    and produces a scored SpreadAnalysisReport.
    """

    def __init__(
        self,
        adf_max_lags: Optional[int] = None,
        significance_level: float = 0.05,
        half_life_max: float = 252.0,
        half_life_min: float = 1.0,
        hurst_max_lags: int = 100,
    ):
        self.adf_max_lags = adf_max_lags
        self.significance_level = significance_level
        self.half_life_max = half_life_max
        self.half_life_min = half_life_min
        self.hurst_max_lags = hurst_max_lags

    # ── Public API ────────────────────────────────────────────

    def full_analysis(
        self,
        prices_x: pd.Series,
        prices_y: pd.Series,
        sym_x: str = "X",
        sym_y: str = "Y",
    ) -> SpreadAnalysisReport:
        """Run complete analysis battery on a price pair."""
        report = SpreadAnalysisReport(
            sym_x=sym_x,
            sym_y=sym_y,
            n_observations=min(len(prices_x), len(prices_y)),
        )

        try:
            # Align
            px, py = self._align(prices_x, prices_y)
            if len(px) < 60:
                report.quality = SpreadQuality(
                    composite_score=0, stationarity_score=0,
                    mean_reversion_score=0, stability_score=0,
                    trading_score=0, grade="F",
                    warnings=["Insufficient data (<60 observations)"],
                )
                return report

            # Hedge ratios
            report.hedge_ratio_ols = self._ols_hedge_ratio(px, py)
            report.hedge_ratio_tls = self._tls_hedge_ratio(px, py)

            # Spread
            spread = py - report.hedge_ratio_ols * px
            report.spread = spread
            report.z_score = self._compute_zscore(spread)

            # Rolling hedge ratio stability
            report.hedge_ratio_rolling_std = self._rolling_hedge_stability(px, py)

            # Cointegration tests
            report.engle_granger = self._engle_granger(px, py)
            report.johansen = self._johansen_test(px, py)
            report.is_cointegrated = (
                (report.engle_granger is not None and report.engle_granger.is_cointegrated)
                or (report.johansen is not None and report.johansen.is_cointegrated)
            )

            # Stationarity of spread
            report.adf_spread = self._adf_test(spread)
            report.kpss_spread = self._kpss_test(spread)

            # Mean-reversion metrics
            report.mean_reversion = self._mean_reversion_metrics(spread)
            if report.mean_reversion:
                report.half_life_days = report.mean_reversion.half_life_days

            # Quality scoring
            report.quality = self._score_quality(report)
            report.composite_score = report.quality.composite_score

        except Exception as exc:
            logger.warning("full_analysis failed for %s/%s: %s", sym_x, sym_y, exc)
            report.quality = SpreadQuality(
                composite_score=0, stationarity_score=0,
                mean_reversion_score=0, stability_score=0,
                trading_score=0, grade="F",
                warnings=[f"Analysis error: {exc}"],
            )

        return report

    # ── Hedge Ratios ──────────────────────────────────────────

    @staticmethod
    def _ols_hedge_ratio(px: pd.Series, py: pd.Series) -> float:
        """OLS hedge ratio: β in Y = α + β*X + ε."""
        X = np.column_stack([np.ones(len(px)), px.values])
        try:
            beta = np.linalg.lstsq(X, py.values, rcond=None)[0]
            return float(beta[1])
        except Exception:
            cov_xy = np.cov(px.values, py.values)[0, 1]
            var_x = np.var(px.values)
            return float(cov_xy / var_x) if var_x > 0 else 1.0

    @staticmethod
    def _tls_hedge_ratio(px: pd.Series, py: pd.Series) -> float:
        """Total Least Squares (Deming regression) hedge ratio."""
        try:
            X = np.column_stack([px.values, py.values])
            X_centered = X - X.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_centered)
            # TLS slope = -V[0,1]/V[1,1] (from last right singular vector)
            return float(-Vt[-1, 0] / Vt[-1, 1]) if abs(Vt[-1, 1]) > 1e-12 else 1.0
        except Exception:
            return SpreadAnalytics._ols_hedge_ratio(px, py)

    def _rolling_hedge_stability(
        self, px: pd.Series, py: pd.Series, window: int = 60,
    ) -> float:
        """Standard deviation of rolling OLS hedge ratio (lower = more stable)."""
        n = len(px)
        if n < window + 10:
            return float("nan")
        betas = []
        for i in range(window, n):
            x_win = px.values[i - window:i]
            y_win = py.values[i - window:i]
            var_x = np.var(x_win)
            if var_x > 1e-12:
                cov_xy = np.cov(x_win, y_win)[0, 1]
                betas.append(cov_xy / var_x)
        return float(np.std(betas)) if betas else float("nan")

    # ── Cointegration Tests ───────────────────────────────────

    def _engle_granger(
        self, px: pd.Series, py: pd.Series,
    ) -> Optional[CointegrationResult]:
        """Engle-Granger 2-step cointegration test."""
        try:
            from statsmodels.tsa.stattools import adfuller

            # Step 1: OLS regression
            beta = self._ols_hedge_ratio(px, py)
            residuals = py.values - beta * px.values

            # Step 2: ADF on residuals
            adf_result = adfuller(residuals, maxlag=self.adf_max_lags, autolag="AIC")
            adf_stat, p_value, n_lags = adf_result[0], adf_result[1], adf_result[2]
            crits = {f"{k}%": v for k, v in adf_result[4].items()}

            return CointegrationResult(
                method="engle_granger",
                is_cointegrated=p_value < self.significance_level,
                test_statistic=float(adf_stat),
                p_value=float(p_value),
                critical_values=crits,
                hedge_ratio=float(beta),
                residual_adf_stat=float(adf_stat),
                details={"n_lags": n_lags, "n_obs": len(residuals)},
            )
        except ImportError:
            logger.debug("statsmodels not available for Engle-Granger test")
            return None
        except Exception as exc:
            logger.debug("Engle-Granger failed: %s", exc)
            return None

    def _johansen_test(
        self, px: pd.Series, py: pd.Series,
    ) -> Optional[CointegrationResult]:
        """Johansen cointegration test (trace statistic)."""
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen

            data = np.column_stack([px.values, py.values])
            # det_order=-1 = no constant in cointegrating relation
            # k_ar_diff=1 = 1 lag in VECM
            result = coint_johansen(data, det_order=0, k_ar_diff=1)

            trace_stat = float(result.lr1[0])  # First eigenvalue trace stat
            crit_90 = float(result.cvt[0, 0])  # 90% critical value
            crit_95 = float(result.cvt[0, 1])  # 95% critical value
            crit_99 = float(result.cvt[0, 2])  # 99% critical value

            n_coint = int(np.sum(result.lr1 > result.cvt[:, 1]))  # At 95%

            # Extract hedge ratio from first eigenvector
            evec = result.evec[:, 0]
            hr = float(-evec[0] / evec[1]) if abs(evec[1]) > 1e-12 else 0.0

            return CointegrationResult(
                method="johansen",
                is_cointegrated=trace_stat > crit_95,
                test_statistic=trace_stat,
                p_value=float("nan"),  # Johansen doesn't give p-values directly
                critical_values={"90%": crit_90, "95%": crit_95, "99%": crit_99},
                hedge_ratio=hr,
                n_cointegrating_vectors=n_coint,
                details={
                    "trace_stats": result.lr1.tolist(),
                    "eigen_stats": result.lr2.tolist(),
                    "eigenvectors": result.evec.tolist(),
                },
            )
        except ImportError:
            logger.debug("statsmodels not available for Johansen test")
            return None
        except Exception as exc:
            logger.debug("Johansen test failed: %s", exc)
            return None

    # ── Stationarity Tests ────────────────────────────────────

    def _adf_test(self, series: pd.Series) -> Optional[StationarityResult]:
        """Augmented Dickey-Fuller test for stationarity."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series.dropna(), maxlag=self.adf_max_lags, autolag="AIC")
            crits = {f"{k}%": float(v) for k, v in result[4].items()}
            return StationarityResult(
                test_name="ADF",
                is_stationary=result[1] < self.significance_level,
                test_statistic=float(result[0]),
                p_value=float(result[1]),
                critical_values=crits,
                n_lags=int(result[2]),
            )
        except Exception as exc:
            logger.debug("ADF test failed: %s", exc)
            return None

    def _kpss_test(self, series: pd.Series) -> Optional[StationarityResult]:
        """KPSS test (H0: stationary; reject = non-stationary)."""
        try:
            from statsmodels.tsa.stattools import kpss
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p_value, n_lags, crits_raw = kpss(
                    series.dropna(), regression="c", nlags="auto",
                )
            crits = {f"{k}": float(v) for k, v in crits_raw.items()}
            # KPSS: stationary if we FAIL to reject (p > alpha)
            return StationarityResult(
                test_name="KPSS",
                is_stationary=p_value > self.significance_level,
                test_statistic=float(stat),
                p_value=float(p_value),
                critical_values=crits,
                n_lags=int(n_lags),
            )
        except Exception as exc:
            logger.debug("KPSS test failed: %s", exc)
            return None

    # ── Mean-Reversion Metrics ────────────────────────────────

    def _mean_reversion_metrics(self, spread: pd.Series) -> Optional[MeanReversionMetrics]:
        """Compute OU process parameters + Hurst + Variance Ratio."""
        try:
            s = spread.dropna().values
            n = len(s)
            if n < 30:
                return None

            # OU estimation via AR(1): s_t = c + phi * s_{t-1} + e
            y = s[1:]
            x = s[:-1]
            X = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            c, phi = float(beta[0]), float(beta[1])

            # OU parameters
            theta = -np.log(max(abs(phi), 1e-10))  # Mean-reversion speed
            mu = c / (1 - phi) if abs(1 - phi) > 1e-10 else float(s.mean())
            residuals = y - X @ beta
            sigma = float(np.std(residuals)) * np.sqrt(2 * theta) if theta > 0 else float(np.std(residuals))

            # Half-life
            half_life = np.log(2) / theta if theta > 1e-10 else 999.0
            half_life = np.clip(half_life, self.half_life_min, self.half_life_max * 2)

            # Hurst exponent (rescaled range)
            hurst = self._hurst_exponent(s)

            # Variance ratio test (lag=10)
            vr, vr_z = self._variance_ratio(s, lag=min(10, n // 5))

            return MeanReversionMetrics(
                half_life_days=float(half_life),
                hurst_exponent=float(hurst),
                variance_ratio=float(vr),
                variance_ratio_z=float(vr_z),
                ar1_coefficient=float(phi),
                ou_mu=float(mu),
                ou_sigma=float(sigma),
                ou_theta=float(theta),
            )
        except Exception as exc:
            logger.debug("Mean-reversion metrics failed: %s", exc)
            return None

    def _hurst_exponent(self, series: np.ndarray) -> float:
        """Hurst exponent via rescaled range (R/S) method."""
        n = len(series)
        if n < 20:
            return 0.5

        max_lag = min(self.hurst_max_lags, n // 2)
        lags = range(2, max_lag)
        rs_values = []

        for lag in lags:
            # Split into subseries of length `lag`
            n_sub = n // lag
            if n_sub < 1:
                continue

            rs_list = []
            for i in range(n_sub):
                sub = series[i * lag:(i + 1) * lag]
                mean_sub = np.mean(sub)
                deviations = np.cumsum(sub - mean_sub)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(sub, ddof=1)
                if S > 1e-12:
                    rs_list.append(R / S)

            if rs_list:
                rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

        if len(rs_values) < 3:
            return 0.5

        log_lags = np.array([v[0] for v in rs_values])
        log_rs = np.array([v[1] for v in rs_values])

        # Linear regression: log(R/S) = H * log(lag) + c
        X = np.column_stack([np.ones(len(log_lags)), log_lags])
        beta = np.linalg.lstsq(X, log_rs, rcond=None)[0]
        return float(np.clip(beta[1], 0.0, 1.0))

    @staticmethod
    def _variance_ratio(series: np.ndarray, lag: int = 10) -> Tuple[float, float]:
        """Lo-MacKinlay variance ratio test. VR < 1 → mean-reverting."""
        n = len(series)
        if n < lag * 2:
            return 1.0, 0.0

        returns = np.diff(series)
        var_1 = np.var(returns, ddof=1)
        if var_1 < 1e-12:
            return 1.0, 0.0

        # q-period returns
        q_returns = series[lag:] - series[:-lag]
        var_q = np.var(q_returns, ddof=1)

        vr = var_q / (lag * var_1)

        # Z-statistic under IID assumption
        z = (vr - 1) / np.sqrt(2 * (2 * lag - 1) * (lag - 1) / (3 * lag * n))
        return float(vr), float(z)

    # ── Quality Scoring ───────────────────────────────────────

    def _score_quality(self, report: SpreadAnalysisReport) -> SpreadQuality:
        """Compute composite quality score from all analysis results."""
        warnings_list: List[str] = []

        # 1. Stationarity score (0-1): ADF + KPSS agreement
        stat_score = 0.0
        if report.adf_spread and report.adf_spread.is_stationary:
            stat_score += 0.5
        if report.kpss_spread and report.kpss_spread.is_stationary:
            stat_score += 0.5
        if report.adf_spread and report.kpss_spread:
            if report.adf_spread.is_stationary and not report.kpss_spread.is_stationary:
                warnings_list.append("ADF/KPSS disagree: possible trend-stationary")
                stat_score = 0.35

        # Boost if Engle-Granger or Johansen confirms
        if report.engle_granger and report.engle_granger.is_cointegrated:
            stat_score = min(1.0, stat_score + 0.25)
        if report.johansen and report.johansen.is_cointegrated:
            stat_score = min(1.0, stat_score + 0.25)

        # 2. Mean-reversion score (0-1)
        mr_score = 0.0
        if report.mean_reversion:
            mr = report.mean_reversion
            # Half-life: ideal 5-60 days
            if self.half_life_min <= mr.half_life_days <= self.half_life_max:
                hl_score = 1.0 - abs(mr.half_life_days - 30) / 60
                mr_score += max(0, hl_score) * 0.4
            else:
                warnings_list.append(f"Half-life {mr.half_life_days:.1f}d outside ideal range")

            # Hurst: < 0.5 = mean-reverting, lower is better
            if mr.hurst_exponent < 0.5:
                mr_score += (0.5 - mr.hurst_exponent) * 0.8
            else:
                warnings_list.append(f"Hurst={mr.hurst_exponent:.3f} > 0.5 (trending)")

            # Variance ratio: < 1 = mean-reverting
            if mr.variance_ratio < 1.0:
                mr_score += (1.0 - mr.variance_ratio) * 0.4
            mr_score = min(1.0, mr_score)

        # 3. Stability score (0-1): hedge ratio stability
        stability_score = 0.5  # Default
        if not np.isnan(report.hedge_ratio_rolling_std):
            if report.hedge_ratio_rolling_std < 0.1:
                stability_score = 1.0
            elif report.hedge_ratio_rolling_std < 0.3:
                stability_score = 0.7
            elif report.hedge_ratio_rolling_std < 0.5:
                stability_score = 0.4
            else:
                stability_score = 0.1
                warnings_list.append("Unstable hedge ratio")

        # 4. Trading score (0-1): spread crosses zero, vol
        trading_score = 0.5
        if report.spread is not None:
            s = report.spread.dropna()
            if len(s) > 20:
                # Zero crossings
                signs = np.sign(s.values - s.mean())
                crossings = np.sum(np.diff(signs) != 0)
                cross_rate = crossings / len(s)
                trading_score = min(1.0, cross_rate * 10)  # ~10% crossing rate = perfect

        # Composite
        w = {"stationarity": 0.30, "mean_reversion": 0.30, "stability": 0.20, "trading": 0.20}
        composite = (
            w["stationarity"] * stat_score
            + w["mean_reversion"] * mr_score
            + w["stability"] * stability_score
            + w["trading"] * trading_score
        )

        # Grade
        if composite >= 0.85:
            grade = "A+"
        elif composite >= 0.75:
            grade = "A"
        elif composite >= 0.65:
            grade = "B+"
        elif composite >= 0.55:
            grade = "B"
        elif composite >= 0.45:
            grade = "C"
        elif composite >= 0.30:
            grade = "D"
        else:
            grade = "F"

        return SpreadQuality(
            composite_score=round(composite, 4),
            stationarity_score=round(stat_score, 4),
            mean_reversion_score=round(mr_score, 4),
            stability_score=round(stability_score, 4),
            trading_score=round(trading_score, 4),
            grade=grade,
            warnings=warnings_list,
        )

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def _align(px: pd.Series, py: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two price series on common dates."""
        common = px.index.intersection(py.index)
        return px.loc[common].dropna(), py.loc[common].dropna()

    @staticmethod
    def _compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
        """Rolling z-score of spread."""
        mu = spread.rolling(window).mean()
        sigma = spread.rolling(window).std()
        sigma = sigma.replace(0, np.nan)
        return (spread - mu) / sigma
