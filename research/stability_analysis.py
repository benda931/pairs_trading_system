# -*- coding: utf-8 -*-
"""
research/stability_analysis.py — Rolling Stability, Structural Breaks, Regime Suitability
==========================================================================================

Provides a StabilityAnalyzer that assesses how stable a pair relationship
has been over time. This is a critical step between candidate generation
and promotion to tradable status:

  "The pair passes statistical tests TODAY — but did it also pass 6 months
   ago? 1 year ago? Or is this a regime-specific artefact?"

Implements:
  1. RollingCointegration — tracks coint p-value across rolling windows
  2. RollingCorrelation — tracks correlation stability
  3. RollingHedgeRatio — tracks hedge ratio drift over time
  4. StructuralBreakDetector — Chow-style and CUSUM-style break detection
  5. RegimeSuitabilityChecker — is the spread currently in a tradable state?
  6. StabilityAnalyzer — combines all the above into a StabilityReport

Design rules:
  - All computations use strictly training data (train_end enforced)
  - Stability score is 0–1 (1 = highly stable across all windows)
  - Every instability finding is logged with RejectionReason
  - Structural breaks downgrade but do not automatically reject
    (a post-break stable period may still be tradable)

Usage:
    analyzer = StabilityAnalyzer()
    report = analyzer.analyze(pair_id, prices, train_end=datetime(2024, 1, 1))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.contracts import PairId
from research.discovery_contracts import (
    RejectionReason,
    StabilityReport,
)

logger = logging.getLogger("research.stability_analysis")


# ── Stability score weights ────────────────────────────────────────

@dataclass
class StabilityWeights:
    """
    Weights for stability score components.
    Default values are domain-expert estimates. Use calibrate_weights()
    to fit from historical validation data.
    """
    correlation_stability: float = 0.25
    beta_stability:        float = 0.30
    adf_pass_rate:         float = 0.25
    cointegration_rate:    float = 0.20

    # Penalty weights
    break_penalty:  float = 0.30
    regime_penalty: float = 0.15

    def __post_init__(self) -> None:
        total = (self.correlation_stability + self.beta_stability +
                 self.adf_pass_rate + self.cointegration_rate)
        if abs(total - 1.0) > 0.01:
            # Auto-normalise
            self.correlation_stability /= total
            self.beta_stability        /= total
            self.adf_pass_rate         /= total
            self.cointegration_rate    /= total

    @classmethod
    def calibrate_weights(
        cls,
        component_scores: list[dict],
        oos_pass_labels:  list[int],
        min_samples: int = 50,
    ) -> "StabilityWeights":
        """
        Fit stability weights via logistic regression.

        Args:
            component_scores: list of dicts with keys
                              corr_stability, beta_stability, adf_pass_rate, coint_rate
            oos_pass_labels:  list of 0/1 labels (1 = pair passed OOS validation)

        Returns calibrated StabilityWeights, or default if insufficient data.
        """
        if len(component_scores) < min_samples:
            logger.info(
                "Insufficient data for weight calibration (%d < %d) — using defaults",
                len(component_scores), min_samples,
            )
            return cls()

        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as np  # noqa: F811 — already imported at module level

            X = np.array([[
                d.get("corr_stability", 0.5),
                d.get("beta_stability", 0.5),
                d.get("adf_pass_rate",  0.5),
                d.get("coint_rate",     0.5),
            ] for d in component_scores])
            y = np.array(oos_pass_labels)

            lr = LogisticRegression(fit_intercept=True, max_iter=500)
            lr.fit(X, y)

            # Normalise coefficients to positive weights
            coefs = lr.coef_[0]
            coefs = np.maximum(coefs, 0.0)  # Force non-negative
            if coefs.sum() > 0:
                coefs /= coefs.sum()
                return cls(
                    correlation_stability=float(coefs[0]),
                    beta_stability=float(coefs[1]),
                    adf_pass_rate=float(coefs[2]),
                    cointegration_rate=float(coefs[3]),
                )
        except ImportError:
            logger.warning("sklearn not available; using default stability weights")
        except Exception as e:
            logger.warning("Weight calibration failed: %s — using defaults", e)

        return cls()


# ── Rolling correlation ────────────────────────────────────────────

def rolling_correlation(
    log_ret_x: pd.Series,
    log_ret_y: pd.Series,
    window: int = 63,
) -> pd.Series:
    """
    Compute rolling window Pearson correlation between two return series.
    Returns a pd.Series with the same index as the inputs.
    """
    df = pd.concat([log_ret_x.rename("x"), log_ret_y.rename("y")], axis=1).dropna()
    return df["x"].rolling(window).corr(df["y"])


# ── Rolling OLS hedge ratio ────────────────────────────────────────

def rolling_hedge_ratio(
    log_x: pd.Series,
    log_y: pd.Series,
    window: int = 63,
    recalc_freq: int = 5,
) -> pd.Series:
    """
    Compute rolling OLS beta of log(x) ~ beta * log(y) + alpha.
    Returns a pd.Series of betas aligned to the input index.
    """
    n = len(log_x)
    betas = pd.Series(np.nan, index=log_x.index)

    for i in range(window, n, recalc_freq):
        start = max(0, i - window)
        lx = log_x.iloc[start:i].values
        ly = log_y.iloc[start:i].values
        mask = ~(np.isnan(lx) | np.isnan(ly))
        if mask.sum() < 30:
            continue
        X = np.column_stack([np.ones(mask.sum()), ly[mask]])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, lx[mask], rcond=None)
            betas.iloc[i - 1] = coeffs[1]
        except Exception:
            pass

    return betas.ffill().bfill()


# ── Rolling cointegration ──────────────────────────────────────────

def rolling_cointegration_pvalues(
    log_x: pd.Series,
    log_y: pd.Series,
    window: int = 252,
    step: int = 21,
) -> pd.Series:
    """
    Compute Engle-Granger cointegration p-values over rolling windows.

    Expensive: runs EG test for every step. Use conservatively (step=21
    means monthly re-evaluation, not daily).

    Returns pd.Series of p-values at each step date.
    """
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        logger.warning("statsmodels not available — rolling coint skipped")
        return pd.Series(dtype=float)

    common = log_x.index.intersection(log_y.index)
    lx = log_x.loc[common]
    ly = log_y.loc[common]
    n = len(common)
    pvalues = pd.Series(np.nan, index=common)

    for i in range(window, n, step):
        start = max(0, i - window)
        x_sub = lx.iloc[start:i].values
        y_sub = ly.iloc[start:i].values
        mask = ~(np.isnan(x_sub) | np.isnan(y_sub))
        if mask.sum() < 100:
            continue
        try:
            _, pval, _ = coint(x_sub[mask], y_sub[mask])
            pvalues.iloc[i - 1] = float(pval)
        except Exception:
            pass

    return pvalues.dropna()


# ── Structural break detector ──────────────────────────────────────

class StructuralBreakDetector:
    """
    Detects structural breaks in a spread series using CUSUM-style logic.

    Full Chow test is omitted (requires knowing the break date a priori).
    Instead uses:
    1. CUSUM of residuals: large cumulative deviation = structural drift
    2. Rolling ADF pass-rate: fraction of windows where spread is stationary
    3. Rolling mean shift: detects regime change in spread level
    """

    # Brown-Durbin-Evans critical values (two-sided) at standard significance levels
    _BDE_CRITICAL = {0.01: 1.63, 0.05: 1.36, 0.10: 1.22}

    def __init__(
        self,
        window: int = 126,          # half-year windows for rolling ADF
        significance_level: float = 0.05,  # BDE significance level for CUSUM test
    ):
        self.window = window
        self.significance_level = significance_level
        # Compute BDE critical value; fall back to 5% (1.36) for unrecognised levels
        self._bde_cv = self._BDE_CRITICAL.get(significance_level, 1.36)

    def detect(
        self,
        spread: pd.Series,
    ) -> dict:
        """
        Detect structural breaks in a spread series.

        Returns:
        - has_break: bool
        - break_date: Optional[datetime]
        - break_confidence: float (0–1)
        - cusum_max: maximum CUSUM value
        - rolling_adf_pass_rate: fraction of windows that pass ADF
        - method: description of detection method
        """
        if len(spread) < self.window * 2:
            return {
                "has_break": False,
                "break_date": None,
                "break_confidence": 0.0,
                "cusum_max": 0.0,
                "rolling_adf_pass_rate": np.nan,
                "method": "insufficient_data",
            }

        spread_clean = spread.dropna()

        # CUSUM of standardised residuals
        mu = spread_clean.mean()
        sigma = spread_clean.std() + 1e-8
        std_resid = (spread_clean - mu) / sigma
        cusum = std_resid.cumsum()
        cusum_range = float(cusum.max() - cusum.min())  # CUSUM range

        # Normalise by sqrt(n) (standard CUSUM scaling)
        n = len(spread_clean)
        cusum_stat = cusum_range / np.sqrt(n)

        # Brown-Durbin-Evans CUSUM: reject H₀ (parameter stability) at chosen significance
        # level if cusum_stat > BDE critical value.  The stat is already normalised by √n,
        # so the BDE 5% critical value is simply 1.36 (not divided by √n again).
        bde_threshold_5pct = self._bde_cv   # Brown-Durbin-Evans critical value
        has_cusum_break = cusum_stat > bde_threshold_5pct

        # Find approximate break date (where CUSUM is most extreme)
        break_date = None
        if has_cusum_break:
            peak_idx = cusum.abs().idxmax()
            break_date = peak_idx.to_pydatetime() if hasattr(peak_idx, 'to_pydatetime') else peak_idx

        # Rolling ADF pass rate
        rolling_adf_pass_rate = self._rolling_adf_pass_rate(spread_clean)

        # Rolling mean shift: large jump in rolling mean
        roll_mean = spread_clean.rolling(self.window).mean()
        mean_changes = roll_mean.diff().abs()
        mean_shift_z = mean_changes / (mean_changes.std() + 1e-8)
        max_mean_shift = float(mean_shift_z.max())
        has_mean_shift = max_mean_shift > 3.0

        has_break = bool(has_cusum_break or has_mean_shift)

        # Confidence: weighted combination of signals
        confidence = 0.0
        if has_cusum_break:
            confidence += min(1.0, (cusum_stat - bde_threshold_5pct) / bde_threshold_5pct) * 0.6
        if has_mean_shift:
            confidence += min(1.0, (max_mean_shift - 3.0) / 3.0) * 0.4
        if rolling_adf_pass_rate < 0.5:
            confidence = min(1.0, confidence + 0.2)

        return {
            "has_break": has_break,
            "break_date": break_date,
            "break_confidence": float(min(1.0, confidence)),
            "cusum_max": float(cusum_stat),
            "rolling_adf_pass_rate": rolling_adf_pass_rate,
            "method": "cusum+rolling_mean",
        }

    def _rolling_adf_pass_rate(self, spread: pd.Series, alpha: float = 0.10) -> float:
        """Fraction of rolling windows where ADF rejects unit root."""
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            return np.nan

        n = len(spread)
        if n < self.window * 2:
            return np.nan

        passes = 0
        total = 0
        step = max(1, self.window // 4)

        for i in range(self.window, n, step):
            sub = spread.iloc[i - self.window:i].values
            try:
                _, pval, *_ = adfuller(sub, maxlag=5, autolag=None)
                total += 1
                if pval < alpha:
                    passes += 1
            except Exception:
                pass

        return passes / max(total, 1)


# ── Regime suitability ─────────────────────────────────────────────

class RegimeSuitabilityChecker:
    """
    Checks whether the CURRENT spread state is suitable for mean-reversion trading.

    A pair can be historically cointegrated but currently in:
    - a trending regime (z-score is persistently far from mean)
    - a high-volatility regime (spread variance has regime-shifted)
    - a broken regime (spread mean has shifted, mean-reversion no longer around 0)

    Does NOT replace historical validation — assesses the CURRENT window.
    """

    def __init__(
        self,
        lookback_days: int = 63,
        trend_threshold: float = 0.5,  # |autocorr lag-1| of z-score for "trending"
        vol_expansion_threshold: float = 2.0,  # current_vol / historical_vol
    ):
        self.lookback_days = lookback_days
        self.trend_threshold = trend_threshold
        self.vol_expansion_threshold = vol_expansion_threshold

    def check(
        self,
        spread: pd.Series,
        *,
        train_end: Optional[datetime] = None,
    ) -> dict:
        """
        Assess regime suitability for the most recent lookback_days of spread.

        Returns:
        - regime_suitable: bool
        - regime_label: str
        - caveats: list[str]
        - current_z: float
        - current_spread_vol: float
        - historical_spread_vol: float
        - autocorr_lag1: float
        """
        if train_end is not None:
            spread = spread[spread.index <= pd.Timestamp(train_end)]

        spread_clean = spread.dropna()
        if len(spread_clean) < self.lookback_days * 2:
            return {
                "regime_suitable": True,  # insufficient data = don't penalise
                "regime_label": "UNKNOWN",
                "caveats": ["Insufficient data for regime check"],
                "current_z": np.nan,
                "current_spread_vol": np.nan,
                "historical_spread_vol": np.nan,
                "autocorr_lag1": np.nan,
            }

        # Historical vs recent volatility
        historical = spread_clean.iloc[:-self.lookback_days]
        recent = spread_clean.iloc[-self.lookback_days:]

        hist_std = float(historical.std()) + 1e-8
        curr_std = float(recent.std())
        vol_ratio = curr_std / hist_std

        # Normalise recent spread against historical parameters
        hist_mean = float(historical.mean())
        z_scores_recent = (recent - hist_mean) / hist_std

        current_z = float(z_scores_recent.iloc[-1]) if len(z_scores_recent) > 0 else 0.0

        # Autocorrelation of recent z-scores (positive = trending, negative = reverting)
        autocorr_1 = float(z_scores_recent.autocorr(lag=1)) if len(z_scores_recent) > 10 else 0.0
        is_trending = abs(autocorr_1) > self.trend_threshold and autocorr_1 > 0

        # Vol expansion check
        is_high_vol = vol_ratio > self.vol_expansion_threshold

        # Mean shift: current mean vs historical mean
        recent_z_mean = float(z_scores_recent.mean())
        is_mean_shifted = abs(recent_z_mean) > 1.5  # recent z has drifted from 0

        caveats = []
        if is_trending:
            caveats.append(f"Trending regime: autocorr_1={autocorr_1:.2f}")
        if is_high_vol:
            caveats.append(f"Vol expansion: current/historical={vol_ratio:.1f}x")
        if is_mean_shifted:
            caveats.append(f"Mean shift: recent_z_mean={recent_z_mean:.2f}")

        regime_suitable = not (is_trending or is_high_vol or is_mean_shifted)

        if is_trending and is_high_vol:
            regime_label = "CRISIS"
        elif is_trending:
            regime_label = "TRENDING"
        elif is_high_vol:
            regime_label = "HIGH_VOL"
        elif is_mean_shifted:
            regime_label = "MEAN_SHIFTED"
        else:
            regime_label = "MEAN_REVERTING"

        return {
            "regime_suitable": regime_suitable,
            "regime_label": regime_label,
            "caveats": caveats,
            "current_z": current_z,
            "current_spread_vol": curr_std,
            "historical_spread_vol": hist_std,
            "autocorr_lag1": autocorr_1,
            "vol_ratio": vol_ratio,
        }


# ── Stability analyzer ─────────────────────────────────────────────

class StabilityAnalyzer:
    """
    Combines rolling correlation, rolling hedge ratio, rolling cointegration,
    structural break detection, and regime suitability into a StabilityReport.

    This is the primary tool for answering:
    "Is this pair's statistical relationship stable over time, or is it
    a recent artefact that won't persist out of sample?"
    """

    def __init__(
        self,
        rolling_window: int = 63,          # ~1 quarter for rolling stats
        coint_window: int = 252,           # ~1 year for rolling cointegration
        coint_step: int = 21,              # Monthly re-evaluation of cointegration
        min_stable_windows: float = 0.60,  # Require 60% of rolling windows to be stable
        weights: Optional[StabilityWeights] = None,  # Score component weights
    ):
        self.rolling_window = rolling_window
        self.coint_window = coint_window
        self.coint_step = coint_step
        self.min_stable_windows = min_stable_windows
        self.weights = weights if weights is not None else StabilityWeights()
        self._break_detector = StructuralBreakDetector(window=rolling_window)
        self._regime_checker = RegimeSuitabilityChecker(lookback_days=63)

    def analyze(
        self,
        pair_id: PairId,
        prices: pd.DataFrame,
        *,
        train_end: Optional[datetime] = None,
        run_rolling_coint: bool = True,  # expensive — disable for speed
    ) -> StabilityReport:
        """
        Run full stability analysis for a pair.

        Parameters
        ----------
        pair_id : PairId
        prices : pd.DataFrame with sym_x and sym_y columns
        train_end : Use only prices up to this date
        run_rolling_coint : Whether to run rolling cointegration (slow)

        Returns
        -------
        StabilityReport
        """
        report = StabilityReport(pair_id=pair_id)
        sx, sy = pair_id.sym_x, pair_id.sym_y

        if sx not in prices.columns or sy not in prices.columns:
            report.is_stable = False
            report.instability_reasons.append(RejectionReason.INSUFFICIENT_OVERLAP)
            return report

        cutoff = pd.Timestamp(train_end) if train_end else prices.index[-1]
        px = prices[prices.index <= cutoff]

        px_x = px[sx].dropna()
        px_y = px[sy].dropna()
        common_idx = px_x.index.intersection(px_y.index)

        if len(common_idx) < self.rolling_window * 3:
            report.is_stable = True  # don't penalise for lack of data
            report.notes = "Insufficient data for rolling stability analysis"
            return report

        px_x = px_x.loc[common_idx]
        px_y = px_y.loc[common_idx]

        log_x = np.log(px_x.clip(lower=1e-8))
        log_y = np.log(px_y.clip(lower=1e-8))
        log_ret_x = log_x.diff().dropna()
        log_ret_y = log_y.diff().dropna()

        # ── Rolling correlation ──────────────────────────────────────
        roll_corr = rolling_correlation(log_ret_x, log_ret_y, window=self.rolling_window).dropna()
        if len(roll_corr) > 0:
            report.corr_mean = float(roll_corr.mean())
            report.corr_std = float(roll_corr.std())
            report.corr_min = float(roll_corr.min())
            report.corr_max = float(roll_corr.max())
            # Trend: is correlation improving or deteriorating?
            if len(roll_corr) > 10:
                t = np.arange(len(roll_corr))
                slope, _ = np.polyfit(t, roll_corr.values, 1)
                report.corr_trend = float(slope)

        # ── Rolling hedge ratio ──────────────────────────────────────
        roll_beta = rolling_hedge_ratio(log_x, log_y, window=self.rolling_window).dropna()
        if len(roll_beta) > 0:
            report.beta_mean = float(roll_beta.mean())
            report.beta_std = float(roll_beta.std())
            report.beta_cv = float(roll_beta.std() / (abs(roll_beta.mean()) + 1e-8))
            if len(roll_beta) > 10:
                t = np.arange(len(roll_beta))
                slope, _ = np.polyfit(t, roll_beta.values, 1)
                report.beta_trend = float(slope)

        # ── Compute spread for structural break analysis ─────────────
        beta_full = float(roll_beta.iloc[-1]) if len(roll_beta) > 0 else 1.0
        intercept_full = float(log_x.mean() - beta_full * log_y.mean())
        spread = log_x - beta_full * log_y - intercept_full

        # ── Rolling ADF ──────────────────────────────────────────────
        break_result = self._break_detector.detect(spread)
        report.adf_rolling_pass_rate = break_result.get("rolling_adf_pass_rate", np.nan)
        report.has_structural_break = break_result["has_break"]
        report.break_date = break_result["break_date"]
        report.break_confidence = break_result["break_confidence"]
        report.break_detection_method = break_result["method"]

        # ── Rolling cointegration (optional) ────────────────────────
        if run_rolling_coint and len(common_idx) >= self.coint_window:
            try:
                coint_pvals = rolling_cointegration_pvalues(
                    log_x, log_y,
                    window=self.coint_window,
                    step=self.coint_step,
                )
                if len(coint_pvals) > 0:
                    report.coint_rolling_pass_rate = float((coint_pvals < 0.10).mean())
            except Exception as exc:
                logger.debug("Rolling cointegration failed: %s", exc)

        # ── Regime suitability ───────────────────────────────────────
        regime_result = self._regime_checker.check(spread, train_end=train_end)
        report.regime_suitable = regime_result["regime_suitable"]
        report.regime_label = regime_result["regime_label"]
        report.regime_caveats = regime_result["caveats"]

        # ── Instability reasons ──────────────────────────────────────
        instability_reasons = []

        if not np.isnan(report.corr_std) and report.corr_std > 0.30:
            instability_reasons.append(RejectionReason.UNSTABLE_CORRELATION)

        if not np.isnan(report.beta_cv) and report.beta_cv > 0.40:
            instability_reasons.append(RejectionReason.UNSTABLE_HEDGE_RATIO)

        if report.has_structural_break and report.break_confidence > 0.5:
            instability_reasons.append(RejectionReason.STRUCTURAL_BREAK)

        if not report.regime_suitable:
            instability_reasons.append(RejectionReason.REGIME_UNSUITABLE)

        if not np.isnan(report.adf_rolling_pass_rate) and report.adf_rolling_pass_rate < self.min_stable_windows:
            instability_reasons.append(RejectionReason.FAILED_ADF_STATIONARITY)

        report.instability_reasons = instability_reasons
        report.is_stable = len(instability_reasons) == 0

        # ── Stability score (0–1) ────────────────────────────────────
        w = self.weights  # StabilityWeights instance

        # Compute each raw component value (0–1 scale)
        corr_stability_component = (
            max(0.0, 1.0 - report.corr_std / 0.5)
            if not np.isnan(report.corr_std) else None
        )
        beta_stability_component = (
            max(0.0, 1.0 - report.beta_cv / 0.6)
            if not np.isnan(report.beta_cv) else None
        )
        adf_pass_rate_component = (
            report.adf_rolling_pass_rate
            if not np.isnan(report.adf_rolling_pass_rate) else None
        )
        coint_pass_rate_component = (
            report.coint_rolling_pass_rate
            if not np.isnan(report.coint_rolling_pass_rate) else None
        )

        # Assemble weighted score — only include components that have data,
        # re-normalising weights on the fly so missing components don't
        # silently drag the score toward zero.
        weighted_sum = 0.0
        weight_total = 0.0
        if corr_stability_component is not None:
            weighted_sum += w.correlation_stability * corr_stability_component
            weight_total += w.correlation_stability
        if beta_stability_component is not None:
            weighted_sum += w.beta_stability * beta_stability_component
            weight_total += w.beta_stability
        if adf_pass_rate_component is not None:
            weighted_sum += w.adf_pass_rate * adf_pass_rate_component
            weight_total += w.adf_pass_rate
        if coint_pass_rate_component is not None:
            weighted_sum += w.cointegration_rate * coint_pass_rate_component
            weight_total += w.cointegration_rate

        raw_score = (weighted_sum / weight_total) if weight_total > 0.0 else 0.5

        # Structural break penalty
        break_confidence = report.break_confidence if report.has_structural_break else 0.0
        raw_score -= w.break_penalty * break_confidence

        # Regime penalty
        raw_score -= w.regime_penalty if not report.regime_suitable else 0.0

        report.stability_score = float(min(1.0, max(0.0, raw_score)))

        logger.debug(
            "%s: stability_score=%.2f, is_stable=%s, regime=%s",
            pair_id.label, report.stability_score, report.is_stable, report.regime_label,
        )

        return report

    def analyze_batch(
        self,
        pair_ids: list[PairId],
        prices: pd.DataFrame,
        *,
        train_end: Optional[datetime] = None,
        run_rolling_coint: bool = False,  # disabled by default for speed in batch mode
    ) -> dict[str, StabilityReport]:
        """Analyze stability for multiple pairs. Returns {pair_label: StabilityReport}."""
        reports = {}
        for pid in pair_ids:
            try:
                reports[pid.label] = self.analyze(
                    pid, prices,
                    train_end=train_end,
                    run_rolling_coint=run_rolling_coint,
                )
            except Exception as exc:
                logger.warning("Stability analysis failed for %s: %s", pid.label, exc)
        return reports
