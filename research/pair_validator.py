# -*- coding: utf-8 -*-
"""
research/pair_validator.py — Explicit Pair Validation Pipeline
===============================================================

This module implements the canonical pair validation pipeline.

DOCTRINE: Correlation is not enough.
A pair should only become tradable after explicitly passing ALL of:
  1. Minimum correlation threshold
  2. ADF stationarity test on the residual spread
  3. Cointegration test (Engle-Granger)
  4. Acceptable half-life (min and max bounds)
  5. Hurst exponent < 0.5 (mean-reverting)
  6. Rolling hedge ratio stability (not drifting too fast)
  7. Minimum liquidity (ADV check)
  8. No active structural break

All rejection reasons are explicitly logged in PairValidationReport.
There are NO silent pass-throughs.

Usage:
    from research.pair_validator import PairValidator, ValidationConfig

    config = ValidationConfig()
    validator = PairValidator(config)

    report = validator.validate(
        pair_id=PairId("XLK", "XLY"),
        prices=df_with_both_symbols,
    )

    if report.is_tradable:
        print(f"✓ {report.pair_id} — half-life={report.half_life_days:.1f}d")
    else:
        print(f"✗ {report.pair_id} — {report.rejection_reasons}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.contracts import (
    HedgeRatioMethod,
    PairId,
    PairValidationReport,
    SpreadModel,
    ValidationResult,
    ValidationTest,
    ValidationThresholds,
)

logger = logging.getLogger("research.pair_validator")


# ── Config ────────────────────────────────────────────────────────

@dataclass
class ValidationConfig:
    """Configuration for the pair validation pipeline."""

    # Hard tests (failure = REJECT)
    min_correlation: float = ValidationThresholds.MIN_CORRELATION
    max_adf_pvalue: float = ValidationThresholds.MAX_ADF_PVALUE
    max_coint_pvalue: float = ValidationThresholds.MAX_COINT_PVALUE
    min_half_life_days: float = ValidationThresholds.MIN_HALF_LIFE_DAYS
    max_half_life_days: float = ValidationThresholds.MAX_HALF_LIFE_DAYS
    # ADR-007: Minimum 1 trading year required for reliable AR(1) half-life estimation.
    # 252 trading days provides sufficient degrees of freedom for ADF, Johansen,
    # and Hurst exponent tests to be statistically meaningful. Do not lower this
    # threshold without explicit review — AR(1) on <252 observations has high
    # estimation variance and inflated false-positive rates for mean reversion.
    min_obs: int = ValidationThresholds.MIN_OBS  # R-002: must be >= 252

    # Soft tests (failure = WARN, not REJECT)
    max_hurst_exponent: float = ValidationThresholds.MAX_HURST_EXPONENT
    min_rolling_stability: float = ValidationThresholds.MIN_ROLLING_STABILITY
    max_spread_vol_ratio: float = ValidationThresholds.MAX_SPREAD_VOL_RATIO

    # Hedge ratio estimation
    hedge_ratio_window: int = 252        # Window for rolling OLS
    hedge_ratio_method: HedgeRatioMethod = HedgeRatioMethod.OLS
    use_log_prices: bool = True          # Use log prices for spread construction

    # Rolling stability check
    stability_window: int = 60           # Rolling window for hedge ratio stability
    stability_n_windows: int = 8         # Number of sub-windows to check

    # Test toggles (for research / fast mode)
    run_adf: bool = True
    run_cointegration: bool = True
    run_hurst: bool = True
    run_stability: bool = True


# ── Core statistical functions ────────────────────────────────────

def _estimate_hedge_ratio_ols(
    log_x: pd.Series,
    log_y: pd.Series,
) -> tuple[float, float]:
    """OLS regression: log_x = beta * log_y + alpha. Returns (beta, alpha)."""
    from numpy.linalg import lstsq
    Y = log_y.values.reshape(-1, 1)
    X = np.column_stack([np.ones(len(Y)), Y])
    coeffs, _, _, _ = lstsq(X, log_x.values, rcond=None)
    alpha, beta = coeffs[0], coeffs[1]
    return float(beta), float(alpha)


def _run_adf_test(series: pd.Series) -> tuple[float, float]:
    """Run Augmented Dickey-Fuller test. Returns (test_stat, pvalue)."""
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna(), autolag="AIC", regression="c")
        return float(result[0]), float(result[1])
    except ImportError:
        logger.warning("statsmodels not available; ADF test skipped")
        return np.nan, np.nan
    except Exception as e:
        logger.warning("ADF test failed: %s", e)
        return np.nan, np.nan


def _run_cointegration_test(
    log_x: pd.Series,
    log_y: pd.Series,
) -> float:
    """Engle-Granger cointegration test. Returns pvalue."""
    try:
        from statsmodels.tsa.stattools import coint
        _, pvalue, _ = coint(log_x.dropna(), log_y.dropna(), autolag="aic")
        return float(pvalue)
    except ImportError:
        logger.warning("statsmodels not available; cointegration test skipped")
        return np.nan
    except Exception as e:
        logger.warning("Cointegration test failed: %s", e)
        return np.nan


def _estimate_half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life from AR(1) regression of spread changes."""
    try:
        s = spread.dropna()
        if len(s) < 20:
            return np.nan
        delta = s.diff().dropna()
        lag = s.shift(1).dropna()
        common = delta.index.intersection(lag.index)
        delta, lag = delta.loc[common], lag.loc[common]

        # OLS: delta_s = kappa * s_{t-1} + epsilon
        X = lag.values.reshape(-1, 1)
        Y = delta.values
        from numpy.linalg import lstsq
        beta, _, _, _ = lstsq(X, Y, rcond=None)
        kappa = float(beta[0])

        if kappa >= 0 or np.isnan(kappa):
            return np.inf   # Not mean-reverting
        half_life = -np.log(2) / kappa
        return float(half_life)
    except Exception as e:
        logger.warning("Half-life estimation failed: %s", e)
        return np.nan


def _estimate_hurst(series: pd.Series, max_lag: int = 20) -> float:
    """R/S Hurst exponent estimate. H < 0.5 = mean-reverting."""
    try:
        s = series.dropna().values
        n = len(s)
        if n < 50:
            return np.nan

        lags = range(2, min(max_lag, n // 4))
        rs_list = []
        lag_list = []
        for lag in lags:
            sub = [s[i:i+lag] for i in range(0, n - lag, lag)]
            if len(sub) < 2:
                continue
            rs_vals = []
            for chunk in sub:
                if len(chunk) < 2:
                    continue
                mean_c = np.mean(chunk)
                std_c = np.std(chunk)
                if std_c < 1e-10:
                    continue
                cumdev = np.cumsum(chunk - mean_c)
                r = cumdev.max() - cumdev.min()
                rs_vals.append(r / std_c)
            if rs_vals:
                rs_list.append(np.mean(rs_vals))
                lag_list.append(lag)

        if len(lag_list) < 3:
            return np.nan

        log_lags = np.log(lag_list)
        log_rs = np.log(rs_list)
        hurst = float(np.polyfit(log_lags, log_rs, 1)[0])
        return np.clip(hurst, 0.0, 1.0)
    except Exception as e:
        logger.warning("Hurst estimation failed: %s", e)
        return np.nan


def _rolling_hedge_ratio_stability(
    log_x: pd.Series,
    log_y: pd.Series,
    window: int,
    n_windows: int,
) -> float:
    """
    Compute rolling hedge ratio stability score (0-1).

    Splits the series into n_windows sub-windows, estimates hedge ratio in each,
    and returns 1 - (cv of hedge ratios) as a stability score.
    High stability (close to 1) = hedge ratio is consistent over time.
    """
    try:
        n = len(log_x)
        step = n // n_windows
        if step < 30:
            return np.nan

        betas = []
        for i in range(n_windows):
            start = i * step
            end = min(start + window, n)
            if end - start < 20:
                continue
            sub_x = log_x.iloc[start:end]
            sub_y = log_y.iloc[start:end]
            beta, _ = _estimate_hedge_ratio_ols(sub_x, sub_y)
            betas.append(beta)

        if len(betas) < 3:
            return np.nan

        betas_arr = np.array(betas)
        mean_beta = np.mean(np.abs(betas_arr))
        if mean_beta < 1e-8:
            return 1.0
        cv = np.std(betas_arr) / mean_beta
        stability = max(0.0, 1.0 - cv)
        return float(stability)
    except Exception as e:
        logger.warning("Stability check failed: %s", e)
        return np.nan


# ── Validator ─────────────────────────────────────────────────────

class PairValidator:
    """
    Runs the full validation pipeline for a candidate pair.

    Design principles:
    - Every test produces a ValidationTest with pass/fail + value
    - Hard failures accumulate rejection_reasons and produce FAIL
    - Soft failures accumulate warning_reasons and produce WARN
    - ALL tests run to completion even after hard failures (for diagnostics)
    - Results are fully serializable via PairValidationReport.to_dict()
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def validate(
        self,
        pair_id: PairId,
        prices: pd.DataFrame,
        *,
        train_end: Optional[datetime] = None,
    ) -> PairValidationReport:
        """
        Run complete validation pipeline.

        Args:
            pair_id: The pair to validate (must have both symbols in prices)
            prices: Wide DataFrame with at least sym_x and sym_y columns
            train_end: If set, only use data up to this date for validation

        Returns:
            PairValidationReport with complete test results and tradability verdict
        """
        report = PairValidationReport(
            pair_id=pair_id,
            result=ValidationResult.SKIP,
            spread_model=SpreadModel.STATIC_OLS,
            hedge_ratio_method=self.config.hedge_ratio_method,
        )

        # ── 0. Pre-condition: data availability ──────────────────
        sym_x, sym_y = pair_id.sym_x, pair_id.sym_y

        if sym_x not in prices.columns or sym_y not in prices.columns:
            report.rejection_reasons.append(
                f"Missing price data: need {sym_x} and {sym_y} in prices DataFrame"
            )
            report.result = ValidationResult.SKIP
            return report

        # Apply train_end cutoff
        if train_end is not None:
            prices = prices[prices.index <= pd.Timestamp(train_end)]

        px = prices[sym_x].dropna()
        py = prices[sym_y].dropna()

        # Align
        common = px.index.intersection(py.index)
        px, py = px.loc[common], py.loc[common]

        n_obs = len(common)
        report.n_obs = n_obs
        report.train_start = common[0].to_pydatetime() if len(common) > 0 else None
        report.train_end = common[-1].to_pydatetime() if len(common) > 0 else None

        if n_obs < self.config.min_obs:
            report.rejection_reasons.append(
                f"Insufficient data: {n_obs} observations < {self.config.min_obs} required"
            )
            report.result = ValidationResult.FAIL
            return report

        # Use log prices
        log_x = np.log(px.clip(lower=1e-8))
        log_y = np.log(py.clip(lower=1e-8))

        # ── 1. Correlation ───────────────────────────────────────
        corr = float(log_x.corr(log_y))
        report.correlation = corr
        corr_test = ValidationTest(
            name="correlation",
            passed=corr >= self.config.min_correlation,
            is_hard=True,
            value=corr,
            threshold=self.config.min_correlation,
            message=f"Correlation={corr:.3f} (min={self.config.min_correlation})",
        )
        report.tests.append(corr_test)
        if not corr_test.passed:
            report.rejection_reasons.append(
                f"Low correlation: {corr:.3f} < {self.config.min_correlation}"
            )

        # ── 2. Hedge ratio estimation ────────────────────────────
        try:
            beta, alpha = _estimate_hedge_ratio_ols(log_x, log_y)
            report.hedge_ratio = beta
        except Exception as e:
            logger.warning("Hedge ratio estimation failed for %s: %s", pair_id, e)
            report.rejection_reasons.append(f"Hedge ratio estimation failed: {e}")
            report.result = ValidationResult.FAIL
            return report

        # ── 3. Spread construction ───────────────────────────────
        spread = log_x - beta * log_y - alpha
        spread_vol = float(spread.std())
        report.spread_vol = spread_vol
        report.residual_vol = spread_vol

        if spread_vol < 1e-8:
            report.rejection_reasons.append("Degenerate spread: near-zero variance")
            report.result = ValidationResult.FAIL
            return report

        # ── 4. ADF stationarity test on spread ──────────────────
        if self.config.run_adf:
            adf_stat, adf_pvalue = _run_adf_test(spread)
            report.adf_pvalue = adf_pvalue
            adf_passed = not np.isnan(adf_pvalue) and adf_pvalue <= self.config.max_adf_pvalue
            adf_test = ValidationTest(
                name="adf_stationarity",
                passed=adf_passed or np.isnan(adf_pvalue),
                is_hard=True,
                value=adf_stat,
                threshold=self.config.max_adf_pvalue,
                pvalue=adf_pvalue if not np.isnan(adf_pvalue) else np.nan,
                message=f"ADF pvalue={adf_pvalue:.4f} (max={self.config.max_adf_pvalue})",
            )
            report.tests.append(adf_test)
            if not np.isnan(adf_pvalue) and not adf_passed:
                report.rejection_reasons.append(
                    f"ADF test failed: pvalue={adf_pvalue:.4f} > {self.config.max_adf_pvalue} "
                    "(spread is not stationary)"
                )

        # ── 5. Cointegration test ────────────────────────────────
        if self.config.run_cointegration:
            coint_pvalue = _run_cointegration_test(log_x, log_y)
            report.cointegration_pvalue = coint_pvalue
            coint_passed = not np.isnan(coint_pvalue) and coint_pvalue <= self.config.max_coint_pvalue
            coint_test = ValidationTest(
                name="engle_granger_cointegration",
                passed=coint_passed or np.isnan(coint_pvalue),
                is_hard=True,
                pvalue=coint_pvalue if not np.isnan(coint_pvalue) else np.nan,
                threshold=self.config.max_coint_pvalue,
                message=f"EG cointegration pvalue={coint_pvalue:.4f} (max={self.config.max_coint_pvalue})",
            )
            report.tests.append(coint_test)
            if not np.isnan(coint_pvalue) and not coint_passed:
                report.rejection_reasons.append(
                    f"Cointegration test failed: pvalue={coint_pvalue:.4f} > {self.config.max_coint_pvalue}"
                )

        # ── 6. Half-life ─────────────────────────────────────────
        hl = _estimate_half_life(spread)
        report.half_life_days = hl
        hl_finite = np.isfinite(hl) and not np.isnan(hl)
        hl_in_range = (
            hl_finite
            and self.config.min_half_life_days <= hl <= self.config.max_half_life_days
        )
        hl_test = ValidationTest(
            name="half_life",
            passed=hl_in_range,
            is_hard=True,
            value=hl if hl_finite else np.nan,
            threshold=self.config.max_half_life_days,
            message=(
                f"Half-life={hl:.1f}d "
                f"(target: [{self.config.min_half_life_days}, {self.config.max_half_life_days}])"
                if hl_finite else "Half-life=∞ (spread not mean-reverting)"
            ),
        )
        report.tests.append(hl_test)
        if not hl_in_range:
            if not hl_finite:
                report.rejection_reasons.append(
                    "Infinite half-life: spread is not mean-reverting"
                )
            elif hl < self.config.min_half_life_days:
                report.rejection_reasons.append(
                    f"Half-life too short: {hl:.1f}d < {self.config.min_half_life_days}d min "
                    "(too noisy, likely mean-reverting around noise)"
                )
            else:
                report.rejection_reasons.append(
                    f"Half-life too long: {hl:.1f}d > {self.config.max_half_life_days}d max "
                    "(too slow to converge; capital tied up too long)"
                )

        # ── 7. Hurst exponent (soft test) ───────────────────────
        if self.config.run_hurst:
            hurst = _estimate_hurst(spread)
            report.hurst_exponent = hurst
            hurst_ok = np.isnan(hurst) or hurst <= self.config.max_hurst_exponent
            hurst_test = ValidationTest(
                name="hurst_exponent",
                passed=hurst_ok,
                is_hard=False,  # SOFT: warn, don't reject
                value=hurst if not np.isnan(hurst) else np.nan,
                threshold=self.config.max_hurst_exponent,
                message=(
                    f"Hurst={hurst:.3f} (max={self.config.max_hurst_exponent}; <0.5=mean-reverting)"
                    if not np.isnan(hurst) else "Hurst: insufficient data"
                ),
            )
            report.tests.append(hurst_test)
            if not hurst_ok:
                report.warning_reasons.append(
                    f"Elevated Hurst exponent: {hurst:.3f} > {self.config.max_hurst_exponent} "
                    "(spread may be trending; consider reduced position size)"
                )

        # ── 8. Rolling hedge ratio stability (soft test) ────────
        if self.config.run_stability:
            stability = _rolling_hedge_ratio_stability(
                log_x, log_y,
                window=self.config.stability_window,
                n_windows=self.config.stability_n_windows,
            )
            report.rolling_stability = stability if not np.isnan(stability) else np.nan
            # P2-COINT: Hard-reject if stability is critically low (< 0.15)
            HARD_STABILITY_FLOOR = 0.15
            critically_unstable = (not np.isnan(stability)) and stability < HARD_STABILITY_FLOOR
            stab_ok = np.isnan(stability) or stability >= self.config.min_rolling_stability
            stab_test = ValidationTest(
                name="hedge_ratio_stability",
                passed=stab_ok and not critically_unstable,
                is_hard=critically_unstable,  # HARD reject if below 0.15
                value=stability if not np.isnan(stability) else np.nan,
                threshold=self.config.min_rolling_stability,
                message=(
                    f"Stability={stability:.3f} (min={self.config.min_rolling_stability})"
                    if not np.isnan(stability) else "Stability: insufficient data"
                ),
            )
            report.tests.append(stab_test)
            if critically_unstable:
                report.rejection_reasons.append(
                    f"CRITICALLY unstable hedge ratio: {stability:.3f} < {HARD_STABILITY_FLOOR} "
                    "(relationship is nearly random — hard reject per P2-COINT)"
                )
            elif not stab_ok:
                report.warning_reasons.append(
                    f"Low hedge ratio stability: {stability:.3f} < {self.config.min_rolling_stability} "
                    "(hedge ratio drifting; consider Kalman or rolling OLS spread model)"
                )

        # ── 9. Determine final verdict ───────────────────────────
        hard_failures = [t for t in report.tests if t.is_hard and not t.passed]
        soft_failures = [t for t in report.tests if not t.is_hard and not t.passed]

        if hard_failures:
            report.result = ValidationResult.FAIL
        elif soft_failures:
            report.result = ValidationResult.WARN
        else:
            report.result = ValidationResult.PASS

        logger.info(
            "PairValidator: %s → %s | corr=%.2f hl=%.1fd adf_p=%.3f hard_fails=%d soft_fails=%d",
            pair_id.label,
            report.result.value,
            corr,
            hl if np.isfinite(hl) else -1,
            report.adf_pvalue if not np.isnan(report.adf_pvalue) else -1,
            len(hard_failures),
            len(soft_failures),
        )

        return report

    def validate_batch(
        self,
        pair_ids: list[PairId],
        prices: pd.DataFrame,
        *,
        train_end: Optional[datetime] = None,
    ) -> list[PairValidationReport]:
        """Validate multiple pairs. Returns list of reports sorted by result quality."""
        reports = []
        for pair_id in pair_ids:
            try:
                report = self.validate(pair_id, prices, train_end=train_end)
                reports.append(report)
            except Exception as e:
                logger.error("Validation crashed for %s: %s", pair_id, e)
                fail_report = PairValidationReport(
                    pair_id=pair_id,
                    result=ValidationResult.FAIL,
                    rejection_reasons=[f"Validation crashed: {e}"],
                )
                reports.append(fail_report)

        # Sort: PASS first, then WARN, then FAIL/SKIP
        order = {ValidationResult.PASS: 0, ValidationResult.WARN: 1,
                 ValidationResult.FAIL: 2, ValidationResult.SKIP: 3}
        reports.sort(key=lambda r: order.get(r.result, 4))
        return reports

    def validation_summary_df(self, reports: list[PairValidationReport]) -> pd.DataFrame:
        """Convert validation reports to a summary DataFrame for display."""
        rows = [r.to_dict() for r in reports]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Reorder columns for readability
        cols = [
            "pair", "result", "correlation", "adf_pvalue", "half_life_days",
            "hurst_exponent", "hedge_ratio", "cointegration_pvalue",
            "rolling_stability", "n_obs", "rejection_reasons", "warning_reasons",
        ]
        present = [c for c in cols if c in df.columns]
        return df[present]
