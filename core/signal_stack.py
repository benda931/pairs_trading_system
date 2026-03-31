# -*- coding: utf-8 -*-
"""
core/signal_stack.py — Multi-Layer Signal Stack for Pairs Trading
=================================================================

4-layer multiplicative scoring system (adapted from srv_quant):

  Layer 1: Spread Distortion Score (S^dist)
    - How far the spread deviates from historical norms
    - Uses z-score of spread volatility + correlation drift

  Layer 2: Pair Dislocation Score (S^disloc)
    - Current spread z-score intensity
    - Capped: S^disloc = min(1, |z| / Z_cap)

  Layer 3: Mean-Reversion Quality (S^mr)
    - Half-life quality (AR(1) sweet spot [5-90] days)
    - ADF stationarity (p-value mapping)
    - Hurst exponent (H < 0.5 = mean-reverting)

  Layer 4: Regime Safety Score (S^safe)
    - VIX penalty, correlation penalty, spread explosion penalty
    - Multiplicative: S^safe = product(1 - w_i * P_i)

  Combined conviction: Score = S^dist * S^disloc * S^mr * S^safe
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Result Dataclasses
# ============================================================================

@dataclass
class MeanReversionResult:
    """Layer 3: Mean-reversion quality per pair."""
    half_life_days: float
    half_life_quality: float  # [0, 1]
    adf_pvalue: float
    adf_quality: float  # [0, 1]
    hurst_exponent: float
    hurst_quality: float  # [0, 1]
    mr_score: float  # weighted combination [0, 1]
    label: str  # "STRONG_MR" / "MODERATE_MR" / "WEAK_MR" / "NO_MR"


@dataclass
class RegimeSafetyResult:
    """Layer 4: Regime safety assessment."""
    vix_penalty: float  # [0, 1]
    correlation_penalty: float  # [0, 1]
    spread_explosion_penalty: float  # [0, 1]
    safety_score: float  # [0, 1]
    regime_label: str  # "SAFE" / "CAUTION" / "DANGER"


@dataclass
class SignalStackResult:
    """Combined output for one pair."""
    pair_label: str
    sym_x: str
    sym_y: str

    # Layer scores
    distortion_score: float  # S^dist [0, 1]
    dislocation_score: float  # S^disloc [0, 1]
    mr_score: float  # S^mr [0, 1]
    safety_score: float  # S^safe [0, 1]

    # Combined
    conviction: float  # product of all layers
    direction: str  # "long_spread" / "short_spread" / "neutral"
    passes_entry: bool

    # Raw data
    spread_z: float
    half_life: float
    correlation: float

    # Detail objects
    mr_detail: Optional[MeanReversionResult] = None
    safety_detail: Optional[RegimeSafetyResult] = None


# ============================================================================
# Layer 1: Spread Distortion Score
# ============================================================================

def _logistic(x: float) -> float:
    """Sigmoid function."""
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def compute_distortion_score(
    spread: pd.Series,
    corr_current: float,
    corr_baseline: float,
    *,
    vol_window: int = 20,
    vol_baseline_window: int = 252,
) -> float:
    """
    Layer 1: How distorted is the spread from normal?

    Uses:
    - Spread volatility z-score (is vol higher than usual?)
    - Correlation drift (has pair correlation changed?)
    """
    if len(spread.dropna()) < vol_baseline_window:
        return 0.5  # neutral if insufficient data

    # Spread vol z-score
    vol_short = spread.rolling(vol_window).std().iloc[-1]
    vol_long = spread.rolling(vol_baseline_window).std().iloc[-1]
    vol_mean = spread.rolling(vol_baseline_window).std().mean()
    vol_std = spread.rolling(vol_baseline_window).std().std()

    vol_z = (vol_short - vol_mean) / max(vol_std, 1e-10) if np.isfinite(vol_std) and vol_std > 0 else 0.0

    # Correlation drift
    corr_drift = abs(corr_current - corr_baseline)

    # Combine via logistic
    logit_input = 0.4 * vol_z + 0.3 * corr_drift * 5.0  # scale drift
    return _logistic(logit_input)


# ============================================================================
# Layer 2: Pair Dislocation Score
# ============================================================================

def compute_dislocation_score(
    spread_z: float,
    z_cap: float = 3.0,
) -> Tuple[float, str]:
    """
    Layer 2: How dislocated is the spread right now?

    Returns (score, direction).
    """
    if not np.isfinite(spread_z):
        return 0.0, "neutral"

    score = min(1.0, abs(spread_z) / z_cap)
    direction = "long_spread" if spread_z < 0 else "short_spread" if spread_z > 0 else "neutral"
    return score, direction


# ============================================================================
# Layer 3: Mean-Reversion Quality
# ============================================================================

def _estimate_half_life(spread: pd.Series, min_obs: int = 60) -> float:
    """AR(1) half-life estimation."""
    x = spread.dropna().astype(float)
    if len(x) < min_obs:
        return float("nan")

    x_lag = x.shift(1).dropna()
    x_now = x.loc[x_lag.index]

    X = np.vstack([np.ones(len(x_lag)), x_lag.values]).T
    y = x_now.values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        phi = float(beta[1])
    except Exception:
        return float("nan")

    if phi <= 0.0 or phi >= 0.999:
        return float("nan")

    return float(-math.log(2.0) / math.log(phi))


def _half_life_quality(hl: float, sweet_lo: float = 5.0, sweet_hi: float = 90.0) -> float:
    """Map half-life to quality [0, 1]. Sweet spot [5, 90] days."""
    if not math.isfinite(hl):
        return 0.0
    if hl < 2.0:
        return 0.15
    if hl < sweet_lo:
        return 0.30 + 0.30 * (hl - 2.0) / (sweet_lo - 2.0)
    if hl <= sweet_hi:
        center = (sweet_lo + sweet_hi) / 2.0
        spread = (sweet_hi - sweet_lo) / 2.0
        return 0.65 + 0.35 * math.exp(-((hl - center) ** 2) / (2 * spread ** 2))
    if hl <= 180:
        return max(0.20, 0.60 * math.exp(-0.01 * (hl - sweet_hi)))
    return 0.10


def _adf_quality(p_value: float) -> float:
    """Map ADF p-value to quality [0, 1]."""
    if not math.isfinite(p_value):
        return 0.0
    if p_value <= 0.01:
        return 1.0
    if p_value <= 0.05:
        return 0.70 + 0.30 * (0.05 - p_value) / 0.04
    if p_value <= 0.10:
        return 0.45 + 0.25 * (0.10 - p_value) / 0.05
    if p_value <= 0.30:
        return 0.15 + 0.30 * (0.30 - p_value) / 0.20
    return max(0.0, 0.15 * (1.0 - p_value))


def _hurst_quality(h: float) -> float:
    """Map Hurst exponent to quality [0, 1]. H < 0.5 = mean-reverting."""
    if not math.isfinite(h):
        return 0.0
    if h <= 0.3:
        return 1.0
    if h <= 0.5:
        return 0.5 + 0.5 * (0.5 - h) / 0.2
    if h <= 0.6:
        return 0.3 * (0.6 - h) / 0.1
    return 0.0


def _hurst_exponent(series: pd.Series, min_window: int = 20, max_window: int = 200) -> float:
    """Hurst exponent via rescaled range (R/S) analysis."""
    x = series.dropna().values
    n = len(x)
    if n < min_window * 2:
        return float("nan")

    lags = []
    rs_values = []
    window = min_window
    while window <= min(max_window, n // 2):
        rs_list = []
        for start in range(0, n - window + 1, window):
            segment = x[start:start + window]
            mean_seg = np.mean(segment)
            deviate = np.cumsum(segment - mean_seg)
            r = np.max(deviate) - np.min(deviate)
            s = np.std(segment, ddof=1)
            if s > 1e-10:
                rs_list.append(r / s)
        if rs_list:
            lags.append(math.log(window))
            rs_values.append(math.log(np.mean(rs_list)))
        window = int(window * 1.5)

    if len(lags) < 3:
        return float("nan")

    try:
        coeffs = np.polyfit(lags, rs_values, 1)
        return float(coeffs[0])
    except Exception:
        return float("nan")


def compute_mean_reversion_score(
    spread: pd.Series,
    *,
    w_hl: float = 0.35,
    w_adf: float = 0.40,
    w_hurst: float = 0.25,
) -> MeanReversionResult:
    """Layer 3: Compute mean-reversion quality for a spread."""
    hl = _estimate_half_life(spread)
    hl_q = _half_life_quality(hl)

    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(spread.dropna().values, maxlag=12, autolag="AIC")
        adf_p = float(result[1])
    except Exception:
        adf_p = 1.0
    adf_q = _adf_quality(adf_p)

    hurst = _hurst_exponent(spread)
    hurst_q = _hurst_quality(hurst)

    mr_score = w_hl * hl_q + w_adf * adf_q + w_hurst * hurst_q

    if mr_score >= 0.7:
        label = "STRONG_MR"
    elif mr_score >= 0.4:
        label = "MODERATE_MR"
    elif mr_score >= 0.2:
        label = "WEAK_MR"
    else:
        label = "NO_MR"

    return MeanReversionResult(
        half_life_days=hl if math.isfinite(hl) else -1.0,
        half_life_quality=hl_q,
        adf_pvalue=adf_p,
        adf_quality=adf_q,
        hurst_exponent=hurst if math.isfinite(hurst) else -1.0,
        hurst_quality=hurst_q,
        mr_score=mr_score,
        label=label,
    )


# ============================================================================
# Layer 4: Regime Safety Score
# ============================================================================

def compute_regime_safety(
    vix_level: float = 15.0,
    vix_percentile: float = 0.5,
    avg_pair_corr: float = 0.7,
    spread_z_max_20d: float = 2.0,
    *,
    vix_soft: float = 0.75,  # percentile thresholds
    vix_hard: float = 0.95,
    vix_kill: float = 0.99,
    w_vix: float = 0.40,
    w_corr: float = 0.30,
    w_spread: float = 0.30,
) -> RegimeSafetyResult:
    """Layer 4: Regime safety gating."""
    # VIX penalty
    if vix_percentile >= vix_kill:
        vix_pen = 1.0
    elif vix_percentile >= vix_hard:
        vix_pen = 0.5 + 0.5 * (vix_percentile - vix_hard) / (vix_kill - vix_hard)
    elif vix_percentile >= vix_soft:
        vix_pen = 0.3 * (vix_percentile - vix_soft) / (vix_hard - vix_soft)
    else:
        vix_pen = 0.0

    # Correlation penalty (high avg corr = flight to quality)
    corr_pen = max(0.0, min(1.0, (avg_pair_corr - 0.6) / 0.3)) if avg_pair_corr > 0.6 else 0.0

    # Spread explosion penalty
    spread_pen = max(0.0, min(1.0, (spread_z_max_20d - 2.5) / 2.0)) if spread_z_max_20d > 2.5 else 0.0

    # Multiplicative safety
    safety = (1.0 - w_vix * vix_pen) * (1.0 - w_corr * corr_pen) * (1.0 - w_spread * spread_pen)
    safety = max(0.0, min(1.0, safety))

    if safety >= 0.7:
        regime_label = "SAFE"
    elif safety >= 0.3:
        regime_label = "CAUTION"
    else:
        regime_label = "DANGER"

    return RegimeSafetyResult(
        vix_penalty=vix_pen,
        correlation_penalty=corr_pen,
        spread_explosion_penalty=spread_pen,
        safety_score=safety,
        regime_label=regime_label,
    )


# ============================================================================
# Combined Signal Stack
# ============================================================================

def compute_pair_conviction(
    pair_label: str,
    sym_x: str,
    sym_y: str,
    spread: pd.Series,
    spread_z: float,
    correlation: float,
    corr_baseline: float = 0.7,
    *,
    entry_threshold: float = 0.05,
    z_cap: float = 3.0,
    vix_level: float = 15.0,
    vix_percentile: float = 0.5,
) -> SignalStackResult:
    """
    Compute full 4-layer conviction score for a pair.

    Returns SignalStackResult with conviction = S^dist * S^disloc * S^mr * S^safe.
    """
    # Layer 1: Distortion
    s_dist = compute_distortion_score(
        spread, correlation, corr_baseline,
    )

    # Layer 2: Dislocation
    s_disloc, direction = compute_dislocation_score(spread_z, z_cap)

    # Layer 3: Mean-Reversion Quality
    mr_result = compute_mean_reversion_score(spread)
    s_mr = mr_result.mr_score

    # Layer 4: Regime Safety
    spread_z_max_20d = float(spread.rolling(20).apply(
        lambda x: abs((x.iloc[-1] - x.mean()) / max(x.std(), 1e-10))
    ).iloc[-1]) if len(spread) >= 20 else abs(spread_z)

    safety_result = compute_regime_safety(
        vix_level=vix_level,
        vix_percentile=vix_percentile,
        avg_pair_corr=abs(correlation),
        spread_z_max_20d=spread_z_max_20d,
    )
    s_safe = safety_result.safety_score

    # Combined conviction
    conviction = s_dist * s_disloc * s_mr * s_safe
    passes = conviction >= entry_threshold

    return SignalStackResult(
        pair_label=pair_label,
        sym_x=sym_x,
        sym_y=sym_y,
        distortion_score=round(s_dist, 4),
        dislocation_score=round(s_disloc, 4),
        mr_score=round(s_mr, 4),
        safety_score=round(s_safe, 4),
        conviction=round(conviction, 4),
        direction=direction,
        passes_entry=passes,
        spread_z=round(spread_z, 4),
        half_life=mr_result.half_life_days,
        correlation=round(correlation, 4),
        mr_detail=mr_result,
        safety_detail=safety_result,
    )


def score_universe(
    pairs: List[Dict[str, Any]],
    prices: pd.DataFrame,
    *,
    window: int = 60,
    entry_threshold: float = 0.05,
    vix_level: float = 15.0,
    vix_percentile: float = 0.5,
) -> List[SignalStackResult]:
    """
    Score all pairs in universe using the 4-layer signal stack.

    Parameters
    ----------
    pairs : list of dicts with keys 'sym_x', 'sym_y', 'pair_label'
    prices : wide DataFrame (date index, symbol columns)
    window : rolling window for z-score
    """
    results = []

    for pair in pairs:
        sym_x = pair.get("sym_x", "")
        sym_y = pair.get("sym_y", "")
        label = pair.get("pair_label", f"{sym_y}-{sym_x}")

        if sym_x not in prices.columns or sym_y not in prices.columns:
            continue

        try:
            px = prices[sym_x].dropna()
            py = prices[sym_y].dropna()
            common = px.index.intersection(py.index)
            if len(common) < window + 20:
                continue

            px = px.loc[common]
            py = py.loc[common]

            # Build spread (simple: Y - beta * X)
            beta = py.rolling(window).cov(px) / px.rolling(window).var()
            beta = beta.fillna(method="ffill").fillna(1.0)
            spread = py - beta * px

            # Z-score
            spread_mean = spread.rolling(window).mean()
            spread_std = spread.rolling(window).std()
            z = (spread - spread_mean) / spread_std.replace(0, np.nan)
            z_last = float(z.dropna().iloc[-1]) if not z.dropna().empty else 0.0

            # Correlation
            corr = float(px.rolling(window).corr(py).iloc[-1])
            corr_base = float(px.rolling(252).corr(py).iloc[-1]) if len(common) >= 252 else corr

            result = compute_pair_conviction(
                pair_label=label,
                sym_x=sym_x,
                sym_y=sym_y,
                spread=spread.dropna(),
                spread_z=z_last,
                correlation=corr,
                corr_baseline=corr_base,
                entry_threshold=entry_threshold,
                vix_level=vix_level,
                vix_percentile=vix_percentile,
            )
            results.append(result)

        except Exception as e:
            logger.warning("Signal stack failed for %s: %s", label, e)
            continue

    # Sort by conviction descending
    results.sort(key=lambda r: r.conviction, reverse=True)
    return results
