# -*- coding: utf-8 -*-
"""
core/correlation_monitor.py — Correlation Regime Monitor
==========================================================

Real-time correlation monitoring with structural break detection:

1. **Rolling Correlation Tracking**
   - Multiple windows (21d, 63d, 126d, 252d)
   - EWMA correlation (exponentially weighted)

2. **Break Detection**
   - CUSUM test for correlation shifts
   - Z-score deviation from long-run correlation
   - Regime change alerts

3. **Pair Health Scoring**
   - Correlation stability index
   - Divergence warning system
   - Pair kill recommendations

4. **Portfolio Correlation Matrix**
   - Eigenvalue decomposition
   - Effective number of bets
   - Concentration risk metrics

Usage:
    from core.correlation_monitor import CorrelationMonitor

    cm = CorrelationMonitor()
    health = cm.check_pair_health(prices_x, prices_y)
    if health.has_break:
        print(f"ALERT: Correlation break detected! ({health.alert_message})")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationSnapshot:
    """Multi-window correlation snapshot for a pair."""
    pair: str
    as_of: str
    corr_21d: float
    corr_63d: float
    corr_126d: float
    corr_252d: float
    corr_ewma: float                     # EWMA correlation
    corr_full_history: float             # All-time correlation
    corr_zscore: float                   # Current vs historical distribution
    is_stable: bool                      # Short-term ≈ long-term
    trend: str                           # "RISING" / "STABLE" / "FALLING" / "BROKEN"


@dataclass
class BreakDetectionResult:
    """Result of correlation break detection."""
    has_break: bool
    break_type: str                      # "CUSUM" / "ZSCORE" / "REGIME_SHIFT" / "NONE"
    break_severity: str                  # "MINOR" / "MAJOR" / "CRITICAL"
    break_date: Optional[str] = None
    cusum_stat: float = 0.0
    cusum_threshold: float = 0.0
    zscore_deviation: float = 0.0
    confidence: float = 0.0              # [0, 1]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairHealthReport:
    """Complete health assessment for a pair."""
    sym_x: str
    sym_y: str
    n_observations: int

    # Correlation
    correlation_snapshot: Optional[CorrelationSnapshot] = None

    # Break detection
    break_detection: Optional[BreakDetectionResult] = None
    has_break: bool = False

    # Health scoring
    health_score: float = 1.0            # [0, 1], 1 = healthy
    stability_index: float = 1.0         # Std of rolling corr (lower = more stable)
    divergence_risk: str = "LOW"         # "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"

    # Recommendations
    action: str = "HOLD"                 # "HOLD" / "REDUCE" / "EXIT" / "MONITOR"
    alert_message: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class PortfolioCorrelationReport:
    """Portfolio-level correlation analysis."""
    n_pairs: int
    avg_correlation: float
    median_correlation: float
    min_correlation: float
    max_correlation: float
    effective_n_bets: float              # Eigenvalue-based diversification
    concentration_ratio: float           # Top eigenvalue / sum
    n_breaks_detected: int
    n_pairs_at_risk: int
    pair_health: Dict[str, PairHealthReport] = field(default_factory=dict)


class CorrelationMonitor:
    """
    Real-time correlation monitoring and break detection engine.
    """

    def __init__(
        self,
        ewma_halflife: int = 30,
        cusum_threshold: float = 3.0,
        zscore_alert_threshold: float = 2.5,
        min_stable_corr: float = 0.50,
        break_lookback: int = 252,
    ):
        self.ewma_halflife = ewma_halflife
        self.cusum_threshold = cusum_threshold
        self.zscore_alert = zscore_alert_threshold
        self.min_stable_corr = min_stable_corr
        self.break_lookback = break_lookback

    # ── Pair Health ───────────────────────────────────────────

    def check_pair_health(
        self,
        prices_x: pd.Series,
        prices_y: pd.Series,
        sym_x: str = "X",
        sym_y: str = "Y",
    ) -> PairHealthReport:
        """Run complete pair health check."""
        from datetime import datetime, timezone

        report = PairHealthReport(
            sym_x=sym_x, sym_y=sym_y,
            n_observations=min(len(prices_x), len(prices_y)),
        )

        try:
            # Align
            common = prices_x.index.intersection(prices_y.index)
            px = prices_x.loc[common].dropna()
            py = prices_y.loc[common].dropna()
            common = px.index.intersection(py.index)
            px, py = px.loc[common], py.loc[common]

            if len(px) < 63:
                report.warnings.append("Insufficient data for health check")
                report.health_score = 0.3
                return report

            report.n_observations = len(px)

            # Returns for correlation
            rx = px.pct_change().dropna()
            ry = py.pct_change().dropna()
            common_r = rx.index.intersection(ry.index)
            rx, ry = rx.loc[common_r], ry.loc[common_r]

            # Correlation snapshot
            report.correlation_snapshot = self._correlation_snapshot(
                rx, ry, f"{sym_x}/{sym_y}",
            )

            # Break detection
            report.break_detection = self._detect_break(rx, ry)
            report.has_break = report.break_detection.has_break if report.break_detection else False

            # Health scoring
            report.health_score = self._compute_health_score(report)
            report.stability_index = self._stability_index(rx, ry)

            # Divergence risk
            if report.has_break and report.break_detection.break_severity == "CRITICAL":
                report.divergence_risk = "CRITICAL"
                report.action = "EXIT"
                report.alert_message = f"Critical correlation break in {sym_x}/{sym_y}"
            elif report.has_break and report.break_detection.break_severity == "MAJOR":
                report.divergence_risk = "HIGH"
                report.action = "REDUCE"
                report.alert_message = f"Major correlation shift in {sym_x}/{sym_y}"
            elif report.correlation_snapshot and report.correlation_snapshot.corr_21d < self.min_stable_corr:
                report.divergence_risk = "MEDIUM"
                report.action = "MONITOR"
                report.alert_message = f"Short-term correlation weakening in {sym_x}/{sym_y}"
            else:
                report.divergence_risk = "LOW"
                report.action = "HOLD"

        except Exception as exc:
            logger.warning("Pair health check failed for %s/%s: %s", sym_x, sym_y, exc)
            report.warnings.append(f"Health check error: {exc}")
            report.health_score = 0.5

        return report

    # ── Portfolio Monitoring ──────────────────────────────────

    def monitor_portfolio(
        self,
        prices: pd.DataFrame,
        pairs: List[Tuple[str, str]],
    ) -> PortfolioCorrelationReport:
        """Monitor correlation health across all portfolio pairs."""
        pair_health: Dict[str, PairHealthReport] = {}
        n_breaks = 0
        n_at_risk = 0
        correlations = []

        for sym_x, sym_y in pairs:
            if sym_x not in prices.columns or sym_y not in prices.columns:
                continue

            health = self.check_pair_health(
                prices[sym_x], prices[sym_y], sym_x, sym_y,
            )
            key = f"{sym_x}/{sym_y}"
            pair_health[key] = health

            if health.has_break:
                n_breaks += 1
            if health.divergence_risk in ("HIGH", "CRITICAL"):
                n_at_risk += 1

            if health.correlation_snapshot:
                correlations.append(health.correlation_snapshot.corr_63d)

        # Portfolio-level metrics
        corr_array = np.array(correlations) if correlations else np.array([0.0])

        # Effective N bets (from eigenvalue decomposition of return correlations)
        eff_n = self._effective_n_bets(prices, [s for pair in pairs for s in pair])

        return PortfolioCorrelationReport(
            n_pairs=len(pairs),
            avg_correlation=round(float(np.mean(corr_array)), 4),
            median_correlation=round(float(np.median(corr_array)), 4),
            min_correlation=round(float(np.min(corr_array)), 4),
            max_correlation=round(float(np.max(corr_array)), 4),
            effective_n_bets=round(eff_n, 2),
            concentration_ratio=round(1.0 / max(eff_n, 0.01), 4),
            n_breaks_detected=n_breaks,
            n_pairs_at_risk=n_at_risk,
            pair_health=pair_health,
        )

    # ── Correlation Snapshot ──────────────────────────────────

    def _correlation_snapshot(
        self, rx: pd.Series, ry: pd.Series, pair_name: str,
    ) -> CorrelationSnapshot:
        """Multi-window correlation snapshot."""
        from datetime import datetime, timezone
        n = len(rx)

        def safe_corr(window):
            if n < window:
                return float("nan")
            return float(rx.iloc[-window:].corr(ry.iloc[-window:]))

        corr_21 = safe_corr(21)
        corr_63 = safe_corr(63)
        corr_126 = safe_corr(126)
        corr_252 = safe_corr(252)
        corr_full = float(rx.corr(ry))

        # EWMA correlation
        ewma_corr = self._ewma_correlation(rx, ry)

        # Z-score: how unusual is current correlation?
        rolling_63 = rx.rolling(63).corr(ry).dropna()
        if len(rolling_63) > 10:
            mean_corr = float(rolling_63.mean())
            std_corr = float(rolling_63.std())
            z = (corr_21 - mean_corr) / max(std_corr, 1e-6)
        else:
            z = 0.0

        # Stability
        is_stable = abs(corr_21 - corr_252) < 0.15 if not np.isnan(corr_252) else True

        # Trend
        if np.isnan(corr_21) or np.isnan(corr_63):
            trend = "STABLE"
        elif corr_21 < corr_63 - 0.15:
            trend = "FALLING"
        elif corr_21 > corr_63 + 0.15:
            trend = "RISING"
        elif corr_21 < 0.3:
            trend = "BROKEN"
        else:
            trend = "STABLE"

        return CorrelationSnapshot(
            pair=pair_name,
            as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            corr_21d=round(corr_21, 4),
            corr_63d=round(corr_63, 4),
            corr_126d=round(corr_126, 4),
            corr_252d=round(corr_252, 4),
            corr_ewma=round(ewma_corr, 4),
            corr_full_history=round(corr_full, 4),
            corr_zscore=round(z, 4),
            is_stable=is_stable,
            trend=trend,
        )

    # ── Break Detection ───────────────────────────────────────

    def _detect_break(self, rx: pd.Series, ry: pd.Series) -> BreakDetectionResult:
        """Detect correlation structural breaks."""
        n = len(rx)
        lookback = min(self.break_lookback, n)

        # CUSUM on rolling correlation
        rolling_corr = rx.rolling(21).corr(ry).dropna()
        if len(rolling_corr) < 30:
            return BreakDetectionResult(has_break=False, break_type="NONE", break_severity="MINOR")

        mean_corr = float(rolling_corr.mean())
        std_corr = float(rolling_corr.std())

        if std_corr < 1e-6:
            return BreakDetectionResult(has_break=False, break_type="NONE", break_severity="MINOR")

        # CUSUM
        cusum = np.cumsum((rolling_corr.values - mean_corr) / std_corr)
        cusum_range = float(np.max(cusum) - np.min(cusum))

        # Z-score deviation (current vs historical)
        current_corr = float(rolling_corr.iloc[-1])
        z_dev = (current_corr - mean_corr) / std_corr

        # Determine break
        has_break = False
        break_type = "NONE"
        severity = "MINOR"

        if cusum_range > self.cusum_threshold * np.sqrt(len(rolling_corr)):
            has_break = True
            break_type = "CUSUM"
            severity = "MAJOR" if cusum_range > self.cusum_threshold * 2 * np.sqrt(len(rolling_corr)) else "MINOR"

        if abs(z_dev) > self.zscore_alert:
            has_break = True
            break_type = "ZSCORE" if break_type == "NONE" else "CUSUM+ZSCORE"
            if abs(z_dev) > self.zscore_alert * 1.5:
                severity = "CRITICAL"
            elif severity != "CRITICAL":
                severity = "MAJOR"

        # Break date (point of max CUSUM deviation)
        break_date = None
        if has_break:
            break_idx = np.argmax(np.abs(cusum))
            if break_idx < len(rolling_corr):
                break_date = str(rolling_corr.index[break_idx])

        return BreakDetectionResult(
            has_break=has_break,
            break_type=break_type,
            break_severity=severity,
            break_date=break_date,
            cusum_stat=round(cusum_range, 4),
            cusum_threshold=round(self.cusum_threshold * np.sqrt(len(rolling_corr)), 4),
            zscore_deviation=round(z_dev, 4),
            confidence=round(min(1.0, cusum_range / max(self.cusum_threshold * np.sqrt(len(rolling_corr)), 1)), 4),
        )

    # ── Helpers ───────────────────────────────────────────────

    def _ewma_correlation(self, rx: pd.Series, ry: pd.Series) -> float:
        """Exponentially weighted moving average correlation."""
        alpha = 2 / (self.ewma_halflife + 1)
        ewm_x = rx.ewm(halflife=self.ewma_halflife).mean()
        ewm_y = ry.ewm(halflife=self.ewma_halflife).mean()
        ewm_xy = (rx * ry).ewm(halflife=self.ewma_halflife).mean()
        ewm_x2 = (rx ** 2).ewm(halflife=self.ewma_halflife).mean()
        ewm_y2 = (ry ** 2).ewm(halflife=self.ewma_halflife).mean()

        cov = ewm_xy - ewm_x * ewm_y
        vol_x = np.sqrt(ewm_x2 - ewm_x ** 2)
        vol_y = np.sqrt(ewm_y2 - ewm_y ** 2)

        corr = cov / (vol_x * vol_y)
        return float(corr.iloc[-1]) if len(corr) > 0 and not np.isnan(corr.iloc[-1]) else 0.0

    @staticmethod
    def _stability_index(rx: pd.Series, ry: pd.Series, window: int = 63) -> float:
        """Correlation stability index (lower = more stable)."""
        rolling_corr = rx.rolling(window).corr(ry).dropna()
        if len(rolling_corr) < 10:
            return 1.0
        return round(float(rolling_corr.std()), 4)

    def _compute_health_score(self, report: PairHealthReport) -> float:
        """Compute overall health score [0, 1]."""
        score = 1.0

        if report.has_break:
            sev = report.break_detection.break_severity if report.break_detection else "MINOR"
            if sev == "CRITICAL":
                score -= 0.50
            elif sev == "MAJOR":
                score -= 0.30
            else:
                score -= 0.10

        if report.correlation_snapshot:
            cs = report.correlation_snapshot
            if cs.corr_21d < 0.3:
                score -= 0.30
            elif cs.corr_21d < 0.5:
                score -= 0.15
            if cs.trend == "BROKEN":
                score -= 0.20
            elif cs.trend == "FALLING":
                score -= 0.10

        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _effective_n_bets(prices: pd.DataFrame, symbols: List[str]) -> float:
        """Effective number of independent bets (eigenvalue-based)."""
        unique_symbols = list(set(s for s in symbols if s in prices.columns))
        if len(unique_symbols) < 2:
            return float(len(unique_symbols))

        returns = prices[unique_symbols].pct_change().dropna()
        if len(returns) < 30:
            return float(len(unique_symbols))

        corr_matrix = returns.corr().values
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = eigenvalues[eigenvalues > 0]
            # Effective N = (sum of eigenvalues)^2 / sum(eigenvalues^2)
            eff_n = float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2))
            return eff_n
        except Exception:
            return float(len(unique_symbols))
