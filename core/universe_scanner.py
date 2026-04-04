# -*- coding: utf-8 -*-
"""
core/universe_scanner.py — Automated Pair Universe Scanner
=============================================================

Scans a universe of instruments to find tradable pairs using:

1. **Correlation pre-filter** — Fast screen (> min_corr)
2. **Cointegration validation** — Engle-Granger + Johansen
3. **Mean-reversion quality** — Half-life, Hurst, variance ratio
4. **Spread quality scoring** — Composite A+ to F grading
5. **Ranking** — Sort by composite score, filter by grade

Produces a ranked PairUniverse ready for the signal pipeline.

Usage:
    from core.universe_scanner import UniverseScanner

    scanner = UniverseScanner(min_correlation=0.6, min_grade="B")
    universe = scanner.scan(prices_df)
    print(f"Found {len(universe.pairs)} tradable pairs")
    for p in universe.pairs[:10]:
        print(f"  {p.sym_x}/{p.sym_y}: score={p.score:.3f} grade={p.grade}")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class ScoredPair:
    """A pair with its analysis scores."""
    sym_x: str
    sym_y: str
    correlation: float
    score: float                         # Composite quality score [0, 1]
    grade: str                           # A+ to F
    half_life: float
    hurst: float
    is_cointegrated: bool
    hedge_ratio: float
    stationarity_score: float
    mean_reversion_score: float
    stability_score: float
    trading_score: float
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Complete universe scan result."""
    n_instruments: int
    n_pairs_screened: int
    n_pairs_passed_correlation: int
    n_pairs_passed_cointegration: int
    n_pairs_final: int
    elapsed_seconds: float
    pairs: List[ScoredPair] = field(default_factory=list)
    rejection_summary: Dict[str, int] = field(default_factory=dict)

    @property
    def yield_rate(self) -> float:
        """Percentage of screened pairs that passed all filters."""
        return self.n_pairs_final / max(self.n_pairs_screened, 1)


# =====================================================================
# Scanner
# =====================================================================

class UniverseScanner:
    """
    Automated pair universe scanner.

    Scans all combinations of instruments, applies statistical filters,
    and returns a ranked list of tradable pairs.
    """

    def __init__(
        self,
        min_correlation: float = 0.60,
        min_grade: str = "C",
        min_observations: int = 252,
        max_half_life: float = 120.0,
        min_half_life: float = 2.0,
        max_hurst: float = 0.50,
        max_pairs: int = 50,
        require_cointegration: bool = True,
    ):
        self.min_correlation = min_correlation
        self.min_grade = min_grade
        self.min_observations = min_observations
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.max_hurst = max_hurst
        self.max_pairs = max_pairs
        self.require_cointegration = require_cointegration

        self._grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]

    def scan(
        self,
        prices: pd.DataFrame,
        symbols: Optional[Sequence[str]] = None,
    ) -> ScanResult:
        """
        Scan a price DataFrame for tradable pairs.

        Parameters
        ----------
        prices : pd.DataFrame
            Wide-format: index=date, columns=ticker, values=close prices.
        symbols : Sequence[str], optional
            Subset of columns to scan. Default: all columns.

        Returns
        -------
        ScanResult with ranked pairs.
        """
        t0 = time.time()

        if symbols is None:
            symbols = list(prices.columns)
        symbols = [s for s in symbols if s in prices.columns]

        n_instruments = len(symbols)
        all_pairs = list(combinations(symbols, 2))
        n_screened = len(all_pairs)

        logger.info(
            "Universe scan: %d instruments, %d pair combinations",
            n_instruments, n_screened,
        )

        rejections: Dict[str, int] = {
            "insufficient_data": 0,
            "low_correlation": 0,
            "not_cointegrated": 0,
            "bad_half_life": 0,
            "high_hurst": 0,
            "low_grade": 0,
            "analysis_error": 0,
        }

        # Phase 1: Correlation pre-filter
        corr_matrix = prices[symbols].corr()
        candidates_phase1 = []

        for sym_x, sym_y in all_pairs:
            corr_val = corr_matrix.loc[sym_x, sym_y]
            if abs(corr_val) >= self.min_correlation:
                candidates_phase1.append((sym_x, sym_y, float(corr_val)))
            else:
                rejections["low_correlation"] += 1

        n_passed_corr = len(candidates_phase1)
        logger.info("Phase 1 (correlation): %d/%d passed (min=%.2f)",
                     n_passed_corr, n_screened, self.min_correlation)

        # Phase 2: Full analysis
        from core.spread_analytics import SpreadAnalytics
        sa = SpreadAnalytics(
            half_life_max=self.max_half_life,
            half_life_min=self.min_half_life,
        )

        scored_pairs: List[ScoredPair] = []
        n_passed_coint = 0

        for sym_x, sym_y, corr_val in candidates_phase1:
            try:
                px = prices[sym_x].dropna()
                py = prices[sym_y].dropna()

                # Data check
                common = px.index.intersection(py.index)
                if len(common) < self.min_observations:
                    rejections["insufficient_data"] += 1
                    continue

                px = px.loc[common]
                py = py.loc[common]

                # Full spread analysis
                report = sa.full_analysis(px, py, sym_x=sym_x, sym_y=sym_y)

                # Cointegration filter
                if self.require_cointegration and not report.is_cointegrated:
                    rejections["not_cointegrated"] += 1
                    continue
                n_passed_coint += 1

                # Mean-reversion filters
                if report.mean_reversion:
                    hl = report.mean_reversion.half_life_days
                    if hl < self.min_half_life or hl > self.max_half_life:
                        rejections["bad_half_life"] += 1
                        continue
                    if report.mean_reversion.hurst_exponent > self.max_hurst:
                        rejections["high_hurst"] += 1
                        continue

                # Grade filter
                if report.quality:
                    if not self._grade_passes(report.quality.grade, self.min_grade):
                        rejections["low_grade"] += 1
                        continue

                # Build scored pair
                sp = ScoredPair(
                    sym_x=sym_x,
                    sym_y=sym_y,
                    correlation=corr_val,
                    score=report.composite_score,
                    grade=report.quality.grade if report.quality else "F",
                    half_life=report.half_life_days,
                    hurst=report.mean_reversion.hurst_exponent if report.mean_reversion else 0.5,
                    is_cointegrated=report.is_cointegrated,
                    hedge_ratio=report.hedge_ratio_ols,
                    stationarity_score=report.quality.stationarity_score if report.quality else 0,
                    mean_reversion_score=report.quality.mean_reversion_score if report.quality else 0,
                    stability_score=report.quality.stability_score if report.quality else 0,
                    trading_score=report.quality.trading_score if report.quality else 0,
                    warnings=report.quality.warnings if report.quality else [],
                )
                scored_pairs.append(sp)

            except Exception as exc:
                rejections["analysis_error"] += 1
                logger.debug("Analysis failed for %s/%s: %s", sym_x, sym_y, exc)

        # Rank by composite score
        scored_pairs.sort(key=lambda p: p.score, reverse=True)

        # Limit
        final_pairs = scored_pairs[:self.max_pairs]

        elapsed = time.time() - t0
        logger.info(
            "Universe scan complete: %d pairs found (%.1fs), yield=%.1f%%",
            len(final_pairs), elapsed, len(final_pairs) / max(n_screened, 1) * 100,
        )

        return ScanResult(
            n_instruments=n_instruments,
            n_pairs_screened=n_screened,
            n_pairs_passed_correlation=n_passed_corr,
            n_pairs_passed_cointegration=n_passed_coint,
            n_pairs_final=len(final_pairs),
            elapsed_seconds=round(elapsed, 2),
            pairs=final_pairs,
            rejection_summary=rejections,
        )

    def _grade_passes(self, grade: str, min_grade: str) -> bool:
        """Check if grade meets minimum threshold."""
        try:
            return self._grade_order.index(grade) <= self._grade_order.index(min_grade)
        except ValueError:
            return False
