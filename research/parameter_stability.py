# -*- coding: utf-8 -*-
# research/parameter_stability.py
"""Parameter Sensitivity & Robustness Analysis
================================================

Institutional-grade diagnostics for optimized strategy parameters.

Three complementary views:
1. **Sensitivity Map** — how does OOS performance degrade as each parameter
   moves away from its optimal value (2-D heatmaps for pairs of params)?
2. **Robustness Radius** — the ±δ neighbourhood around the optimum where
   Sharpe stays above a floor (e.g. 50 % of peak).  A large radius → robust.
3. **Fragility Score** — scalar summary: high sensitivity + small radius → fragile.

Usage
-----
from research.parameter_stability import (
    ParameterStabilityAnalyzer,
    SensitivityResult,
    RobustnessRadius,
    FragilityReport,
)

# results_df = output of run_optimization()
analyzer = ParameterStabilityAnalyzer(results_df, param_names=["entry_z", "exit_z", "window"])
report = analyzer.full_report()
print(report.fragility_score, report.is_fragile)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """1-D sensitivity around a single parameter."""
    param_name: str
    # Grid of parameter values tested
    param_values: List[float]
    # Corresponding median Sharpe in each neighbourhood bucket
    sharpe_values: List[float]
    # Slope of linear fit: dSharpe/dParam
    sensitivity_slope: float
    # R² of the linear fit — how monotonic is the degradation?
    r_squared: float
    # Peak Sharpe observed in this slice
    peak_sharpe: float
    # Value at which peak was observed
    peak_value: float


@dataclass
class RobustnessRadius:
    """How far from the optimum performance stays acceptable."""
    param_name: str
    optimal_value: float
    peak_sharpe: float
    # Threshold = peak × floor_ratio (default 0.5)
    floor_sharpe: float
    # Symmetric ±radius around optimal (in param units) where Sharpe > floor
    radius_lower: float  # how far below optimal is still acceptable
    radius_upper: float  # how far above optimal is still acceptable
    radius_pct_of_range: float  # as fraction of total param range (0–1)
    # True if robustness radius covers < 20 % of range on either side
    is_narrow: bool


@dataclass
class FragilityReport:
    """Aggregate fragility score for a parameter set.

    Fragility is high when:
    - Sensitivity slopes are steep (performance cliffs)
    - Robustness radii are narrow (parameter values must be near-perfect)
    - Cross-validation of top-N param sets shows high coefficient of variation

    fragility_score ∈ [0, 1]:  0 = fully robust, 1 = extremely fragile.
    """
    # Per-parameter diagnostics
    sensitivity: List[SensitivityResult]
    radii: List[RobustnessRadius]
    # Aggregate
    fragility_score: float          # scalar ∈ [0, 1]
    is_fragile: bool                # True if fragility_score > 0.6
    # Parameter CV (from walk-forward folds or top-N trials)
    param_cv: Dict[str, float]      # param → coefficient of variation
    mean_cv: float
    cv_warning: bool                # True if mean_cv > 0.5
    # Sharpe degradation: how much Sharpe drops as params deviate 1 std-dev
    sharpe_degradation_per_sigma: Dict[str, float]
    # Summary narrative
    narrative: str = ""


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class ParameterStabilityAnalyzer:
    """Compute sensitivity, robustness, and fragility from optimization results.

    Parameters
    ----------
    results_df:
        DataFrame output of ``run_optimization()``.  Must contain at minimum:
        - Parameter columns (named in ``param_names``)
        - ``"sharpe"`` column
    param_names:
        Explicit list of parameter column names to analyse.  If *None*, all
        numeric columns that are not well-known metric columns are used.
    sharpe_col:
        Column name for the Sharpe ratio metric (default ``"sharpe"``).
    floor_ratio:
        Fraction of peak Sharpe that defines the acceptable floor for
        robustness-radius calculation (default 0.5).
    n_buckets:
        Number of equal-width buckets along each parameter axis for
        sensitivity estimation (default 10).
    top_n_for_cv:
        Number of top Sharpe trials used for parameter CV calculation
        (default 20).
    """

    _METRIC_COLS = frozenset({
        "trial_id", "trial_number", "sharpe", "return", "drawdown", "win_rate",
        "score", "score_mode", "classic_score", "hf_score", "runtime_sec",
        "trial_dsr", "dsr_gate_passed", "_source",
        "ES_95", "VaR_95", "Trades", "Turnover", "Exposure",
        "hf_base_score", "hf_final_score", "hf_dd_soft_penalty",
        "hf_tail_penalty", "hf_behavior_penalty", "hf_wf_sharpe_mean",
        "hf_wf_sharpe_std",
    })

    def __init__(
        self,
        results_df: pd.DataFrame,
        param_names: Optional[List[str]] = None,
        sharpe_col: str = "sharpe",
        floor_ratio: float = 0.5,
        n_buckets: int = 10,
        top_n_for_cv: int = 20,
    ) -> None:
        if results_df is None or results_df.empty:
            raise ValueError("results_df must be a non-empty DataFrame")

        self._df = results_df.copy()
        self._sharpe_col = sharpe_col
        self._floor_ratio = float(floor_ratio)
        self._n_buckets = int(max(n_buckets, 3))
        self._top_n_for_cv = int(max(top_n_for_cv, 2))

        # Resolve parameter column names
        if param_names is not None:
            self._param_names = [p for p in param_names if p in self._df.columns]
        else:
            numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
            self._param_names = [
                c for c in numeric_cols
                if c not in self._METRIC_COLS and not c.startswith("hf_")
            ]

        if not self._param_names:
            raise ValueError(
                "No parameter columns found. Provide param_names or ensure the "
                "DataFrame contains numeric parameter columns."
            )

        # Ensure sharpe column is numeric
        if sharpe_col not in self._df.columns:
            raise ValueError(f"sharpe_col={sharpe_col!r} not found in results_df")
        self._df[sharpe_col] = pd.to_numeric(self._df[sharpe_col], errors="coerce")

        # Drop rows with NaN Sharpe
        self._df = self._df.dropna(subset=[sharpe_col])
        if self._df.empty:
            raise ValueError("No finite Sharpe values found in results_df after cleaning")

    # ------------------------------------------------------------------
    # 1-D Sensitivity
    # ------------------------------------------------------------------

    def compute_sensitivity(self, param: str) -> SensitivityResult:
        """Compute 1-D sensitivity of Sharpe for a single parameter.

        Algorithm
        ---------
        1. Bin ``param`` into ``n_buckets`` equal-width intervals.
        2. Compute median Sharpe in each occupied bucket.
        3. Fit a linear regression (slope = sensitivity gradient).
        4. Record peak Sharpe and the bucket median at which it occurs.
        """
        if param not in self._df.columns:
            raise ValueError(f"Parameter {param!r} not in DataFrame")

        col = pd.to_numeric(self._df[param], errors="coerce")
        sharpe = self._df[self._sharpe_col]
        mask = col.notna() & sharpe.notna()
        x = col[mask].values.astype(float)
        y = sharpe[mask].values.astype(float)

        if len(x) < 3:
            return SensitivityResult(
                param_name=param,
                param_values=[],
                sharpe_values=[],
                sensitivity_slope=float("nan"),
                r_squared=float("nan"),
                peak_sharpe=float("nan"),
                peak_value=float("nan"),
            )

        p_min, p_max = x.min(), x.max()
        if p_max <= p_min:
            p_max = p_min + 1e-9

        edges = np.linspace(p_min, p_max, self._n_buckets + 1)
        bucket_centers: List[float] = []
        bucket_sharpes: List[float] = []
        for i in range(self._n_buckets):
            lo, hi = edges[i], edges[i + 1]
            # Include upper bound on last bucket
            if i == self._n_buckets - 1:
                mask_b = (x >= lo) & (x <= hi)
            else:
                mask_b = (x >= lo) & (x < hi)
            if mask_b.sum() == 0:
                continue
            bucket_centers.append(float((lo + hi) / 2.0))
            bucket_sharpes.append(float(np.median(y[mask_b])))

        if len(bucket_centers) < 2:
            slope, r2 = float("nan"), float("nan")
        else:
            bc = np.array(bucket_centers)
            bs = np.array(bucket_sharpes)
            try:
                res = stats.linregress(bc, bs)
                slope = float(res.slope)
                r2 = float(res.rvalue ** 2)
            except Exception:
                slope, r2 = float("nan"), float("nan")

        if bucket_sharpes:
            peak_idx = int(np.argmax(bucket_sharpes))
            peak_sharpe = bucket_sharpes[peak_idx]
            peak_value = bucket_centers[peak_idx]
        else:
            peak_sharpe, peak_value = float("nan"), float("nan")

        return SensitivityResult(
            param_name=param,
            param_values=bucket_centers,
            sharpe_values=bucket_sharpes,
            sensitivity_slope=slope,
            r_squared=r2,
            peak_sharpe=peak_sharpe,
            peak_value=peak_value,
        )

    def compute_all_sensitivities(self) -> List[SensitivityResult]:
        """Compute 1-D sensitivity for every parameter."""
        results = []
        for p in self._param_names:
            try:
                results.append(self.compute_sensitivity(p))
            except Exception as exc:
                logger.debug("Sensitivity failed for %r: %s", p, exc)
        return results

    # ------------------------------------------------------------------
    # Robustness Radius
    # ------------------------------------------------------------------

    def compute_robustness_radius(
        self, sensitivity: SensitivityResult, param_range: Optional[Tuple[float, float]] = None
    ) -> RobustnessRadius:
        """Compute robustness radius for a parameter given its 1-D sensitivity.

        The radius is the distance from the optimal value to the nearest point
        where Sharpe falls below ``peak_sharpe × floor_ratio``.
        """
        param = sensitivity.param_name
        peak_sharpe = sensitivity.peak_sharpe
        peak_value = sensitivity.peak_value

        if not np.isfinite(peak_sharpe) or not np.isfinite(peak_value):
            return RobustnessRadius(
                param_name=param,
                optimal_value=float("nan"),
                peak_sharpe=float("nan"),
                floor_sharpe=float("nan"),
                radius_lower=float("nan"),
                radius_upper=float("nan"),
                radius_pct_of_range=float("nan"),
                is_narrow=True,
            )

        floor_sharpe = peak_sharpe * self._floor_ratio

        pv = np.array(sensitivity.param_values)
        sv = np.array(sensitivity.sharpe_values)
        above_floor = sv >= floor_sharpe

        # Lower radius: distance from peak_value to nearest point to the LEFT
        # where performance drops below floor
        lower_vals = pv[pv <= peak_value]
        lower_above = above_floor[pv <= peak_value]
        if lower_vals.size > 0 and lower_above.any():
            # Furthest left where still above floor
            furthest_left = lower_vals[lower_above].min()
            radius_lower = float(peak_value - furthest_left)
        else:
            radius_lower = 0.0

        # Upper radius: distance from peak_value to nearest point to the RIGHT
        upper_vals = pv[pv >= peak_value]
        upper_above = above_floor[pv >= peak_value]
        if upper_vals.size > 0 and upper_above.any():
            furthest_right = upper_vals[upper_above].max()
            radius_upper = float(furthest_right - peak_value)
        else:
            radius_upper = 0.0

        # Normalise by total parameter range
        if param_range is not None:
            total_range = max(float(param_range[1] - param_range[0]), 1e-9)
        elif pv.size > 0:
            total_range = max(float(pv.max() - pv.min()), 1e-9)
        else:
            total_range = 1e-9

        radius_pct = float(min(radius_lower, radius_upper) / total_range)
        is_narrow = radius_pct < 0.20  # < 20% of range on either side → narrow

        return RobustnessRadius(
            param_name=param,
            optimal_value=peak_value,
            peak_sharpe=peak_sharpe,
            floor_sharpe=floor_sharpe,
            radius_lower=radius_lower,
            radius_upper=radius_upper,
            radius_pct_of_range=radius_pct,
            is_narrow=is_narrow,
        )

    # ------------------------------------------------------------------
    # Parameter CV (top-N trials)
    # ------------------------------------------------------------------

    def compute_param_cv(self) -> Dict[str, float]:
        """Coefficient of variation of each parameter across top-N Sharpe trials.

        CV = std / |mean|.  High CV → parameter is unstable across winners.
        """
        top_df = (
            self._df.nlargest(self._top_n_for_cv, self._sharpe_col)
            if len(self._df) > self._top_n_for_cv
            else self._df
        )
        cv_dict: Dict[str, float] = {}
        for p in self._param_names:
            if p not in top_df.columns:
                continue
            col = pd.to_numeric(top_df[p], errors="coerce").dropna()
            if len(col) < 2:
                cv_dict[p] = float("nan")
                continue
            mean_abs = abs(float(col.mean()))
            std = float(col.std(ddof=1))
            cv_dict[p] = std / mean_abs if mean_abs > 1e-9 else float("inf")
        return cv_dict

    # ------------------------------------------------------------------
    # Sharpe degradation per sigma
    # ------------------------------------------------------------------

    def compute_sharpe_degradation(self) -> Dict[str, float]:
        """Estimate how much Sharpe drops if each parameter shifts by ±1 std-dev.

        Uses the sensitivity slope × std-dev(param) as a linear approximation.
        """
        degradation: Dict[str, float] = {}
        for p in self._param_names:
            col = pd.to_numeric(self._df[p], errors="coerce").dropna()
            if len(col) < 2:
                degradation[p] = float("nan")
                continue
            param_std = float(col.std(ddof=1))
            try:
                sens = self.compute_sensitivity(p)
                slope = sens.sensitivity_slope
                degradation[p] = abs(slope * param_std) if np.isfinite(slope) else float("nan")
            except Exception:
                degradation[p] = float("nan")
        return degradation

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(
        self,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> FragilityReport:
        """Generate the complete fragility and robustness report.

        Parameters
        ----------
        param_ranges:
            Optional dict of {param_name: (low, high)} for normalising radius.
            If not provided, ranges are inferred from the data.
        """
        sensitivities = self.compute_all_sensitivities()
        radii = [
            self.compute_robustness_radius(
                s,
                param_range=param_ranges.get(s.param_name) if param_ranges else None,
            )
            for s in sensitivities
        ]

        param_cv = self.compute_param_cv()
        sharpe_degradation = self.compute_sharpe_degradation()

        # Mean CV across parameters (ignoring NaN)
        valid_cvs = [v for v in param_cv.values() if np.isfinite(v)]
        mean_cv = float(np.mean(valid_cvs)) if valid_cvs else float("nan")
        cv_warning = bool(mean_cv > 0.5) if np.isfinite(mean_cv) else False

        # Fragility score: convex combination of:
        # (a) normalised slope steepness
        # (b) fraction of narrow radii
        # (c) mean CV / 1.0 (clipped to [0,1])
        fragility_components: List[float] = []

        # Component A: average |slope| normalised by peak Sharpe std across params
        all_sharpes = self._df[self._sharpe_col].dropna()
        sharpe_scale = max(float(all_sharpes.std()), 1e-9)
        slopes = [
            abs(s.sensitivity_slope) / sharpe_scale
            for s in sensitivities
            if np.isfinite(s.sensitivity_slope)
        ]
        if slopes:
            # Sigmoid transform: slope → fragility
            mean_slope_norm = float(np.mean(slopes))
            comp_a = float(2.0 / (1.0 + np.exp(-2.0 * mean_slope_norm)) - 1.0)  # sigmoid shifted to [0,1]
        else:
            comp_a = 0.0
        fragility_components.append(comp_a)

        # Component B: fraction of narrow radii
        narrow_count = sum(1 for r in radii if r.is_narrow and np.isfinite(r.radius_pct_of_range))
        total_finite = sum(1 for r in radii if np.isfinite(r.radius_pct_of_range))
        comp_b = float(narrow_count / total_finite) if total_finite > 0 else 0.0
        fragility_components.append(comp_b)

        # Component C: mean CV clipped to [0, 1]
        comp_c = float(min(max(mean_cv, 0.0), 1.0)) if np.isfinite(mean_cv) else 0.5
        fragility_components.append(comp_c)

        # Weighted average: slopes 30%, radii 40%, CV 30%
        weights = [0.30, 0.40, 0.30]
        fragility_score = float(sum(w * c for w, c in zip(weights, fragility_components)))
        fragility_score = float(min(max(fragility_score, 0.0), 1.0))
        is_fragile = fragility_score > 0.6

        # Narrative
        parts = []
        if is_fragile:
            parts.append(
                f"FRAGILE strategy (fragility={fragility_score:.2f}). "
                "Parameters are sensitive to small changes — DO NOT promote to production."
            )
        else:
            parts.append(
                f"Robustness acceptable (fragility={fragility_score:.2f})."
            )
        if cv_warning:
            parts.append(
                f"High parameter instability: mean CV={mean_cv:.2f} > 0.5 across top-{self._top_n_for_cv} trials."
            )
        narrow_params = [r.param_name for r in radii if r.is_narrow and np.isfinite(r.radius_pct_of_range)]
        if narrow_params:
            parts.append(
                f"Narrow robustness radius on: {', '.join(narrow_params)}. "
                "Small parameter shifts degrade Sharpe below 50% of peak."
            )
        steep_params = [
            s.param_name for s in sensitivities
            if np.isfinite(s.sensitivity_slope) and abs(s.sensitivity_slope) > 0.5
        ]
        if steep_params:
            parts.append(
                f"Steep sensitivity gradients on: {', '.join(steep_params)}."
            )
        narrative = " ".join(parts)

        return FragilityReport(
            sensitivity=sensitivities,
            radii=radii,
            fragility_score=fragility_score,
            is_fragile=is_fragile,
            param_cv=param_cv,
            mean_cv=mean_cv,
            cv_warning=cv_warning,
            sharpe_degradation_per_sigma=sharpe_degradation,
            narrative=narrative,
        )


# ---------------------------------------------------------------------------
# Convenience: 2-D sensitivity heatmap data
# ---------------------------------------------------------------------------

def compute_2d_sensitivity_heatmap(
    results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    sharpe_col: str = "sharpe",
    n_bins: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2-D sensitivity heatmap for a pair of parameters.

    Returns
    -------
    x_centers : np.ndarray, shape (n_bins,)
    y_centers : np.ndarray, shape (n_bins,)
    sharpe_grid : np.ndarray, shape (n_bins, n_bins)
        Median Sharpe in each cell.  NaN where no trials exist.
    """
    df = results_df.dropna(subset=[param_x, param_y, sharpe_col]).copy()
    df[param_x] = pd.to_numeric(df[param_x], errors="coerce")
    df[param_y] = pd.to_numeric(df[param_y], errors="coerce")
    df[sharpe_col] = pd.to_numeric(df[sharpe_col], errors="coerce")
    df = df.dropna()

    if df.empty or n_bins < 2:
        empty = np.full((n_bins, n_bins), np.nan)
        return np.array([]), np.array([]), empty

    x = df[param_x].values
    y = df[param_y].values
    s = df[sharpe_col].values

    x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    y_edges = np.linspace(y.min(), y.max(), n_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    grid = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        x_lo, x_hi = x_edges[i], x_edges[i + 1]
        mask_x = (x >= x_lo) & (x <= x_hi if i == n_bins - 1 else x < x_hi)
        for j in range(n_bins):
            y_lo, y_hi = y_edges[j], y_edges[j + 1]
            mask_y = (y >= y_lo) & (y <= y_hi if j == n_bins - 1 else y < y_hi)
            cell_mask = mask_x & mask_y
            if cell_mask.sum() > 0:
                grid[i, j] = float(np.median(s[cell_mask]))

    return x_centers, y_centers, grid


# ---------------------------------------------------------------------------
# Convenience: standalone fragility check (accepts optimizer DataFrame directly)
# ---------------------------------------------------------------------------

def check_fragility(
    results_df: pd.DataFrame,
    param_names: Optional[List[str]] = None,
    sharpe_col: str = "sharpe",
    floor_ratio: float = 0.5,
) -> FragilityReport:
    """One-liner fragility check.  Returns a full FragilityReport.

    Example
    -------
    >>> df = run_optimization(candidates, config, n_trials=100)
    >>> report = check_fragility(df, param_names=["entry_z", "exit_z"])
    >>> print(report.fragility_score, report.narrative)
    """
    analyzer = ParameterStabilityAnalyzer(
        results_df,
        param_names=param_names,
        sharpe_col=sharpe_col,
        floor_ratio=floor_ratio,
    )
    return analyzer.full_report()
