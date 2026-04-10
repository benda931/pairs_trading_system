# -*- coding: utf-8 -*-
# research/permutation_test.py
"""Permutation (Label-Shuffle) Baseline for Strategy Validation
================================================================

Answers the question: "Is this strategy's Sharpe ratio better than what
random chance would produce on the same price series?"

Method (López de Prado, 2018 — "Advances in Financial Machine Learning"):
1. Run the strategy on the real data → observed Sharpe S*.
2. Shuffle the return/signal sequence N_PERM times.
3. For each permuted dataset, run the same strategy → null distribution {S_i}.
4. Permutation p-value = fraction of {S_i} that exceed S*.

Interpretations
---------------
- p < 0.05   → Reject H0; strategy likely has real edge.
- p ≥ 0.05   → Cannot reject H0; performance may be noise.
- Designed for FINAL validation after OOS split, NOT during search.

Usage
-----
from research.permutation_test import PermutationTest, PermutationResult

perm_test = PermutationTest(
    strategy_fn=my_backtest,
    price_a=price_series,
    price_b=hedge_series,
    params=best_params,
    n_permutations=500,
)
result = perm_test.run()
print(result.p_value, result.is_significant)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class PermutationResult:
    """Result of a permutation significance test."""
    observed_sharpe: float
    null_sharpe_mean: float
    null_sharpe_std: float
    null_sharpe_95th_pct: float
    p_value: float
    is_significant: bool     # p_value < alpha
    alpha: float             # significance level used
    n_permutations: int
    n_permutations_completed: int
    # Z-score: (observed - null_mean) / null_std
    z_score: float
    # 5th/95th percentile of null distribution
    null_sharpe_5th_pct: float
    # Full null distribution (can be used for plotting)
    null_distribution: List[float] = field(default_factory=list, repr=False)
    # Narrative summary
    narrative: str = ""


# ---------------------------------------------------------------------------
# Shuffle strategies
# ---------------------------------------------------------------------------

def _shuffle_returns(returns: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle the return series (destroys time structure, preserves marginal distribution)."""
    rng = np.random.default_rng(seed)
    shuffled = returns.copy()
    rng.shuffle(shuffled)
    return shuffled


def _block_shuffle_returns(
    returns: np.ndarray, block_size: int, seed: int
) -> np.ndarray:
    """Block-shuffle: shuffle blocks of ``block_size`` returns (preserves short-range autocorrelation)."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    blocks: List[np.ndarray] = []
    i = 0
    while i < n:
        blocks.append(returns[i : i + block_size])
        i += block_size
    rng.shuffle(blocks)
    return np.concatenate(blocks)[:n]


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def _compute_sharpe_from_returns(returns: np.ndarray) -> float:
    """Annualised Sharpe ratio from daily returns.  Returns NaN on error."""
    if len(returns) < 4:
        return float("nan")
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(mean / std * np.sqrt(252))


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class PermutationTest:
    """Permutation significance test for a pairs-trading strategy.

    The test supports two shuffle modes:

    - ``"returns"`` — shuffle the spread/position return series directly.
      Fast and model-free.  Best for testing whether the PnL is due to
      return structure, not just the price level.
    - ``"signals"`` — shuffle the entry/exit signal dates.  Tests whether
      the timing of entries has any edge.  Requires ``strategy_fn``.
    - ``"block_returns"`` — block-shuffle (preserves short-range autocorrelation).
      More conservative than full shuffle.

    Parameters
    ----------
    strategy_fn:
        Callable(price_a, price_b, params) → dict with at minimum ``"sharpe"``
        key.  Required for ``mode="signals"``; optional for other modes if
        ``returns_series`` is provided.
    price_a, price_b:
        Price series for the two legs (pd.Series with DatetimeIndex).
    params:
        Parameter dict forwarded to ``strategy_fn``.
    returns_series:
        Pre-computed strategy return series.  If provided and mode is
        ``"returns"`` or ``"block_returns"``, ``strategy_fn`` is not called
        during permutations (much faster).
    n_permutations:
        Number of permutations (default 500; use ≥ 1000 for publication).
    alpha:
        Significance level for the p-value gate (default 0.05).
    mode:
        Shuffle mode: ``"returns"`` | ``"block_returns"`` | ``"signals"``.
    block_size:
        Block size in days for ``"block_returns"`` mode (default 21).
    seed:
        Base random seed (deterministic).
    """

    def __init__(
        self,
        strategy_fn: Optional[Callable[..., Any]] = None,
        price_a: Optional[pd.Series] = None,
        price_b: Optional[pd.Series] = None,
        params: Optional[Dict[str, Any]] = None,
        returns_series: Optional[pd.Series] = None,
        n_permutations: int = 500,
        alpha: float = 0.05,
        mode: str = "returns",
        block_size: int = 21,
        seed: int = 42,
    ) -> None:
        self._strategy_fn = strategy_fn
        self._price_a = price_a
        self._price_b = price_b
        self._params = params or {}
        self._returns_series = returns_series
        self._n_permutations = int(max(n_permutations, 10))
        self._alpha = float(alpha)
        self._mode = str(mode).lower()
        self._block_size = int(max(block_size, 2))
        self._seed = int(seed)

        if self._mode not in {"returns", "block_returns", "signals"}:
            raise ValueError(
                f"mode must be 'returns', 'block_returns', or 'signals'. Got {mode!r}"
            )
        if self._mode == "signals" and strategy_fn is None:
            raise ValueError("strategy_fn is required for mode='signals'")
        if self._returns_series is None and strategy_fn is None:
            raise ValueError("Either returns_series or strategy_fn must be provided")

    # ------------------------------------------------------------------
    # Observed Sharpe
    # ------------------------------------------------------------------

    def _compute_observed(self) -> Tuple[float, Optional[np.ndarray]]:
        """Compute observed Sharpe and return the return array."""
        if self._returns_series is not None:
            rs = self._returns_series
            if hasattr(rs, "dropna"):
                rets = rs.dropna().values.astype(float)
            else:
                rets = np.asarray(rs, dtype=float)
                rets = rets[~np.isnan(rets)]
            sharpe = _compute_sharpe_from_returns(rets)
            return sharpe, rets

        # Call strategy_fn
        if self._strategy_fn is not None:
            try:
                result = self._strategy_fn(self._price_a, self._price_b, self._params)
                sharpe = float(
                    result.get("sharpe", result.get("Sharpe", float("nan")))
                    if isinstance(result, dict)
                    else getattr(result, "sharpe", float("nan"))
                )
                # Try to extract returns from result
                rets = None
                for attr in ("returns", "equity_curve", "pnl"):
                    candidate = (
                        result.get(attr) if isinstance(result, dict)
                        else getattr(result, attr, None)
                    )
                    if candidate is not None:
                        if isinstance(candidate, pd.Series):
                            candidate = candidate.pct_change().dropna()
                        rets = np.array(candidate, dtype=float)
                        break
                return sharpe, rets
            except Exception as exc:
                logger.error("strategy_fn failed on observed data: %s", exc)
                raise

        return float("nan"), None

    # ------------------------------------------------------------------
    # Null distribution
    # ------------------------------------------------------------------

    def _run_permutation(self, perm_idx: int, base_returns: Optional[np.ndarray]) -> float:
        """Run a single permutation and return its Sharpe."""
        seed_i = self._seed + perm_idx * 997  # deterministic per permutation

        if self._mode == "signals":
            # Shuffle the price_b series (acts as synthetic pair randomisation)
            # — equivalent to randomising which ticker we pair with
            if self._price_b is not None:
                rng = np.random.default_rng(seed_i)
                shuffled_b = pd.Series(
                    rng.permutation(self._price_b.values),
                    index=self._price_b.index,
                    name=self._price_b.name,
                )
                try:
                    result = self._strategy_fn(  # type: ignore[misc]
                        self._price_a, shuffled_b, self._params
                    )
                    sharpe = float(
                        result.get("sharpe", result.get("Sharpe", float("nan")))
                        if isinstance(result, dict)
                        else getattr(result, "sharpe", float("nan"))
                    )
                    return sharpe
                except Exception as exc:
                    logger.debug("Permutation %d failed: %s", perm_idx, exc)
                    return float("nan")
            return float("nan")

        # Return-shuffle modes (faster — no need to rerun full strategy)
        if base_returns is None:
            return float("nan")

        if self._mode == "block_returns":
            perm_rets = _block_shuffle_returns(base_returns, self._block_size, seed_i)
        else:
            perm_rets = _shuffle_returns(base_returns, seed_i)

        return _compute_sharpe_from_returns(perm_rets)

    # ------------------------------------------------------------------
    # Public run
    # ------------------------------------------------------------------

    def run(self) -> PermutationResult:
        """Execute the permutation test.

        Returns
        -------
        PermutationResult
        """
        logger.info(
            "[PermutationTest] mode=%s, n_perm=%d, alpha=%.2f",
            self._mode, self._n_permutations, self._alpha,
        )

        # Step 1: observed Sharpe
        observed_sharpe, base_returns = self._compute_observed()
        if not np.isfinite(observed_sharpe):
            return _empty_result(self._n_permutations, self._alpha, "Observed Sharpe is NaN")

        # Step 2: null distribution
        null_sharpes: List[float] = []
        for i in range(self._n_permutations):
            s = self._run_permutation(i, base_returns)
            if np.isfinite(s):
                null_sharpes.append(s)

        n_completed = len(null_sharpes)
        if n_completed < 10:
            return _empty_result(
                self._n_permutations, self._alpha,
                f"Only {n_completed} permutations completed — insufficient for p-value",
            )

        null_arr = np.array(null_sharpes)
        null_mean = float(np.mean(null_arr))
        null_std = float(np.std(null_arr, ddof=1))
        null_95th = float(np.percentile(null_arr, 95))
        null_5th = float(np.percentile(null_arr, 5))

        # Step 3: p-value = fraction of null Sharpes ≥ observed
        p_value = float(np.mean(null_arr >= observed_sharpe))
        # Minimum achievable p-value is 1/n_completed
        p_value = max(p_value, 1.0 / n_completed)

        is_significant = p_value < self._alpha
        z_score = (
            (observed_sharpe - null_mean) / null_std
            if null_std > 1e-9 else float("nan")
        )

        # Narrative
        sig_str = "SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
        narrative = (
            f"Permutation test ({self._mode} shuffle, {n_completed} permutations): "
            f"Observed SR={observed_sharpe:.3f}, Null 95th={null_95th:.3f}, "
            f"p={p_value:.4f} → {sig_str} at α={self._alpha}. "
            f"Z-score={z_score:.2f}."
        )
        if not is_significant:
            narrative += (
                " WARNING: Cannot reject H0 (SR = noise). "
                "Strategy may not have genuine predictive power."
            )

        logger.info("[PermutationTest] %s", narrative)

        return PermutationResult(
            observed_sharpe=observed_sharpe,
            null_sharpe_mean=null_mean,
            null_sharpe_std=null_std,
            null_sharpe_95th_pct=null_95th,
            null_sharpe_5th_pct=null_5th,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self._alpha,
            n_permutations=self._n_permutations,
            n_permutations_completed=n_completed,
            z_score=z_score,
            null_distribution=null_sharpes,
            narrative=narrative,
        )


def _empty_result(n_perm: int, alpha: float, reason: str) -> PermutationResult:
    return PermutationResult(
        observed_sharpe=float("nan"),
        null_sharpe_mean=float("nan"),
        null_sharpe_std=float("nan"),
        null_sharpe_95th_pct=float("nan"),
        null_sharpe_5th_pct=float("nan"),
        p_value=1.0,
        is_significant=False,
        alpha=alpha,
        n_permutations=n_perm,
        n_permutations_completed=0,
        z_score=float("nan"),
        null_distribution=[],
        narrative=f"Permutation test incomplete: {reason}",
    )


# ---------------------------------------------------------------------------
# Convenience: run permutation test on returns series directly
# ---------------------------------------------------------------------------

def permutation_test_returns(
    observed_returns: pd.Series,
    n_permutations: int = 500,
    alpha: float = 0.05,
    mode: str = "block_returns",
    block_size: int = 21,
    seed: int = 42,
) -> PermutationResult:
    """Quick permutation test on a pre-computed return series.

    Useful when you already have the OOS return stream and want to check
    whether its Sharpe is above the null distribution.

    Parameters
    ----------
    observed_returns:
        Daily return series (not equity curve).
    n_permutations:
        Number of shuffles.
    alpha:
        Significance level.
    mode:
        ``"returns"`` (full shuffle) or ``"block_returns"`` (preserves autocorrelation).
    block_size:
        Block size for block shuffle (default 21 — one trading month).
    seed:
        Random seed.
    """
    tester = PermutationTest(
        returns_series=observed_returns,
        n_permutations=n_permutations,
        alpha=alpha,
        mode=mode,
        block_size=block_size,
        seed=seed,
    )
    return tester.run()
