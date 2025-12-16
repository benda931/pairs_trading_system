# -*- coding: utf-8 -*-
"""
risk_parity.py — Professional Risk-Parity utilities for the pairs-trading system.
==============================================================================

This module provides **hedge-fund–grade** risk-parity tools at two layers:

1. Portfolio-level risk parity
   --------------------------------
   - risk_parity_weights_from_cov:
       Solve for weights w such that each asset contributes a desired
       fraction of portfolio risk, given a covariance matrix Σ.

   - risk_parity_naive_weights:
       Fast approximation using 1/vol.

   - risk_parity_from_returns:
       Convenience wrapper that takes a returns DataFrame → cov → weights.

   - compute_risk_contributions:
       Decompose portfolio risk into per-asset contributions (RC, MRC, %).

   - risk_parity_diagnostics:
       Compact diagnostic dict suitable for logging / debugging.

2. Parameter-level risk parity
   --------------------------------
   - apply_risk_parity_to_params:
       Rescale a parameter dictionary so that each numeric parameter gets
       a more balanced "risk weight" (e.g., for hyper-parameters feeding
       Optuna/Backtest engines).

Design goals
============
- Hedge-fund–grade numerics (iterative solver with convergence checks).
- Robustness to noisy covariance matrices (ridge regularization, symmetry fix).
- No heavy external dependencies beyond NumPy/Pandas.
- Backwards-compatible enough to be called from existing optimization code.

This module is intentionally *standalone* so it can be reused by multiple
parts of your system (optimization, backtest sizing, dashboards, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

Number = Union[int, float, np.number]

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Configuration
# =====================================================================


@dataclass
class RiskParityConfig:
    """
    Configuration for portfolio-level risk parity solvers.

    Attributes
    ----------
    max_iter : int
        Maximum number of iterations for the fixed-point solver.
    tol : float
        Convergence tolerance on risk contributions.
    min_weight : float
        Optional minimum weight per asset (long-only). If >0, all weights
        are clipped to at least `min_weight` and renormalized.
    use_naive_if_fail : bool
        If True, fallback to naive 1/vol weights when iterative solver
        does not converge.
    verbose : bool
        If True, log convergence information via the module logger.
    ridge : float
        Ridge regularization added to the covariance diagonal to improve
        numerical stability.
    allow_short : bool
        If True, the solver *allows* negative weights (very basic support).
        Note: risk-parity is most natural in long-only settings — use with care.
    target_risk_budget : Optional[np.ndarray]
        If not None, vector of size N specifying the desired percentage of
        risk contributed by each asset (sums to 1). Default is equal budget.
    target_vol : Optional[float]
        If not None, post-scale the weights to target an annualized volatility.
        (You may prefer to handle leverage elsewhere — this is optional.)
    """

    max_iter: int = 10_000
    tol: float = 1e-8
    min_weight: float = 0.0
    use_naive_if_fail: bool = True
    verbose: bool = False
    ridge: float = 1e-8
    allow_short: bool = False
    target_risk_budget: Optional[np.ndarray] = None
    target_vol: Optional[float] = None  # annualized target vol (optional)

    def log(self, msg: str, *args: object) -> None:
        if self.verbose:
            logger.info(msg, *args)


# =====================================================================
# 2. Utility helpers
# =====================================================================


def _to_cov_matrix(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Convert a DataFrame or ndarray to a clean covariance matrix."""
    if isinstance(data, pd.DataFrame):
        cov = np.asarray(data.values, dtype=float)
    else:
        cov = np.asarray(data, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance must be square matrix, got shape={cov.shape!r}")

    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance matrix contains non-finite values")

    return cov


def _ensure_symmetric(cov: np.ndarray) -> np.ndarray:
    """Force covariance matrix to be symmetric by averaging with its transpose."""
    return 0.5 * (cov + cov.T)


def _regularize_cov(cov: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Apply ridge regularization to a covariance matrix."""
    Σ = np.array(cov, dtype=float, copy=True)
    n = Σ.shape[0]
    if ridge > 0:
        Σ.flat[:: n + 1] += ridge
    return Σ


def _safe_normalize(weights: np.ndarray, min_weight: float = 0.0) -> np.ndarray:
    """
    Normalize weight vector to sum to 1, enforcing optional minimum weight.

    - Ensures non-negativity if min_weight >= 0 and allow_short=False.
    - Handles degenerate cases by falling back to equal weights.
    """
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError("Weights contain non-finite values")

    if w.ndim != 1:
        raise ValueError("Weights must be 1-D array")

    if np.allclose(w, 0):
        w = np.ones_like(w)

    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)

    w = w / w.sum()

    if min_weight > 0.0:
        w = np.maximum(w, min_weight)
        w = w / w.sum()

    return w


def _normalize_risk_budget(n: int, budget: Optional[np.ndarray]) -> np.ndarray:
    """
    Ensure risk-budget vector has length n and sums to 1.

    If budget is None → equal risk budget.
    """
    if budget is None:
        b = np.ones(n, dtype=float) / n
        return b

    b = np.asarray(budget, dtype=float).reshape(-1)
    if b.shape[0] != n:
        raise ValueError(f"risk budget length {b.shape[0]} != n={n}")
    if np.any(b < 0):
        raise ValueError("risk budget entries must be non-negative")
    if not np.isfinite(b).all():
        raise ValueError("risk budget contains non-finite values")

    s = b.sum()
    if s <= 0:
        raise ValueError("sum of risk budget <= 0")
    return b / s


# =====================================================================
# 3. Portfolio-level risk parity
# =====================================================================


def risk_parity_naive_weights(
    cov: Union[pd.DataFrame, np.ndarray],
    *,
    min_weight: float = 0.0,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Fast approximation of risk-parity weights using inverse volatility.

    w_i ∝ 1 / sqrt(Σ_ii)

    Parameters
    ----------
    cov : DataFrame or ndarray
        Covariance matrix Σ.
    min_weight : float, optional
        Minimum weight per asset (long-only). Default is 0.0.
    ridge : float, optional
        Ridge regularization added to the diagonal of Σ before computing
        volatilities.
    """
    Σ = _regularize_cov(_ensure_symmetric(_to_cov_matrix(cov)), ridge=ridge)
    diag = np.diag(Σ)
    if np.any(diag <= 0):
        raise ValueError("Covariance diagonal must be positive for naive RP")

    inv_vol = 1.0 / np.sqrt(diag)
    return _safe_normalize(inv_vol, min_weight=min_weight)


def risk_parity_weights_from_cov(
    cov: Union[pd.DataFrame, np.ndarray],
    config: RiskParityConfig | None = None,
) -> np.ndarray:
    """
    Solve for full risk-parity weights given covariance matrix Σ.

    Problem (Maillard, Roncalli & Teïletche 2010, simplified):
        find w > 0, sum(w)=1
        such that each asset's risk contribution matches target risk budget.

    Implementation:
        - Uses a multiplicative-update fixed-point scheme.
        - Supports arbitrary risk budgets (equal by default).
        - Robust to moderate dimensionality (N up to a few hundred).
    """
    if config is None:
        config = RiskParityConfig()

    Σ_raw = _to_cov_matrix(cov)
    Σ = _regularize_cov(_ensure_symmetric(Σ_raw), ridge=config.ridge)
    n = Σ.shape[0]

    # Normalize risk budget
    rb = _normalize_risk_budget(n, config.target_risk_budget)

    # Start from naive 1/vol as initialization (good warmstart)
    w = risk_parity_naive_weights(Σ, min_weight=config.min_weight, ridge=0.0)

    for it in range(config.max_iter):
        sigma_p = float(np.sqrt(w @ Σ @ w))
        if sigma_p <= 0:
            raise ValueError("Portfolio std became non-positive during iterations")

        # Marginal risk contributions: mrc_i = (Σ w)_i / σ_p
        mrc = (Σ @ w) / sigma_p

        # Total risk contribution from asset i: RC_i = w_i * mrc_i
        rc = w * mrc
        total_rc = rc.sum()
        if total_rc <= 0:
            raise ValueError("Total risk contribution became non-positive")

        # Normalize RC to become risk contribution percentages
        rc_pct = rc / total_rc

        # Check convergence vs target risk budget
        diff = rc_pct - rb
        max_dev = float(np.max(np.abs(diff)))
        if max_dev < config.tol:
            config.log("risk_parity converged in %d iterations", it + 1)
            w = _safe_normalize(w, min_weight=config.min_weight)
            break

        # Multiplicative update towards target risk budget:
        #   w_i <- w_i * (rb_i / rc_pct_i)
        adj = rb / (rc_pct + 1e-12)
        w = w * adj

        if not config.allow_short:
            w = np.maximum(w, 0.0)
        w = _safe_normalize(w, min_weight=config.min_weight)
    else:
        # Not converged
        config.log("risk_parity did not fully converge in %d iterations", config.max_iter)
        if config.use_naive_if_fail:
            config.log("falling back to naive 1/vol risk parity weights")
            w = risk_parity_naive_weights(Σ_raw, min_weight=config.min_weight, ridge=config.ridge)

    # Optional target volatility scaling (leverage)
    if config.target_vol is not None and config.target_vol > 0:
        sigma_p = float(np.sqrt(w @ Σ @ w))
        if sigma_p > 0:
            leverage = float(config.target_vol / sigma_p)
            w = w * leverage
            # NOTE: now sum(w) may be != 1 (this is intended: leveraged portfolio).
        else:
            config.log("target_vol requested but portfolio std=0, skipping leveraging")

    return w


def risk_parity_weights_series(
    cov: pd.DataFrame,
    config: RiskParityConfig | None = None,
) -> pd.Series:
    """
    Convenience wrapper that returns weights as a pandas Series.

    The Series index is aligned to the covariance DataFrame's columns.
    """
    w = risk_parity_weights_from_cov(cov, config=config)
    return pd.Series(w, index=cov.columns, name="w_risk_parity")


def risk_parity_from_returns(
    returns: pd.DataFrame,
    config: RiskParityConfig | None = None,
    *,
    demean: bool = True,
) -> pd.Series:
    """
    Compute risk-parity weights directly from a returns DataFrame.

    Parameters
    ----------
    returns : DataFrame
        Historical returns (T x N) for N assets.
    config : RiskParityConfig, optional
        Configuration for the solver.
    demean : bool, optional
        If True, demean returns before covariance estimation (default True).

    Returns
    -------
    weights : Series
        Risk-parity weights (index = column names).
    """
    if returns.empty:
        raise ValueError("returns DataFrame is empty")

    data = returns.copy()
    if demean:
        data = data - data.mean()

    cov = data.cov()
    return risk_parity_weights_series(cov, config=config)


# =====================================================================
# 4. Risk Contributions & Diagnostics
# =====================================================================


def compute_risk_contributions(
    weights: Union[np.ndarray, pd.Series],
    cov: Union[pd.DataFrame, np.ndarray],
) -> pd.DataFrame:
    """
    Compute marginal and total risk contributions for a portfolio.

    Returns a DataFrame with columns:
        - weight: portfolio weight w_i
        - mrc:    marginal risk contribution (∂σ_p/∂w_i)
        - rc:     total risk contribution (w_i * mrc_i)
        - pct_rc: percentage of total risk contributed by asset i
    """
    if isinstance(weights, pd.Series):
        w = np.asarray(weights.values, dtype=float)
        index = list(weights.index)
    else:
        w = np.asarray(weights, dtype=float)
        index = [f"asset_{i}" for i in range(len(w))]

    Σ = _regularize_cov(_ensure_symmetric(_to_cov_matrix(cov)))
    sigma_p = float(np.sqrt(w @ Σ @ w))
    if sigma_p <= 0:
        raise ValueError("Portfolio std is non-positive")

    mrc = (Σ @ w) / sigma_p
    rc = w * mrc
    total_rc = rc.sum()
    pct_rc = rc / total_rc if total_rc != 0 else np.zeros_like(rc)

    return pd.DataFrame(
        {
            "weight": w,
            "mrc": mrc,
            "rc": rc,
            "pct_rc": pct_rc,
        },
        index=index,
    )


def risk_parity_diagnostics(
    weights: Union[np.ndarray, pd.Series],
    cov: Union[pd.DataFrame, np.ndarray],
) -> Dict[str, Any]:
    """
    Return a diagnostics dict for a given portfolio.

    Useful for logging / debugging / integration with dashboards.
    """
    contrib = compute_risk_contributions(weights, cov)
    w = np.asarray(weights, dtype=float)
    Σ = _ensure_symmetric(_to_cov_matrix(cov))
    portfolio_std = float(np.sqrt(w @ Σ @ w))
    return {
        "portfolio_std": portfolio_std,
        "contributions": contrib,
        "max_contribution_pct": float(contrib["pct_rc"].max()),
        "min_contribution_pct": float(contrib["pct_rc"].min()),
    }


# =====================================================================
# 5. Parameter-level risk parity
# =====================================================================


def _extract_numeric_params(param_dict: Mapping[str, object]) -> Tuple[List[str], np.ndarray]:
    """Extract numeric parameters (finite values) from any mapping."""
    keys: List[str] = []
    vals: List[float] = []

    for k, v in param_dict.items():
        try:
            val = float(v)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        keys.append(k)
        vals.append(val)

    return keys, np.asarray(vals, dtype=float)


def _compute_param_weights(
    values: np.ndarray,
    *,
    mode: str = "inverse_deviation",
    eps: float = 1e-8,
    min_weight: float = 0.0,
) -> np.ndarray:
    """
    Compute risk-parity-style weights for parameter values.

    Parameters
    ----------
    values : np.ndarray
        Raw numeric parameter values.
    mode : {"inverse_deviation", "inverse_abs", "equal"}
        - "inverse_deviation": weights ∝ 1 / |v - mean(v)|
        - "inverse_abs":      weights ∝ 1 / |v|
        - "equal":            uniform weights
    eps : float
        Numerical stability term.
    min_weight : float
        Minimum weight after normalization.
    """
    n = len(values)
    if n == 0:
        raise ValueError("_compute_param_weights: empty values array")

    if mode == "equal":
        raw = np.ones(n, dtype=float)
    elif mode == "inverse_deviation":
        dev = np.abs(values - values.mean())
        raw = 1.0 / (dev + eps)
    elif mode == "inverse_abs":
        raw = 1.0 / (np.abs(values) + eps)
    else:
        raise ValueError(f"Unknown param risk parity mode: {mode!r}")

    if not np.all(np.isfinite(raw)) or raw.sum() <= 0:
        logger.warning("Invalid raw param weights; falling back to equal")
        raw = np.ones(n, dtype=float)

    w = raw / raw.sum()
    if min_weight > 0.0:
        w = np.maximum(w, min_weight)
        w = w / w.sum()

    return w


def apply_risk_parity_to_params(
    param_dict: Mapping[str, object],
    *,
    mode: str = "inverse_deviation",
    min_weight: float = 0.0,
    exclude_keys: Optional[Iterable[str]] = None,
    return_weights: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]]]:
    """
    Apply risk-parity style scaling to a parameter dictionary.

    Idea
    ----
    We view each numeric parameter as a "risk dimension". The goal is not
    to enforce strict portfolio risk parity, but to avoid configurations
    where one parameter is effectively dominating the search just because
    its numerical scale is much larger.

    Parameters
    ----------
    param_dict : Mapping[str, object]
        Arbitrary dictionary of parameters.
    mode : {"inverse_deviation", "inverse_abs", "equal"}
        Strategy for computing relative weights.
    min_weight : float, optional
        Minimum weight (after normalization) for each numeric parameter.
    exclude_keys : Iterable[str], optional
        Parameter names to ignore when computing numeric weights (e.g.
        "seed", "label", "name", etc.).
    return_weights : bool, optional
        If True, also return a dict of parameter weights.

    Returns
    -------
    scaled_params : Dict[str, float]
        Parameter dictionary with numeric values scaled by their weights.
    weights : Dict[str, float], optional
        Only returned when return_weights=True.
    """
    exclude = set(exclude_keys or [])
    filtered: Dict[str, object] = {k: v for k, v in param_dict.items() if k not in exclude}

    keys, vals = _extract_numeric_params(filtered)

    if not vals.size:
        logger.debug("apply_risk_parity_to_params: no numeric params found")
        if return_weights:
            return dict(param_dict), {}
        return dict(param_dict)

    if np.isclose(vals.std(), 0.0) and mode != "equal":
        logger.debug("apply_risk_parity_to_params: std≈0 → switching to equal mode")
        mode_effective = "equal"
    else:
        mode_effective = mode

    weights_arr = _compute_param_weights(vals, mode=mode_effective, min_weight=min_weight)

    scaled: Dict[str, float] = dict(param_dict)
    weight_dict: Dict[str, float] = {}

    for k, v, w_param in zip(keys, vals, weights_arr):
        scaled_val = float(v * w_param)
        scaled[k] = scaled_val
        weight_dict[k] = float(w_param)

    if return_weights:
        return scaled, weight_dict
    return scaled


# =====================================================================
# 6. Public exports
# =====================================================================

__all__ = [
    "RiskParityConfig",
    "risk_parity_naive_weights",
    "risk_parity_weights_from_cov",
    "risk_parity_weights_series",
    "risk_parity_from_returns",
    "compute_risk_contributions",
    "risk_parity_diagnostics",
    "apply_risk_parity_to_params",
]
