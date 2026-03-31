# -*- coding: utf-8 -*-
"""
common/risk_helpers.py ג€” Risk plumbing for macro/pairs engines (MVP)
--------------------------------------------------------------------
Utility functions for portfolio risk: realized/exp vol, simple shrinkage
covariance, risk-parity weights, target-vol sizing, portfolio VaR/ES (CVaR),
position caps, and a stateful drawdown gate.

Design goals
------------
- Pure numpy/pandas, no heavy deps. Works with Python 3.9+.
- Deterministic, sideג€‘effect free (except DrawdownGate state).
- Small, documented, and easy to swap for a more advanced library later.

All functions expect clean inputs (NaN-safe); call `.dropna()` upfront if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple, List
import numpy as np
import pandas as pd

# ---------------------------- Volatility & Covariance -------------------------

def realized_vol(r: pd.Series, window: int = 20, annualize: bool = True) -> float:
    """Rolling realized volatility of simple returns over `window`.
    Returns last available vol; if insufficient data, returns NaN.
    """
    r = pd.Series(r).dropna()
    if len(r) < max(2, window):
        return np.nan
    vol = r.rolling(window).std(ddof=0).iloc[-1]
    return float(vol * np.sqrt(252.0)) if annualize else float(vol)


def ewma_vol(r: pd.Series, span: int = 20, annualize: bool = True) -> float:
    """EWMA volatility (RiskMetrics-style) of simple returns.
    """
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return np.nan
    var = r.ewm(span=span, adjust=False).var(bias=False).iloc[-1]
    return float(np.sqrt(var) * (252.0 ** 0.5) if annualize else np.sqrt(var))


def shrink_cov(R: pd.DataFrame, window: int = 60, shrink: float = 0.0) -> pd.DataFrame:
    """Simple rolling covariance with optional scalar shrinkage to identity.
    `R` is Tֳ—N matrix of returns. Returns Nֳ—N covariance matrix.
    """
    R = R.dropna().astype(float)
    if len(R) < max(2, window):
        return R.cov()
    S = R.tail(window).cov()
    if shrink > 0:
        I = np.eye(S.shape[0])
        S = (1 - shrink) * S + shrink * (np.trace(S) / S.shape[0]) * pd.DataFrame(I, index=S.index, columns=S.columns)
    return S

# ---------------------------- Sizing & Weights -------------------------------

def target_vol_weights(vols: pd.Series, target_vol: float, max_leverage: float = 1.0,
                       floor: float = 0.0, cap: float = 1.0) -> pd.Series:
    """Inverse-vol weights scaled to hit `target_vol` (approx.).
    `vols` is per-asset annualized vol. Returns weights summing to 1 (after caps).
    """
    w_raw = 1.0 / vols.replace(0, np.nan)
    w_raw = w_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if w_raw.sum() == 0:
        return pd.Series(0.0, index=vols.index)
    w = w_raw / w_raw.sum()
    # enforce caps/floors
    w = w.clip(lower=floor)
    if cap < 1.0:
        excess = (w - cap).clip(lower=0.0)
        if excess.sum() > 0:
            w = w - excess + (excess.sum() * (w < cap) / max((w < cap).sum(), 1))
    # normalize and scale to target leverage
    w = w / w.sum()
    return w * min(max_leverage, 1.0)


def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 500, tol: float = 1e-8) -> pd.Series:
    """Simple iterative risk-parity (equal risk contribution) weights.
    Returns weights summing to 1. Assumes PSD `cov`.
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        m = cov.values @ w
        rc = w * m  # risk contributions
        avg = rc.mean()
        grad = (rc - avg)
        step = 0.5 / (np.linalg.norm(grad) + 1e-12)
        w_new = np.maximum(w - step * grad, 1e-12)
        w_new /= w_new.sum()
        if np.linalg.norm(w_new - w, ord=1) < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=cov.index)


def apply_caps(weights: pd.Series, caps: Mapping[str, float], groups: Mapping[str, str]) -> pd.Series:
    """Apply per-group caps to weights.
    `weights` indexed by asset id; `groups` maps asset -> group key; `caps` maps group -> max weight.
    Excess over a group cap is redistributed proportionally to uncapped names.
    """
    w = weights.copy().clip(lower=0.0)
    w /= max(w.sum(), 1e-12)
    if not caps:
        return w
    # compute group sums
    df = pd.DataFrame({"w": w, "g": [groups.get(i, "__ungrouped__") for i in w.index]})
    for g, cap in caps.items():
        mask = df["g"] == g
        over = df.loc[mask, "w"].sum() - cap
        if over > 1e-12:
            # zero out proportional share of overweight and redistribute to others
            take = df.loc[mask, "w"]
            reduce = (take / take.sum()) * over
            df.loc[mask, "w"] = (take - reduce).clip(lower=0.0)
            # redistribute to non-mask
            rest_mask = ~mask
            if rest_mask.any():
                df.loc[rest_mask, "w"] += df.loc[rest_mask, "w"] / df.loc[rest_mask, "w"].sum() * over
    df["w"] /= max(df["w"].sum(), 1e-12)
    return df["w"].reindex(weights.index).fillna(0.0)

# ---------------------------- VaR / ES (CVaR) --------------------------------

def portfolio_var(weights: pd.Series, cov: pd.DataFrame, alpha: float = 0.95) -> float:
    """Parametric VaR (normal) at level `alpha` using covariance matrix (annualized).
    Returns annualized VaR (positive number)."""
    w = weights.values.reshape(-1, 1)
    sigma = float(np.sqrt((w.T @ cov.values @ w).item()))
    # Gaussian quantile (one-sided)
    try:
        from scipy.stats import norm  # type: ignore
        z = float(norm.ppf(alpha))
    except Exception:
        # simple approx for common alphas
        z_map = {0.95: 1.64485, 0.975: 1.95996, 0.99: 2.32635}
        z = z_map.get(round(alpha, 3), 1.645)
    return z * sigma


def cvar_historical(portfolio_returns: pd.Series, alpha: float = 0.95) -> float:
    """Historical Expected Shortfall (CVaR) at level `alpha` on (non-annualized) portfolio returns.
    Returns positive number (magnitude of expected loss in the tail)."""
    x = pd.Series(portfolio_returns).dropna()
    if x.empty:
        return np.nan
    q = x.quantile(1 - alpha)
    tail = x[x <= q]
    if tail.empty:
        return 0.0
    return float(-tail.mean())

# ---------------------------- Drawdown gate ----------------------------------

@dataclass
class DrawdownGate:
    """Stateful drawdown gate with cooldown.

    Parameters
    ----------
    threshold : float
        Max tolerated drawdown (e.g., 0.1 for 10%). If breached, gate triggers.
    cooloff : int
        Minimum number of subsequent observations to wait before resetting.
    """
    threshold: float = 0.1
    cooloff: int = 5

    # internal state
    _peak: float = 1.0
    _cool: int = 0

    def reset(self) -> None:
        self._peak = 1.0
        self._cool = 0

    def update(self, equity: float) -> Tuple[bool, float]:
        """Update with latest equity; return (gate_tripped, drawdown).
        Expects `equity` as cumulative PnL index (starts ~1.0)."""
        if np.isnan(equity):
            return False, 0.0
        self._peak = max(self._peak, float(equity))
        dd = 1.0 - float(equity) / max(self._peak, 1e-12)
        tripped = False
        if self._cool > 0:
            self._cool -= 1
            tripped = True
        elif dd >= self.threshold:
            tripped = True
            self._cool = self.cooloff
        return tripped, dd



# ---------------------------- Additional helpers -----------------------------

def compute_vols(R: pd.DataFrame, method: str = "ewma", window: int = 20, span: int = 20, *, annualize: bool = True) -> pd.Series:
    """Compute per-asset volatility from returns matrix `R`.
    Parameters
    ----------
    R : DataFrame (Tֳ—N) of simple returns
    method : 'ewma' or 'realized'
    window : rolling window for realized vol
    span   : EWMA span for EWMA vol
    annualize : scale by sqrt(252)
    """
    R = R.copy().astype(float)
    vols: Dict[str, float] = {}
    if method == "ewma":
        for c in R.columns:
            vols[c] = ewma_vol(R[c], span=span, annualize=annualize)
    else:
        for c in R.columns:
            vols[c] = realized_vol(R[c], window=window, annualize=annualize)
    return pd.Series(vols)


def target_weights_from_returns(R: pd.DataFrame, *, target_vol: float = 0.1, method: str = "invvol",
                                cov_window: int = 60, shrink: float = 0.1, max_leverage: float = 1.0,
                                floor: float = 0.0, cap: float = 1.0) -> pd.Series:
    """Compute portfolio weights from a returns matrix `R` using either inverse-vol or risk-parity.
    - If method == 'invvol': use inverse of EWMA vol then scale to target leverage.
    - If method == 'erc'   : compute shrinked cov and use risk-parity weights; then scale to max_leverage.
    """
    R = R.dropna().astype(float)
    if R.empty:
        return pd.Series(dtype=float)
    if method.lower() == "erc":
        C = shrink_cov(R, window=cov_window, shrink=shrink)
        w = risk_parity_weights(C)
        return w * max_leverage
    else:
        vols = compute_vols(R, method="ewma", span=max(5, min(cov_window, 60)))
        w = target_vol_weights(vols, target_vol=target_vol, max_leverage=max_leverage, floor=floor, cap=cap)
        return w


def apply_multipliers(weights: pd.Series, multipliers: Mapping[str, float], *, cap: float = 1.0,
                      floor: float = 0.0, renorm: bool = True) -> pd.Series:
    """Apply multiplicative adjustments per asset and (optionally) renormalize.
    Unknown assets keep their original weight. Caps/Floors applied post-scaled.
    """
    w = weights.copy().astype(float)
    for k, m in multipliers.items():
        if k in w.index and pd.notna(m):
            w.loc[k] = w.loc[k] * float(m)
    # enforce bounds & renorm
    w = w.clip(lower=floor)
    if cap < 1.0:
        excess = (w - cap).clip(lower=0.0)
        if excess.sum() > 0:
            w = w - excess + (excess.sum() * (w < cap) / max((w < cap).sum(), 1))
    if renorm:
        s = w.sum()
        if s > 0:
            w = w / s
    return w


def max_drawdown(x: pd.Series, *, from_returns: bool = True) -> Tuple[float, pd.Series]:
    """Compute max drawdown and drawdown series.
    If `from_returns` is True, `x` is simple returns; otherwise `x` is equity index.
    Returns (max_drawdown, drawdown_series)."""
    x = pd.Series(x).dropna()
    if x.empty:
        return np.nan, pd.Series(dtype=float)
    if from_returns:
        eq = (1.0 + x).cumprod()
    else:
        eq = x.copy()
    peak = eq.cummax()
    dd = 1.0 - (eq / peak).clip(upper=np.inf)
    return float(dd.max()), dd


def portfolio_metrics(weights: pd.Series, returns: pd.DataFrame, *, alpha: float = 0.95,
                      cov_window: int = 60, shrink: float = 0.1) -> Dict[str, float]:
    """Compute basic portfolio risk metrics using historical returns.
    Returns dict with annualized vol, mean return, Sharpe (naive), VaR, ES, max drawdown.
    """
    w = weights.reindex(returns.columns).fillna(0.0)  # ensure alignment
    r = (returns.fillna(0.0) * w).sum(axis=1)
    vol = realized_vol(r, window=cov_window, annualize=True)
    mu = float(r.mean() * 252.0)
    sharpe = float(mu / (vol + 1e-12))
    C = shrink_cov(returns, window=cov_window, shrink=shrink)
    var = portfolio_var(w, C, alpha=alpha)
    es = cvar_historical(r, alpha=alpha)
    mdd, _ = max_drawdown(r, from_returns=True)
    return {
        "ann_vol": float(vol),
        "ann_return": float(mu),
        "sharpe": float(sharpe),
        "var": float(var),
        "es": float(es),
        "max_dd": float(mdd),
    }

# ---------------------------- Extras: normalization & turnover ---------------

def normalize_weights(w: pd.Series, *, method: str = "sum") -> pd.Series:
    """Normalize weights to sum to 1 (or other schemes later)."""
    w = pd.Series(w).astype(float).fillna(0.0)
    s = w.sum()
    return w / s if s != 0 else w


def portfolio_turnover(w_prev: pd.Series, w_curr: pd.Series) -> float:
    """Compute L1 turnover between two weight vectors (sum |־”w|)."""
    a = pd.Series(w_prev).astype(float).fillna(0.0)
    b = pd.Series(w_curr).astype(float).fillna(0.0).reindex(a.index, fill_value=0.0)
    return float(np.abs(a - b).sum())


def exp_cov(R: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """Exponential covariance estimator (component-wise EWMA)."""
    R = R.astype(float).dropna()
    if R.empty:
        return R.cov()
    demean = R - R.ewm(span=span, adjust=False).mean()
    return (demean.ewm(span=span, adjust=False).cov().dropna()
            .groupby(level=1).tail(1).droplevel(0))


def beta_to_index(returns: pd.DataFrame, index: pd.Series) -> pd.Series:
    """OLS beta of each column in `returns` to `index` (no intercept)."""
    X = pd.Series(index).astype(float).dropna()
    betas: Dict[str, float] = {}
    for c in returns.columns:
        y = returns[c].reindex(X.index).astype(float).fillna(0.0)
        num = float((y * X).mean())
        den = float((X * X).mean()) + 1e-12
        betas[c] = num / den
    return pd.Series(betas)


def marginal_contrib_to_risk(cov: pd.DataFrame, w: pd.Series) -> Tuple[pd.Series, pd.Series, float]:
    """Return (MRC, RC, portfolio_vol). MRC = (־£w)_i / vol; RC = w_i*MRC_i."""
    wv = pd.Series(w).astype(float).reindex(cov.index).fillna(0.0)
    sig = float(np.sqrt((wv.values @ cov.values @ wv.values))) + 1e-12
    mrc = pd.Series((cov.values @ wv.values) / sig, index=cov.index)
    rc = wv * mrc
    return mrc, rc, sig


def decompose_risk(cov: pd.DataFrame, w: pd.Series) -> Dict[str, object]:
    mrc, rc, sig = marginal_contrib_to_risk(cov, w)
    return {"vol": float(sig), "mrc": mrc, "rc": rc, "rc_pct": rc / max(rc.sum(), 1e-12)}


__all__ = [
    # vol/cov
    "realized_vol", "ewma_vol", "shrink_cov", "exp_cov",
    # sizing/weights
    "risk_parity_weights", "target_vol_weights", "normalize_weights", "apply_caps",
    "compute_vols", "target_weights_from_returns", "apply_multipliers",
    # risk
    "portfolio_var", "cvar_historical", "max_drawdown", "portfolio_metrics",
    "marginal_contrib_to_risk", "decompose_risk", "beta_to_index", "portfolio_turnover",
    # gate
    "DrawdownGate",
]
