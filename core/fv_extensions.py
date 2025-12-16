"""
core/fv_extensions.py ג€” Advanced helpers for Fair Value Engine

Pure NumPy/Pandas utilities (no SciPy) used optionally by the engine:
- Dynamic (alpha, beta) via Kalman/RLS
- Covariance shrinkage (ridge / "lw" alias)
- Portfolio allocators: ERC (Equal Risk Contribution), HRP (Hierarchical Risk Parity)
- DSR (Deflated Sharpe Ratio) built on PSR
- Robust volatility & z-score helpers (EWMA, MAD, Huber)
- Utility helpers: quantile thresholds, weight caps/renorm, sector caps

All functions are standalone and safe to import. The engine will call them
if present; otherwise it falls back to internal implementations.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Kalman / RLS for timeג€‘varying alpha,beta
# ---------------------------------------------------------------------------

def kalman_alpha_beta(
    y: Sequence[float],
    x: Sequence[float],
    q: float = 1e-6,
    r: float = 1e-3,
    init: Optional[Tuple[float, float]] = None,
    p0: float = 1e3,
    q_alpha_beta_ratio: float = 1.0,
    r_decay: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kalman filter for y_t ג‰ˆ alpha_t + beta_t * x_t (timeג€‘varying ־±,־²).

    Parameters
    ----------
    y, x : arrays of shape (T,)
    q : float
        Base process noise scalar for the state.
    r : float
        Measurement noise variance.
    init : (alpha0, beta0) or None
        Initial state. If None, starts at (0,0).
    p0 : float
        Initial state covariance scale (diffuse prior).
    q_alpha_beta_ratio : float
        Split of process noise between alpha/beta (beta gets q*ratio).
    r_decay : float in (0,1] or None
        If provided, measurement noise decays as R_t = R0 * (r_decay**t).

    Returns
    -------
    (alpha_path, beta_path) : ndarrays of shape (T,)
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = int(y.shape[0])
    if n == 0:
        return np.asarray([]), np.asarray([])

    a_path = np.zeros(n)
    b_path = np.zeros(n)

    theta = np.array([0.0, 0.0]) if init is None else np.array([float(init[0]), float(init[1])])
    P = np.eye(2) * float(p0)
    q_beta = float(q) * float(max(0.0, q_alpha_beta_ratio))
    Q = np.diag([float(q), q_beta])
    R0 = float(r)

    I = np.eye(2)
    for t in range(n):
        xt = float(x[t])
        H = np.array([[1.0, xt]], dtype=float)  # observation row
        # predict
        P_pred = P + Q
        # adapt R if decaying
        Rt = R0 if (r_decay is None) else (R0 * (float(r_decay) ** t))
        # innovation
        y_pred = float(H @ theta)
        e = float(y[t] - y_pred)
        S = float(H @ P_pred @ H.T + Rt)
        K = (P_pred @ H.T) / S
        # update
        theta = theta + (K.flatten() * e)
        P = (I - K @ H) @ P_pred
        a_path[t], b_path[t] = theta[0], theta[1]

    return a_path, b_path


# ---------------------------------------------------------------------------
# Covariance shrinkage
# ---------------------------------------------------------------------------

def cov_shrinkage(
    R: np.ndarray | pd.DataFrame,
    method: str = "ridge",
    lam: float = 0.2,
) -> np.ndarray:
    """Shrinked covariance of returns matrix R (T x N).

    method="ridge": ־£_shrunk = (1-־»)ֲ·־£_sample + ־»ֲ·diag(diag(־£_sample))
    method="lw": alias to ridge here (no SciPy); set ־» via config.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] < 2:
        n = R.shape[1] if R.ndim == 2 else 1
        return np.eye(n)
    S = np.cov(R, rowvar=False, ddof=1)
    lam = float(np.clip(lam, 0.0, 1.0))
    if method.lower() in ("ridge", "lw"):
        D = np.diag(np.diag(S))
        return (1.0 - lam) * S + lam * D
    # fallback
    return S


# ---------------------------------------------------------------------------
# ERC ג€” Equal Risk Contribution weights
# ---------------------------------------------------------------------------

def erc_weights_from_cov(C: np.ndarray, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Compute ERC portfolio weights from covariance matrix C (N x N).

    Iterative proportional update on risk contributions; returns wג‰¥0, sum(w)=1.
    Fallback: equal weights if numerical issues.
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    if n == 1:
        return np.array([1.0])
    w = np.ones(n) / n
    for _ in range(max_iter):
        mrc = C @ w
        port_var = float(w @ mrc)
        if not np.isfinite(port_var) or port_var <= 0:
            break
        rc = w * mrc
        target = port_var / n
        w_new = w * (target / (rc + 1e-12))
        w_new = np.maximum(w_new, 0.0)
        s = float(w_new.sum())
        if s <= 0:
            break
        w_new /= s
        if np.linalg.norm(w_new - w, ord=1) < tol:
            w = w_new
            break
        w = w_new
    w = np.maximum(w, 0.0)
    return w / (w.sum() if w.sum() > 0 else 1.0)


# ---------------------------------------------------------------------------
# HRP ג€” Hierarchical Risk Parity (simple, SciPyג€‘free)
# ---------------------------------------------------------------------------

def _corr_from_cov(_C: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(_C), 1e-12, np.inf))
    Dinv = np.diag(1.0 / d)
    return Dinv @ _C @ Dinv


def _quasi_diag_order(_C: np.ndarray) -> np.ndarray:
    R = _corr_from_cov(_C)
    vals, vecs = np.linalg.eigh(R)
    v1 = vecs[:, -1]
    return np.argsort(v1)


def _cluster_variance(C: np.ndarray, idx: np.ndarray) -> float:
    sub = C[np.ix_(idx, idx)]
    w = 1.0 / np.clip(np.diag(sub), 1e-8, np.inf)
    w /= w.sum()
    return float(w @ sub @ w)


def hrp_weights_from_cov(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    order = _quasi_diag_order(C)
    idx = order.tolist()

    def _bisect(indices: List[int]) -> np.ndarray:
        if len(indices) == 1:
            return np.array([indices[0]])
        k = len(indices) // 2
        left = _bisect(indices[:k])
        right = _bisect(indices[k:])
        return np.concatenate([left, right])

    sort_idx = _bisect(idx)
    w = np.ones(n)
    clusters = [sort_idx]
    while clusters:
        cl = clusters.pop(0)
        if cl.size <= 1:
            continue
        k = cl.size // 2
        L = cl[:k]
        R = cl[k:]
        varL = _cluster_variance(C, L)
        varR = _cluster_variance(C, R)
        alpha = 1.0 - varL / (varL + varR)
        w[L] *= alpha
        w[R] *= (1.0 - alpha)
        clusters.extend([L, R])
    w = w / w.sum()
    final = np.zeros(n)
    for i, j in enumerate(sort_idx):
        final[j] = w[i]
    return final


# ---------------------------------------------------------------------------
# DSR ג€” Deflated Sharpe Ratio (approx) and PSR glue
# ---------------------------------------------------------------------------

def _sharpe_annualized(returns: pd.Series) -> float:
    if returns.size < 2:
        return np.nan
    mu = float(returns.mean()) * 252.0
    sd = float(returns.std(ddof=1)) * np.sqrt(252.0)
    return float(mu / sd) if sd > 0 else np.nan

# Normal inverse CDF (Acklamג€‘like) for DSR penalty; no SciPy
_a = [-3.969683028665376e+01,  2.209460984245205e+02,
      -2.759285104469687e+02,  1.383577518672690e+02,
      -3.066479806614716e+01,  2.506628277459239e+00]
_b = [-5.447609879822406e+01,  1.615858368580409e+02,
      -1.556989798598866e+02,  6.680131188771972e+01,
      -1.328068155288572e+01]
_c = [-7.784894002430293e-03, -3.223964580411365e-01,
      -2.400758277161838e+00, -2.549732539343734e+00,
       4.374664141464968e+00,  2.938163982698783e+00]
_d = [ 7.784695709041462e-03,  3.224671290700398e-01,
       2.445134137142996e+00,  3.754408661907416e+00]


def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((_c[0]*q + _c[1])*q + _c[2])*q + _c[3])*q + _c[4])*q + _c[5]) / \
               ((((_d[0]*q + _d[1])*q + _d[2])*q + _d[3])*q + 1)
    if phigh < p:
        q = np.sqrt(-2 * np.log(1-p))
        return -(((((_c[0]*q + _c[1])*q + _c[2])*q + _c[3])*q + _c[4])*q + _c[5]) / \
                  ((((_d[0]*q + _d[1])*q + _d[2])*q + _d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((_a[0]*r + _a[1])*r + _a[2])*r + _a[3])*r + _a[4])*r + _a[5])*q / \
           (((((_b[0]*r + _b[1])*r + _b[2])*r + _b[3])*r + _b[4])*r + 1)


def dsr(returns: pd.Series, sr_star: float = 0.0, n_trials: int = 30) -> float:
    """Deflated Sharpe Ratio approximation (Bailey & Lֳ³pez de Prado).

    Steps:
    1) Compute SR (annualized) and PSR Zג€‘equivalent with skew/kurtosis adj.
    2) Penalize for multiple trials via z_alpha = ־¦^{-1}(1 - 1/T).
    3) Return ־¦(Z - z_alpha).
    """
    m = returns.dropna()
    n = m.size
    if n < 3:
        return np.nan
    sr = _sharpe_annualized(m)
    if not np.isfinite(sr):
        return np.nan
    g1 = float(((m - m.mean())**3).mean() / (m.std(ddof=1)**3 + 1e-12))
    g2 = float(((m - m.mean())**4).mean() / (m.var(ddof=1)**2 + 1e-12))
    denom = np.sqrt(1.0 - g1*sr + 0.25*(g2 - 1.0)*(sr**2))
    if denom <= 0:
        return np.nan
    Z = (sr - sr_star) * np.sqrt(n - 1) / denom
    z_alpha = _norm_ppf(1.0 - 1.0 / max(2, int(n_trials)))
    Z_adj = Z - z_alpha
    return 0.5 * (1.0 + np.math.erf(Z_adj / np.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Robust volatility & zג€‘score helpers
# ---------------------------------------------------------------------------

def ewma_std(series: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """EWMA std (J.P. Morgan style) on a series (e.g., returns or diffs)."""
    if not 0.0 < lambda_ < 1.0:
        lambda_ = 0.94
    # pandas ewm std already estimates the mean internally
    return series.ewm(alpha=(1.0 - lambda_), adjust=False).std(bias=False)


def mad_z(x_last: float, series: pd.Series) -> float:
    """MADג€‘based zג€‘score: (x - median)/ (1.4826*MAD)."""
    med = float(series.median())
    mad = float((series - med).abs().median())
    denom = 1.4826 * mad if mad > 0 else np.nan
    return float((x_last - med) / denom) if np.isfinite(denom) else np.nan


def huber_z(x_last: float, series: pd.Series, c: float = 1.345, iters: int = 25) -> float:
    """Huber Mג€‘estimator z using IRLS; returns (x_last - mu_hat)/sigma_hat."""
    x = series.dropna().values.astype(float)
    if x.size < 3:
        return np.nan
    mu = float(np.median(x))
    sigma = float(1.4826 * np.median(np.abs(x - mu))) or float(np.std(x, ddof=1) or 1.0)
    for _ in range(iters):
        t = (x - mu) / (sigma + 1e-12)
        w = np.clip(c / np.abs(t + 1e-12), 0.0, 1.0)  # Huber weights
        mu_new = float(np.sum(w * x) / np.sum(w))
        sigma_new = float(np.sqrt(np.sum(w * (x - mu_new)**2) / np.sum(w)))
        if not (np.isfinite(mu_new) and np.isfinite(sigma_new)):
            break
        if abs(mu_new - mu) < 1e-10 and abs(sigma_new - sigma) < 1e-10:
            mu, sigma = mu_new, sigma_new
            break
        mu, sigma = mu_new, sigma_new
    return float((x_last - mu) / (sigma if sigma > 0 else np.nan))


def robust_z_from_series(x_last: float, series: pd.Series, mode: str = "std", huber_c: float = 1.345) -> float:
    mode = (mode or "std").lower()
    if mode == "mad":
        return mad_z(x_last, series)
    if mode == "huber":
        return huber_z(x_last, series, c=huber_c)
    # std
    mu = float(series.mean())
    sd = float(series.std(ddof=1))
    return float((x_last - mu) / sd) if sd > 0 else np.nan


# ---------------------------------------------------------------------------
# Thresholds & weight utilities
# ---------------------------------------------------------------------------

def quantile_thresholds(z_series: pd.Series, z_in_q: float, z_out_q: float) -> Tuple[float, float]:
    """Compute dynamic entry/exit thresholds from |Z| quantiles (within window)."""
    if z_series.dropna().empty:
        return np.nan, np.nan
    absz = z_series.abs().dropna()
    q_in = float(absz.quantile(np.clip(z_in_q, 0.5, 0.999)))
    q_out = float(absz.quantile(np.clip(z_out_q, 0.0, min(0.98, z_in_q - 1e-3))))
    return q_in, q_out


def cap_and_renorm(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Cap each w_i at max_weight, then renormalize to sum=1."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w
    cap = float(max_weight)
    w = np.clip(w, 0.0, cap)
    s = float(w.sum())
    return w / s if s > 0 else np.ones_like(w) / w.size


def apply_sector_cap(weights: np.ndarray, sectors: Sequence[int], cap: float) -> np.ndarray:
    """Apply perג€‘sector cap; renormalize. `sectors` must map each asset to a sector id."""
    w = np.asarray(weights, dtype=float)
    sectors = np.asarray(sectors)
    out = w.copy()
    for sec in np.unique(sectors):
        idx = np.where(sectors == sec)[0]
        sec_sum = float(out[idx].sum())
        lim = float(cap)
        if sec_sum > lim:
            out[idx] *= lim / sec_sum
    s = float(out.sum())
    return out / s if s > 0 else np.ones_like(out) / out.size


__all__ = [
    # Kalman
    "kalman_alpha_beta",
    # Covariance & allocators
    "cov_shrinkage", "erc_weights_from_cov", "hrp_weights_from_cov",
    # DSR
    "dsr",
    # Robust vol & z
    "ewma_std", "mad_z", "huber_z", "robust_z_from_series",
    # Thresholds / weights utilities
    "quantile_thresholds", "cap_and_renorm", "apply_sector_cap",
]
