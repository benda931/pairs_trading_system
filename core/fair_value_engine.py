"""
Fair Value Engine ג€” final, canvasג€‘editable

Stable, extensible fairג€‘value engine for pairs trading with:
- Robust OLS (numpy lstsq) on linear/log domain
- Diagnostics: spread/z/HL/bands/corr/dCor/ADF/Hurst/coint (incl. Residualג€‘ADF)
- Volג€‘adjusted mispricing; costs; hysteresis; inג€‘window P&L; PSR/DSR
- Position sizing via target vol + quality weight + Halfג€‘Kelly
- Portfolio layer: invג€‘vol / ERC / HRP with covariance shrinkage
- Optional dynamic (alpha,beta) via Kalman (RLSג€‘style)
- Clean SECTION MARKERS for point edits in canvas

No heavy deps (numpy/pandas only). Optional hooks to `params`,
`matrix_helpers`, `advanced_metrics`, and `core.fv_extensions` (if present).
"""

from __future__ import annotations

# === FV:IMPORTS START ===
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd

# Optional system modules (if present in your repo)
try:
    import params  # type: ignore
except Exception:  # noqa: E722
    params = None  # type: ignore

try:
    import matrix_helpers  # type: ignore
except Exception:  # noqa: E722
    matrix_helpers = None  # type: ignore

try:
    import advanced_metrics  # type: ignore
except Exception:  # noqa: E722
    advanced_metrics = None  # type: ignore

# Logger (lightweight default)
logger = logging.getLogger("fair_value_engine")
if not logger.handlers:
    _h = logging.StreamHandler()
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Optional advanced extensions (kalman, shrinkage, ERC/HRP, DSR)
try:
    from core import fv_extensions as fvx  # preferred package path
except Exception:  # noqa: E722
    try:
        import fv_extensions as fvx  # fallback if same folder
    except Exception:  # noqa: E722
        fvx = None  # type: ignore
# === FV:IMPORTS END ===


# === FV:CONFIG START ===
@dataclass
class Config:
    """Configuration for the FairValueEngine. Override via `params` if present."""
    # Core
    window: int = 252
    min_overlap: int = 60
    log_mode: bool = True

    # Diagnostics / stability
    secondary_windows: Tuple[int, ...] = field(default_factory=lambda: (63, 126))
    use_winsor: bool = False
    winsor_p: float = 0.01
    zscore_clip: Tuple[float, float] = (-6.0, 6.0)
    volatility_adjust: bool = True  # mispricing / spread vol

    # Correlation/metrics inputs
    use_returns_for_corr: bool = True
    use_returns_for_dcor: bool = True

    # Ensemble controls
    ensemble_mode: str = "none"   # "none" | "weighted"
    ensemble_target: str = "zscore"  # or "mispricing"

    # Meanג€‘reversion criteria
    mean_revert_pvalue: float = 0.05

    # Cointegration (Residualג€‘ADF)
    residual_adf_enabled: bool = True
    coint_pvalue: float = 0.05

    # Beta / Kalman
    beta_mode: str = "static"        # "static" | "kalman"
    kalman_q: float = 1e-6
    kalman_r: float = 1e-3

    # Portfolio & covariance
    rp_method: str = "erc"           # "invvol" | "erc" | "hrp"
    cov_method: str = "ridge"        # "sample" | "ridge" | "lw"
    cov_shrink_lambda: float = 0.2

    # Costs & hysteresis
    costs_bps: float = 1.0         # per leg
    slippage_bps: float = 1.0      # per leg
    borrow_bps: float = 0.0        # annualized proxy
    z_in: float = 1.0              # enter threshold
    z_out: float = 0.5             # exit threshold (hysteresis)

    # Sizing (target vol + Halfג€‘Kelly)
    target_vol_ann: float = 0.10   # 10% annualized target vol per pair position
    kelly_fraction: float = 0.5    # halfג€‘Kelly
    max_leverage: float = 5.0

    # Evaluation (PSR/DSR)
    psr_sr_star: float = 0.0
    dsr_n_trials: int = 30

    # Universe
    pairs: List[Tuple[str, str]] = field(default_factory=list)

    # Providers (hooks only; disabled if no provider injected)
    data_provider: Optional[str] = None
    data_api_endpoint: Optional[str] = None
    data_api_key: Optional[str] = None
    data_period: Optional[str] = None
    data_interval: Optional[str] = None


def _load_config_from_params() -> Config:
    cfg = Config()
    if params is None:
        return cfg

    def getp(name: str, default: Any) -> Any:
        return getattr(params, name, default)

    # Core
    cfg.window = int(getp("FAIR_VALUE_WINDOW", cfg.window))
    cfg.min_overlap = int(getp("FAIR_VALUE_MIN_OVERLAP", cfg.min_overlap))
    cfg.log_mode = bool(getp("FAIR_VALUE_LOG_MODE", cfg.log_mode))
    sec = getp("FAIR_VALUE_SECONDARY_WINDOWS", cfg.secondary_windows)
    if isinstance(sec, (list, tuple)) and len(sec) > 0:
        cfg.secondary_windows = tuple(int(x) for x in sec if int(x) > 0)

    # Robustness
    cfg.use_winsor = bool(getp("FAIR_VALUE_WINSOR", cfg.use_winsor))
    cfg.winsor_p = float(getp("FAIR_VALUE_WINSOR_P", cfg.winsor_p))
    cfg.zscore_clip = tuple(getp("FAIR_VALUE_Z_CLIP", cfg.zscore_clip))  # type: ignore
    cfg.volatility_adjust = bool(getp("FAIR_VALUE_VOL_ADJ", cfg.volatility_adjust))

    # Corr / dCor inputs
    cfg.use_returns_for_corr = bool(getp("FAIR_VALUE_USE_RET_FOR_CORR", cfg.use_returns_for_corr))
    cfg.use_returns_for_dcor = bool(getp("FAIR_VALUE_USE_RET_FOR_DCOR", cfg.use_returns_for_dcor))

    # Ensemble
    cfg.ensemble_mode = str(getp("FAIR_VALUE_ENSEMBLE_MODE", cfg.ensemble_mode))
    cfg.ensemble_target = str(getp("FAIR_VALUE_ENSEMBLE_TARGET", cfg.ensemble_target))

    # Mean reversion
    cfg.mean_revert_pvalue = float(getp("FAIR_VALUE_MR_PVALUE", cfg.mean_revert_pvalue))

    # Cointegration
    cfg.residual_adf_enabled = bool(getp("FAIR_VALUE_RESIDUAL_ADF", cfg.residual_adf_enabled))
    cfg.coint_pvalue = float(getp("FAIR_VALUE_COINT_PVALUE", cfg.coint_pvalue))

    # Beta / Kalman
    cfg.beta_mode = str(getp("FAIR_VALUE_BETA_MODE", cfg.beta_mode))
    cfg.kalman_q = float(getp("FAIR_VALUE_KALMAN_Q", cfg.kalman_q))
    cfg.kalman_r = float(getp("FAIR_VALUE_KALMAN_R", cfg.kalman_r))

    # Portfolio & covariance
    cfg.rp_method = str(getp("FAIR_VALUE_RP_METHOD", cfg.rp_method))
    cfg.cov_method = str(getp("FAIR_VALUE_COV_METHOD", cfg.cov_method))
    cfg.cov_shrink_lambda = float(getp("FAIR_VALUE_COV_SHRINK_LAMBDA", cfg.cov_shrink_lambda))

    # Costs & sizing
    cfg.costs_bps = float(getp("FAIR_VALUE_COSTS_BPS", cfg.costs_bps))
    cfg.slippage_bps = float(getp("FAIR_VALUE_SLIPPAGE_BPS", cfg.slippage_bps))
    cfg.borrow_bps = float(getp("FAIR_VALUE_BORROW_BPS", cfg.borrow_bps))
    cfg.z_in = float(getp("FAIR_VALUE_Z_IN", cfg.z_in))
    cfg.z_out = float(getp("FAIR_VALUE_Z_OUT", cfg.z_out))
    cfg.target_vol_ann = float(getp("FAIR_VALUE_TARGET_VOL", cfg.target_vol_ann))
    cfg.kelly_fraction = float(getp("FAIR_VALUE_KELLY_FRAC", cfg.kelly_fraction))
    cfg.max_leverage = float(getp("FAIR_VALUE_MAX_LEV", cfg.max_leverage))

    # Evaluation
    cfg.psr_sr_star = float(getp("FAIR_VALUE_PSR_SR_STAR", cfg.psr_sr_star))
    cfg.dsr_n_trials = int(getp("FAIR_VALUE_DSR_TRIALS", cfg.dsr_n_trials))

    # Universe
    pairs_param = getp("PAIRS", None)
    pairs_matrix = getp("PAIRS_MATRIX", None)
    pairs: List[Tuple[str, str]] = []
    if isinstance(pairs_param, (list, tuple)):
        for p in pairs_param:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pairs.append((str(p[0]), str(p[1])))
            elif isinstance(p, dict) and {"y", "x"} <= set(p):
                pairs.append((str(p["y"]), str(p["x"])))
    elif isinstance(pairs_matrix, (list, tuple)):
        for row in pairs_matrix:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                pairs.append((str(row[0]), str(row[1])))
    cfg.pairs = pairs

    # data hooks
    cfg.data_provider = getp("DATA_PROVIDER", cfg.data_provider)
    cfg.data_api_endpoint = getp("DATA_API_ENDPOINT", cfg.data_api_endpoint)
    cfg.data_api_key = getp("DATA_API_KEY", cfg.data_api_key)
    cfg.data_period = getp("DATA_PERIOD", cfg.data_period)
    cfg.data_interval = getp("DATA_INTERVAL", cfg.data_interval)
    return cfg
# === FV:CONFIG END ===


# === FV:CORE_MATH START ===

def _safe_lstsq(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (alpha, beta) solving y ג‰ˆ a + bֲ·x via least squares."""
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(theta[0]), float(theta[1])


def _ols_alpha_beta(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    X = np.column_stack([np.ones_like(x), x])
    return _safe_lstsq(X, y)


def _winsorize(s: pd.Series, p: float) -> pd.Series:
    if p <= 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def _estimate_halflife(spread: pd.Series) -> float:
    s = spread.dropna()
    if s.size < 20:
        return np.nan
    lag = s.shift(1).dropna()
    delta = (s - lag).dropna()
    lag = lag.loc[delta.index]
    X = np.column_stack([np.ones_like(lag.values), lag.values])
    a, b = _safe_lstsq(X, delta.values)
    phi = 1.0 + b
    if not (0.0 < phi < 1.0):
        return float("inf")
    try:
        return float(-math.log(2.0) / math.log(phi))
    except Exception:
        return np.nan


def _rolling_corr(y: pd.Series, x: pd.Series) -> float:
    window = len(y)
    if matrix_helpers is not None and hasattr(matrix_helpers, "rolling_corr"):
        try:
            c = matrix_helpers.rolling_corr(y, x, window=window).iloc[-1]
            return float(c)
        except Exception:
            pass
    return float(y.corr(x))


def _safe_zscore(value: float, mean: float, std: float, clip: Tuple[float, float]) -> float:
    if np.isfinite(std) and std > 0:
        z = (value - mean) / std
        return float(np.clip(z, clip[0], clip[1]))
    return np.nan


def _check_log_safe(y: pd.Series, x: pd.Series) -> bool:
    return (y > 0).all() and (x > 0).all()


def _quality_weight(adf_p: Optional[float], halflife: Optional[float], corr: Optional[float]) -> float:
    """Quality weight (higher is better): (1-p), invג€‘HL, |corr|."""
    w_adf = (1.0 - float(adf_p)) if (adf_p is not None and np.isfinite(adf_p)) else 0.5
    w_adf = float(np.clip(w_adf, 0.0, 1.0))
    w_hl = 1.0 / (1.0 + float(halflife)) if (halflife is not None and np.isfinite(halflife) and halflife > 0) else 0.0
    w_corr = abs(float(corr)) if (corr is not None and np.isfinite(corr)) else 0.0
    return float(0.5 * w_adf + 0.3 * w_hl + 0.2 * w_corr)

# === FV:COSTS (helpers) START ===

def _bps_to_decimal(bps: float) -> float:
    return float(bps) / 1e4


def _spread_trade_notional(y_last: float, x_last: float, beta: float) -> float:
    """Approx notional traded for 1 unit of spread (|Y| + |־²ֲ·X|)."""
    return abs(y_last) + abs(beta) * abs(x_last)


def _spread_cost_units(y_last: float, x_last: float, beta: float,
                       costs_bps: float, slippage_bps: float, borrow_bps: float) -> float:
    """Approx roundג€‘trip cost in price units for 1 unit of spread."""
    notional = _spread_trade_notional(y_last, x_last, beta)
    legs_bps = 2.0 * (_bps_to_decimal(costs_bps + slippage_bps))  # enter+exit
    borrow = _bps_to_decimal(borrow_bps) * 1.0  # simple proxy
    return float(notional * (legs_bps + borrow))
# === FV:COSTS (helpers) END ===

# === FV:SIZING (helpers) START ===

def _ann_vol_of_spread(spread: pd.Series) -> float:
    """Annualized vol estimate of spread changes (first differences)."""
    ds = spread.diff().dropna()
    if ds.size < 2:
        return np.nan
    daily = float(ds.std(ddof=1))
    return float(daily * np.sqrt(252.0))


def _size_from_quality_and_vol(z: float, wq: float, target_vol_ann: float,
                               spread_vol_ann: float, kelly_fraction: float,
                               max_leverage: float) -> float:
    if not (np.isfinite(spread_vol_ann) and spread_vol_ann > 0):
        return 0.0
    base = (target_vol_ann / spread_vol_ann)
    size = float(base * (z / 2.0) * (0.5 + 0.5 * wq) * kelly_fraction)
    return float(np.clip(size, -max_leverage, max_leverage))
# === FV:SIZING (helpers) END ===

# === FV:EVAL (helpers) START ===

def _simulate_hysteresis_pnl(spread: pd.Series, z_series: pd.Series,
                             z_in: float, z_out: float,
                             cost_per_round: float) -> Tuple[pd.Series, int, float]:
    """Simple inג€‘window backtest of unit spread with hysteresis."""
    pos = 0  # -1 short spread, +1 long spread
    pnl = []
    trades = 0
    hold_lengths = []
    hold = 0
    prev = spread.iloc[0]
    for i in range(1, len(spread)):
        zt = z_series.iloc[i]
        if pos == 0:
            if np.isfinite(zt) and zt > z_in:
                pos = -1; trades += 1; pnl.append(-cost_per_round); hold = 0
            elif np.isfinite(zt) and zt < -z_in:
                pos = 1; trades += 1; pnl.append(-cost_per_round); hold = 0
            else:
                pnl.append(0.0)
        else:
            if np.isfinite(zt) and abs(zt) < z_out:
                pos = 0; trades += 1; pnl.append(-cost_per_round)
                if hold > 0: hold_lengths.append(hold); hold = 0
            else:
                cur = spread.iloc[i]
                pnl.append(float((cur - prev) * pos))
                prev = cur
                hold += 1
    avg_hold = float(np.mean(hold_lengths)) if hold_lengths else 0.0
    return pd.Series(pnl, index=spread.index[1:]), trades, avg_hold


def _sharpe_annualized(returns: pd.Series) -> float:
    if returns.size < 2:
        return np.nan
    mu = float(returns.mean()) * 252.0
    sd = float(returns.std(ddof=1)) * np.sqrt(252.0)
    return float(mu / sd) if sd > 0 else np.nan


def _skew_kurt(returns: pd.Series) -> Tuple[float, float]:
    m = returns.dropna()
    if m.size < 2:
        return (np.nan, np.nan)
    g1 = float(((m - m.mean())**3).mean() / (m.std(ddof=1)**3 + 1e-12))
    g2 = float(((m - m.mean())**4).mean() / (m.var(ddof=1)**2 + 1e-12))
    return (g1, g2)


def _psr(returns: pd.Series, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lֳ³pez de Prado)."""
    m = returns.dropna(); n = m.size
    if n < 3:
        return np.nan
    sr = _sharpe_annualized(m)
    if not np.isfinite(sr):
        return np.nan
    g1, g2 = _skew_kurt(m)
    denom = np.sqrt(1.0 - g1 * sr + 0.25 * (g2 - 1.0) * (sr**2))
    z = ((sr - sr_star) * np.sqrt(n - 1)) / denom if denom > 0 else np.nan
    from math import erf, sqrt
    if not np.isfinite(z):
        return np.nan
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _dsr(returns: pd.Series, sr_star: float, n_trials: int) -> float:
    """Wrapper for DSR using fv_extensions if available; fallback to PSR."""
    try:
        if fvx is not None and hasattr(fvx, 'dsr'):
            return float(fvx.dsr(returns, sr_star=sr_star, n_trials=int(n_trials)))
    except Exception:
        pass
    return _psr(returns, sr_star=sr_star)
# === FV:EVAL (helpers) END ===

# === FV:PORTFOLIO (helpers) START ===

def _inv_vol_weights(series_map: Dict[Tuple[str,str], pd.Series]) -> Dict[Tuple[str,str], float]:
    vols = {k: float(s.std(ddof=1)) for k, s in series_map.items()}
    valid = {k: 1.0/v for k, v in vols.items() if np.isfinite(v) and v > 0}
    if not valid:
        return {k: np.nan for k in series_map}
    norm = sum(valid.values())
    return {k: w / norm for k, w in valid.items()}

# Local HRP helpers (fallback if fv_extensions is absent)
import numpy as _np

def _corr_from_cov(_C: _np.ndarray) -> _np.ndarray:
    d = _np.sqrt(_np.clip(_np.diag(_C), 1e-12, _np.inf))
    Dinv = _np.diag(1.0 / d)
    return Dinv @ _C @ Dinv

def _quasi_diag_order(_C: _np.ndarray) -> _np.ndarray:
    R = _corr_from_cov(_C)
    vals, vecs = _np.linalg.eigh(R)
    v1 = vecs[:, -1]
    order = _np.argsort(v1)
    return order

def _cluster_variance(C: _np.ndarray, idx: _np.ndarray) -> float:
    sub = C[_np.ix_(idx, idx)]
    w = 1.0 / _np.clip(_np.diag(sub), 1e-8, _np.inf)
    w /= w.sum()
    return float(w @ sub @ w)

def _hrp_weights_from_cov_local(C: _np.ndarray) -> _np.ndarray:
    C = _np.asarray(C, dtype=float)
    n = C.shape[0]
    order = _quasi_diag_order(C)
    idx = order.tolist()
    def _bisect(indices: List[int]) -> _np.ndarray:
        if len(indices) == 1:
            return _np.array([indices[0]])
        k = len(indices) // 2
        left = _bisect(indices[:k])
        right = _bisect(indices[k:])
        return _np.concatenate([left, right])
    sort_idx = _bisect(idx)
    w = _np.ones(n)
    clusters = [sort_idx]
    while clusters:
        cl = clusters.pop(0)
        if cl.size <= 1:
            continue
        k = cl.size // 2
        L = cl[:k]; R = cl[k:]
        varL = _cluster_variance(C, L)
        varR = _cluster_variance(C, R)
        alpha = 1.0 - varL / (varL + varR)
        w[L] *= alpha; w[R] *= (1.0 - alpha)
        clusters.extend([L, R])
    w = w / w.sum()
    final = _np.zeros(n)
    for i, j in enumerate(sort_idx):
        final[j] = w[i]
    return final
# === FV:PORTFOLIO (helpers) END ===
# === FV:CORE_MATH END ===


# === FV:PROVIDERS START ===
class DataProvider:
    """Abstract provider interface (hook).
    Implement get_prices(tickers: Sequence[str], period: str, interval: str) -> pd.DataFrame.
    """
    def get_prices(self, tickers: Sequence[str], period: str, interval: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


def _maybe_fetch(prices_wide: Optional[pd.DataFrame], provider: Optional[DataProvider],
                 tickers: Sequence[str], period: Optional[str], interval: Optional[str]) -> pd.DataFrame:
    if prices_wide is not None:
        return prices_wide
    if provider is None:
        raise RuntimeError("No price matrix provided and no provider injected. Supply `prices_wide` or pass a DataProvider.")
    return provider.get_prices(tickers, period or "2y", interval or "1d")
# === FV:PROVIDERS END ===


# === FV:ENGINE START ===
class FairValueEngine:
    """Main engine for fair value & diagnostics on pairs.

    Public API: run(prices_wide=None, pairs=None) -> DataFrame
    """
    def __init__(self, config: Optional[Config] = None, provider: Optional[DataProvider] = None) -> None:
        self.config = config or _load_config_from_params()
        self.provider = provider
        for w in self.config.secondary_windows:
            if w <= 0:
                raise ValueError(f"secondary window must be positive, got {w}")

    def run(self, prices_wide: Optional[pd.DataFrame] = None,
            pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        cfg = self.config
        pairs_use = pairs or cfg.pairs
        if not pairs_use:
            raise ValueError("No pairs provided (cfg.pairs is empty).")

        # Fetch / validate prices
        tickers = sorted({t for p in pairs_use for t in p})
        prices = _maybe_fetch(prices_wide, self.provider, tickers, cfg.data_period, cfg.data_interval)
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices.sort_index().select_dtypes(include=[float, int]).astype(float)
        asof = prices.index.max()

        # Precompute returns
        rets = prices.pct_change()

        rows: List[Dict[str, Any]] = []
        windows = (cfg.window,) + tuple(w for w in cfg.secondary_windows if w != cfg.window)
        rp_series: Dict[Tuple[str, str], pd.Series] = {}

        for (y_sym, x_sym) in pairs_use:
            if y_sym not in prices.columns or x_sym not in prices.columns:
                logger.warning(f"missing ticker(s) for pair ({y_sym},{x_sym})")
                rows.append({"asof": asof, "pair": (y_sym, x_sym), "window": cfg.window, "reason": "missing ticker(s)"})
                continue

            base = pd.concat({"y": prices[y_sym], "x": prices[x_sym]}, axis=1).dropna()
            if base.shape[0] < cfg.min_overlap:
                rows.append({"asof": asof, "pair": (y_sym, x_sym), "window": cfg.window,
                             "reason": f"insufficient history: {base.shape[0]} < {cfg.min_overlap}"})
                continue

            pair_rows: List[Dict[str, Any]] = []
            for w in windows:
                dfw = base.tail(w)
                if dfw.shape[0] < cfg.min_overlap:
                    pair_rows.append({"asof": asof, "pair": (y_sym, x_sym), "window": w,
                                      "reason": f"insufficient overlap: {dfw.shape[0]} < {cfg.min_overlap}"})
                    continue

                y, x = dfw["y"], dfw["x"]
                if cfg.use_winsor:
                    y, x = _winsorize(y, cfg.winsor_p), _winsorize(x, cfg.winsor_p)

                # Domain selection
                reason = None
                if cfg.log_mode and _check_log_safe(y, x):
                    y_fit, x_fit, use_log = np.log(y.values), np.log(x.values), True
                elif cfg.log_mode:
                    reason = "nonג€‘positive prices for log mode"; y_fit, x_fit, use_log = y.values, x.values, False
                else:
                    y_fit, x_fit, use_log = y.values, x.values, False

                # OLS fit (static)
                alpha, beta = _ols_alpha_beta(y_fit, x_fit)
                y_last, x_last = float(y.iloc[-1]), float(x.iloc[-1])

                # Optional Kalman dynamic alpha/beta
                beta_mode_used = "static"
                if fvx is not None and cfg.beta_mode.lower() == "kalman" and hasattr(fvx, 'kalman_alpha_beta'):
                    try:
                        a_path, b_path = fvx.kalman_alpha_beta(y_fit, x_fit, q=cfg.kalman_q, r=cfg.kalman_r, init=(alpha, beta))
                        if use_log:
                            y_fair_series = np.exp(a_path + b_path * np.log(x.values))
                        else:
                            y_fair_series = a_path + b_path * x.values
                        y_fair = float(y_fair_series[-1])
                        spread_series = pd.Series(y.values - y_fair_series, index=y.index)
                        alpha, beta = float(a_path[-1]), float(b_path[-1])
                        beta_mode_used = "kalman"
                    except Exception:
                        beta_mode_used = "static"

                if beta_mode_used == "static":
                    if use_log:
                        y_fair = float(np.exp(alpha + beta * np.log(x_last)))
                        spread_series = y - np.exp(alpha + beta * np.log(x))
                    else:
                        y_fair = float(alpha + beta * x_last)
                        spread_series = y - (alpha + beta * x)

                # Diagnostics
                spread_last = float(spread_series.iloc[-1])
                mu = float(spread_series.mean())
                sd = float(spread_series.std(ddof=1)) if spread_series.size > 1 else np.nan
                z = _safe_zscore(spread_last, mu, sd, cfg.zscore_clip)
                hl = _estimate_halflife(spread_series)
                band = float(1.96 * sd) if np.isfinite(sd) else np.nan
                upper = mu + band if np.isfinite(band) else np.nan
                lower = mu - band if np.isfinite(band) else np.nan

                # Correlation base: returns if requested
                if cfg.use_returns_for_corr:
                    yy = rets[y_sym].reindex(dfw.index).dropna()
                    xx = rets[x_sym].reindex(dfw.index).dropna()
                    ii = yy.index.intersection(xx.index)
                    corr = _rolling_corr(yy.loc[ii], xx.loc[ii]) if ii.size >= 2 else np.nan
                else:
                    corr = _rolling_corr(y, x)

                # Advanced metrics (optional)
                dcor = adf_p = hurst = coint_stat = coint_p = rv = rsharpe = np.nan
                if advanced_metrics is not None:
                    try:
                        if hasattr(advanced_metrics, "distance_correlation"):
                            y_d = rets[y_sym].reindex(dfw.index) if cfg.use_returns_for_dcor else y
                            x_d = rets[x_sym].reindex(dfw.index) if cfg.use_returns_for_dcor else x
                            yy2, xx2 = y_d.dropna(), x_d.dropna()
                            ii2 = yy2.index.intersection(xx2.index)
                            if ii2.size >= 2:
                                dcor = float(advanced_metrics.distance_correlation(yy2.loc[ii2].values, xx2.loc[ii2].values))
                    except Exception:
                        pass
                    try:
                        if hasattr(advanced_metrics, "adf_pvalue"):
                            adf_p = float(advanced_metrics.adf_pvalue(spread_series.values))
                    except Exception:
                        pass
                    try:
                        if hasattr(advanced_metrics, "hurst_exponent"):
                            hurst = float(advanced_metrics.hurst_exponent(spread_series.values))
                    except Exception:
                        pass
                    try:
                        if hasattr(advanced_metrics, "cointegration_test"):
                            res = advanced_metrics.cointegration_test(y.values, x.values)
                            if isinstance(res, (list, tuple)) and len(res) >= 2:
                                coint_stat, coint_p = float(res[0]), float(res[1])
                            elif res is not None:
                                coint_stat = float(res)
                    except Exception:
                        pass
                    try:
                        if hasattr(advanced_metrics, "realized_volatility"):
                            rv = float(advanced_metrics.realized_volatility(spread_series.values))
                    except Exception:
                        pass
                    try:
                        if hasattr(advanced_metrics, "rolling_sharpe"):
                            rs = advanced_metrics.rolling_sharpe(spread_series, window=len(spread_series))
                            rsharpe = float(rs.iloc[-1])
                    except Exception:
                        pass

                # Residual ADF preference for quality
                residual_adf_p = np.nan
                if advanced_metrics is not None and hasattr(advanced_metrics, "adf_pvalue") and cfg.residual_adf_enabled:
                    try:
                        resid_fit = y_fit - (alpha + beta * x_fit)
                        residual_adf_p = float(advanced_metrics.adf_pvalue(resid_fit))
                    except Exception:
                        residual_adf_p = np.nan
                adf_for_quality = residual_adf_p if np.isfinite(residual_adf_p) else adf_p

                # Volatility-adjusted mispricing
                if cfg.volatility_adjust:
                    denom = sd if np.isfinite(sd) and sd > 0 else (rv if np.isfinite(rv) and rv > 0 else np.nan)
                    vol_adj = (spread_last / denom) if np.isfinite(denom) and denom > 0 else np.nan
                else:
                    vol_adj = np.nan

                # Sizing & action
                wq = _quality_weight(adf_for_quality, hl, corr)
                spread_vol_ann = _ann_vol_of_spread(spread_series)
                target_units = _size_from_quality_and_vol(z if np.isfinite(z) else 0.0,
                                                          wq, cfg.target_vol_ann,
                                                          spread_vol_ann, cfg.kelly_fraction,
                                                          cfg.max_leverage)
                side = "long_spread" if (np.isfinite(z) and z < -cfg.z_in) else ("short_spread" if (np.isfinite(z) and z > cfg.z_in) else "flat")

                cost_units = _spread_cost_units(y_last, x_last, beta, cfg.costs_bps, cfg.slippage_bps, cfg.borrow_bps)
                net_edge_z = ((abs(spread_last) - cost_units) / sd) if (np.isfinite(sd) and sd > 0) else np.nan

                # Eval for primary window
                sr_net = psr_net = dsr_net = turnover_est = avg_hold = np.nan; trades = 0
                if w == cfg.window:
                    z_series = (spread_series - mu) / sd if (np.isfinite(sd) and sd > 0) else pd.Series(np.nan, index=spread_series.index)
                    pnl_series, trades, avg_hold = _simulate_hysteresis_pnl(spread_series, z_series, cfg.z_in, cfg.z_out, cost_units)
                    sr_net = _sharpe_annualized(pnl_series)
                    psr_net = _psr(pnl_series, sr_star=cfg.psr_sr_star)
                    dsr_net = _dsr(pnl_series, sr_star=cfg.psr_sr_star, n_trials=cfg.dsr_n_trials)
                    turnover_est = float(np.mean(np.abs(np.diff(np.clip(z_series.values, -1, 1))))) if z_series.notna().sum()>2 else np.nan
                    rp_series[(y_sym, x_sym)] = spread_series.diff().dropna()

                pair_rows.append({
                    "asof": asof,
                    "pair": (y_sym, x_sym),
                    "window": int(w),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "beta_mode_used": beta_mode_used,
                    "beta_dyn_last": float(beta),
                    "y_last": y_last,
                    "x_last": x_last,
                    "y_fair": float(y_fair),
                    "spread": spread_last,
                    "spread_mean": mu,
                    "spread_std": sd,
                    "zscore": z,
                    "band_p95": band,
                    "band_upper": upper,
                    "band_lower": lower,
                    "halflife": hl,
                    "rolling_corr": corr,
                    "distance_corr": dcor,
                    "adf_p": adf_p,
                    "residual_adf_p": residual_adf_p,
                    "is_coint": bool(np.isfinite(residual_adf_p) and residual_adf_p < cfg.coint_pvalue),
                    "hurst": hurst,
                    "coint_stat": coint_stat,
                    "coint_p": coint_p,
                    "realized_vol": rv,
                    "rolling_sharpe": rsharpe,
                    "mispricing": spread_last,
                    "vol_adj_mispricing": vol_adj,
                    "is_mean_reverting": bool((np.isfinite(hl) and hl not in (float("inf"),)) and ((np.isfinite(adf_for_quality) and adf_for_quality < cfg.mean_revert_pvalue) or (not np.isfinite(adf_for_quality)))),
                    "n_obs": int(dfw.shape[0]),
                    "reason": reason,
                    # action/sizing/costs/eval
                    "action": side,
                    "target_pos_units": float(target_units),
                    "quality_weight": float(wq),
                    "cost_spread_units": float(cost_units),
                    "net_edge_z": float(net_edge_z) if np.isfinite(net_edge_z) else np.nan,
                    "z_in": cfg.z_in,
                    "z_out": cfg.z_out,
                    "sr_net": float(sr_net) if np.isfinite(sr_net) else np.nan,
                    "psr_net": float(psr_net) if np.isfinite(psr_net) else np.nan,
                    "dsr_net": float(dsr_net) if np.isfinite(dsr_net) else np.nan,
                    "turnover_est": float(turnover_est) if np.isfinite(turnover_est) else np.nan,
                    "avg_hold_days": float(avg_hold) if np.isfinite(avg_hold) else np.nan,
                })

            # Append per-window rows
            rows.extend(pair_rows)

            # Ensemble row (weighted) if requested
            if cfg.ensemble_mode.lower() == "weighted" and len(pair_rows) > 0:
                wts, tgt_vals = [], []
                for r in pair_rows:
                    wq_en = _quality_weight(r.get("adf_p"), r.get("halflife"), r.get("rolling_corr"))
                    tgt = r.get(cfg.ensemble_target)
                    if tgt is None or not np.isfinite(tgt):
                        continue
                    if np.isfinite(wq_en) and wq_en > 0:
                        wts.append(wq_en)
                        tgt_vals.append((tgt, r))
                if len(wts) > 0 and len(tgt_vals) > 0:
                    def wavg(field: str, default=np.nan):
                        num = 0.0; den = 0.0
                        for (_t, r), w in zip(tgt_vals, wts):
                            v = r.get(field)
                            if v is None or not np.isfinite(v):
                                continue
                            num += w * float(v); den += w
                        return float(num/den) if den > 0 else default
                    ens = {
                        "asof": asof,
                        "pair": (y_sym, x_sym),
                        "window": -1,
                        "y_last": pair_rows[-1]["y_last"],
                        "x_last": pair_rows[-1]["x_last"],
                        "y_fair": wavg("y_fair"),
                        "mispricing": wavg("mispricing"),
                        "vol_adj_mispricing": wavg("vol_adj_mispricing"),
                        "zscore": wavg("zscore"),
                        "band_lower": wavg("band_lower"),
                        "band_upper": wavg("band_upper"),
                        "spread": wavg("spread"),
                        "spread_mean": wavg("spread_mean"),
                        "spread_std": wavg("spread_std"),
                        "band_p95": wavg("band_p95"),
                        "alpha": wavg("alpha"),
                        "beta": wavg("beta"),
                        "halflife": wavg("halflife"),
                        "rolling_corr": wavg("rolling_corr"),
                        "distance_corr": wavg("distance_corr"),
                        "adf_p": wavg("adf_p"),
                        "hurst": wavg("hurst"),
                        "coint_stat": wavg("coint_stat"),
                        "coint_p": wavg("coint_p"),
                        "realized_vol": wavg("realized_vol"),
                        "rolling_sharpe": wavg("rolling_sharpe"),
                        "n_obs": int(min(r["n_obs"] for r in pair_rows if isinstance(r.get("n_obs"), int))) if any(isinstance(r.get("n_obs"), int) for r in pair_rows) else 0,
                        "reason": "ensemble(weighted)",
                    }
                    ens["is_mean_reverting"] = any(bool(r.get("is_mean_reverting")) for r in pair_rows)
                    rows.append(ens)

        # Portfolio weights on spread changes ג€” inv-vol / ERC / HRP
        rp_w: Dict[Tuple[str,str], float] = {}
        if rp_series:
            try:
                rp_df = pd.DataFrame(rp_series).dropna()
                if rp_df.shape[1] >= 2:
                    if cfg.cov_method.lower() in ("ridge", "lw") and fvx is not None and hasattr(fvx, 'cov_shrinkage'):
                        C = fvx.cov_shrinkage(rp_df.values, method=cfg.cov_method.lower(), lam=cfg.cov_shrink_lambda)
                    else:
                        C = np.cov(rp_df.values, rowvar=False, ddof=1)
                    if cfg.rp_method.lower() == "hrp":
                        if fvx is not None and hasattr(fvx, 'hrp_weights_from_cov'):
                            w = fvx.hrp_weights_from_cov(C)
                        else:
                            w = _hrp_weights_from_cov_local(C)
                        for i, col in enumerate(rp_df.columns):
                            rp_w[col] = float(w[i])
                    elif cfg.rp_method.lower() == "erc":
                        if fvx is not None and hasattr(fvx, 'erc_weights_from_cov'):
                            w = fvx.erc_weights_from_cov(C)
                            for i, col in enumerate(rp_df.columns):
                                rp_w[col] = float(w[i])
                if not rp_w:
                    rp_w = _inv_vol_weights(rp_series)
            except Exception:
                rp_w = _inv_vol_weights(rp_series)

        for r in rows:
            if r.get("window") == cfg.window and tuple(r.get("pair")) in rp_w:
                r["rp_weight"] = float(rp_w[tuple(r.get("pair"))])
            elif "rp_weight" not in r:
                r["rp_weight"] = np.nan

        out = pd.DataFrame(rows)
        preferred = [
            "asof","pair","window","y_last","x_last","y_fair","mispricing","vol_adj_mispricing",
            "zscore","band_lower","band_upper","spread","spread_mean","spread_std","band_p95",
            "alpha","beta","beta_mode_used","beta_dyn_last","halflife","rolling_corr","distance_corr","adf_p","residual_adf_p","is_coint","hurst",
            "coint_stat","coint_p","realized_vol","rolling_sharpe","is_mean_reverting",
            "action","target_pos_units","quality_weight","cost_spread_units","net_edge_z","z_in","z_out",
            "sr_net","psr_net","dsr_net","turnover_est","avg_hold_days","rp_weight",
            "n_obs","reason",
        ]
        cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
        return out[cols]
# === FV:ENGINE END ===


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run Fair Value Engine on a wide CSV")
    p.add_argument("--csv", required=False, help="Path to wide CSV (date index in first col or 'Date')")
    p.add_argument("--pairs", nargs="*", help="Pairs as Y:X")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    cfg = _load_config_from_params()
    if args.window is not None:
        cfg.window = args.window
    if args.log:
        cfg.log_mode = True

    if args.csv:
        df = pd.read_csv(args.csv)
        # autodetect date
        date_col = None
        for c in df.columns:
            if str(c).lower() in {"date","datetime"}:
                date_col = c; break
        if date_col is None:
            # assume first col is index
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors="coerce")
            prices_wide = df.set_index(df.columns[0])
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            prices_wide = df.set_index(date_col)
        prices_wide = prices_wide.select_dtypes(include=[float, int])
    else:
        prices_wide = None

    if args.pairs:
        pairs_list: List[Tuple[str,str]] = []
        for s in args.pairs:
            if ":" in s:
                y,x = s.split(":",1)
                pairs_list.append((y.strip(), x.strip()))
    else:
        pairs_list = cfg.pairs

    eng = FairValueEngine(cfg)
    res = eng.run(prices_wide=prices_wide, pairs=pairs_list)
    print(res.head(30).to_string(index=False))
