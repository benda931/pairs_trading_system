# core/metrics.py
"""Metrics utilities for the optimisation pipeline.

- Dynamic plugin registry via ``@register_metric``
- Normalisation helpers with correct directionality (e.g., lower drawdown is better)
- End-to-end ``score_performance`` helper
- Fast Hurst exponent (Numba fallback)
- Rich ``PerformanceMetrics`` class (abridged)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional acceleration with Numba
# ---------------------------------------------------------------------------
try:
    from numba import njit  # type: ignore

    @njit
    def _numba_hurst(arr: np.ndarray, max_lag: int) -> float:  # pragma: no cover — numba fallback
        lags = np.arange(2, max_lag)
        tau = np.sqrt(np.array([np.std(np.diff(arr, lag)) for lag in lags]))
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return 2 * poly[0]

    @njit
    def _numba_es(arr: np.ndarray, alpha: float) -> float:  # pragma: no cover — numba fallback
        losses = -arr
        var = np.quantile(losses, 1 - alpha)
        mask = losses >= var
        return losses[mask].mean() if mask.any() else 0.0

    @njit
    def _numba_tail_ratio(arr: np.ndarray, upper: float, lower: float) -> float:  # pragma: no cover — numba fallback
        upper_q = np.quantile(arr, upper)
        lower_q = abs(np.quantile(arr, lower))
        return upper_q / lower_q if lower_q else np.inf

    NUMBA_OK = True
except Exception:  # pragma: no cover — Numba unavailable
    NUMBA_OK = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Type aliases & constants
# ---------------------------------------------------------------------------
class PerfDict(TypedDict, total=False):
    Sharpe: float
    Profit: float
    Drawdown: float
    WinRate: float
    ProfitFactor: float
    Calmar: float
    Crowding: float  # 0..1 where higher=crowded (penalised)

BoundsDict = Dict[str, Tuple[float, float]]
WeightsDict = Dict[str, float]

__all__ = [
    'PerfDict', 'BoundsDict', 'WeightsDict',
    'register_metric', 'extract_metrics', 'get_metric_bounds',
    'normalize_metrics', 'compute_weighted_score', 'compute_weighted_score_normalized',
    'normalize_weights', 'score_performance', 'objective_from_perf', 'score_with_crowding',
    'load_metric_plugins', 'get_metric_bounds_from_history', 'score_geometric', 'rank_runs',
    'PerformanceMetrics',
]

DEFAULT_BOUNDS: BoundsDict = {
    'Sharpe': (0.0, 3.0),
    'Profit': (0.0, 10_000.0),
    'Drawdown': (0.0, 2_000.0),
    'WinRate': (0.0, 1.0),
    'ProfitFactor': (0.0, 5.0),
    'Calmar': (0.0, 3.0),
    'Crowding': (0.0, 1.0),
}

METRIC_NAMES: Dict[str, str] = {
    'Sharpe': 'Sharpe Ratio',
    'Profit': 'Total Profit',
    'Drawdown': 'Max Drawdown',
    'WinRate': 'Win Rate',
    'ProfitFactor': 'Profit Factor',
    'Calmar': 'Calmar Ratio',
    'Crowding': 'Crowding Score',
}

# Metrics where *lower* is better (invert normalisation)
LOWER_IS_BETTER = {'Drawdown', 'Crowding'}

# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------
MetricFn = Callable[[PerfDict], float]
_METRIC_PLUGINS: Dict[str, MetricFn] = {}

def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator to register a metric extraction function."""
    def _decorator(fn: MetricFn) -> MetricFn:
        _METRIC_PLUGINS[name] = fn
        return fn
    return _decorator

# Default metric extractors
@register_metric('Sharpe')
def _get_sharpe(p: PerfDict) -> float:  # noqa: D401 — simple getter
    return p.get('Sharpe', 0.0)

@register_metric('Profit')
def _get_profit(p: PerfDict) -> float:
    return p.get('Profit', 0.0)

@register_metric('Drawdown')
def _get_drawdown(p: PerfDict) -> float:
    return p.get('Drawdown', float('inf'))

@register_metric('WinRate')
def _get_winrate(p: PerfDict) -> float:
    return p.get('WinRate', 0.0)

@register_metric('ProfitFactor')
def _get_pf(p: PerfDict) -> float:
    return p.get('ProfitFactor', 0.0)

@register_metric('Calmar')
def _get_calmar(p: PerfDict) -> float:
    return p.get('Calmar', 0.0)

@register_metric('Crowding')
def _get_crowding(p: PerfDict) -> float:
    return p.get('Crowding', 0.0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_metrics(perf: PerfDict) -> PerfDict:
    """Return a dict with all registered metrics extracted from *perf*."""
    return {name: fn(perf) for name, fn in _METRIC_PLUGINS.items()}


@lru_cache(maxsize=None)
def get_metric_bounds(capital: float) -> BoundsDict:
    """Return bounds scaled by *capital* for Profit & Drawdown."""
    return {
        **DEFAULT_BOUNDS,
        'Profit': (0.0, capital),
        'Drawdown': (0.0, capital * 0.2),
    }


def normalize_metrics(perf: PerfDict, bounds: Optional[BoundsDict] = None) -> PerfDict:
    """Min–max normalise *perf* into [0,1] using *bounds*.

    For metrics where *lower* is better (e.g., Drawdown, Crowding), the score is inverted
    so that 1.0 is always better across metrics.
    """
    b = bounds or DEFAULT_BOUNDS
    out: PerfDict = {}
    for m, (lo, hi) in b.items():
        val = perf.get(m, 0.0)
        norm = (val - lo) / (hi - lo) if hi > lo else 0.0
        norm = max(0.0, min(1.0, norm))
        if m in LOWER_IS_BETTER:
            norm = 1.0 - norm
        out[m] = norm
    return out


def compute_weighted_score(norm: PerfDict, weights: WeightsDict) -> float:
    """Weighted average of *norm* according to *weights*."""
    ttl = sum(weights.values())
    if ttl <= 0:
        logger.error('Total weight must be positive')
        return 0.0
    return sum(norm.get(m, 0.0) * w for m, w in weights.items()) / ttl


def normalize_weights(weights: WeightsDict) -> WeightsDict:
    """Return a copy of *weights* normalised to sum to 1.0 (filters non‑positive)."""
    wpos = {k: float(v) for k, v in weights.items() if float(v) > 0}
    ttl = sum(wpos.values())
    return {k: v / ttl for k, v in wpos.items()} if ttl > 0 else {}


def compute_weighted_score_normalized(norm: PerfDict, weights: WeightsDict) -> float:
    """Compute weighted score after normalising *weights* to sum to 1.0."""
    w = normalize_weights(weights)
    return sum(norm.get(m, 0.0) * w.get(m, 0.0) for m in norm.keys())


def score_geometric(norm: PerfDict, weights: WeightsDict) -> float:
    """Geometric‑mean composite score (all inputs in [0,1]; weights normalised).

    Useful when penalising low components more strongly than arithmetic mean.
    """
    w = normalize_weights(weights)
    if not w:
        logger.error('No positive weights for geometric score')
        return 0.0
    prod = 1.0
    for m, wm in w.items():
        prod *= max(1e-9, float(norm.get(m, 0.0))) ** wm
    return float(prod)


def rank_runs(df: pd.DataFrame, weights: WeightsDict, capital: float | None = None,
              use_geometric: bool = False) -> pd.DataFrame:
    """Return *df* with an added 'score' column and sorted by descending score.

    Expects columns matching metric names in *weights*. Non‑existent metrics are treated as 0.
    """
    rows = []
    for _, row in df.iterrows():
        perf: PerfDict = {m: float(row.get(m, 0.0)) for m in weights.keys()}
        raw = extract_metrics(perf)
        bounds = get_metric_bounds(capital or DEFAULT_BOUNDS['Profit'][1])
        norm = normalize_metrics(raw, bounds)
        score = score_geometric(norm, weights) if use_geometric else compute_weighted_score_normalized(norm, weights)
        rows.append(score)
    out = df.copy()
    out['score'] = rows
    return out.sort_values('score', ascending=False).reset_index(drop=True)


def score_performance(perf: PerfDict, weights: WeightsDict, capital: float | None = None) -> float:
    """High-level helper to go from raw *perf* to a single [0,1] score.

    Parameters
    ----------
    perf : dict
        Raw performance metrics (e.g., {"Sharpe": 1.4, "Profit": 5000, "Drawdown": 900}).
    weights : dict
        Weights per metric, not necessarily summing to 1 (normalised internally).
    capital : float, optional
        Scaling base for Profit/Drawdown bounds; if None use default bounds.

    Examples
    --------
    >>> perf = {"Sharpe": 1.5, "Profit": 5000, "Drawdown": 1000}
    >>> w = {"Sharpe": 0.5, "Profit": 0.4, "Drawdown": 0.1}
    >>> s = score_performance(perf, w, capital=10_000)
    >>> 0.0 <= s <= 1.0
    True
    """
    raw = extract_metrics(perf)
    bounds = get_metric_bounds(capital or DEFAULT_BOUNDS['Profit'][1])
    norm = normalize_metrics(raw, bounds)
    return compute_weighted_score(norm, weights)


def objective_from_perf(
    perf: PerfDict,
    weights: WeightsDict,
    capital: float | None = None,
    minimize: bool = True,
) -> float:
    """Optuna-friendly objective from metrics.

    Returns a value to minimise by default (negative score), or to maximise if ``minimize=False``.
    """
    score = score_performance(perf, weights, capital)
    return -score if minimize else score


def score_with_crowding(
    perf: PerfDict,
    weights: WeightsDict,
    capital: float | None = None,
    crowding_weight: float = 0.10,
) -> float:
    """Score with crowding penalty (keeps [0,1] scale).

    The penalty scales linearly with the normalised Crowding metric.
    """
    raw = extract_metrics(perf)
    bounds = get_metric_bounds(capital or DEFAULT_BOUNDS['Profit'][1])
    norm = normalize_metrics(raw, bounds)
    base = compute_weighted_score(norm, weights)
    crowd = norm.get('Crowding', 0.0)
    penalty = max(0.0, 1.0 - crowding_weight * crowd)
    return max(0.0, min(1.0, base * penalty))

# ---------------------------------------------------------------------------
# Optional plugin loader & adaptive bounds
# ---------------------------------------------------------------------------

def load_metric_plugins(path: str) -> int:
    """Dynamically load metrics_*.py plugins from *path*.

    Returns number of successfully loaded modules. Each plugin can call
    ``@register_metric`` to add custom extractors.
    """
    import importlib.util, pathlib
    loaded = 0
    for p in pathlib.Path(path).glob('metrics_*.py'):
        try:
            spec = importlib.util.spec_from_file_location(p.stem, p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[arg-type]
                loaded += 1
        except Exception as e:
            logger.warning("failed loading plugin %s: %s", p, e)
    return loaded


def get_metric_bounds_from_history(db_path: str, table: str = 'backtests', horizon_days: int = 365,
                                   default_capital: float = DEFAULT_BOUNDS['Profit'][1]) -> BoundsDict:
    """Derive metric bounds from historical results stored in DuckDB.

    Falls back to ``get_metric_bounds(default_capital)`` if DuckDB or table is unavailable.
    """
    try:
        import duckdb  # type: ignore
        con = duckdb.connect(db_path, read_only=True)
        q = f"""
            WITH recent AS (
                SELECT * FROM {table}
                WHERE date >= DATE 'now' - INTERVAL {horizon_days} DAY
            )
            SELECT
                MAX(COALESCE(total_pnl,0))  AS max_profit,
                MAX(COALESCE(max_dd_usd,0)) AS max_dd
            FROM recent;
        """
        row = con.execute(q).fetchone()
        con.close()
        if row:
            max_profit = float(row[0] or 0.0)
            max_dd = float(row[1] or 0.0)
            cap = max(default_capital, max_profit)
            b = get_metric_bounds(cap)
            b['Drawdown'] = (0.0, max(max_dd, cap*0.2))
            return b
    except Exception as e:
        logger.info("adaptive bounds fallback: %s", e)
    return get_metric_bounds(default_capital)

# ---------------------------------------------------------------------------
# PerformanceMetrics class (trimmed)
# ---------------------------------------------------------------------------
class PerformanceMetrics:
    """Rolling & point-in-time statistics for a returns series."""

    def __init__(self, returns: pd.Series):
        self.returns = returns.dropna()

    # --- Basic metrics --------------------------------------------------
    def sharpe(self) -> float:
        std = self.returns.std()
        return np.sqrt(252) * self.returns.mean() / std if std else 0.0

    def sortino(self) -> float:
        downside = self.returns[self.returns < 0].std()
        return np.sqrt(252) * self.returns.mean() / downside if downside else 0.0

    # --- Fast Hurst exponent (Numba) ------------------------------------
    def hurst_exponent(self, max_lag: int = 100) -> float:
        arr = self.returns.values.astype(np.float64)
        if NUMBA_OK:
            try:
                return float(_numba_hurst(arr, max_lag))
            except Exception:  # pragma: no cover — fallback
                pass
        lags = np.arange(2, max_lag)
        tau = [np.sqrt(np.std(np.diff(arr, lag))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return 2 * poly[0]

    # --- Advanced risk metrics -----------------------------------------
    def expected_shortfall(self, alpha: float = 0.05) -> float:
        """Conditional VaR / Expected Shortfall at confidence level *alpha*."""
        arr = self.returns.values.astype(np.float64)
        if NUMBA_OK:
            try:
                return float(_numba_es(arr, alpha))
            except Exception:
                pass
        losses = -arr
        var = np.quantile(losses, 1 - alpha)
        tail = losses[losses >= var]
        return float(tail.mean()) if tail.size else 0.0

    def tail_ratio(self, upper: float = 0.95, lower: float = 0.05) -> float:
        """Tail ratio: *upper* quantile gain divided by *lower* quantile loss."""
        arr = self.returns.values.astype(np.float64)
        if NUMBA_OK:
            try:
                return float(_numba_tail_ratio(arr, upper, lower))
            except Exception:
                pass
        upper_q = np.quantile(arr, upper)
        lower_q = abs(np.quantile(arr, lower))
        return float(upper_q / lower_q) if lower_q else np.inf

    # --- Additional risk & efficiency metrics -------------------------
    def calmar(self) -> float:
        """Calmar ratio: annualised return divided by max drawdown."""
        ann = self.returns.mean() * 252
        equity = (1 + self.returns).cumprod()
        max_dd = ((equity - equity.cummax()) / equity.cummax()).min()
        return float(ann / abs(max_dd)) if max_dd < 0 else np.inf

    def ulcer_index(self) -> float:
        """Ulcer index: root-mean-square percentage drawdown."""
        equity = (1 + self.returns).cumprod()
        dd = (equity - equity.cummax()) / equity.cummax()
        return float(np.sqrt((dd ** 2).mean()))

    def kelly_fraction(self) -> float:
        """Kelly fraction: mean / variance."""
        var = self.returns.var()
        return float(self.returns.mean() / var) if var else 0.0

    def drawdown_duration(self) -> int:
        """Longest drawdown duration (bars)."""
        equity = (1 + self.returns).cumprod()
        peak = equity.cummax()
        underwater = equity < peak
        grp = (underwater != underwater.shift()).cumsum()
        return int(underwater.groupby(grp).cumsum().max())

    def recovery_time(self) -> int:
        """Bars taken to recover from max drawdown (np.nan if not recovered)."""
        equity = (1 + self.returns).cumprod()
        peak_idx = equity.idxmax()
        after_peak = equity.loc[peak_idx:]
        recov = after_peak[after_peak >= equity.loc[peak_idx]]
        if not recov.empty:
            return int((recov.index[0] - peak_idx)) if hasattr(recov.index[0], '__sub__') else int(recov.index[0] - peak_idx)
        return np.nan

    def upside_capture(self, benchmark: pd.Series) -> float:
        """Returns capture during benchmark up days."""
        mask = benchmark > 0
        bmk_up = benchmark[mask].mean()
        strat_up = self.returns[mask].mean()
        return float(strat_up / bmk_up) if bmk_up else np.nan

    def downside_capture(self, benchmark: pd.Series) -> float:
        """Returns capture during benchmark down days."""
        mask = benchmark < 0
        bmk_down = benchmark[mask].mean()
        strat_down = self.returns[mask].mean()
        return float(strat_down / bmk_down) if bmk_down else np.nan

    def bayesian_sharpe(self, rf: float = 0.0, prior_mean: float = 0.0, prior_var: float = 1e-4) -> float:
        """Bayesian Sharpe ratio using a conjugate prior."""
        sample_mean = self.returns.mean() - rf / 252
        sample_var = self.returns.var()
        post_mean = (prior_var * sample_mean + sample_var * prior_mean) / (prior_var + sample_var)
        post_var = (prior_var * sample_var) / (prior_var + sample_var)
        return np.sqrt(252) * post_mean / np.sqrt(post_var) if post_var else 0.0

    def factor_betas(self, factors: pd.DataFrame) -> pd.Series:
        """OLS betas to supplied factor returns DataFrame."""
        try:
            import statsmodels.api as sm  # type: ignore
        except ImportError:  # pragma: no cover
            logger.warning("statsmodels not available – skipping factor_betas")
            return pd.Series(dtype=float)
        y = self.returns.reindex(factors.index).dropna()
        X = factors.loc[y.index].fillna(0)  # type: ignore[attr-defined]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model.params.drop('const')

    def fama_french_exposure(self, start: str | None = None, end: str | None = None) -> pd.Series:
        """Download daily FF factors & MOM and return betas."""
        try:
            from pandas_datareader import data as web  # type: ignore
        except ImportError:  # pragma: no cover
            logger.warning("pandas_datareader not available – skipping fama_french_exposure")
            return pd.Series(dtype=float)
        ff = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench')[0]
        mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench')[0]
        ff.index = pd.to_datetime(ff.index, format='%Y%m%d')
        mom.index = pd.to_datetime(mom.index, format='%Y%m%d')
        df = ff.join(mom, how='inner') / 100
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return self.factor_betas(df)

    # --- Rolling analytics ---------------------------------------------
    def rolling_beta(self, benchmark: pd.Series, window: int = 63) -> pd.Series:
        """Rolling CAPM beta to *benchmark* using *window* observations."""
        cov = self.returns.rolling(window).cov(benchmark.rolling(window))
        var = benchmark.rolling(window).var()
        return cov / var

    def rolling_expected_shortfall(self, window: int = 63, alpha: float = 0.05) -> pd.Series:
        """Rolling Conditional VaR / Expected Shortfall."""
        def es_window(x):
            losses = -x
            var = np.quantile(losses, 1 - alpha)
            tail = losses[losses >= var]
            return tail.mean() if len(tail) else 0.0
        return self.returns.rolling(window).apply(es_window, raw=False)

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """Rolling Sharpe ratio (window annualised)."""
        return (
            self.returns.rolling(window).mean() / self.returns.rolling(window).std()
        ) * np.sqrt(252)

    def rolling_sortino(self, window: int = 63) -> pd.Series:
        """Rolling Sortino ratio (window annualised)."""
        def sortino_win(x):
            downside = x[x < 0].std()
            return (x.mean() / downside) * np.sqrt(252) if downside else 0.0
        return self.returns.rolling(window).apply(sortino_win, raw=False)

    def rolling_calmar(self, window: int = 252) -> pd.Series:
        """Rolling Calmar ratio over *window* bars."""
        def calmar_win(x):
            eq = (1 + x).cumprod()
            max_dd = ((eq - eq.cummax()) / eq.cummax()).min()
            ann = x.mean() * 252
            return ann / abs(max_dd) if max_dd < 0 else np.inf
        return self.returns.rolling(window).apply(calmar_win, raw=False)

    def rolling_ulcer(self, window: int = 252) -> pd.Series:
        """Rolling Ulcer index."""
        def ulcer_win(x):
            eq = (1 + x).cumprod()
            dd = (eq - eq.cummax()) / eq.cummax()
            return np.sqrt((dd ** 2).mean())
        return self.returns.rolling(window).apply(ulcer_win, raw=False)

    def rolling_tail_ratio(self, window: int = 126, upper: float = 0.95, lower: float = 0.05) -> pd.Series:
        """Rolling tail ratio in a sliding window."""
        def tail_win(x):
            upper_q = np.quantile(x, upper)
            lower_q = abs(np.quantile(x, lower))
            return upper_q / lower_q if lower_q else np.inf
        return self.returns.rolling(window).apply(tail_win, raw=False)

    def rolling_drawdown(self, window: int = 252) -> pd.Series:
        """Rolling max drawdown over *window* bars (percentage negative)."""
        def dd_win(x):
            eq = (1 + x).cumprod()
            draw = (eq - eq.cummax()) / eq.cummax()
            return draw.min()
        return self.returns.rolling(window).apply(dd_win, raw=False)

    # Representation helpers -------------------------------------------
    def summary(self, extended: bool = False) -> pd.Series:
        """Return a pandas Series with core (or extended) metrics."""
        core = {
            'Sharpe': self.sharpe(),
            'Sortino': self.sortino(),
            'Calmar': self.calmar(),
            'ES_95': self.expected_shortfall(),
            'Ulcer': self.ulcer_index(),
        }
        if extended:
            core.update({
                'Hurst': self.hurst_exponent(),
                'Kelly': self.kelly_fraction(),
                'TailRatio': self.tail_ratio(),
            })
        return pd.Series(core)

    def to_frame(self, extended: bool = False) -> pd.DataFrame:
        """Return metrics as single-row DataFrame (nice for concatenation)."""
        return self.summary(extended).to_frame().T

    # ------------------------------------------------------------------
    # Export & visualisation helpers
    # ------------------------------------------------------------------
    def equity_curve(self) -> pd.Series:
        """Cumulative equity curve Series (starting at 1.0)."""
        return (1 + self.returns).cumprod()

    def plot_equity(self):  # pragma: no cover — optional Plotly
        """Return a Plotly Figure of the equity curve (if Plotly available)."""
        try:
            import plotly.express as px  # type: ignore
        except ImportError:
            logger.warning("plotly not installed – cannot plot equity curve")
            return None
        curve = self.equity_curve()
        fig = px.line(curve, title="Equity Curve", labels={"value": "Equity", "index": "Date"})
        return fig

    def export_summary_csv(self, path: str | None = None, extended: bool = False) -> str:
        """Export summary metrics to CSV. Returns CSV string (and saves if *path*)."""
        csv_str = self.summary(extended).to_frame().T.to_csv(index=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(csv_str)
        return csv_str

    def export_summary_json(self, path: str | None = None, extended: bool = False, indent: int | None = 2) -> str:
        """Export summary metrics to JSON string (and saves if *path*)."""
        json_str = self.summary(extended).to_json(indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def export_summary_md(self, path: str | None = None, extended: bool = False) -> str:
        """Export summary metrics to GitHub-flavoured Markdown table."""
        ser = self.summary(extended)
        md_lines = ["| Metric | Value |", "|---|---|"]
        for k, v in ser.items():
            try:
                md_lines.append(f"| {k} | {float(v):.4f} |")
            except Exception:
                md_lines.append(f"| {k} | {v} |")
        md = "\n".join(md_lines)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
        return md

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PerformanceMetrics Sharpe={self.sharpe():.2f} n={len(self.returns)}>"
