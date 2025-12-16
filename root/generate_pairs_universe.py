# -*- coding: utf-8 -*-
"""
generate_pairs_universe.py — Build a high-quality pairs universe
----------------------------------------------------------------
Refactored for robustness, speed, and reuse.

✓ Works as a **module** (import & call `build_pairs_universe(...)`) or **CLI**
✓ Guards optional scientific deps; has safe fallbacks
✓ Preloads price series (threaded) to avoid N×M I/O
✓ Parallel, deterministic scoring of candidate pairs
✓ Deterministic filtering/dedup for stable outputs
✓ Optional progress bars (tqdm) and rich detail columns
✓ Uses core.stat_tests.compute_stat_quality_score for hedge-fund-grade TS diagnostics

Base output columns (always):
    symbol_1, symbol_2, corr, p_value, half_life, vol_zscore, quality
Optional detail columns (controlled by `include_details=True`):
    n_obs, beta, intercept, start, end, quality_stat, quality_ts_score

Assumptions:
- `load_price_data(symbol)` returns a DataFrame with a DateTimeIndex and 'close' column
- If `statsmodels` is unavailable, cointegration p-value falls back to ADF on hedge residual (or 1.0)
- If `utils` module not available, built-in fallbacks are used for half-life & vol ratio & quality
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Literal
import itertools
import math
import os
import sys
import argparse
import warnings
import contextlib
import concurrent.futures as _cf
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from core.app_context import AppContext
from core.sql_store import SqlStore


log = logging.getLogger("pairs_universe")

# ---- Optional scientific libs (guarded) ----
try:  # pragma: no cover
    from scipy.stats import pearsonr, spearmanr  # type: ignore
except Exception:  # pragma: no cover
    pearsonr = None  # type: ignore
    spearmanr = None  # type: ignore

try:  # pragma: no cover
    from statsmodels.tsa.stattools import coint as _coint, adfuller as _adfuller  # type: ignore
except Exception:  # pragma: no cover
    _coint = None  # type: ignore
    _adfuller = None  # type: ignore

# ---- Project loaders/helpers (guarded) ----
# דואג ש-root יראה את תיקיית הפרויקט, ואז משתמש ב-loader החדש מ-common.data_loader
from pathlib import Path
import sys as _sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover
    # ⚙️ זה ה-loader החדש, שכבר בדקנו שעובד (load_price_data("SPY",...) עובד)
    from common.data_loader import load_price_data  # type: ignore[import]
except Exception as _exc:  # pragma: no cover
    load_price_data = None  # type: ignore
    log.warning(
        "generate_pairs_universe: failed to import load_price_data from common.data_loader: %s",
        _exc,
    )


try:  # pragma: no cover
    from common.utils import (
        calculate_half_life as _hl_ext,
        calculate_volatility_zscore as _vz_ext,
        evaluate_pair_quality as _qual_ext,
    )  # type: ignore
except Exception:  # pragma: no cover
    _hl_ext = _vz_ext = _qual_ext = None  # type: ignore

# 🔹 חיבור ל־core/pairs_universe — כדי להשתמש בזוגות מתוך JSON
try:  # pragma: no cover
    from core.pairs_universe import (
        load_pairs_universe,
        as_symbol_pairs,
        describe_universe,
    )
except ImportError:  # pragma: no cover
    load_pairs_universe = None  # type: ignore
    as_symbol_pairs = None  # type: ignore
    describe_universe = None  # type: ignore

# 🔹 חיבור ל־core/stat_tests — ציון איכות סטטיסטי לספראד
try:  # pragma: no cover
    from core.stat_tests import compute_stat_quality_score  # type: ignore
except Exception:  # pragma: no cover
    compute_stat_quality_score = None  # type: ignore

warnings.simplefilter("ignore", category=FutureWarning)
log = logging.getLogger("pairs_universe")

# =========================
# Fallback helpers (when utils/scipy/statsmodels missing)
# =========================

def _half_life_fallback(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life via AR(1) OLS proxy (no statsmodels)."""
    s = pd.Series(spread).dropna()
    if len(s) < 10:
        return float("nan")
    x = s.shift(1)
    y = s
    X = np.vstack([x.fillna(method="bfill").values, np.ones(len(x))]).T
    try:
        beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
        phi = float(beta[0])
    except Exception:
        return float("nan")
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return float("inf")
    return float(-math.log(2) / math.log(phi))


def _vol_zscore_fallback(px: pd.Series, py: pd.Series) -> float:
    r1 = pd.to_numeric(px, errors="coerce").pct_change().dropna()
    r2 = pd.to_numeric(py, errors="coerce").pct_change().dropna()
    v1 = float(np.std(r1))
    v2 = float(np.std(r2))
    if not np.isfinite(v1) or not np.isfinite(v2) or v2 == 0:
        return float("inf")
    return float(v1 / v2) if v1 >= v2 else float(v2 / v1)


def _quality_fallback(
    *,
    corr: float,
    p_value: float,
    half_life: float,
    vol_zscore: float,
    n_obs: int,
    min_obs: int,
) -> float:
    """Professional composite score in [0,1] for pair quality.

    Components
    ----------
    corr_score   : abs(corr) in [0,1] – strength of linear co-movement.
    coint_score  : cointegration strength from p-value (Engle-Granger/ADF).
    hl_score     : mean-reversion speed from half-life (shorter is better).
    vol_score    : how close vol_zscore is to 1 (balanced legs).
    len_score    : stability bonus based on sample size vs `min_overlap`.

    Aggregation
    -----------
    base = 0.40*corr + 0.25*coint + 0.20*hl + 0.15*vol
    quality = base * (0.5 + 0.5*len_score)

    Notes
    -----
    - All intermediate scores are clipped into [0,1].
    - If anything explodes (NaN/inf), we degrade gracefully to 0.
    - This is only used when `evaluate_pair_quality` is not provided.
    """
    try:
        # --- Corr score (|ρ|) ---
        c_raw = float(corr)
        if not np.isfinite(c_raw):
            c_raw = 0.0
        corr_score = max(0.0, min(1.0, abs(c_raw)))

        # --- Cointegration score (from p-value) ---
        p = float(p_value)
        if not np.isfinite(p) or p <= 0.0:
            coint_score = 0.0
        else:
            p = max(1e-12, min(1.0, p))
            if p <= 1e-4:
                coint_score = 1.0
            elif p <= 0.1:
                coint_score = (-math.log10(p) / 4.0)
                coint_score = max(0.0, min(1.0, coint_score))
            else:
                coint_score = 0.0

        # --- Half-life score (shorter = better, but robust) ---
        hl = float(half_life)
        if not np.isfinite(hl) or hl <= 0.0:
            hl_score = 0.0
        else:
            hl_score = 1.0 / (1.0 + math.log1p(max(hl, 0.0)))
            hl_score = max(0.0, min(1.0, hl_score))

        # --- Volatility balance score (vol_zscore ~ 1 is ideal) ---
        vz = float(vol_zscore)
        if not np.isfinite(vz) or vz <= 0.0:
            vol_score = 0.0
        else:
            penalty = abs(math.log(vz))
            vol_score = 1.0 / (1.0 + 3.0 * penalty)
            vol_score = max(0.0, min(1.0, vol_score))

        # --- Length / stability score ---
        try:
            ratio = float(n_obs) / max(float(min_obs), 1.0)
        except Exception:
            ratio = 0.0
        ratio = max(0.0, ratio)
        ratio = min(ratio, 5.0)
        len_score = (math.tanh((ratio - 1.0) / 2.0) + 1.0) / 2.0
        len_score = max(0.0, min(1.0, len_score))

        base = (
            0.40 * corr_score
            + 0.25 * coint_score
            + 0.20 * hl_score
            + 0.15 * vol_score
        )

        quality = base * (0.5 + 0.5 * len_score)

        if not np.isfinite(quality):
            return 0.0
        return float(max(0.0, min(1.0, quality)))
    except Exception:
        return 0.0


# =========================
# Config & datamodel
# =========================

@dataclass
class BuildConfig:
    # statistical thresholds
    min_corr: float = 0.70
    max_pvalue: float = 0.05
    min_overlap: int = 60            # min shared observations after alignment
    # optional stability filters
    max_half_life: Optional[float] = None
    max_vol_zscore: Optional[float] = None  # e.g., 1.5 means vols within 50%
    # returns/correlation settings
    return_mode: Literal["log", "pct"] = "log"  # log or pct returns for corr/vol
    corr_method: Literal["pearson", "spearman"] = "pearson"
    # time windowing / resampling
    lookback_days: Optional[int] = 365 * 3        # None → use full history
    resample: Optional[str] = None                # e.g., '1D', '1W'
    # candidate generation
    top_n_aux: int = 300
    within_sector_only: bool = False
    sector_map: Optional[Dict[str, str]] = None   # symbol->sector mapping
    # performance & UX
    parallel: bool = True
    workers: Optional[int] = None                 # None → os.cpu_count()
    progress: bool = True
    preload: bool = True
    # details & reproducibility
    include_details: bool = True
    seed: int = 42
    # candidate sampling (נקבעים ע"י CLI ומועברים ל-meta)
    candidate_mode: Literal["all", "random", "latin", "sobol"] = "all"
    candidate_count: Optional[int] = None


DEFAULT_INITIAL_PAIRS: List[Tuple[str, str]] = [
    ("AAPL", "MSFT"), ("GOOGL", "META"), ("CSCO", "JNPR"), ("INTC", "AMD"),
    ("NVDA", "AMD"), ("ORCL", "IBM"), ("TXN", "ADI"), ("HPQ", "DELL"),
    ("WDC", "STX"), ("MU", "WDC"), ("V", "MA"), ("JPM", "BAC"),
]

DEFAULT_MANDATORY_PAIRS: List[Tuple[str, str]] = [
    ("XLY", "XLC"), ("XLY", "XLP"), ("XLI", "XLB"), ("IBIT", "MSTR"),
]

DEFAULT_AUX_SYMBOLS: List[str] = [
    "SPY", "QQQ", "DIA", "IWM", "VOO", "VTI", "VEA", "VWO", "VEU", "VNQ",
    "XLC", "XLY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLB", "XLRE",
    "IBIT", "MSTR", "GBTC", "ETHE", "BTC",
    "GLD", "SLV", "USO", "UNG", "UUP", "FXE", "FXA", "IYR", "EEM", "EFA",
    "LQD", "HYG", "SHY", "IEF", "TLT", "TIP", "BND", "AGG", "LIT", "XOP",
]

# =========================
# Price preparation & alignment
# =========================

def _prepare_series(symbol: str, cfg: BuildConfig) -> pd.Series:
    # במקום datetime.utcnow() או datetime.now(timezone.utc)
    # נשתמש ב-Pandas Timestamp, strip timezone, ונישאר אחידים עם האינדקסים:
    start_dt = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=int(cfg.lookback_days))

    if load_price_data is None:
        raise RuntimeError("load_price_data is not available")

    df = load_price_data(
        symbol,
        start_date=start_dt,
        end_date=None,
    )

    if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
        raise ValueError(f"Bad data for {symbol}: expected DataFrame with 'close'")

    # נוודא שהאינדקס ממויין
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # 3) raw series לפני ניקוי (ל-fallback)
    s_raw = df["close"]

    # אם הכל NaN כבר בשלב הזה – אין מה לעשות
    if s_raw.isna().all():
        raise ValueError(f"No valid closes for {symbol}: all NaN before cleaning")

    # 4) ניקוי ראשוני "חכם"
    # נשתמש בגרסה כבר מנורמלת של validate_price_df,
    # אבל נוודא שוב שאנחנו בנומרי + finite.
    s = pd.to_numeric(s_raw, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.dropna()

    # אם יש price_floor כלשהו בעתיד, אפשר להשתמש פה:
    # if getattr(cfg, "price_floor", None) is not None:
    #     s = s[s >= float(cfg.price_floor)]

    # 5) אם הניקוי האגרסיבי מחק הכל, אבל raw כן מכיל דאטה → fallback
    if s.empty and s_raw.notna().sum() > 0:
        log.warning(
            "prepare_series(%s): aggressive cleaning removed all points — "
            "falling back to basic dropna-only cleaning",
            symbol,
        )
        s = pd.to_numeric(s_raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
        s = s.dropna()

    # 6) אחרי fallback – אם עדיין אין כלום → באמת בעייתי
    if s.empty:
        raise ValueError(f"No valid closes for {symbol} after filtering")

    # 7) מגביל ל-lookback המקורי (במקרה שה-loader החזיר יותר)
    s = s[s.index >= s.index.min()]  # רק לוודא אינדקס, אפשר להשאיר ככה
    # אפשר גם:
    # s = s[max(s.index.min(), pd.Timestamp(start_dt)) :]

    # 8) מחזירים סדרה נקייה
    return s

def _align_returns(
    px: pd.Series,
    py: pd.Series,
    *,
    cfg: BuildConfig,
) -> Tuple[pd.Series, pd.Series, int, pd.Timestamp, pd.Timestamp]:
    x, y = px.align(py, join="inner")
    if cfg.return_mode == "log":
        rx = np.log(x).diff().dropna()
        ry = np.log(y).diff().dropna()
    else:
        rx = x.pct_change().dropna()
        ry = y.pct_change().dropna()
    rx, ry = rx.align(ry, join="inner")
    n = len(rx)
    if n == 0:
        return rx, ry, 0, pd.NaT, pd.NaT
    start = max(rx.index.min(), ry.index.min())
    end = min(rx.index.max(), ry.index.max())
    return rx, ry, n, start, end


def _hedge(px: pd.Series, py: pd.Series) -> Tuple[float, float, pd.Series]:
    """OLS hedge: py ≈ a + b*px; returns (beta=b, intercept=a, residual=py - (a+b*px))."""
    x, y = px.align(py, join="inner")
    X = np.vstack([np.ones(len(x)), x.values]).T
    b, a = 0.0, 0.0
    try:
        coeffs, *_ = np.linalg.lstsq(X, y.values, rcond=None)
        a = float(coeffs[0])
        b = float(coeffs[1])
    except Exception:
        pass
    resid = y - (a + b * x)
    return b, a, resid.dropna()


def _corr(rx: pd.Series, ry: pd.Series, *, cfg: BuildConfig) -> float:
    if cfg.corr_method == "spearman":
        if spearmanr is not None:
            return float(spearmanr(rx.values, ry.values).correlation)
        return float(rx.corr(ry, method="spearman"))
    if pearsonr is not None:
        return float(pearsonr(rx.values, ry.values)[0])
    return float(rx.corr(ry, method="pearson"))


def _cointegration_pval(px: pd.Series, py: pd.Series) -> float:
    x, y = px.align(py, join="inner")
    if len(x) < 20:
        return 1.0
    if _coint is not None:
        try:
            _, pval, _ = _coint(np.log(x.values), np.log(y.values))
            return float(pval)
        except Exception:
            pass
    b, a, resid = _hedge(x, y)
    if _adfuller is not None:
        try:
            return float(_adfuller(resid.values, regression="c")[1])
        except Exception:
            return 1.0
    return 1.0


# =========================
# Pair scoring
# =========================

def _score_pair(
    sym1: str,
    sym2: str,
    cfg: BuildConfig,
    price_map: Optional[Dict[str, pd.Series]] = None,
    seed_priority_map: Optional[Dict[Tuple[str, str], float]] = None,
) -> Optional[Dict[str, object]]:
    """
    HF-grade scoring for a single pair (sym1, sym2).

    שכבות הציון (Multi-layer):
    ---------------------------
    1. Data layer:
       - טוען מחירי close (דרך _prepare_series → load_price_data מ-common.data_loader).
       - חיתוך לפי cfg.lookback_days + resample אם צריך.
       - פילטור overlap מינימלי (cfg.min_overlap).

    2. Core stats layer:
       - returns alignment (log/pct, cfg.return_mode).
       - corr (pearson/spearman לפי cfg.corr_method).
       - cointegration p-value (Engle–Granger, fallback ADF על residual).
       - OLS hedge (beta, intercept, spread).
       - half-life (calculate_half_life אם קיים, אחרת AR(1) fallback).
       - volatility z-score בין שתי הרגליים.

    3. HF metrics layer (על הספראד):
       - r_spread = Δspread (או pct_change).
       - Sharpe של הספראד (שנתי).
       - Sortino (downside std בלבד).
       - Max drawdown על cumulative spread-returns.
       - Volatility of spread.
       - fraction of positive vs negative returns.

    4. Quality scores:
       - quality_stat  (0–1) מהמודול הישן (evaluate_pair_quality) או _quality_fallback.
       - quality_ts_score (0–1) מה-core.stat_tests.compute_stat_quality_score(spread) אם זמין.
       - quality_hf (0–1) משולב עם Sharpe/Sortino/DD של הספראד.
       - core_quality = mix(quality_stat, quality_ts_score, quality_hf).

    5. Seed priority overlay:
       - אם seed_priority_map קיים (pairs.json) → 80% סטטיסטיקה, 20% priority.

    החזרה:
    -------
    dict עם:
        symbol_1, symbol_2, corr, p_value, half_life, vol_zscore, quality
    ואם cfg.include_details=True:
        n_obs, beta, intercept, start, end,
        quality_stat, quality_ts_score, quality_hf,
        spread_sharpe, spread_sortino, spread_max_dd,
        spread_vol, spread_pos_ratio, spread_neg_ratio.
    """
    try:
        # ---------- 1) Load price series ----------
        px = None
        py = None
        if price_map is not None:
            px = price_map.get(sym1)
            py = price_map.get(sym2)

        if px is None:
            px = _prepare_series(sym1, cfg=cfg)
        if py is None:
            py = _prepare_series(sym2, cfg=cfg)

        # ---------- 2) Align returns & basic overlap ----------
        rx, ry, n, start, end = _align_returns(px, py, cfg=cfg)
        if n < cfg.min_overlap:
            return None

        # ---------- 3) Correlation ----------
        corr = _corr(rx, ry, cfg=cfg)
        if not np.isfinite(corr) or abs(corr) < float(cfg.min_corr):
            return None

        # ---------- 4) Cointegration ----------
        pval = _cointegration_pval(px, py)
        if not np.isfinite(pval) or pval > float(cfg.max_pvalue):
            return None

        # ---------- 5) Hedge & spread ----------
        beta, intercept, spread = _hedge(px, py)
        spread = spread.dropna()
        if spread.empty or spread.var() <= 0:
            return None

        # Half-life
        try:
            hl = float(_hl_ext(spread) if _hl_ext is not None else _half_life_fallback(spread))
        except Exception:
            hl = float("nan")

        # Volatility z-score (legs)
        try:
            vz = float(_vz_ext(px, py) if _vz_ext is not None else _vol_zscore_fallback(px, py))
        except Exception:
            vz = float("inf")

        # Hard filters (optional)
        if cfg.max_half_life is not None and np.isfinite(hl) and hl > float(cfg.max_half_life):
            return None
        if cfg.max_vol_zscore is not None and np.isfinite(vz) and vz > float(cfg.max_vol_zscore):
            return None

        # ---------- 6) HF metrics on spread ----------
        # נקבע returns על הספראד (Δspread) — יותר יציב מ-pct_change על מחירים.
        spr_ret = spread.diff().dropna()
        if spr_ret.empty:
            spr_ret = spread.pct_change().dropna()

        if spr_ret.empty:
            # אם גם זה נכשל — אין מה לציין על האדג'
            spr_sharpe = 0.0
            spr_sortino = 0.0
            spr_max_dd = 0.0
            spr_vol = 0.0
            spr_pos_ratio = 0.0
            spr_neg_ratio = 0.0
        else:
            r = spr_ret.to_numpy(dtype=float)
            # Sharpe (שנתי, הנחה דומה ל-daily)
            mean_r = float(np.mean(r))
            std_r = float(np.std(r, ddof=0))
            spr_sharpe = float((mean_r / std_r) * np.sqrt(252.0)) if std_r > 0 else 0.0

            # downside std → Sortino
            downside = r[r < 0]
            ds = float(np.std(downside, ddof=0)) if downside.size > 0 else 0.0
            spr_sortino = float((mean_r * np.sqrt(252.0)) / ds) if ds > 0 else 0.0

            # Max drawdown (על cumulative sum של spread returns)
            cum = np.cumsum(r)
            if cum.size:
                peak = np.maximum.accumulate(cum)
                dd = peak - cum
                spr_max_dd = float(dd.max())
            else:
                spr_max_dd = 0.0

            spr_vol = float(std_r * np.sqrt(252.0))
            spr_pos_ratio = float((r > 0).mean())
            spr_neg_ratio = float((r < 0).mean())

        # ---------- 7) Base statistical quality (ישן) ----------
        try:
            if _qual_ext is not None:
                quality_stat = float(
                    _qual_ext(
                        corr=corr,
                        p_value=pval,
                        half_life=hl,
                        vol_zscore=vz,
                        n_obs=n,
                        min_obs=cfg.min_overlap,
                    )
                )
            else:
                quality_stat = _quality_fallback(
                    corr=corr,
                    p_value=pval,
                    half_life=hl,
                    vol_zscore=vz,
                    n_obs=n,
                    min_obs=cfg.min_overlap,
                )
        except Exception:
            quality_stat = 0.0

        if not np.isfinite(quality_stat):
            quality_stat = 0.0
        quality_stat = max(0.0, min(1.0, float(quality_stat)))

        # ---------- 8) Time-series quality (core.stat_tests) ----------
        ts_score_norm = 0.0
        if compute_stat_quality_score is not None:
            try:
                ts_raw = float(compute_stat_quality_score(spread))
                if np.isfinite(ts_raw):
                    # מניחים תחום אופייני ~0–4 → Scale ל-[0,1]
                    ts_score_norm = max(0.0, min(1.0, ts_raw / 4.0))
            except Exception:
                ts_score_norm = 0.0

        # ---------- 9) HF quality layer from spread metrics ----------
        # Sharpe / Sortino / DD נותנים "צל" על האדג' בפועל.
        # נבנה score רך מ-3 ממדים:
        #   - sharpe_score  ~ tanh(Sharpe/3)
        #   - sortino_score ~ tanh(Sortino/3)
        #   - dd_score      ~ penalty על DD גבוה מדי.
        sharpe_score = 0.5 + math.tanh(spr_sharpe / 3.0) / 2.0 if np.isfinite(spr_sharpe) else 0.5
        sortino_score = 0.5 + math.tanh(spr_sortino / 3.0) / 2.0 if np.isfinite(spr_sortino) else 0.5

        if np.isfinite(spr_max_dd) and spr_max_dd > 0:
            # DD גבוה → score נמוך; משתמשים ב-log soft penalty
            dd_penalty = 1.0 / (1.0 + math.log1p(spr_max_dd))
            dd_score = max(0.0, min(1.0, dd_penalty))
        else:
            dd_score = 1.0

        quality_hf = float(
            0.5 * sharpe_score +
            0.3 * sortino_score +
            0.2 * dd_score
        )
        quality_hf = max(0.0, min(1.0, quality_hf))

        # ---------- 10) Combine all quality layers ----------
        # משקל קצת יותר גבוה למטריקות היסוד (quality_stat),
        # ואחוזים משמעותיים ל-TS ול-HF spread metrics.
        w_stat = 0.45
        w_ts = 0.25
        w_hf = 0.30
        core_quality = (
            w_stat * quality_stat +
            w_ts * ts_score_norm +
            w_hf * quality_hf
        )
        if not np.isfinite(core_quality):
            core_quality = 0.0
        core_quality = max(0.0, min(1.0, float(core_quality)))

        # ---------- 11) Blend with seed priority (pairs.json) ----------
        final_quality = core_quality
        if seed_priority_map is not None:
            key = tuple(sorted((str(sym1).strip(), str(sym2).strip())))
            prio_raw = float(seed_priority_map.get(key, 0.0) or 0.0)
            if not np.isfinite(prio_raw) or prio_raw < 0.0:
                prio_raw = 0.0
            if prio_raw > 1.0:
                prio_raw = 1.0
            alpha = 0.8  # 80% סטטיסטיקה, 20% priority
            final_quality = alpha * core_quality + (1.0 - alpha) * prio_raw

        if not np.isfinite(final_quality):
            final_quality = 0.0
        final_quality = max(0.0, min(1.0, float(final_quality)))

        # ---------- 12) Build output record ----------
        out: Dict[str, object] = {
            "symbol_1": sym1,
            "symbol_2": sym2,
            "corr": float(corr),
            "p_value": float(pval),
            "half_life": float(hl),
            "vol_zscore": float(vz),
            "quality": float(final_quality),
        }

        if cfg.include_details:
            out.update(
                {
                    "n_obs": int(n),
                    "beta": float(beta),
                    "intercept": float(intercept),
                    "start": pd.to_datetime(start).isoformat() if pd.notna(start) else None,
                    "end": pd.to_datetime(end).isoformat() if pd.notna(end) else None,
                    "quality_stat": float(quality_stat),
                    "quality_ts_score": float(ts_score_norm),
                    "quality_hf": float(quality_hf),
                    "spread_sharpe": float(spr_sharpe),
                    "spread_sortino": float(spr_sortino),
                    "spread_max_dd": float(spr_max_dd),
                    "spread_vol": float(spr_vol),
                    "spread_pos_ratio": float(spr_pos_ratio),
                    "spread_neg_ratio": float(spr_neg_ratio),
                }
            )

        return out

    except Exception as e:
        # קרן גידור לא מתעלמת משגיאות בשקט — רושמים ERROR עם trace מלא.
        log.error(
            "score_pair failed for %s-%s (min_corr=%.3f, max_pval=%.3f, min_overlap=%d): %s",
            sym1,
            sym2,
            cfg.min_corr,
            cfg.max_pvalue,
            cfg.min_overlap,
            e,
            exc_info=True,
        )
        return None



# =========================
# Utility: canonical pair set
# =========================

def _unique_ordered_pairs(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return canonicalized unique (a,b) where a<b to avoid duplicates."""
    seen = set()
    out: List[Tuple[str, str]] = []
    for a, b in pairs:
        if a == b:
            continue
        key = tuple(sorted((str(a).strip(), str(b).strip())))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# =========================
# Public API
# =========================

def build_pairs_universe(
    *,
    initial_pairs: Sequence[Tuple[str, str]] = DEFAULT_INITIAL_PAIRS,
    mandatory_pairs: Sequence[Tuple[str, str]] = DEFAULT_MANDATORY_PAIRS,
    aux_symbols: Sequence[str] = DEFAULT_AUX_SYMBOLS,
    cfg: BuildConfig = BuildConfig(),
    seed_priority_map: Optional[Dict[Tuple[str, str], float]] = None,
) -> pd.DataFrame:
    """Build the pairs universe as a DataFrame."""
    np.random.seed(cfg.seed)

    init_pairs = _unique_ordered_pairs(initial_pairs)
    mand_pairs = _unique_ordered_pairs(mandatory_pairs)

    # preload prices to avoid repeated I/O
    price_map: Optional[Dict[str, pd.Series]] = None
    if cfg.preload:
        to_load = {s for pair in (init_pairs + mand_pairs) for s in pair}
        to_load |= set(str(s).strip() for s in aux_symbols)
        price_map = {}
        syms = sorted(to_load)
        workers = cfg.workers or (os.cpu_count() or 4)
        try:
            with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_prepare_series, sym, cfg=cfg): sym for sym in syms}
                for fut in futs:
                    pass
                for fut in _cf.as_completed(futs):
                    sym = futs[fut]
                    try:
                        price_map[sym] = fut.result()
                    except Exception as e:
                        log.warning("skip %s: %s", sym, e)
        except Exception:
            # fallback serial
            for sym in syms:
                try:
                    price_map[sym] = _prepare_series(sym, cfg=cfg)
                except Exception as e:
                    log.warning("skip %s: %s", sym, e)

    rows: List[Dict[str, object]] = []

    def _score_wrapper(pair: Tuple[str, str]) -> Optional[Dict[str, object]]:
        return _score_pair(pair[0], pair[1], cfg, price_map, seed_priority_map)

    # 1) score initial & mandatory
    for pair in itertools.chain(init_pairs, mand_pairs):
        rec = _score_wrapper(pair)
        if rec:
            rows.append(rec)

    # 2) aux candidates
    syms_aux = [str(s).strip() for s in aux_symbols if str(s).strip()]
    candidates = _unique_ordered_pairs(itertools.combinations(syms_aux, 2))

    # sector constraint
    if cfg.within_sector_only and cfg.sector_map:
        candidates = [p for p in candidates if cfg.sector_map.get(p[0]) == cfg.sector_map.get(p[1])]

    # optional candidate sampling (candidate_mode / candidate_count)
    if cfg.candidate_mode != "all" and cfg.candidate_count is not None and cfg.candidate_count > 0:
        total = len(candidates)
        k = min(int(cfg.candidate_count), total)
        if 0 < k < total:
            rng = np.random.RandomState(cfg.seed)
            idx = rng.choice(total, size=k, replace=False)
            idx_sorted = sorted(int(i) for i in idx)
            candidates = [candidates[i] for i in idx_sorted]
            log.info(
                "Candidate sampling: mode=%s, requested=%d, total=%d, used=%d",
                cfg.candidate_mode,
                cfg.candidate_count,
                total,
                len(candidates),
            )

    # progress helper
    def _iter_progress(it, total=None, desc=""):
        if cfg.progress:
            try:
                from tqdm import tqdm  # type: ignore
                return tqdm(it, total=total, desc=desc)
            except Exception:
                return it
        return it

    # score aux (parallel or serial)
    if cfg.parallel:
        max_workers = cfg.workers or (os.cpu_count() or 4)
        with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_score_wrapper, p) for p in candidates]
            for fut in _iter_progress(_cf.as_completed(futs), total=len(futs), desc="Scoring aux"):
                try:
                    rec = fut.result()
                except Exception:
                    rec = None
                if rec:
                    rows.append(rec)
    else:
        for p in _iter_progress(candidates, total=len(candidates), desc="Scoring aux"):
            rec = _score_wrapper(p)
            if rec:
                rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["symbol_1", "symbol_2", "corr", "p_value", "half_life", "vol_zscore", "quality"])

    df = pd.DataFrame.from_records(rows)

    base_cols = ["symbol_1", "symbol_2", "corr", "p_value", "half_life", "vol_zscore", "quality"]
    for col in base_cols:
        if col not in df.columns:
            df[col] = np.nan
    ordered = base_cols + [c for c in df.columns if c not in set(base_cols)]
    df = df[ordered]

    # deterministic canonical key and dedup
    df["_key"] = df.apply(lambda r: tuple(sorted((str(r["symbol_1"]), str(r["symbol_2"])))), axis=1)
    df = (
        df.sort_values(["quality", "corr"], ascending=[False, False])
        .drop_duplicates(subset=["_key"])
        .drop(columns=["_key"])
    )
    if cfg.top_n_aux is not None and len(df) > cfg.top_n_aux:
        df = df.head(cfg.top_n_aux)
    df.reset_index(drop=True, inplace=True)
    return df


# =========================
# Packaging helpers (ZIP bundle)
# =========================

def _sha256_str(data: bytes) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _to_parquet_bytes(df: pd.DataFrame) -> Optional[bytes]:
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        import io
        table = pa.Table.from_pandas(df)
        bio = io.BytesIO()
        pq.write_table(table, bio)
        return bio.getvalue()
    except Exception:
        return None


def _build_zip_bundle(
    df: pd.DataFrame,
    output_csv_path: str,
    cfg: BuildConfig,
    *,
    include_parquet: bool = True,
) -> str:
    import io, json, zipfile, os as _os
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    meta = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
        "config": {
            "min_corr": cfg.min_corr,
            "max_pvalue": cfg.max_pvalue,
            "min_overlap": cfg.min_overlap,
            "lookback_days": cfg.lookback_days,
            "resample": cfg.resample,
            "return_mode": cfg.return_mode,
            "corr_method": cfg.corr_method,
            "top_n_aux": cfg.top_n_aux,
            "max_half_life": cfg.max_half_life,
            "max_vol_zscore": cfg.max_vol_zscore,
            "candidate_mode": cfg.candidate_mode,
            "candidate_count": cfg.candidate_count,
        },
    }
    out_zip = _os.path.splitext(output_csv_path)[0] + ".zip"
    with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pairs_universe.csv", csv_bytes)
        zf.writestr("pairs_universe.sha256", _sha256_str(csv_bytes))
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        if include_parquet:
            pbytes = _to_parquet_bytes(df)
            if pbytes is not None:
                zf.writestr("pairs_universe.parquet", pbytes)
    return out_zip


# =========================
# CLI helpers & entry point
# =========================

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate high-quality pairs universe CSV")
    p.add_argument("--min-corr", type=float, default=0.70, help="Minimum absolute correlation of returns")
    p.add_argument("--max-pval", type=float, default=0.05, help="Maximum cointegration p-value")
    p.add_argument("--min-obs", type=int, default=60, help="Minimum overlapping observations after alignment")
    p.add_argument("--lookback-days", type=int, default=1095, help="Limit history to last N days (None = full)")
    p.add_argument("--resample", type=str, default=None, help="Optional resample rule, e.g. '1D','1W','1M'")
    p.add_argument("--return-mode", choices=["log", "pct"], default="log")
    p.add_argument("--corr", choices=["pearson", "spearman"], default="pearson")
    p.add_argument("--top-n", type=int, default=300, help="Max number of auxiliary pairs to keep")
    p.add_argument("--max-hl", type=float, default=None, help="Optional max half-life filter")
    p.add_argument("--max-vz", type=float, default=None, help="Optional max volatility z-score filter")
    p.add_argument("--within-sector-only", action="store_true", help="Only form aux pairs within same sector")
    p.add_argument("--sector-file", type=str, default=None, help="CSV with columns [symbol,sector]")
    p.add_argument(
        "--candidate-mode",
        choices=["all", "random", "latin", "sobol"],
        default="all",
        help="Candidate sampling mode",
    )
    p.add_argument(
        "--candidate-count",
        type=int,
        default=None,
        help="When not 'all', how many pairs to score",
    )
    p.add_argument("--no-parallel", action="store_true", help="Disable threaded prefetch & scoring")
    p.add_argument("--workers", type=int, default=None, help="Max workers for threading")
    p.add_argument("--no-details", action="store_true", help="Do not include extra detail columns")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    p.add_argument("--package-zip", action="store_true", help="Build ZIP bundle (meta+csv[+parquet])")
    p.add_argument("--parquet", action="store_true", help="Include Parquet in ZIP (if pyarrow available)")
    p.add_argument("--min-seed-priority", type=float, default=0.0, help="Minimum priority for pairs from pairs.json (0-1)")
    p.add_argument("--max-seed-pairs", type=int, default=None, help="Maximum number of seed pairs to load from pairs.json")
    p.add_argument("--output", type=str, default="pairs_universe.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    # optional sector map
    sector_map: Optional[Dict[str, str]] = None
    if args.sector_file:
        try:
            dfm = pd.read_csv(args.sector_file)
            sym_col = "symbol" if "symbol" in dfm.columns else dfm.columns[0]
            sec_col = "sector" if "sector" in dfm.columns else dfm.columns[1]
            sector_map = {str(s): str(t) for s, t in zip(dfm[sym_col], dfm[sec_col])}
        except Exception as e:
            log.warning("Failed to load sector map: %s", e)

    cfg = BuildConfig(
        min_corr=args.min_corr,
        max_pvalue=args.max_pval,
        min_overlap=args.min_obs,
        top_n_aux=args.top_n,
        max_half_life=args.max_hl,
        max_vol_zscore=args.max_vz,
        return_mode=args.return_mode,
        corr_method=args.corr,
        lookback_days=args.lookback_days,
        resample=args.resample,
        within_sector_only=bool(args.within_sector_only),
        sector_map=sector_map,
        parallel=not args.no_parallel,
        workers=args.workers,
        progress=not args.no_progress,
        include_details=not args.no_details,
        seed=args.seed,
        candidate_mode=args.candidate_mode,
        candidate_count=args.candidate_count,
    )

    np.random.seed(cfg.seed)

    # 🔹 שימוש בזוגות מ-json (pairs.json בשורש הפרויקט) אם core/pairs_universe זמין
    initial_pairs = DEFAULT_INITIAL_PAIRS
    mandatory_pairs = DEFAULT_MANDATORY_PAIRS
    seed_priority_map: Optional[Dict[Tuple[str, str], float]] = None
    seed_meta_map: Optional[Dict[Tuple[str, str], Tuple[float, str]]] = None

    if load_pairs_universe is not None and as_symbol_pairs is not None:
        try:
            project_root = Path(__file__).resolve().parent.parent
            seed_records = load_pairs_universe(
                base_dir=project_root,
                filenames=["pairs.json"],
            )
            min_priority = float(getattr(args, "min_seed_priority", 0.0) or 0.0)
            max_seed = getattr(args, "max_seed_pairs", None)
            filtered = [r for r in seed_records if getattr(r, "priority", 0.0) >= min_priority]
            if max_seed is not None and max_seed > 0:
                filtered.sort(key=lambda r: getattr(r, "priority", 0.0), reverse=True)
                filtered = filtered[: max_seed]
            if filtered:
                seed_records = filtered
            seed_pairs = as_symbol_pairs(seed_records)
            if seed_pairs:
                mandatory_pairs = seed_pairs
                seed_priority_map = {r.key: float(getattr(r, "priority", 0.0) or 0.0) for r in seed_records}
                seed_meta_map = {
                    r.key: (
                        float(getattr(r, "priority", float("nan")) or float("nan")),
                        str(getattr(r, "category", "")),
                    )
                    for r in seed_records
                }
                initial_pairs = list(DEFAULT_INITIAL_PAIRS) + seed_pairs[: max(0, 10 - len(DEFAULT_INITIAL_PAIRS))]
                log.info(
                    "Loaded %d seed pairs from pairs.json (mandatory_pairs, min_priority=%.2f, max_seed=%s)",
                    len(seed_pairs),
                    min_priority,
                    str(max_seed),
                )
                if describe_universe is not None:
                    log.debug("Pairs universe summary: %s", describe_universe(seed_records))
        except Exception as e:
            log.warning(
                "Failed to load pairs.json universe from project root; falling back to DEFAULT_MANDATORY_PAIRS: %s",
                e,
            )

    try:
        df = build_pairs_universe(
            initial_pairs=initial_pairs,
            mandatory_pairs=mandatory_pairs,
            aux_symbols=DEFAULT_AUX_SYMBOLS,
            cfg=cfg,
            seed_priority_map=seed_priority_map,
        )
    except Exception as e:
        print(f"ERROR: build_pairs_universe failed: {e}", file=sys.stderr)
        return 2

    if seed_meta_map is not None and not df.empty:
        def _key_row(r: pd.Series) -> Tuple[str, str]:
            return tuple(sorted((str(r["symbol_1"]), str(r["symbol_2"]))))
        keys = df.apply(_key_row, axis=1)
        seed_priorities: List[float] = []
        seed_categories: List[str] = []
        for k in keys:
            prio, cat = seed_meta_map.get(k, (float("nan"), ""))

            seed_priorities.append(prio)
            seed_categories.append(cat)
        df["seed_priority"] = seed_priorities
        df["seed_category"] = seed_categories

    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"✓ Wrote {args.output} with {len(df)} rows (top aux={cfg.top_n_aux}).")
        if args.package_zip:
            try:
                _build_zip_bundle(df, args.output, cfg, include_parquet=args.parquet)
                print("✓ Built ZIP bundle")
            except Exception as e:
                print(f"WARN: packaging failed: {e}")

        # ===== Persist pairs_universe → SqlStore (dq_pairs) =====
        try:
            app_ctx = AppContext.get_global()
            store = SqlStore.from_settings(app_ctx.settings)

            # df אמור להכיל כבר sym_x, sym_y, score, z_entry, z_exit וכו' לפי הלוגיקה של generate_pairs_universe
            store.save_pair_quality(
                df,
                run_id="pairs_universe_bootstrap",
                section="data_quality",
                env=store.default_env,  # בדרך כלל 'dev'
            )
            print(f"✓ Persisted {len(df)} pairs into SqlStore.dq_pairs (env={store.default_env})")
        except Exception as e:
            print(f"WARN: failed to persist pairs_universe to SqlStore: {e}")

        return 0
    except Exception as e:
        print(f"ERROR: saving csv failed: {e}", file=sys.stderr)
        return 3



if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
