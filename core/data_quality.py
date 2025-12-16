# -*- coding: utf-8 -*-
"""
core/data_quality.py — Data Quality Engine (HF-grade)
=====================================================

מודול מרכזי למדידת איכות דאטה במערכת:

1. Symbol-level:
   - coverage_ratio, missing_days_est
   - nan_fraction (price vs all columns)
   - zero_volume_fraction
   - max_gap_days
   - duplicated_rows / non_monotonic_index
   - suspicious_jumps_count (תשואות קיצוניות)
   - return_vol_daily/annual

2. Pair-level:
   - sym_x_quality / sym_y_quality (embedded dicts)
   - overlap_ratio
   - aligned_points
   - corr_approx_price / corr_approx_rets
   - mismatch_days
   - warnings / status

3. Universe-level:
   - DataFrames מוכנים ל-SQL (symbols_df, pairs_df)
   - UniverseDataQualitySummary – סטטיסטיקות גלובליות
   - רשימות worst symbols / pairs

המודול לא תלוי ב-UI; מחזיר רק dataclasses, dictים ו-DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

JSONDict = Dict[str, Any]

# פונקציית טעינת מחירים – חופשית (תיבנה מבחוץ)
PriceLoaderFn = Callable[[str, date, date], pd.DataFrame]
PairLegsLoaderFn = Callable[[Any, date, date], Tuple[pd.Series, pd.Series, str]]  # (s1, s2, label)


# ========= Dataclasses =========

@dataclass
class SymbolDataQuality:
    """
    איכות דאטה לסימבול בודד.

    חשוב: בנוי בצורה "SQL-friendly" (ניתן להפוך ישר ל-DataFrame/טבלה).
    """

    symbol: str

    # טווח זמין בפועל
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    # כמות שורות
    rows: int = 0

    # טווח ימים כולל (end-start+1)
    span_days: Optional[int] = None
    unique_days: Optional[int] = None

    # יחס כיסוי: unique_days / span_days
    coverage_ratio: Optional[float] = None

    # הערכת ימים חסרים
    missing_days_est: Optional[int] = None

    # איכות אינדקס
    non_monotonic_index: bool = False
    duplicated_rows: int = 0

    # NaNים
    nan_fraction_price: Optional[float] = None
    nan_fraction_all: Optional[float] = None

    # Volume
    zero_volume_fraction: Optional[float] = None

    # gaps
    max_gap_days: Optional[int] = None

    # Volatility & "חיות" הסדרה
    return_vol_daily: Optional[float] = None
    return_vol_annual: Optional[float] = None

    # Suspicious jumps
    suspicious_jumps_count: int = 0
    suspicious_jump_threshold_sigma: float = 8.0

    # סטטוס כללי
    status: str = "ok"        # "ok" / "warn" / "bad"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["warnings"] = list(self.warnings)
        return d


@dataclass
class PairDataQuality:
    """
    איכות דאטה לזוג:

    • sym_x_quality / sym_y_quality – מילון של SymbolDataQuality.
    • overlap_ratio – יחס ימי חפיפה.
    • aligned_points – כמה נקודות משותפות בפועל.
    • corr_approx_price / corr_approx_rets – קורלציה גסה.
    • mismatch_days – union - intersection.
    """

    pair_label: str
    sym_x: str
    sym_y: str

    sym_x_quality: JSONDict = field(default_factory=dict)
    sym_y_quality: JSONDict = field(default_factory=dict)

    aligned_points: int = 0
    overlap_ratio: Optional[float] = None

    corr_approx_price: Optional[float] = None
    corr_approx_rets: Optional[float] = None

    mismatch_days: Optional[int] = None

    status: str = "ok"  # "ok" / "warn" / "bad"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["warnings"] = list(self.warnings)
        return d


@dataclass
class UniverseDataQualitySummary:
    """
    תקציר איכות יקום:

    • מבוסס על two DataFrames:
        - symbols_df (metadata על סימבולים)
        - pairs_df   (metadata על זוגות)
    """

    num_symbols: int = 0
    num_pairs: int = 0

    avg_symbol_coverage: Optional[float] = None
    min_symbol_coverage: Optional[float] = None
    max_symbol_gap_days: Optional[int] = None

    avg_pair_overlap: Optional[float] = None
    min_pair_overlap: Optional[float] = None
    avg_pair_corr_price: Optional[float] = None

    # רשימות של שמות "גרועים"
    worst_symbols_by_coverage: List[str] = field(default_factory=list)
    worst_symbols_by_nan: List[str] = field(default_factory=list)
    worst_pairs_by_overlap: List[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["worst_symbols_by_coverage"] = list(self.worst_symbols_by_coverage)
        d["worst_symbols_by_nan"] = list(self.worst_symbols_by_nan)
        d["worst_pairs_by_overlap"] = list(self.worst_pairs_by_overlap)
        return d


# ========= Helpers =========

_PRICE_COL_PRIORITY = ("Close", "Adj Close", "close", "adj_close", "price", "LAST_PRICE")
_VOLUME_COL_CANDIDATES = ("Volume", "volume", "VOL", "vol")


def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    מוודא:
    - index DatetimeIndex
    - sorted
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    try:
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.dropna(subset=[out.index.name or out.index.to_series().name])
    except Exception:
        try:
            out.index = pd.to_datetime(out.index, errors="coerce")
        except Exception:
            pass

    out = out.sort_index()
    return out


def _pick_price_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for col in _PRICE_COL_PRIORITY:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    # fallback – העמודה הראשונה
    return pd.to_numeric(df.iloc[:, 0], errors="coerce")


def _pick_volume_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for col in _VOLUME_COL_CANDIDATES:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


# ========= Symbol-level quality computation =========

def compute_symbol_data_quality(
    df: pd.DataFrame,
    symbol: str,
    *,
    suspicious_jump_sigma: float = 8.0,
) -> SymbolDataQuality:
    """
    מחשב איכות דאטה לסימבול בודד מתוך DataFrame של prices (סדרות זמן).

    df: DataFrame עם index (תאריכים) ועמודות מחיר/ווליום.
    """
    df = _normalize_price_df(df)
    dq = SymbolDataQuality(symbol=symbol, suspicious_jump_threshold_sigma=float(suspicious_jump_sigma))

    if df.empty:
        dq.status = "bad"
        dq.warnings.append("empty_df")
        return dq

    dq.rows = int(len(df))

    # index info
    idx = df.index
    try:
        first = idx[0].date()
        last = idx[-1].date()
        dq.start_date = first
        dq.end_date = last
        span_days = (last - first).days + 1
        dq.span_days = span_days
    except Exception:
        dq.start_date = dq.end_date = None
        dq.span_days = None

    # unique days / coverage
    try:
        days = idx.normalize().unique()
        dq.unique_days = int(len(days))
        if dq.span_days and dq.span_days > 0:
            dq.coverage_ratio = float(dq.unique_days / dq.span_days)
            dq.missing_days_est = max(dq.span_days - dq.unique_days, 0)
    except Exception:
        dq.unique_days = None
        dq.coverage_ratio = None
        dq.missing_days_est = None

    # monotonic / duplicates
    dq.non_monotonic_index = not idx.is_monotonic_increasing
    dq.duplicated_rows = int(df.index.duplicated().sum())

    # price series
    price = _pick_price_series(df)
    if price.empty:
        dq.status = "bad"
        dq.warnings.append("no_price_series")
        return dq

    price_nan_frac = float(price.isna().sum() / len(price))
    dq.nan_fraction_price = price_nan_frac

    # all columns NaNs
    all_nan = float(df.isna().sum().sum() / (len(df) * max(len(df.columns), 1)))
    dq.nan_fraction_all = all_nan

    # volume
    vol = _pick_volume_series(df)
    if vol is not None and len(vol) == len(df):
        zero_vol_frac = float((vol.fillna(0) == 0).sum() / len(vol))
        dq.zero_volume_fraction = zero_vol_frac
    else:
        dq.zero_volume_fraction = None

    # gaps
    try:
        unique_days = idx.normalize().unique()
        if len(unique_days) > 1:
            diffs = np.diff(unique_days.astype("datetime64[D]").astype("int"))
            dq.max_gap_days = int(max(diffs))
        else:
            dq.max_gap_days = None
    except Exception:
        dq.max_gap_days = None

    # returns & volatility
    try:
        ret = price.ffill().pct_change().dropna()
        if not ret.empty:
            vol_daily = float(ret.std(ddof=1))
            vol_annual = float(vol_daily * np.sqrt(252))
            dq.return_vol_daily = vol_daily
            dq.return_vol_annual = vol_annual
        else:
            dq.return_vol_daily = dq.return_vol_annual = None
    except Exception:
        dq.return_vol_daily = dq.return_vol_annual = None

    # suspicious jumps – |ret| > suspicious_jump_sigma * std
    try:
        ret = price.ffill().pct_change().dropna()
        if len(ret) > 10:
            std = float(ret.std(ddof=1))
            thr = suspicious_jump_sigma * std if std > 0 else 0
            jumps = (ret.abs() > thr).sum() if thr > 0 else 0
            dq.suspicious_jumps_count = int(jumps)
        else:
            dq.suspicious_jumps_count = 0
    except Exception:
        dq.suspicious_jumps_count = 0

    # status & warnings
    warnings: List[str] = []

    if dq.coverage_ratio is not None and dq.coverage_ratio < 0.5:
        warnings.append("low_coverage")
    if dq.nan_fraction_price is not None and dq.nan_fraction_price > 0.05:
        warnings.append("many_nans_price")
    if dq.nan_fraction_all is not None and dq.nan_fraction_all > 0.1:
        warnings.append("many_nans_all")
    if dq.max_gap_days is not None and dq.max_gap_days > 10:
        warnings.append("large_gaps")
    if dq.duplicated_rows > 0:
        warnings.append("duplicated_rows")
    if dq.non_monotonic_index:
        warnings.append("non_monotonic_index")
    if dq.suspicious_jumps_count > 0:
        warnings.append("suspicious_jumps")

    dq.warnings = warnings

    if "low_coverage" in warnings or "many_nans_price" in warnings or "large_gaps" in warnings:
        dq.status = "warn"
    if "many_nans_all" in warnings or dq.rows == 0:
        dq.status = "bad"

    return dq


def compute_symbol_quality_from_loader(
    loader_fn: PriceLoaderFn,
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    suspicious_jump_sigma: float = 8.0,
) -> SymbolDataQuality:
    """
    עטיפה נוחה כשיש loader_fn(symbol, start, end) שמחזיר DataFrame.
    """
    try:
        df = loader_fn(symbol, start_date, end_date)
    except Exception as e:
        logger.warning("loader_fn failed for symbol %s: %s", symbol, e)
        dq = SymbolDataQuality(symbol=symbol)
        dq.status = "bad"
        dq.warnings.append(f"loader_failed:{e}")
        return dq

    return compute_symbol_data_quality(
        df,
        symbol=symbol,
        suspicious_jump_sigma=suspicious_jump_sigma,
    )


# ========= Pair-level quality =========

def compute_pair_data_quality(
    pair_obj: Any,
    start_date: date,
    end_date: date,
    *,
    pair_legs_loader: PairLegsLoaderFn,
) -> PairDataQuality:
    """
    מחשב איכות דאטה לזוג על בסיס pair_legs_loader:

    pair_legs_loader(pair_obj, start, end) -> (s1, s2, label)
    """
    try:
        s1, s2, label = pair_legs_loader(pair_obj, start_date, end_date)
    except Exception as e:
        logger.warning("pair_legs_loader failed for %r: %s", pair_obj, e)
        return PairDataQuality(
            pair_label=str(pair_obj),
            sym_x="",
            sym_y="",
            status="bad",
            warnings=[f"pair_legs_loader_failed:{e}"],
        )

    sym_x = str(getattr(s1, "name", "X"))
    sym_y = str(getattr(s2, "name", "Y"))

    # נבנה DataFrame לכל leg לחישוב איכות
    df_x = pd.DataFrame({"price": s1}).copy()
    df_y = pd.DataFrame({"price": s2}).copy()

    dq_x = compute_symbol_data_quality(df_x, symbol=sym_x)
    dq_y = compute_symbol_data_quality(df_y, symbol=sym_y)

    # יישור ימי מסחר
    s1 = _normalize_price_df(df_x)["price"]
    s2 = _normalize_price_df(df_y)["price"]

    days_x = set(pd.to_datetime(s1.index).normalize())
    days_y = set(pd.to_datetime(s2.index).normalize())
    inter = days_x & days_y
    union = days_x | days_y

    aligned_points = 0
    overlap_ratio = None
    mismatch_days = None

    if union:
        aligned_points = int(len(inter))
        overlap_ratio = float(len(inter) / len(union)) if len(union) > 0 else None
        mismatch_days = int(len(union) - len(inter))

    # קורלציות
    corr_price = None
    corr_rets = None
    try:
        s1_al, s2_al = s1.align(s2, join="inner")
        if len(s1_al) >= 5:
            corr_price = float(np.corrcoef(s1_al.values, s2_al.values)[0, 1])

            r1 = s1_al.pct_change().dropna()
            r2 = s2_al.pct_change().dropna()
            r1, r2 = r1.align(r2, join="inner")
            if len(r1) >= 5:
                corr_rets = float(np.corrcoef(r1.values, r2.values)[0, 1])
    except Exception:
        pass

    pdq = PairDataQuality(
        pair_label=label,
        sym_x=sym_x,
        sym_y=sym_y,
        sym_x_quality=dq_x.to_dict(),
        sym_y_quality=dq_y.to_dict(),
        aligned_points=aligned_points,
        overlap_ratio=overlap_ratio,
        corr_approx_price=corr_price,
        corr_approx_rets=corr_rets,
        mismatch_days=mismatch_days,
        status="ok",
        warnings=[],
    )

    # warnings / status
    ws: List[str] = []
    if overlap_ratio is not None and overlap_ratio < 0.5:
        ws.append("low_overlap")
    if corr_price is not None and abs(corr_price) < 0.2:
        ws.append("low_corr_price")
    if dq_x.status != "ok" or dq_y.status != "ok":
        ws.append("leg_quality_issue")

    pdq.warnings = ws
    if "low_overlap" in ws or "leg_quality_issue" in ws:
        pdq.status = "warn"
    if dq_x.status == "bad" or dq_y.status == "bad":
        pdq.status = "bad"

    return pdq


# ========= Universe-level quality (symbols / pairs) =========

def compute_universe_symbol_quality(
    symbols: List[str],
    loader_fn: PriceLoaderFn,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    מחזיר DataFrame עם שורות של SymbolDataQuality – מוכן ל-SQL/CSV.
    """
    records: List[JSONDict] = []
    for sym in symbols:
        dq = compute_symbol_quality_from_loader(loader_fn, sym, start_date, end_date)
        records.append(dq.to_dict())
    return pd.DataFrame(records)


def compute_universe_pair_quality(
    pairs: List[Any],
    pair_legs_loader: PairLegsLoaderFn,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    מחזיר DataFrame עם שורות של PairDataQuality – מוכן ל-SQL/CSV.
    """
    records: List[JSONDict] = []
    for p in pairs:
        dq = compute_pair_data_quality(p, start_date, end_date, pair_legs_loader=pair_legs_loader)
        records.append(dq.to_dict())
    return pd.DataFrame(records)


def summarize_universe_quality(
    symbols_df: Optional[pd.DataFrame],
    pairs_df: Optional[pd.DataFrame],
) -> UniverseDataQualitySummary:
    """
    בונה UniverseDataQualitySummary מ-DataFrames של סימבולים וזוגות.
    """
    summary = UniverseDataQualitySummary()

    if symbols_df is not None and not symbols_df.empty:
        df_s = symbols_df.copy()
        summary.num_symbols = int(df_s.shape[0])

        if "coverage_ratio" in df_s.columns:
            cov = pd.to_numeric(df_s["coverage_ratio"], errors="coerce").dropna()
            if not cov.empty:
                summary.avg_symbol_coverage = float(cov.mean())
                summary.min_symbol_coverage = float(cov.min())
                # worst by coverage
                worst = df_s.loc[cov.sort_values().index][:10]
                summary.worst_symbols_by_coverage = worst["symbol"].astype(str).tolist()

        if "max_gap_days" in df_s.columns:
            gaps = pd.to_numeric(df_s["max_gap_days"], errors="coerce").dropna()
            if not gaps.empty:
                summary.max_symbol_gap_days = int(gaps.max())

        if "nan_fraction_price" in df_s.columns:
            nans = pd.to_numeric(df_s["nan_fraction_price"], errors="coerce").fillna(0.0)
            worst_nan = df_s.loc[nans.sort_values(ascending=False).index][:10]
            summary.worst_symbols_by_nan = worst_nan["symbol"].astype(str).tolist()

    if pairs_df is not None and not pairs_df.empty:
        df_p = pairs_df.copy()
        summary.num_pairs = int(df_p.shape[0])

        if "overlap_ratio" in df_p.columns:
            ov = pd.to_numeric(df_p["overlap_ratio"], errors="coerce").dropna()
            if not ov.empty:
                summary.avg_pair_overlap = float(ov.mean())
                summary.min_pair_overlap = float(ov.min())
                worst = df_p.loc[ov.sort_values().index][:10]
                summary.worst_pairs_by_overlap = worst["pair_label"].astype(str).tolist()

        if "corr_approx_price" in df_p.columns:
            cp = pd.to_numeric(df_p["corr_approx_price"], errors="coerce").dropna()
            if not cp.empty:
                summary.avg_pair_corr_price = float(cp.mean())

    return summary


# ========= Helpers ל-SQL / Persist =========

def data_quality_symbols_to_sql_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    פונקציה קטנה שהופכת DataFrame של SymbolDataQuality ל-"SQL-ready":
    - מוודאת שאין רשימות (warnings → '|' joined string).
    """
    if df is None or df.empty:
        return df

    df_sql = df.copy()
    if "warnings" in df_sql.columns:
        df_sql["warnings"] = df_sql["warnings"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else str(x)
        )
    return df_sql


def data_quality_pairs_to_sql_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    הופך DataFrame של PairDataQuality ל-"SQL-ready" (warnings→string).
    """
    if df is None or df.empty:
        return df

    df_sql = df.copy()
    if "warnings" in df_sql.columns:
        df_sql["warnings"] = df_sql["warnings"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else str(x)
        )
    # sym_x_quality / sym_y_quality הם dictים – ל-SQL אפשר לשמור כ-JSON string
    for col in ("sym_x_quality", "sym_y_quality"):
        if col in df_sql.columns:
            df_sql[col] = df_sql[col].apply(
                lambda d: dict(d) if isinstance(d, dict) else {}
            )
    return df_sql


__all__ = [
    "SymbolDataQuality",
    "PairDataQuality",
    "UniverseDataQualitySummary",
    "compute_symbol_data_quality",
    "compute_symbol_quality_from_loader",
    "compute_pair_data_quality",
    "compute_universe_symbol_quality",
    "compute_universe_pair_quality",
    "summarize_universe_quality",
    "data_quality_symbols_to_sql_ready",
    "data_quality_pairs_to_sql_ready",
]
