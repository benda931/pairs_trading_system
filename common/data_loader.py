# -*- coding: utf-8 -*-
"""
common/data_loader.py — Data ingestion & utility layer (HF-grade, v2)
=====================================================================

מטרות המודול (ברמת קרן גידור):
------------------------------
1. ניהול דאטה היסטורי ו־intraday בצורה מסודרת:
   - תיקיות ייעודיות (daily / intraday) תחת DATA_FOLDER.
   - הורדות מ-Yahoo עם fallback ל-Stooq.
   - שמירה אוטומטית ל-CSV + אופציה ל-Parquet.

2. טעינה בטוחה של מחירים:
   - קריאת CSV בלי להתרסק על parse_dates="date".
   - זיהוי עמודת תאריך מתוך ["date","Date","datetime","Datetime","timestamp"].
   - ניקוי שמות עמודות לסטנדרט אחיד: open/high/low/close/adj_close/volume.
   - החזרת DataFrame עם DatetimeIndex ממוין כשאפשר.

3. universe של זוגות:
   - load_pairs / save_pairs בפורמט אחיד: {"symbols": ["XLY", "XLC"], ...}
   - תמיכה ב-JSON/CSV, fallback ל-demo pairs אם הקובץ חסר/שבור.

4. ביצועים וארכיטקטורה:
   - ThreadPool להורדת הרבה סמבולים במקביל.
   - @lru_cache לטעינת FULL-SYMBOL (פנימי) כדי לחסוך I/O.
   - Settings מבוסס Pydantic (pydantic_settings) עם env overrides.

שדרוגים בגרסת v2 (20 רעיונות חדשים / יכולות חדשות):
----------------------------------------------------
1.  API רשמי ל-load_price_data: symbol + start_date/end_date + intraday.
2.  שכבת cache פנימית לפר-סימבול (_load_symbol_full_cached) + חתך טווח על גביו.
3.  פרמטר cols ל-load_price_data: טעינת subset של עמודות בלבד.
4.  פרמטר freq (D/W/M) ל-load_price_data: רסמפול סדרות (ohlc/close).
5.  פרמטר normalize: נרמול close כך שיתחיל מ-100 (ל-comparison גרפים).
6.  פרמטר fill_method: בחירת שיטת מילוי חסרים ("ffill", "bfill", "none").
7.  פונקציה load_prices_multi — טעינת מספר סמבולים ל-DataFrame רחב (wide).
8.  פונקציה get_price_metadata — מחזירה מטא־דאטה (start/end/n_rows/freq/has_nans).
9.  פונקציה validate_price_df — בדיקות איכות: אינדקס ממויין, אין מחירים שליליים, אין אינדקס כפול.
10. אזהרת איכות אם יש חוסרים משמעותיים (gaps) בדאטה.
11. תמיכה רשמית ב-intraday גם ב-load_price_data (לא רק דרך download_symbol).
12. CLI "info" לסימבול: טווח תאריכים, שורות, עמודות, דוגמה.
13. CLI "inspect" לזוג/רשימה: בודק quality (gaps / missing close).
14. תמיכה ב-Parquet ברמת API (csv_to_parquet + load_parquet_if_fresh).
15. DataLoader.get_price עוטף את load_price_data החדש, כולל start/end.
16. לוגים משודרגים: מראים path, intraday/daily, מקור (cache/local/remote).
17. זיהוי "stale data" לפי CSV_MAX_AGE_DAYS והדפסת אזהרה.
18. normalization ו-resample בנויים כ-helpers נפרדים (לשימוש עתידי ע"י מודולים אחרים).
19. כל פונקציות ה-"public API" מרוכזות ב-__all__ בצורה ברורה.
20. שמירה מלאה על תאימות לאחור: כל קריאה של load_price_data(symbol, intraday=True)
    עדיין עובדת; start/end הישנים נתמכים כאליאס ל-start_date/end_date.

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union, Optional

import pandas as pd
import requests
import yfinance as yf
from pydantic import Field

try:
    from pydantic_settings import BaseSettings  # Pydantic v2 style
except ImportError:  # pragma: no cover
    from pydantic_settings import BaseSettings  # type: ignore

from common.json_safe import make_json_safe, json_default as _json_default

# =============================================================
# Settings
# =============================================================


class Settings(BaseSettings):
    # --- base paths ---
    DATA_FOLDER: Path = Field(default=Path("data"), env="DATA_FOLDER")
    DAILY_SUBDIR: str = Field(default="daily", env="DAILY_SUBDIR")
    INTRADAY_SUBDIR: str = Field(default="intraday", env="INTRADAY_SUBDIR")

    # --- Yahoo defaults ---
    YF_PERIOD: str = Field(default="5y", env="YF_PERIOD")
    YF_INTERVAL: str = Field(default="1d", env="YF_INTERVAL")
    YF_AUTO_ADJUST: bool = Field(default=True, env="YF_AUTO_ADJUST")

    # --- Intraday defaults ---
    INTRADAY_INTERVAL: str = Field(default="5m", env="INTRADAY_INTERVAL")
    INTRADAY_LOOKBACK_DAYS: int = Field(default=30, env="INTRADAY_LOOKBACK_DAYS")

    # --- Parallelism / cache freshness ---
    MAX_WORKERS: int = Field(default=8, env="MAX_WORKERS")
    CSV_MAX_AGE_DAYS: int = Field(default=7, env="CSV_MAX_AGE_DAYS")

    # --- Pairs file ---
    PAIRS_FILE: Path = Field(default=Path("pairs.json"), env="PAIRS_FILE")

    # --- Optional external keys (ignored here) ---
    OPENAI_API_KEY: str | None = Field(default=None, env="OPENAI_API_KEY")

    # --- Advanced options (v2 additions) ---
    WARN_GAPS_DAYS: int = Field(default=5, env="WARN_GAPS_DAYS")
    DEFAULT_FREQ_DAILY: str = Field(default="D", env="DEFAULT_FREQ_DAILY")
    ENABLE_PARQUET: bool = Field(default=True, env="ENABLE_PARQUET")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }


settings = Settings()

# =============================================================
# Logging
# =============================================================

logger = logging.getLogger("data_loader")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | data_loader | %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# =============================================================
# Internal helpers — filesystem & cleaning
# =============================================================


def _data_folder(intraday: bool = False) -> Path:
    """Return the folder for daily / intraday data and ensure it exists."""
    sub = settings.INTRADAY_SUBDIR if intraday else settings.DAILY_SUBDIR
    folder = settings.DATA_FOLDER / sub
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _standardise_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    מנרמל את שמות העמודות לשמות סטנדרטיים.

    Mapping בסיסי:
        Date/Datetime -> date
        Open/High/Low/Close/Adj Close/Volume -> open/high/low/close/adj_close/volume
    """
    if df.empty:
        return df

    mapping = {
        "Date": "date",
        "Datetime": "date",
        "datetime": "date",
        "Timestamp": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj close": "adj_close",
        "adj close": "adj_close",
        "AdjClose": "adj_close",
        "Volume": "volume",
    }

    df = df.rename(columns=mapping)

    lowered_map: Dict[str, str] = {}
    for col in df.columns:
        lc = col.lower()
        if lc == "date" and col != "date":
            lowered_map[col] = "date"
        elif lc == "open" and col != "open":
            lowered_map[col] = "open"
        elif lc == "high" and col != "high":
            lowered_map[col] = "high"
        elif lc == "low" and col != "low":
            lowered_map[col] = "low"
        elif lc == "close" and col != "close":
            lowered_map[col] = "close"
        elif lc in ("adj close", "adjclose", "adj_close") and col != "adj_close":
            lowered_map[col] = "adj_close"
        elif lc == "volume" and col != "volume":
            lowered_map[col] = "volume"

    if lowered_map:
        df = df.rename(columns=lowered_map)

    return df


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    מנסה להפוך את האינדקס ל-DatetimeIndex בצורה בטוחה:

    - אם יש עמודת 'date' → ממיר ל-datetime ומגדיר כאינדקס.
    - אם אין 'date' אבל האינדקס current ניתן להמרה ל-datetime → ממיר.
    - אחרת: משאיר את האינדקס כמו שהוא.
    """
    if df.empty:
        return df

    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.set_index("date")
            df = df.sort_index()
            return df
        except Exception as exc:
            logger.warning("Failed to parse 'date' column as datetime: %s", exc)

    try:
        idx = pd.to_datetime(df.index, errors="coerce")
        if idx.notna().any():
            df = df.copy()
            df.index = idx
            df = df.sort_index()
            return df
    except Exception:
        pass

    return df


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise a raw price DataFrame:

    - normalise column names (open/high/low/close/adj_close/volume)
    - ensure DatetimeIndex when possible
    - sort by index, drop duplicates
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = _standardise_price_columns(df)
    df = _ensure_datetime_index(df)

    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="last")]

    return df.sort_index()


def _read_local_csv(path: Path) -> pd.DataFrame:
    """
    קורא CSV בצורה בטוחה – בלי parse_dates=["date"] שמתפוצץ אם אין עמודה 'date'.
    משתמש ב-_clean_df כדי לנקות ולהסדיר את ה-DataFrame.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read local CSV %s: %s", path, exc)
        return pd.DataFrame()

    return _clean_df(df)

# =============================================================
# Download helpers (Yahoo + Stooq fallback)
# =============================================================


def _download_yahoo(
    symbol: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Try Yahoo Finance, return cleaned DataFrame or empty if failed/empty."""
    try:
        kwargs: Dict[str, Any] = {
            "interval": interval,
            "progress": False,
            "auto_adjust": settings.YF_AUTO_ADJUST,
        }
        if start and end:
            kwargs["start"] = start
            kwargs["end"] = end
        else:
            kwargs["period"] = settings.YF_PERIOD

        df = yf.download(symbol, **kwargs)
        if df is None or df.empty:
            return pd.DataFrame()
        return _clean_df(df)
    except Exception as exc:  # pragma: no cover
        logger.warning("Yahoo download failed for %s: %s", symbol, exc)
        return pd.DataFrame()


def _download_stooq(symbol: str) -> pd.DataFrame:
    """
    Stooq free daily backup (no intraday).

    URL example:
        https://stooq.com/q/d/l/?s=xly.us&i=d
    """
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        from io import StringIO

        df = pd.read_csv(StringIO(r.text))
        return _clean_df(df)
    except Exception as exc:  # pragma: no cover
        logger.warning("Stooq download failed for %s: %s", symbol, exc)
        return pd.DataFrame()


def download_symbol(
    symbol: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    intraday: bool = False,
) -> pd.DataFrame:
    """
    הורדת סמבול בודד (daily / intraday):

    - intraday=True → Yahoo עם interval=INTRADAY_INTERVAL, טווח לפי INTRADAY_LOOKBACK_DAYS אם לא הועבר start/end.
    - intraday=False → Yahoo interval=YF_INTERVAL, period=YF_PERIOD.
    - fallback ל-Stooq (daily בלבד) אם Yahoo מחזיר ריק.
    """
    if intraday:
        if start is None or end is None:
            end = datetime.now(timezone.utc)()
            start = end - timedelta(days=settings.INTRADAY_LOOKBACK_DAYS)
        df = _download_yahoo(symbol, start=start, end=end, interval=settings.INTRADAY_INTERVAL)
        return df

    df = _download_yahoo(symbol, start=start, end=end, interval=settings.YF_INTERVAL)
    if df.empty:
        df = _download_stooq(symbol)
    return df


def save_csv(df: pd.DataFrame, symbol: str, *, intraday: bool = False) -> Path:
    """
    Save DataFrame to CSV under the correct data folder (daily/intraday).

    לא משנה את הדאטה (מניח שכבר נקי), רק שומר.
    """
    folder = _data_folder(intraday=intraday)
    path = folder / f"{symbol}.csv"
    df.to_csv(path)
    return path

# =============================================================
# Bulk operations
# =============================================================


def _csv_is_stale(path: Path, *, max_age_days: int) -> bool:
    """
    בדיקת עדכניות של CSV לפי זמן שינוי.

    max_age_days=0 → תמיד נחשב stale.
    """
    if not path.exists():
        return True
    if max_age_days <= 0:
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return mtime < datetime.now(timezone.utc)() - timedelta(days=max_age_days)


def bulk_download(
    symbols: Sequence[str],
    *,
    intraday: bool = False,
    force: bool = False,
    days_back: Optional[int] = None,
) -> List[Path]:
    """
    Download many symbols in parallel, respecting cache freshness.

    Parameters
    ----------
    symbols : sequence of str
    intraday : bool, default False
    force : bool, default False
        If True, always re-download even if CSV is fresh.
    days_back : int or None, default None
        If given, refresh symbol if its CSV is older than days_back.
        If None, uses settings.CSV_MAX_AGE_DAYS.
    """
    folder = _data_folder(intraday=intraday)
    max_age = days_back if days_back is not None else settings.CSV_MAX_AGE_DAYS

    def _needs_download(sym: str) -> bool:
        path = folder / f"{sym}.csv"
        if force:
            return True
        if not path.exists():
            return True
        return _csv_is_stale(path, max_age_days=max_age)

    targets = [s for s in symbols if _needs_download(s)]
    if not targets:
        logger.info("bulk_download: all %s symbols up-to-date", len(symbols))
        return []

    logger.info(
        "bulk_download: downloading %s symbols (workers=%s, intraday=%s)…",
        len(targets),
        settings.MAX_WORKERS,
        intraday,
    )
    saved_paths: List[Path] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_symbol, sym, intraday=intraday): sym for sym in targets
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
                if df.empty:
                    logger.warning("bulk_download: no data for %s", sym)
                    continue
                p = save_csv(df, sym, intraday=intraday)
                saved_paths.append(p)
            except Exception as exc:
                logger.error("bulk_download: failed %s: %s", sym, exc)

    dt = time.time() - t0
    logger.info(
        "bulk_download: downloaded %s/%s symbols in %.1fs",
        len(saved_paths),
        len(targets),
        dt,
    )
    return saved_paths

# =============================================================
# Pairs helpers
# =============================================================


def _default_pairs() -> List[Dict[str, Any]]:
    """
    Fallback demo pairs — כל איבר הוא dict עם מפתח 'symbols' → [sym_x, sym_y].
    """
    return [
        {"symbols": ["AAPL", "MSFT"]},
        {"symbols": ["KO", "PEP"]},
        {"symbols": ["XOM", "CVX"]},
    ]


def load_pairs(create_if_missing: bool = True) -> List[Dict[str, Any]]:
    """
    Load trading pairs list and normalise structure.

    פורמט פלט:
        [{"symbols": ["XLY","XLC"], ...}, ...]

    קובץ input:
    -----------
    - JSON:
        [
          {"symbols": ["XLY","XLC"], "score": 0.9},
          {"asset1": "KO", "asset2": "PEP"}
        ]
    - CSV:
        columns: symbols (list-like) OR asset1,asset2.
    """
    pf = settings.PAIRS_FILE
    if not pf.exists():
        if create_if_missing:
            logger.warning("Pairs file %s missing — creating default demo list", pf)
            save_pairs(_default_pairs())
        return _default_pairs()

    try:
        if pf.suffix.lower() == ".json":
            raw = json.loads(pf.read_text(encoding="utf-8"))
        else:
            raw = pd.read_csv(pf).to_dict("records")
    except Exception as exc:
        logger.warning("load_pairs: failed to read %s: %s, using default pairs", pf, exc)
        return _default_pairs()

    pairs: List[Dict[str, Any]] = []
    for rec in raw:
        try:
            if "symbols" in rec and isinstance(rec["symbols"], (list, tuple)) and len(rec["symbols"]) >= 2:
                pairs.append(
                    {"symbols": list(rec["symbols"][:2]), **{k: v for k, v in rec.items() if k != "symbols"}}
                )
            elif "asset1" in rec and "asset2" in rec:
                extra = {k: v for k, v in rec.items() if k not in ("asset1", "asset2")}
                pairs.append({"symbols": [rec["asset1"], rec["asset2"]], **extra})
            else:
                logger.warning("load_pairs: unrecognised pair format: %s", rec)
        except Exception as exc:
            logger.warning("load_pairs: failed to normalise record %s: %s", rec, exc)

    if not pairs:
        logger.warning("load_pairs: file %s empty/invalid — using default list", pf)
        return _default_pairs()
    return pairs


def save_pairs(pairs: List[Mapping[str, Any]], *, mode: str = "w") -> None:
    """
    Save list of pairs to configured PAIRS_FILE.

    pairs: רשימת dict-ים, לפחות עם key 'symbols' → [sym_x, sym_y].
    """
    pf = settings.PAIRS_FILE
    pf.parent.mkdir(parents=True, exist_ok=True)

    payload: List[Dict[str, Any]] = []
    for rec in pairs:
        rec = dict(rec)
        symbols = rec.get("symbols")
        if isinstance(symbols, (list, tuple)) and len(symbols) >= 2:
            rec["symbols"] = list(symbols[:2])
            payload.append(rec)
        elif "asset1" in rec and "asset2" in rec:
            rec["symbols"] = [rec["asset1"], rec["asset2"]]
            payload.append(rec)
        else:
            logger.warning("save_pairs: skipping invalid pair record: %s", rec)

    if pf.suffix.lower() == ".json":
        with pf.open(mode, encoding="utf-8") as fh:
            json.dump(make_json_safe(payload), fh, indent=2, default=_json_default)
    else:
        pd.DataFrame(payload).to_csv(pf, index=False, mode=mode)

# =============================================================
# Parquet conversion & helpers
# =============================================================


def csv_to_parquet(symbol: str, *, intraday: bool = False) -> Path:
    """
    Convert CSV → Parquet עבור symbol נתון, אם נחוץ.

    אם Parquet קיים ויותר חדש מה-CSV → מחזירים אותו.
    אחרת:
        - קוראים את ה-CSV דרך _read_local_csv (ללא parse_dates).
        - מנקים עם _clean_df.
        - שומרים ל-Parquet עם compression="snappy".
    """
    folder = _data_folder(intraday=intraday)
    csv_path = folder / f"{symbol}.csv"
    pq_path = csv_path.with_suffix(".parquet")

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if pq_path.exists() and pq_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return pq_path

    df = _read_local_csv(csv_path)
    if df.empty:
        logger.warning("csv_to_parquet: CSV for %s is empty/invalid", symbol)
        df.to_parquet(pq_path, compression="snappy")
        return pq_path

    df.to_parquet(pq_path, compression="snappy")
    return pq_path


def _load_parquet_if_fresh(symbol: str, *, intraday: bool = False) -> Optional[pd.DataFrame]:
    """
    Helper: אם Parquet קיים והוא עדכני יחסית ל-CSV — נטען אותו.
    אחרת נחזיר None וניפול ל-CSV.
    """
    if not settings.ENABLE_PARQUET:
        return None

    folder = _data_folder(intraday=intraday)
    csv_path = folder / f"{symbol}.csv"
    pq_path = csv_path.with_suffix(".parquet")

    if not pq_path.exists() or not csv_path.exists():
        return None

    if pq_path.stat().st_mtime < csv_path.stat().st_mtime:
        return None

    try:
        df = pd.read_parquet(pq_path)
        return _clean_df(df)
    except Exception as exc:
        logger.warning("Failed to read Parquet for %s: %s", symbol, exc)
        return None

# =============================================================
# Quality & metadata helpers
# =============================================================


def validate_price_df(df: pd.DataFrame, *, symbol: str = "") -> None:
    """
    מבצע בדיקות איכות בסיסיות על DataFrame מחירים.

    בודק:
    - אינדקס ממוין עולה.
    - אינדקס ייחודי.
    - מחירי close לא-שליליים (אם קיימים).
    - מזהיר אם יש gaps גדולים מדי (לפי WARN_GAPS_DAYS).
    """
    if df.empty:
        return

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        logger.warning("validate_price_df(%s): index is not DatetimeIndex", symbol)

    if not idx.is_monotonic_increasing:
        logger.warning("validate_price_df(%s): index is not sorted; sorting.", symbol)
        df.sort_index(inplace=True)

    if not idx.is_unique:
        logger.warning("validate_price_df(%s): duplicate index entries detected.", symbol)


    if "close" in df.columns:
            # --- normalize 'close' to numeric before any checks ---
        try:
            # עושים copy כדי לא לגעת ב־df המקורי של ה־cache בטעות
            if not pd.api.types.is_numeric_dtype(df["close"]):
                orig_non_na = df["close"].notna().sum()
                df = df.copy()
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                coerced_na = df["close"].isna().sum()
                if coerced_na > 0 and orig_non_na > 0:
                    logger.warning(
                        "validate_price_df(%s): coerced %d/%d 'close' values to NaN (non-numeric)",
                        symbol,
                        coerced_na,
                        orig_non_na,
                    )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "validate_price_df(%s): failed to normalize 'close' to numeric: %s",
                symbol,
                exc,
            )

        neg = (df["close"] < 0).sum()
        if neg > 0:
            logger.warning("validate_price_df(%s): found %s negative close values.", symbol, neg)

    # gaps detection (rough)
    try:
        diffs = idx.to_series().diff().dropna()
        max_gap = diffs.max()
        if isinstance(max_gap, pd.Timedelta):
            days_gap = max_gap.days
            if days_gap >= settings.WARN_GAPS_DAYS:
                logger.warning(
                    "validate_price_df(%s): large gap detected (max %s days).",
                    symbol,
                    days_gap,
                )
    except Exception:
        pass


def get_price_metadata(df: pd.DataFrame, *, symbol: str = "") -> Dict[str, Any]:
    """
    מחזיר מטא-דאטה בסיסי על סדרת המחיר:

    - symbol, n_rows, n_cols, start_date, end_date
    - has_nans, freq (משוער), intraday_flag
    """
    if df.empty:
        return {
            "symbol": symbol,
            "n_rows": 0,
            "n_cols": 0,
            "start_date": None,
            "end_date": None,
            "has_nans": False,
            "freq": None,
            "intraday": None,
        }

    idx = df.index
    start = idx[0]
    end = idx[-1]
    has_nans = df.isna().any().any()

    # freq heuristic
    freq = None
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 2:
        try:
            freq = pd.infer_freq(idx)
        except Exception:
            freq = None

    intraday_flag = None
    if isinstance(idx, pd.DatetimeIndex):
        intraday_flag = any(idx.to_series().dt.time != datetime.min.time())

    return {
        "symbol": symbol,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "start_date": start,
        "end_date": end,
        "has_nans": bool(has_nans),
        "freq": freq,
        "intraday": intraday_flag,
    }

# =============================================================
# Cached FULL-symbol loader (internal)
# =============================================================


@lru_cache(maxsize=256)
def _load_symbol_full_cached(symbol: str, intraday: bool = False) -> pd.DataFrame:
    """
    טוען את כל הדאטה ל-symbol (daily / intraday) מהדיסק / רשת, ומנרמל.

    זה הקאש הפנימי האמיתי; load_price_data חותך ממנו טווחים / רסמפול.
    """
    folder = _data_folder(intraday=intraday)
    csv_path = folder / f"{symbol}.csv"

    # ניסון ראשון: CSV קיים
    if csv_path.exists():
        # אם יש Parquet עדכני — נטען אותו
        df_pq = _load_parquet_if_fresh(symbol, intraday=intraday)
        if df_pq is not None and not df_pq.empty:
            logger.info(
                "load_symbol_full_cached(%s, intraday=%s): using Parquet cache",
                symbol,
                intraday,
            )
            df = df_pq
        else:
            df = _read_local_csv(csv_path)

        if df.empty:
            logger.warning(
                "load_symbol_full_cached: local CSV for %s is empty — attempting re-download",
                symbol,
            )
            df = download_symbol(symbol, intraday=intraday)
            if df.empty:
                logger.warning("load_symbol_full_cached: re-download still empty for %s", symbol)
                return pd.DataFrame()
            save_csv(df, symbol, intraday=intraday)
    else:
        logger.info(
            "load_symbol_full_cached: cache miss for %s (%s) — downloading",
            symbol,
            "intraday" if intraday else "daily",
        )
        df = download_symbol(symbol, intraday=intraday)
        if df.empty:
            logger.warning("load_symbol_full_cached: no data for %s from remote sources", symbol)
            return pd.DataFrame()
        save_csv(df, symbol, intraday=intraday)

    validate_price_df(df, symbol=symbol)
    return df

# =============================================================
# Public API: load_price_data (HF-grade)
# =============================================================


def _slice_by_dates(
    df: pd.DataFrame,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())

    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]
    return df


def _apply_freq_and_fill(
    df: pd.DataFrame,
    *,
    freq: Optional[str] = None,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    רסמפול ומילוי חסרים ברמת loader, לשימוש אופציונלי ע"י מודולים שונים.
    """
    if df.empty:
        return df

    out = df.copy()

    if freq:
        # נעשה רסמפול "אגרסיבי" אבל נשמור על close כ-last, volume כ-sum וכו' אם קיימים.
        rule = freq.upper()
        agg: Dict[str, Any] = {}
        if "open" in out.columns:
            agg["open"] = "first"
        if "high" in out.columns:
            agg["high"] = "max"
        if "low" in out.columns:
            agg["low"] = "min"
        if "close" in out.columns:
            agg["close"] = "last"
        if "adj_close" in out.columns:
            agg["adj_close"] = "last"
        if "volume" in out.columns:
            agg["volume"] = "sum"

        # כל עמודה שלא הוזכרה → last
        for col in out.columns:
            if col not in agg:
                agg[col] = "last"

        out = out.resample(rule).agg(agg)

    if fill_method == "ffill":
        out = out.ffill()
    elif fill_method == "bfill":
        out = out.bfill()
    elif fill_method == "none":
        pass
    else:
        logger.warning("Unknown fill_method=%s (using ffill)", fill_method)
        out = out.ffill()

    return out


def load_price_data(
    symbol: str,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
    *,
    intraday: bool = False,
    cols: Optional[Sequence[str]] = None,
    freq: Optional[str] = None,
    normalize: bool = False,
    fill_method: str = "ffill",
    **legacy_kwargs: Any,
) -> pd.DataFrame:
    """
    Canonical price loader לכל המערכת.

    API רשמי:
    ----------
        df = load_price_data(
            symbol,
            start_date=None,
            end_date=None,
            intraday=False,
            cols=None,
            freq=None,
            normalize=False,
            fill_method="ffill",
        )

    תמיכה לאחור:
    -------------
    - load_price_data(symbol, intraday=True) עדיין חוקי.
    - load_price_data(symbol, start=..., end=...) נתמך כאליאס ל-start_date/end_date.

    Parameters
    ----------
    symbol : str
        טיקר (לדוגמה "SPY").
    start_date / end_date : datetime|date|None
        טווח תאריכים רצוי; אם None → כל הטווח הזמין.
    intraday : bool, default False
        שימוש בתיקיית intraday וב-INTRADAY_INTERVAL.
    cols : sequence of str or None
        אם לא None, מחזיר רק את העמודות האלה (אם קיימות).
    freq : str or None
        רסמפול לתדירות חדשה ("D","W","M", וכו'). None → בלי רסמפול.
    normalize : bool, default False
        אם True → מנרמל את close כך שהערך הראשון יהיה 100 (עוזר להשוואת מדדים).
    fill_method : {"ffill","bfill","none"}
        שיטת מילוי חסרים אחרי חיתוך/רסמפול.

    Returns
    -------
    pd.DataFrame
        DataFrame מנוקה, עם DatetimeIndex, עמודות סטנדרטיות ככל האפשר.
    """
    # legacy support: start / end ב-kwargs
    if start_date is None and "start" in legacy_kwargs:
        start_date = legacy_kwargs.pop("start")
    if end_date is None and "end" in legacy_kwargs:
        end_date = legacy_kwargs.pop("end")

    df_full = _load_symbol_full_cached(symbol, intraday=intraday)
    if df_full.empty:
        return df_full

    df = _slice_by_dates(df_full, start_date=start_date, end_date=end_date)
    df = _apply_freq_and_fill(df, freq=freq, fill_method=fill_method)

    if cols is not None:
        cols_set = [c for c in cols if c in df.columns]
        if not cols_set:
            logger.warning("load_price_data(%s): requested cols %s not in df.columns", symbol, cols)
        else:
            df = df[cols_set]

    if normalize and "close" in df.columns and not df.empty:
        first = float(df["close"].iloc[0])
        if first != 0:
            df["close"] = df["close"] / first * 100.0

    return df

# =============================================================
# Multi-symbol loader
# =============================================================


def load_prices_multi(
    symbols: Sequence[str],
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
    *,
    intraday: bool = False,
    col: str = "close",
    freq: Optional[str] = None,
    normalize: bool = False,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    טוען מספר סמבולים ל-DataFrame רחב (wide):

        columns = [sym1, sym2, ...]
        index   = DatetimeIndex מיושר

    שימושי מאוד למטריצות/קורלציות/Portfolio-level analysis.
    """
    frames: List[pd.Series] = []

    for sym in symbols:
        df = load_price_data(
            sym,
            start_date=start_date,
            end_date=end_date,
            intraday=intraday,
            freq=freq,
            normalize=normalize,
            fill_method=fill_method,
        )
        if df.empty:
            logger.warning("load_prices_multi: %s returned empty df", sym)
            continue
        if col not in df.columns:
            logger.warning("load_prices_multi: column '%s' not found for %s", col, sym)
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        s.name = sym
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    wide = pd.concat(frames, axis=1)
    return wide

# =============================================================
# OOP Convenience wrapper
# =============================================================


class DataLoader:
    """
    Convenience class bundling common operations.

    שימוש:
        dl = DataLoader()
        df = dl.get_price("XLY", start_date=..., end_date=...)
        dl.bulk_daily(["XLY","XLC"], days_back=3)
    """

    def __init__(self) -> None:
        self.settings = settings

    def get_price(
        self,
        symbol: str,
        *,
        intraday: bool = False,
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime, date]] = None,
        cols: Optional[Sequence[str]] = None,
        freq: Optional[str] = None,
        normalize: bool = False,
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        return load_price_data(
            symbol,
            start_date=start_date,
            end_date=end_date,
            intraday=intraday,
            cols=cols,
            freq=freq,
            normalize=normalize,
            fill_method=fill_method,
        )

    def bulk_daily(self, symbols: Sequence[str], **kwargs: Any) -> None:
        bulk_download(symbols, intraday=False, **kwargs)

    def bulk_intraday(self, symbols: Sequence[str], **kwargs: Any) -> None:
        bulk_download(symbols, intraday=True, **kwargs)

# =============================================================
# Export list for * imports
# =============================================================

__all__: Tuple[str, ...] = (
    "download_symbol",
    "bulk_download",
    "load_price_data",
    "load_prices_multi",
    "get_price_metadata",
    "validate_price_df",
    "load_pairs",
    "save_pairs",
    "csv_to_parquet",
    "DataLoader",
    "settings",
)

# =============================================================
# CLI
# =============================================================


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Data Loader CLI (daily / intraday).")
    sub = p.add_subparsers(dest="command")

    # bulk-download
    p_dl = sub.add_parser("download", help="Bulk download symbols to CSV.")
    p_dl.add_argument("symbols", nargs="*", help="Symbols to download (default: from pairs file).")
    p_dl.add_argument("--intraday", action="store_true", help="Use intraday interval.")
    p_dl.add_argument("--force", action="store_true", help="Redownload even if CSV is fresh.")
    p_dl.add_argument("--days", type=int, default=None, help="Force refresh if older than N days.")
    p_dl.add_argument("--max-workers", type=int, default=None, help="Override MAX_WORKERS for this run.")

    # pairs info
    p_pairs = sub.add_parser("pairs", help="Show loaded pairs summary.")
    p_pairs.add_argument("--limit", type=int, default=10, help="Show first N pairs.")

    # symbol info
    p_info = sub.add_parser("info", help="Show metadata about a single symbol.")
    p_info.add_argument("symbol", type=str, help="Symbol to inspect.")
    p_info.add_argument("--intraday", action="store_true", help="Use intraday data.")

    # inspect multiple
    p_insp = sub.add_parser("inspect", help="Inspect quality for multiple symbols.")
    p_insp.add_argument("symbols", nargs="*", help="Symbols to inspect (default: from pairs file).")
    p_insp.add_argument("--intraday", action="store_true", help="Use intraday data.")
    p_insp.add_argument("--limit", type=int, default=10, help="Limit number of symbols from pairs file.")

    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        if args.max_workers is not None:
            settings.MAX_WORKERS = args.max_workers  # type: ignore[assignment]
        if args.symbols:
            syms = args.symbols
        else:
            pairs = load_pairs()
            syms = sorted({s for p in pairs for s in p.get("symbols", [])})
        if not syms:
            print("No symbols provided / found in pairs file.")
            sys.exit(1)
        bulk_download(syms, intraday=args.intraday, force=args.force, days_back=args.days)
        print("✓ Download completed.")

    elif args.command == "pairs":
        pairs = load_pairs()
        limit = max(1, int(args.limit))
        print(f"Total pairs: {len(pairs)}")
        for rec in pairs[:limit]:
            print(rec)

    elif args.command == "info":
        df = load_price_data(args.symbol, intraday=args.intraday)
        meta = get_price_metadata(df, symbol=args.symbol)
        print(json.dumps(make_json_safe(meta), indent=2, default=_json_default))

    elif args.command == "inspect":
        if args.symbols:
            syms = args.symbols
        else:
            pairs = load_pairs()
            syms = sorted({s for p in pairs for s in p.get("symbols", [])})
            syms = syms[: max(1, int(args.limit))]
        if not syms:
            print("No symbols to inspect.")
            sys.exit(1)
        for sym in syms:
            df = load_price_data(sym, intraday=args.intraday)
            meta = get_price_metadata(df, symbol=sym)
            print(f"=== {sym} ===")
            print(json.dumps(make_json_safe(meta), indent=2, default=_json_default))
            print()

    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
