# -*- coding: utf-8 -*-
"""
common/price_loader.py — Legacy-compatible price loader adapter (HF-grade)
=========================================================================

תפקיד המודול הזה:
-----------------
1. **תאימות לאחור**:
   - כל קוד ישן שעושה:
        from common.price_loader import load_price_data
     יקבל היום את ה-API החדש מ-common.data_loader.

2. **שכבת נוחות**:
   - get_daily_price(...)        — עטיפה נוחה ל-daily.
   - get_intraday_price(...)     — עטיפה נוחה ל-intraday.
   - ensure_prices_cached(...)   — הורדה/רענון של CSV-ים לדיסק.
   - load_prices_multi(...)      — טעינה ל-DataFrame רחב (מטריצות / קורלציה).
   - DataLoader                  — עטיפת OOP (כניסה מרכזית לדאטה).

3. **שקיפות**:
   - לוג ברור אם data_loader לא זמין.
   - התנהגות אחידה בכל המערכת (Backtest, Optimization, Universe).

הערה:
------
- זהו מודול "bridge". רוב הלוגיקה החכמה יושבת ב-common.data_loader.
- אם אתה כותב קוד חדש, עדיף לייבא ישירות מ-common.data_loader,
  אבל price_loader נשאר כדי לא לשבור קוד קיים כמו generate_pairs_universe.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import pandas as pd

logger = logging.getLogger("price_loader")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | price_loader | %(message)s"
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Try to import the new HF-grade loader from common.data_loader
# ---------------------------------------------------------------------------

try:
    from .data_loader import (  # type: ignore[import]
        load_price_data as _core_load_price_data,
        load_prices_multi as _core_load_prices_multi,
        bulk_download as _core_bulk_download,
        DataLoader as _CoreDataLoader,
        settings as _core_settings,
        get_price_metadata as _core_get_price_metadata,
        validate_price_df as _core_validate_price_df,
    )
    _HAS_CORE = True
except Exception as exc:  # pragma: no cover
    logger.warning(
        "Failed to import core data_loader; price_loader will be in degraded mode: %s",
        exc,
    )
    _core_load_price_data = None  # type: ignore[assignment]
    _core_load_prices_multi = None  # type: ignore[assignment]
    _core_bulk_download = None  # type: ignore[assignment]
    _CoreDataLoader = object  # type: ignore[assignment]
    _core_settings = None  # type: ignore[assignment]
    _core_get_price_metadata = None  # type: ignore[assignment]
    _core_validate_price_df = None  # type: ignore[assignment]
    _HAS_CORE = False

Number = Union[int, float]

# ---------------------------------------------------------------------------
# Public API — legacy-compatible wrappers
# ---------------------------------------------------------------------------


def load_price_data(
    symbol: str,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
    *,
    intraday: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    עטיפה תאימה לאחור ל-loader החדש.

    חתימה נתמכת:
        load_price_data(symbol, start_date=None, end_date=None, intraday=False, **kwargs)
    וגם:
        load_price_data(symbol, start=..., end=..., intraday=False, **kwargs)
    (start/end כ-kwargs יעברו ל-start_date/end_date).

    זה מאפשר קוד ישן כמו generate_pairs_universe.py לקרוא ל-load_price_data
    בלי לדעת שהתשתית כבר עברה ל-common.data_loader.
    """
    if not _HAS_CORE or _core_load_price_data is None:
        logger.warning("load_price_data called but data_loader core is not available.")
        return pd.DataFrame()

    # תמיכה בקריאות legacy: load_price_data(symbol, start=..., end=...)
    if start_date is None and "start" in kwargs:
        start_date = kwargs.pop("start")
    if end_date is None and "end" in kwargs:
        end_date = kwargs.pop("end")

    return _core_load_price_data(
        symbol,
        start_date=start_date,
        end_date=end_date,
        intraday=intraday,
        **kwargs,
    )


def get_daily_price(
    symbol: str,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    עטיפה נוחה לבקשת דאטה יומי (Daily).

    דוגמה:
        df = get_daily_price("SPY", start_date="2020-01-01", end_date="2024-01-01")
    """
    return load_price_data(symbol, start_date=start_date, end_date=end_date, intraday=False, **kwargs)


def get_intraday_price(
    symbol: str,
    *,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    עטיפה נוחה לדאטה Intraday.

    אם start_date / end_date לא נמסרים, data_loader כבר משתמש ב-INTRADAY_LOOKBACK_DAYS
    מתוך settings.
    """
    return load_price_data(symbol, start_date=start_date, end_date=end_date, intraday=True, **kwargs)


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
    עטיפה ל-load_prices_multi המלא מה-DataLoader החדש.

    מחזיר DataFrame רחב:
        index   = DatetimeIndex
        columns = symbols (SPY, QQQ, XLY, XLP, ...)
    """
    if not _HAS_CORE or _core_load_prices_multi is None:
        logger.warning("load_prices_multi called but data_loader core is not available.")
        return pd.DataFrame()

    return _core_load_prices_multi(
        symbols,
        start_date=start_date,
        end_date=end_date,
        intraday=intraday,
        col=col,
        freq=freq,
        normalize=normalize,
        fill_method=fill_method,
    )


def ensure_prices_cached(
    symbols: Sequence[str],
    *,
    intraday: bool = False,
    force: bool = False,
    days_back: Optional[int] = None,
) -> None:
    """
    מבטיח שלסימבולים הנתונים יש CSV טרי על הדיסק (cache):

        ensure_prices_cached(["SPY","QQQ","XLY","XLP"], intraday=False)

    מתבסס על bulk_download מ-data_loader.
    """
    if not _HAS_CORE or _core_bulk_download is None:
        logger.warning("ensure_prices_cached called but data_loader core is not available.")
        return

    _core_bulk_download(
        list(symbols),
        intraday=intraday,
        force=force,
        days_back=days_back,
    )


class DataLoader(_CoreDataLoader):  # type: ignore[misc]
    """
    עטיפה ל-DataLoader מתוך common.data_loader לצורכי תאימות.

    שימוש:
        from common.price_loader import DataLoader

        dl = DataLoader()
        df = dl.get_price("XLY", start_date="2020-01-01", end_date="2024-01-01")
        dl.bulk_daily(["XLY","XLP"])
    """

    # אין צורך להוסיף קוד – כל הלוגיקה כבר יושבת ב-CoreDataLoader.
    # המחלקה נשמרת כדי לא לשבור importים קיימים.
    pass


# ---------------------------------------------------------------------------
# Metadata & validation passthrough (אופציונלי)
# ---------------------------------------------------------------------------


def get_price_metadata(df: pd.DataFrame, *, symbol: str = "") -> Dict[str, Any]:
    """
    Proxy ל-get_price_metadata מ-data_loader:

        meta = get_price_metadata(df, symbol="SPY")
    """
    if not _HAS_CORE or _core_get_price_metadata is None:
        return {
            "symbol": symbol,
            "n_rows": int(df.shape[0]) if df is not None else 0,
            "n_cols": int(df.shape[1]) if df is not None else 0,
            "start_date": None,
            "end_date": None,
            "has_nans": bool(df.isna().any().any()) if df is not None else False,
            "freq": None,
            "intraday": None,
        }
    return _core_get_price_metadata(df, symbol=symbol)


def validate_price_df(df: pd.DataFrame, *, symbol: str = "") -> None:
    """
    Proxy ל-validate_price_df (בדיקות איכות: אינדקס, gaps, מחירים שליליים וכו').
    """
    if not _HAS_CORE or _core_validate_price_df is None:
        return
    _core_validate_price_df(df, symbol=symbol)


# ---------------------------------------------------------------------------
# __all__ — מה שאנחנו רוצים לחשוף למודולים אחרים
# ---------------------------------------------------------------------------

__all__ = [
    "load_price_data",
    "get_daily_price",
    "get_intraday_price",
    "load_prices_multi",
    "ensure_prices_cached",
    "DataLoader",
    "get_price_metadata",
    "validate_price_df",
]
