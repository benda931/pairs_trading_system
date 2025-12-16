# -*- coding: utf-8 -*-
"""
core/data_ingest_bridge.py — Market-Data Ingestion Bridge (SqlStore ⇄ Providers)
===============================================================================

תפקיד המודול:
-------------
שכבת "גשר" קטנה בין SqlStore לבין ספקי דאטה (למשל IBKR, Yahoo fallback, וכו'),
עם מטרה אחת:

    ensure_prices_for_symbol / ensure_prices_for_pair

כלומר:
    * לבדוק איזה טווח תאריכים כבר קיים בטבלת prices (דרך SqlStore.get_price_range).
    * לחשב אילו תאריכים חסרים.
    * לקרוא לפונקציית fetch חיצונית (IBKR / YF) רק על הטווחים החסרים.
    * לשמור את המחירים ל-SqlStore דרך save_price_history.

המודול *לא* יודע להתחבר ל-IBKR בעצמו – הוא מקבל מבחוץ:
    - SqlStore (core.sql_store.SqlStore)
    - callable fetch_fn(symbol, start_date, end_date) -> pd.DataFrame

כך הוא נשאר דק, בדיק וניתן לשימוש גם ע"י סקריפטים/Agents.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import logging

import pandas as pd

from core.sql_store import SqlStore

logger = logging.getLogger(__name__)

DateLike = Union[date, datetime, str]
FetchFunc = Callable[[str, date, date], pd.DataFrame]


# ---------------------------------------------------------------------------
# פנימיים: המרות תאריכים וחישוב טווחים חסרים
# ---------------------------------------------------------------------------

def _to_date(d: DateLike) -> date:
    """
    ממיר קלט (date/datetime/str) ל-date.

    - str → pandas.to_datetime → .date()
    - datetime → date()
    - date → מחזיר כפי שהוא
    """
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    return pd.to_datetime(d).date()  # type: ignore[arg-type]


def _compute_missing_ranges_for_symbol(
    existing_min: Optional[pd.Timestamp],
    existing_max: Optional[pd.Timestamp],
    requested_start: date,
    requested_end: date,
) -> List[Tuple[date, date]]:
    """
    מחזיר רשימת טווחים [start, end] (כולל) שחסרים עבור הסימבול:

    לוגיקה:
    -------
    - אם אין בכלל דאטה קיים → מחזירים [(requested_start, requested_end)].
    - אחרת:
        * אם requested_start < existing_min.date():
              חסר טווח מ-requested_start ועד יום-לפני existing_min.
        * אם requested_end > existing_max.date():
              חסר טווח מיום-אחרי existing_max ועד requested_end.

    אין ניסיון "למזג" את הדאטה הקיים – רק להימנע ממשיכה כפולה ברורה.
    """
    missing: List[Tuple[date, date]] = []

    # אין דאטה קיים בכלל
    if existing_min is None or existing_max is None:
        if requested_start <= requested_end:
            missing.append((requested_start, requested_end))
        return missing

    have_start = existing_min.date()
    have_end = existing_max.date()

    # אם הטווח המבוקש הפוך – לא מושכים כלום
    if requested_start > requested_end:
        return []

    # gap שמאלי (לפני הדאטה הקיים)
    if requested_start < have_start:
        # נמשוך עד יום-לפני have_start כדי לא לייצר כפילות
        gap_end = min(have_start - timedelta(days=1), requested_end)
        if requested_start <= gap_end:
            missing.append((requested_start, gap_end))

    # gap ימני (אחרי הדאטה הקיים)
    if requested_end > have_end:
        # נתחיל יום-אחרי have_end כדי לא לייצר כפילות
        gap_start = max(have_end + timedelta(days=1), requested_start)
        if gap_start <= requested_end:
            missing.append((gap_start, requested_end))

    return missing


def _split_range_into_chunks(
    start: date,
    end: date,
    *,
    max_chunk_days: int = 365,
) -> List[Tuple[date, date]]:
    """
    מפצל טווח תאריכים [start, end] לרשימת chunks בגודל עד max_chunk_days.

    הסיבה:
    -------
    - פקודות IBKR/ספקים חיצוניים לא תמיד אוהבות טווחי שנים רבים בבת אחת.
    - עם chunking נוכל למשוך, למשל, 10 שנים כ-10× שנה.
    """
    if start > end:
        return []

    max_chunk_days = max(1, int(max_chunk_days))
    out: List[Tuple[date, date]] = []

    cur_start = start
    while cur_start <= end:
        cur_end = min(cur_start + timedelta(days=max_chunk_days - 1), end)
        out.append((cur_start, cur_end))
        cur_start = cur_end + timedelta(days=1)

    return out


# ---------------------------------------------------------------------------
# API ראשי: ensure_prices_for_symbol / ensure_prices_for_pair
# ---------------------------------------------------------------------------

def ensure_prices_for_symbol(
    store: SqlStore,
    symbol: str,
    *,
    start: DateLike,
    end: DateLike,
    env: Optional[str] = None,
    fetch_fn: FetchFunc,
    max_chunk_days: int = 365,
    log_prefix: str = "Ingest",
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    מוודא שב-SqlStore יש מחירי EOD עבור symbol בטווח [start, end].

    מה הוא עושה:
    -------------
    1. ממיר start/end ל-date ודואג ש-start <= end.
    2. שואל את SqlStore.get_price_range(symbol, env=env) כדי לדעת מה כבר קיים.
    3. מחשב אילו טווחים חסרים (למשל לפני הדאטה הקיים ואחרי).
    4. מפצל כל טווח חסר לצ'אנקים של עד max_chunk_days.
    5. עבור כל chunk:
         - קורא fetch_fn(symbol, chunk_start, chunk_end) → DataFrame מחירים.
         - אם לא ריק → קורא store.save_price_history(symbol, df, env=env).

    פרמטרים:
    ---------
    store : SqlStore
        שכבת ה-SQL (DuckDB/SQLite/Postgres) עם get_price_range + save_price_history.
    symbol : str
        סימבול (למשל "SPY", "XLY", "BTC-USD").
    start, end : DateLike
        טווח התאריכים המבוקש (date/datetime/str).
    env : Optional[str]
        סביבה לוגית ("dev"/"paper"/"live"), תישמר בעמודת env ב-prices.
        אם None → ייעשה שימוש ב-store.default_env.
    fetch_fn : Callable[[symbol, start_date, end_date], pd.DataFrame]
        פונקציה שמחזירה DataFrame מחירים עבור הטווח הנתון.
        חייבת להחזיר index DatetimeIndex או עמודת 'date' + עמודות open/high/low/close/volume.
    max_chunk_days : int
        מקסימום ימים לכל chunk למשיכה מספק הדאטה (דיפולט: 365).
    log_prefix : str
        prefix ללוגים (למשל "IBKR", "YF", "Ingest").

    החזרה:
    -------
    (min_date, max_date) כפי שנמדדו לפני / אחרי ההרצה (כפי שמוחזרים מ-get_price_range).
    """
    symbol = str(symbol).strip()
    if not symbol:
        logger.warning("ensure_prices_for_symbol: empty symbol, nothing to do.")
        return None, None

    start_d = _to_date(start)
    end_d = _to_date(end)
    if start_d > end_d:
        logger.warning(
            "ensure_prices_for_symbol(%s): start > end (%s > %s) — skipping.",
            symbol,
            start_d,
            end_d,
        )
        return None, None

    eff_env = (env or store.default_env or "dev").strip()

    # 1) מה כבר קיים בבסיס?
    have_min, have_max = store.get_price_range(symbol, table_name=None, env=eff_env)
    logger.info(
        "[%s] %s: existing price range env=%s → (%s, %s)",
        log_prefix,
        symbol,
        eff_env,
        have_min,
        have_max,
    )

    # 2) לחשב אילו טווחים חסרים
    missing_ranges = _compute_missing_ranges_for_symbol(
        existing_min=have_min,
        existing_max=have_max,
        requested_start=start_d,
        requested_end=end_d,
    )
    if not missing_ranges:
        logger.info(
            "[%s] %s: no missing ranges for [%s, %s] in env=%s — nothing to ingest.",
            log_prefix,
            symbol,
            start_d,
            end_d,
            eff_env,
        )
        return have_min, have_max

    logger.info(
        "[%s] %s: missing ranges for [%s, %s] in env=%s → %s",
        log_prefix,
        symbol,
        start_d,
        end_d,
        eff_env,
        missing_ranges,
    )

    # 3) למשוך ולהכניס כל טווח חסר
    for gap_start, gap_end in missing_ranges:
        for chunk_start, chunk_end in _split_range_into_chunks(
            gap_start,
            gap_end,
            max_chunk_days=max_chunk_days,
        ):
            logger.info(
                "[%s] %s: fetching prices for chunk [%s, %s] (env=%s)",
                log_prefix,
                symbol,
                chunk_start,
                chunk_end,
                eff_env,
            )
            try:
                df = fetch_fn(symbol, chunk_start, chunk_end)
            except Exception as exc:  # pragma: no cover — הגנה קשיחה
                logger.warning(
                    "[%s] %s: fetch_fn failed for [%s, %s]: %s",
                    log_prefix,
                    symbol,
                    chunk_start,
                    chunk_end,
                    exc,
                    exc_info=True,
                )
                continue

            if df is None or df.empty:
                logger.info(
                    "[%s] %s: no data returned for [%s, %s] — skipping save.",
                    log_prefix,
                    symbol,
                    chunk_start,
                    chunk_end,
                )
                continue

            try:
                store.save_price_history(
                    symbol=symbol,
                    df_prices=df,
                    env=eff_env,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] %s: save_price_history failed for [%s, %s]: %s",
                    log_prefix,
                    symbol,
                    chunk_start,
                    chunk_end,
                    exc,
                    exc_info=True,
                )

    # 4) להחזיר טווח מעודכן אחרי ingestion (אופציונלי אבל נוח)
    new_min, new_max = store.get_price_range(symbol, table_name=None, env=eff_env)
    logger.info(
        "[%s] %s: final price range env=%s → (%s, %s)",
        log_prefix,
        symbol,
        eff_env,
        new_min,
        new_max,
    )
    return new_min, new_max


def ensure_prices_for_pair(
    store: SqlStore,
    sym_x: str,
    sym_y: str,
    *,
    start: DateLike,
    end: DateLike,
    env: Optional[str] = None,
    fetch_fn: FetchFunc,
    max_chunk_days: int = 365,
    log_prefix: str = "Ingest",
) -> Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
    """
    מוודא שמחירי EOD קיימים עבור זוג סימבולים (sym_x, sym_y) בטווח [start, end].

    זה רק wrapper נוח סביב ensure_prices_for_symbol פעמיים.

    החזרה:
        {
          "sym_x": (min_x, max_x),
          "sym_y": (min_y, max_y),
        }
    """
    start_d = _to_date(start)
    end_d = _to_date(end)

    res_x = ensure_prices_for_symbol(
        store=store,
        symbol=sym_x,
        start=start_d,
        end=end_d,
        env=env,
        fetch_fn=fetch_fn,
        max_chunk_days=max_chunk_days,
        log_prefix=log_prefix,
    )
    res_y = ensure_prices_for_symbol(
        store=store,
        symbol=sym_y,
        start=start_d,
        end=end_d,
        env=env,
        fetch_fn=fetch_fn,
        max_chunk_days=max_chunk_days,
        log_prefix=log_prefix,
    )
    return {"sym_x": res_x, "sym_y": res_y}
