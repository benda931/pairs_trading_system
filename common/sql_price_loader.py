# -*- coding: utf-8 -*-
"""
common/sql_price_loader.py — HF-grade loaders from SqlStore
===========================================================

תפקיד המודול:
-------------
- לטעון היסטוריית מחירים מ-SqlStore (DuckDB / SQLite / Postgres) ברמת:
  • סימבול יחיד.
  • זוג סימבולים (Pair) כולל spread.
  • רשימת סימבולים עם יישור מלא על ציר הזמן.

- להחזיר DataFrame-ים:
  • מסודרים, מסונכרנים, עם אינדקס זמן נקי.
  • מוכנים ל-Backtest / Optimizer / Agents / Dashboard.

עקרונות:
---------
1. SqlStore הוא מקור האמת למחירים (לא Yahoo / APIs אחרים).
2. כל הלוגיקה של "איך טוענים מה-Sql" יושבת כאן ולא מפוזרת בקוד.
3. תמיכה ב:
   - סינון לפי תאריכים (start/end).
   - resample לתדירויות אחרות (W, M, 1h וכו').
   - חישוב תשואות (אחוזיות / לוגיות) בנוסף למחירים.
   - בדיקות sanity בסיסיות (מיון, אינדקס, duplicates).

שימוש טיפוסי:
-------------
    from pathlib import Path
    from common.sql_price_loader import (
        init_sql_store_from_config,
        load_symbol_history,
        load_pair_history,
        load_pair_with_spread,
        load_multi_symbol_matrix,
    )

    project_root = Path(__file__).resolve().parent.parent
    store = init_sql_store_from_config(project_root, env="dev")

    # סימבול אחד
    df_xlp = load_symbol_history("XLP", store=store, env="dev")

    # זוג
    df_pair = load_pair_with_spread("XLP", "XLY", store=store, env="dev")

    # מטריצה של כמה סימבולים
    df_mat = load_multi_symbol_matrix(["SPY", "QQQ", "IWM"], store=store)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Literal, Dict, Any

import logging
import pandas as pd

from common.config_manager import load_settings
from core.sql_store import SqlStore


# ========= Logger =========

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | sql_price_loader | %(message)s"
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ========= Dataclasses / Types =========

ReturnsMode = Literal["none", "pct", "log"]


@dataclass(frozen=True)
class PriceLoadConfig:
    """
    PriceLoadConfig — קונפיגורציה לטעינת מחירי סימבול/זוג.

    נועד לשימוש פנימי/חיצוני כאחד, כדי לשמור על הגדרות עקביות.
    """

    price_col: str = "close"          # עמודת מחיר שבה נשתמש (בד"כ "close" / "adj_close").
    freq: Optional[str] = None        # למשל "1D", "W", "M", "H" – אם None, אין resample.
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    returns: ReturnsMode = "none"     # "none" / "pct" / "log".
    drop_na: bool = True              # האם לזרוק שורות עם NaN אחרי היישור.


# ========= Init helpers =========

def init_sql_store_from_config(
    project_root: Optional[Path] = None,
    *,
    env: str = "dev",
    config_name: str = "config.json",
) -> SqlStore:
    """
    יוצר SqlStore מתוך קובץ קונפיגורציה (config.json).

    Parameters
    ----------
    project_root : Path | None
        תיקיית פרויקט הבסיס. אם None → נלקח יחסית למיקום המודול.
    env : str
        environment לוגי (dev / paper / live).
    config_name : str
        שם קובץ הקונפיג (ברירת מחדל: "config.json" בשורש הפרויקט).

    Returns
    -------
    SqlStore
        מופע מוכן לשימוש.

    הערות:
    ------
    - אנחנו לא מעבירים engine_url ידנית כי SqlStore.from_settings כבר יודע
      לבחור DuckDB ברירת מחדל לפי ההגדרות שלך.
    """
    if project_root is None:
        # מניח שהקובץ יושב ב-common/ → project_root = parent.parent
        project_root = Path(__file__).resolve().parent.parent

    config_path = project_root / config_name
    settings = load_settings(config_path)
    store = SqlStore.from_settings(settings, env=env)
    logger.info(
        "init_sql_store_from_config: initialized SqlStore (env=%s, config=%s)",
        env,
        config_path,
    )
    return store


# ========= Internal helpers =========

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    מוודא של-DataFrame יש DatetimeIndex תקין וממויין.

    - אם אין אינדקס מסוג DatetimeIndex אך קיימת עמודת 'date' → מעביר אותה לאינדקס.
    - מסיר duplicated index (שומר את השורה האחרונה).
    - ממיין לפי האינדקס.
    """
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        else:
            # ניסיון אחרון: לנסות לעשות to_datetime על האינדקס כפי שהוא
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                logger.warning(
                    "DataFrame has no DatetimeIndex and no 'date' column — returning as is."
                )
                return df

    # להסיר duplicates באינדקס (שומר את האחרון)
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    df = df.sort_index()
    return df


def _slice_by_date(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """
    גוזר את DataFrame לפי start/end אם ניתנו.
    """
    if df.empty:
        return df

    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]

    return df


def _resample_ohlcv(
    df: pd.DataFrame,
    freq: str,
    *,
    price_cols: Sequence[str] = ("open", "high", "low", "close"),
    volume_cols: Sequence[str] = ("volume",),
) -> pd.DataFrame:
    """
    Resample ל-OHLCV בצורה סטנדרטית:

    - open  → first
    - high  → max
    - low   → min
    - close → last
    - volume → sum

    שאר העמודות (אם יש) → last (כדי לא לזרוק אותן לגמרי).
    """
    if df.empty:
        return df

    agg: Dict[str, Any] = {}

    cols = list(df.columns)

    for c in cols:
        lc = c.lower()
        if lc in price_cols:
            if lc == "open":
                agg[c] = "first"
            elif lc == "high":
                agg[c] = "max"
            elif lc == "low":
                agg[c] = "min"
            elif lc == "close":
                agg[c] = "last"
        elif lc in volume_cols:
            agg[c] = "sum"
        else:
            agg[c] = "last"

    df_resampled = df.resample(freq).agg(agg)
    return df_resampled


def _apply_returns(
    df: pd.DataFrame,
    price_col: str,
    mode: ReturnsMode,
    *,
    new_col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    מוסיף/מחליף עמודת תשואות על בסיס עמודת מחיר נתונה.

    mode:
        - "none" → לא עושה כלום.
        - "pct"  → אחוזית: pct_change().
        - "log"  → לוגית: log(p_t / p_{t-1}).
    """
    if mode == "none" or df.empty:
        return df

    if price_col not in df.columns:
        logger.warning(
            "apply_returns: price_col %r not in columns %r — skipping returns.",
            price_col,
            list(df.columns),
        )
        return df

    if new_col_name is None:
        new_col_name = f"{price_col}_ret_{mode}"

    price = df[price_col].astype(float)

    if mode == "pct":
        ret = price.pct_change()
    elif mode == "log":
        # log-return: ln(p_t / p_{t-1})
        ret = (price / price.shift(1)).apply(lambda x: pd.NA if x <= 0 else float(pd.np.log(x)))  # type: ignore
    else:
        return df

    df[new_col_name] = ret
    return df


# ========= Symbol-level loader =========

def load_symbol_history(
    symbol: str,
    *,
    store: SqlStore,
    env: str = "dev",
    cols: Sequence[str] = ("open", "high", "low", "close", "volume"),
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: Optional[str] = None,
    returns: ReturnsMode = "none",
    drop_na: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """
    טוען היסטוריית מחירים לסימבול יחיד מ-SqlStore.

    Parameters
    ----------
    symbol : str
        טיקר, למשל "XLP".
    store : SqlStore
        אובייקט SqlStore (נוצר פעם אחת באפליקציה).
    env : str, default "dev"
        תג סביבת הריצה (dev / paper / live).
    cols : Sequence[str]
        אילו עמודות להביא (רק אם קיימות בטבלה).
    start, end : str | None
        טווח תאריכים (ISO, למשל "2018-01-01"). אם None → לא חותכים.
    freq : str | None
        אם לא None → עושים resample לפי תדירות זו (למשל "W", "M", "1H").
    returns : {"none", "pct", "log"}
        אפשרות לחישוב תשואות על price_col ("close" בד"כ) בתוך אותו DF.
    drop_na : bool
        האם לזרוק שורות עם NaN אחרי עיבוד (חתיכה + resample + returns).
    validate : bool
        אם True → מבצע sanity checks (אינדקס זמן, duplicates וכו').

    Returns
    -------
    pd.DataFrame
        DataFrame עם אינדקס של תאריכים ועמודות המחירים.
    """
    df = store.load_price_history(symbol, env=env)

    if df is None or df.empty:
        logger.warning("load_symbol_history(%s): no data in SqlStore (env=%s).", symbol, env)
        return pd.DataFrame()

    if validate:
        df = _ensure_datetime_index(df)

    # סינון לעמודות שנתבקשו (רק אלו שקיימות)
    cols_existing: List[str] = [c for c in cols if c in df.columns]
    if cols_existing:
        df = df[cols_existing].copy()

    # חיתוך לפי תאריכים (אם ניתנו)
    start_ts = pd.to_datetime(start) if start is not None else None
    end_ts = pd.to_datetime(end) if end is not None else None
    df = _slice_by_date(df, start_ts, end_ts)

    # resample אם ביקשנו freq
    if freq is not None:
        df = _resample_ohlcv(df, freq=freq)

    # חישוב תשואות אם ביקשנו
    if returns != "none" and "close" in df.columns:
        df = _apply_returns(df, price_col="close", mode=returns)

    if drop_na:
        df = df.dropna(how="any")

    return df


# ========= Pair-level loader =========

def load_pair_history(
    sym_x: str,
    sym_y: str,
    *,
    store: SqlStore,
    env: str = "dev",
    price_col: str = "close",
    drop_na: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: Optional[str] = None,
    returns: ReturnsMode = "none",
) -> pd.DataFrame:
    """
    טוען היסטוריית מחירים לשני סימבולים ומחזיר DataFrame מסונכרן.

    DataFrame החוזר:
        index: DatetimeIndex (תאריכים משותפים)
        columns:
            {sym_x}_{price_col}, {sym_y}_{price_col}
        ואם returns != "none":
            {sym_x}_{price_col}_ret_{mode}, {sym_y}_{price_col}_ret_{mode}

    Parameters
    ----------
    sym_x, sym_y : str
        שני הטיקרים של הזוג.
    store : SqlStore
        מופע SqlStore משותף לכל המערכת.
    env : str
        environment לוגי (dev / paper / live).
    price_col : str
        איזו עמודת מחיר להשתמש לצורך העבודה על הזוג.
    drop_na : bool
        האם לזרוק שורות עם NaN אחרי ה-join.
    start, end, freq, returns
        כמו ב-load_symbol_history.

    Returns
    -------
    pd.DataFrame
    """
    df_x = load_symbol_history(
        sym_x,
        store=store,
        env=env,
        cols=("open", "high", "low", price_col, "volume"),
        start=start,
        end=end,
        freq=freq,
        returns="none",  # תשואות, אם צריך, נוסיף אחרי ה-join
        drop_na=False,
        validate=True,
    )
    df_y = load_symbol_history(
        sym_y,
        store=store,
        env=env,
        cols=("open", "high", "low", price_col, "volume"),
        start=start,
        end=end,
        freq=freq,
        returns="none",
        drop_na=False,
        validate=True,
    )

    if df_x.empty or df_y.empty:
        logger.warning(
            "load_pair_history(%s,%s): one of the legs is empty (env=%s).",
            sym_x,
            sym_y,
            env,
        )
        return pd.DataFrame()

    df_x = df_x.sort_index()
    df_y = df_y.sort_index()

    col_x = f"{sym_x}_{price_col}"
    col_y = f"{sym_y}_{price_col}"

    out = (
        df_x[[price_col]]
        .rename(columns={price_col: col_x})
        .join(
            df_y[[price_col]].rename(columns={price_col: col_y}),
            how="inner",  # רק תאריכים משותפים לזוג
        )
    )

    if drop_na:
        out = out.dropna(how="any")

    # חישוב תשואות לכל רגל אם ביקשנו
    if returns != "none":
        out = _apply_returns(out, price_col=col_x, mode=returns)
        out = _apply_returns(out, price_col=col_y, mode=returns)

    return out


def load_pair_with_spread(
    sym_x: str,
    sym_y: str,
    *,
    store: SqlStore,
    env: str = "dev",
    price_col: str = "close",
    spread_name: Optional[str] = None,
    drop_na: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    הרחבה נוחה: מחזירה גם עמודת spread בסיסית.

    spread = {sym_x}_{price_col} - {sym_y}_{price_col}
    """
    df = load_pair_history(
        sym_x,
        sym_y,
        store=store,
        env=env,
        price_col=price_col,
        drop_na=drop_na,
        start=start,
        end=end,
        freq=freq,
        returns="none",
    )
    if df.empty:
        return df

    if spread_name is None:
        spread_name = f"spread_{sym_x}_{sym_y}"

    cols = list(df.columns)
    if len(cols) < 2:
        logger.warning(
            "load_pair_with_spread(%s,%s): expected at least 2 price columns, got %r",
            sym_x,
            sym_y,
            cols,
        )
        return df

    col_x, col_y = cols[0], cols[1]
    df[spread_name] = df[col_x].astype(float) - df[col_y].astype(float)
    return df


# ========= Multi-symbol loader (matrix) =========

def load_multi_symbol_matrix(
    symbols: Sequence[str],
    *,
    store: SqlStore,
    env: str = "dev",
    price_col: str = "close",
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: Optional[str] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    טוען מטריצה של כמה סימבולים ביחד, מיושרים על ציר הזמן.

    DataFrame החוזר:
        index: DatetimeIndex
        columns:
            {sym}_{price_col}  לכל סימבול ב-symbols.

    שימוש טיפוסי:
    --------------
        df = load_multi_symbol_matrix(
            ["SPY", "QQQ", "IWM"],
            store=store,
            env="dev",
            start="2018-01-01",
        )

    זה בסיס מעולה למטריצות קורלציה, PCA, וכו'.
    """
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []

    for sym in symbols:
        df_sym = load_symbol_history(
            sym,
            store=store,
            env=env,
            cols=("open", "high", "low", price_col, "volume"),
            start=start,
            end=end,
            freq=freq,
            returns="none",
            drop_na=False,
            validate=True,
        )
        if df_sym.empty:
            logger.warning("load_multi_symbol_matrix: no data for %s (env=%s).", sym, env)
            continue

        df_sym = df_sym.sort_index()
        col_name = f"{sym}_{price_col}"
        df_sym = df_sym[[price_col]].rename(columns={price_col: col_name})
        dfs.append(df_sym)

    if not dfs:
        return pd.DataFrame()

    # inner join על כל הסדרות
    out = dfs[0]
    for df in dfs[1:]:
        out = out.join(df, how="inner")

    if drop_na:
        out = out.dropna(how="any")

    return out
