# -*- coding: utf-8 -*-
"""
common/fundamental_loader.py — Index/ETF Fundamentals Ingestion Layer (HF-grade)
===============================================================================

מטרת המודול
-----------
שכבת **ingestion + ניהול** אחת לכל הדאטה הפנדומנטלי ברמת מדד / ETF / Basket:

- מדדים כלליים: SPY, QQQ, IWM, DIA, VTI, VT...
- מדדי אזורים: EEM, EFA, EWJ, EWU, VNQ, וכו'.
- סקטורים: XLF, XLK, XLE, XLY, XLV, XLU, XLI, XLB...
- מדדי Style / Factor: Value, Growth, Quality, Small, Momentum וכו' (אם תוסיף).

המערכת נבנית כך שתתמוך *בכל מה שתכננו*:
- טווחי זמן ארוכים (עד 10 שנים ויותר) לכל המדדים.
- כיסוי רחב "באופן קיצוני" של פיצ'רים פנדומנטליים:
  P/E, Forward P/E, P/B, Dividend Yield, Earnings Yield,
  ROE, ROA, ROIC, Net Margin, Operating Margin,
  Revenue/EPS/FCF Growth (3Y/5Y),
  Net Debt / EBITDA, Interest Coverage, Payout Ratio,
  ועוד תתי-פיצ'רים שנרצה להוסיף.

- תמיכה במקורות דאטה שונים (Providers):
  * Local files (CSV/Parquet) – for backfills / offline research.
  * Tiingo / Financial APIs – כשנרצה להזין נתונים אמיתיים.
  * IBKR fundamentals – התאמה למערכת המסחר החיה.
  * Provider מותאם אישית (למשל DuckDB פנימי).

- ניהול Cache לוקלי מלא:
  * שמירה של כל symbol בקבצי Parquet/CSV + קובץ meta.json.
  * TTL (time to live) לפי ימים.
  * אפשרות ל-force refresh, partial refresh (incremental), ולוגים.

- API פנימי נקי לשאר המערכת (core/index_fundamentals, core/index_clustering וכו'):
  * load_index_fundamentals(...)
  * load_indices_fundamentals(...)
  * build_fundamentals_panel(...)
  * later: incremental_update_index_fundamentals(...)

מבנה הקובץ (4 חלקים)
---------------------
חלק 1/4 (כאן):
    - תיעוד כללי.
    - imports, logger.
    - FundamentalsSettings (Pydantic Settings).
    - טיפוסים בסיסיים (FundamentalFrame / Panel).
    - Provider registry בסיסי (without actual API calls).
    - עזרי path ו-naming לסימולים.

חלק 2/4:
    - I/O לוקלי: קריאה/כתיבה Parquet/CSV.
    - מטא-דאטה (meta.json) ו-JSON safe.
    - Normalize DataFrame (אינדקס זמן, שמות עמודות).
    - תבנית Provider abstraction (BaseFundamentalsProvider + registry).

חלק 3/4:
    - פונקציות public:
        * load_index_fundamentals
        * load_index_fundamentals_cached
        * load_indices_fundamentals
        * build_fundamentals_panel
    - תמיכה ב-fields, טווחי זמן, provider override.
    - לוגים ו-handling חכם של cache vs remote.

חלק 4/4:
    - הרחבות מתקדמות:
        * incremental_refresh (append-only מה-provider).
        * diagnostics / health-check (מי symbol חסר / מאחר).
        * utilities ל-core (למשל cast לפורמט wide/long).
        * hooks לשילוב עם DuckDB / local DB (אופציונלי).

שימו לב:
--------
בחלק 1/4 **אין** עדיין קריאת API אמיתית – זה יגיע בחלקים הבאים כ־Base Provider + מימושים.
כבר עכשיו המודול בנוי כך שקל להרחיבו מבלי לשבור את שאר המערכת.
"""

from __future__ import annotations

# =========================
# Imports
# =========================

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd
from pydantic import Field
try:  # Pydantic v2 style (מועדף)
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover - fallback לסביבות בלי pydantic_settings
    from pydantic import BaseSettings  # type: ignore

from common.json_safe import make_json_safe, json_default as _json_default
from common.helpers import get_logger
# אם יש לך common/typing_compat עם StrPath/Sequence תוכל לייבא בעתיד:
# from common.typing_compat import StrPath  # לא חובה בשלב זה


# =========================
# Logger
# =========================

logger = get_logger("common.fundamental_loader")


# =========================
# Settings
# =========================


class FundamentalsSettings(BaseSettings):
    """
    הגדרות לטעינה/שמירה/רענון של דאטה פנדומנטלי למדדים/ETF.

    ניתן לשלוט דרך משתני סביבה / .env, לדוגמה:
    ------------------------------------------------
    FUNDAMENTALS_DATA_DIR=./data/fundamentals
    FUNDAMENTALS_CACHE_TTL_DAYS=7
    FUNDAMENTALS_DEFAULT_PROVIDER=tiingo
    FUNDAMENTALS_ALWAYS_REFRESH=false
    FUNDAMENTALS_ALLOW_PARTIAL=true
    FUNDAMENTALS_PANEL_FREQ=D
    """

    # 📂 תיקיית בסיס לדאטה פנדומנטלי
    FUNDAMENTALS_DATA_DIR: Path = Field(
        default=Path("data") / "fundamentals",
        env="FUNDAMENTALS_DATA_DIR",
    )

    # 🕒 TTL – כמה זמן (בימים) קובץ נחשב "טרי"
    FUNDAMENTALS_CACHE_TTL_DAYS: int = Field(
        default=7,
        env="FUNDAMENTALS_CACHE_TTL_DAYS",
    )

    # 🔌 ספק ברירת מחדל (logical name) – יוגדר ב-registry:
    # לדוגמה: "tiingo", "ibkr", "local_duckdb", "csv_only"
    FUNDAMENTALS_DEFAULT_PROVIDER: str = Field(
        default="tiingo",
        env="FUNDAMENTALS_DEFAULT_PROVIDER",
    )

    # 🔁 האם תמיד לנסות רענון מהספק, גם אם הקובץ טרי
    FUNDAMENTALS_ALWAYS_REFRESH: bool = Field(
        default=False,
        env="FUNDAMENTALS_ALWAYS_REFRESH",
    )

    # ⚠️ האם מותר partial data (למשל חסרים חלק מהשדות) או שנדרוש full schema
    FUNDAMENTALS_ALLOW_PARTIAL: bool = Field(
        default=True,
        env="FUNDAMENTALS_ALLOW_PARTIAL",
    )

    # 🧱 תדירות ברירת מחדל לפאנל (resample אם צריך): "D", "W", "M", "Q"
    FUNDAMENTALS_PANEL_FREQ: str = Field(
        default="M",
        env="FUNDAMENTALS_PANEL_FREQ",
    )

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }


SETTINGS = FundamentalsSettings()


# =========================
# Typing aliases
# =========================

FundamentalFrame = pd.DataFrame  # DataFrame עבור symbol בודד
FundamentalPanel = pd.DataFrame  # MultiIndex (date, symbol) עבור universe


@dataclass
class FundamentalMeta:
    """
    Metadata בסיסי עבור קובץ פנדומנטל אחד (per symbol).

    נשמר כ-JSON לצד קובץ הדאטה, למשל:
        data/fundamentals/SPY.meta.json
    """

    symbol: str
    source: str
    last_refresh: datetime
    n_rows: int
    n_cols: int
    fields: Sequence[str]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FundamentalMeta":
        last_refresh_raw = data.get("last_refresh")
        if isinstance(last_refresh_raw, str):
            try:
                last_refresh_val = datetime.fromisoformat(last_refresh_raw)
            except ValueError:
                last_refresh_val = datetime.min
        elif isinstance(last_refresh_raw, datetime):
            last_refresh_val = last_refresh_raw
        else:
            last_refresh_val = datetime.min

        return cls(
            symbol=str(data.get("symbol") or ""),
            source=str(data.get("source") or ""),
            last_refresh=last_refresh_val,
            n_rows=int(data.get("n_rows") or 0),
            n_cols=int(data.get("n_cols") or 0),
            fields=list(data.get("fields") or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "last_refresh": self.last_refresh.isoformat(),
            "n_rows": int(self.n_rows),
            "n_cols": int(self.n_cols),
            "fields": list(self.fields),
        }


# =========================
# Provider registry (Skeleton)
# =========================

class BaseFundamentalsProvider:
    """
    בסיס מופשט לכל ספקי הדאטה הפנדומנטליים.

    הרעיון:
    -------
    - כל provider (Tiingo/IBKR/CSV/DuckDB) יירש מהמחלקה הזאת.
    - המודול fundamental_loader ידבר תמיד עם API אחיד:
        .fetch(symbol, start, end, fields)

    החוזה:
    -------
    - מחזירים DataFrame עם:
        index: DatetimeIndex (תאריך דו"ח / period end)
        columns: שמות פנדומנטליים (pe, pb, roe, dividend_yield וכו').
    - normalization של שמות עמודות ואינדקס ייעשה במודול זה (לא ב-provider).
    """

    name: str = "base"

    def fetch(
        self,
        symbol: str,
        *,
        start: date | None = None,
        end: date | None = None,
        fields: Sequence[str] | None = None,
    ) -> FundamentalFrame:
        raise NotImplementedError("BaseFundamentalsProvider.fetch must be overridden")


# registry: שם לוגי -> instance של provider
_FUNDAMENTALS_PROVIDERS: Dict[str, BaseFundamentalsProvider] = {}


def register_fundamentals_provider(provider: BaseFundamentalsProvider) -> None:
    """
    רישום ספק חדש ב-registry.

    דוגמאות:
    ---------
        from common.data_providers import TiingoFundamentalsProvider

        register_fundamentals_provider(TiingoFundamentalsProvider())

    ואז:
        load_index_fundamentals(..., provider="tiingo")
    """
    name = getattr(provider, "name", None) or provider.__class__.__name__.lower()
    name = str(name).lower().strip()
    if not name:
        raise ValueError("Provider must have a non-empty name")

    if name in _FUNDAMENTALS_PROVIDERS:
        logger.warning(
            "Overriding existing fundamentals provider %r with %r",
            name,
            provider,
        )
    _FUNDAMENTALS_PROVIDERS[name] = provider
    logger.info("Registered fundamentals provider: %s -> %r", name, provider)


def get_fundamentals_provider(name: str | None = None) -> BaseFundamentalsProvider:
    """
    מקבל את ספק הדאטה לפי שם לוגי.

    אם name=None – משתמש ב-SETTINGS.FUNDAMENTALS_DEFAULT_PROVIDER.
    אם לא נמצא – זורק KeyError עם הודעה ברורה.
    """
    if name is None or not str(name).strip():
        name = SETTINGS.FUNDAMENTALS_DEFAULT_PROVIDER
    key = str(name).lower().strip()

    if key not in _FUNDAMENTALS_PROVIDERS:
        raise KeyError(
            f"No fundamentals provider registered under name {key!r}. "
            "Make sure to call register_fundamentals_provider(...) at startup."
        )
    return _FUNDAMENTALS_PROVIDERS[key]


# =========================
# Path helpers (symbols → files)
# =========================

def _normalize_symbol(symbol: str) -> str:
    """
    מנרמל סימול לשימוש בבסיס קבצים:
    - מחליף '/' ו-' ' ב-'_'
    - מחזיר באותיות גדולות (SPY, QQQ...)
    """
    return str(symbol).strip().replace("/", "_").replace(" ", "_").upper()


def _symbol_base_path(symbol: str) -> Path:
    """
    בסיס path עבור symbol – ללא סיומת:
        data/fundamentals/SPY
    """
    sym = _normalize_symbol(symbol)
    base_dir = SETTINGS.FUNDAMENTALS_DATA_DIR
    return base_dir / sym


def _parquet_path(symbol: str) -> Path:
    """
    קובץ Parquet עבור symbol:
        data/fundamentals/SPY.parquet
    """
    return _symbol_base_path(symbol).with_suffix(".parquet")


def _csv_path(symbol: str) -> Path:
    """
    קובץ CSV עבור symbol:
        data/fundamentals/SPY.csv
    """
    return _symbol_base_path(symbol).with_suffix(".csv")


def _meta_path(symbol: str) -> Path:
    """
    קובץ meta.json עבור symbol:
        data/fundamentals/SPY.meta.json
    """
    return _symbol_base_path(symbol).with_suffix(".meta.json")


def _ensure_data_dir() -> None:
    """
    מבטיח שהתיקייה FUNDAMENTALS_DATA_DIR קיימת.
    """
    SETTINGS.FUNDAMENTALS_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# חלק 1/4 נגמר כאן.
# בחלק 2/4 נוסיף:
#   - קריאת/כתיבת קבצים (parquet/csv)
#   - קריאה/כתיבת meta.json
#   - normalization ל-DataFrame (index = date, columns = snake_case)
#   - מעטפת בסיסית לשימוש ב-BaseFundamentalsProvider
# ============================================================

# ============================================================
# Part 2/4 — Local I/O, Metadata, Normalization & Provider glue
# ============================================================
"""
בחלק הזה אנחנו מוסיפים את היכולות הבאות:

1. 📂 עבודה עם קבצים לוקליים (Parquet / CSV):
   - _load_from_disk(symbol, require_fresh=True)
   - _save_to_disk(symbol, df, source, extra_meta)

2. 🧾 Metadata:
   - _is_fresh(path) – בדיקת TTL לפי SETTINGS.FUNDAMENTALS_CACHE_TTL_DAYS
   - _load_metadata(symbol) -> FundamentalMeta | None
   - _save_metadata(symbol, FundamentalMeta | dict)

3. 🧼 נירמול DataFrame לפורמט אחיד:
   - _normalize_fundamentals_df(df)
     * אינדקס זמן (DatetimeIndex) על בסיס עמודות date/period_end/report_date וכו'
     * שמות עמודות lower_snake_case
     * מיון לפי תאריך, הסרת כפילויות, הסרת כל-NaN בקצוות

4. 🔌 מעטפת בסיסית לעבודה עם Provider:
   - _fetch_via_provider(symbol, start, end, fields, provider_name)
   - לא נוגעים עדיין ב-load_index_fundamentals (זה יהיה בחלק 3).

שימו לב:
--------
- אין כאן כניסה ל-API חיצוני – זה נעשה דרך BaseFundamentalsProvider.fetch.
- הקוד תואם גם לסביבות בהן קיימים רק CSV/Parquet (ללא אינטרנט).
"""

import json


# =========================
# Freshness / TTL helpers
# =========================

def _is_fresh(path: Path) -> bool:
    """
    בדיקה אם קובץ נחשב "טרי" לפי הגדרת ה-TTL ב-SETTINGS.

    לוגיקה:
    -------
    - אם FUNDAMENTALS_CACHE_TTL_DAYS <= 0 → כל קובץ קיים נחשב טרי.
    - אם הקובץ לא קיים → False.
    - אחרת, משווים את זמן המודיפיקציה של הקובץ לעומת עכשיו.
    """
    ttl_days = SETTINGS.FUNDAMENTALS_CACHE_TTL_DAYS
    if not path.exists():
        return False
    if ttl_days <= 0:
        return True

    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:  # pragma: no cover - edge filesystem
        return False

    age = datetime.now() - mtime
    return age <= timedelta(days=ttl_days)


# =========================
# Metadata helpers
# =========================

def _load_metadata(symbol: str) -> FundamentalMeta | None:
    """
    טוען FundamentalMeta מסביבת העבודה, אם קיים.

    - אם אין meta.json → מחזיר None.
    - אם יש בעיה בקריאה → כותב אזהרה ומחזיר None.
    """
    meta_p = _meta_path(symbol)
    if not meta_p.exists():
        return None

    try:
        with meta_p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        meta = FundamentalMeta.from_dict(raw)
        return meta
    except Exception as exc:  # pragma: no cover - לוג שקט בייצור
        logger.warning(
            "Failed to read fundamentals meta for %s from %s: %s",
            symbol,
            meta_p,
            exc,
        )
        return None


def _save_metadata(
    symbol: str,
    meta: FundamentalMeta | Mapping[str, Any],
) -> None:
    """
    שומר FundamentalMeta (או dict דומה) לקובץ meta.json.

    שימוש פנימי מתוך _save_to_disk.
    """
    if isinstance(meta, FundamentalMeta):
        data = meta.to_dict()
    else:
        data = dict(meta)

    safe = make_json_safe(data)
    meta_p = _meta_path(symbol)

    try:
        with meta_p.open("w", encoding="utf-8") as f:
            json.dump(
                safe,
                f,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        logger.debug("Saved fundamentals meta for %s to %s", symbol, meta_p)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Failed to write fundamentals meta for %s to %s: %s",
            symbol,
            meta_p,
            exc,
        )


# =========================
# DataFrame normalization
# =========================

_DATE_COLUMN_CANDIDATES = (
    "date",
    "datetime",
    "period_end",
    "report_date",
    "as_of_date",
    "statement_date",
)


def _guess_date_column(df: pd.DataFrame) -> str | None:
    """
    מזהה עמודת תאריך סבירה מתוך DataFrame לפי רשימת שמות טיפוסיים.

    אם לא מוצא – מחזיר None.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in _DATE_COLUMN_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def _normalize_column_name(name: Any) -> str:
    """
    מנרמל שם עמודה לפורמט lower_snake_case פשוט.
    דוגמאות:
        'P/E' -> 'p_e'
        'Price to Book' -> 'price_to_book'
        'ROE (%)' -> 'roe'
    """
    s = str(name).strip()

    # הסרה של סימולים נפוצים
    remove_chars = ["%", "(", ")", "[", "]"]
    for ch in remove_chars:
        s = s.replace(ch, "")

    # הפרדה בסיסית
    s = (
        s.replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )

    s = s.lower()
    return s


def _normalize_fundamentals_df(df: pd.DataFrame) -> FundamentalFrame:
    """
    מנרמל DataFrame שמגיע מספק/קובץ לפורמט אחיד:

    1. מבטיח index מבוסס זמן (DatetimeIndex).
       - אם אין אינדקס כזה, מחפש עמודת תאריך טיפוסית (date/period_end/...).
       - אם מוצא – הופך אותה ל-DatetimeIndex.
       - אם לא – משאיר אינדקס כפי שהוא אבל מתריע בלוג.

    2. מנרמל שמות עמודות ל-lower_snake_case.

    3. ממיין לפי זמן, מסיר כפילויות על index, ומסיר שורות ריקות לגמרי.

    שים לב:
    -------
    - לא מבצע resample לתדירות קבועה (זה יהיה תפקיד build_fundamentals_panel).
    - לא כופה טיפוסים מספריים – אבל ניתן להוסיף בהמשך coercion לפי צורך.
    """
    if df is None or df.empty:
        # מחזיר DataFrame ריק אבל שומר על טיפוס
        return pd.DataFrame()

    df = df.copy()

    # --- טיפול באינדקס זמן ---
    if not isinstance(df.index, pd.DatetimeIndex):
        # ננסה לזהות עמודת תאריך
        date_col = _guess_date_column(df)
        if date_col is not None:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to convert column %r to datetime index in fundamentals DF: %s",
                    date_col,
                    exc,
                )
        else:
            # אין עמודת תאריך – נתריע, אבל לא נזרוק שגיאה
            logger.warning(
                "Fundamentals DataFrame has no obvious date column; "
                "consider including 'date' or 'period_end'."
            )

    # אם יש אינדקס זמן – נמיין
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        # הסרת כפילויות בתאריך – נשמור את האחרונה (או אפשר לבחור אחרת)
        df = df[~df.index.duplicated(keep="last")]

    # --- נירמול שמות עמודות ---
    df.columns = [_normalize_column_name(c) for c in df.columns]

    # --- הסרת שורות ריקות לחלוטין ---
    # (כל הערכים NaN או None)
    df = df.dropna(how="all")

    return df


# =========================
# Local I/O (Parquet / CSV)
# =========================

def _load_from_parquet(path: Path) -> pd.DataFrame | None:
    """
    ניסיון לטעון DataFrame מקובץ Parquet.

    אם יש כשל – מחזיר None ולא זורק החוצה (משאיר ללוג לטפל).
    """
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        logger.debug("Loaded fundamentals parquet from %s", path)
        return df
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read parquet fundamentals %s: %s", path, exc)
        return None


def _load_from_csv(path: Path) -> pd.DataFrame | None:
    """
    ניסיון לטעון DataFrame מקובץ CSV.

    אם יש כשל – מחזיר None ולא זורק החוצה (משאיר ללוג לטפל).
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        logger.debug("Loaded fundamentals csv from %s", path)
        return df
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read csv fundamentals %s: %s", path, exc)
        return None


def _load_from_disk(
    symbol: str,
    *,
    require_fresh: bool = True,
) -> FundamentalFrame | None:
    """
    טוען פנדומנטל ל-symbol מהדיסק (Parquet/CSV), אם קיים.

    פרמטרים
    --------
    symbol : str
        סימול המדד/ETF (SPY, QQQ וכו').
    require_fresh : bool
        אם True – נבדוק TTL עם _is_fresh, ונחזיר None אם הקובץ ישן מדי.
        אם False – נטען גם קובץ ישן.

    החזר
    -----
    DataFrame או None:
        - DataFrame מנורמל באמצעות _normalize_fundamentals_df אם נמצא קובץ.
        - None אם אין קובץ / לא טרי / כשל קריאה.
    """
    _ensure_data_dir()
    sym_norm = _normalize_symbol(symbol)
    pq = _parquet_path(sym_norm)
    csv = _csv_path(sym_norm)

    if require_fresh:
        # אם לא טרי – לא ננסה בכלל לטעון (נעדיף provider)
        if not (_is_fresh(pq) or _is_fresh(csv)):
            logger.debug(
                "No fresh fundamentals cache for %s (ttl=%sd) – skipping local load",
                sym_norm,
                SETTINGS.FUNDAMENTALS_CACHE_TTL_DAYS,
            )
            return None

    df_raw: pd.DataFrame | None = None

    # עדיף Parquet על CSV
    df_raw = _load_from_parquet(pq)
    if df_raw is None:
        df_raw = _load_from_csv(csv)

    if df_raw is None:
        return None

    df_norm = _normalize_fundamentals_df(df_raw)
    return df_norm


def _save_to_disk(
    symbol: str,
    df: FundamentalFrame,
    *,
    source: str,
    extra_meta: Mapping[str, Any] | None = None,
) -> None:
    """
    שומר DataFrame פנדומנטל לקבצי Parquet/CSV + קובץ מטא-דאטה.

    פרמטרים
    --------
    symbol : str
        סימול המדד/ETF (SPY, QQQ וכו').
    df : DataFrame
        דאטה מנורמל (עדיף שכבר עבר _normalize_fundamentals_df).
    source : str
        תיאור מקור הדאטה (למשל "tiingo", "ibkr", "local_file").
    extra_meta : Mapping[str, Any] | None
        שדות מטא נוספים לשמירה (למשל כמות nulls, גרסת schema וכו').
    """
    _ensure_data_dir()
    sym_norm = _normalize_symbol(symbol)
    pq = _parquet_path(sym_norm)
    csv = _csv_path(sym_norm)

    # קודם נשמור את הדאטה עצמו
    try:
        df.to_parquet(pq)
        logger.info("Saved fundamentals for %s to %s", sym_norm, pq)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to write parquet fundamentals %s: %s", pq, exc)

    try:
        df.to_csv(csv, index=True)
        logger.debug("Also saved fundamentals for %s to %s", sym_norm, csv)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to write csv fundamentals %s: %s", csv, exc)

    # כעת נבנה ונשמור meta
    meta = FundamentalMeta(
        symbol=sym_norm,
        source=source,
        last_refresh=datetime.now(),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        fields=list(df.columns),
    )
    if extra_meta:
        # נכלול מידע נוסף בתוך dict ונשאיר ל-FundamentalMeta רק את הבסיס
        # ניתן להרחיב בעתיד לסכימה עשירה יותר
        merged = {**meta.to_dict(), **dict(extra_meta)}
        _save_metadata(sym_norm, merged)
    else:
        _save_metadata(sym_norm, meta)


# =========================
# Provider glue
# =========================

def _fetch_via_provider(
    symbol: str,
    *,
    provider_name: str | None = None,
    start: date | None = None,
    end: date | None = None,
    fields: Sequence[str] | None = None,
) -> tuple[FundamentalFrame, str]:
    """
    Fetch fundamentals עבור symbol דרך Provider הרשום ב-registry.

    החוזה:
    -------
    - מחזירים tuple:
        (df_norm, provider_name_effective)

    - df_norm:
        DataFrame **מנורמל** (_normalize_fundamentals_df) עם:
        index: DatetimeIndex (אם אפשר)
        columns: lower_snake_case.

    - provider_name_effective:
        השם שהשתמשנו בו בפועל (למשל "tiingo", גם אם provider_name=None).

    התנהגות:
    --------
    - אם אין ספק ברישום → KeyError.
    - אם הספק זורק שגיאה → נעביר אותה (שימושי כדי לדעת מה קרה).
    """
    if provider_name is None or not str(provider_name).strip():
        provider_name = SETTINGS.FUNDAMENTALS_DEFAULT_PROVIDER
    provider_name = str(provider_name).lower().strip()

    provider = get_fundamentals_provider(provider_name)

    logger.info(
        "Fetching fundamentals for %s via provider %s (start=%s, end=%s, fields=%s)",
        symbol,
        provider_name,
        start,
        end,
        fields,
    )

    df_raw = provider.fetch(
        symbol,
        start=start,
        end=end,
        fields=fields,
    )

    df_norm = _normalize_fundamentals_df(df_raw)
    return df_norm, provider_name


# =========================
# Diagnostics helpers (optional)
# =========================

def list_cached_symbols() -> list[str]:
    """
    מחזיר רשימת סימולים שיש להם קבצי fundamentals בתיקיית הדאטה.

    שימושי ל-debug / לוחות בקרה:
    - לראות במה המערכת כבר מכוסה.
    - לעקוב אחרי כמות הסימולים המנוטרים.
    """
    base = SETTINGS.FUNDAMENTALS_DATA_DIR
    if not base.exists():
        return []

    symbols: set[str] = set()
    for p in base.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".parquet", ".csv"):
            continue
        stem = p.stem  # למשל "SPY"
        symbols.add(stem.upper())

    return sorted(symbols)


def get_cached_meta(symbol: str) -> FundamentalMeta | None:
    """
    Wrapper ידידותי ל-_load_metadata: מחזיר FundamentalMeta אם יש.

    זה יכול לשמש:
    - להצגת "Last refresh" ב-UI.
    - ניטור סמוי (לוגים/health check).
    """
    sym_norm = _normalize_symbol(symbol)
    return _load_metadata(sym_norm)


# סוף חלק 2/4.
# בחלק 3/4 נוסיף את ה-API הציבורי:
#   load_index_fundamentals / load_index_fundamentals_cached /
#   load_indices_fundamentals / build_fundamentals_panel
# המשתמשים ביכולות שהוגדרו כאן.
# ============================================================
# Part 3/4 — Public API: load / cache / panel construction
# ============================================================
"""
בחלק הזה אנחנו בונים את ה-API הציבורי של שכבת הפנדומנטל למדדים:

1. 🔍 load_index_fundamentals(...)
   - טוען DataFrame פנדומנטל למדד/ETF בודד.
   - משלב Cache לוקלי + ספק חיצוני (provider).
   - שולט ב-TTL, force_refresh, allow_remote, allow_partial.

2. 🧠 load_index_fundamentals_cached(...)
   - עטיפה עם LRU cache לשימוש בתוך ריצה אחת (session).
   - מיועדת להימנע מטעינות חוזרות לאותו סימול/טווח/fields.

3. 📚 load_indices_fundamentals(...)
   - טעינה ל-universe שלם: {symbol -> DataFrame}.

4. 🧱 build_fundamentals_panel(...)
   - בונה Panel בפורמט MultiIndex (date, symbol) לכל ה-universe.
   - אופציונלית: resample לתדירות אחידה (D/W/M/Q).
   - תומך ב-forward-fill (למשל להפוך רבעון/שנה לנתון חודשי).

הכל נבנה כך שיתאים למה שתכננו:
- 10 שנים של דאטה.
- כיסוי רחב של פיצ'רים.
- תמיכה בהרחבה ל-Regimes / Clustering / Signals ברמת core.
"""


# =========================
# Internal helpers (fields & ranges)
# =========================

def _select_fields(
    df: FundamentalFrame,
    fields: Sequence[str] | None,
    *,
    allow_partial: bool,
    symbol: str,
) -> FundamentalFrame:
    """
    בחירת עמודות לפי רשימת fields, עם תמיכה ב-allow_partial.

    - fields=None → מחזיר את df כמו שהוא.
    - אחרת:
        * מנרמל את השמות לבסיס lower-case להשוואה.
        * אם allow_partial=True → מחזיר רק מה שקיים, מתריע על חסרים.
        * אם allow_partial=False → אם חסרות עמודות, מעלה ValueError.
    """
    if df.empty:
        return df

    if not fields:
        return df

    # normalization ל-lower עבור mapping
    cols_lower = {c.lower(): c for c in df.columns}
    requested_norm = [str(f).lower().strip() for f in fields]

    found_cols: list[str] = []
    missing: list[str] = []
    for f in requested_norm:
        if f in cols_lower:
            found_cols.append(cols_lower[f])
        else:
            missing.append(f)

    if missing:
        msg = (
            f"Fundamental fields {missing!r} not found for symbol {symbol!r}. "
            f"Available fields: {list(df.columns)!r}"
        )
        if allow_partial:
            logger.warning(msg + " (allow_partial=True → continuing with subset)")
        else:
            raise ValueError(msg + " (allow_partial=False → aborting)")

    if found_cols:
        return df[found_cols].copy()
    else:
        # אין אפילו עמודה אחת מהמבוקשות
        if allow_partial:
            # נחזיר DataFrame ריק עם אותם אינדקסים (אולי עדיין נרצה רק Timeline)
            logger.warning(
                "No requested fundamental fields were found for symbol %r; "
                "returning empty DataFrame with same index.",
                symbol,
            )
            return pd.DataFrame(index=df.index)
        else:
            raise ValueError(
                f"No requested fundamental fields found for {symbol!r} and "
                "allow_partial=False."
            )


def _apply_date_range(
    df: FundamentalFrame,
    start: date | None,
    end: date | None,
) -> FundamentalFrame:
    """
    חותך DataFrame לפי טווח תאריכים (אם יש אינדקס זמן).
    """
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    df_out = df
    if start is not None:
        df_out = df_out[df_out.index >= pd.to_datetime(start)]
    if end is not None:
        df_out = df_out[df_out.index <= pd.to_datetime(end)]
    return df_out


def _coerce_allow_partial(allow_partial: bool | None) -> bool:
    """
    אם לא סופק allow_partial, נשתמש בברירת מחדל מה-SETTINGS.
    """
    if allow_partial is None:
        return SETTINGS.FUNDAMENTALS_ALLOW_PARTIAL
    return bool(allow_partial)


def _coerce_require_fresh_local(require_fresh_local: bool | None) -> bool:
    """
    ברירת מחדל: if require_fresh_local=None → True
    (כלומר, באופן דיפולטי לא נסתמך על קובץ ישן אם TTL פג).
    """
    if require_fresh_local is None:
        return True
    return bool(require_fresh_local)


def _coerce_force_refresh(force_refresh: bool | None) -> bool:
    """
    ברירת מחדל: לוקח מה-SETTINGS.FUNDAMENTALS_ALWAYS_REFRESH.
    """
    if force_refresh is None:
        return SETTINGS.FUNDAMENTALS_ALWAYS_REFRESH
    return bool(force_refresh)


# =========================
# Public API — single symbol
# =========================

def load_index_fundamentals(
    symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
    fields: Sequence[str] | None = None,
    provider: str | None = None,
    allow_remote: bool = True,
    force_refresh: bool | None = None,
    require_fresh_local: bool | None = None,
    allow_partial: bool | None = None,
) -> FundamentalFrame:
    """
    טוען את כל היסטוריית הדאטה הפנדומנטלי למדד/ETF בודד, תוך ניהול cache ברמת
    קרן גידור:

    לוגיקה high-level:
    -------------------
    1. ניסיון טעינה מהדיסק:
       - אם require_fresh_local=True → נבדוק TTL (טריות) עם _is_fresh.
       - אם force_refresh=True → נדלג על הקובץ ונלך ישר ל-provider.

    2. אם יש df לוקלי:
       - ננרמל (אם לא כבר).
       - נחתוך לפי טווח תאריכים (start/end).
       - נבחר עמודות (fields) לפי allow_partial.
       - אם זה מספיק לנו → נחזיר.

       "מספיק לנו" = או שאין fields ספציפיים,
                      או שכל ה-fields קיימים,
                      או allow_partial=True (מותר subset).

    3. אם אין df לוקלי מתאים / force_refresh=True / חסרים שדות קריטיים:
       - אם allow_remote=False → נזרוק FileNotFoundError/ValueError רלוונטי.
       - אחרת:
           * נשתמש ב-provider דרך _fetch_via_provider.
           * ננרמל, נחתוך טווח תאריכים, נבחר fields.
           * נשמור לדיסק (Parquet/CSV + meta).
           * נחזיר.

    פרמטרים
    --------
    symbol : str
        סימול המדד/ETF (SPY, QQQ, IWM, EEM וכו').
    start, end : date | None
        טווח תאריכים לפילטור האינדקס.
    fields : Sequence[str] | None
        רשימת שדות פנדומנטליים (pe, pb, roe, dividend_yield וכו').
        אם None – מחזיר כל העמודות.
    provider : str | None
        שם ספק (כפי שנרשם ב-register_fundamentals_provider).
        אם None – ישתמש ב-SETTINGS.FUNDAMENTALS_DEFAULT_PROVIDER.
    allow_remote : bool
        אם False – מונע פניה לספק, יסתמך רק על קבצים לוקליים או יזרוק שגיאה.
    force_refresh : bool | None
        אם True – יתעלם מ-cache ויפנה לספק (אם allow_remote=True).
        אם None – משתמש ב-SETTINGS.FUNDAMENTALS_ALWAYS_REFRESH.
    require_fresh_local : bool | None
        אם True – לא יטען קובץ ישן (TTL עבר).
        אם False – יסכים לטעון גם קובץ ישן.
        אם None – ברירת מחדל True.
    allow_partial : bool | None
        אם True – אם חסרים חלק מהשדות, נמשיך עם מה שיש (ונרשום אזהרה).
        אם False – אם חסרים שדות, נזרוק ValueError.
        אם None – משתמש ב-SETTINGS.FUNDAMENTALS_ALLOW_PARTIAL.

    החזר
    -----
    DataFrame:
        אינדקס = DatetimeIndex (אם אפשר),
        עמודות = שדות פנדומנטליים מנורמלים (lower_snake_case).
    """
    if not symbol or not str(symbol).strip():
        raise ValueError("symbol must be a non-empty string")

    sym_norm = _normalize_symbol(symbol)
    use_force_refresh = _coerce_force_refresh(force_refresh)
    use_require_fresh_local = _coerce_require_fresh_local(require_fresh_local)
    use_allow_partial = _coerce_allow_partial(allow_partial)

    logger.debug(
        "load_index_fundamentals(symbol=%s, start=%s, end=%s, fields=%s, "
        "provider=%s, allow_remote=%s, force_refresh=%s, require_fresh_local=%s, "
        "allow_partial=%s)",
        sym_norm,
        start,
        end,
        fields,
        provider,
        allow_remote,
        use_force_refresh,
        use_require_fresh_local,
        use_allow_partial,
    )

    # ---------------------------------------------------------
    # 1) נסיון טעינה לוקאלית (אם לא ביקשו force_refresh)
    # ---------------------------------------------------------
    df_local: FundamentalFrame | None = None
    if not use_force_refresh:
        df_local = _load_from_disk(
            sym_norm,
            require_fresh=use_require_fresh_local,
        )

    if df_local is not None and not df_local.empty:
        # נחתוך טווח + שדות
        df_local = _apply_date_range(df_local, start, end)
        df_local = _select_fields(
            df_local,
            fields,
            allow_partial=use_allow_partial,
            symbol=sym_norm,
        )

        # אם ביקשו partial ומצאנו לפחות חלק → אפשר להחזיר
        if fields is None or df_local.shape[1] > 0:
            logger.info(
                "Returning fundamentals for %s from local cache (rows=%s, cols=%s)",
                sym_norm,
                df_local.shape[0],
                df_local.shape[1],
            )
            return df_local

    # ---------------------------------------------------------
    # 2) אם אין או לא מספיק → ספק מרוחק (אם מותר)
    # ---------------------------------------------------------
    if not allow_remote:
        # אין remote, ואין df_local טוב → שגיאה
        if df_local is None:
            raise FileNotFoundError(
                f"No suitable fundamentals cache for {sym_norm} and allow_remote=False"
            )
        # df_local קיים אבל חסר fields קריטיים (allow_partial=False)
        raise ValueError(
            f"Local fundamentals cache for {sym_norm} is insufficient and "
            f"allow_remote=False."
        )

    # פניה לספק (שם ספק אפקטיבי)
    df_remote, effective_provider = _fetch_via_provider(
        sym_norm,
        provider_name=provider,
        start=start,
        end=end,
        fields=fields,  # מותר לספק לנסות לפלטר בעצמו
    )

    # ננרמל / נחתוך שוב (לפחות כדי להבטיח עקביות מלאה)
    df_remote = _normalize_fundamentals_df(df_remote)
    df_remote = _apply_date_range(df_remote, start, end)
    df_remote = _select_fields(
        df_remote,
        fields,
        allow_partial=use_allow_partial,
        symbol=sym_norm,
    )

    # שמירה לקבצים
    _save_to_disk(
        sym_norm,
        df_remote,
        source=effective_provider,
        extra_meta={
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "requested_fields": list(fields) if fields else None,
        },
    )

    logger.info(
        "Returning fundamentals for %s from provider %s (rows=%s, cols=%s)",
        sym_norm,
        effective_provider,
        df_remote.shape[0],
        df_remote.shape[1],
    )

    return df_remote


# =========================
# Cached single-symbol API
# =========================

@lru_cache(maxsize=256)
def load_index_fundamentals_cached(
    symbol: str,
    start: date | None = None,
    end: date | None = None,
    fields: tuple[str, ...] | None = None,
) -> FundamentalFrame:
    """
    עטיפה עם LRU cache עבור load_index_fundamentals, לשימוש בתוך Session אחד.

    הערות:
    ------
    - fields מקבל tuple כדי שיהיה hashable עבור ה-cache.
    - לא כולל פרמטרים כמו provider/allow_remote/force_refresh – זה
      מיועד בעיקר לשימוש בתוך core כשאנחנו כבר ב-trusted environment.
    - אם תרצה cache-aware גם לפרמטרים מתקדמים, אפשר להרחיב לגרסה
      נוספת (או להעביר dict frozen).
    """
    # כאן נבחר התנהגות שמרנית:
    # - allow_remote=True
    # - force_refresh=False
    # - require_fresh_local=True
    # - allow_partial=True (כי מי שפונה לכאן כנראה בסדר עם subset/ברירת מחדל)
    fields_list: Sequence[str] | None = list(fields) if fields is not None else None
    return load_index_fundamentals(
        symbol,
        start=start,
        end=end,
        fields=fields_list,
        allow_remote=True,
        force_refresh=False,
        require_fresh_local=True,
        allow_partial=True,
    )


# =========================
# Public API — multi-symbol
# =========================

def load_indices_fundamentals(
    symbols: Sequence[str],
    *,
    start: date | None = None,
    end: date | None = None,
    fields: Sequence[str] | None = None,
    allow_remote: bool = True,
    force_refresh: bool | None = None,
    require_fresh_local: bool | None = None,
    allow_partial: bool | None = None,
    ignore_errors: bool = True,
) -> Dict[str, FundamentalFrame]:
    """
    טוען fundamentals למספר מדדים (universe) ומחזיר dict {symbol -> DataFrame}.

    פרמטרים
    --------
    symbols : Sequence[str]
        רשימת סימולים (מדדים/ETF).
    start, end, fields, allow_remote, force_refresh, require_fresh_local, allow_partial :
        עוברים ל-load_index_fundamentals.
    ignore_errors : bool
        אם True → אם יש כשל עבור סימול מסוים, נרשום אזהרה ונמשיך.
        אם False → נזרוק את השגיאה הראשונה שתופיע.

    החזר
    -----
    dict[str, DataFrame]
    """
    use_allow_partial = _coerce_allow_partial(allow_partial)
    results: Dict[str, FundamentalFrame] = {}

    for sym in symbols:
        try:
            df = load_index_fundamentals(
                sym,
                start=start,
                end=end,
                fields=fields,
                allow_remote=allow_remote,
                force_refresh=force_refresh,
                require_fresh_local=require_fresh_local,
                allow_partial=use_allow_partial,
            )
            results[_normalize_symbol(sym)] = df
        except Exception as exc:
            if ignore_errors:
                logger.warning(
                    "Failed to load fundamentals for %s: %s (ignore_errors=True)",
                    sym,
                    exc,
                )
                continue
            else:
                raise

    return results


# =========================
# Panel construction (MultiIndex)
# =========================

def _harmonize_columns_for_panel(
    frames: Mapping[str, FundamentalFrame],
) -> Dict[str, FundamentalFrame]:
    """
    מבטיח שכל ה-DataFrame-ים ב-universe יכילו את אותה רשימת עמודות,
    כך שאפשר לעשות concat בלי הפתעות.

    - מזהה את איחוד כל השדות.
    - עבור כל סימול → מוסיף עמודות חסרות כ-NaN.
    """
    if not frames:
        return {}

    all_cols: set[str] = set()
    for df in frames.values():
        all_cols.update(df.columns)

    all_cols_list = sorted(all_cols)
    out: Dict[str, FundamentalFrame] = {}
    for sym, df in frames.items():
        if df.empty:
            # נשמור DataFrame ריק עם אותן עמודות לכל הפחות
            out[sym] = pd.DataFrame(columns=all_cols_list)
            continue

        # reindex columns כדי להשלים חסרים ב-NaN
        df2 = df.reindex(columns=all_cols_list)
        out[sym] = df2

    return out


def build_fundamentals_panel(
    symbols: Sequence[str],
    *,
    start: date | None = None,
    end: date | None = None,
    fields: Sequence[str] | None = None,
    allow_remote: bool = True,
    force_refresh: bool | None = None,
    require_fresh_local: bool | None = None,
    allow_partial: bool | None = None,
    freq: str | None = None,
    ffill: bool = True,
    ignore_errors: bool = True,
) -> FundamentalPanel:
    """
    בונה Panel של fundamentals לכל ה-universe בפורמט DataFrame עם MultiIndex:

        index  = (date, symbol)
        columns = השדות הפנדומנטליים.

    יכול גם:
    - לבצע resample לתדירות אחידה (freq="M"/"Q"/"D"/...).
    - לבצע forward-fill (ffill=True) כדי "למרוח" נתונים רבעוניים לחודשים.

    פרמטרים
    --------
    symbols : Sequence[str]
        רשימת סימולים (מדדים/ETF).
    start, end, fields, allow_remote, force_refresh, require_fresh_local,
    allow_partial, ignore_errors :
        עוברים ל-load_indices_fundamentals.
    freq : str | None
        תדירות לפאנל (למשל "M" לחודשי, "Q" לרבעוני).
        אם None → לא מבוצע resample (משאירים את אינדקס הנתונים כפי שהוא).
        אם None ו-SETTINGS.FUNDAMENTALS_PANEL_FREQ לא None → אפשר לבחור להשתמש
        בברירת המחדל ממבנה ההגדרות בעתיד (כרגע freq מנצח אם סופק).
    ffill : bool
        אם True → לאחר resample, נבצע forward-fill בתוך כל סימול.
        מתאים לנתונים כמו EPS/Book/ROE שהם בדרך כלל "נתונים תקופתיים"
        שמתעדכנים אחת לרבעון/שנה.

    החזר
    -----
    FundamentalPanel (DataFrame):
        index: MultiIndex (date, symbol)
        columns: מאוחדים לכל ה-universe.
    """
    if not symbols:
        return pd.DataFrame()

    use_allow_partial = _coerce_allow_partial(allow_partial)
    sym_list = list(symbols)

    logger.info(
        "Building fundamentals panel for universe=%s, start=%s, end=%s, "
        "fields=%s, freq=%s, ffill=%s",
        sym_list,
        start,
        end,
        fields,
        freq or SETTINGS.FUNDAMENTALS_PANEL_FREQ,
        ffill,
    )

    # שלב 1: טעינת DataFrame לכל סימול
    frames_raw = load_indices_fundamentals(
        sym_list,
        start=start,
        end=end,
        fields=fields,
        allow_remote=allow_remote,
        force_refresh=force_refresh,
        require_fresh_local=require_fresh_local,
        allow_partial=use_allow_partial,
        ignore_errors=ignore_errors,
    )

    if not frames_raw:
        return pd.DataFrame()

    # שלב 2: אחידות עמודות
    frames = _harmonize_columns_for_panel(frames_raw)

    # שלב 3: resample (אם ביקשו freq)
    panel_pieces: list[pd.DataFrame] = []
    effective_freq = freq or SETTINGS.FUNDAMENTALS_PANEL_FREQ

    for sym, df in frames.items():
        if df.empty:
            continue

        df_sym = df.copy()

        # רק אם יש אינדקס זמן
        if isinstance(df_sym.index, pd.DatetimeIndex) and effective_freq:
            try:
                # לוקחים את הערך האחרון בכל bucket (סביר לפנדומנטל)
                df_sym = df_sym.resample(effective_freq).last()
                if ffill:
                    df_sym = df_sym.ffill()
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Resample failed for %s with freq=%s: %s",
                    sym,
                    effective_freq,
                    exc,
                )

        df_sym["symbol"] = _normalize_symbol(sym)
        panel_pieces.append(df_sym)

    if not panel_pieces:
        return pd.DataFrame()

    panel = pd.concat(panel_pieces)

    # בניית MultiIndex (date, symbol)
    if "symbol" in panel.columns:
        panel = panel.reset_index()  # "index" -> date (או שם אחר)
        # ננסה לזהות עמודת זמן
        date_col = _guess_date_column(panel) or "index"
        if date_col not in panel.columns:
            # אם זה המצב, משהו מאוד לא רגיל קרה – אבל ניפול חזרה לטיפול מינימלי
            logger.warning(
                "Failed to infer date column when building panel; "
                "result will not have proper MultiIndex."
            )
            return panel  # נחזיר as-is

        panel = panel.rename(columns={date_col: "date"})
        panel = panel.set_index(["date", "symbol"])
        panel.index = panel.index.set_names(["date", "symbol"])
        # נוודא אינדקס זמן
        panel = panel.sort_index()

    return panel


# ============================================================
# Part 4/4 — Advanced utilities, health checks & integrations
# ============================================================
"""
בחלק האחרון אנחנו מוסיפים שכבת כלים מתקדמים סביב שכבת הפנדומנטל:

10 רעיונות/יכולות מרכזיות שכבר תכננו:

1. incremental_refresh_index_fundamentals
   - רענון אינקרמנטלי למדד בודד (רק מ-last_date ואילך).

2. refresh_universe_fundamentals
   - רענון אינקרמנטלי/מלא ל-universe שלם.

3. summarize_fundamentals_coverage
   - סיכום כיסוי פנדומנטלי (rows/dates/meta) לכל סימול.

4. validate_fundamentals_schema
   - בדיקת schema – האם כל השדות הקריטיים קיימים? חסרים? מסוגים נכונים?

5. detect_fundamental_anomalies
   - זיהוי אנומליות פשוטות (קפיצות קיצוניות, ערכים בלתי סבירים).

6. fundamentals_health_check
   - "דוח בריאות" לדאטה – עדכניות, כיסוי, אנומליות.

7. get_latest_fundamentals_snapshot
   - snapshot חד-נקודתי (תאריך אחרון) לכל השדות עבור symbol/Universe.

8. fundamentals_panel_to_long
   - המרת Panel MultiIndex לפורמט long (date, symbol, field, value).

9. write_fundamentals_panel_to_duckdb
   - כתיבת Panel ל-DuckDB לטובת מחקר/שאילתות כבדות.

10. load_fundamentals_panel_from_duckdb
    - טעינת Panel מ-DuckDB כ-DataFrame MultiIndex.

בנוסף אתה ביקשת **עוד 6 רעיונות חדשים**, אז הוספנו:

11. compute_fundamentals_coverage_stats
    - סטטיסטיקות high-level על הכיסוי (כמה סימולים, ממוצע rows, טווח תאריכים).

12. compare_fundamentals_between_universes
    - השוואת ממוצעים/חציון של fundamentals בין שני יקומים (Universe A vs Universe B).

13. prune_stale_fundamentals_files
    - ניקוי קבצים ישנים/לא בשימוש – Housekeeping ברמת מערכת.

14. export_latest_fundamentals_to_dict
    - החזרת snapshot אחרון כ- dict נוח ל-UI / JSON.

15. ensure_minimum_history
    - בדיקה שכל סימול עומד בדרישת היסטוריה מינימלית (למשל 10 שנים אחורה).

16. generate_fundamentals_markdown_report
    - יצירת דוח Markdown אוטומטי לסקירה/תצוגה ב-UI/לוג.

שום דבר כאן לא משנה את ה-API הבסיסי – הכל הרחבות סביבו.
"""

from typing import Optional

# DuckDB אופציונלי – בסביבה שאין, פשוט נדלג על הפונקציות שמשתמשות בו
try:  # pragma: no cover
    import duckdb  # type: ignore
    _HAS_DUCKDB = True
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore
    _HAS_DUCKDB = False


# =========================
# 1) Incremental refresh (single symbol)
# =========================

def incremental_refresh_index_fundamentals(
    symbol: str,
    *,
    provider: str | None = None,
    fields: Sequence[str] | None = None,
    allow_partial: bool | None = None,
) -> FundamentalFrame:
    """
    מבצע רענון אינקרמנטלי למדד בודד:

    לוגיקה:
    -------
    1. מנסה לטעון קובץ קיים מהדיסק (גם אם לא טרי).
    2. מזהה את התאריך האחרון שיש לנו (last_date).
    3. קורא ל-provider החל מיום אחרי last_date (או מתחילת היסטוריה אם אין נתונים).
    4. מאחד (concat) בין הישן לחדש, מנקה כפילויות, מנרמל ושומר לדיסק.
    5. מחזיר את ה-DataFrame המלא (ישן+חדש).
    """
    sym_norm = _normalize_symbol(symbol)
    use_allow_partial = _coerce_allow_partial(allow_partial)

    # 1) דאטה קיים
    df_existing = _load_from_disk(sym_norm, require_fresh=False)

    last_date: date | None = None
    if df_existing is not None and not df_existing.empty:
        if isinstance(df_existing.index, pd.DatetimeIndex):
            last_ts = df_existing.index.max()
            last_date = last_ts.date()

    # 2) תחילת רענון
    if last_date is not None:
        start_refresh = last_date + timedelta(days=1)
    else:
        start_refresh = None

    logger.info(
        "Incremental refresh for %s starting from %s (prev_last_date=%s)",
        sym_norm,
        start_refresh,
        last_date,
    )

    # 3) משיכת דאטה חדש מה-provider
    df_new, effective_provider = _fetch_via_provider(
        sym_norm,
        provider_name=provider,
        start=start_refresh,
        end=None,
        fields=fields,
    )

    df_new = _normalize_fundamentals_df(df_new)

    # אם אין חדש – נחזיר את הקיים
    if df_existing is not None and not df_existing.empty:
        if df_new is None or df_new.empty:
            logger.info(
                "No new fundamentals found for %s; returning existing data only.",
                sym_norm,
            )
            df_existing = _select_fields(
                df_existing,
                fields,
                allow_partial=use_allow_partial,
                symbol=sym_norm,
            )
            return df_existing

        df_combined = pd.concat([df_existing, df_new])
    else:
        df_combined = df_new

    df_combined = _normalize_fundamentals_df(df_combined)
    df_combined = _select_fields(
        df_combined,
        fields,
        allow_partial=use_allow_partial,
        symbol=sym_norm,
    )

    _save_to_disk(
        sym_norm,
        df_combined,
        source=effective_provider,
        extra_meta={
            "incremental": True,
            "prev_last_date": last_date.isoformat() if last_date else None,
        },
    )

    logger.info(
        "Completed incremental refresh for %s (rows=%s, cols=%s)",
        sym_norm,
        df_combined.shape[0],
        df_combined.shape[1],
    )

    return df_combined


# =========================
# 2) Incremental refresh for universe
# =========================

def refresh_universe_fundamentals(
    symbols: Sequence[str],
    *,
    provider: str | None = None,
    fields: Sequence[str] | None = None,
    allow_partial: bool | None = None,
    ignore_errors: bool = True,
) -> Dict[str, FundamentalFrame]:
    """
    רענון אינקרמנטלי (או מלא, אם אין דאטה) ל-universe שלם.

    עבור כל סימול:
    - קורא ל-incremental_refresh_index_fundamentals.
    - מחזיר dict {symbol -> DataFrame}.
    """
    results: Dict[str, FundamentalFrame] = {}
    sym_list = list(symbols)
    if not sym_list:
        return results

    logger.info(
        "Refreshing universe fundamentals incrementally for %s symbols: %s",
        len(sym_list),
        sym_list,
    )

    for sym in sym_list:
        try:
            df = incremental_refresh_index_fundamentals(
                sym,
                provider=provider,
                fields=fields,
                allow_partial=allow_partial,
            )
            results[_normalize_symbol(sym)] = df
        except Exception as exc:
            if ignore_errors:
                logger.warning(
                    "Failed incremental refresh for %s: %s (ignore_errors=True)",
                    sym,
                    exc,
                )
                continue
            else:
                raise

    return results


# =========================
# 3) Coverage summary & stats
# =========================

def summarize_fundamentals_coverage(
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    מחזיר DataFrame עם סיכום כיסוי הפנדומנטל לכל סימול:

    עמודות:
        symbol
        has_file
        n_rows
        n_cols
        min_date
        max_date
        last_refresh
        fields (comma-separated string)
    """
    if symbols is None:
        symbols = list_cached_symbols()

    rows: list[dict[str, Any]] = []
    for sym in symbols:
        sym_norm = _normalize_symbol(sym)
        meta = _load_metadata(sym_norm)
        df = _load_from_disk(sym_norm, require_fresh=False)

        has_file = df is not None and not df.empty
        n_rows = int(df.shape[0]) if has_file else 0
        n_cols = int(df.shape[1]) if has_file else 0

        min_date = None
        max_date = None
        if has_file and isinstance(df.index, pd.DatetimeIndex):
            try:
                min_date = df.index.min().date().isoformat()
                max_date = df.index.max().date().isoformat()
            except Exception:  # pragma: no cover
                min_date = None
                max_date = None

        last_refresh = None
        fields_list: list[str] = []
        if meta is not None:
            last_refresh = meta.last_refresh.isoformat()
            fields_list = list(meta.fields or [])
        elif has_file:
            fields_list = list(df.columns)

        rows.append(
            {
                "symbol": sym_norm,
                "has_file": has_file,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "min_date": min_date,
                "max_date": max_date,
                "last_refresh": last_refresh,
                "fields": ", ".join(sorted(set(fields_list))) if fields_list else None,
            }
        )

    cov_df = pd.DataFrame(rows)
    cov_df = cov_df.sort_values("symbol").reset_index(drop=True)
    return cov_df


def compute_fundamentals_coverage_stats(
    symbols: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    סטטיסטיקות high-level על כיסוי fundamentals:

    מחזיר dict עם:
        n_symbols
        n_with_files
        avg_rows
        median_rows
        min_start_date
        max_end_date
    """
    cov = summarize_fundamentals_coverage(symbols)
    if cov.empty:
        return {
            "n_symbols": 0,
            "n_with_files": 0,
            "avg_rows": 0.0,
            "median_rows": 0.0,
            "min_start_date": None,
            "max_end_date": None,
        }

    with_files = cov[cov["has_file"]]

    def _safe_min(col: str) -> Optional[str]:
        vals = [v for v in with_files[col].dropna().tolist() if v]
        return min(vals) if vals else None

    def _safe_max(col: str) -> Optional[str]:
        vals = [v for v in with_files[col].dropna().tolist() if v]
        return max(vals) if vals else None

    stats = {
        "n_symbols": int(len(cov)),
        "n_with_files": int(with_files.shape[0]),
        "avg_rows": float(with_files["n_rows"].mean()) if not with_files.empty else 0.0,
        "median_rows": float(with_files["n_rows"].median()) if not with_files.empty else 0.0,
        "min_start_date": _safe_min("min_date"),
        "max_end_date": _safe_max("max_date"),
    }
    return stats


# =========================
# 4) Schema validation & anomalies
# =========================

def validate_fundamentals_schema(
    df: FundamentalFrame,
    required_fields: Sequence[str],
    *,
    symbol: str | None = None,
) -> Dict[str, Any]:
    """
    בודק האם DataFrame של fundamentals מכיל את כל השדות הנדרשים.

    מחזיר dict:
        {
            "ok": bool,
            "missing": [list of missing fields],
            "present": [list of present fields]
        }
    """
    df_cols_lower = {c.lower() for c in df.columns}
    requested_lower = [str(f).lower().strip() for f in required_fields]

    missing = [f for f in requested_lower if f not in df_cols_lower]
    present = [f for f in requested_lower if f in df_cols_lower]

    ok = len(missing) == 0
    if not ok:
        logger.warning(
            "Fundamentals schema validation failed for %s: missing=%s",
            symbol or "<unknown>",
            missing,
        )

    return {
        "ok": ok,
        "missing": missing,
        "present": present,
    }


def detect_fundamental_anomalies(
    df: FundamentalFrame,
    *,
    symbol: str | None = None,
    numeric_fields: Sequence[str] | None = None,
    zscore_threshold: float = 6.0,
) -> pd.DataFrame:
    """
    מזהה אנומליות בסיסיות בפנדומנטל:

    כללים:
    -------
    - אם numeric_fields=None → נשתמש בכל העמודות המספריות.
    - נחשב Z-score לכל סדרה (field) ונזהה ערכים עם |z| > zscore_threshold.
    - בנוסף, עבור פיצ'רים שלא אמורים להיות שליליים (כמו pe, pb, dividend_yield),
      נזהה ערכים שליליים.

    מחזיר DataFrame של anomalies:
        columns: date, field, value, zscore, rule
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "field", "value", "zscore", "rule"])

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(
            "detect_fundamental_anomalies: DataFrame index is not DatetimeIndex "
            "for %s; anomaly detection may be limited.",
            symbol or "<unknown>",
        )

    # בחירת שדות מספריים
    if numeric_fields is None:
        num_df = df.select_dtypes(include=["number"])
        numeric_fields = list(num_df.columns)

    records: list[dict[str, Any]] = []

    for field in numeric_fields:
        series = df.get(field)
        if series is None:
            continue
        try:
            s = pd.to_numeric(series, errors="coerce").dropna()
        except Exception:  # pragma: no cover
            continue
        if s.empty:
            continue

        mean = s.mean()
        std = s.std(ddof=0) or 1e-9
        zscores = (s - mean) / std

        # כלל 1: |z| גדול
        mask_extreme = zscores.abs() > zscore_threshold
        for ts, z in zscores[mask_extreme].items():
            records.append(
                {
                    "date": ts,
                    "field": field,
                    "value": s.loc[ts],
                    "zscore": float(z),
                    "rule": f"|z|>{zscore_threshold}",
                }
            )

        # כלל 2: שלילי לפיצ'רים שלא אמורים להיות שליליים
        non_negative_candidates = ("pe", "pb", "dividend", "yield", "margin")
        if any(tok in field for tok in non_negative_candidates):
            negative_mask = s < 0
            for ts, val in s[negative_mask].items():
                records.append(
                    {
                        "date": ts,
                        "field": field,
                        "value": float(val),
                        "zscore": None,
                        "rule": "negative_for_non_negative_field",
                    }
                )

    if not records:
        return pd.DataFrame(columns=["date", "field", "value", "zscore", "rule"])

    anomalies = pd.DataFrame(records)
    anomalies = anomalies.sort_values(["date", "field"]).reset_index(drop=True)
    return anomalies


# =========================
# 5) Health check
# =========================

def fundamentals_health_check(
    symbols: Sequence[str] | None = None,
    *,
    required_fields: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    "דוח בריאות" לדאטה הפנדומנטלי:

    מחזיר dict:
        {
            "coverage": DataFrame (summarize_fundamentals_coverage),
            "coverage_stats": dict (compute_fundamentals_coverage_stats),
            "schema_issues": dict[symbol -> {missing/present}],
            "anomaly_counts": dict[symbol -> int]
        }
    """
    if symbols is None:
        symbols = list_cached_symbols()

    coverage = summarize_fundamentals_coverage(symbols)
    coverage_stats = compute_fundamentals_coverage_stats(symbols)

    schema_issues: Dict[str, Any] = {}
    anomaly_counts: Dict[str, int] = {}

    for sym in symbols:
        sym_norm = _normalize_symbol(sym)
        df = _load_from_disk(sym_norm, require_fresh=False)
        if df is None or df.empty:
            continue

        if required_fields:
            schema_issues[sym_norm] = validate_fundamentals_schema(
                df,
                required_fields=required_fields,
                symbol=sym_norm,
            )

        anomalies = detect_fundamental_anomalies(df, symbol=sym_norm)
        anomaly_counts[sym_norm] = int(anomalies.shape[0])

    return {
        "coverage": coverage,
        "coverage_stats": coverage_stats,
        "schema_issues": schema_issues,
        "anomaly_counts": anomaly_counts,
    }


# =========================
# 6) Snapshots & export
# =========================

def get_latest_fundamentals_snapshot(
    symbol_or_symbols: str | Sequence[str],
    *,
    fields: Sequence[str] | None = None,
    allow_remote: bool = True,
) -> pd.DataFrame:
    """
    מחזיר snapshot אחרון (תאריך אחרון) עבור symbol אחד או רשימת symbols.

    החזר:
    ------
    DataFrame עם:
        index: symbol
        columns: fields (אם צוין) או כל העמודות.

    הערה:
    ------
    לא מחזיר את עמודת התאריך – זו כבר "חתיכת מצב" אחרונה.
    """
    if isinstance(symbol_or_symbols, str):
        symbols = [symbol_or_symbols]
    else:
        symbols = list(symbol_or_symbols)

    rows: list[pd.Series] = []
    index: list[str] = []

    for sym in symbols:
        df = load_index_fundamentals(
            sym,
            start=None,
            end=None,
            fields=fields,
            allow_remote=allow_remote,
            force_refresh=False,
            require_fresh_local=False,
            allow_partial=True,
        )
        if df is None or df.empty:
            continue

        if isinstance(df.index, pd.DatetimeIndex):
            last_row = df.iloc[-1]
        else:
            last_row = df.iloc[-1]

        rows.append(last_row)
        index.append(_normalize_symbol(sym))

    if not rows:
        return pd.DataFrame()

    snap_df = pd.DataFrame(rows, index=index)
    snap_df.index.name = "symbol"
    return snap_df


def export_latest_fundamentals_to_dict(
    symbol_or_symbols: str | Sequence[str],
    *,
    fields: Sequence[str] | None = None,
    allow_remote: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    מחזיר snapshot אחרון כ- dict נוח ל-UI / JSON:

        {
            "SPY": {"pe": 27.3, "pb": 4.9, ...},
            "QQQ": {...},
            ...
        }
    """
    snap_df = get_latest_fundamentals_snapshot(
        symbol_or_symbols,
        fields=fields,
        allow_remote=allow_remote,
    )
    if snap_df.empty:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for sym, row in snap_df.iterrows():
        out[str(sym)] = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    return out


# =========================
# 7) Panel transformations
# =========================

def fundamentals_panel_to_long(
    panel: FundamentalPanel,
) -> pd.DataFrame:
    """
    ממיר Panel MultiIndex (date, symbol) לפורמט long:

        columns: ["date", "symbol", "field", "value"]

    שימושי ל-analytics / plotting / שמירה כ-CSV/DB.
    """
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["date", "symbol", "field", "value"])

    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["date", "symbol"]:
        logger.warning(
            "fundamentals_panel_to_long: panel index is not MultiIndex ['date', 'symbol']; attempting best-effort reshape."
        )

    long_df = panel.reset_index().melt(
        id_vars=["date", "symbol"],
        var_name="field",
        value_name="value",
    )
    return long_df


# =========================
# 8) DuckDB integration (optional)
# =========================

def write_fundamentals_panel_to_duckdb(
    panel: FundamentalPanel,
    *,
    db_path: str | Path,
    table_name: str = "fundamentals_panel",
    mode: str = "replace",
) -> None:
    """
    כותב Panel ל-DuckDB (אם מותקן):

    פרמטרים:
        db_path   : הנתיב לקובץ DuckDB (למשל "data/fundamentals.duckdb").
        table_name: שם הטבלה.
        mode      : "replace" | "append".
    """
    if not _HAS_DUCKDB:
        logger.warning(
            "write_fundamentals_panel_to_duckdb: duckdb not installed; skipping."
        )
        return

    if panel is None or panel.empty:
        logger.info("write_fundamentals_panel_to_duckdb: panel is empty; nothing to write.")
        return

    db_path = Path(db_path)
    long_df = fundamentals_panel_to_long(panel)

    conn = duckdb.connect(str(db_path))
    try:
        if mode == "replace":
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.register("tmp_fundamentals", long_df)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} AS
            SELECT * FROM tmp_fundamentals
            """
        )
        if mode == "append":
            conn.execute(
                f"""
                INSERT INTO {table_name}
                SELECT * FROM tmp_fundamentals
                """
            )
        logger.info(
            "Wrote fundamentals panel to DuckDB %s (table=%s, rows=%s)",
            db_path,
            table_name,
            long_df.shape[0],
        )
    finally:
        conn.close()


def load_fundamentals_panel_from_duckdb(
    *,
    db_path: str | Path,
    table_name: str = "fundamentals_panel",
) -> FundamentalPanel:
    """
    טוען Panel בסיסי מ-DuckDB:

    מצופה שפורמט הטבלה יהיה:
        [date, symbol, field, value]

    מחזיר DataFrame MultiIndex (date, symbol) בעיצוב "wide".
    """
    if not _HAS_DUCKDB:
        logger.warning(
            "load_fundamentals_panel_from_duckdb: duckdb not installed; returning empty DataFrame."
        )
        return pd.DataFrame()

    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning(
            "load_fundamentals_panel_from_duckdb: db_path %s does not exist.",
            db_path,
        )
        return pd.DataFrame()

    conn = duckdb.connect(str(db_path))
    try:
        df_long = conn.execute(f"SELECT * FROM {table_name}").fetch_df()
    finally:
        conn.close()

    if df_long.empty:
        return pd.DataFrame()

    # מצופה שיש עמודות date, symbol, field, value
    required_cols = {"date", "symbol", "field", "value"}
    if not required_cols.issubset(df_long.columns):
        logger.warning(
            "load_fundamentals_panel_from_duckdb: table %s has columns %s, "
            "expected at least %s.",
            table_name,
            list(df_long.columns),
            required_cols,
        )
        return pd.DataFrame()

    df_wide = df_long.pivot_table(
        index=["date", "symbol"],
        columns="field",
        values="value",
    )
    df_wide.columns = [str(c) for c in df_wide.columns]
    df_wide = df_wide.sort_index()

    return df_wide


# =========================
# 9) Universe comparison & history requirements
# =========================

def compare_fundamentals_between_universes(
    universe_a: Sequence[str],
    universe_b: Sequence[str],
    *,
    fields: Sequence[str],
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """
    השוואת ממוצעים/חציון של fundamentals בין שני יקומים (Universe A vs B).

    מחזיר DataFrame עם:
        index: field
        columns: ["mean_a", "mean_b", "diff", "median_a", "median_b"]
    """
    panel_a = build_fundamentals_panel(
        universe_a,
        start=start,
        end=end,
        fields=fields,
        allow_remote=False,
        force_refresh=False,
        require_fresh_local=False,
        allow_partial=True,
        freq=None,
        ffill=False,
        ignore_errors=True,
    )
    panel_b = build_fundamentals_panel(
        universe_b,
        start=start,
        end=end,
        fields=fields,
        allow_remote=False,
        force_refresh=False,
        require_fresh_local=False,
        allow_partial=True,
        freq=None,
        ffill=False,
        ignore_errors=True,
    )

    long_a = fundamentals_panel_to_long(panel_a)
    long_b = fundamentals_panel_to_long(panel_b)

    stats: list[dict[str, Any]] = []
    for field in fields:
        fa = long_a[long_a["field"] == field].dropna(subset=["value"])
        fb = long_b[long_b["field"] == field].dropna(subset=["value"])
        if fa.empty and fb.empty:
            continue

        mean_a = float(fa["value"].mean()) if not fa.empty else float("nan")
        mean_b = float(fb["value"].mean()) if not fb.empty else float("nan")
        median_a = float(fa["value"].median()) if not fa.empty else float("nan")
        median_b = float(fb["value"].median()) if not fb.empty else float("nan")

        stats.append(
            {
                "field": field,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "diff": mean_a - mean_b if pd.notna(mean_a) and pd.notna(mean_b) else float("nan"),
                "median_a": median_a,
                "median_b": median_b,
            }
        )

    if not stats:
        return pd.DataFrame()

    df_stats = pd.DataFrame(stats).set_index("field")
    return df_stats


def ensure_minimum_history(
    symbols: Sequence[str],
    *,
    min_start_date: date,
) -> Dict[str, bool]:
    """
    בודק שכל סימול עומד בדרישת היסטוריה מינימלית (למשל 10 שנים אחורה).

    מחזיר dict {symbol -> bool} האם עומד בדרישה.
    """
    result: Dict[str, bool] = {}
    for sym in symbols:
        sym_norm = _normalize_symbol(sym)
        df = _load_from_disk(sym_norm, require_fresh=False)
        ok = False
        if df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex):
            try:
                min_dt = df.index.min().date()
                ok = min_dt <= min_start_date
            except Exception:  # pragma: no cover
                ok = False
        result[sym_norm] = ok
        if not ok:
            logger.warning(
                "Symbol %s does not meet minimum history requirement (min_date>%s).",
                sym_norm,
                min_start_date,
            )
    return result


# =========================
# 10) Housekeeping & reports
# =========================

def prune_stale_fundamentals_files(
    *,
    older_than_days: int,
    dry_run: bool = True,
) -> list[Path]:
    """
    מוחק (או מסמן) קבצי fundamentals ישנים מתיקיית הדאטה.

    פרמטרים:
        older_than_days : קבצים שגילם גדול מכך ייחשבו "ישנים".
        dry_run         : אם True → לא מוחק בפועל, רק מחזיר רשימת קבצים למחק.

    החזר:
        רשימת Paths שנמצאו ישנים (ואם dry_run=False – נמחקו בפועל).
    """
    base = SETTINGS.FUNDAMENTALS_DATA_DIR
    if not base.exists():
        return []

    cutoff = datetime.now() - timedelta(days=older_than_days)
    candidates: list[Path] = []
    for p in base.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".parquet", ".csv", ".json"):
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
        except OSError:  # pragma: no cover
            continue
        if mtime < cutoff:
            candidates.append(p)

    deleted: list[Path] = []
    for p in candidates:
        if dry_run:
            logger.info("[DRY-RUN] Would delete stale fundamentals file: %s", p)
        else:
            try:
                p.unlink()
                logger.info("Deleted stale fundamentals file: %s", p)
                deleted.append(p)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to delete stale fundamentals file %s: %s", p, exc)

    return candidates if dry_run else deleted


def generate_fundamentals_markdown_report(
    symbols: Sequence[str] | None = None,
    *,
    required_fields: Sequence[str] | None = None,
) -> str:
    """
    יוצר דוח Markdown מסוכם על כיסוי ובריאות הפנדומנטל.

    מתאים להזרקה לטאב "דוח תחזוקה" בדשבורד.
    """
    health = fundamentals_health_check(
        symbols=symbols,
        required_fields=required_fields,
    )
    cov: pd.DataFrame = health["coverage"]
    stats: Dict[str, Any] = health["coverage_stats"]
    anomaly_counts: Dict[str, int] = health["anomaly_counts"]

    lines: list[str] = []
    lines.append("# Fundamentals Health Report")
    lines.append("")
    lines.append("## Coverage Stats")
    lines.append("")
    lines.append(f"- Number of symbols tracked: **{stats.get('n_symbols', 0)}**")
    lines.append(f"- Symbols with files: **{stats.get('n_with_files', 0)}**")
    lines.append(f"- Avg rows per symbol: **{stats.get('avg_rows', 0):.1f}**")
    lines.append(f"- Median rows per symbol: **{stats.get('median_rows', 0):.1f}**")
    lines.append(
        f"- Earliest start date: **{stats.get('min_start_date') or 'N/A'}**, "
        f"latest end date: **{stats.get('max_end_date') or 'N/A'}**"
    )
    lines.append("")

    if not cov.empty:
        lines.append("## Per-Symbol Coverage (top 20)")
        lines.append("")
        # ניקח עד 20 שורות לצמצום
        head_cov = cov.head(20).copy()
        # נהפוך לטבלה Markdown בסיסית
        lines.append(head_cov.to_markdown(index=False))
        lines.append("")

    if anomaly_counts:
        lines.append("## Anomaly Counts")
        lines.append("")
        for sym, cnt in sorted(anomaly_counts.items()):
            lines.append(f"- **{sym}**: {cnt} anomalies")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Final __all__ — covers all 4 parts of fundamental_loader.py
# ============================================================

__all__ = [
    # Settings & Types
    "FundamentalsSettings",
    "SETTINGS",
    "FundamentalFrame",
    "FundamentalPanel",
    "FundamentalMeta",

    # Providers
    "BaseFundamentalsProvider",
    "register_fundamentals_provider",
    "get_fundamentals_provider",

    # Public API לטעינה / פאנל
    "load_index_fundamentals",
    "load_index_fundamentals_cached",
    "load_indices_fundamentals",
    "build_fundamentals_panel",

    # Diagnostics בסיסיים
    "list_cached_symbols",
    "get_cached_meta",

    # Incremental & Universe refresh
    "incremental_refresh_index_fundamentals",
    "refresh_universe_fundamentals",

    # Coverage & Health
    "summarize_fundamentals_coverage",
    "compute_fundamentals_coverage_stats",
    "validate_fundamentals_schema",
    "detect_fundamental_anomalies",
    "fundamentals_health_check",
    "ensure_minimum_history",

    # Snapshots & Export
    "get_latest_fundamentals_snapshot",
    "export_latest_fundamentals_to_dict",

    # Panel ops & DuckDB
    "fundamentals_panel_to_long",
    "write_fundamentals_panel_to_duckdb",
    "load_fundamentals_panel_from_duckdb",

    # Universe comparison, housekeeping & reports
    "compare_fundamentals_between_universes",
    "prune_stale_fundamentals_files",
    "generate_fundamentals_markdown_report",
]
