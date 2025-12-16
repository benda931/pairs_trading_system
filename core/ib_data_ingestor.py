# -*- coding: utf-8 -*-
"""
core/ib_data_ingestor.py — Historical Price Ingestor from IBKR (HF-grade)
=========================================================================

תפקידים עיקריים:
-----------------
1. למשוך דאטה היסטורי מ-IBKR עבור Universe ש-SqlStore מחזיק (dq_pairs / universe אחר).
2. לשמור את הדאטה לטבלת prices ב-SqlStore (מקור אמת אחיד למחירים).
3. לתמוך ב-Ingestion מלא או אינקרמנטלי (רק תאריכים חסרים).
4. לספק מחלקת IBDataIngestor ברמת קרן גידור לסקריפטים ודשבורד.
5. (אופציונלי) להשתמש ב-bridge חכם (core.data_ingest_bridge) כדי למשוך רק gaps חסרים.

תלות:
------
- core.app_context.AppContext
- core.sql_store.SqlStore
- root.ibkr_connection.get_ib_instance / ib_connection_status
- ib_insync (לקריאה ל-IBKR)
- (אופציונלי) core.data_ingest_bridge.ensure_prices_for_symbol / ensure_prices_for_pair

שימוש (Script חיצוני, ingest לסימבול בודד):
-----------------------------------------
    from datetime import date
    from core.ib_data_ingestor import IBDataIngestor
    from core.sql_store import SqlStore
    from core.config_manager import load_settings  # לדוגמה

    settings = load_settings(PROJECT_ROOT)
    store = SqlStore.from_settings(settings, env=settings.env)
    ingestor = IBDataIngestor.from_settings(settings=settings, store=store)

    ingestor.ensure_prices_for_symbol(
        symbol="XLP",
        start_date=date(2015, 1, 1),
        end_date=date.today(),
        use_bridge=True,      # אם data_ingest_bridge + get_price_range זמינים
    )

שימוש (Universe ingestion ב-CLI):
---------------------------------
    python -m core.ib_data_ingestor --start 2020-01-01 --end 2024-12-31 \
        --env dev --max-symbols 20 --force-ib-enable --use-bridge
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from time import sleep
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
)

import logging

import pandas as pd

from core.app_context import AppContext
from core.sql_store import SqlStore
from root.ibkr_connection import get_ib_instance, ib_connection_status

if TYPE_CHECKING:  # רק לטייפ-הינטס; בזמן ריצה נשתמש ב-import דינמי
    from ib_insync import IB  # pragma: no cover

# ניסיון טעינה של ה-bridge (אופציונלי, לא חובה)
try:
    from core.data_ingest_bridge import (
        ensure_prices_for_symbol as _bridge_ensure_symbol,
        ensure_prices_for_pair as _bridge_ensure_pair,
    )
except Exception:  # pragma: no cover - הגנה רכה
    _bridge_ensure_symbol = None
    _bridge_ensure_pair = None


logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s [ib_data_ingestor] %(message)s")
    )
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# טיפוסי עזר
DateLike = Union[date, datetime, str]
FetchFunc = Callable[[str, date, date], pd.DataFrame]


# =========================
# Config dataclass (Universe-level)
# =========================

@dataclass
class IBIngestConfig:
    """
    קונפיגורציה ל-Ingestion היסטורי מ-IBKR ברמת Universe.

    Attributes
    ----------
    start : str
        תאריך התחלה (YYYY-MM-DD).
    end : str
        תאריך סוף (YYYY-MM-DD).
    env : str
        Environment לוגי (dev/paper/live) – משמש לסימון ב-SqlStore.
    universe_table : str
        טבלת Universe ב-SqlStore (ברירת מחדל: dq_pairs).
    max_symbols : int | None
        הגבלת מספר סימבולים (להגנה / בדיקות).
    bar_size : str
        barSizeSetting עבור IBKR (למשל "1 day", "1 hour").
    what_to_show : str
        TRADES / MIDPOINT / BID_ASK / ...
    incremental : bool
        אם True → מושך רק אחרי התאריך האחרון שכבר שמור ב-prices עבור כל סימבול
        (fallback פשוט גם בלי bridge).
    sleep_between_symbols_sec : float
        זמן המתנה קטן בין סימבולים (pacing מול IB API).
    dry_run : bool
        אם True → לא שומר ל-SqlStore, רק מדווח ללוג.
    force_ib_enable : bool
        אם True → מתעלם מ-settings.ib_enable לצורך ingestion ידני.
    use_bridge : bool
        אם True → מנסה להשתמש ב-core.data_ingest_bridge.ensure_prices_for_symbol
        + SqlStore.get_price_range (אם קיימים) כדי למשוך רק gaps חסרים בצורה חכמה.
    max_chunk_days : int
        גודל מקסימלי של chunk ימים למשיכה אחת (רלוונטי ל-bridge).
    log_prefix : str
        prefix ללוגים (IBKR / Ingest / וכו').
    """

    start: str
    end: str
    env: str
    universe_table: str = "dq_pairs"
    max_symbols: Optional[int] = None
    bar_size: str = "1 day"
    what_to_show: str = "TRADES"
    incremental: bool = True
    sleep_between_symbols_sec: float = 0.2
    dry_run: bool = False
    force_ib_enable: bool = False

    # תוספות HF-grade
    use_bridge: bool = True
    max_chunk_days: int = 365
    log_prefix: str = "IBKR"


# =========================
# OO wrapper — IBDataIngestor (Fund-level API)
# =========================

@dataclass
class IBDataIngestor:
    """
    IBDataIngestor — עטיפה אובייקטית ל-Ingestion מ-IBKR ל-SqlStore.

    Responsibilities
    ----------------
    - להחזיק:
        * store : SqlStore
        * settings : אובייקט ה-settings שלך (מה-config_manager)
        * env : תגית סביבת הרצה ("dev"/"paper"/"live")
        * ib : אובייקט IB של ib_insync (או None אם לא מחוברים)
        * default_bar_size / default_what_to_show / exchange / currency

    - לספק:
        * from_settings(...) — בנאי "חכם" שמבין את ה־settings שלך.
        * ensure_connected() — דואג שיהיה חיבור ל-IB (או זורק RuntimeError).
        * ingest_symbol(...) — מושך היסטוריה לסימבול ושומר ל-SqlStore.
        * ingest_pair(...) — לזוג כמו "XLP-XLY".
        * ingest_symbols(...) — לרשימה של טיקרים.
        * ensure_prices_for_symbol / ensure_prices_for_pair — עטיפות חכמות שמנסות
          להשתמש ב-bridge (אם זמין) ורק אם לא — נופלות למצב incremental הפשוט.
    """

    store: SqlStore
    settings: Any
    env: str = "dev"

    default_bar_size: str = "1 day"
    default_what_to_show: str = "TRADES"
    default_exchange: str = "SMART"
    default_currency: str = "USD"

    incremental_default: bool = True

    ib: Optional["IB"] = None

    # ---------- Factory חכם מה-settings ----------

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        store: Optional[SqlStore] = None,
        env: Optional[str] = None,
        connect: bool = True,
    ) -> "IBDataIngestor":
        """
        יצירת Ingestor מתוך settings קיימים.

        - אם לא הועבר store → נבנה SqlStore.from_settings.
        - אם לא הועבר env → ננסה settings.env, אחרת "dev".
        - שואב ברירות מחדל ל-bar_size / what_to_show / exchange / currency מה-settings אם קיימים.
        - אם ib_enable=False → ib יישאר None (ועדיין אפשר להשתמש בדאטה מ־SqlStore).
        - אם connect=True ו-ib_enable=True → ננסה להתחבר ל-IB מיד.
        """
        if env is None:
            env = getattr(settings, "env", "dev")

        if store is None:
            store = SqlStore.from_settings(settings, env=env)

        default_bar_size = getattr(settings, "ib_bar_size", "1 day")
        default_what_to_show = getattr(settings, "ib_what_to_show", "TRADES")
        default_exchange = getattr(settings, "ib_exchange", "SMART")
        default_currency = getattr(settings, "ib_currency", "USD")
        incremental_default = bool(getattr(settings, "ib_incremental_default", True))

        ingestor = cls(
            store=store,
            settings=settings,
            env=env,
            default_bar_size=default_bar_size,
            default_what_to_show=default_what_to_show,
            default_exchange=default_exchange,
            default_currency=default_currency,
            incremental_default=incremental_default,
            ib=None,
        )

        ib_enable = getattr(settings, "ib_enable", True)
        if not ib_enable:
            logger.warning(
                "IBDataIngestor.from_settings: ib_enable is False – "
                "starting without IB connection (ib=None)."
            )
            return ingestor

        if connect:
            ingestor.ensure_connected()

        return ingestor

    # ---------- Utilities פנימיים ----------

    @staticmethod
    def _normalize_date(value: Any) -> str:
        """
        מקבל date/datetime/str ומחזיר string בפורמט YYYY-MM-DD.
        """
        if isinstance(value, (date, datetime)):
            return value.strftime("%Y-%m-%d")
        return str(value)

    def _make_fetch_fn(
        self,
        *,
        bar_size: Optional[str] = None,
        what_to_show: Optional[str] = None,
    ) -> FetchFunc:
        """
        מחזיר פונקציית fetch(symbol, start_date, end_date) שמתאימה ל-bridge,
        ומבוססת על fetch_history_for_symbol + self.ib.
        """
        def _fn(symbol: str, start_d: date, end_d: date) -> pd.DataFrame:
            self.ensure_connected()
            return fetch_history_for_symbol(
                self.ib,  # type: ignore[arg-type]
                symbol=symbol,
                currency=self.default_currency,
                exchange=self.default_exchange,
                start=start_d.strftime("%Y-%m-%d"),
                end=end_d.strftime("%Y-%m-%d"),
                bar_size=bar_size or self.default_bar_size,
                what_to_show=what_to_show or self.default_what_to_show,
            )

        return _fn

    # ---------- חיבור / בדיקת חיבור ----------

    def ensure_connected(self, *, force_reconnect: bool = False) -> None:
        """
        דואג שיש חיבור ל-IB:

        - אם כבר יש self.ib מחובר ו-force_reconnect=False → לא עושה כלום.
        - אם ib_enable=False → זורק RuntimeError עם הסבר.
        - אחרת מנסה להתחבר דרך _connect_ib_from_settings.
        """
        ib_enable = getattr(self.settings, "ib_enable", True)
        if not ib_enable:
            raise RuntimeError(
                "IBDataIngestor.ensure_connected called but ib_enable=False in settings. "
                "עדכן את הקונפיג (ib_enable=True) או אל תקרא לפונקציות שמצריכות חיבור ל-IB."
            )

        if self.ib is not None and not force_reconnect:
            try:
                if self.ib.isConnected():
                    return
            except Exception:
                self.ib = None

        self.ib = _connect_ib_from_settings(self.settings)
        if self.ib is None:
            raise RuntimeError(
                "IBDataIngestor.ensure_connected: failed to obtain IB connection "
                "(self.ib is None). בדוק ש-TWS/Gateway פתוח, ה-API מאופשר "
                "ושה-host/port/client_id נכונים ב-settings."
            )

    # ---------- ingest לסימבול אחד (Low-level, ללא bridge) ----------

    def ingest_symbol(
        self,
        symbol: str,
        *,
        start_date: Any,
        end_date: Any,
        bar_size: Optional[str] = None,
        what_to_show: Optional[str] = None,
        force: bool = False,  # שמור להרחבה עתידית (מחיקה/החלפה)
        dry_run: bool = False,
    ) -> int:
        """
        ingest_symbol — משיכת דאטה היסטורי לסימבול אחד ושמירה ל-SqlStore.

        הערה:
        -----
        זו פונקציה "פשוטה": תמיד מושכת את כל הטווח start_date→end_date.
        ל-Incremental חכם יותר (רק gaps חסרים) השתמש ב:
            ensure_prices_for_symbol(..., use_bridge=True)
        או ב־IBIngestConfig.use_bridge ב-ingest_history_for_universe.

        Parameters
        ----------
        symbol : str
            טיקר, למשל "XLP".
        start_date, end_date : date | datetime | str
            טווח תאריכים (כולל) לבקשת דאטה מ-IBKR.
        bar_size : str, default from self.default_bar_size
        what_to_show : str, default from self.default_what_to_show
        force : bool
            כרגע רק בלוג; בעתיד אפשר להרחיב למחיקת טווח קיים לפני כתיבה.
        dry_run : bool
            אם True → לא כותב ל-SqlStore, רק מושך ומחזיר מספר שורות פוטנציאלי.

        Returns
        -------
        int
            מספר השורות שנשמרו בפועל (או שהיו נשמרות ב-dry_run).
        """
        log = logger

        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)

        if pd.to_datetime(start_str) > pd.to_datetime(end_str):
            log.warning(
                "ingest_symbol(%s): start_date %s is after end_date %s – skipping.",
                symbol,
                start_str,
                end_str,
            )
            return 0

        bar_size_eff = bar_size or self.default_bar_size
        what_eff = what_to_show or self.default_what_to_show

        log.info(
            "IBDataIngestor.ingest_symbol: %s | %s → %s | bar_size=%s | what=%s | force=%s | dry_run=%s",
            symbol,
            start_str,
            end_str,
            bar_size_eff,
            what_eff,
            force,
            dry_run,
        )

        self.ensure_connected()

        df = fetch_history_for_symbol(
            self.ib,  # type: ignore[arg-type]
            symbol,
            currency=self.default_currency,
            exchange=self.default_exchange,
            start=start_str,
            end=end_str,
            bar_size=bar_size_eff,
            what_to_show=what_eff,
        )

        if df is None or df.empty:
            log.info(
                "No data returned from IB for %s in range %s → %s.",
                symbol,
                start_str,
                end_str,
            )
            return 0

        if dry_run:
            log.info(
                "Dry-run: would save %d rows for %s into SqlStore (env=%s).",
                len(df),
                symbol,
                getattr(self, "env", None),
            )
            return int(len(df))

        env = getattr(self, "env", None)
        self.store.save_price_history(
            symbol,
            df,
            env=env,
            if_exists="append",
        )

        log.info(
            "Saved %d rows for %s into SqlStore (env=%s).",
            len(df),
            symbol,
            env,
        )

        return int(len(df))

    # ---------- ingest לזוג (XLP-XLY) — fallback incremental פשוט ----------

    def ingest_pair(
        self,
        pair: str,
        *,
        start_date: Any,
        end_date: Any,
        bar_size: Optional[str] = None,
        what_to_show: Optional[str] = None,
        incremental: Optional[bool] = None,
        dry_run: bool = False,
    ) -> int:
        """
        ingest_pair — מושך דאטה לשני הסימבולים בזוג 'XLP-XLY' ושומר ל-SqlStore.

        אם incremental=True → משתמש ב-SqlStore.load_price_history כדי
        להתחיל מיום אחרי התאריך האחרון הקיים (fallback פשוט, לא bridge מלא).
        """
        if "-" not in pair:
            raise ValueError(f"pair צריך להיות בפורמט 'AAA-BBB', קיבלתי: {pair!r}")

        sym_x, sym_y = [s.strip().upper() for s in pair.split("-", 1)]
        log = logger

        log.info(
            "IBDataIngestor.ingest_pair(pair=%s → %s,%s) | start=%s | end=%s | incremental=%s | dry_run=%s",
            pair,
            sym_x,
            sym_y,
            start_date,
            end_date,
            incremental,
            dry_run,
        )

        inc_flag = self.incremental_default if incremental is None else incremental

        total_rows = 0
        for sym in (sym_x, sym_y):
            s_start = start_date
            s_end = end_date

            if inc_flag:
                try:
                    df_existing = self.store.load_price_history(sym, env=self.env)
                except Exception:
                    df_existing = pd.DataFrame()

                if not df_existing.empty:
                    if not isinstance(df_existing.index, pd.DatetimeIndex):
                        if "date" in df_existing.columns:
                            df_existing = df_existing.set_index("date")
                    if isinstance(df_existing.index, pd.DatetimeIndex):
                        last_dt = df_existing.index.max()
                        if pd.notna(last_dt):
                            next_dt = (last_dt + pd.Timedelta(days=1)).date()
                            if isinstance(s_start, (date, datetime)):
                                if next_dt > s_start:
                                    s_start = next_dt
                            else:
                                base_dt = pd.to_datetime(s_start).date()
                                if next_dt > base_dt:
                                    s_start = next_dt

            n_rows = self.ingest_symbol(
                symbol=sym,
                start_date=s_start,
                end_date=s_end,
                bar_size=bar_size,
                what_to_show=what_to_show,
                force=False,
                dry_run=dry_run,
            )
            total_rows += n_rows

        return total_rows

    # ---------- ingest לרשימת סימבולים (fallback incremental פשוט) ----------

    def ingest_symbols(
        self,
        symbols: Sequence[str],
        *,
        start_date: Any,
        end_date: Any,
        bar_size: Optional[str] = None,
        what_to_show: Optional[str] = None,
        incremental: Optional[bool] = None,
        sleep_between_sec: float = 0.0,
        dry_run: bool = False,
    ) -> int:
        """
        ingest_symbols — ריצה על רשימת סימבולים.

        משמש בעיקר לסקריפטים ברמת Universe / Campaign.

        הערה:
        -----
        ל-Ingestion חכם של gaps בלבד (בהתבסס על SqlStore.get_price_range +
        data_ingest_bridge) עדיף להשתמש במתודות
        ensure_prices_for_symbol / ingest_history_for_universe עם use_bridge=True.
        """
        inc_flag = self.incremental_default if incremental is None else incremental
        total_rows = 0
        for idx, sym in enumerate(symbols, start=1):
            sym = sym.strip().upper()
            if not sym:
                continue

            logger.info(
                "(%d/%d) Ingesting %s ... (incremental=%s, dry_run=%s)",
                idx,
                len(symbols),
                sym,
                inc_flag,
                dry_run,
            )

            s_start = start_date
            s_end = end_date

            if inc_flag:
                try:
                    df_existing = self.store.load_price_history(sym, env=self.env)
                except Exception:
                    df_existing = pd.DataFrame()

                if not df_existing.empty:
                    if not isinstance(df_existing.index, pd.DatetimeIndex):
                        if "date" in df_existing.columns:
                            df_existing = df_existing.set_index("date")
                    if isinstance(df_existing.index, pd.DatetimeIndex):
                        last_dt = df_existing.index.max()
                        if pd.notna(last_dt):
                            next_dt = (last_dt + pd.Timedelta(days=1)).date()
                            if isinstance(s_start, (date, datetime)):
                                if next_dt > s_start:
                                    s_start = next_dt
                            else:
                                base_dt = pd.to_datetime(s_start).date()
                                if next_dt > base_dt:
                                    s_start = next_dt

            n_rows = self.ingest_symbol(
                symbol=sym,
                start_date=s_start,
                end_date=s_end,
                bar_size=bar_size,
                what_to_show=what_to_show,
                force=False,
                dry_run=dry_run,
            )
            total_rows += n_rows

            if sleep_between_sec > 0:
                sleep(sleep_between_sec)

        return total_rows

    # ---------- High-level bridge: ensure_prices_* ----------

    def ensure_prices_for_symbol(
        self,
        symbol: str,
        *,
        start_date: DateLike,
        end_date: DateLike,
        use_bridge: bool = True,
        max_chunk_days: int = 365,
        log_prefix: str = "IBKR",
        dry_run: bool = False,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        High-level API: מוודא שיש מחירים ב-SqlStore עבור symbol בטווח [start,end].

        עדיפות:
        --------
        1. אם use_bridge=True, יש core.data_ingest_bridge וה-SqlStore תומך get_price_range →
           שימוש ב-bridge (ensure_prices_for_symbol) עם fetch_fn על בסיס self.ib.

        2. אחרת → fallback ל-ingest_symbol incremental-style:
           - בודק existing range דרך load_price_history.
           - אם incremental נדרש → מעדכן start_date.
           - מריץ ingest_symbol על הטווח המלא.

        החזרה:
        -------
        (min_date, max_date) כפי שמוחזרים מ-SqlStore (אם get_price_range קיים),
        או (None, None) אם אין מידע.
        """
        from datetime import timedelta

        symbol = str(symbol).strip()
        if not symbol:
            logger.warning("ensure_prices_for_symbol: empty symbol.")
            return None, None

        # 1) נתיב bridge אם אפשר
        has_bridge = (
            use_bridge
            and _bridge_ensure_symbol is not None
            and hasattr(self.store, "get_price_range")
        )

        start_d = pd.to_datetime(start_date).date()
        end_d = pd.to_datetime(end_date).date()
        if start_d > end_d:
            logger.warning(
                "ensure_prices_for_symbol(%s): start_date %s > end_date %s — skipping.",
                symbol,
                start_d,
                end_d,
            )
            return None, None

        eff_env = self.env or getattr(self.store, "default_env", "dev")

        # dry_run + bridge — רק נשתמש ב-get_price_range ונספר gaps לוגית
        if has_bridge and not dry_run:
            fetch_fn = self._make_fetch_fn()
            return _bridge_ensure_symbol(  # type: ignore[call-arg]
                store=self.store,
                symbol=symbol,
                start=start_d,
                end=end_d,
                env=eff_env,
                fetch_fn=fetch_fn,
                max_chunk_days=max_chunk_days,
                log_prefix=log_prefix,
            )

        # bridge לא זמין / dry_run=True → fallback incremental פשוט
        logger.info(
            "ensure_prices_for_symbol: using fallback incremental path "
            "(bridge_available=%s, dry_run=%s) for %s.",
            has_bridge,
            dry_run,
            symbol,
        )

        # נבדוק טווח קיים בסיסי דרך load_price_history
        try:
            df_existing = self.store.load_price_history(symbol, env=eff_env)
        except Exception:
            df_existing = pd.DataFrame()

        existing_min: Optional[pd.Timestamp]
        existing_max: Optional[pd.Timestamp]
        if df_existing is not None and not df_existing.empty:
            if not isinstance(df_existing.index, pd.DatetimeIndex):
                if "date" in df_existing.columns:
                    df_existing = df_existing.set_index("date")
            if isinstance(df_existing.index, pd.DatetimeIndex):
                existing_min = df_existing.index.min()
                existing_max = df_existing.index.max()
            else:
                existing_min = existing_max = None
        else:
            existing_min = existing_max = None

        # אם incremental: נתחיל מהיום שאחרי existing_max
        s_start = start_d
        s_end = end_d
        if existing_max is not None:
            next_day = (existing_max + pd.Timedelta(days=1)).date()
            if next_day > s_start:
                s_start = next_day

        if s_start <= s_end:
            self.ingest_symbol(
                symbol=symbol,
                start_date=s_start,
                end_date=s_end,
                dry_run=dry_run,
            )
        else:
            logger.info(
                "ensure_prices_for_symbol(%s): nothing new to ingest for [%s, %s].",
                symbol,
                start_d,
                end_d,
            )

        # טווח מעודכן — אם ל-SqlStore אין get_price_range פשוט נחזיר None,None
        if hasattr(self.store, "get_price_range"):
            try:
                return self.store.get_price_range(symbol, env=eff_env)  # type: ignore[call-arg]
            except Exception:
                return None, None

        return existing_min, existing_max

    def ensure_prices_for_pair(
        self,
        pair: str,
        *,
        start_date: DateLike,
        end_date: DateLike,
        use_bridge: bool = True,
        max_chunk_days: int = 365,
        log_prefix: str = "IBKR",
        dry_run: bool = False,
    ) -> Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
        """
        High-level API: מוודא מחירים עבור זוג 'SYM_X-SYM_Y'.

        אם bridge זמין → משתמש ב-ensure_prices_for_pair מה-bridge.
        אחרת → פשוט קורא ensure_prices_for_symbol פעמיים (fallback).
        """
        if "-" not in pair:
            raise ValueError(f"pair צריך להיות בפורמט 'AAA-BBB', קיבלתי: {pair!r}")
        sym_x, sym_y = [s.strip().upper() for s in pair.split("-", 1)]

        start_d = pd.to_datetime(start_date).date()
        end_d = pd.to_datetime(end_date).date()

        eff_env = self.env or getattr(self.store, "default_env", "dev")
        has_bridge = (
            use_bridge
            and _bridge_ensure_pair is not None
            and hasattr(self.store, "get_price_range")
        )

        if has_bridge and not dry_run:
            fetch_fn = self._make_fetch_fn()
            return _bridge_ensure_pair(  # type: ignore[call-arg]
                store=self.store,
                sym_x=sym_x,
                sym_y=sym_y,
                start=start_d,
                end=end_d,
                env=eff_env,
                fetch_fn=fetch_fn,
                max_chunk_days=max_chunk_days,
                log_prefix=log_prefix,
            )

        # fallback: שתי קריאות נפרדות
        logger.info(
            "ensure_prices_for_pair: using fallback symbol-by-symbol path "
            "(bridge_available=%s, dry_run=%s) for pair=%s.",
            has_bridge,
            dry_run,
            pair,
        )
        res_x = self.ensure_prices_for_symbol(
            symbol=sym_x,
            start_date=start_d,
            end_date=end_d,
            use_bridge=False,
            max_chunk_days=max_chunk_days,
            log_prefix=log_prefix,
            dry_run=dry_run,
        )
        res_y = self.ensure_prices_for_symbol(
            symbol=sym_y,
            start_date=start_d,
            end_date=end_d,
            use_bridge=False,
            max_chunk_days=max_chunk_days,
            log_prefix=log_prefix,
            dry_run=dry_run,
        )
        return {"sym_x": res_x, "sym_y": res_y}


# =========================
# Internal helpers (module-level)
# =========================

def _import_ib_insync() -> Any:
    """
    מייבא ib_insync בצורה בטוחה.

    מחזיר את המודול עצמו או None אם לא מותקן.
    """
    try:
        import ib_insync  # type: ignore
        return ib_insync
    except Exception as e:  # pragma: no cover
        logger.error("Failed to import ib_insync: %s", e)
        return None


def _connect_ib_via_connection_manager(settings: Any) -> Optional["IB"]:
    """
    משתמש ב-root.ibkr_connection.get_ib_instance כדי לקבל IB (אם אפשר).

    הקטע החשוב: ממיר את settings ל-dict כדי למנוע TypeError
    כש-get_ib_instance מצפה למבנה ניתן-לאיטרציה (dict / mapping).
    """
    cfg: Dict[str, Any]
    try:
        if hasattr(settings, "as_dict"):
            cfg_candidate = settings.as_dict()  # type: ignore[call-arg]
            cfg = dict(cfg_candidate) if isinstance(cfg_candidate, dict) else {}
        elif hasattr(settings, "__dict__"):
            cfg = dict(settings.__dict__)  # SimpleNamespace / dataclass / וכו'
        else:
            cfg = {}
    except Exception as e:
        logger.warning("Failed to serialize settings for get_ib_instance: %s", e)
        cfg = {}

    ib = None
    try:
        ib = get_ib_instance(
            readonly=True,
            use_singleton=False,
            profile=getattr(settings, "ib_mode", None),
            settings=cfg,
        )
    except Exception as e:
        logger.warning("get_ib_instance(...) raised: %s", e)
        ib = None

    if ib is not None:
        try:
            status = ib_connection_status(ib)
            logger.info("IB connection status: %r", status)
        except Exception:
            pass

    return ib


def _connect_ib_from_settings(settings: Any) -> Optional["IB"]:
    """
    מחבר ל-IBKR:

    1. מנסה דרך get_ib_instance (ibkr_connection).
    2. אם נכשל, fallback ל-ib_insync.IB().connect(host,port,clientId).
    """
    ib = _connect_ib_via_connection_manager(settings)
    if ib is not None:
        try:
            connected = ib.isConnected()
        except Exception:
            connected = None
        logger.info("IB connected via ibkr_connection: %r", connected)
        if connected:
            return ib

    ib_insync = _import_ib_insync()
    if ib_insync is None:
        return None

    host = getattr(settings, "ib_host", "127.0.0.1")
    port = int(getattr(settings, "ib_port", 7497))
    client_id = int(getattr(settings, "ib_client_id", 1))

    ib = ib_insync.IB()
    logger.info("Connecting directly to IBKR at %s:%s (clientId=%s)...", host, port, client_id)
    try:
        ib.connect(host, port, clientId=client_id)
    except Exception as e:
        logger.exception("Direct IBKR connect failed: %s", e)
        return None

    try:
        connected = ib.isConnected()
    except Exception:
        connected = None
    if not connected:
        logger.error("Direct IBKR connection failed (ib.isConnected() is False).")
        return None

    logger.info("Connected to IBKR directly: %r", connected)
    return ib


def fetch_history_for_symbol(
    ib: "IB",
    symbol: str,
    *,
    currency: str = "USD",
    exchange: str = "SMART",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
) -> pd.DataFrame:
    """
    מושך דאטה היסטורי לסימבול אחד מ-IBKR באמצעות ib_insync.

    מחזיר DataFrame עם index=date ועמודות:
        open, high, low, close, volume
    """
    ib_insync = _import_ib_insync()
    if ib_insync is None:
        return pd.DataFrame()

    contract = ib_insync.Stock(symbol, exchange, currency)
    ib.qualifyContracts(contract)

    try:
        # endDateTime='' → עד "עכשיו"; durationStr גדול → נסנן מקומית ל-start/end
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="10 Y",
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=False,
            formatDate=1,
        )
    except Exception as e:
        logger.warning("reqHistoricalData failed for %s: %s", symbol, e)
        return pd.DataFrame()

    if not bars:
        logger.info("No historical data returned for %s.", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(b.date),
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(getattr(b, "volume", 0)),
            }
            for b in bars
        ]
    )
    df = df.set_index("date").sort_index()

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

    return df


def _get_universe_symbols_from_store(
    store: SqlStore,
    *,
    env: str,
    universe_table: str = "dq_pairs",
    max_symbols: Optional[int] = None,
) -> List[str]:
    """
    מחלץ רשימת סימבולים ל-Ingestion מ-SqlStore.

    ברירת מחדל:
        - טוען מ-load_pair_quality (dq_pairs), אם קיימת.
    אפשרויות הרחבה:
        - universe_table אחר (למשל "universe_equities").
    """
    if universe_table == "dq_pairs":
        df_pairs = None
        try:
            df_pairs = store.load_pair_quality(env=env, section="data_quality")
        except TypeError:
            try:
                df_pairs = store.load_pair_quality()  # type: ignore[call-arg]
            except Exception:
                df_pairs = pd.DataFrame()
        except AttributeError:
            df_pairs = store.read_table("dq_pairs")
    else:
        df_pairs = store.read_table(universe_table)

    if df_pairs is None or df_pairs.empty:
        logger.warning("No pairs found in universe table '%s'.", universe_table)
        return []

    sym_x = df_pairs["sym_x"].astype(str).tolist() if "sym_x" in df_pairs.columns else []
    sym_y = df_pairs["sym_y"].astype(str).tolist() if "sym_y" in df_pairs.columns else []
    symbols = sorted(set(sym_x + sym_y))

    if max_symbols is not None:
        symbols = symbols[:max_symbols]

    return symbols


def _determine_incremental_start_for_symbol(
    store: SqlStore,
    symbol: str,
    cfg: IBIngestConfig,
) -> str:
    """
    קובע תאריך התחלה אינקרמנטלי לסימבול:

    - אם cfg.incremental=False → מחזיר cfg.start.
    - אם True → בודק מה התאריך האחרון ב-prices עבור הסימבול + 1 יום.
      אם אין דאטה בכלל → cfg.start.
    """
    if not cfg.incremental:
        return cfg.start

    try:
        df_existing = store.load_price_history(symbol, env=cfg.env)
    except Exception:
        df_existing = pd.DataFrame()

    if df_existing.empty:
        return cfg.start

    if not isinstance(df_existing.index, pd.DatetimeIndex):
        if "date" in df_existing.columns:
            df_existing = df_existing.set_index("date")
        else:
            return cfg.start

    last_date = df_existing.index.max()
    if pd.isna(last_date):
        return cfg.start

    next_day = last_date + pd.Timedelta(days=1)
    start_dt = pd.to_datetime(cfg.start)
    effective_dt = max(start_dt, next_day)
    return effective_dt.strftime("%Y-%m-%d")


# =========================
# Public API — Universe ingestion (CLI-friendly)
# =========================

def ingest_history_for_universe(cfg: IBIngestConfig) -> None:
    """
    משיכת דאטה היסטורי עבור Universe שמוגדר ב-SqlStore,
    ושמירתו לטבלת prices.

    זרימה:
    -------
    1. AppContext + settings + SqlStore + IBDataIngestor.from_settings.
    2. בדיקת ib_enable (עם override דרך cfg.force_ib_enable).
    3. טעינת Universe (dq_pairs או טבלה אחרת) → רשימת סימבולים.
    4. אם cfg.use_bridge=True ויש bridge + get_price_range:
        * לכל סימבול: ensure_prices_for_symbol עם bridge (gaps בלבד).
       אחרת:
        * לכל סימבול: ingest_symbol אינקרמנטלי (fallback).
    5. אם cfg.dry_run=True → מושכים / מודדים טווחים, אבל לא כותבים ל-SqlStore.
    """
    app_ctx = AppContext.get_global()
    settings = app_ctx.settings
    store = SqlStore.from_settings(settings, env=getattr(settings, "env", None))

    ib_enable = getattr(settings, "ib_enable", None)
    if ib_enable is False and not cfg.force_ib_enable:
        logger.error(
            "ib_enable is False in settings — ingestion aborted. "
            "אם אתה מריץ ידנית ורוצה להתעלם, השתמש force_ib_enable=True."
        )
        return
    elif ib_enable is False and cfg.force_ib_enable:
        logger.warning(
            "ib_enable is False in settings, אבל force_ib_enable=True — "
            "ממשיך Ingestion בכל זאת."
        )
    elif ib_enable is None:
        logger.warning(
            "ib_enable is None / not set in settings — "
            "מתייחס לזה כאילו ib_enable=True ל-Ingestion ידני."
        )

    ingestor = IBDataIngestor.from_settings(
        settings=settings,
        store=store,
        env=cfg.env,
        connect=not cfg.dry_run,  # ב-dry_run אפשר גם בלי חיבור (אם לא צריך fetch אמיתי)
    )

    symbols = _get_universe_symbols_from_store(
        store,
        env=cfg.env,
        universe_table=cfg.universe_table,
        max_symbols=cfg.max_symbols,
    )
    if not symbols:
        logger.warning("No symbols in universe — nothing to ingest.")
        return

    logger.info(
        "Starting IBKR ingestion for %d symbols (env=%s) between %s and %s. "
        "use_bridge=%s, incremental=%s, dry_run=%s",
        len(symbols),
        cfg.env,
        cfg.start,
        cfg.end,
        cfg.use_bridge,
        cfg.incremental,
        cfg.dry_run,
    )

    use_bridge_effective = (
        cfg.use_bridge
        and _bridge_ensure_symbol is not None
        and hasattr(store, "get_price_range")
    )

    total_rows = 0
    for idx, sym in enumerate(symbols, start=1):
        logger.info("(%d/%d) Ingesting %s ...", idx, len(symbols), sym)

        if use_bridge_effective and not cfg.dry_run:
            # נתיב bridge – gaps בלבד, עם chunking
            start_effective = pd.to_datetime(cfg.start).date()
            end_effective = pd.to_datetime(cfg.end).date()
            ingestor.ensure_prices_for_symbol(
                symbol=sym,
                start_date=start_effective,
                end_date=end_effective,
                use_bridge=True,
                max_chunk_days=cfg.max_chunk_days,
                log_prefix=cfg.log_prefix,
                dry_run=False,
            )
            # אין לנו ספירה ישירה של rows; אפשר להעריך בעתיד (מכיסוי לפני/אחרי)
        else:
            # fallback incremental פשוט (או dry_run)
            start_effective = _determine_incremental_start_for_symbol(store, sym, cfg)
            if pd.to_datetime(start_effective) > pd.to_datetime(cfg.end):
                logger.info(
                    "Symbol %s already up-to-date (latest >= end=%s); skipping.",
                    sym,
                    cfg.end,
                )
            else:
                n_rows = ingestor.ingest_symbol(
                    symbol=sym,
                    start_date=start_effective,
                    end_date=cfg.end,
                    bar_size=cfg.bar_size,
                    what_to_show=cfg.what_to_show,
                    force=False,
                    dry_run=cfg.dry_run,
                )
                total_rows += n_rows

        if cfg.sleep_between_symbols_sec > 0:
            sleep(cfg.sleep_between_symbols_sec)

    logger.info(
        "IBKR ingestion completed. Total rows inserted (approx, fallback path): %d",
        total_rows,
    )


# =========================
# CLI entry point
# =========================

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest historical prices from IBKR into SqlStore.prices"
    )
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument("--env", type=str, default="dev", help="Environment label (dev/paper/live)")
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit number of symbols")
    parser.add_argument(
        "--universe-table", type=str, default="dq_pairs", help="Universe table name"
    )
    parser.add_argument("--bar-size", type=str, default="1 day", help="IB barSizeSetting")
    parser.add_argument(
        "--what-to-show",
        type=str,
        default="TRADES",
        help="IB whatToShow (TRADES/MIDPOINT/...)",
    )
    parser.add_argument(
        "--full-refresh",
        dest="full_refresh",
        action="store_true",
        help="Disable incremental mode (reload full history for each symbol)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write to SQL, only log")
    parser.add_argument(
        "--force-ib-enable",
        action="store_true",
        help="Ignore ib_enable=False in settings for manual ingestion",
    )
    parser.add_argument(
        "--use-bridge",
        action="store_true",
        help="Use core.data_ingest_bridge + SqlStore.get_price_range for gap-only ingestion when available",
    )
    parser.add_argument(
        "--max-chunk-days",
        type=int,
        default=365,
        help="Maximum days per single IBKR request chunk (bridge mode)",
    )
    parser.add_argument(
        "--log-prefix",
        type=str,
        default="IBKR",
        help="Prefix for log lines (helpful when multiple ingestion processes run)",
    )

    args = parser.parse_args()

    cfg = IBIngestConfig(
        start=args.start,
        end=args.end,
        env=args.env,
        universe_table=args.universe_table,
        max_symbols=args.max_symbols,
        bar_size=args.bar_size,
        what_to_show=args.what_to_show,
        incremental=not args.full_refresh,
        dry_run=args.dry_run,
        force_ib_enable=args.force_ib_enable,
        use_bridge=bool(args.use_bridge),
        max_chunk_days=int(args.max_chunk_days),
        log_prefix=str(args.log_prefix),
    )

    ingest_history_for_universe(cfg)
