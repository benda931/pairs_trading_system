# -*- coding: utf-8 -*-
"""
common/market_data_router.py — Smart Market-Data Router (HF-grade, v2)
======================================================================

מטרה:
-----
1. לנהל רשימת ספקי דאטה (IBKR, Yahoo, וכו').
2. לבחור באופן חכם מאיזה ספק להביא דאטה:
   - לפי בחירת המשתמש (preferred_source)
   - לפי סדר עדיפויות (priority)
   - עם fallback לספקים אחרים במקרה של כישלון.
3. להחזיר תמיד DataFrame אחיד (דרך ה-providers ב־data_providers.py)
   + שם הספק שממנו הגיע הדאטה, לצורך לוגים ושקיפות.

שדרוגי v2 (HF-grade):
----------------------
- Circuit-Breaker לספקים "חולים" (דלג על ספק שנכשל שוב ושוב לזמן קצוב).
- סטטיסטיקות Router ו-per-provider (כמה הצלחות / כשלונות, היסטוריה, וכו').
- describe_providers() שמחזיר DataFrame נוח להצגה בטאבי קונפיג/דיאגנוסטיקה.
- get_last() אחיד למחיר אחרון, עם אותם מנגנוני fallback ו-circuit-breaker.
- לוגים עדינים בסגנון ספרייה (NullHandler) – המערכת הראשית מחליטה מה להדפיס.

שימוש טיפוסי:
--------------
    from ib_insync import IB
    from common.data_providers import IBKRProvider, YahooProvider
    from common.market_data_router import MarketDataRouter, build_default_router

    ib = IB()
    ib.connect(...)

    router = build_default_router(ib=ib, use_yahoo=True)

    # היסטוריה
    res = router.get_history(
        symbols=["XLY", "XLC"],
        period="6mo",
        bar_size="1d",
        preferred_source="ibkr",   # או None ל-Auto
    )
    df = res.df
    print("נתונים הגיעו מ:", res.source)

    # מחיר אחרון
    last = router.get_last("XLY")
    if last.ok:
        print("מחיר אחרון:", last.price, "מ־", last.source)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from .data_providers import (
    BarSize,
    WhatToShow,
    MarketDataProvider,
    IBKRProvider,
    YahooProvider,
    normalize_symbols,
    available_providers_summary,
)

# לוגר בסגנון ספרייה – לא כופה handlers על כל המערכת
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ========================= Data structures =======================


@dataclass
class RouterResult:
    """
    תוצאה מרוטינג להיסטוריה:
    - df        : הדאטה שהתקבל
    - source    : שם הספק שממנו הגיע הדאטה (ibkr / yahoo / וכו')
    - tried     : רשימת שמות ספקים שניסינו (לשקיפות / Debug)
    - errors    : Mapping ספק -> הודעת שגיאה אחרונה
    """

    df: pd.DataFrame
    source: str
    tried: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.df is not None and not self.df.empty and bool(self.source)


@dataclass
class LastPriceResult:
    """
    תוצאה של get_last:
    - price : מחיר אחרון (אם הצליח)
    - source: שם הספק שממנו הגיע המחיר
    - tried : רשימת ספקים שניסינו
    - errors: Mapping ספק -> הודעת שגיאה אחרונה
    """

    price: float | None
    source: str | None
    tried: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.price is not None and bool(self.source)


@dataclass
class ProviderStats:
    """
    Counters per provider – שימוש לדיאגנוסטיקה.
    """

    history_successes: int = 0
    history_failures: int = 0
    last_successes: int = 0
    last_failures: int = 0


@dataclass
class RouterStats:
    """
    Counters כלליים – לכמה בקשות וקצת חלוקה per-provider.
    """

    total_history_requests: int = 0
    total_last_requests: int = 0
    history_successes: int = 0
    history_failures: int = 0
    last_successes: int = 0
    last_failures: int = 0
    per_provider: Dict[str, ProviderStats] = field(default_factory=dict)


@dataclass
class ProviderState:
    """
    State פנימי לספק – עבור Circuit-Breaker בסיסי.
    """

    name: str
    priority: int
    consecutive_failures: int = 0
    last_error: str | None = None
    last_failure_ts: float | None = None
    last_success_ts: float | None = None
    circuit_open_until: float | None = None  # timestamp עד מתי אנחנו מדלגים


# ========================= MarketDataRouter ======================


class MarketDataRouter:
    """
    Smart router over multiple MarketDataProvider instances.

    Features:
    ---------
    - בחירה מפורשת בספק ע"פ preferred_source.
    - Auto-fallback לספקים אחרים לפי priority (או לפי סדר שהועבר).
    - healthcheck בסיסי כדי לדלג על ספקים "מתים".
    - Circuit-Breaker לספקים שנכשלים שוב ושוב (backoff אוטומטי).
    - סטטיסטיקות ו-describe_providers() לדיאגנוסטיקה.
    - החזרת RouterResult / LastPriceResult עם פירוט מי ניסינו ומי נכשל.
    """

    def __init__(
        self,
        providers: Sequence[MarketDataProvider],
        *,
        prefer: Sequence[str] | None = None,
        allow_fallbacks: bool = True,
        strict: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        providers : list[MarketDataProvider]
            רשימת ספקים זמינים (IBKRProvider, YahooProvider, וכו').
        prefer : list[str] | None
            אם None → ינוצל סדר priority של הספקים.
            אם רשימה → זהו סדר העדיפויות לספקים (למשל ["ibkr","yahoo"]).
        allow_fallbacks : bool
            אם True → אם ספק אחד נכשל, מנסים את הבא בתור.
        strict : bool
            אם True → במקרה של כישלון אצל כל הספקים & require_non_empty=True,
                       ניתן לזרוק Exception במקום לחזור עם df ריק.
        """
        if not providers:
            raise ValueError("MarketDataRouter requires at least one provider.")

        # Map name → provider (אחרון מנצח אם יש שם כפול)
        self.providers: Dict[str, MarketDataProvider] = {
            p.name: p for p in providers
        }
        self.allow_fallbacks = allow_fallbacks
        self.strict = strict

        # סדר עדיפויות
        if prefer is not None:
            self.prefer: List[str] = list(prefer)
        else:
            self.prefer = sorted(
                self.providers.keys(),
                key=lambda n: getattr(self.providers[n], "priority", 100),
            )

        # מצב פנימי לספקים + סטטיסטיקות
        self._provider_state: Dict[str, ProviderState] = {}
        self._stats = RouterStats()

        for name, prov in self.providers.items():
            priority = getattr(prov, "priority", 100)
            self._provider_state[name] = ProviderState(name=name, priority=priority)
            self._stats.per_provider[name] = ProviderStats()

        logger.info(
            "MarketDataRouter initialized with providers=%s, prefer=%s, allow_fallbacks=%s",
            list(self.providers.keys()),
            self.prefer,
            self.allow_fallbacks,
        )

    # ---------- Internal helpers: circuit-breaker & stats ----------

    def _can_use_provider(self, name: str) -> bool:
        state = self._provider_state.get(name)
        if state is None:
            return True
        if state.circuit_open_until is None:
            return True
        now = time.time()
        if now >= state.circuit_open_until:
            # circuit נפתח מחדש
            state.circuit_open_until = None
            return True
        logger.debug(
            "Skipping provider '%s' – circuit open until %s",
            name,
            datetime.fromtimestamp(state.circuit_open_until),
        )
        return False

    def _mark_success(self, name: str) -> None:
        state = self._provider_state.get(name)
        if state is None:
            return
        state.consecutive_failures = 0
        state.last_error = None
        state.circuit_open_until = None
        state.last_success_ts = time.time()

    def _mark_failure(self, name: str, error: str) -> None:
        state = self._provider_state.get(name)
        if state is None:
            return
        now = time.time()
        state.consecutive_failures += 1
        state.last_error = error
        state.last_failure_ts = now

        # backoff אקספוננציאלי עד מקסימום 5 דקות
        backoff = min(300.0, 2.0 ** min(state.consecutive_failures, 8))
        state.circuit_open_until = now + backoff

        logger.debug(
            "Provider '%s' marked as failure #%d (backoff=%.1fs, until=%s): %s",
            name,
            state.consecutive_failures,
            backoff,
            datetime.fromtimestamp(state.circuit_open_until),
            error,
        )

    def _update_stats(self, name: str, *, success: bool, kind: str) -> None:
        ps = self._stats.per_provider.setdefault(name, ProviderStats())
        if kind == "history":
            if success:
                self._stats.history_successes += 1
                ps.history_successes += 1
            else:
                self._stats.history_failures += 1
                ps.history_failures += 1
        elif kind == "last":
            if success:
                self._stats.last_successes += 1
                ps.last_successes += 1
            else:
                self._stats.last_failures += 1
                ps.last_failures += 1

    # ---------- Public helpers: stats & diagnostics ----------

    def get_stats(self, *, as_dict: bool = False) -> RouterStats | Dict[str, Any]:
        """
        החזרת סטטיסטיקות Router ברמת-high level.
        ניתן להשתמש בזה לטאב דיאגנוסטיקה.

        Parameters
        ----------
        as_dict : bool
            אם True → מחזיר dict (בעזרת dataclasses.asdict) שנוח ל-json / yaml.

        Returns
        -------
        RouterStats | dict
        """
        if as_dict:
            return asdict(self._stats)
        return self._stats

    def describe_providers(self) -> pd.DataFrame:
        """
        תיאור סטטוס הספקים (state + stats) כ-DataFrame.

        עמודות לדוגמה:
        --------------
        name, priority, consecutive_failures, circuit_open,
        last_error, last_failure_ts, last_success_ts,
        history_successes, history_failures,
        last_successes, last_failures
        """
        rows: List[Dict[str, Any]] = []
        for name, prov in self.providers.items():
            st = self._provider_state.get(name)
            ps = self._stats.per_provider.get(name)
            rows.append(
                {
                    "name": name,
                    "priority": getattr(prov, "priority", None),
                    "consecutive_failures": st.consecutive_failures if st else 0,
                    "circuit_open": bool(st.circuit_open_until) if st else False,
                    "circuit_open_until": (
                        datetime.fromtimestamp(st.circuit_open_until)
                        if st and st.circuit_open_until
                        else None
                    ),
                    "last_error": st.last_error if st else None,
                    "last_failure_ts": (
                        datetime.fromtimestamp(st.last_failure_ts)
                        if st and st.last_failure_ts
                        else None
                    ),
                    "last_success_ts": (
                        datetime.fromtimestamp(st.last_success_ts)
                        if st and st.last_success_ts
                        else None
                    ),
                    "history_successes": ps.history_successes if ps else 0,
                    "history_failures": ps.history_failures if ps else 0,
                    "last_successes": ps.last_successes if ps else 0,
                    "last_failures": ps.last_failures if ps else 0,
                }
            )
        return pd.DataFrame(rows)

    # ---------- Provider management ----------

    def register_provider(self, provider: MarketDataProvider, *, override: bool = True) -> None:
        """
        הוספת ספק חדש בזמן ריצה.
        """
        if provider.name in self.providers and not override:
            raise ValueError(f"Provider '{provider.name}' already registered.")
        self.providers[provider.name] = provider
        if provider.name not in self.prefer:
            self.prefer.append(provider.name)

        priority = getattr(provider, "priority", 100)
        self._provider_state[provider.name] = ProviderState(
            name=provider.name,
            priority=priority,
        )
        self._stats.per_provider.setdefault(provider.name, ProviderStats())

        logger.info(
            "Registered provider '%s' [priority=%s]",
            provider.name,
            priority,
        )

    def unregister_provider(self, name: str) -> None:
        """
        הסרת ספק מהרוטר.
        """
        self.providers.pop(name, None)
        self._provider_state.pop(name, None)
        # סטטיסטיקות אנחנו שומרים – כדי שלא נאבד היסטוריה
        if name in self.prefer:
            self.prefer = [n for n in self.prefer if n != name]
        logger.info("Unregistered provider '%s'", name)

    # ---------- Core routing API: history ----------

    def get_history(
        self,
        symbols: Sequence[str] | Mapping[str, Any] | str,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        period: str | None = "6mo",
        bar_size: BarSize = "1d",
        what_to_show: WhatToShow = "TRADES",
        use_rth: bool = True,
        preferred_source: str | None = None,
        require_non_empty: bool = False,
        min_symbols: int = 1,
        **kwargs: Any,
    ) -> RouterResult:
        """
        נסיון חכם להביא דאטה היסטורי עבור סימבולים מסוימים.

        Parameters
        ----------
        symbols : Sequence[str] | Mapping | str
            רשימת סימבולים, מילון עם "SYMBOLS", או מחרוזת מסובכת.
        start, end, period, bar_size, what_to_show, use_rth :
            עוברים הלאה לספק עצמו.
        preferred_source : str | None
            שם ספק מועדף (ibkr / yahoo / וכו').
            אם None → יבחר לפי self.prefer.
        require_non_empty : bool
            אם True → df ריק ייחשב ככישלון (וננסה ספק אחר).
        min_symbols : int
            אם > 1 → דורש מינימום כמות סימבולים כתקינים ב-DataFrame (ע"פ עמודת symbol).

        Returns
        -------
        RouterResult
            df, source, tried, errors

        התנהגות:
        --------
        - אם כל הספקים נכשלו:
            - strict=False → df ריק, source="".
            - strict=True  → ייזרק RuntimeError.
        """
        self._stats.total_history_requests += 1

        norm_syms = normalize_symbols(symbols)
        if not norm_syms:
            logger.warning("MarketDataRouter.get_history: empty symbol list after normalize.")
            return RouterResult(df=pd.DataFrame(), source="", tried=[], errors={})

        tried: List[str] = []
        errors: Dict[str, str] = {}

        def _try_provider(name: str) -> pd.DataFrame | None:
            prov = self.providers.get(name)
            if prov is None:
                errors[name] = "Provider not registered"
                logger.warning("Provider '%s' is not registered.", name)
                return None

            tried.append(name)

            if not self._can_use_provider(name):
                msg = "circuit open (backoff active)"
                errors[name] = msg
                logger.info("Provider '%s' skipped in get_history: %s", name, msg)
                return None

            # Healthcheck פשוט: אם נפל → לא ננסה בכלל
            try:
                if not prov.healthcheck():
                    msg = "healthcheck failed"
                    errors[name] = msg
                    logger.warning("Provider '%s' healthcheck failed.", name)
                    self._mark_failure(name, msg)
                    self._update_stats(name, success=False, kind="history")
                    return None
            except Exception as exc:
                msg = f"healthcheck exception: {exc}"
                errors[name] = msg
                logger.warning("Provider '%s' healthcheck raised: %s", name, exc)
                self._mark_failure(name, msg)
                self._update_stats(name, success=False, kind="history")
                return None

            try:
                logger.info(
                    "Trying provider '%s' for symbols=%s period=%s bar_size=%s",
                    name,
                    norm_syms,
                    period,
                    bar_size,
                )
                df = prov.get_history(
                    norm_syms,
                    start=start,
                    end=end,
                    period=period,
                    bar_size=bar_size,
                    what_to_show=what_to_show,
                    use_rth=use_rth,
                    **kwargs,
                )
                if df is None or df.empty:
                    msg = "empty result"
                    errors[name] = msg
                    logger.info("Provider '%s' returned empty DataFrame.", name)
                    if require_non_empty:
                        self._mark_failure(name, msg)
                        self._update_stats(name, success=False, kind="history")
                    return None

                # min_symbols check
                if min_symbols > 1 and "symbol" in df.columns:
                    unique_syms = df["symbol"].dropna().unique()
                    if len(unique_syms) < min_symbols:
                        msg = f"only {len(unique_syms)} symbols in data (<{min_symbols})"
                        errors[name] = msg
                        logger.info(
                            "Provider '%s' has insufficient symbols: %s",
                            name,
                            msg,
                        )
                        self._mark_failure(name, msg)
                        self._update_stats(name, success=False, kind="history")
                        return None

                # הצלחה
                self._mark_success(name)
                self._update_stats(name, success=True, kind="history")
                return df

            except Exception as exc:
                msg = f"exception: {exc}"
                errors[name] = msg
                logger.warning(
                    "Provider '%s' failed for %s: %s",
                    name,
                    norm_syms,
                    exc,
                )
                self._mark_failure(name, msg)
                self._update_stats(name, success=False, kind="history")
                return None

        # 1) נסיון עם preferred_source (אם קיים)
        candidate_order: List[str] = []

        if preferred_source:
            candidate_order.append(preferred_source)

        # 2) שאר הספקים לפי סדר עדיפויות
        if self.allow_fallbacks:
            for name in self.prefer:
                if name not in candidate_order:
                    candidate_order.append(name)

        # הרצה בפועל
        for name in candidate_order:
            df = _try_provider(name)
            if df is not None:
                # הצלחנו
                return RouterResult(df=df, source=name, tried=tried, errors=errors)

        # אם הגענו עד כאן – כל הספקים נכשלו
        if self.strict and require_non_empty:
            raise RuntimeError(
                f"All providers failed for symbols={norm_syms}. Errors={errors}"
            )

        logger.error(
            "All providers failed or returned empty for symbols=%s. Errors=%s",
            norm_syms,
            errors,
        )
        return RouterResult(df=pd.DataFrame(), source="", tried=tried, errors=errors)

    # ---------- Core routing API: last price ----------

    def get_last(
        self,
        symbol: str,
        *,
        preferred_source: str | None = None,
        allow_fallbacks: bool | None = None,
    ) -> LastPriceResult:
        """
        ניסיון חכם להביא מחיר "אחרון" (לייב / מעודכן) עבור סמל אחד.

        מחזיר:
            LastPriceResult(price, source, tried, errors)

        strict:
            אם self.strict=True וכל הספקים נכשלו → ייזרק RuntimeError.
        """
        if allow_fallbacks is None:
            allow_fallbacks = self.allow_fallbacks

        self._stats.total_last_requests += 1

        tried: List[str] = []
        errors: Dict[str, str] = {}

        candidates: List[str] = []
        if preferred_source:
            candidates.append(preferred_source)
        if allow_fallbacks:
            for name in self.prefer:
                if name not in candidates:
                    candidates.append(name)

        for name in candidates:
            prov = self.providers.get(name)
            if prov is None:
                errors[name] = "Provider not registered"
                continue

            tried.append(name)

            if not self._can_use_provider(name):
                msg = "circuit open (backoff active)"
                errors[name] = msg
                logger.info("Provider '%s' skipped in get_last: %s", name, msg)
                continue

            # Healthcheck
            try:
                if not prov.healthcheck():
                    msg = "healthcheck failed"
                    errors[name] = msg
                    logger.debug("Provider '%s' healthcheck failed for get_last.", name)
                    self._mark_failure(name, msg)
                    self._update_stats(name, success=False, kind="last")
                    continue
            except Exception as exc:
                msg = f"healthcheck exception: {exc}"
                errors[name] = msg
                logger.debug(
                    "Provider '%s' healthcheck raised in get_last: %s",
                    name,
                    exc,
                )
                self._mark_failure(name, msg)
                self._update_stats(name, success=False, kind="last")
                continue

            # locate getter method
            if hasattr(prov, "get_last"):
                getter_name = "get_last"
            elif hasattr(prov, "get_last_price"):
                getter_name = "get_last_price"
            else:
                errors[name] = "no get_last/get_last_price on provider"
                continue

            try:
                getter = getattr(prov, getter_name)
                price = getter(symbol)
                if price is None:
                    msg = "None price"
                    errors[name] = msg
                    self._mark_failure(name, msg)
                    self._update_stats(name, success=False, kind="last")
                    continue

                self._mark_success(name)
                self._update_stats(name, success=True, kind="last")
                return LastPriceResult(
                    price=float(price),
                    source=name,
                    tried=tried,
                    errors=errors,
                )
            except Exception as exc:
                msg = f"exception: {exc}"
                errors[name] = msg
                logger.debug(
                    "Provider '%s' failed in get_last(%s): %s",
                    name,
                    symbol,
                    exc,
                )
                self._mark_failure(name, msg)
                self._update_stats(name, success=False, kind="last")
                continue

        if self.strict:
            raise RuntimeError(
                f"All providers failed for symbol={symbol!r} in get_last. Errors={errors}"
            )

        logger.warning(
            "MarketDataRouter.get_last: all providers failed for symbol=%s; errors=%s",
            symbol,
            errors,
        )
        return LastPriceResult(price=None, source=None, tried=tried, errors=errors)


# ========================= Factory helpers =======================


def build_default_router(
    ib: Any | None = None,
    *,
    use_yahoo: bool = True,
    yahoo_auto_adjust: bool = False,
    prefer: Sequence[str] | None = None,
    allow_fallbacks: bool = True,
    strict: bool = False,
) -> MarketDataRouter:
    """
    Helper לחיבור מהיר בתוך המערכת שלך.

    בונה Router עם:
    - IBKRProvider (אם ib לא None)
    - YahooProvider (אופציונלי, כ-fallback)

    Parameters
    ----------
    ib : IB | None
        אובייקט IB פעיל (ib_insync.IB). אם None → לא נשתמש ב-IBKR.
    use_yahoo : bool
        אם True → נוסיף YahooProvider (אם yfinance מותקן).
    yahoo_auto_adjust : bool
        האם להשתמש במחירים מתוקנים ב-Yahoo (Adj Close).
    prefer : Sequence[str] | None
        סדר עדיפויות שמות ספקים (לדוגמה ["ibkr","yahoo"]).
        אם None → ייגזר מתוך priority של הספקים.
    allow_fallbacks : bool
        אם True → ננסה ספקים נוספים אם הראשון נכשל.
    strict : bool
        אם True + require_non_empty=True → ייזרק Exception אם הכל נכשל.

    Returns
    -------
    MarketDataRouter
    """
    providers: List[MarketDataProvider] = []

    # IBKR
    if ib is not None:
        try:
            providers.append(IBKRProvider(ib=ib))
            logger.info("build_default_router: IBKRProvider added.")
        except Exception as exc:
            logger.warning("build_default_router: failed to init IBKRProvider: %s", exc)

    # Yahoo
    if use_yahoo:
        try:
            providers.append(YahooProvider(auto_adjust=yahoo_auto_adjust))
            logger.info("build_default_router: YahooProvider added.")
        except Exception as exc:
            logger.warning("build_default_router: failed to init YahooProvider: %s", exc)

    if not providers:
        raise RuntimeError(
            "build_default_router: no providers could be initialized. "
            "Ensure at least IBKR or yfinance is available."
        )

    return MarketDataRouter(
        providers=providers,
        prefer=prefer,
        allow_fallbacks=allow_fallbacks,
        strict=strict,
    )


def describe_available_providers() -> pd.DataFrame:
    """
    עטיפה נוחה ל-available_providers_summary() מתוך data_providers.py.

    שימוש:
    -------
        df_prov = describe_available_providers()
        st.table(df_prov)  # בטאב קונפיג
    """
    return available_providers_summary()
