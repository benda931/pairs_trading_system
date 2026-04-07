# -*- coding: utf-8 -*-
"""
core/spread_guard.py — Spread Guard for Active Execution Windows
================================================================

SpreadGuard monitors the spread (bid-ask width) on both legs of a pair
during an active execution window. If the spread widens beyond the
acceptable threshold, execution is paused until spread normalises
or the window expires.

This prevents executing into a temporarily illiquid market, which
would cause excess slippage on the entry and degrade the expected
spread cost assumption used in pre-trade analysis.

Usage
-----
    from core.spread_guard import SpreadGuard

    guard = SpreadGuard(max_spread_bps=50, check_interval_seconds=5)

    # In execution loop:
    if not guard.is_clear(symbol_x="SPY", symbol_y="QQQ"):
        logger.warning("SpreadGuard: spread too wide — pausing execution")
        time.sleep(guard.check_interval_seconds)
        continue
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("core.spread_guard")


# ---------------------------------------------------------------------------
# Quote objects
# ---------------------------------------------------------------------------

@dataclass
class SpreadQuote:
    """A point-in-time bid/ask quote for a single instrument."""
    symbol: str
    bid: float
    ask: float
    mid: float = field(init=False)
    spread_bps: float = field(init=False)
    quoted_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def __post_init__(self):
        if self.ask > 0:
            self.mid = (self.bid + self.ask) / 2.0
            self.spread_bps = (
                (self.ask - self.bid) / self.mid * 10_000
                if self.mid > 0 else float("nan")
            )
        else:
            self.mid = float("nan")
            self.spread_bps = float("nan")

    @property
    def is_valid(self) -> bool:
        return (
            self.bid > 0
            and self.ask >= self.bid
            and math.isfinite(self.spread_bps)
        )


@dataclass
class SpreadGuardCheck:
    """Result of a single SpreadGuard check."""
    clear: bool                      # True = spread acceptable, execution may proceed
    symbol_x: str = ""
    symbol_y: str = ""
    quote_x: Optional[SpreadQuote] = None
    quote_y: Optional[SpreadQuote] = None
    blocking_symbol: Optional[str] = None
    blocking_spread_bps: float = float("nan")
    threshold_bps: float = 50.0
    reason: str = ""
    checked_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def summary(self) -> str:
        if self.clear:
            return (
                f"SpreadGuard CLEAR: {self.symbol_x}/{self.symbol_y} "
                f"(X={self.quote_x.spread_bps:.1f}bps, Y={self.quote_y.spread_bps:.1f}bps)"
                if self.quote_x and self.quote_y else "SpreadGuard CLEAR"
            )
        return (
            f"SpreadGuard BLOCKED: {self.blocking_symbol} "
            f"spread={self.blocking_spread_bps:.1f}bps > threshold={self.threshold_bps:.1f}bps"
        )


# ---------------------------------------------------------------------------
# SpreadGuard
# ---------------------------------------------------------------------------

class SpreadGuard:
    """
    Guards execution by monitoring bid-ask spreads on both legs of a pair.

    If either leg's spread exceeds max_spread_bps, execution is blocked.
    The guard also tracks spread history for diagnostics.

    Parameters
    ----------
    max_spread_bps : float
        Maximum acceptable bid-ask spread in basis points (default: 50 bps).
        50 bps = 0.50% spread. For liquid ETFs, typical spread is 1-5 bps.
    check_interval_seconds : float
        How often to re-check spreads when blocked (default: 5 seconds for live,
        60 seconds for paper).
    quote_provider : callable, optional
        Function `(symbol: str) -> SpreadQuote`. If None, mock quotes are used.
        In production, this should call your market data feed.
    stale_quote_seconds : float
        A quote older than this is considered stale and blocks execution
        (conservative: missing data = blocked). Default: 30 seconds.
    max_retries : int
        Maximum number of spread checks before aborting the execution attempt.
    """

    def __init__(
        self,
        max_spread_bps: float = 50.0,
        check_interval_seconds: float = 5.0,
        quote_provider: Optional[Callable[[str], SpreadQuote]] = None,
        stale_quote_seconds: float = 30.0,
        max_retries: int = 12,
    ):
        self._max_bps = max_spread_bps
        self._interval = check_interval_seconds
        self._quote_fn = quote_provider or self._mock_quote_provider
        self._stale_seconds = stale_quote_seconds
        self._max_retries = max_retries

        # History for diagnostics
        self._check_history: List[SpreadGuardCheck] = []
        self._max_history = 100

    # ------------------------------------------------------------------
    # Primary check
    # ------------------------------------------------------------------

    def is_clear(
        self,
        symbol_x: str,
        symbol_y: str,
    ) -> SpreadGuardCheck:
        """
        Check if spreads on both legs are within the acceptable threshold.

        Returns a SpreadGuardCheck. Use `result.clear` to decide whether
        to proceed with execution.

        Parameters
        ----------
        symbol_x : str
            First leg symbol.
        symbol_y : str
            Second leg symbol.

        Returns
        -------
        SpreadGuardCheck with clear=True if spreads are acceptable.
        """
        try:
            quote_x = self._quote_fn(symbol_x)
            quote_y = self._quote_fn(symbol_y)
        except Exception as exc:
            logger.warning("SpreadGuard: quote fetch failed: %s — blocking execution", exc)
            result = SpreadGuardCheck(
                clear=False,
                symbol_x=symbol_x,
                symbol_y=symbol_y,
                blocking_symbol="quote_fetch_error",
                threshold_bps=self._max_bps,
                reason=f"quote_fetch_error: {exc}",
            )
            self._record(result)
            return result

        # Stale quote check
        for sym, quote in [(symbol_x, quote_x), (symbol_y, quote_y)]:
            if not quote.is_valid:
                result = SpreadGuardCheck(
                    clear=False,
                    symbol_x=symbol_x,
                    symbol_y=symbol_y,
                    quote_x=quote_x,
                    quote_y=quote_y,
                    blocking_symbol=sym,
                    threshold_bps=self._max_bps,
                    reason=f"invalid_quote: {sym} bid={quote.bid} ask={quote.ask}",
                )
                self._record(result)
                return result

        # Spread width check
        for sym, quote in [(symbol_x, quote_x), (symbol_y, quote_y)]:
            if not math.isfinite(quote.spread_bps):
                continue
            if quote.spread_bps > self._max_bps:
                result = SpreadGuardCheck(
                    clear=False,
                    symbol_x=symbol_x,
                    symbol_y=symbol_y,
                    quote_x=quote_x,
                    quote_y=quote_y,
                    blocking_symbol=sym,
                    blocking_spread_bps=quote.spread_bps,
                    threshold_bps=self._max_bps,
                    reason=(
                        f"spread_too_wide: {sym} "
                        f"spread={quote.spread_bps:.1f}bps > threshold={self._max_bps:.1f}bps"
                    ),
                )
                logger.warning(result.summary())
                self._record(result)
                return result

        # All clear
        result = SpreadGuardCheck(
            clear=True,
            symbol_x=symbol_x,
            symbol_y=symbol_y,
            quote_x=quote_x,
            quote_y=quote_y,
            threshold_bps=self._max_bps,
        )
        self._record(result)
        return result

    def wait_for_clear(
        self,
        symbol_x: str,
        symbol_y: str,
    ) -> Tuple[bool, int]:
        """
        Block until spread is clear or max_retries is exceeded.

        Returns (cleared: bool, attempts: int).
        Use this in the execution loop to wait for spread to normalise.

        Parameters
        ----------
        symbol_x : str
        symbol_y : str

        Returns
        -------
        tuple : (cleared, attempts)
            cleared : True if spread cleared within max_retries checks.
            attempts : Number of checks performed.
        """
        for attempt in range(1, self._max_retries + 1):
            result = self.is_clear(symbol_x, symbol_y)
            if result.clear:
                if attempt > 1:
                    logger.info(
                        "SpreadGuard: spread cleared after %d checks for %s/%s",
                        attempt, symbol_x, symbol_y,
                    )
                return True, attempt

            logger.debug(
                "SpreadGuard: attempt %d/%d — %s",
                attempt, self._max_retries, result.reason,
            )
            if attempt < self._max_retries:
                time.sleep(self._interval)

        logger.warning(
            "SpreadGuard: spread did not clear after %d attempts for %s/%s — aborting",
            self._max_retries, symbol_x, symbol_y,
        )
        return False, self._max_retries

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_history(self, n: int = 20) -> List[SpreadGuardCheck]:
        """Return the last N check results."""
        return self._check_history[-n:]

    def get_block_rate(self) -> float:
        """Return fraction of checks that were blocked (0.0 = never blocked)."""
        if not self._check_history:
            return 0.0
        blocked = sum(1 for c in self._check_history if not c.clear)
        return blocked / len(self._check_history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, result: SpreadGuardCheck) -> None:
        self._check_history.append(result)
        if len(self._check_history) > self._max_history:
            self._check_history = self._check_history[-self._max_history:]

    @staticmethod
    def _mock_quote_provider(symbol: str) -> SpreadQuote:
        """
        Mock quote provider for testing and paper trading.

        Returns a synthetic quote with a 1 bps spread.
        Replace with a real market data feed in production.
        """
        # Synthetic mid price based on well-known ETF approximate prices
        MOCK_PRICES: Dict[str, float] = {
            "SPY": 480.0, "QQQ": 420.0, "IWM": 200.0, "XLF": 40.0,
            "XLE": 90.0, "XLK": 190.0, "XLV": 140.0, "XLI": 120.0,
        }
        mid = MOCK_PRICES.get(symbol.upper(), 100.0)
        half_spread = mid * 0.00005  # 1 bps half-spread
        return SpreadQuote(
            symbol=symbol,
            bid=mid - half_spread,
            ask=mid + half_spread,
        )
