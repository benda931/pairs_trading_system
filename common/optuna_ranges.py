# common/optuna_ranges.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass(frozen=True)
class IntRange:
    low: int
    high: int
    step: int = 1

    def sanitize(self) -> "IntRange":
        if self.step <= 0:
            return IntRange(self.low, self.high, 1)
        if self.high < self.low:
            return IntRange(self.high, self.low, self.step)

        span = self.high - self.low
        k = span // self.step
        high2 = self.low + k * self.step
        # אם יצא ש-high2 == low כי הטווח קטן מה-step, עדיין חוקי, אבל כנראה לא רצוי:
        return IntRange(self.low, high2, self.step)

    def as_tuple(self) -> Tuple[int, int, int]:
        r = self.sanitize()
        return (r.low, r.high, r.step)


@dataclass(frozen=True)
class FloatRange:
    low: float
    high: float
    step: Optional[float] = None  # None => continuous
    ndigits: int = 12             # הגנה נגד floating drift

    def sanitize(self) -> "FloatRange":
        low, high = float(self.low), float(self.high)
        if high < low:
            low, high = high, low

        if self.step is None:
            return FloatRange(low, high, None, self.ndigits)

        step = float(self.step)
        if step <= 0:
            return FloatRange(low, high, None, self.ndigits)

        span = high - low
        k = math.floor(span / step + 1e-12)
        high2 = low + k * step
        high2 = round(high2, self.ndigits)
        low2 = round(low, self.ndigits)
        return FloatRange(low2, high2, step, self.ndigits)

    def as_tuple(self) -> Tuple[float, float, Optional[float]]:
        r = self.sanitize()
        return (r.low, r.high, r.step)
