# -*- coding: utf-8 -*-
"""
common/error_telemetry.py — Structured Error Telemetry
=========================================================

Replaces silent `except Exception: pass` patterns across core/.

Anti-pattern being fixed (AP-5):
    try:
        ...
    except Exception:
        pass   # <-- bug hides here for weeks

Fixed pattern:
    from common.error_telemetry import log_exception, get_error_stats

    @log_exception(module="risk_analytics")
    def compute_var(...):
        ...

    # Or inline:
    try:
        ...
    except Exception as exc:
        telemetry.record("risk_analytics", exc, context={"pair": "XLY/XLC"})

Features:
- Per-module error counters
- Automatic alerting when error rate exceeds threshold
- Structured context preservation
- Never raises itself (telemetry failures don't cascade)

Usage:
    from common.error_telemetry import get_telemetry

    t = get_telemetry()
    t.record("signal_pipeline", exc, severity="WARNING")
    stats = t.get_stats()
    # {"signal_pipeline": {"count": 3, "last_seen": "...", "last_error": "..."}}
"""
from __future__ import annotations

import functools
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ErrorRecord:
    """One captured error with context."""
    module: str
    exception_type: str
    message: str
    ts_utc: str
    severity: str = "WARNING"
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleErrorStats:
    """Rolling error statistics for a single module."""
    module: str
    count: int = 0
    last_seen: str = ""
    last_error: str = ""
    last_severity: str = ""
    errors_last_hour: int = 0
    errors_last_day: int = 0


class ErrorTelemetry:
    """
    Thread-safe error telemetry collector.

    Records errors per module, tracks rates, and emits alerts when
    rates exceed thresholds.
    """

    # Alert thresholds — emit Telegram alert when exceeded
    ALERT_THRESHOLD_PER_HOUR = 20
    ALERT_THRESHOLD_PER_DAY = 100

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[str, list] = defaultdict(list)
        self._counts: Dict[str, int] = defaultdict(int)
        self._last_alert_ts: Dict[str, float] = {}
        self._alert_cool_down_seconds = 3600  # Don't re-alert same module for 1h

    def record(
        self,
        module: str,
        exception: BaseException,
        severity: str = "WARNING",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an exception occurrence.

        Never raises — telemetry failures don't cascade.
        """
        try:
            rec = ErrorRecord(
                module=module,
                exception_type=type(exception).__name__,
                message=str(exception)[:500],
                ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                severity=severity,
                context=dict(context) if context else {},
            )
            with self._lock:
                self._records[module].append(rec)
                # Keep last 1000 per module
                if len(self._records[module]) > 1000:
                    self._records[module] = self._records[module][-1000:]
                self._counts[module] += 1

            # Log with structured context
            logger.warning(
                "[%s/%s] %s: %s%s",
                module, severity, rec.exception_type, rec.message,
                f" ctx={context}" if context else "",
            )

            # Check alert threshold (throttled)
            self._maybe_alert(module)
        except Exception:
            # Telemetry failure: fall back to direct logger
            logger.debug("ErrorTelemetry.record failed", exc_info=True)

    def get_stats(self, module: Optional[str] = None) -> Dict[str, ModuleErrorStats]:
        """Return rolling stats for one module or all modules."""
        with self._lock:
            modules = [module] if module else list(self._records.keys())
            out = {}
            now = time.time()
            for mod in modules:
                records = self._records.get(mod, [])
                if not records:
                    continue
                last = records[-1]

                # Count errors in rolling windows
                one_hour_ago = now - 3600
                one_day_ago = now - 86400
                errors_1h = 0
                errors_1d = 0
                for r in reversed(records):
                    try:
                        rec_time = datetime.fromisoformat(r.ts_utc.replace("Z", "+00:00")).timestamp()
                        if rec_time < one_day_ago:
                            break
                        errors_1d += 1
                        if rec_time >= one_hour_ago:
                            errors_1h += 1
                    except Exception:
                        continue

                out[mod] = ModuleErrorStats(
                    module=mod,
                    count=self._counts[mod],
                    last_seen=last.ts_utc,
                    last_error=f"{last.exception_type}: {last.message}",
                    last_severity=last.severity,
                    errors_last_hour=errors_1h,
                    errors_last_day=errors_1d,
                )
            return out

    def reset(self, module: Optional[str] = None) -> None:
        """Reset counters (useful for tests)."""
        with self._lock:
            if module:
                self._records.pop(module, None)
                self._counts.pop(module, None)
            else:
                self._records.clear()
                self._counts.clear()

    def _maybe_alert(self, module: str) -> None:
        """Emit alert if error rate exceeds threshold (throttled)."""
        try:
            now = time.time()
            last_alert = self._last_alert_ts.get(module, 0)
            if now - last_alert < self._alert_cool_down_seconds:
                return

            stats = self.get_stats(module).get(module)
            if stats is None:
                return

            if (stats.errors_last_hour >= self.ALERT_THRESHOLD_PER_HOUR or
                    stats.errors_last_day >= self.ALERT_THRESHOLD_PER_DAY):
                self._last_alert_ts[module] = now
                # Best-effort Telegram alert
                try:
                    from core.alerts import alert_system
                    alert_system(
                        "error_telemetry",
                        "WARNING",
                        f"Module {module} error rate: "
                        f"{stats.errors_last_hour}/h, {stats.errors_last_day}/d. "
                        f"Last: {stats.last_error}",
                    )
                except Exception:
                    pass
        except Exception:
            pass


# ── Module singleton ──────────────────────────────────────────────

_TELEMETRY: Optional[ErrorTelemetry] = None
_TELEMETRY_LOCK = threading.Lock()


def get_telemetry() -> ErrorTelemetry:
    """Return process-wide telemetry singleton."""
    global _TELEMETRY
    if _TELEMETRY is None:
        with _TELEMETRY_LOCK:
            if _TELEMETRY is None:
                _TELEMETRY = ErrorTelemetry()
    return _TELEMETRY


# ── Decorator for explicit error logging ─────────────────────────

def log_exception(
    module: str,
    severity: str = "WARNING",
    reraise: bool = False,
    default: Any = None,
) -> Callable[[F], F]:
    """
    Decorator that wraps a function with structured exception logging.

    Replaces `try: ... except Exception: pass` pattern.

    Parameters
    ----------
    module : str
        Logical module name for telemetry grouping.
    severity : str
        WARNING | ERROR | CRITICAL.
    reraise : bool
        If True, re-raise after logging. Default: swallow + return `default`.
    default : Any
        Value returned on exception when reraise=False.

    Example:
        @log_exception(module="cycle_detector", default={})
        def analyze(spread): ...
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                get_telemetry().record(
                    module=module,
                    exception=exc,
                    severity=severity,
                    context={"func": fn.__name__},
                )
                if reraise:
                    raise
                return default
        return wrapper  # type: ignore[return-value]
    return decorator
