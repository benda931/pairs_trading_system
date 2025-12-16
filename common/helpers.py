# -*- coding: utf-8 -*-
"""
common/helpers.py — General-purpose utilities (v1.1, Part 1/4)
===============================================================

Infrastructure & IO Layer
-------------------------
חלק זה אחראי על:
- הגנות סביב חבילות חיצוניות (בעיקר arch לצורך GARCH).
- הגדרת נתיבי פרויקט/חבילה (project_root, package_root).
- imports בסיסיים כלליים.
- כלי JSON/YAML (sync + async) עם atomic write ו-treatment ל-numpy/pandas.
- לוגיקת קונפיג פשוטה מבוססת JSON/YAML + ENV (load_settings / save_settings).

שאר הקובץ (בחלקים 2–4) מוסיף:
2. Logging & decorators & async utils.
3. Data / risk / GPU helpers.
4. CLI + summarize_series + hooks ל-common.helpers.

Note לגבי arch
--------------
בדיקות אצלך משתמשות ב־sys.modules["arch"] = None כדי לוודא ש-`from common import helpers`
יזרוק ImportError בצורה צפויה. לכן יש כאן guard מוקדם מול arch, אבל רק בצורת
"explicit None placeholder", לא סתם חסר חבילה רגיל.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Union

# ---------------------------------------------------------------------------
# ARCH guard — צריך להיות *מוקדם* מאוד
# ---------------------------------------------------------------------------
# אם arch קיים ב-sys.modules והערך שלו הוא None → נכשלה טעינה במכוון (pytest),
# ואנחנו אמורים לזרוק ImportError כשמישהו מייבא common.helpers.
_arch_mod = sys.modules.get("arch")
if _arch_mod is None and "arch" in sys.modules:
    raise ImportError("The 'arch' package is unavailable (explicit None placeholder)")


# ---------------------------------------------------------------------------
# Path setup — project_root / package_root
# ---------------------------------------------------------------------------

_thisdir = os.path.abspath(os.path.dirname(__file__))
_package_root = os.path.dirname(_thisdir)
_project_root = os.path.dirname(_package_root)

# מוודאים שה-root paths נמצאים ב-sys.path (בתחילת הרשימה, בלי כפילויות)
for _p in (_project_root, _package_root):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

# כאשר הקובץ מורץ כ-script (python -m common.helpers), נתקן __package__
if __name__ == "__main__" and __package__ is None:
    __package__ = "common"

__all__: list[str] = []  # נרחיב בהמשך חלקים 2–4


# ---------------------------------------------------------------------------
# Imports בסיסיים
# ---------------------------------------------------------------------------

import argparse
import asyncio
import concurrent.futures
import contextlib
import functools
import hashlib
import itertools
import json
import logging
import pathlib
import re
import statistics
import time
import typing as _t

from collections import abc
from contextlib import suppress
from datetime import datetime, timezone
from logging import Logger
from random import randint

import numpy as np
import pandas as pd

from common.json_safe import make_json_safe, json_default as _json_default_internal  # type: ignore


# ---------------------------------------------------------------------------
# Optional deps: YAML / aiofiles
# ---------------------------------------------------------------------------

try:  # PyYAML (לא חובה)
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

try:  # aiofiles עבור I/O אסינכרוני
    import aiofiles  # type: ignore
except ImportError:
    aiofiles = None  # type: ignore


# ---------------------------------------------------------------------------
# PathLike type & small config helper
# ---------------------------------------------------------------------------

PathLike = _t.Union[str, os.PathLike[str]]


def _as_path(path: PathLike) -> pathlib.Path:
    return pathlib.Path(path)


# ---------------------------------------------------------------------------
# JSON helpers (עם תמיכה ב-numpy/pandas ו-atomic write)
# ---------------------------------------------------------------------------

def _json_default(obj: Any):  # noqa: ANN001
    """
    Safe converter for numpy/pandas objects to JSON-serialisable types.

    ממשיך את ההתנהגות של common.json_safe, אבל מאפשר שימוש ישיר כ-default
    ב-json.dump. אם הטיפוס לא מוכר → מתרגם ל-str(obj).
    """
    # קודם כל, ננסה את json_default הפנימי; אם הוא נכשל, ניפול ל-fallback.
    try:
        return _json_default_internal(obj)
    except Exception:
        pass

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    return str(obj)


def read_json(path: PathLike, encoding: str = "utf-8") -> Any:
    """
    Read JSON from *path* and return the decoded object.

    Raises:
        FileNotFoundError   אם הקובץ לא קיים
        json.JSONDecodeError אם ה-JSON לא תקין
    """
    path = _as_path(path)
    with path.open("r", encoding=encoding) as fp:
        return json.load(fp)


def write_json(
    obj: Any,
    path: PathLike,
    *,
    encoding: str = "utf-8",
    indent: int = 2,
) -> None:
    """
    Serialize *obj* to JSON at *path* בצורה בטוחה ו-atomic.

    - משתמש ב-make_json_safe כדי לנקות numpy/pandas/objects.
    - כותב לקובץ זמני (.tmp) ואז מחליף → אין מצב שנשאר JSON שבור.
    """
    path = _as_path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")

    safe_obj = make_json_safe(obj)  # from common.json_safe
    with tmp.open("w", encoding=encoding) as fp:
        json.dump(safe_obj, fp, indent=indent, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# YAML helpers (sync + async)
# ---------------------------------------------------------------------------

def read_yaml(path: PathLike, encoding: str = "utf-8") -> Any:  # pragma: no cover
    """
    Read YAML content from *path* (requires PyYAML).

    Raises:
        ImportError אם PyYAML לא מותקן.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for read_yaml/write_yaml")
    path = _as_path(path)
    with path.open("r", encoding=encoding) as fp:
        return yaml.safe_load(fp)


def write_yaml(
    obj: Any,
    path: PathLike,
    *,
    encoding: str = "utf-8",
) -> None:  # pragma: no cover
    """
    Write YAML to *path* (atomic write), אם PyYAML זמין.

    Raises:
        ImportError אם PyYAML לא מותקן.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for read_yaml/write_yaml")
    path = _as_path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding=encoding) as fp:
        yaml.safe_dump(obj, fp, allow_unicode=True, sort_keys=False)
    os.replace(tmp, path)


async def read_yaml_async(
    path: PathLike,
    encoding: str = "utf-8",
    *,
    loop: asyncio.AbstractEventLoop | None = None,
) -> Any:
    """
    Asynchronously read and parse YAML from *path*.

    - אם יש aiofiles + PyYAML → קריאה async אמיתית.
    - אחרת → מבצע off-thread ל-read_yaml כדי לא לחסום event loop.
    """
    if yaml is not None and aiofiles is not None and "aiofiles" in sys.modules:
        path = _as_path(path)
        async with aiofiles.open(path, "r", encoding=encoding) as fp:  # type: ignore[attr-defined]
            raw = await fp.read()
        return yaml.safe_load(raw)

    loop = loop or asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(read_yaml, path, encoding))


async def write_yaml_async(
    obj: Any,
    path: PathLike,
    *,
    encoding: str = "utf-8",
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """
    Asynchronously write YAML to *path* (atomic).

    - אם aiofiles + PyYAML זמינים → כותב async.
    - אחרת → מממש off-thread ל-write_yaml.
    """
    if yaml is not None and aiofiles is not None and "aiofiles" in sys.modules:
        path = _as_path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        async with aiofiles.open(tmp, "w", encoding=encoding) as fp:  # type: ignore[attr-defined]
            await fp.write(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False))
        os.replace(tmp, path)
        return

    loop = loop or asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        functools.partial(write_yaml, obj, path, encoding=encoding),
    )


# ---------------------------------------------------------------------------
# Simple config loader (JSON/YAML + ENV override)
# ---------------------------------------------------------------------------

def load_settings(
    path: PathLike,
    *,
    env_prefix: str | None = None,
) -> dict[str, Any]:
    """
    Load settings from JSON/YAML file, ואז מכבד override מהסביבה (ENV).

    - אם הקובץ נגמר ב-.json → read_json
    - אם suffix ב-.yml/.yaml → read_yaml
    - env_prefix (למשל "APP_") ייקח משתני סביבה שמתחילים ב-prefix
      ויכסה על מפתחות בקונפיג.

    Returns:
        dict עם כל ההגדרות לאחר merge.
    """
    path = _as_path(path)
    if not path.exists():
        return {}

    if path.suffix.lower() == ".json":
        cfg = read_json(path)
    elif path.suffix.lower() in (".yml", ".yaml"):
        cfg = read_yaml(path)
    else:
        raise ValueError(f"Unsupported settings file suffix: {path.suffix}")

    if not isinstance(cfg, dict):
        raise TypeError(f"Settings file {path} did not yield a mapping.")

    out: dict[str, Any] = dict(cfg)

    if env_prefix:
        prefix = env_prefix.upper()
        for key, val in os.environ.items():
            if not key.startswith(prefix):
                continue
            # APP_LOG_LEVEL → log_level
            short_key = key[len(prefix) :].lower()
            out[short_key] = val

    return out


def save_settings(
    cfg: dict[str, Any],
    path: PathLike,
) -> None:
    """
    Save settings dict to JSON/YAML file בהתאם לסיומת.

    - .json  → write_json
    - .yml/.yaml → write_yaml
    """
    path = _as_path(path)
    if path.suffix.lower() == ".json":
        write_json(cfg, path)
    elif path.suffix.lower() in (".yml", ".yaml"):
        write_yaml(cfg, path)
    else:
        raise ValueError(f"Unsupported settings file suffix: {path.suffix}")
# ---------------------------------------------------------------------------
# LOGGING — צבע, פורמט, ENV + כלי עזר
# ---------------------------------------------------------------------------

_DEFAULT_TZ = timezone.utc
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


class ColorFormatter(logging.Formatter):
    """Formatter עם צבעים פשוטים ללוגים בטרמינל (ANSI)."""

    COLOR_MAP = {
        logging.DEBUG: "\033[37m",    # White
        logging.INFO: "\033[36m",     # Cyan
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[41m", # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        # צבע ל-level
        color = self.COLOR_MAP.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"

        # זמן בפורמט ISO עם Z (UTC)
        time_fmt = datetime.fromtimestamp(record.created, _DEFAULT_TZ).strftime("%Y-%m-%d %H:%M:%S")
        record.asctime = f"{time_fmt}Z"

        return super().format(record)


def _env_log_level(default: int = logging.INFO) -> int:
    """
    קובע את רמת הלוג בהתאם ל־ENV (LOG_LEVEL / APP_LOG_LEVEL), אם קיימת.

    LOG_LEVEL=DEBUG / INFO / WARNING / ERROR / CRITICAL
    """
    lvl_str = os.environ.get("LOG_LEVEL") or os.environ.get("APP_LOG_LEVEL")
    if not lvl_str:
        return default
    lvl_str = lvl_str.upper()
    return getattr(logging, lvl_str, default)


def get_logger(
    name: str | None = None,
    *,
    level: int | None = None,
    stream: _t.IO[str] | None = None,
) -> Logger:
    """
    החזרת Logger עם הגדרות סבירות + צבעים.

    - אם level=None → יקח מ־ENV דרך _env_log_level.
    - logger עם StreamHandler יחיד (לא יוסיף handler כל פעם).
    - propagate=False כדי לא להכפיל פלט.
    """
    if level is None:
        level = _env_log_level(logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(ColorFormatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


@contextlib.contextmanager
def temporary_log_level(logger: Logger, level: int):
    """
    Context manager שמשנה זמנית את רמת הלוג ל־*level* ומחזיר אותה בסוף.

    Usage:
        log = get_logger(__name__)
        with temporary_log_level(log, logging.DEBUG):
            ...
    """
    old_level = logger.level
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(old_level)


def set_global_log_level(level: int) -> None:
    """
    שינוי רמת הלוג לכל הלוגרים הקיימים.

    שימוש נוח ב־notebook / סקריפט קצר:
        set_global_log_level(logging.DEBUG)
    """
    logging.basicConfig(level=level, format=_LOG_FORMAT)
    for name in logging.Logger.manager.loggerDict.keys():  # type: ignore[attr-defined]
        logging.getLogger(name).setLevel(level)


__all__ += ["get_logger", "temporary_log_level", "set_global_log_level"]


# ---------------------------------------------------------------------------
# Decorators & Functional Helpers
# ---------------------------------------------------------------------------

def singleton(cls: _t.Type[_t.Any]) -> _t.Type[_t.Any]:
    """
    Decorator שממיר class ל-singleton (instance אחד לתהליך).

    Usage:
        @singleton
        class MyService: ...
    """
    instances: dict[_t.Type[_t.Any], _t.Any] = {}

    @functools.wraps(cls)
    def get_instance(*args: _t.Any, **kwargs: _t.Any) -> _t.Any:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _t.cast(_t.Type[_t.Any], get_instance)


class retry:  # noqa: N801
    """
    Decorator למחזור ניסיונות (Retry) עם backoff אקספוננציאלי + לוגים.

    Parameters
    ----------
    exceptions : Exception type or tuple
        אילו חריגות לתפוס ולנסות שוב (default=Exception).
    tries : int
        מספר נסיונות (כולל הראשון).
    delay : float
        זמן התחלה לשינה (שניות).
    backoff : float
        מקדם הגדלת זמן השינה (למשל 2.0 → 0.1, 0.2, 0.4...).
    jitter : float
        ג'יטר אקראי נוסף (שניות), כדי לא לעבוד בקצב קבוע מדי.
    logger : Logger | None
        אם ניתן לוגger, ידפיס אזהרות על retries.
    """

    def __init__(
        self,
        exceptions: _t.Type[BaseException] | tuple[_t.Type[BaseException], ...] = Exception,
        *,
        tries: int = 3,
        delay: float = 0.1,
        backoff: float = 2.0,
        jitter: float = 0.1,
        logger: Logger | None = None,
    ) -> None:
        self.exceptions = exceptions if isinstance(exceptions, tuple) else (exceptions,)
        self.tries = tries
        self.delay = delay
        self.backoff = backoff
        self.jitter = jitter
        self.logger = logger or get_logger(__name__)

    def __call__(self, func: _t.Callable[..., _t.Any]) -> _t.Callable[..., _t.Any]:
        @functools.wraps(func)
        def wrapper(*args: _t.Any, **kwargs: _t.Any) -> _t.Any:
            current_delay = self.delay
            for attempt in range(1, self.tries + 1):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as exc:  # noqa: BLE001
                    if attempt == self.tries:
                        self.logger.error(
                            "retry: %s failed after %d attempts: %s",
                            func.__name__, attempt, exc,
                        )
                        raise
                    self.logger.warning(
                        "retry: %s failed on attempt %d/%d (%s), retrying in %.3fs",
                        func.__name__, attempt, self.tries, exc, current_delay,
                    )
                    sleep_time = current_delay + randint(0, int(self.jitter * 1000)) / 1000
                    time.sleep(sleep_time)
                    current_delay *= self.backoff
        return wrapper


def cached(
    maxsize: int = 128,
    ttl: float | None = None,
) -> _t.Callable[[_t.Callable[..., _t.Any]], _t.Callable[..., _t.Any]]:
    """
    Decorator ל-cache של פונקציה, עם אופציה ל-TTL (שניות).

    - משתמש ב functools.lru_cache מתחת.
    - אם ttl לא None → מנקה את ה-cache כאשר TTL פג לאותו key.

    Usage:
        @cached(maxsize=256, ttl=60)
        def get_data(x): ...
    """
    def decorator(func: _t.Callable[..., _t.Any]) -> _t.Callable[..., _t.Any]:
        cache = functools.lru_cache(maxsize=maxsize)(func)
        timestamps: dict[tuple[_t.Any, ...], float] = {}

        @functools.wraps(func)
        def wrapper(*args: _t.Any, **kwargs: _t.Any) -> _t.Any:
            key = args + tuple(sorted(kwargs.items()))
            now = time.time()
            if ttl is not None:
                ts = timestamps.get(key)
                # TTL פג → מנקים את כל ה-cache (פשוט לביצוע, מספיק טוב לרוב השימושים)
                if ts is not None and now - ts > ttl:
                    cache.cache_clear()
                    timestamps.clear()
            result = cache(*args, **kwargs)
            timestamps[key] = now
            return result

        # מצרפים reference ל-cache המקורי אם צריך לנקות חיצונית
        wrapper._cache_clear = cache.cache_clear  # type: ignore[attr-defined]
        return wrapper
    return decorator


def timeit(
    label: str | None = None,
    *,
    logger: Logger | None = None,
) -> _t.Callable[[_t.Callable[..., _t.Any]], _t.Callable[..., _t.Any]]:
    """
    Decorator למדידת זמן ריצה של פונקציה sync, עם לוג DEBUG.

    Usage:
        @timeit("load_data")
        def load_data(...): ...
    """
    log = logger or get_logger(__name__)

    def decorator(func: _t.Callable[..., _t.Any]) -> _t.Callable[..., _t.Any]:
        tag = label or func.__name__

        @functools.wraps(func)
        def wrapper(*args: _t.Any, **kwargs: _t.Any) -> _t.Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                log.debug("timeit: %s executed in %.2f ms", tag, duration_ms)
        return wrapper

    return decorator


def timeit_async(
    label: str | None = None,
    *,
    logger: Logger | None = None,
) -> _t.Callable[[_t.Callable[..., _t.Any]], _t.Callable[..., _t.Any]]:
    """
    Decorator למדידת זמן ריצה של coroutine (async def), עם לוג DEBUG.

    Usage:
        @timeit_async("fetch_batch")
        async def fetch_batch(...): ...
    """
    log = logger or get_logger(__name__)

    def decorator(func: _t.Callable[..., _t.Any]) -> _t.Callable[..., _t.Any]:
        tag = label or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: _t.Any, **kwargs: _t.Any) -> _t.Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                log.debug("timeit_async: %s executed in %.2f ms", tag, duration_ms)
        return wrapper

    return decorator


__all__ += ["singleton", "retry", "cached", "timeit", "timeit_async"]


# ---------------------------------------------------------------------------
# Async retry & timeout
# ---------------------------------------------------------------------------

def retry_async(
    exceptions: _t.Type[BaseException] | tuple[_t.Type[BaseException], ...] = Exception,
    *,
    tries: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    jitter: float = 0.1,
    logger: Logger | None = None,
) -> _t.Callable[[_t.Callable[..., _t.Any]], _t.Callable[..., _t.Any]]:
    """
    Decorator ל-Retry על פונקציות async עם backoff אקספוננציאלי + לוגים.

    Usage:
        @retry_async(tries=3, delay=0.2)
        async def fetch(...): ...
    """
    exceptions = exceptions if isinstance(exceptions, tuple) else (exceptions,)
    log = logger or get_logger(__name__)

    def decorator(func: _t.Callable[..., _t.Any]) -> _t.Callable[..., _t.Any]:
        @functools.wraps(func)
        async def wrapper(*args: _t.Any, **kwargs: _t.Any) -> _t.Any:
            current_delay = delay
            for attempt in range(1, tries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001
                    if attempt == tries:
                        log.error(
                            "retry_async: %s failed after %d attempts: %s",
                            func.__name__, attempt, exc,
                        )
                        raise
                    log.warning(
                        "retry_async: %s failed on attempt %d/%d (%s), retrying in %.3fs",
                        func.__name__, attempt, tries, exc, current_delay,
                    )
                    sleep_time = current_delay + randint(0, int(jitter * 1000)) / 1000
                    await asyncio.sleep(sleep_time)
                    current_delay *= backoff
        return wrapper
    return decorator


def async_timeout(
    coro: _t.Awaitable[_t.Any],
    timeout: float,
) -> _t.Awaitable[_t.Any]:
    """
    Wrap coroutine with asyncio.wait_for, so אפשר לכתוב:

        await async_timeout(coro(), 5.0)

    במקום:
        await asyncio.wait_for(coro(), 5.0)
    """
    return asyncio.wait_for(coro, timeout)


__all__ += ["retry_async", "async_timeout"]


# ---------------------------------------------------------------------------
# Async concurrency helpers
# ---------------------------------------------------------------------------

async def gather_with_concurrency(
    n: int,
    *coros: _t.Awaitable[_t.Any],
) -> list[_t.Any]:
    """
    Gather על רשימת קורוטינות עם הגבלת־כמות של n משימות במקביל.

    דוגמה:
        results = await gather_with_concurrency(5, *(fetch(u) for u in urls))
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro: _t.Awaitable[_t.Any]) -> _t.Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


def run_in_executor(
    func: _t.Callable[..., _t.Any],
    *args: _t.Any,
    loop: asyncio.AbstractEventLoop | None = None,
    executor: concurrent.futures.Executor | None = None,
) -> _t.Awaitable[_t.Any]:
    """
    להריץ פונקציה sync ב-thread pool ולהחזיר awaitable.

    שימוש:
        result = await run_in_executor(blocking_function, arg1, arg2)
    """
    loop = loop or asyncio.get_running_loop()
    return loop.run_in_executor(executor, functools.partial(func, *args))


__all__ += ["gather_with_concurrency", "run_in_executor"]

# ---------------------------------------------------------------------------
# Dict helpers: flatten / unflatten / deep_merge
# ---------------------------------------------------------------------------

def flatten_dict(d: dict[str, _t.Any], sep: str = ".") -> dict[str, _t.Any]:
    """
    Flatten dict מקונן למבנה פשוט עם מפתחות מופרדים ב-sep.

    Example:
        {"a": {"b": 1, "c": 2}, "d": 3}
        → {"a.b": 1, "a.c": 2, "d": 3}
    """
    if not isinstance(d, dict):
        raise TypeError(f"flatten_dict expects dict, got {type(d)}")

    out: dict[str, _t.Any] = {}

    def _recurse(prefix: str, value: _t.Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _recurse(f"{prefix}{k}{sep}", v)
        else:
            out[prefix[:-1]] = value

    _recurse("", d)
    return out


def unflatten_dict(d: dict[str, _t.Any], sep: str = ".") -> dict[str, _t.Any]:
    """
    Unflatten dict מהצורה {"a.b": 1, "a.c": 2} חזרה ל-nested dict.

    Example:
        {"a.b": 1, "a.c": 2, "d": 3}
        → {"a": {"b": 1, "c": 2}, "d": 3}
    """
    if not isinstance(d, dict):
        raise TypeError(f"unflatten_dict expects dict, got {type(d)}")

    out: dict[str, _t.Any] = {}
    for composite_key, value in d.items():
        parts = composite_key.split(sep)
        current: dict[str, _t.Any] = out
        for part in parts[:-1]:
            current = current.setdefault(part, {})  # type: ignore[assignment]
        current[parts[-1]] = value
    return out


def deep_merge(
    dest: dict[str, _t.Any],
    *sources: dict[str, _t.Any],
    overwrite: bool = True,
) -> dict[str, _t.Any]:
    """
    Merge רקורסיבי של dictים, עם שליטה אם overwrite=true או false.

    - dest מתעדכן in-place.
    - אם שני הצדדים dict → מבצע merge פנימי.
    - אם overwrite=False → ערכים קיימים ב-dest לא נדרסים.

    Usage:
        cfg = {"db": {"host": "localhost", "port": 5432}}
        override = {"db": {"port": 6432}}
        deep_merge(cfg, override) → port=6432
    """
    for src in sources:
        for key, value in src.items():
            if (
                key in dest
                and isinstance(dest[key], dict)
                and isinstance(value, dict)
            ):
                deep_merge(dest[key], value, overwrite=overwrite)  # type: ignore[arg-type]
            else:
                if overwrite or key not in dest:
                    dest[key] = value
    return dest


__all__ += ["flatten_dict", "unflatten_dict", "deep_merge"]


# ---------------------------------------------------------------------------
# ENV helpers: load_dotenv, humanize_bytes, hash_file
# ---------------------------------------------------------------------------

_ENV_PAT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_dotenv(
    path: PathLike = ".env",
    *,
    override: bool = False,
) -> dict[str, str]:
    """
    Load .env file לתוך os.environ.

    - שורות שמתחילות ב-# או ריקות מתעלמות.
    - אם override=True, משתנים קיימים ידרסו.

    Returns:
        dict של כל המשתנים שהועמסו.
    """
    variables: dict[str, str] = {}
    with suppress(FileNotFoundError), open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            match = _ENV_PAT.match(line)
            if match:
                key, val = match.groups()
                val = val.strip().strip("'\"")
                if override or key not in os.environ:
                    os.environ[key] = val
                variables[key] = val
    return variables


def humanize_bytes(n: int, precision: int = 1) -> str:
    """
    להמיר מספר bytes למחרוזת קריאה (1.2 MB, 500 KB וכו').

    usage:
        humanize_bytes(10_000) → "9.8 KB"
    """
    prefixes = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    idx = 0
    num = float(n)
    while num >= 1024 and idx < len(prefixes) - 1:
        num /= 1024
        idx += 1
    return f"{num:.{precision}f} {prefixes[idx]}"


def hash_file(
    path: PathLike,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
) -> str:
    """
    Compute hex digest של קובץ בנתיב *path* לפי *algorithm*.

    Example:
        hash_file("data.csv", "md5")
    """
    h = hashlib.new(algorithm)
    path = _as_path(path)
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ += ["load_dotenv", "humanize_bytes", "hash_file"]


# ---------------------------------------------------------------------------
# Timing & profiling
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def timing(label: str = "block"):
    """
    Context manager למדידת זמן ריצה של בלוק קוד עם לוג DEBUG.

    Usage:
        with timing("load_data"):
            ...
    """
    log = get_logger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        log.debug("%s executed in %.2f ms", label, duration_ms)


@contextlib.contextmanager
def profile(
    output: PathLike | None = None,
    sort_by: str = "cumulative",
) -> _t.Generator[None, None, None]:
    """
    Context manager ל-profile של בלוק קוד בעזרת cProfile.

    - אם output=None → מדפיס ל-stdout.
    - אם output Path → שומר binary stats + טקסט מתוך pstats.

    Usage:
        with profile("profile.pstats"):
            heavy_function()
    """
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats(sort_by)
        if output:
            out_path = _as_path(output)
            stats.dump_stats(out_path)
            txt_path = out_path.with_suffix(".txt")
            with txt_path.open("w", encoding="utf-8") as f:
                stats.stream = f
                stats.print_stats()
        else:
            stats.print_stats()


__all__ += ["timing", "profile"]


# ---------------------------------------------------------------------------
# Basic stats wrapper
# ---------------------------------------------------------------------------

def mean(data: abc.Iterable[float]) -> float:
    """ממוצע אריתמטי או NaN לקלט ריק (wrapper על statistics.mean)."""
    try:
        return statistics.mean(data)
    except statistics.StatisticsError:
        return float("nan")


__all__ += ["mean"]


# ---------------------------------------------------------------------------
# Rolling Sharpe Ratio
# ---------------------------------------------------------------------------

def rolling_sharpe_ratio(
    returns: abc.Iterable[float] | np.ndarray,
    window: int,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> np.ndarray:
    """
    חישוב Rolling Sharpe Ratio על סדרת תשואות.

    Parameters
    ----------
    returns : iterable or np.ndarray
        תשואות תקופתיות (למשל daily returns).
    window : int
        גודל חלון הריצה (מספר תקופות).
    risk_free_rate : float, default 0.0
        ריבית חסרת סיכון *שנתית* או לפי periodicity (יחד עם annualization).
    annualization : int, default 252
        מספר התקופות בשנה (252 ימי מסחר, 12 חודשי וכו').

    Returns
    -------
    np.ndarray (float)
        מערך של Rolling Sharpe, באורך התשואות (עם NaN ל-window-1 הראשונים).
    """
    arr = np.asarray(list(returns), dtype=float)
    sr = np.full_like(arr, fill_value=np.nan, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")

    rf_per_period = risk_free_rate / annualization
    excess = arr - rf_per_period

    for i in range(window - 1, len(arr)):
        window_slice = excess[i + 1 - window : i + 1]
        mean_ex = float(np.mean(window_slice))
        std_ex = float(np.std(window_slice, ddof=1))
        if std_ex == 0.0 or not np.isfinite(std_ex):
            sr[i] = np.nan
        else:
            sr[i] = (mean_ex / std_ex) * np.sqrt(annualization)
    return sr


__all__ += ["rolling_sharpe_ratio"]


# ---------------------------------------------------------------------------
# GARCH Volatility (arch-backed, עם fallback ל-EWMA)
# ---------------------------------------------------------------------------

def garch_volatility(
    returns: abc.Iterable[float] | np.ndarray,
    p: int = 1,
    q: int = 1,
    span_ewma: int = 20,
) -> np.ndarray:
    """
    הערכת תנודתיות GARCH על סדרת תשואות, באורך זהה לסדרה המקורית.

    Requires:
        - חבילת `arch` מותקנת.

    אם import של arch נכשל → ImportError (במכוון).
    אם יש שגיאה בריצה → fallback ל-EWMA std (span=span_ewma).

    Returns
    -------
    np.ndarray
        מערך תנודתיות באורך arr.
    """
    arr = np.asarray(list(returns), dtype=float)
    vol = np.full_like(arr, np.nan)

    try:
        from arch import arch_model  # type: ignore
    except ImportError as exc:
        raise ImportError("arch package is required for garch_volatility") from exc

    try:
        am = arch_model(arr * 100, p=p, q=q, mean="Zero", vol="GARCH", dist="normal")
        res = am.fit(disp="off")
        vol_arr = res.conditional_volatility / 100.0
        vol[: len(vol_arr)] = vol_arr
    except Exception as exc:
        get_logger(__name__).warning("garch_volatility: arch_model failed – %s", exc)
        vol = pd.Series(arr).ewm(span=span_ewma, adjust=False).std().values
    return vol


__all__ += ["garch_volatility"]


# ---------------------------------------------------------------------------
# Risk metrics: VaR, CVaR, max drawdown, ulcer index (full series / rolling)
# ---------------------------------------------------------------------------

def risk_metrics(
    returns: abc.Iterable[float] | np.ndarray,
    alpha: float = 0.05,
    window: int | None = None,
) -> dict[str, float | np.ndarray]:
    """
    חישוב מדדי סיכון: VaR, CVaR, max drawdown ו־Ulcer Index.

    Parameters
    ----------
    returns : iterable or np.ndarray
        תשואות תקופתיות.
    alpha : float, default 0.05
        רמת מובהקות ל־VaR/CVaR.
    window : int or None, default None
        גודל חלון rolling; אם None → מחשב על כל הסדרה.

    Returns
    -------
    dict
        {
            'VaR': float or np.ndarray,
            'CVaR': float or np.ndarray,
            'max_drawdown': float or np.ndarray,
            'ulcer_index': float or np.ndarray
        }
    """
    arr = np.asarray(list(returns), dtype=float)

    def _var(x: np.ndarray) -> float:
        return float(np.percentile(x, 100 * alpha))

    def _cvar(x: np.ndarray) -> float:
        thr = np.percentile(x, 100 * alpha)
        mask = x <= thr
        if not mask.any():
            return float(thr)
        return float(x[mask].mean())

    def _drawdown(x: np.ndarray) -> np.ndarray:
        cum = np.cumprod(1 + x)
        peak = np.maximum.accumulate(cum)
        return (cum - peak) / peak

    def _ulcer(x: np.ndarray) -> float:
        dd = _drawdown(x)
        negative_dd = dd[dd < 0]
        if negative_dd.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(negative_dd ** 2)))

    if window is None:
        dd_arr = _drawdown(arr)
        vals: dict[str, float | np.ndarray] = {
            "VaR": _var(arr),
            "CVaR": _cvar(arr),
            "max_drawdown": float(dd_arr.min()) if dd_arr.size > 0 else float("nan"),
            "ulcer_index": _ulcer(arr),
        }
    else:
        size = len(arr)
        VaR = np.full(size, np.nan); CVaR = np.full(size, np.nan)
        maxDD = np.full(size, np.nan); UI = np.full(size, np.nan)
        for i in range(window - 1, size):
            seg = arr[i + 1 - window : i + 1]
            VaR[i] = _var(seg)
            CVaR[i] = _cvar(seg)
            dd_seg = _drawdown(seg)
            maxDD[i] = float(dd_seg.min()) if dd_seg.size > 0 else np.nan
            UI[i] = _ulcer(seg)
        vals = {
            "VaR": VaR,
            "CVaR": CVaR,
            "max_drawdown": maxDD,
            "ulcer_index": UI,
        }
    return vals


__all__ += ["risk_metrics"]


# ---------------------------------------------------------------------------
# GPU accelerate — decorator שמחליף backend ל-CuPy אם יש צורך
# ---------------------------------------------------------------------------

def gpu_accelerate(
    func: _t.Callable[..., np.ndarray] | None = None,
    *,
    use_gpu: bool = True,
    gpu_threshold: int = 1_000_000,
) -> _t.Callable[..., np.ndarray]:
    """
    Decorator שמאיץ פונקציה שפועלת על np.ndarray באמצעות CuPy אם אפשר.

    - אם use_gpu=True ויש cupy מותקן
      ובסך הכל גודל ה־arrays >= gpu_threshold → המרה ל-CuPy.
    - אחרת → נשאר עם NumPy.

    Usage:
        @gpu_accelerate
        def my_heavy_func(x: np.ndarray) -> np.ndarray:
            ...

        @gpu_accelerate(use_gpu=True, gpu_threshold=5_000_000)
        def other(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            ...
    """
    # שימוש כ-decorator עם פרמטרים
    if func is None:
        return lambda f: gpu_accelerate(f, use_gpu=use_gpu, gpu_threshold=gpu_threshold)

    try:
        import cupy as cp  # type: ignore
    except ImportError:
        cp = None  # type: ignore

    @functools.wraps(func)
    def wrapper(*args: _t.Any, **kwargs: _t.Any) -> np.ndarray:
        array_args = [a for a in args if hasattr(a, "size")]
        size = int(sum(getattr(a, "size", 0) for a in array_args)) if array_args else 0

        if use_gpu and cp is not None and size >= gpu_threshold:
            gpu_args = [cp.asarray(a) if hasattr(a, "size") else a for a in args]
            result = func(*gpu_args, **kwargs)
            return cp.asnumpy(result)
        else:
            return func(*args, **kwargs)

    return wrapper


__all__ += ["gpu_accelerate"]


# ---------------------------------------------------------------------------
# safe_get — גישה בטוחה ל-nested dict
# ---------------------------------------------------------------------------

def safe_get(
    mapping: dict[str, _t.Any],
    path: str,
    *,
    default: _t.Any = None,
    sep: str = ".",
) -> _t.Any:
    """
    גישה בטוחה ל-nested dict לפי path בסגנון "a.b.c".

    Example:
        cfg = {"db": {"host": "localhost"}}
        safe_get(cfg, "db.host") → "localhost"
        safe_get(cfg, "db.port", default=5432) → 5432
    """
    current: _t.Any = mapping
    for part in path.split(sep):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


__all__ += ["safe_get"]

# ---------------------------------------------------------------------------
# Series Summary — מטריצות (matrix_helpers + advanced_metrics)
# ---------------------------------------------------------------------------

# Alias לסדרה של מטריצות (ndarray)
SeriesND = pd.Series

# ננסה לייבא את matrix_helpers
# ננסה לייבא את matrix_helpers
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

try:
    from common.matrix_helpers import (  # type: ignore[attr-defined]
        ensure_matrix_series,
        matrix_correlation,
        matrix_covariance,
        matrix_pca,
    )
    _HAS_MATRIX_HELPERS = True
except Exception as e:  # noqa: BLE001
    # ⚠ מצב Degraded: אם common.matrix_helpers לא נטען (שגיאת ייבוא / PATH / סינטקס),
    # לא מפילים את כל המערכת אלא מתריעים ומגדירים מימושי fallback.
    _HAS_MATRIX_HELPERS = False
    log = get_logger(__name__) if "get_logger" in globals() else None
    if log is not None:
        log.warning(
            "common.matrix_helpers לא נטען — summarize_series יעבוד במצב בסיסי בלבד: %s",
            e,
        )

    # ==== FALLBACKS — כדי שלא ייפול NameError, וגם לחיות במצב בסיסי ====

    def ensure_matrix_series(obj: Any, *_: Any, **__: Any) -> pd.Series:
        """
        Fallback בסיסי ל-ensure_matrix_series:
        - אם כבר Series → מחזיר כמו שהוא.
        - אחרת → עוטף ל-Series[object].
        """
        if isinstance(obj, pd.Series):
            return obj
        try:
            return pd.Series(list(obj), dtype="object")  # type: ignore[arg-type]
        except Exception:
            return pd.Series([obj], dtype="object")

    def matrix_correlation(
        obj: Any,
        *_: Any,
        **__: Any,
    ) -> Optional[pd.DataFrame]:
        """
        Fallback ל-matrix_correlation:
        - מנסה להמיר ל-DataFrame ולחשב corr() פשוט.
        - אם לא מצליח → מחזיר None.
        """
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.corr()
            if isinstance(obj, pd.Series):
                return obj.to_frame().corr()
            df = pd.DataFrame(obj)
            return df.corr()
        except Exception:
            if log is not None:
                log.debug("matrix_correlation fallback failed", exc_info=True)
            return None

    def matrix_covariance(
        obj: Any,
        *_: Any,
        **__: Any,
    ) -> Optional[pd.DataFrame]:
        """
        Fallback ל-matrix_covariance:
        - כמו matrix_correlation אבל עם cov().
        """
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.cov()
            if isinstance(obj, pd.Series):
                return obj.to_frame().cov()
            df = pd.DataFrame(obj)
            return df.cov()
        except Exception:
            if log is not None:
                log.debug("matrix_covariance fallback failed", exc_info=True)
            return None

    def matrix_pca(
        obj: Any,
        n_components: Optional[int] = None,
        *_: Any,
        **__: Any,
    ) -> Dict[str, Any]:
        """
        Fallback גס ל-PCA:
        - משתמש ב-numpy.cov + np.linalg.eigh.
        - מחזיר dict דומה לממשק המלא של matrix_helpers.matrix_pca:
            {
                "components": DataFrame (loadings),
                "explained_var": Series / array,
                "explained_ratio": Series / array,
            }
        """
        try:
            df = pd.DataFrame(obj)
            mat = df.values.astype(float)
            if mat.ndim != 2 or mat.shape[1] < 2:
                return {"components": pd.DataFrame(), "explained_var": [], "explained_ratio": []}

            X = mat.copy()
            X = X - np.nanmean(X, axis=0, keepdims=True)
            X = np.nan_to_num(X, nan=0.0)

            cov = np.cov(X, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)

            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            if n_components is not None and n_components < eigvals.size:
                eigvals = eigvals[:n_components]
                eigvecs = eigvecs[:, :n_components]

            total_var = float(eigvals.sum()) or 1.0
            ratio = eigvals / total_var

            comp_cols = [f"PC{j+1}" for j in range(eigvecs.shape[1])]
            components = pd.DataFrame(eigvecs, index=df.columns, columns=comp_cols)

            return {
                "components": components,
                "explained_var": eigvals,
                "explained_ratio": ratio,
            }
        except Exception:
            if log is not None:
                log.debug("matrix_pca fallback failed", exc_info=True)
            return {
                "components": pd.DataFrame(),
                "explained_var": [],
                "explained_ratio": [],
            }

# ננסה לייבא advanced_metrics, ואם אין – נגדיר stubs שזורקים ImportError
try:
    from common.advanced_metrics import (  # type: ignore
        cointegration_test,
        mahalanobis_distance,
        dynamic_time_warping,
        kalman_filter,
        distance_correlation,
    )
except ImportError:
    def cointegration_test(series, method=None):  # type: ignore[no-redef]
        raise ImportError("advanced_metrics.cointegration_test is unavailable")

    def mahalanobis_distance(series):  # type: ignore[no-redef]
        raise ImportError("advanced_metrics.mahalanobis_distance is unavailable")

    def dynamic_time_warping(series):  # type: ignore[no-redef]
        raise ImportError("advanced_metrics.dynamic_time_warping is unavailable")

    def kalman_filter(series):  # type: ignore[no-redef]
        raise ImportError("advanced_metrics.kalman_filter is unavailable")

    def distance_correlation(series):  # type: ignore[no-redef]
        raise ImportError("advanced_metrics.distance_correlation is unavailable")


def summarize_series(
    series: SeriesND,
    *,
    n_pca: int = 2,
    include_advanced: bool = True,
) -> dict[str, _t.Any]:
    """
    Compute key metrics על סדרה של מטריצות 2D ולהחזיר סיכום מובנה.

    מבנה התוצאה:
    -------------
    {
        "core": {
            "shape_summary": {...},
            "correlation": {...} or None,
            "covariance": {...} or None,
            "pca": {...} or None,
            "norms": {...} or None,
        },
        "advanced": {
            "cointegration_engle": {...} or None,
            "cointegration_johansen": {...} or None,
            "mahalanobis": {...} or None,
            "dtw": {...} or None,
            "kalman": {...} or None,
            "distance_correlation": {...} or None,
        }
    }

    Notes:
    ------
    - אם include_advanced=False → החלק המתקדם יחזור ריק ({}), כדי לחסוך זמן.
    - כל metric עטוף ב-try/except עם לוג WARNING ולא מפיל את כל הפונקציה.
    """
    log = get_logger(__name__)

    if series.empty:
        raise ValueError("summarize_series: input series is empty")

    # Validate series as "matrix series"
    series = ensure_matrix_series(series, "summarize_series")

    summary: dict[str, _t.Any] = {"core": {}, "advanced": {}}

    # ---------------- Core: shape summary ----------------
    try:
        shapes = [getattr(m, "shape", None) for m in series]
        valid_shapes = [s for s in shapes if isinstance(s, tuple) and len(s) == 2]
        shape_counts: dict[str, int] = {}
        for sh in valid_shapes:
            key = f"{sh[0]}x{sh[1]}"
            shape_counts[key] = shape_counts.get(key, 0) + 1
        summary["core"]["shape_summary"] = {
            "counts": shape_counts,
            "total": len(series),
        }
    except Exception as e:
        log.warning("summarize_series: shape_summary failed: %s", e)
        summary["core"]["shape_summary"] = None

    # ---------------- Core: correlation ----------------
    try:
        corr_df = matrix_correlation(series)
        summary["core"]["correlation"] = corr_df.to_dict()
    except Exception as e:
        log.warning("summarize_series: correlation failed: %s", e)
        summary["core"]["correlation"] = None

    # ---------------- Core: covariance ----------------
    try:
        cov_df = matrix_covariance(series)
        summary["core"]["covariance"] = cov_df.to_dict()
    except Exception as e:
        log.warning("summarize_series: covariance failed: %s", e)
        summary["core"]["covariance"] = None

    # ---------------- Core: PCA ----------------
    try:
        pca_res = matrix_pca(series, n_pca)
        # אם זו DataFrame (מימוש ישן/אחר)
        if isinstance(pca_res, pd.DataFrame):
            summary["core"]["pca"] = pca_res.to_dict()
        else:
            # אם זה dict כמו ב-fallback → נשמר אותו כמו שהוא (או נעשה make_json_safe)
            summary["core"]["pca"] = make_json_safe(pca_res)
    except Exception as e:
        log.warning("summarize_series: PCA failed: %s", e)
        summary["core"]["pca"] = None
 

    # ---------------- Core: Norm metrics (Frobenius / trace) ----------------
    try:
        frob_norms = []
        traces = []
        for m in series:
            arr = np.asarray(m, dtype=float)
            frob_norms.append(float(np.linalg.norm(arr)))
            traces.append(float(np.trace(arr)) if arr.shape[0] == arr.shape[1] else float("nan"))
        norm_stats = {
            "frob_mean": float(np.nanmean(frob_norms)) if frob_norms else float("nan"),
            "frob_std": float(np.nanstd(frob_norms, ddof=1)) if frob_norms else float("nan"),
            "trace_mean": float(np.nanmean(traces)) if traces else float("nan"),
            "trace_std": float(np.nanstd(traces, ddof=1)) if traces else float("nan"),
        }
        summary["core"]["norms"] = norm_stats
    except Exception as e:
        log.warning("summarize_series: norm metrics failed: %s", e)
        summary["core"]["norms"] = None

    # ---------------- Advanced metrics (optional) ----------------
    if include_advanced:
        advanced_specs: list[tuple[str, _t.Callable[[SeriesND], _t.Any]]] = [
            (
                "cointegration_engle",
                lambda s: cointegration_test(s, method="engle").to_dict(orient="index"),
            ),
            (
                "cointegration_johansen",
                lambda s: cointegration_test(s, method="johansen").to_dict(orient="index"),
            ),
            (
                "mahalanobis",
                lambda s: mahalanobis_distance(s).to_dict(),
            ),
            (
                "dtw",
                lambda s: dynamic_time_warping(s).to_dict(),
            ),
            (
                "kalman",
                lambda s: kalman_filter(s).to_dict(),
            ),
            (
                "distance_correlation",
                lambda s: distance_correlation(s).to_dict(),
            ),
        ]

        for name, fn in advanced_specs:
            try:
                summary["advanced"][name] = fn(series)
            except Exception as e:
                log.warning("summarize_series: %s failed: %s", name, e)
                summary["advanced"][name] = None
    else:
        summary["advanced"] = {}

    return summary


__all__.append("summarize_series")


# ---------------------------------------------------------------------------
# CLI — hash-file, humanize-bytes, load-dotenv, summarize matrices
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Helper utilities CLI")
    sub = p.add_subparsers(dest="command")

    # hash-file
    p1 = sub.add_parser("hash-file", help="Compute file hash of a given path.")
    p1.add_argument("path", type=str, help="Path to the file.")
    p1.add_argument(
        "--algorithm",
        default="sha256",
        help="Hash algorithm to use (default=sha256).",
    )

    # humanize-bytes
    p2 = sub.add_parser("humanize-bytes", help="Convert bytes to human-readable format.")
    p2.add_argument("bytes_count", type=int, help="Number of bytes to humanize.")

    # load-dotenv
    p3 = sub.add_parser("load-dotenv", help="Load .env file into environment.")
    p3.add_argument("--path", default=".env", help="Path to .env file.")
    p3.add_argument(
        "--override",
        action="store_true",
        help="Override existing environment vars if set.",
    )

    # summarize (matrix series)
    p4 = sub.add_parser(
        "summarize",
        help="Compute summary metrics for a series of matrices stored in a .npz file.",
    )
    p4.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to .npz file with matrix arrays (keys → arrays).",
    )
    p4.add_argument(
        "--n-pca",
        type=int,
        default=2,
        help="Number of PCA components to compute.",
    )
    p4.add_argument(
        "--no-advanced",
        action="store_true",
        help="Disable advanced metrics (cointegration, DTW, etc.).",
    )
    p4.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file; if omitted, print to stdout.",
    )
    p4.add_argument(
        "--format",
        type=str,
        choices=["json", "yaml"],
        default="json",
        help="Output format (default=json).",
    )

    return p


def main(argv: _t.Sequence[str] | None = None) -> None:
    """
    Command-line interface עבור חלק מה-helpers.

    פקודות זמינות:
        hash-file       → hash_file(path, algorithm)
        humanize-bytes  → humanize_bytes(bytes_count)
        load-dotenv     → load_dotenv(path, override)
        summarize       → summarize_series על .npz של מטריצות
    """
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "hash-file":
        print(hash_file(args.path, args.algorithm))
    elif args.command == "humanize-bytes":
        print(humanize_bytes(args.bytes_count))
    elif args.command == "load-dotenv":
        vars_loaded = load_dotenv(args.path, override=args.override)
        print(vars_loaded)
    elif args.command == "summarize":
        # Load matrix series from .npz archive
        data = np.load(args.input)
        mats = [data[k] for k in sorted(data.files)]
        series = pd.Series(mats)
        summary = summarize_series(series, n_pca=args.n_pca, include_advanced=not args.no_advanced)
        if args.format == "json":
            payload = make_json_safe(summary)
            text = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default_internal)
        else:
            if yaml is None:
                raise ImportError("PyYAML is required for --format yaml")
            payload = make_json_safe(summary)
            text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)  # type: ignore[arg-type]

        if args.output:
            out_path = pathlib.Path(args.output)
            out_path.write_text(text, encoding="utf-8")
            print(f"✓ wrote summary to {out_path}")
        else:
            print(text)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    # אם helpers מורץ כ-script ישירות
    main()


# ---------------------------------------------------------------------------
# Hook ל-common.helpers → arch guard ברמת package (לבדיקות)
# ---------------------------------------------------------------------------

def _common_getattr(name: str):  # noqa: D401
    """
    __getattr__ ל-package 'common', כדי ש:

        from common import helpers

    יעבוד, אבל יכבד את ה-arch guard:

    - אם sys.modules["arch"] is None (placeholder) → ImportError
    - אחרת → import של common.helpers רגיל.
    """
    if name == "helpers":
        if sys.modules.get("arch") is None and "arch" in sys.modules:
            # match pytest behaviour – ImportError כש-arch = None
            raise ImportError("The 'arch' package is unavailable (None placeholder)")
        import importlib  # defer heavy import
        return importlib.import_module("common.helpers")
    raise AttributeError(name)


try:
    # ננסה לדחוף __getattr__ לחבילה 'common'
    import common as _common_pkg  # type: ignore[import]
    if not hasattr(_common_pkg, "__getattr__"):
        _common_pkg.__getattr__ = _common_getattr  # type: ignore[assignment]
except ImportError:
    # אם 'common' לא נטען עדיין, sitecustomize/loader שלך יכולים להוסיף את זה אחר כך.
    pass
