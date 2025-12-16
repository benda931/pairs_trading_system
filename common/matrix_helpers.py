# -*- coding: utf-8 -*-
"""
matrix_helpers.py — Part 1/6 — Core Infrastructure & Validation (HF-grade)
==========================================================================

This is **Part 1/6** of `matrix_helpers.py` (rebuilt & expanded).

What this part provides
-----------------------
1. **Logging layer (quant-grade)**
   - Per-module logger with:
     • Environment-driven log level (MATRIX_HELPERS_LOG_LEVEL / LOG_LEVEL)
     • Optional colored output for local dev
     • Lightweight, dependency-free (no import from common.helpers to avoid cycles)
   - Safe to import very early in the dependency graph.

2. **Backend orchestration (NumPy / CuPy)**
   - Global `xp` backend alias (NumPy by default, CuPy when available).
   - `set_backend("numpy"|"cupy")`, `backend()`, `backend_context(...)`.
   - Best-effort GPU telemetry (`backend_info`, `gpu_memory_info`, `gpu_sync`).
   - Environment overrides:
       MATRIX_HELPERS_BACKEND = "numpy" | "cupy"
       MATRIX_HELPERS_GPU_MIN_SIZE = int (min total elements before using GPU helpers)

3. **Typed aliases (backend-agnostic)**
   - `NDArray`: union of NumPy & CuPy ndarray types.
   - `SeriesND`: pandas.Series of matrices; canonical container for all helpers.

Later parts (2–6) build on this API:
- Part 2 — Apply & Parallel Engine
- Part 3 — Rolling Window Tensor Operations
- Part 4 — Statistical Helpers (cov/corr/eig/PCA)
- Part 5 — Broadcast & Reshape Utilities
- Part 6 — Persistence & Tail-Dependence / Registry IO

IMPORTANT
---------
- This file **must not** import `common.helpers` or any module that imports
  `matrix_helpers` back; otherwise you get circular imports.
- All logging goes through the local `logger` defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import contextlib
import json
import logging
import os

import numpy as _np
import pandas as _pd

# ---------------------------------------------------------------------------
# Logging (hedge-fund style, but local only — no imports from common.helpers)
# ---------------------------------------------------------------------------

def _env_log_level(default: int = logging.INFO) -> int:
    """
    Resolve log level from env, in decreasing priority:

    1. MATRIX_HELPERS_LOG_LEVEL
    2. LOG_LEVEL
    3. APP_LOG_LEVEL
    """
    name = (
        os.environ.get("MATRIX_HELPERS_LOG_LEVEL")
        or os.environ.get("LOG_LEVEL")
        or os.environ.get("APP_LOG_LEVEL")
    )
    if not name:
        return default
    name = str(name).upper()
    return getattr(logging, name, default)


def _configure_logger(name: str) -> logging.Logger:
    """
    Configure a per-module logger exactly once.

    - No external dependencies.
    - In library / dashboard contexts the root logger may already be configured,
      so we add a handler **only if** there are no handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured by application / tests
        return logger

    level = _env_log_level(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03dZ [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = _configure_logger(__name__)

# ---------------------------------------------------------------------------
# Backend management — NumPy (CPU) / CuPy (GPU)
# ---------------------------------------------------------------------------

@dataclass
class BackendConfig:
    """
    Global backend configuration for matrix helpers.

    Parameters
    ----------
    preferred : {'numpy','cupy'}
        Initial preference; environment override via MATRIX_HELPERS_BACKEND.
    gpu_min_size : int
        Minimal total number of elements before it makes sense to consider GPU
        acceleration in heavy operations. This is a **hint** only; most helpers
        remain NumPy-only unless explicitly GPU-aware.
    allow_fallback : bool
        If True (default), silently fall back to NumPy when CuPy is unavailable
        or errors; if False, raise a hard error when CuPy cannot be used.
    """

    preferred: Literal["numpy", "cupy"] = "numpy"
    gpu_min_size: int = 1_000_000
    allow_fallback: bool = True


def _load_backend_config_from_env() -> BackendConfig:
    preferred_env = os.environ.get("MATRIX_HELPERS_BACKEND", "").strip().lower()
    preferred: Literal["numpy", "cupy"] = "numpy"
    if preferred_env in ("numpy", "cupy"):
        preferred = preferred_env  # type: ignore[assignment]

    gpu_min = os.environ.get("MATRIX_HELPERS_GPU_MIN_SIZE")
    try:
        gpu_min_size = int(gpu_min) if gpu_min is not None else 1_000_000
    except Exception:
        gpu_min_size = 1_000_000

    allow_fallback_env = os.environ.get("MATRIX_HELPERS_GPU_NO_FALLBACK", "")
    allow_fallback = not (allow_fallback_env.strip().lower() in ("1", "true", "yes"))

    return BackendConfig(
        preferred=preferred,
        gpu_min_size=gpu_min_size,
        allow_fallback=allow_fallback,
    )


_BACKEND_CONFIG: BackendConfig = _load_backend_config_from_env()

# Global backend alias:
# - xp = numpy (default)
# - xp = cupy  (if available + configured)
_AVAILABLE_BACKENDS: List[str] = ["numpy"]
_CURRENT_BACKEND: str = "numpy"

try:  # pragma: no cover (optional CuPy import)
    import cupy as _cp  # type: ignore

    _AVAILABLE_BACKENDS.insert(0, "cupy")
    # Honor env preference; default stays NumPy unless explicitly requested
    if _BACKEND_CONFIG.preferred == "cupy":
        xp = _cp  # type: ignore[assignment]
        _CURRENT_BACKEND = "cupy"
        logger.info("matrix_helpers: CuPy detected — GPU backend active (preferred)")
    else:
        xp = _np  # type: ignore[assignment]
        logger.info("matrix_helpers: CuPy available but using NumPy backend (preferred=numpy)")
except Exception:  # pragma: no cover
    xp = _np  # type: ignore[assignment]
    logger.info("matrix_helpers: CuPy not available — using NumPy (CPU only)")


def set_backend(name: Literal["numpy", "cupy"]) -> None:
    """
    Switch global numerical backend.

    Parameters
    ----------
    name : {'numpy','cupy'}
        Desired backend. If 'cupy' is not installed and allow_fallback=True,
        logs a warning and keeps NumPy. If allow_fallback=False, raises ValueError.
    """
    global xp, _CURRENT_BACKEND

    if name == _CURRENT_BACKEND:
        return

    if name == "cupy":
        if "cupy" not in _AVAILABLE_BACKENDS:
            msg = "CuPy backend not available — install 'cupy' or set MATRIX_HELPERS_BACKEND=numpy"
            if _BACKEND_CONFIG.allow_fallback:
                logger.warning("matrix_helpers.set_backend(cupy): %s; staying on NumPy", msg)
                return
            raise ValueError(msg)
        try:
            import importlib
            xp = importlib.import_module("cupy")  # type: ignore[assignment]
            _CURRENT_BACKEND = "cupy"
            logger.info("matrix_helpers: backend switched to CuPy")
        except Exception as exc:  # pragma: no cover
            msg = f"failed to activate CuPy backend: {exc}"
            if _BACKEND_CONFIG.allow_fallback:
                logger.warning("matrix_helpers.set_backend(cupy): %s; falling back to NumPy", msg)
                xp = _np  # type: ignore[assignment]
                _CURRENT_BACKEND = "numpy"
            else:
                raise ValueError(msg) from exc

    elif name == "numpy":
        xp = _np  # type: ignore[assignment]
        _CURRENT_BACKEND = "numpy"
        logger.info("matrix_helpers: backend switched to NumPy")
    else:
        raise ValueError(f"Unknown backend '{name}'. Options: {sorted(set(_AVAILABLE_BACKENDS))}")


def backend() -> str:
    """Return current backend name ('numpy' or 'cupy')."""
    return _CURRENT_BACKEND


@contextlib.contextmanager
def backend_context(name: Literal["numpy", "cupy"]) -> Iterator[None]:
    """
    Temporarily switch backend within a `with` block and restore afterwards.

    Example
    -------
    >>> with backend_context("numpy"):
    ...     # force CPU computations here
    ...     ...
    """
    prev = backend()
    if name == prev:
        # Fast-path — no switch
        yield
        return
    try:
        set_backend(name)
        yield
    finally:
        set_backend(prev)


def is_gpu_active() -> bool:
    """True if CuPy backend is currently active."""
    return backend() == "cupy"


def gpu_memory_info() -> dict[str, int] | None:
    """
    Best-effort GPU memory probe (bytes).

    Returns
    -------
    dict | None
        {'free': int, 'total': int} when CuPy is active and runtime info
        is available; otherwise None.
    """
    if not is_gpu_active():
        return None
    try:  # pragma: no cover
        free, total = _cp.cuda.runtime.memGetInfo()  # type: ignore[attr-defined]
        return {"free": int(free), "total": int(total)}
    except Exception:
        return None


def gpu_sync() -> None:
    """
    Synchronize default CUDA stream.

    - Safe no-op when running on NumPy.
    - Used in long-running backtests to avoid “hidden” async lag.
    """
    if not is_gpu_active():
        return
    try:  # pragma: no cover
        _cp.cuda.Stream.null.synchronize()  # type: ignore[attr-defined]
    except Exception:
        # We don't want GPU sync failures to kill the dashboard
        logger.debug("matrix_helpers.gpu_sync: failed to synchronize CUDA stream", exc_info=True)


def backend_info() -> dict[str, Any]:
    """
    Return a small telemetry snapshot for debugging/support/UX.

    Example payload
    ---------------
    {
        "backend": "numpy",
        "numpy": "1.26.4",
        "cupy": null,
        "gpu": null,
        "pid": 12345,
        "config": {
            "preferred": "cupy",
            "gpu_min_size": 1000000,
            "allow_fallback": true
        }
    }
    """
    info: dict[str, Any] = {
        "backend": backend(),
        "numpy": getattr(_np, "__version__", "n/a"),
        "cupy": None,
        "gpu": gpu_memory_info(),
        "pid": os.getpid(),
        "config": asdict(_BACKEND_CONFIG),
    }
    if is_gpu_active():
        try:  # pragma: no cover
            info["cupy"] = getattr(_cp, "__version__", "n/a")  # type: ignore[attr-defined]
        except Exception:
            info["cupy"] = "n/a"
    return info

# ---------------------------------------------------------------------------
# Typing aliases & module meta
# ---------------------------------------------------------------------------

from numpy.typing import NDArray as _NDArray  # type: ignore  # noqa: E402

# NDArray is "backend-aware": can be NumPy ndarray or CuPy ndarray (xp.ndarray)
NDArray = Union[_NDArray[Any], "xp.ndarray"]
SeriesND = _pd.Series

__version__ = "3.0.0-part1"

__all__ = [
    "__version__",
    # backend / telemetry
    "NDArray",
    "SeriesND",
    "backend",
    "set_backend",
    "backend_context",
    "is_gpu_active",
    "gpu_memory_info",
    "gpu_sync",
    "backend_info",
    # conversions (defined later in Part 1)
    "to_backend",
    "to_numpy",
    # validation (defined later in Part 1)
    "ValidationPolicy",
    "MatrixValidationError",
    "ensure_matrix_series",
]


# ---------------------------------------------------------------------------
# Conversions helpers (NumPy <-> active backend)
# ---------------------------------------------------------------------------

def to_numpy(arr: Any) -> _np.ndarray:
    """
    Convert any backend array/Series/DataFrame to a NumPy ndarray.

    - אם זה CuPy → נקרא get()
    - אם זה pandas עם to_numpy() → נשתמש בזה
    - אחרת → np.asarray
    """
    try:
        # CuPy array
        if hasattr(arr, "get"):
            return arr.get()  # type: ignore[call-arg]
        # pandas objects
        if hasattr(arr, "to_numpy"):
            return _np.asarray(arr.to_numpy())
        # כבר ndarray / list
        return _np.asarray(arr)
    except Exception:
        # fallback קשוח – לא מפיל את המערכת
        return _np.asarray(arr)


def to_backend(arr: Any) -> NDArray:
    """
    Convert *arr* to the active backend ndarray (xp).

    - תומך ב-pandas (to_numpy)
    - תומך ב-NumPy / CuPy / list
    - אם המרה ל-backend נכשלת → נופל חזרה ל-NumPy
    """
    if hasattr(arr, "to_numpy"):
        arr = arr.to_numpy()

    try:
        return xp.asarray(arr)
    except Exception as exc:
        logger.warning(
            "matrix_helpers.to_backend: conversion on backend '%s' failed (%s); falling back to NumPy",
            backend(),
            exc,
        )
        return _np.asarray(arr)

# ---------------------------------------------------------------------------
# Validation — Series of homogeneous 2‑D matrices
# ---------------------------------------------------------------------------

class MatrixValidationError(ValueError):
    """Raised when a collection cannot be coerced into a homogeneous 2‑D series."""


@dataclass
class ValidationPolicy:
    """Validation knobs for `ensure_matrix_series`.

    Parameters
    ----------
    enforce_2d : bool
        Require each item to be exactly 2‑D (ndim==2).
    uniform_shape : bool
        Require all matrices to share identical shape (rows, cols).
    promote_1d : None | Literal["row","col"]
        If a 1‑D vector is encountered and this is set, promote to 2‑D
        row/col vector. Ignored when `enforce_2d=True` and ndim!=1.
    dtype : str | _np.dtype | None
        If provided, cast each matrix to this dtype (on the active backend).
    order : None | Literal["C","F"]
        Memory order hint for casts (backend‑specific, best‑effort).
    copy : bool
        Force copy on cast. If False, backend may return a view.
    allow_none : bool
        If True, allows `None` entries which will be dropped.
    dropna : bool
        If True, drop `NaN`/`None` items before validation.
    min_rows, min_cols : int | None
        If set, enforce minimum rows/cols per matrix.
    strict_backend : bool
        If True, ensure returned arrays are on the **active** backend. If False,
        inputs already on NumPy are allowed to pass through.
    """

    enforce_2d: bool = True
    uniform_shape: bool = True
    promote_1d: Optional[Literal["row", "col"]] = None
    dtype: Optional[Union[str, _np.dtype]] = None
    order: Optional[Literal["C", "F"]] = None
    copy: bool = False
    allow_none: bool = False
    dropna: bool = False
    min_rows: Optional[int] = None
    min_cols: Optional[int] = None
    strict_backend: bool = True


def _cast_backend(mat: Any, *, policy: ValidationPolicy) -> NDArray:
    """Cast/copy `mat` to the active backend with policy hints."""
    arr = mat
    if hasattr(arr, "to_numpy"):
        arr = arr.to_numpy()
    # Promote 1‑D vector if policy allows
    if getattr(arr, "ndim", None) == 1 and policy.promote_1d:
        arr = arr[None, :] if policy.promote_1d == "row" else arr[:, None]
    # Enforce dtype/order if requested
    try:
        arr = xp.asarray(arr, dtype=policy.dtype, order=policy.order)
        if policy.copy:
            arr = xp.array(arr, dtype=policy.dtype, order=policy.order, copy=True)
    except Exception as exc:
        logger.warning("backend cast failed (%s), falling back to NumPy", exc)
        arr = _np.asarray(arr, dtype=policy.dtype, order=policy.order)
        if policy.copy:
            arr = _np.array(arr, dtype=policy.dtype, order=policy.order, copy=True)
    return arr  # NDArray on active backend or NumPy fallback


def ensure_matrix_series(
    obj: Any,
    name: str = "series",
    policy: ValidationPolicy | None = None,
    **kwargs: Any,
) -> SeriesND:
    """Validate/convert *obj* to a `Series` whose **elements are 2‑D ndarrays**.

    This is the **single source of truth** validator used by higher‑level helpers.
    It converts, enforces dimensionality, optionally promotes 1‑D vectors, and
    guarantees uniform shape when requested.

    Parameters
    ----------
    obj : Any
        Iterable/array‑like/Series/list‑of‑lists or nested np/cp arrays.
    name : str
        Label used in diagnostics.
    policy : ValidationPolicy | None
        Validation knobs. If None, a sane default policy is used.
    **kwargs : Any
        Shorthand overrides for `ValidationPolicy` fields (e.g., `dtype='float32'`,
        `promote_1d='row'`, `uniform_shape=False`).

    Returns
    -------
    pandas.Series
        Series(dtype=object) whose items are 2‑D ndarrays (active backend unless
        `strict_backend=False`).

    Raises
    ------
    MatrixValidationError
        When an element fails conversion/dimensionality, or shapes violate the policy.
    """
    pol = policy or ValidationPolicy()
    # override via kwargs
    for k, v in kwargs.items():
        if hasattr(pol, k):
            setattr(pol, k, v)
        else:
            raise TypeError(f"Unknown policy field: {k}")

    # Construct Series
    series: SeriesND = obj if isinstance(obj, _pd.Series) else _pd.Series(obj)

    # Optionally drop None/NaN before processing
    if pol.dropna:
        series = series.dropna()
    elif not pol.allow_none and series.isna().any():
        bad_idx = [i for i, v in series.items() if v is None]
        if bad_idx:
            raise MatrixValidationError(f"{name}: None items not allowed (set allow_none=True); at {bad_idx[:5]}…")

    mats: List[NDArray] = []

    for idx, mat in series.items():
        try:
            arr = _cast_backend(mat, policy=pol)
        except Exception as exc:
            raise MatrixValidationError(f"{name}[{idx}] could not be converted to ndarray (type={type(mat)})") from exc

        if pol.enforce_2d and getattr(arr, "ndim", None) != 2:
            raise MatrixValidationError(f"{name}[{idx}] must be 2‑D (found ndim={getattr(arr, 'ndim', None)})")

        # Minimum shape constraints
        if getattr(arr, "ndim", 0) == 2:
            r, c = arr.shape
            if pol.min_rows is not None and r < pol.min_rows:
                raise MatrixValidationError(f"{name}[{idx}] has too few rows: {r} < {pol.min_rows}")
            if pol.min_cols is not None and c < pol.min_cols:
                raise MatrixValidationError(f"{name}[{idx}] has too few cols: {c} < {pol.min_cols}")

        # Ensure arrays are on the active backend when strict_backend=True
        if pol.strict_backend and backend() == "numpy" and type(arr).__module__.startswith("cupy"):
            arr = _np.asarray(arr)
        if pol.strict_backend and backend() == "cupy" and type(arr).__module__.startswith("numpy"):
            arr = xp.asarray(arr)

        mats.append(arr)

    # Uniform shape enforcement
    if pol.uniform_shape and mats:
        shape0 = mats[0].shape
        bad_shapes = {m.shape for m in mats}
        if len(bad_shapes) > 1:
            raise MatrixValidationError(f"All matrices in '{name}' must share the same shape; found {sorted(bad_shapes)}")

    out = _pd.Series(mats, index=series.index, dtype="object")
    logger.debug("ensure_matrix_series(%s): %d matrices (shape=%s) | policy=%s", name, len(mats), mats[0].shape if mats else None, asdict(pol))
    return out

# ---------------------------------------------------------------------------
# (End of Part 1/6)
# Paste Part 2/6 **after this**: Apply helpers (serial & parallel)
# ---------------------------------------------------------------------------

# =============================================================================
# Part 2/6 — Apply & Parallel Engine (HF‑grade)
# =============================================================================
from typing import Dict
import time
import math
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future

# (version is defined in Part 1; later parts must not redefine __version__)
__all__.extend([
    "ApplyPolicy",
    "MultiApplyError",
    "apply_matrix_series",
    "apply_matrix_series_parallel",
    "map_with_index",
    "map_reduce",
])

@dataclass
class ApplyPolicy:
    """Execution policy for apply helpers.

    Parameters
    ----------
    max_workers : int | None
        Degree of parallelism. None → min(32, os.cpu_count() or 1).
    executor : {'thread','process'}
        Thread pool (default) supports lambdas/closures; process pool needs picklable callables.
    chunk_size : int
        Submit tasks in chunks of this size to reduce overhead (≥1).
    progress : bool
        Show tqdm progress if installed.
    ordered : bool
        Preserve original Series order when collecting results.
    timeout : float | None
        Per‑task timeout in seconds. None → wait indefinitely.
    retry_n : int
        Retries per task on exception (0 = no retry).
    retry_backoff : float
        Backoff factor (seconds) between retries (exponential).
    return_exceptions : bool
        If True, embed exceptions in output; else raise aggregated error.
    fail_fast : bool
        If True, cancel remaining tasks on first failure (ignored if return_exceptions=True).
    device_affinity : {'auto','cpu','gpu'}
        Try to keep tensors on chosen device before invoking fn (best‑effort).
    """
    max_workers: Optional[int] = None
    executor: Literal["thread","process"] = "thread"
    chunk_size: int = 1
    progress: bool = False
    ordered: bool = True
    timeout: Optional[float] = None
    retry_n: int = 0
    retry_backoff: float = 0.25
    return_exceptions: bool = False
    fail_fast: bool = False
    device_affinity: Literal["auto","cpu","gpu"] = "auto"

class MultiApplyError(RuntimeError):
    """Aggregated exceptions from parallel apply.

    Attributes
    ----------
    errors : Dict[Any, BaseException]
        Mapping from index → exception raised by that task.
    """
    def __init__(self, message: str, errors: Dict[Any, BaseException]):
        super().__init__(message)
        self.errors = errors

# ------------------------------- helpers ------------------------------------

def _iter_progress(it, total=None, desc:str=""):
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(it, total=total, desc=desc) if total is not None else tqdm(it, desc=desc)
    except Exception:
        return it


def _maybe_move_device(arr: NDArray, policy: ApplyPolicy) -> NDArray:
    try:
        if policy.device_affinity == "auto":
            return arr
        if policy.device_affinity == "cpu" and is_gpu_active():
            return _np.asarray(arr)
        if policy.device_affinity == "gpu" and not is_gpu_active():
            try:
                import cupy as _cp  # type: ignore
                return _cp.asarray(arr)
            except Exception:
                return arr
        return arr
    except Exception:
        return arr


def _task_apply(idx, mat: NDArray, fn: Callable[[NDArray], Any], policy: ApplyPolicy):
    mat2 = _maybe_move_device(mat, policy)
    tries = policy.retry_n + 1
    delay = policy.retry_backoff
    last_exc = None
    for k in range(tries):
        try:
            return idx, fn(mat2)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if k < policy.retry_n:
                time.sleep(delay)
                delay *= 2
            else:
                break
    # exhausted
    assert last_exc is not None
    raise last_exc

# ------------------------------- public API ---------------------------------

def apply_matrix_series(series: SeriesND, fn: Callable[[NDArray], Any], *, policy: ApplyPolicy | None = None, **overrides) -> _pd.Series:
    """Serial apply with strict validation & optional device affinity.

    Notes
    -----
    - Always validates via `ensure_matrix_series`.
    - If `return_exceptions=True`, exceptions are embedded as values.
    """
    pol = policy or ApplyPolicy()
    for k, v in overrides.items():
        if hasattr(pol, k):
            setattr(pol, k, v)
        else:
            raise TypeError(f"Unknown policy field: {k}")

    ser = ensure_matrix_series(series, "apply_matrix_series")
    out: Dict[Any, Any] = {}
    errors: Dict[Any, BaseException] = {}
    it = ser.items()
    if pol.progress:
        it = _iter_progress(it, total=len(ser), desc="apply(serial)")
    for idx, mat in it:
        try:
            _, res = _task_apply(idx, mat, fn, pol)
            out[idx] = res
        except Exception as e:  # noqa: BLE001
            if pol.return_exceptions:
                out[idx] = e
            else:
                errors[idx] = e
                if pol.fail_fast:
                    break
    if errors and not pol.return_exceptions:
        msg = f"{len(errors)} task(s) failed in apply(serial). First: {next(iter(errors.values()))}"
        raise MultiApplyError(msg, errors)
    return _pd.Series(out, index=ser.index)


def apply_matrix_series_parallel(series: SeriesND, fn: Callable[[NDArray], Any], *, policy: ApplyPolicy | None = None, **overrides) -> _pd.Series:
    """Parallel apply over a Series of matrices with retries, timeouts & ordering.

    - Thread pool by default (safe for lambdas/closures). Use `executor='process'`
      רק אם הפונקציה והאובייקטים picklable.
    - Preserves order by default; set `ordered=False` לאסוף as‑completed.
    - `return_exceptions=True` יטמיע חריגות בפלט במקום לזרוק.
    """
    pol = policy or ApplyPolicy()
    for k, v in overrides.items():
        if hasattr(pol, k):
            setattr(pol, k, v)
        else:
            raise TypeError(f"Unknown policy field: {k}")

    ser = ensure_matrix_series(series, "apply_matrix_series_parallel")
    if len(ser) == 0:
        return ser.copy()
    if len(ser) == 1 or (os.cpu_count() or 1) == 1:
        return apply_matrix_series(ser, fn, policy=pol)

    max_workers = pol.max_workers or min(32, os.cpu_count() or 1)
    Pool = ThreadPoolExecutor if pol.executor == "thread" else ProcessPoolExecutor

    # task iterator
    def _iter_tasks():
        for idx, mat in ser.items():
            yield idx, mat

    errors: Dict[Any, BaseException] = {}
    results: Dict[Any, Any] = {}

    with Pool(max_workers=max_workers) as ex:
        futures: Dict[Future, Any] = {}
        for idx, mat in _iter_tasks():
            fut = ex.submit(_task_apply, idx, mat, fn, pol)
            futures[fut] = idx

        if pol.ordered:
            # preserve order by waiting on each future by index
            for idx, mat in (_iter_progress(list(ser.items()), total=len(ser), desc="apply(parallel,ordered)") if pol.progress else ser.items()):
                # locate the future for this idx
                fut_for_idx = next((f for f,i in futures.items() if i == idx), None)
                if fut_for_idx is None:
                    continue
                try:
                    _, res = fut_for_idx.result(timeout=pol.timeout)
                    results[idx] = res
                except Exception as e:  # noqa: BLE001
                    if pol.return_exceptions:
                        results[idx] = e
                    else:
                        errors[idx] = e
                        if pol.fail_fast:
                            # cancel others best‑effort
                            for f in futures:
                                f.cancel()
                            break
        else:
            # collect as completed
            iterator = as_completed(futures, timeout=None)
            if pol.progress:
                iterator = _iter_progress(iterator, total=len(futures), desc="apply(parallel)")
            for fut in iterator:
                idx = futures.get(fut)
                try:
                    _, res = fut.result(timeout=pol.timeout)
                    results[idx] = res
                except Exception as e:  # noqa: BLE001
                    if pol.return_exceptions:
                        results[idx] = e
                    else:
                        errors[idx] = e
                        if pol.fail_fast:
                            for f in futures:
                                f.cancel()
                            break

    if errors and not pol.return_exceptions:
        msg = f"{len(errors)} task(s) failed in apply(parallel). First: {next(iter(errors.values()))}"
        raise MultiApplyError(msg, errors)

    # materialize in original index order
    return _pd.Series({idx: results.get(idx, _np.nan) for idx in ser.index}, index=ser.index)

# ------------------------------ extras --------------------------------------
def map_with_index(
    series: SeriesND,
    fn: Callable[[Any, NDArray], Any],
    *,
    policy: ApplyPolicy | None = None,
    parallel: bool = False,
    **overrides
) -> _pd.Series:
    """
    Apply a function that needs both (index, matrix).
    - Serial path iterates directly over (idx, mat) → O(n)
    - Parallel path uses a pool and preserves index order on output
    - Honors ApplyPolicy: max_workers, executor ('thread'/'process'),
      return_exceptions, fail_fast, timeout, progress, etc.
    """
    ser = ensure_matrix_series(series, "map_with_index")

    pol = policy or ApplyPolicy()
    for k, v in overrides.items():
        if hasattr(pol, k):
            setattr(pol, k, v)
        else:
            raise TypeError(f"Unknown policy field: {k}")

    # ---- Serial path ----
    if not parallel or len(ser) <= 1 or (os.cpu_count() or 1) == 1:
        out: Dict[Any, Any] = {}
        errors: Dict[Any, BaseException] = {}
        it = ser.items()
        if pol.progress:
            it = _iter_progress(it, total=len(ser), desc="map_with_index(serial)")
        for idx, mat in it:
            try:
                out[idx] = fn(idx, mat)
            except Exception as e:
                if pol.return_exceptions:
                    out[idx] = e
                else:
                    errors[idx] = e
                    if pol.fail_fast:
                        break
        if errors and not pol.return_exceptions:
            msg = f"{len(errors)} task(s) failed in map_with_index(serial). First: {next(iter(errors.values()))}"
            raise MultiApplyError(msg, errors)
        return _pd.Series(out, index=ser.index)

    # ---- Parallel path ----
    max_workers = pol.max_workers or min(32, os.cpu_count() or 1)
    Pool = ThreadPoolExecutor if pol.executor == "thread" else ProcessPoolExecutor

    errors: Dict[Any, BaseException] = {}
    results: Dict[Any, Any] = {}

    def _task(idx_mat: Tuple[Any, NDArray]) -> Tuple[Any, Any]:
        idx, mat = idx_mat
        # נשתמש בלוגיקת ריטרייז כמו ב־_task_apply
        tries = pol.retry_n + 1
        delay = pol.retry_backoff
        last_exc = None
        # התאמת affinity אם נדרש
        mat2 = _maybe_move_device(mat, pol) if "device_affinity" in pol.__dict__ else mat
        for _ in range(tries):
            try:
                return idx, fn(idx, mat2)
            except Exception as e:
                last_exc = e
                if delay and tries > 1:
                    time.sleep(delay)
                    delay *= 2
        assert last_exc is not None
        raise last_exc

    tasks = list(ser.items())
    with Pool(max_workers=max_workers) as ex:
        futs = {ex.submit(_task, item): item[0] for item in tasks}
        iterator = as_completed(futs)
        if pol.progress:
            iterator = _iter_progress(iterator, total=len(futs), desc="map_with_index(parallel)")
        for fut in iterator:
            idx = futs[fut]
            try:
                i, val = fut.result(timeout=pol.timeout)
                results[i] = val
            except Exception as e:
                if pol.return_exceptions:
                    results[idx] = e
                else:
                    errors[idx] = e
                    if pol.fail_fast:
                        # ביטול משימות שנותרו (best-effort)
                        for f in futs:
                            f.cancel()
                        break

    if errors and not pol.return_exceptions:
        msg = f"{len(errors)} task(s) failed in map_with_index(parallel). First: {next(iter(errors.values()))}"
        raise MultiApplyError(msg, errors)

    # איסוף לפי סדר האינדקס המקורי
    return _pd.Series({idx: results.get(idx, _np.nan) for idx in ser.index}, index=ser.index)

def map_reduce(series: SeriesND, map_fn: Callable[[NDArray], Any], reduce_fn: Callable[[Any, Any], Any], *, init: Any | None = None, policy: ApplyPolicy | None = None, parallel: bool = True, **overrides) -> Any:
    """Map over matrices (optionally parallel) then reduce the results to a single value."""
    if parallel:
        out = apply_matrix_series_parallel(series, map_fn, policy=policy, **overrides)
    else:
        out = apply_matrix_series(series, map_fn, policy=policy, **overrides)
    it = out.values
    acc = init
    for v in it:
        acc = v if acc is None else reduce_fn(acc, v)
    return acc

# ---------------------------------------------------------------------------
# (End of Part 2/6)
# Next: Part 3/6 — Rolling window tensor operations
# ---------------------------------------------------------------------------

# =============================================================================
# Part 3/6 — Rolling Window Tensor Operations (HF‑grade)
# =============================================================================
from typing import Sequence

# (version is defined in Part 1; later parts must not redefine __version__)
__all__.extend([
    "RollingPolicy",
    "rolling_tensor",
    "rolling_map",
    "rolling_pair_map",
    "rolling_reduce",
    # conveniences
    "rolling_mean",
    "rolling_std",
    "rolling_cov",
    "rolling_corr",
    # advanced extensions
    "rolling_map_parallel",
    "rolling_zscore",
    "rolling_quantile",
    "rolling_ewm_mean",
    "rolling_ewm_cov",
])

@dataclass
class RollingPolicy:
    """Policy/config for rolling window operations.

    Parameters
    ----------
    window : int
        Number of matrices per window (must be ≥ 1).
    min_periods : int | None
        Minimum #matrices required to emit a value; if None uses `window`.
    step : int
        Slide step between windows (≥1). For `step>1`, windows are decimated.
    center : bool
        If True, align the output index to the **center** of the window; otherwise
        align to the **right** edge (i.e., window end).
    return_tensor : bool
        If True, return the 3‑D tensor per window (H×W×T); otherwise return the
        function result.
    progress : bool
        Show tqdm progress bars when available.
    preserve_index : bool
        Preserve the original index on the output Series.
    pad_with_nan : bool
        For windows that do not meet `min_periods`, emit NaN (or None) instead of
        skipping the slot.
    device_affinity : {'auto','cpu','gpu'}
        Attempt to place constructed tensors on the selected device (best‑effort).
    nan_policy : {'propagate','omit','fill'}
        How to handle NaNs inside matrices when forming tensors/stats.
    nan_fill_value : float | None
        Value to fill when `nan_policy='fill'`.
    ewm_alpha : float | None
        If set (0<alpha<=1), enables exponential weighting in helpers that support it.
    """
    window: int
    min_periods: Optional[int] = None
    step: int = 1
    center: bool = False
    return_tensor: bool = False
    progress: bool = False
    preserve_index: bool = True
    pad_with_nan: bool = True
    device_affinity: Literal["auto","cpu","gpu"] = "auto"
    nan_policy: Literal["propagate","omit","fill"] = "propagate"
    nan_fill_value: Optional[float] = None
    ewm_alpha: Optional[float] = None


def _iter_prog(it, total: Optional[int], desc: str, enable: bool):
    if enable:
        try:
            from tqdm import tqdm  # type: ignore
            return tqdm(it, total=total, desc=desc)
        except Exception:
            return it
    return it


def _maybe_affinity(tensor: NDArray, policy: RollingPolicy) -> NDArray:
    if policy.device_affinity == "auto":
        return tensor
    if policy.device_affinity == "cpu" and is_gpu_active():
        return _np.asarray(tensor)
    if policy.device_affinity == "gpu" and not is_gpu_active():
        try:
            import cupy as _cp  # type: ignore
            return _cp.asarray(tensor)
        except Exception:
            return tensor
    return tensor


def rolling_tensor(series: SeriesND, policy: RollingPolicy) -> _pd.Series:
    """Build a 3‑D tensor per rolling window according to *policy*.

    Returns a `Series(dtype=object)` whose items are H×W×T tensors (backend‑aware).
    """
    if policy.window < 1:
        raise ValueError("window must be ≥ 1")
    minp = int(policy.min_periods or policy.window)

    ser = ensure_matrix_series(series, "rolling_tensor")
    mats = list(ser)
    n = len(mats)

    out_idx = ser.index if policy.preserve_index else _pd.RangeIndex(n)
    out: List[Any] = [(_np.nan) for _ in range(n)]  # replaced as we go

    rng = range(policy.window - 1, n, policy.step)
    it = _iter_prog(rng, math.ceil((n - (policy.window - 1)) / policy.step), "rolling(tensor)", policy.progress)

    for end_i in it:
        start_i = end_i - policy.window + 1
        if start_i < 0:
            continue
        window_mats = mats[start_i : end_i + 1]
        if len(window_mats) < minp:
            continue
        tensor = xp.stack(window_mats, axis=2)  # H×W×T
        tensor = _maybe_affinity(tensor, policy)
        if policy.center:
            pos = start_i + (policy.window // 2)
        else:
            pos = end_i
        out[pos] = tensor

    return _pd.Series(out, index=out_idx, dtype="object")


def rolling_map(series: SeriesND, fn: Callable[[NDArray], Any], policy: RollingPolicy) -> _pd.Series:
    """Apply *fn* to each rolling window tensor. Honors `policy.return_tensor`."""
    ser = ensure_matrix_series(series, "rolling_map")
    tens = rolling_tensor(ser, policy)
    if policy.return_tensor:
        return tens
    out: Dict[Any, Any] = {}
    it = _iter_prog(tens.items(), len(tens), "rolling(map)", policy.progress)
    for idx, t in it:
        if isinstance(t, float) and _np.isnan(t):
            out[idx] = _np.nan
        else:
            out[idx] = fn(t)  # user fn sees H×W×T
    return _pd.Series(out, index=tens.index)


def rolling_pair_map(
    series_x: SeriesND,
    series_y: SeriesND,
    fn: Callable[[NDArray, NDArray], Any],
    policy: RollingPolicy,
    *,
    align: Literal["inner","left","right"] = "inner",
) -> _pd.Series:
    """Rolling map over **two** aligned series. Applies *fn(tensor_x, tensor_y)*.

    Both windows are H×W×T tensors; shapes must be identical in H×W per window.
    """
    sx = ensure_matrix_series(series_x, "rolling_pair_map_x")
    sy = ensure_matrix_series(series_y, "rolling_pair_map_y")
    if align == "inner" and not sx.index.equals(sy.index):
        common = sx.index.intersection(sy.index)
        sx = sx.loc[common]
        sy = sy.loc[common]
    elif align == "left":
        sy = sy.reindex(sx.index)
    elif align == "right":
        sx = sx.reindex(sy.index)

    tx = rolling_tensor(sx, policy)
    ty = rolling_tensor(sy, policy)

    out: Dict[Any, Any] = {}
    for idx in tx.index:
        a = tx.loc[idx]; b = ty.loc[idx]
        if any(isinstance(v, float) and _np.isnan(v) for v in (a, b)):
            out[idx] = _np.nan
            continue
        out[idx] = fn(a, b)
    return _pd.Series(out, index=tx.index)


def rolling_reduce(
    series: SeriesND,
    map_fn: Callable[[NDArray], Any],
    reduce_fn: Callable[[Any, Any], Any],
    *,
    init: Any | None = None,
    policy: RollingPolicy,
) -> _pd.Series:
    """Map windows via *map_fn* then **reduce inside each window** to a scalar/obj."""
    ser = ensure_matrix_series(series, "rolling_reduce")
    tens = rolling_tensor(ser, policy)
    out = {}
    for idx, t in tens.items():
        if isinstance(t, float) and _np.isnan(t):
            out[idx] = _np.nan
            continue
        acc = init
        # user map_fn receives the whole tensor; if it returns an iterable, we reduce it
        val = map_fn(t)
        if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
            for v in val:
                acc = v if acc is None else reduce_fn(acc, v)
            out[idx] = acc
        else:
            out[idx] = val if acc is None else reduce_fn(acc, val)
    return _pd.Series(out, index=tens.index)

# -------------------- convenience rolling statistics ------------------------

def _nanify(arr: NDArray, policy: RollingPolicy) -> NDArray:
    if policy.nan_policy == "propagate":
        return arr
    if policy.nan_policy == "omit":
        return xp.where(xp.isnan(arr), 0, arr)
    if policy.nan_policy == "fill":
        fill = 0.0 if policy.nan_fill_value is None else float(policy.nan_fill_value)
        return xp.where(xp.isnan(arr), fill, arr)
    return arr

def rolling_mean(series: SeriesND, policy: RollingPolicy) -> _pd.Series:
    def _fn(t: NDArray) -> NDArray:
        t2 = _nanify(t, policy)  # מיישם nan_policy
        # EWM
        if policy.ewm_alpha and 0.0 < policy.ewm_alpha <= 1.0:
            alpha = float(policy.ewm_alpha)
            acc = t2[..., 0]
            for i in range(1, t2.shape[2]):
                acc = alpha * t2[..., i] + (1 - alpha) * acc
            return acc[None, :]
        # mean עמיד ל-NaN (ללא where)
        mask = ~xp.isnan(t2)
        num = xp.where(mask, t2, 0).sum(axis=2)
        den = mask.sum(axis=2)
        den = xp.where(den == 0, 1, den)
        mu = num / den
        return mu[None, :]
    return rolling_map(series, _fn, policy)


def rolling_std(series: SeriesND, policy: RollingPolicy) -> _pd.Series:
    """Column‑wise std over each window (returns 1×W vector)."""
    def _fn(t: NDArray) -> NDArray:
        t2 = _nanify(t, policy)
        return xp.std(t2, axis=2, ddof=1)[None, :]
    return rolling_map(series, _fn, policy)

def rolling_cov(series: SeriesND, policy: RollingPolicy) -> _pd.Series:
    """Sample covariance over each window (W×W)."""
    def _cov(t: NDArray) -> NDArray:
        mat = xp.mean(_nanify(t, policy), axis=2)
        return xp.cov(mat, rowvar=False)
    return rolling_map(series, _cov, policy)

def rolling_corr(series: SeriesND, policy: RollingPolicy) -> _pd.Series:
    """Correlation over each window (W×W)."""
    def _corr(t: NDArray) -> NDArray:
        mat = xp.mean(_nanify(t, policy), axis=2)
        cov = xp.cov(mat, rowvar=False)
        std = xp.sqrt(xp.diag(cov))
        denom = std[:, None] * std[None, :]
        return xp.where(denom == 0, 0, cov / denom)
    return rolling_map(series, _corr, policy)

# -------------------- advanced rolling extensions ---------------------------

def rolling_zscore(series: SeriesND, policy: RollingPolicy) -> _pd.Series:
    """Column‑wise z‑score per window (returns 1×W vector)."""
    def _fn(t: NDArray) -> NDArray:
        mat = _nanify(t, policy)
        mu = xp.mean(mat, axis=2)
        sd = xp.std(mat, axis=2, ddof=1)
        sd = xp.where(sd == 0, 1, sd)
        z = (mat[..., -1] - mu) / sd
        return z[None, :]
    return rolling_map(series, _fn, policy)

def rolling_quantile(series: SeriesND, q: float, policy: RollingPolicy) -> _pd.Series:
    """Column‑wise quantile per window (returns 1×W vector)."""
    q = float(q)
    def _fn(t: NDArray) -> NDArray:
        mat = _nanify(t, policy)
        # move axis to last then compute quantile along time
        return xp.quantile(mat, q=q, axis=2)[None, :]
    return rolling_map(series, _fn, policy)

def rolling_ewm_mean(series: SeriesND, alpha: float, policy: RollingPolicy) -> _pd.Series:
    pol = RollingPolicy(**{**policy.__dict__, "ewm_alpha": float(alpha)})
    return rolling_mean(series, pol)

def rolling_ewm_cov(series: SeriesND, alpha: float, policy: RollingPolicy) -> _pd.Series:
    pol = RollingPolicy(**{**policy.__dict__, "ewm_alpha": float(alpha)})
    return rolling_cov(series, pol)

# -------------------- parallel rolling map ----------------------------------

def rolling_map_parallel(series: SeriesND, fn: Callable[[NDArray], Any], policy: RollingPolicy, *, max_workers: Optional[int] = None) -> _pd.Series:
    """Parallel version of `rolling_map` — builds tensors once, maps in threads."""
    tens = rolling_tensor(ensure_matrix_series(series, "rolling_map_parallel"), policy)
    if policy.return_tensor:
        return tens
    # build tasks skipping NaNs
    tasks = [(i, t) for i, t in tens.items() if not (isinstance(t, float) and _np.isnan(t))]
    results: Dict[Any, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers or min(32, os.cpu_count() or 1)) as ex:
        futs = {ex.submit(fn, t): i for i, t in tasks}
        iterator = as_completed(futs)
        if policy.progress:
            iterator = _iter_prog(iterator, total=len(futs), desc="rolling(map_parallel)", enable=True)
        for fut in iterator:
            idx = futs[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = e
    # stitch back in index order
    out = {i: (results[i] if i in results else _np.nan) for i in tens.index}
    return _pd.Series(out, index=tens.index)

# ---------------------------------------------------------------------------
# (End of Part 3/6)
# Next: Part 4/6 — Statistical helpers (cov/corr/eig/PCA, robust variants)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# =============================================================================
# Part 4/6 — Statistical Helpers (HF-grade, expanded & dedup-safe)
# =============================================================================
# NOTE: Relies on symbols from Part 1: xp, _np, _pd, NDArray, SeriesND,
#       ensure_matrix_series, to_backend, to_numpy, logger

__all__.extend([
    # diagnostics
    "is_symmetric", "assert_symmetric", "symmetrize", "is_psd",
    "make_psd", "nearest_psd", "condition_number", "matrix_trace",
    # core stats
    "stat_center", "stat_scale", "stat_var", "stat_std",
    "stat_cov", "stat_corr",
    # eig / factorizations & whitening
    "eig_sorted", "cholesky_psd", "chol_inverse", "whiten_zca", "whiten_pca",
    # PCA family
    "pca_decompose", "pca_components_df", "pca_transform",
    "pca_scores_df", "pca_reconstruct", "pca_scree",
    # robust / shrinkage (optional)
    "robust_cov_mcd", "shrinkage_cov_lw",
])

# ------------------------------ diagnostics ----------------------------------

def is_symmetric(mat: NDArray, *, atol: float = 1e-8) -> bool:
    """Fast symmetry check (real Hermitian) on current backend."""
    A = to_backend(mat)
    try:
        return bool(xp.all(xp.abs(A - A.T) <= atol))
    except Exception:
        A = to_numpy(A)
        return bool(_np.all(_np.abs(A - A.T) <= atol))

def assert_symmetric(mat: NDArray, *, name: str = "matrix", atol: float = 1e-8) -> None:
    if not is_symmetric(mat, atol=atol):
        raise ValueError(f"{name} must be symmetric within atol={atol}")

def symmetrize(mat: NDArray) -> NDArray:
    """Return (A + A^T)/2 on current backend."""
    A = to_backend(mat)
    return (A + A.T) / 2.0

def is_psd(mat: NDArray, *, atol: float = 1e-10) -> bool:
    """Numerical PSD test: all eigenvalues >= -atol."""
    A = to_backend(mat)
    try:
        w = xp.linalg.eigvalsh(A)
        return bool(xp.all(w >= -atol))
    except Exception:
        try:
            w = _np.linalg.eigvalsh(to_numpy(A))
            return bool(_np.all(w >= -atol))
        except Exception:
            return False

def condition_number(mat: NDArray) -> float:
    """Spectral condition number λ_max/λ_min (clipped), inf if singular."""
    A = to_backend(mat)
    try:
        w = xp.linalg.eigvalsh(A)
        w = xp.where(w < 0, 0, w)
        wmax = float(xp.max(w)) if w.size else 0.0
        wmin = float(xp.min(xp.where(w > 0, w, xp.inf)))
        return float(_np.inf if wmin == 0 else wmax / wmin)
    except Exception:
        w = _np.linalg.eigvalsh(to_numpy(A))
        w = _np.where(w < 0, 0, w)
        wmax = float(_np.max(w)) if w.size else 0.0
        wmin = float(_np.min(_np.where(w > 0, w, _np.inf)))
        return float(_np.inf if wmin == 0 else wmax / wmin)

def matrix_trace(mat: NDArray) -> float:
    """Trace(A) on current backend."""
    try:
        return float(xp.trace(to_backend(mat)))
    except Exception:
        return float(_np.trace(to_numpy(mat)))

def make_psd(mat: NDArray, *, jitter: float = 1e-10, max_tries: int = 7) -> NDArray:
    """Return a PSD matrix by adding diagonal jitter geometrically; final fallback clamps eigs ≥ 0."""
    A = to_backend(mat)
    if is_psd(A):
        return A
    jj = float(max(jitter, 0.0))
    I = xp.eye(A.shape[0], dtype=A.dtype)
    for _ in range(int(max_tries)):
        B = A + jj * I
        if is_psd(B):
            return B
        jj *= 10.0
    # clamp negative eigs
    try:
        w, V = xp.linalg.eigh(A)
        w = xp.where(w < 0, 0, w)
        return (V @ xp.diag(w) @ V.T).astype(A.dtype)
    except Exception:
        w, V = _np.linalg.eigh(to_numpy(A))
        w = _np.where(w < 0, 0, w)
        return to_backend(V @ _np.diag(w) @ V.T)

def nearest_psd(mat: NDArray, *, max_iter: int = 50, tol: float = 1e-9) -> NDArray:
    """Nearest PSD via Higham's projection (simple implementation)."""
    A = to_backend(mat)
    A = symmetrize(A)
    Y = A.copy()
    delta = xp.zeros_like(A)
    for _ in range(int(max_iter)):
        R = Y - delta
        # eigen clip
        w, V = xp.linalg.eigh(R)
        w = xp.where(w < 0, 0, w)
        X = V @ xp.diag(w) @ V.T
        delta = X - R
        Y = X
        if float(xp.linalg.norm(delta)) <= tol:
            break
    return Y

# ------------------------------- core stats ----------------------------------

def stat_center(series: SeriesND, *, mean: bool = True) -> SeriesND:
    """Center matrices column-wise (subtract mean) or return mean row (1×W)."""
    ser = ensure_matrix_series(series, "stat_center")
    if mean:
        return ser.apply(lambda m: (m - xp.mean(m, axis=0)))
    return ser.apply(lambda m: xp.mean(m, axis=0)[None, :])

def stat_scale(series: SeriesND, *, with_mean: bool = True, with_std: bool = True, ddof: int = 1) -> SeriesND:
    """Standardize matrices column-wise (Z-score); guards std=0."""
    ser = ensure_matrix_series(series, "stat_scale")
    def _sc(m: NDArray) -> NDArray:
        X = m
        if with_mean:
            X = X - xp.mean(X, axis=0)
        if with_std:
            sd = xp.std(X, axis=0, ddof=ddof)
            sd = xp.where(sd == 0, 1, sd)
            X = X / sd
        return X
    return ser.apply(_sc)

def stat_var(series: SeriesND, *, ddof: int = 1) -> SeriesND:
    ser = ensure_matrix_series(series, "stat_var")
    return ser.apply(lambda m: xp.var(m, axis=0, ddof=ddof)[None, :])

def stat_std(series: SeriesND, *, ddof: int = 1) -> SeriesND:
    ser = ensure_matrix_series(series, "stat_std")
    return ser.apply(lambda m: xp.std(m, axis=0, ddof=ddof)[None, :])

def stat_cov(series: SeriesND, *, ddof: int = 1) -> SeriesND:
    ser = ensure_matrix_series(series, "stat_cov")
    return ser.apply(lambda m: xp.cov(m, rowvar=False, ddof=ddof))

def stat_corr(series: SeriesND, *, ddof: int = 1) -> SeriesND:
    covs = stat_cov(series, ddof=ddof)
    def _corr(C: NDArray) -> NDArray:
        std = xp.sqrt(xp.diag(C))
        denom = std[:, None] * std[None, :]
        return xp.where(denom == 0, 0, C / denom)
    return covs.apply(_corr)

# ------------------------- eigen, cholesky & whitening -----------------------

def eig_sorted(mat: NDArray, *, descending: bool = True) -> Tuple[NDArray, NDArray]:
    """Eigenvalues/eigenvectors sorted (desc by default)."""
    w, V = xp.linalg.eigh(to_backend(mat))
    order = xp.argsort(w)
    if descending:
        order = order[::-1]
    return w[order], V[:, order]

def cholesky_psd(C: NDArray, *, jitter: float = 1e-10) -> NDArray:
    """Cholesky factor of PSD matrix with jitter if needed (returns lower-triangular)."""
    A = to_backend(C)
    try:
        return xp.linalg.cholesky(A)
    except Exception:
        return xp.linalg.cholesky(make_psd(A, jitter=jitter))

def chol_inverse(C: NDArray, *, jitter: float = 1e-10) -> NDArray:
    """Inverse using Cholesky; more stable than eig for PSD matrices."""
    L = cholesky_psd(C, jitter=jitter)  # C = L L^T
    Linv = xp.linalg.inv(L)
    return Linv.T @ Linv

def whiten_zca(C: NDArray, *, eps: float = 1e-8) -> NDArray:
    """ZCA whitening matrix C^(-1/2) using eigendecomposition."""
    w, V = eig_sorted(C, descending=False)
    w = xp.where(w < eps, eps, w)
    W = V @ xp.diag(1.0 / xp.sqrt(w)) @ V.T
    return W

def whiten_pca(C: NDArray, *, eps: float = 1e-8) -> NDArray:
    """PCA whitening using eigendecomposition (columns are eigenvectors)."""
    w, V = eig_sorted(C, descending=True)
    w = xp.where(w < eps, eps, w)
    return xp.diag(1.0 / xp.sqrt(w)) @ V.T

# ---------------------------------- PCA -------------------------------------

def _pca_from_cov(C: NDArray, *, n_components: Optional[int]) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return (eigvals_pos, eigvecs_top_k, eigvals_top_k, ratio_top_k)."""
    w, V = eig_sorted(C, descending=True)
    w_pos = xp.where(w < 0, 0, w)
    tot = float(xp.sum(w_pos)) or 1.0
    k = int(n_components) if n_components is not None else int(len(w_pos))
    w_k = w_pos[:k]
    V_k = V[:, :k]
    ratio = to_numpy(w_k / tot)
    return to_backend(w_pos), to_backend(V_k), to_backend(w_k), to_backend(ratio)

def pca_decompose(series: SeriesND, *, n_components: Optional[int] = None, center: bool = True) -> Dict[str, _pd.DataFrame]:
    """Fit PCA per matrix → {'components','explained_var','explained_ratio'} DataFrames."""
    ser = ensure_matrix_series(series, "pca_decompose")
    comps, vars_, ratios = [], [], []
    k_first = None
    n_features = ser.iloc[0].shape[1] if len(ser) else 0
    for _, X in ser.items():
        Xc = X - xp.mean(X, axis=0) if center else X
        C = xp.cov(Xc, rowvar=False)
        w_pos, V_k, w_k, ratio = _pca_from_cov(C, n_components=n_components)
        k = V_k.shape[1]
        if k_first is None:
            k_first = k
        comps.append(to_numpy(V_k).flatten())
        vars_.append(to_numpy(w_k))
        ratios.append(to_numpy(ratio))
    comp_cols = [f"PC{j+1}_f{i+1}" for j in range(k_first or 0) for i in range(n_features)]
    var_cols  = [f"PC{j+1}" for j in range(k_first or 0)]
    comp_df  = _pd.DataFrame(_np.vstack(comps) if comps else _np.empty((0, len(comp_cols))), index=ser.index, columns=comp_cols)
    var_df   = _pd.DataFrame(_np.vstack(vars_) if vars_ else _np.empty((0, len(var_cols))), index=ser.index, columns=var_cols)
    ratio_df = _pd.DataFrame(_np.vstack(ratios) if ratios else _np.empty((0, len(var_cols))), index=ser.index, columns=var_cols)
    return {"components": comp_df, "explained_var": var_df, "explained_ratio": ratio_df}

def pca_components_df(series: SeriesND, *, n_components: Optional[int] = None, center: bool = True) -> _pd.DataFrame:
    """Return flattened component loadings only."""
    return pca_decompose(series, n_components=n_components, center=center)["components"]

def pca_transform(series: SeriesND, components: _pd.DataFrame, *, center: bool = True) -> SeriesND:
    """Project each matrix on `components` (from pca_decompose)."""
    ser = ensure_matrix_series(series, "pca_transform")
    if components.empty:
        return _pd.Series([], dtype="object")
    # infer K, F from columns: PC{j}_f{i}
    pcs   = sorted({int(c.split("_f")[0].replace("PC", "")) for c in components.columns})
    feats = sorted({int(c.split("_f")[1]) for c in components.columns})
    K, F = len(pcs), len(feats)
    scores: Dict[Any, NDArray] = {}
    for idx, X in ser.items():
        vec = components.loc[idx].to_numpy().reshape(K, F)
        W = to_backend(vec.T)  # F×K
        Xc = X - xp.mean(X, axis=0) if center else X
        S  = Xc @ W  # H×K
        scores[idx] = S
    return _pd.Series(scores, index=ser.index)

def pca_scores_df(series: SeriesND, components: _pd.DataFrame, *, center: bool = True) -> _pd.DataFrame:
    """Convenience: return a (rows × K) DataFrame of PCA scores."""
    scores = pca_transform(series, components, center=center)
    if scores.empty:
        return _pd.DataFrame()
    K = scores.iloc[0].shape[1]
    cols = [f"PC{j+1}" for j in range(K)]
    rows = [to_numpy(M).mean(axis=0) for M in scores.values]  # summary per row (mean across H)
    return _pd.DataFrame(_np.vstack(rows), index=scores.index, columns=cols)

def pca_reconstruct(series: SeriesND, components: _pd.DataFrame, *, center: bool = True) -> SeriesND:
    """Approximate reconstruction X̂ ≈ scores @ componentsᵀ (+mean if centered)."""
    ser = ensure_matrix_series(series, "pca_reconstruct")
    if components.empty:
        return _pd.Series([], dtype="object")
    pcs   = sorted({int(c.split("_f")[0].replace("PC", "")) for c in components.columns})
    feats = sorted({int(c.split("_f")[1]) for c in components.columns})
    K, F = len(pcs), len(feats)
    Xhat: Dict[Any, NDArray] = {}
    for idx, X in ser.items():
        vec = components.loc[idx].to_numpy().reshape(K, F)
        W   = to_backend(vec.T)  # F×K
        mu  = xp.mean(X, axis=0) if center else 0.0
        S   = (X - mu) @ W if center else X @ W
        Xr  = S @ W.T
        Xr  = Xr + mu if center else Xr
        Xhat[idx] = Xr
    return _pd.Series(Xhat, index=ser.index)

def pca_scree(series: SeriesND, *, center: bool = True) -> _pd.DataFrame:
    """Return cumulative explained ratio per sample (for scree plots)."""
    ser = ensure_matrix_series(series, "pca_scree")
    rows = []
    maxk = 0
    for _, X in ser.items():
        Xc = X - xp.mean(X, axis=0) if center else X
        C  = xp.cov(Xc, rowvar=False)
        w, _ = eig_sorted(C, descending=True)
        w    = xp.where(w < 0, 0, w)
        tot  = float(xp.sum(w)) or 1.0
        ratio_cum = _np.cumsum(to_numpy(w) / tot)
        rows.append(ratio_cum)
        maxk = max(maxk, len(ratio_cum))
    rows_pad = [_np.pad(r, (0, maxk - len(r)), mode="edge") for r in rows]
    cols     = [f"PC{j+1}" for j in range(maxk)]
    return _pd.DataFrame(_np.vstack(rows_pad), index=ser.index, columns=cols)

# === High-level helpers for common.helpers.summarize_series =================

def matrix_correlation(
    obj: Any,
    *,
    name: str = "matrix_correlation",
) -> _pd.DataFrame:
    """
    High-level helper: turn 'obj' into SeriesND of 2D matrices and compute
    a **single** correlation matrix that summarises the series.

    Design:
    -------
    - אם obj הוא:
        * Series של מטריצות (H×W) → כל מטריצה נפרסת לשורה (1×(H·W)),
          ואז מחשבים corr() בין השורות (correlation בין "מטריצות").
        * DataFrame / ndarray / רשימת מטריצות → אותו רעיון.
    - מתאים לשימוש של summarize_series: “כמה דומות המטריצות אחת לשניה”.
    """
    ser = ensure_matrix_series(obj, name)
    if ser.empty:
        return _pd.DataFrame()

    # flatten each matrix to a row vector
    rows = []
    idx = []
    for i, m in ser.items():
        A = to_numpy(m)
        rows.append(A.reshape(1, -1))
        idx.append(i)

    X = _np.vstack(rows)  # n_mats × (H·W)
    df = _pd.DataFrame(X, index=idx)
    return df.T.corr()  # correlation בין המטריצות (לא בין הפיצ'רים)


def matrix_covariance(
    obj: Any,
    *,
    name: str = "matrix_covariance",
) -> _pd.DataFrame:
    """
    High-level helper: כמו matrix_correlation אבל מחזיר covariance בין המטריצות.

    - flatten לכל מטריצה → שורה אחת.
    - cov() בין השורות.
    """
    ser = ensure_matrix_series(obj, name)
    if ser.empty:
        return _pd.DataFrame()

    rows = []
    idx = []
    for i, m in ser.items():
        A = to_numpy(m)
        rows.append(A.reshape(1, -1))
        idx.append(i)

    X = _np.vstack(rows)
    df = _pd.DataFrame(X, index=idx)
    cov = _np.cov(df.values, rowvar=True)
    return _pd.DataFrame(cov, index=idx, columns=idx)


def matrix_pca(
    obj: Any,
    n_components: Optional[int] = None,
    *,
    name: str = "matrix_pca",
) -> Dict[str, Any]:
    """
    High-level PCA over a series of matrices.

    - flatten לכל מטריצה → וקטור.
    - מפעיל PCA על כל הוקטורים (בדומה ל-feature space).
    - התוצאה תואמת לסכימה שה-fallback של helpers השתמש בה:
        {
            "components": DataFrame (loadings per PC),
            "explained_var": array,
            "explained_ratio": array,
        }
    """
    ser = ensure_matrix_series(obj, name)
    if ser.empty:
        return {
            "components": _pd.DataFrame(),
            "explained_var": _np.array([]),
            "explained_ratio": _np.array([]),
        }

    # flatten each matrix
    rows = []
    idx = []
    for i, m in ser.items():
        A = to_numpy(m)
        rows.append(A.reshape(-1))
        idx.append(i)

    X = _np.vstack(rows)  # n_mats × (H·W)
    X = X - _np.nanmean(X, axis=0, keepdims=True)
    X = _np.nan_to_num(X, nan=0.0)

    cov = _np.cov(X, rowvar=False)
    eigvals, eigvecs = _np.linalg.eigh(cov)
    order = _np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if n_components is not None and n_components < eigvals.size:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    total = float(eigvals.sum()) or 1.0
    ratio = eigvals / total

    comp_cols = [f"PC{j+1}" for j in range(eigvecs.shape[1])]
    components_df = _pd.DataFrame(eigvecs, index=range(eigvecs.shape[0]), columns=comp_cols)

    return {
        "components": components_df,
        "explained_var": eigvals,
        "explained_ratio": ratio,
    }


# --------------------- robust / shrinkage (optional) -------------------------

def robust_cov_mcd(series: SeriesND) -> SeriesND:
    """Minimum Covariance Determinant (requires scikit-learn)."""
    try:
        from sklearn.covariance import MinCovDet  # type: ignore
    except Exception as e:
        raise ImportError("robust_cov_mcd requires 'scikit-learn'") from e
    ser = ensure_matrix_series(series, "robust_cov_mcd")
    return ser.apply(lambda m: MinCovDet().fit(to_numpy(m)).covariance_)

def shrinkage_cov_lw(series: SeriesND) -> SeriesND:
    """Ledoit–Wolf shrinkage covariance (requires scikit-learn)."""
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore
    except Exception as e:
        raise ImportError("shrinkage_cov_lw requires 'scikit-learn'") from e
    ser = ensure_matrix_series(series, "shrinkage_cov_lw")
    return ser.apply(lambda m: LedoitWolf().fit(to_numpy(m)).covariance_)

# ---------------------------------------------------------------------------
# (End of Part 4/6)
# Next: Part 5/6 — Broadcast & Reshape utilities
# ---------------------------------------------------------------------------
# =============================================================================
# Part 5/6 — Broadcast & Reshape Utilities (HF-grade)
# =============================================================================
# NOTE: Dedup-safe (no new imports). Relies on Part 1 symbols:
#       xp, _np, _pd, NDArray, SeriesND, ensure_matrix_series, to_backend, to_numpy, logger

__all__.extend([
    # reshape & layout
    "reshape_safe", "flatten_row", "unflatten_row", "permute_axes", "repeat_interleave",
    # tile/pad/crop
    "tile_rows", "tile_cols", "pad_matrix", "crop_matrix", "pad_to_shape", "align_shapes",
    # concat/stack
    "concat_series_h", "concat_series_v", "stack_series",
    # algebra
    "block_diag_series", "kronecker_product",
])

# --------------------------- reshape & layout --------------------------------

def reshape_safe(mat: NDArray, shape: Tuple[int, int]) -> NDArray:
    """Reshape **with** exact size check (backend-aware)."""
    A = to_backend(mat)
    r, c = int(shape[0]), int(shape[1])
    if r * c != int(A.size):
        raise ValueError(f"reshape_safe: incompatible size {A.size} → {shape}")
    return A.reshape(r, c)

def flatten_row(series: SeriesND) -> SeriesND:
    """Flatten each matrix to a single **row** (1×(R·C))."""
    ser = ensure_matrix_series(series, "flatten_row")
    return ser.apply(lambda m: to_backend(m).reshape(1, -1))

def unflatten_row(series: SeriesND, *, shape: Tuple[int, int]) -> SeriesND:
    """Inverse of `flatten_row` — reshape row back to 2-D."""
    ser = ensure_matrix_series(series, "unflatten_row")
    r, c = int(shape[0]), int(shape[1])
    return ser.apply(lambda v: reshape_safe(v, (r, c)))

def permute_axes(series: SeriesND, order: Tuple[int, int]) -> SeriesND:
    """Swap axes (0,1) or (1,0) for each matrix."""
    ser = ensure_matrix_series(series, "permute_axes")
    if tuple(order) not in {(0, 1), (1, 0)}:
        raise ValueError("permute_axes: only 2D permutations supported: (0,1) or (1,0)")
    return ser.apply(lambda m: xp.swapaxes(m, order[0], order[1]))

def repeat_interleave(series: SeriesND, repeats: int, *, axis: int = 0) -> SeriesND:
    """Repeat rows/cols (`axis=0/1`) `repeats` times (guarded ≥1)."""
    ser = ensure_matrix_series(series, "repeat_interleave")
    rep = int(max(1, repeats))
    return ser.apply(lambda m: xp.repeat(m, rep, axis=int(axis)))

# --------------------------- tile / pad / crop -------------------------------

def tile_rows(series: SeriesND, reps: int) -> SeriesND:
    ser = ensure_matrix_series(series, "tile_rows")
    t = int(max(1, reps))
    return ser.apply(lambda m: xp.tile(m, (t, 1)))

def tile_cols(series: SeriesND, reps: int) -> SeriesND:
    ser = ensure_matrix_series(series, "tile_cols")
    t = int(max(1, reps))
    return ser.apply(lambda m: xp.tile(m, (1, t)))

def pad_matrix(series: SeriesND, target_shape: Tuple[int, int], *, value: float = 0.0, anchor: Literal["tl","br","center"] = "tl") -> SeriesND:
    """Pad each matrix to `target_shape` with constant `value` (top-left/center/bottom-right anchor)."""
    ser = ensure_matrix_series(series, "pad_matrix")
    R, C = int(target_shape[0]), int(target_shape[1])
    val = float(value)
    def _pad(m: NDArray) -> NDArray:
        r, c = m.shape
        if r > R or c > C:
            raise ValueError("pad_matrix: target smaller than input")
        out = xp.full((R, C), val, dtype=m.dtype)
        if anchor == "tl":
            out[:r, :c] = m
        elif anchor == "br":
            out[R-r:, C-c:] = m
        else:  # center
            rs = (R - r) // 2; cs = (C - c) // 2
            out[rs:rs+r, cs:cs+c] = m
        return out
    return ser.apply(_pad)

def crop_matrix(series: SeriesND, target_shape: Tuple[int, int], *, anchor: Literal["tl","br","center"] = "tl") -> SeriesND:
    """Crop each matrix **down** to `target_shape` by anchor (top-left/center/bottom-right)."""
    ser = ensure_matrix_series(series, "crop_matrix")
    R, C = int(target_shape[0]), int(target_shape[1])
    def _crop(m: NDArray) -> NDArray:
        r, c = m.shape
        if r < R or c < C:
            raise ValueError("crop_matrix: target larger than input")
        if anchor == "tl":
            return m[:R, :C]
        elif anchor == "br":
            return m[r-R:, c-C:]
        else:
            rs = (r - R) // 2; cs = (c - C) // 2
            return m[rs:rs+R, cs:cs+C]
    return ser.apply(_crop)

def pad_to_shape(series: SeriesND, shape: Tuple[int, int], *, value: float = 0.0) -> SeriesND:
    """Pad each matrix to **exact** shape (sugar for `pad_matrix`)."""
    ser = ensure_matrix_series(series, "pad_to_shape")
    return pad_matrix(ser, (int(shape[0]), int(shape[1])), value=value)

def align_shapes(series_list: List[SeriesND], *, mode: Literal["pad","crop"] = "pad", value: float = 0.0) -> List[SeriesND]:
    """Align multiple Series to a common shape by pad/crop (keeps original order)."""
    if not series_list:
        return []
    # derive common shape
    s0 = ensure_matrix_series(series_list[0], "align_shapes_0")
    R, C = s0.iloc[0].shape
    for s in series_list[1:]:
        s = ensure_matrix_series(s, "align_shapes_i")
        r, c = s.iloc[0].shape
        R, C = max(R, r), max(C, c)
    if mode == "pad":
        return [pad_matrix(ensure_matrix_series(s), (R, C), value=value) for s in series_list]
    else:
        return [crop_matrix(ensure_matrix_series(s), (R, C)) for s in series_list]

# --------------------------- concat / stack ---------------------------------

def _align_common_index(series_list: List[SeriesND]) -> Tuple[List[SeriesND], _pd.Index]:
    """Best-effort common index alignment across multiple Series."""
    if not series_list:
        return [], _pd.Index([])
    idx = series_list[0].index
    for s in series_list[1:]:
        if not s.index.equals(idx):
            idx = idx.intersection(s.index)
    return [s.loc[idx] for s in series_list], idx

def concat_series_h(series_list: List[SeriesND]) -> SeriesND:
    """Horizontal concat (stack **columns**; axis=1) with shape alignment by pad."""
    if not series_list:
        return _pd.Series([], dtype="object")
    aligned = align_shapes(series_list, mode="pad")
    aligned, idx = _align_common_index(aligned)
    out = {}
    for i in idx:
        mats = [s.loc[i] for s in aligned]
        out[i] = xp.concatenate(mats, axis=1)
    return _pd.Series(out, index=idx)

def concat_series_v(series_list: List[SeriesND]) -> SeriesND:
    """Vertical concat (stack **rows**; axis=0) with shape alignment by pad."""
    if not series_list:
        return _pd.Series([], dtype="object")
    aligned = align_shapes(series_list, mode="pad")
    aligned, idx = _align_common_index(aligned)
    out = {}
    for i in idx:
        mats = [s.loc[i] for s in aligned]
        out[i] = xp.concatenate(mats, axis=0)
    return _pd.Series(out, index=idx)

def stack_series(series: SeriesND, *, axis: int = 2) -> _pd.Series:
    """Stack each matrix along a **new** axis (default time/planes axis)."""
    ser = ensure_matrix_series(series, "stack_series")
    ax = int(axis)
    if ax not in (0, 1, 2):
        raise ValueError("stack_series: axis must be 0,1,2 for 2-D inputs")
    return _pd.Series({i: xp.expand_dims(m, axis=ax) for i, m in ser.items()}, index=ser.index)

# ------------------------------- algebra ------------------------------------

def block_diag_series(series_list: List[SeriesND]) -> SeriesND:
    """Block-diagonal compose multiple Series (pad-aligned), per index."""
    if not series_list:
        return _pd.Series([], dtype="object")
    aligned = align_shapes(series_list, mode="pad")
    aligned, idx = _align_common_index(aligned)
    out = {}
    for i in idx:
        blocks = [s.loc[i] for s in aligned]
        R = sum(b.shape[0] for b in blocks)
        C = sum(b.shape[1] for b in blocks)
        BD = xp.zeros((R, C), dtype=blocks[0].dtype)
        r0 = c0 = 0
        for b in blocks:
            r, c = b.shape
            BD[r0:r0+r, c0:c0+c] = b
            r0 += r; c0 += c
        out[i] = BD
    return _pd.Series(out, index=idx)

def kronecker_product(series_a: SeriesND, series_b: SeriesND) -> SeriesND:
    """Kronecker product A⊗B per index; aligns to common index first."""
    sa = ensure_matrix_series(series_a, "kronecker_a")
    sb = ensure_matrix_series(series_b, "kronecker_b")
    if not sa.index.equals(sb.index):
        common = sa.index.intersection(sb.index)
        sa = sa.loc[common]; sb = sb.loc[common]
    out = {}
    for i in sa.index:
        A = sa.loc[i]; B = sb.loc[i]
        try:
            kron = xp.kron(A, B)  # type: ignore[attr-defined]
        except Exception:
            kron = to_backend(_np.kron(to_numpy(A), to_numpy(B)))
        out[i] = kron
    return _pd.Series(out, index=sa.index)

# ---------------------------------------------------------------------------
# (End of Part 5/6)
# Next: Part 6/6 — Persistence & Robust/Advanced Integrations
# ---------------------------------------------------------------------------
# =============================================================================
# Part 6/6 — Persistence & Robust/Advanced Integrations (HF-grade)
# =============================================================================
# NOTE: No new imports — dedup-safe. Relies on Part 1 symbols:
#       xp, _np, _pd, NDArray, SeriesND, ensure_matrix_series, to_backend, to_numpy, logger

__all__.extend([
    # tensor series IO
    "tensor_series_to_parquet", "tensor_series_from_parquet",
    "tensor_series_to_npz", "tensor_series_from_npz",
    # packaging
    "build_zip_bundle", "series_manifest", "sha256_bytes",
    # registry (multi-series) persistence
    "save_registry", "load_registry",
    # DuckDB/flat export (optional, best-effort)
    "flatten_series_table",
])

# ------------------------------- utilities -----------------------------------

def sha256_bytes(data: bytes) -> str:
    """Return SHA256 hex of *data* (tiny helper)."""
    try:
        import hashlib
        h = hashlib.sha256(); h.update(data)
        return h.hexdigest()
    except Exception:
        return ""

def series_manifest(series: SeriesND, *, meta: dict | None = None) -> dict:
    """Build a minimal manifest for a tensor series (for packaging)."""
    ser = ensure_matrix_series(series, "series_manifest")
    shape = tuple(int(x) for x in (ser.iloc[0].shape if len(ser) else (0, 0)))
    return {
        "count": int(len(ser)),
        "shape": shape,
        "index_type": type(ser.index).__name__,
        "backend": backend(),
        "extra": meta or {},
    }

# -------------------------- tensor series ⇄ Parquet --------------------------

def tensor_series_to_parquet(series: SeriesND, path: str) -> None:
    """Write a Series of 2-D matrices to Parquet (one row per item, flattened).

    Columns are named c0..c{R*C-1}; original index is preserved in a column
    "__index__" (stringified). This is a CPU-friendly export (NumPy space).
    """
    ser = ensure_matrix_series(series, "tensor_series_to_parquet")
    rows = []
    for idx, mat in ser.items():
        v = to_numpy(mat).reshape(1, -1)
        rows.append((str(idx), v))
    if not rows:
        _pd.DataFrame({"__index__": []}).to_parquet(path, index=False)
        return
    idxs = [r[0] for r in rows]
    data = _np.vstack([r[1] for r in rows])
    cols = [f"c{i}" for i in range(data.shape[1])]
    df = _pd.DataFrame(data, columns=cols)
    df.insert(0, "__index__", idxs)
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        raise RuntimeError(f"tensor_series_to_parquet: failed to write '{path}': {e}")

def tensor_series_from_parquet(path: str, *, shape: tuple[int, int]) -> SeriesND:
    """Read Parquet written by `tensor_series_to_parquet` back to a Series.

    You must provide the *shape* (R,C) to unflatten.
    """
    try:
        df = _pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(f"tensor_series_from_parquet: failed to read '{path}': {e}")
    if "__index__" not in df.columns:
        raise ValueError("tensor_series_from_parquet: missing __index__ column")
    cols = [c for c in df.columns if c != "__index__"]
    R, C = int(shape[0]), int(shape[1])
    out = {}
    for _, row in df.iterrows():
        v = row[cols].to_numpy(dtype=float)
        if v.size != R * C:
            raise ValueError(f"row has {v.size} values but shape={shape} requires {R*C}")
        out[row["__index__"]] = to_backend(v.reshape(R, C))
    return ensure_matrix_series(_pd.Series(out), "tensor_series_from_parquet")

# ---------------------------- tensor series ⇄ NPZ ----------------------------

def tensor_series_to_npz(series: SeriesND, path: str, *, overwrite: bool = False, compressed: bool = True) -> None:
    """Store the Series as NPZ where each key is the stringified index.

    Creates an adjacent `.meta.json` with a manifest for reproducibility.
    """
    ser = ensure_matrix_series(series, "tensor_series_to_npz")
    path = str(path)
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File exists: {path}")
    # build dict of arrays (CPU for portability)
    arrays = {str(i): to_numpy(m) for i, m in ser.items()}
    if compressed:
        _np.savez_compressed(path, **arrays)
    else:
        _np.savez(path, **arrays)
    # write manifest
    try:
        manifest = series_manifest(ser)
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

def tensor_series_from_npz(path: str) -> SeriesND:
    """Load NPZ written by `tensor_series_to_npz` into a Series (NumPy arrays)."""
    try:
        data = _np.load(path)
    except Exception as e:
        raise RuntimeError(f"tensor_series_from_npz: failed to read '{path}': {e}")
    mats = {k: data[k] for k in data.files}
    # preserve natural order (sorted by key)
    keys = sorted(mats.keys())
    ser = _pd.Series([mats[k] for k in keys], index=keys, dtype="object")
    return ensure_matrix_series(ser, "tensor_series_from_npz")

# -------------------------------- packaging ---------------------------------

def build_zip_bundle(series: SeriesND, zip_path: str, *, include_parquet: bool = True) -> str:
    """Package a tensor series into a ZIP: CSV(flat), optional Parquet, SHA256 and manifest.

    Returns the produced zip path.
    """
    ser = ensure_matrix_series(series, "build_zip_bundle")
    import io, zipfile
    # CSV (flat)
    rows = []
    for idx, m in ser.items():
        v = to_numpy(m).reshape(1, -1)
        rows.append((str(idx), v))
    if rows:
        idxs = [r[0] for r in rows]
        data = _np.vstack([r[1] for r in rows])
        cols = [f"c{i}" for i in range(int(data.shape[1]))]
        df = _pd.DataFrame(data, columns=cols)
        df.insert(0, "__index__", idxs)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
    else:
        csv_bytes = b"__index__\n"
    # Manifest
    manifest = series_manifest(ser)
    # Optional Parquet
    pq_bytes = b""
    if include_parquet:
        try:
            import pyarrow as pa, pyarrow.parquet as pq
            table = pa.Table.from_pandas(df if rows else _pd.DataFrame({"__index__": []}))
            bio = io.BytesIO(); pq.write_table(table, bio)
            pq_bytes = bio.getvalue()
        except Exception:
            pq_bytes = b""
    # Write ZIP
    out_zip = str(zip_path)
    with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("series.csv", csv_bytes)
        zf.writestr("series.sha256", sha256_bytes(csv_bytes))
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        if pq_bytes:
            zf.writestr("series.parquet", pq_bytes)
    return out_zip

# ------------------------------- DuckDB export -------------------------------

def flatten_series_table(series: SeriesND, *, include_index: bool = True) -> _pd.DataFrame:
    """Return a long DataFrame: (index, r, c, value) for each matrix cell.

    Useful to push into SQL engines (DuckDB) with simple schema.
    """
    ser = ensure_matrix_series(series, "flatten_series_table")
    recs = []
    for idx, m in ser.items():
        M = to_numpy(m)
        R, C = int(M.shape[0]), int(M.shape[1])
        for r in range(R):
            for c in range(C):
                if include_index:
                    recs.append((str(idx), r, c, float(M[r, c])))
                else:
                    recs.append((r, c, float(M[r, c])))
    cols = ["index", "r", "c", "value"] if include_index else ["r", "c", "value"]
    return _pd.DataFrame.from_records(recs, columns=cols)

# ------------------------------- registry IO --------------------------------

def save_registry(registry: dict[str, SeriesND], path: str, *, overwrite: bool = False) -> None:
    """Save a dict[name → SeriesND] into a single NPZ (namespaced keys)."""
    path = str(path)
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File exists: {path}")
    pkg: dict[str, _np.ndarray] = {}
    meta: dict[str, dict] = {}
    for name, ser in registry.items():
        ser2 = ensure_matrix_series(ser, f"save_registry:{name}")
        for idx, m in ser2.items():
            key = f"{name}:::{str(idx)}"
            pkg[key] = to_numpy(m)
        meta[name] = series_manifest(ser2)
    _np.savez_compressed(path, **pkg)
    try:
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_registry(path: str) -> dict[str, SeriesND]:
    """Load a registry saved by `save_registry` (dict[name → SeriesND])."""
    data = _np.load(path)
    buckets: dict[str, dict[str, _np.ndarray]] = {}
    for k in data.files:
        if ":::" not in k:
            # legacy/unknown — skip
            continue
        name, idx = k.split(":::", 1)
        buckets.setdefault(name, {})[idx] = data[k]
    out: dict[str, SeriesND] = {}
    for name, mats in buckets.items():
        keys = sorted(mats.keys())
        ser = _pd.Series([mats[k] for k in keys], index=keys, dtype="object")
        out[name] = ensure_matrix_series(ser, f"load_registry:{name}")
    return out

# =============================================================================
# Tail-Dependence Helpers (HF-grade, market-aware)
# =============================================================================
# NOTE:
# - נשארים backend-agnostic (NumPy לשימוש העיקרי כאן מספיק).
# - import של load_price_data מתבצע לוקלית בתוך הפונקציה
#   כדי להקטין סיכוי ל-circular imports.

__all__.extend([
    "compute_tail_dep_matrix",
])


def _pair_to_label_and_legs_for_tail(pair_obj: Any) -> Tuple[str, Tuple[str, ...]]:
    """
    Normalize pair representation into (label, legs) for tail-dependence.

    תומך ב:
    - "AAPL"                    -> label="AAPL", legs=("AAPL",)
    - ("XOM", "CVX")            -> label="XOM-CVX", legs=("XOM","CVX")
    - {"symbol_1": "A", "symbol_2": "B"}
    - {"leg1": "A", "leg2": "B"}
    """
    if isinstance(pair_obj, str):
        return pair_obj, (pair_obj,)

    if isinstance(pair_obj, (tuple, list)) and len(pair_obj) >= 2:
        s1 = str(pair_obj[0])
        s2 = str(pair_obj[1])
        return f"{s1}-{s2}", (s1, s2)

    if isinstance(pair_obj, dict):
        if "symbol_1" in pair_obj and "symbol_2" in pair_obj:
            s1 = str(pair_obj["symbol_1"])
            s2 = str(pair_obj["symbol_2"])
            return f"{s1}-{s2}", (s1, s2)
        if "leg1" in pair_obj and "leg2" in pair_obj:
            s1 = str(pair_obj["leg1"])
            s2 = str(pair_obj["leg2"])
            return f"{s1}-{s2}", (s1, s2)

    raise TypeError(f"Unsupported pair format for tail-dependence: {type(pair_obj)!r}")


def _load_price_series_for_tail(
    symbol: str,
    start_date: Any,
    end_date: Any,
    field: str = "Close",
) -> _pd.Series:
    """
    טוען סדרת מחירים לסימול יחיד ומחזיר סדרה נקייה (float, dropna).

    - משתמש ב-load_price_data מהמערכת (import לוקלי).
    - חותך טווח תאריכים.
    - תומך בשמות עמודות כמו 'close' / 'adj_close' וכו'.
    """
    try:
        from common.utils import load_price_data  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.warning(
            "tail_dep: cannot import load_price_data from common.utils (%s) — returning empty series",
            e,
        )
        return _pd.Series(dtype=float)

    try:
        df = load_price_data(symbol)
    except Exception as e:
        logger.warning("tail_dep: load_price_data(%s) failed: %s", symbol, e)
        return _pd.Series(dtype=float)

    if df is None or df.empty:
        return _pd.Series(dtype=float)

    df = df.sort_index()
    # חתך טווח תאריכים
    try:
        start_ts = _pd.to_datetime(start_date)
        end_ts = _pd.to_datetime(end_date)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    except Exception:
        try:
            df = df.copy()
            df.index = _pd.to_datetime(df.index)
            start_ts = _pd.to_datetime(start_date)
            end_ts = _pd.to_datetime(end_date)
            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        except Exception as e:
            logger.warning("tail_dep: date filtering failed for %s (%s)", symbol, e)
            return _pd.Series(dtype=float)

    if df.empty:
        return _pd.Series(dtype=float)

    cols_lower = {c.lower(): c for c in df.columns}
    col_name = None
    for key in (str(field).lower(), "close", "adj_close"):
        if key in cols_lower:
            col_name = cols_lower[key]
            break
    if col_name is None:
        col_name = df.columns[0]

    s = df[col_name].astype(float)
    return s.dropna()


def _build_pair_return_series_for_tail(
    legs: Tuple[str, ...],
    start_date: Any,
    end_date: Any,
    field: str = "Close",
) -> _pd.Series:
    """
    בונה סדרת תשואות עבור "pair" לצורך Tail-Dependence:

    - leg יחיד: log-return של הסימול עצמו.
    - שני legs: spread-return בסיסי: log_ret(leg1) - log_ret(leg2).

    אפשר להחליף להגדרה מתקדמת יותר (beta-hedged / z-spread) בהמשך.
    """
    if not legs:
        return _pd.Series(dtype=float)

    if len(legs) == 1:
        px = _load_price_series_for_tail(legs[0], start_date, end_date, field=field)
        if px.empty:
            return _pd.Series(dtype=float)
        lr = _np.log(px).diff()
        return lr.dropna()

    # שני רגליים
    px1 = _load_price_series_for_tail(legs[0], start_date, end_date, field=field)
    px2 = _load_price_series_for_tail(legs[1], start_date, end_date, field=field)
    if px1.empty or px2.empty:
        return _pd.Series(dtype=float)

    df = _pd.concat([px1, px2], axis=1, join="inner")
    if df.shape[1] < 2:
        return _pd.Series(dtype=float)

    a = _np.log(df.iloc[:, 0]).diff()
    b = _np.log(df.iloc[:, 1]).diff()
    spread_ret = a - b
    return spread_ret.dropna()


def compute_tail_dep_matrix(
    pairs: Sequence[Any],
    start_date: Any,
    end_date: Any,
    *,
    field: str = "Close",
    q: float = 0.90,
) -> _pd.DataFrame:
    """
    compute_tail_dep_matrix — מטריצת Tail-Dependence ברמת זוגות/נכסים.

    Tail-Dependence מוגדרת כאן כ:

        P(|r_i| קיצוני ו-|r_j| קיצוני) / P(|r_i| קיצוני או |r_j| קיצוני)

    כלומר: מתוך כל הימים שבהם **לפחות אחד** מהשניים קיצוני, כמה פעמים **שניהם**
    קיצוניים יחד.

    Parameters
    ----------
    pairs : Sequence[Any]
        כל אובייקט שמייצג זוג/נכס, לדוגמה:
        - "AAPL"
        - ("XOM", "CVX")
        - {"symbol_1": "XOM", "symbol_2": "CVX"}
    start_date, end_date : Any
        טווח תאריכים (ימופו ל-Timestamp).
    field : str, default "Close"
        שם עמודת המחיר (case-insensitive).
    q : float, default 0.90
        Quantile לזנב (0.9 = 10% הימים הכי קיצוניים במונחי |תשואה|).

    Returns
    -------
    pandas.DataFrame
        מטריצה סימטרית N×N עם Tail-Dependence, index/columns = labels של ה-"pairs".
    """
    if not pairs:
        return _pd.DataFrame()

    # בונים map: label → סדרת תשואות
    series_map: Dict[str, _pd.Series] = {}
    for p in pairs:
        try:
            label, legs = _pair_to_label_and_legs_for_tail(p)
        except TypeError as e:
            logger.debug("tail_dep: skipping unsupported pair format %r (%s)", p, e)
            continue

        try:
            s_ret = _build_pair_return_series_for_tail(legs, start_date, end_date, field=field)
        except Exception as e:
            logger.warning("tail_dep: failed to build return series for %r: %s", p, e)
            continue

        if s_ret is None or s_ret.empty:
            continue

        # guard מפני label כפול
        lbl = label
        k = 1
        while lbl in series_map:
            lbl = f"{label}#{k}"
            k += 1
        series_map[lbl] = s_ret

    if not series_map:
        return _pd.DataFrame()

    # מאחדים לסדרת תשואות טבלאית (inner-join על המדד המשותף)
    df_ret = _pd.concat(series_map.values(), axis=1, join="inner")
    df_ret.columns = list(series_map.keys())
    df_ret = df_ret.replace([_np.inf, -_np.inf], _np.nan).dropna(how="all")

    if df_ret.shape[1] == 0:
        return _pd.DataFrame()

    # אם רק עמודה אחת – נחזיר מטריצה 1×1 עם 1
    if df_ret.shape[1] == 1:
        lbl = df_ret.columns[0]
        return _pd.DataFrame([[1.0]], index=[lbl], columns=[lbl])

    abs_ret = df_ret.abs()
    # quantile per column
    try:
        thr = abs_ret.quantile(q=float(q), axis=0)
    except Exception as e:
        logger.warning("tail_dep: quantile computation failed (q=%s): %s", q, e)
        return _pd.DataFrame()

    is_extreme = abs_ret.ge(thr)

    labels = list(df_ret.columns)
    n = len(labels)
    tail_dep = _np.full((n, n), _np.nan, dtype=float)

    for i in range(n):
        ei = is_extreme.iloc[:, i]
        for j in range(i, n):
            ej = is_extreme.iloc[:, j]
            both = ei & ej
            either = ei | ej
            denom = int(either.sum())
            if denom == 0:
                val = _np.nan
            else:
                val = float(both.sum() / denom)
            tail_dep[i, j] = val
            tail_dep[j, i] = val

    # אלכסון = 1
    for k in range(n):
        tail_dep[k, k] = 1.0

    return _pd.DataFrame(tail_dep, index=labels, columns=labels)

# ---------------------------------------------------------------------------
# (End of Part 6/6)
# You have completed the 6-part matrix_helpers build. 🎉
# ---------------------------------------------------------------------------
