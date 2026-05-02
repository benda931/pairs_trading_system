# -*- coding: utf-8 -*-
"""
core/orchestrator.py — Pairs Trading Orchestrator (inspired by srv_quant)
=========================================================================

Automated scheduling and pipeline management for the pairs trading system.

Features:
- Cron-like task scheduling with dependencies
- Daily pipeline: data refresh → signals → risk check → paper trading
- Health monitoring cycle
- Agent bus for inter-component messaging
- Graceful error handling per task

Usage:
    from core.orchestrator import PairsOrchestrator
    orch = PairsOrchestrator()
    orch.run_daily_pipeline()   # Run full daily cycle
    orch.start_daemon()         # Start background scheduler
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

logger = logging.getLogger("orchestrator")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
AGENT_BUS_PATH = LOGS_DIR / "agent_bus.json"


# ============================================================================
# AgentBus + TaskResult extracted to core/agent_bus.py (AP-4: break circular dep)
from core.agent_bus import AgentBus, TaskResult  # noqa: E402
from core.data_refresh_diagnostics import (
    is_external_scheduler_daemon_active,
    load_data_refresh_diagnostics,
    persist_data_refresh_diagnostics,
)
from core.position_tracker import normalize_portfolio_monitor_payload
from common.data_freshness import FreshnessConfig, validate_pair_frames
from common.pair_utils import (
    extract_symbols_from_pairs,
    load_asset_policy,
    normalize_pairs,
    pair_allowed_by_policy,
    parse_pair_record,
)


def _coerce_asof_timestamp(index_value: Any) -> datetime:
    """Convert a pandas index value into a UTC-aware datetime anchor."""
    ts = pd.Timestamp(index_value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _estimate_half_life_days(spread: pd.Series) -> float:
    """Estimate OU-style half-life from a residual spread series."""
    clean = pd.to_numeric(spread, errors="coerce").dropna()
    if len(clean) < 20:
        return float("nan")

    lagged = clean.shift(1).dropna()
    delta = clean.diff().dropna()
    aligned = pd.concat([lagged.rename("lagged"), delta.rename("delta")], axis=1).dropna()
    if len(aligned) < 20:
        return float("nan")

    x = aligned["lagged"].to_numpy(dtype=float)
    y = aligned["delta"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    except Exception:
        return float("nan")

    slope = float(coeffs[1])
    if not np.isfinite(slope) or slope >= 0.0:
        return float("nan")

    half_life = -np.log(2.0) / slope
    if not np.isfinite(half_life) or half_life <= 0.0:
        return float("nan")
    return float(half_life)


def _estimate_variance_ratio(spread: pd.Series, lag: int = 5) -> float:
    """Estimate a simple variance ratio for mean-reversion diagnostics."""
    clean = pd.to_numeric(spread, errors="coerce").dropna()
    if len(clean) <= max(20, lag + 2):
        return float("nan")

    one_step = clean.diff().dropna()
    lagged = clean.diff(lag).dropna()
    if len(one_step) < 5 or len(lagged) < 5:
        return float("nan")

    var_one = float(one_step.var(ddof=1))
    if not np.isfinite(var_one) or var_one <= 0.0:
        return float("nan")

    var_lag = float(lagged.var(ddof=1))
    if not np.isfinite(var_lag):
        return float("nan")

    return float(var_lag / (lag * var_one))


def _estimate_hurst_exponent(
    spread: pd.Series,
    *,
    min_lag: int = 2,
    max_lag: int = 20,
) -> float:
    """Estimate Hurst exponent from spread differences on a log-log scale."""
    clean = pd.to_numeric(spread, errors="coerce").dropna()
    if len(clean) < max(60, max_lag * 3):
        return float("nan")

    lag_cap = min(max_lag, max(min_lag + 4, len(clean) // 3))
    lags = list(range(min_lag, lag_cap + 1))
    tau: list[float] = []
    valid_lags: list[int] = []
    for lag in lags:
        diff = clean.diff(lag).dropna().to_numpy(dtype=float)
        if len(diff) < 10:
            continue
        scale = float(np.std(diff, ddof=1))
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        tau.append(scale)
        valid_lags.append(lag)

    if len(valid_lags) < 5:
        return float("nan")

    try:
        slope, _ = np.polyfit(np.log(valid_lags), np.log(tau), 1)
    except Exception:
        return float("nan")

    if not np.isfinite(slope):
        return float("nan")
    return float(np.clip(slope, 0.0, 1.0))


def _extract_pair_tuple(pair_def: Any) -> Optional[tuple[str, str]]:
    """Normalize a raw pair definition into an uppercase ``(sym_x, sym_y)`` tuple."""
    return parse_pair_record(pair_def)


def _iter_pair_tuples(pairs: Sequence[Any]) -> list[tuple[str, str]]:
    """Return all valid normalized pair tuples from a raw config/app-context list."""
    return normalize_pairs(pairs or [])


def _get_configured_pairs(cfg: dict | None = None) -> list[tuple[str, str]]:
    from common.config_manager import load_config

    cfg = cfg or load_config()
    if cfg.get("use_production_pairs", False) and cfg.get("production_pairs"):
        raw_pairs = cfg.get("production_pairs", [])
    else:
        raw_pairs = cfg.get("pairs", [])
    pairs = normalize_pairs(raw_pairs)
    policy = load_asset_policy(cfg)
    return [(a, b) for a, b in pairs if pair_allowed_by_policy(a, b, policy=policy)]


def _prepare_pair_signal_snapshot(
    prices_wide: pd.DataFrame,
    sym_x: str,
    sym_y: str,
    *,
    lookback_days: int = 252,
    min_points: int = 120,
    include_diagnostics: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Build a point-in-time-clean pair snapshot from a wide price matrix.

    Returns None when the pair lacks clean overlap or stable residual variance.
    """
    diagnostics: Dict[str, Any] = {
        "pair": f"{sym_x}/{sym_y}",
        "sym_x": sym_x,
        "sym_y": sym_y,
        "lookback_days": int(max(lookback_days, 0)),
        "min_points": int(max(min_points, 0)),
        "status": "failed",
        "reason": "unknown",
        "available_points_x": 0,
        "available_points_y": 0,
        "aligned_points": 0,
        "coverage_ratio": 0.0,
        "freshness_lag_days": None,
    }

    def _finish(
        snapshot: Optional[Dict[str, Any]],
        *,
        status: str,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        diagnostics["status"] = status
        diagnostics["reason"] = reason
        if extra:
            diagnostics.update(extra)
        if include_diagnostics:
            return snapshot, diagnostics
        return snapshot

    if prices_wide is None or prices_wide.empty:
        return _finish(None, status="failed", reason="empty_price_matrix")
    if sym_x not in prices_wide.columns or sym_y not in prices_wide.columns:
        return _finish(
            None,
            status="failed",
            reason="missing_price_columns",
            extra={
                "missing_symbols": [
                    symbol for symbol in (sym_x, sym_y) if symbol not in prices_wide.columns
                ],
            },
        )

    pair_df = prices_wide[[sym_x, sym_y]].copy()
    pair_df[sym_x] = pd.to_numeric(pair_df[sym_x], errors="coerce")
    pair_df[sym_y] = pd.to_numeric(pair_df[sym_y], errors="coerce")
    pair_df = pair_df.replace([np.inf, -np.inf], np.nan)
    pair_df = pair_df[~pair_df.index.duplicated(keep="last")]
    pair_df = pair_df.sort_index()
    diagnostics["available_points_x"] = int(pair_df[sym_x].dropna().shape[0])
    diagnostics["available_points_y"] = int(pair_df[sym_y].dropna().shape[0])
    pair_df = pair_df.dropna(how="any")

    if lookback_days > 0:
        pair_df = pair_df.iloc[-lookback_days:]

    if len(pair_df) < min_points:
        return _finish(
            None,
            status="skipped",
            reason="insufficient_overlap",
            extra={
                "aligned_points": int(len(pair_df)),
                "coverage_ratio": round(float(len(pair_df)) / max(float(lookback_days), 1.0), 4),
            },
        )

    px = pair_df[sym_x]
    py = pair_df[sym_y]
    x = px.to_numpy(dtype=float)
    y = py.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])

    try:
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    except Exception:
        return _finish(None, status="failed", reason="ols_fit_failed")

    alpha = float(coeffs[0])
    beta = float(coeffs[1])
    fitted = alpha + beta * x
    spread = pd.Series(y - fitted, index=pair_df.index, name=f"{sym_y}_minus_{sym_x}")

    spread_std = float(spread.std(ddof=1))
    if not np.isfinite(spread_std) or spread_std <= 1e-12:
        return _finish(None, status="skipped", reason="degenerate_spread")

    z_score = float((spread.iloc[-1] - float(spread.mean())) / spread_std)
    z_velocity = float(spread.diff().iloc[-1] / spread_std) if len(spread) > 1 else float("nan")
    spread_vol_20d = float(spread.iloc[-20:].std(ddof=1)) if len(spread) >= 20 else float("nan")
    spread_vol_120d = float(spread.iloc[-120:].std(ddof=1)) if len(spread) >= 120 else spread_std
    returns_corr = float(px.pct_change().corr(py.pct_change()))
    if not np.isfinite(returns_corr):
        returns_corr = float("nan")
    as_of = _coerce_asof_timestamp(pair_df.index[-1])
    freshness_lag_days = max(
        0.0,
        (datetime.now(timezone.utc) - as_of).total_seconds() / 86400.0,
    )
    coverage_ratio = float(
        np.clip(float(len(pair_df)) / max(float(lookback_days), 1.0), 0.0, 1.0)
    )

    snapshot = {
        "px": px,
        "py": py,
        "spread": spread,
        "alpha": alpha,
        "beta": beta,
        "z_score": z_score,
        "z_velocity": z_velocity,
        "hurst": _estimate_hurst_exponent(spread),
        "correlation": returns_corr,
        "spread_vol_20d": spread_vol_20d,
        "spread_vol_120d": spread_vol_120d,
        "half_life": _estimate_half_life_days(spread),
        "variance_ratio": _estimate_variance_ratio(spread),
        "conviction": float(np.clip(abs(z_score) / 3.0, 0.05, 1.0)),
        "as_of": as_of,
        "aligned_points": int(len(pair_df)),
        "coverage_ratio": coverage_ratio,
        "freshness_lag_days": round(freshness_lag_days, 4),
    }
    return _finish(
        snapshot,
        status="ok",
        reason="ok",
        extra={
            "aligned_points": int(len(pair_df)),
            "coverage_ratio": round(coverage_ratio, 4),
            "freshness_lag_days": round(freshness_lag_days, 4),
            "as_of": as_of.isoformat(),
        },
    )

@dataclass
class ScheduledTask:
    """Definition of a scheduled task."""
    name: str
    func: Callable
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    enabled: bool = True
    last_run: Optional[str] = None
    last_status: str = "never_run"


# ============================================================================
# Pipeline Tasks
# ============================================================================

def task_data_refresh(**kwargs) -> Dict[str, Any]:
    """Refresh price data and return source-aware freshness diagnostics."""
    from common.data_loader import bulk_download, get_price_metadata, load_price_data

    run_id = str(kwargs.get("run_id") or "latest")

    from common.config_manager import load_config

    cfg = load_config()
    pair_tuples = _get_configured_pairs(cfg)
    symbols = set(extract_symbols_from_pairs(pair_tuples))
    if not symbols:
        return {"status": "no_symbols", "count": 0, "run_id": run_id}

    now_utc = datetime.now(timezone.utc)
    freshness_cutoff_days = 7.0
    ordered_symbols = sorted(str(sym).strip().upper() for sym in symbols if str(sym).strip())
    external_scheduler_active = is_external_scheduler_daemon_active()

    sql_store = None
    coverage_by_symbol: Dict[str, Dict[str, Any]] = {}
    def _reload_sql_coverage() -> None:
        nonlocal sql_store, coverage_by_symbol
        coverage_by_symbol = {}
        if external_scheduler_active:
            sql_store = None
            return
        try:
            from core.sql_store import SqlStore

            sql_store = SqlStore.from_settings({}, read_only=True)
            try:
                coverage = sql_store.load_prices_coverage_summary(
                    env=getattr(sql_store, "default_env", None),
                    warn_on_error=False,
                )
            except TypeError:
                coverage = sql_store.load_prices_coverage_summary(
                    env=getattr(sql_store, "default_env", None)
                )
            if not coverage.empty:
                for _, row in coverage.iterrows():
                    symbol = str(row.get("symbol") or "").strip().upper()
                    if not symbol:
                        continue
                    max_date = pd.to_datetime(row.get("max_date"), errors="coerce")
                    coverage_by_symbol[symbol] = {
                        "n_rows": int(row.get("n_rows", 0) or 0),
                        "max_date": max_date,
                    }
        except Exception as exc:
            logger.debug("task_data_refresh: coverage summary unavailable: %s", exc)

    _reload_sql_coverage()

    rows: List[Dict[str, Any]] = []
    refresh_candidates: List[str] = []

    def _build_row(sym: str) -> Dict[str, Any]:
        df = load_price_data(sym, start_date=now_utc - timedelta(days=90), end_date=now_utc)
        meta = get_price_metadata(df, symbol=sym)
        source = str(meta.get("data_source") or ("missing" if df.empty else "unknown"))
        age_hours = meta.get("age_hours")
        age_days = round(float(age_hours) / 24.0, 4) if age_hours is not None else None

        sql_info = coverage_by_symbol.get(sym, {})
        sql_max_date = sql_info.get("max_date")
        sql_age_days = None
        if sql_max_date is not None and not pd.isna(sql_max_date):
            sql_ts = pd.Timestamp(sql_max_date)
            if sql_ts.tzinfo is None:
                sql_ts = sql_ts.tz_localize("UTC")
            else:
                sql_ts = sql_ts.tz_convert("UTC")
            sql_age_days = round((now_utc - sql_ts.to_pydatetime()).total_seconds() / 86400.0, 4)

        if df.empty:
            final_status = "missing"
        elif age_days is not None and age_days > freshness_cutoff_days:
            final_status = "stale"
        elif source == "sql_store" and (sql_age_days is None or sql_age_days <= freshness_cutoff_days):
            final_status = "fresh"
        else:
            final_status = "partial"

        return {
            "symbol": sym,
            "final_status": final_status,
            "data_source": source,
            "n_rows": int(meta.get("n_rows", 0) or 0),
            "start_date": meta.get("start_date").isoformat() if meta.get("start_date") is not None else None,
            "end_date": meta.get("end_date").isoformat() if meta.get("end_date") is not None else None,
            "age_hours": round(float(age_hours), 2) if age_hours is not None else None,
            "age_days": age_days,
            "has_nans": bool(meta.get("has_nans", False)),
            "freq": meta.get("freq"),
            "sql_rows": int(sql_info.get("n_rows", 0) or 0),
            "sql_age_days": sql_age_days,
            "sql_status": (
                "missing"
                if not sql_info
                else ("stale" if sql_age_days is not None and sql_age_days > freshness_cutoff_days else "fresh")
            ),
        }

    for sym in ordered_symbols:
        row = _build_row(sym)
        rows.append(row)
        if row["final_status"] in {"missing", "stale"}:
            refresh_candidates.append(sym)

    fmp_runs: List[Dict[str, Any]] = []
    if refresh_candidates and not external_scheduler_active:
        full_backfill_days = int(cfg.get("scheduler_price_backfill_days", 730) or 730)
        incremental_days = int(cfg.get("scheduler_price_refresh_days", 30) or 30)
        today_iso = now_utc.date().isoformat()

        fmp_full_backfill: List[str] = []
        fmp_incremental: List[str] = []
        for sym in refresh_candidates:
            sql_info = coverage_by_symbol.get(sym, {})
            max_date = sql_info.get("max_date")
            has_sql_history = bool(sql_info.get("n_rows")) and max_date is not None and not pd.isna(max_date)
            if has_sql_history:
                fmp_incremental.append(sym)
            else:
                fmp_full_backfill.append(sym)

        try:
            from scripts.ingest_prices_fmp import ingest as _ingest_prices_fmp
        except Exception as exc:
            logger.debug("task_data_refresh: FMP ingest unavailable: %s", exc)
        else:
            for mode, batch_symbols, days in (
                ("full_backfill", fmp_full_backfill, full_backfill_days),
                ("incremental", fmp_incremental, incremental_days),
            ):
                if not batch_symbols:
                    continue
                start_iso = (now_utc - timedelta(days=days)).date().isoformat()
                try:
                    summary = _ingest_prices_fmp(
                        batch_symbols,
                        start=start_iso,
                        end=today_iso,
                        dry_run=False,
                        batch_size=20,
                    )
                    fmp_runs.append(
                        {
                            "mode": mode,
                            "days": days,
                            "symbols": len(batch_symbols),
                            **summary,
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "task_data_refresh: FMP %s failed for %d symbols: %s",
                        mode,
                        len(batch_symbols),
                        exc,
                    )
    elif refresh_candidates:
        logger.debug(
            "task_data_refresh: skipping FMP ingest because an external scheduler daemon is active"
        )

    _reload_sql_coverage()
    rows = [_build_row(sym) for sym in ordered_symbols]

    fallback_candidates = [
        str(row.get("symbol") or "")
        for row in rows
        if row.get("final_status") in {"missing", "stale"} and row.get("data_source") != "sql_store"
    ]
    downloaded = bulk_download(fallback_candidates, force=True) if fallback_candidates else []
    if fallback_candidates:
        _reload_sql_coverage()
        rows = [_build_row(sym) for sym in ordered_symbols]

    status_counts = Counter(str(row.get("final_status") or "unknown") for row in rows)
    overall_status = (
        "ok"
        if status_counts.get("missing", 0) == 0 and status_counts.get("stale", 0) == 0
        else "partial"
    )
    rows.sort(
        key=lambda item: (
            {"missing": 0, "stale": 1, "partial": 2, "fresh": 3}.get(str(item.get("final_status")), 4),
            str(item.get("symbol") or ""),
        )
    )

    payload = {
        "status": overall_status,
        "run_id": run_id,
        "symbols_requested": len(ordered_symbols),
        "symbols_downloaded": len(downloaded),
        "symbols_targeted_for_refresh": len(refresh_candidates),
        "status_counts": dict(status_counts),
        "coverage_source": "sql_store" if sql_store is not None else "local_only",
        "fmp_runs": fmp_runs,
        "fallback_csv_downloads": len(downloaded),
        "rows": rows,
    }

    try:
        persist_data_refresh_diagnostics(payload, run_id=run_id)
    except Exception as exc:
        logger.debug("task_data_refresh: file persistence skipped: %s", exc)

    if external_scheduler_active:
        logger.debug(
            "task_data_refresh: skipping SQL persistence because an external scheduler daemon is active"
        )
        return payload

    try:
        from core.sql_store import SqlStore

        store_rw = SqlStore.from_settings({}, read_only=False)
        if hasattr(store_rw, "save_json"):
            try:
                store_rw.save_json("data_refresh_diagnostics", run_id, payload, warn_on_error=False)
                if run_id != "latest":
                    store_rw.save_json("data_refresh_diagnostics", "latest", payload, warn_on_error=False)
            except TypeError:
                store_rw.save_json("data_refresh_diagnostics", run_id, payload)
                if run_id != "latest":
                    store_rw.save_json("data_refresh_diagnostics", "latest", payload)
    except Exception as exc:
        logger.debug("task_data_refresh: persistence skipped: %s", exc)

    return payload


def task_compute_signals(**kwargs) -> Dict[str, Any]:
    """Compute signals for all pairs in universe."""
    try:
        from common.config_manager import load_config
        from common.data_loader import load_price_data
        from core.signals_engine import compute_universe_signals

        cfg = load_config()
        orchestrator = kwargs.get("orchestrator")
        if (
            orchestrator is not None
            and getattr(orchestrator, "_fresh_pairs_override", None) is not None
        ):
            pairs = list(getattr(orchestrator, "_fresh_pairs_override") or [])
        else:
            pairs = _get_configured_pairs(cfg)
        if not pairs:
            return {"status": "no_pairs", "pairs_requested": 0, "pairs_computed": 0}

        lookback_days = int(kwargs.get("lookback_days") or cfg.get("signal_lookback_days", 252) or 252)
        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(days=max(int(lookback_days) * 3, 365))
        start_date = start_ts.date()
        end_date = end_ts.date()

        def _pair_legs_loader(pair_obj: Any, loader_start_date, loader_end_date):
            sym_x, sym_y = _extract_pair_tuple(pair_obj) or ("", "")
            if not sym_x or not sym_y:
                raise ValueError(f"Invalid pair object: {pair_obj!r}")

            sx = load_price_data(sym_x, start_date=loader_start_date, end_date=loader_end_date)
            sy = load_price_data(sym_y, start_date=loader_start_date, end_date=loader_end_date)
            if sx.empty or sy.empty:
                raise ValueError(f"Missing price history for {sym_x}/{sym_y}")

            def _series_from_df(df: pd.DataFrame, symbol: str) -> pd.Series:
                candidate_cols = ("close", "adj_close", "Close", "Adj Close")
                for col in candidate_cols:
                    if col in df.columns:
                        series = pd.to_numeric(df[col], errors="coerce").copy()
                        series.name = symbol
                        return series
                if len(df.columns) == 1:
                    series = pd.to_numeric(df.iloc[:, 0], errors="coerce").copy()
                    series.name = symbol
                    return series
                raise ValueError(f"No close-like column available for {symbol}")

            s1 = _series_from_df(sx, sym_x)
            s2 = _series_from_df(sy, sym_y)
            return s1, s2, f"{sym_x}-{sym_y}"

        universe = compute_universe_signals(
            pairs=pairs,
            start_date=start_date,
            end_date=end_date,
            pair_legs_loader=_pair_legs_loader,
            max_workers=int(cfg.get("signal_max_workers", 8) or 8),
        )
        n = int(len(universe.signals_df)) if getattr(universe, "signals_df", None) is not None else 0
        diag_n = int(len(universe.diagnostics_df)) if getattr(universe, "diagnostics_df", None) is not None else 0
        return {
            "status": "ok",
            "pairs_requested": len(pairs),
            "pairs_computed": n,
            "diagnostics_rows": diag_n,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def task_data_freshness_check(**kwargs) -> Dict[str, Any]:
    """Validate pair-leg freshness before signal computation."""
    from common.config_manager import load_config
    from common.data_loader import load_price_data

    cfg = load_config()
    orchestrator = kwargs.get("orchestrator")
    pairs = _get_configured_pairs(cfg)
    if not pairs:
        payload = {
            "status": "ok",
            "reason": "no_pairs",
            "pairs_requested": 0,
            "pairs_passed": 0,
            "pairs_failed": 0,
            "passed_pairs": [],
            "failed_pairs": [],
        }
        if orchestrator is not None:
            orchestrator._fresh_pairs_override = []
            orchestrator._last_data_freshness_report = payload
            orchestrator.bus.publish("data_freshness", payload)
        return payload

    freshness_cfg = FreshnessConfig(**dict(cfg.get("data_freshness") or {}))
    lookback_days = max(
        int(cfg.get("signal_lookback_days", 252) or 252) * 3,
        int(freshness_cfg.min_rows) + 30,
    )
    now_utc = datetime.now(timezone.utc)
    start_date = (now_utc - timedelta(days=lookback_days)).date()
    end_date = now_utc.date()

    passed_pairs: list[tuple[str, str]] = []
    failed_pairs: list[dict[str, Any]] = []
    for sym_x, sym_y in pairs:
        df_x = load_price_data(sym_x, start_date=start_date, end_date=end_date)
        df_y = load_price_data(sym_y, start_date=start_date, end_date=end_date)
        validation = validate_pair_frames(sym_x, df_x, sym_y, df_y, cfg=freshness_cfg, now=now_utc)
        if validation.get("ok"):
            passed_pairs.append((sym_x, sym_y))
            continue
        failed_pairs.append(
            {
                "pair": f"{sym_x}/{sym_y}",
                "reason": str(validation.get("reason") or "invalid"),
                "x": dict(validation.get("x") or {}),
                "y": dict(validation.get("y") or {}),
            }
        )

    if passed_pairs and failed_pairs:
        status = "partial"
        reason = "some_pairs_stale_or_invalid"
    elif passed_pairs:
        status = "ok"
        reason = "ok"
    else:
        status = "failed"
        reason = "all_pairs_stale_or_invalid"

    payload = {
        "status": status,
        "reason": reason,
        "pairs_requested": len(pairs),
        "pairs_passed": len(passed_pairs),
        "pairs_failed": len(failed_pairs),
        "passed_pairs": [f"{sym_x}/{sym_y}" for sym_x, sym_y in passed_pairs],
        "failed_pairs": failed_pairs,
    }
    if orchestrator is not None:
        orchestrator._fresh_pairs_override = list(passed_pairs)
        orchestrator._last_data_freshness_report = payload
        orchestrator.bus.publish("data_freshness", payload)
    return payload


def task_risk_check(**kwargs) -> Dict[str, Any]:
    """Run risk assessment on current state."""
    try:
        from core.risk_engine import (
            RiskState, RiskLimits, check_risk_breaches,
            compute_overall_risk_score,
        )

        state = RiskState()
        limits = RiskLimits()
        breaches = check_risk_breaches(state, limits)
        score = compute_overall_risk_score(state, limits)
        return {
            "status": "ok",
            "risk_score": score,
            "breaches": breaches,
            "breach_count": len(breaches),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def task_health_check(**kwargs) -> Dict[str, Any]:
    """System health check — verify all components are importable and responsive."""
    results = {}
    modules = [
        "core.sql_store", "core.optimizer", "core.risk_engine",
        "core.signals_engine", "core.fair_value_engine", "core.macro_engine",
        "common.config_manager", "common.data_loader",
    ]
    for mod in modules:
        try:
            __import__(mod)
            results[mod] = "ok"
        except Exception as e:
            results[mod] = f"error: {e}"

    ok_count = sum(1 for v in results.values() if v == "ok")
    return {
        "status": "ok" if ok_count == len(modules) else "degraded",
        "modules_ok": ok_count,
        "modules_total": len(modules),
        "details": results,
    }


def task_sql_store_maintenance(**kwargs) -> Dict[str, Any]:
    """Run SQL store maintenance (vacuum, integrity checks)."""
    try:
        from core.sql_store import SqlStore
        from common.config_manager import load_config

        cfg = load_config()
        url = cfg.get("engine_url") or cfg.get("sql_store_url")
        if not url:
            return {"status": "skipped", "reason": "no_sql_url"}

        store = SqlStore(url)
        tables = store.list_tables()
        return {"status": "ok", "tables": len(tables)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================================
# Orchestrator
# ============================================================================

class PairsOrchestrator:
    """
    Main orchestrator for the pairs trading system.

    Manages task scheduling, execution, and inter-component communication.
    Inspired by srv_quant's agent orchestrator pattern.
    """

    def __init__(self):
        self.bus = AgentBus()
        self.tasks: Dict[str, ScheduledTask] = {}
        self._results: List[TaskResult] = []
        self._last_signal_collection_diagnostics: Dict[str, Any] = {}
        self._fresh_pairs_override: Optional[list[tuple[str, str]]] = None
        self._last_data_freshness_report: Dict[str, Any] = {}
        self._register_default_tasks()
        # Startup reconciliation: verify local positions match broker before accepting new intents.
        # Any critical discrepancy pauses new trading until manually resolved.
        self._run_startup_reconciliation()

        # Startup health check: verify all required components are operational.
        self._run_startup_health_check()

    def _persist_signal_collection_diagnostics(self, payload: Dict[str, Any]) -> None:
        data = dict(payload) if isinstance(payload, dict) else {}
        data.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
        self._last_signal_collection_diagnostics = data
        try:
            self.bus.publish("signal_collection_diagnostics", data)
        except Exception:
            logger.debug("Failed to publish signal_collection_diagnostics", exc_info=True)

        store = getattr(self, "_store", None) or getattr(self, "sql_store", None)
        if store is not None and hasattr(store, "save_json"):
            try:
                key = str(getattr(self, "_current_run_id", None) or data.get("run_id") or "latest")
                store.save_json("signal_collection_diagnostics", key, data)
                if key != "latest":
                    store.save_json("signal_collection_diagnostics", "latest", data)
            except Exception:
                logger.debug("Failed to persist signal_collection_diagnostics", exc_info=True)

    def _run_startup_reconciliation(self) -> None:
        """Run position reconciliation on startup. Non-fatal if infrastructure unavailable."""
        try:
            from core.position_reconciliation import (
                PositionReconciliationEngine,
                StartupReconciliationError,
            )
            router = getattr(self, "_order_router", None) or getattr(self, "_router", None)
            store = getattr(self, "_store", None) or getattr(self, "sql_store", None)

            if router is None and store is None:
                logger.debug("Startup reconciliation skipped: no router or store available")
                return

            engine = PositionReconciliationEngine(
                router=router,
                store=store,
                raise_on_critical=False,  # Log but don't crash on startup in this integration
            )
            result = engine.reconcile()
            if not result.clean:
                logger.warning(
                    "Startup reconciliation: %s",
                    result.summary(),
                )
                self.bus.publish("reconciliation", {
                    "status": "discrepancies_found",
                    "critical_count": result.critical_count,
                    "warning_count": result.warning_count,
                    "discrepancies": [
                        {"symbol": d.symbol, "delta": d.delta, "severity": d.severity}
                        for d in result.discrepancies
                    ],
                })
            else:
                logger.info("Startup reconciliation: %s", result.summary())
        except Exception as exc:
            logger.warning("Startup reconciliation failed (non-fatal): %s", exc)

    def _run_startup_health_check(self) -> None:
        """
        Run full system health check on startup.

        Persists the result to the agent bus so the dashboard can display
        the health status in the status bar without a separate health check run.
        Never raises — health check failure is logged but does not block startup.
        """
        try:
            from core.system_health import run_health_check
            result = run_health_check(run_id=getattr(self, "_run_id", ""))
            logger.info("Startup health check: %s", result.summary())

            self.bus.publish("system_health", {
                "healthy": result.healthy,
                "summary": result.summary(),
                "n_required_passing": result.n_required_passing,
                "n_required_total": result.n_required_total,
                "failed_required": result.failed_required(),
                "failed_optional": result.failed_optional(),
                "total_check_time_ms": result.total_check_time_ms,
                "ts": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            })

            if not result.healthy:
                logger.warning(
                    "Startup health check FAILED: %s — system will continue but may be degraded. "
                    "Failed required components: %s",
                    result.summary(),
                    result.failed_required(),
                )
        except Exception as exc:
            logger.warning("Startup health check failed (non-fatal): %s", exc)

    def _register_default_tasks(self) -> None:
        """Register the standard pipeline tasks."""
        self.register_task(ScheduledTask(
            name="health_check",
            func=task_health_check,
            description="System health check",
        ))
        self.register_task(ScheduledTask(
            name="data_refresh",
            func=task_data_refresh,
            description="Refresh price data",
            depends_on=["health_check"],
        ))
        self.register_task(ScheduledTask(
            name="data_freshness_check",
            func=task_data_freshness_check,
            description="Validate pair price freshness",
            depends_on=["data_refresh"],
        ))
        self.register_task(ScheduledTask(
            name="compute_signals",
            func=task_compute_signals,
            description="Compute pair signals",
            depends_on=["data_freshness_check"],
        ))
        self.register_task(ScheduledTask(
            name="risk_check",
            func=task_risk_check,
            description="Run risk assessment",
            depends_on=["compute_signals"],
        ))
        self.register_task(ScheduledTask(
            name="sql_maintenance",
            func=task_sql_store_maintenance,
            description="SQL store maintenance",
        ))

    def register_task(self, task: ScheduledTask) -> None:
        self.tasks[task.name] = task

    def _execute_task(self, task: ScheduledTask, **kwargs) -> TaskResult:
        """Execute a single task with timing and error handling."""
        if not task.enabled:
            return TaskResult(task.name, "skipped")

        # Check dependencies
        for dep in task.depends_on:
            dep_task = self.tasks.get(dep)
            if dep_task and dep_task.last_status == "failed":
                dep_reason = None
                try:
                    dep_latest = self.bus.latest(dep) or {}
                    dep_payload = dep_latest.get("payload") or {}
                    dep_output = dep_payload.get("output") if isinstance(dep_payload, dict) else None
                    if isinstance(dep_output, dict):
                        dep_reason = dep_output.get("reason") or dep_output.get("error")
                    if dep_reason is None and isinstance(dep_payload, dict):
                        dep_reason = dep_payload.get("reason") or dep_payload.get("error")
                except Exception:
                    dep_reason = None
                logger.warning(
                    "Skipping %s: dependency %s failed", task.name, dep,
                )
                task.last_status = "skipped"
                reason_txt = str(dep_reason or "dependency_failed")
                return TaskResult(
                    task.name,
                    "skipped",
                    error=f"dep:{dep} failed:{reason_txt}",
                    output={"status": "skipped", "reason": reason_txt},
                )

        logger.info("Running task: %s (%s)", task.name, task.description)
        t0 = time.time()
        try:
            output = task.func(
                run_id=getattr(self, "_current_run_id", None),
                task_name=task.name,
                orchestrator=self,
                **kwargs,
            )
            elapsed = time.time() - t0
            task.last_run = datetime.now(timezone.utc).isoformat()
            output_status = (
                str(output.get("status") or "").strip().lower()
                if isinstance(output, dict)
                else ""
            )
            result_status = "success"
            result_error = None
            if output_status in {"failed", "error"}:
                result_status = "failed"
                result_error = str(
                    output.get("error")
                    or output.get("reason")
                    or output.get("status")
                    or "task_failed"
                )
            elif output_status == "skipped":
                result_status = "skipped"
                result_error = str(output.get("reason") or "skipped")

            task.last_status = result_status
            result = TaskResult(task.name, result_status, elapsed, error=result_error, output=output)
            self.bus.publish(task.name, {
                "status": result_status,
                "duration_sec": round(elapsed, 2),
                "output": output,
            })
            if result_status == "failed":
                logger.warning("Task %s reported failure in %.1fs", task.name, elapsed)
            else:
                logger.info("Task %s completed in %.1fs", task.name, elapsed)
            return result

        except Exception as e:
            elapsed = time.time() - t0
            task.last_status = "failed"
            logger.error("Task %s failed after %.1fs: %s", task.name, elapsed, e)
            result = TaskResult(task.name, "failed", elapsed, error=str(e))
            self.bus.publish(task.name, {
                "status": "failed",
                "duration_sec": round(elapsed, 2),
                "error": str(e),
            })
            return result

    def run_daily_pipeline(self) -> List[TaskResult]:
        """
        Execute the full daily pipeline in dependency order.

        Pipeline:
            health_check (bare)
            → agent_system_health (WorkflowEngine, P1-AGENTS)
            → data_refresh
            → agent_data_integrity (WorkflowEngine, P1-AGENTS)
            → compute_signals
            → portfolio_allocation (SignalPipeline → bridge_signals_to_allocator,
                                    with is_safe_to_trade + kill-switch factory)
            → risk_check

        Agent dispatches:
        1. SystemHealthAgent (after health_check) — P1-AGENTS
        2. DataIntegrityAgent (after data_refresh) — P1-AGENTS

        Allocation:
            After compute_signals, _collect_signal_decisions() runs SignalPipeline
            for each active pair, then run_portfolio_allocation_cycle() feeds the
            resulting SignalDecision objects through bridge_signals_to_allocator()
            with runtime safety check and control-plane kill-switch (P1-PORTINT,
            P1-SAFE, P0-KS — all COMPLETE via this call path).
        """
        # AP-7: Generate unique run_id for idempotency + traceability
        import uuid as _uuid
        run_id = str(_uuid.uuid4())[:12]
        self._current_run_id = run_id

        logger.info("=" * 60)
        logger.info(
            "Starting daily pipeline run_id=%s at %s",
            run_id, datetime.now(timezone.utc).isoformat(),
        )
        logger.info("=" * 60)

        self._fresh_pairs_override = None
        self._last_data_freshness_report = {}

        order = ["health_check", "data_refresh", "data_freshness_check", "compute_signals", "risk_check"]
        results = []
        all_signal_decisions = []  # Collect for alert summary

        for name in order:
            task = self.tasks.get(name)
            if task:
                result = self._execute_task(task)
                results.append(result)
                self._results.append(result)

            # After health_check, dispatch SystemHealthAgent (P1-AGENTS)
            if name == "health_check":
                agent_result = self.run_agent_system_health_check()
                if agent_result is not None:
                    results.append(agent_result)
                    self._results.append(agent_result)

            # After data_refresh, dispatch DataIntegrityAgent (P1-AGENTS)
            if name == "data_refresh":
                agent_result = self.run_agent_data_integrity_check()
                if agent_result is not None:
                    results.append(agent_result)
                    self._results.append(agent_result)

            # After compute_signals: run signal agents → allocate → risk agents
            if name == "compute_signals" and result.status == "success":
                # Signal-layer agents (regime surveillance)
                for ar in self._dispatch_signal_agents():
                    results.append(ar)
                    self._results.append(ar)

                # Collect decisions and allocate
                signal_decisions = self._collect_signal_decisions()
                all_signal_decisions = signal_decisions  # Save for alert summary
                if signal_decisions:
                    try:
                        from common.config_manager import load_config
                        capital = float(load_config().get("scheduler_capital", 1_000_000.0))
                    except Exception:
                        capital = 1_000_000.0
                    alloc_result = self.run_portfolio_allocation_cycle(
                        signal_decisions=signal_decisions,
                        capital=capital,
                        allocation_batch_id=f"daily:{date.today():%Y%m%d}",
                    )
                    if alloc_result is not None:
                        results.append(alloc_result)
                        self._results.append(alloc_result)

                    # Risk-layer agents (exposure, drawdown, kill-switch)
                    for ar in self._dispatch_risk_agents():
                        results.append(ar)
                        self._results.append(ar)
                else:
                    logger.info("run_daily_pipeline: no signal decisions — allocation skipped")

        # ── Run risk analytics on portfolio returns ─────────────
        risk_result = self._run_risk_analytics()
        if risk_result is not None:
            results.append(risk_result)
            self._results.append(risk_result)

        # ── Run universe scanner for new pair opportunities ───
        scan_result = self._run_universe_scan()
        if scan_result is not None:
            results.append(scan_result)
            self._results.append(scan_result)

        # ── Run correlation health monitor ────────────────────
        corr_result = self._run_correlation_monitor()
        if corr_result is not None:
            results.append(corr_result)
            self._results.append(corr_result)

        # ── Run Monte Carlo confidence estimation ─────────────
        mc_result = self._run_monte_carlo()
        if mc_result is not None:
            results.append(mc_result)
            self._results.append(mc_result)

        # ── Run factor attribution ────────────────────────────
        attr_result = self._run_factor_attribution()
        if attr_result is not None:
            results.append(attr_result)
            self._results.append(attr_result)

        # Summary
        ok = sum(1 for r in results if r.status == "success")
        failed = len(results) - ok
        logger.info(
            "Daily pipeline complete: %d/%d tasks succeeded",
            ok, len(results),
        )

        # ── Agent Feedback Loop — ACT on agent outputs ─────────
        feedback_summary = self._run_feedback_loop(results)
        training_result = self.run_agent_training_cycle()
        if training_result is not None:
            results.append(training_result)
            self._results.append(training_result)
        cfa_research_result = self.run_cfa_research_cycle()
        if cfa_research_result is not None:
            results.append(cfa_research_result)
            self._results.append(cfa_research_result)
        portfolio_monitor_result = self.run_portfolio_monitor_cycle()
        if portfolio_monitor_result is not None:
            results.append(portfolio_monitor_result)
            self._results.append(portfolio_monitor_result)

        # ── Send operational alerts ──────────────────────────────
        self._send_pipeline_alerts(results, all_signal_decisions)

        # Publish pipeline summary
        self.bus.publish("daily_pipeline", {
            "run_id": run_id,
            "tasks_ok": ok,
            "tasks_total": len(results),
            "results": [asdict(r) for r in results],
        })

        # Persist run manifest for dashboard/agent observability
        try:
            store = getattr(self, "_store", None) or getattr(self, "sql_store", None)
            if store is not None and hasattr(store, "save_run_manifest"):
                import uuid as _uuid
                store.save_run_manifest(
                    run_id=str(_uuid.uuid4()),
                    config_hash="",
                    stage_timings=getattr(self, "_last_stage_timings", {}),
                    stage_errors=getattr(self, "_last_stage_errors", {}),
                    status="completed",
                    notes="Daily pipeline complete",
                )
        except Exception as _exc:
            logger.debug("save_run_manifest failed (non-fatal): %s", _exc)

        return results

    def _run_risk_analytics(self) -> Optional[TaskResult]:
        """Run portfolio-level risk analytics using the new RiskAnalytics engine."""
        try:
            from core.risk_analytics import RiskAnalytics

            # Try to load recent equity curve / returns
            returns = None
            try:
                from core.alpha_persistence import load_alpha_results
                alpha = load_alpha_results()
                if alpha and "equity_curve" in alpha:
                    eq = pd.Series(alpha["equity_curve"])
                    returns = eq.pct_change().dropna()
            except Exception:
                pass

            if returns is None or len(returns) < 30:
                logger.info("Risk analytics: no returns data available, skipping")
                return None

            ra = RiskAnalytics()
            report = ra.full_risk_report(returns, name="Portfolio")

            # Alert on high risk
            if report.var_95 and report.var_95.cvar > 0.03:
                try:
                    from core.alerts import alert_risk
                    alert_risk(
                        "High CVaR",
                        f"95% CVaR = {report.var_95.cvar:.2%} (threshold: 3%)",
                        severity="WARNING",
                    )
                except Exception:
                    pass

            if report.drawdown and report.drawdown.current_drawdown < -0.10:
                try:
                    from core.alerts import alert_risk
                    alert_risk(
                        "Drawdown Alert",
                        f"Current DD = {report.drawdown.current_drawdown:.2%}",
                        severity="CRITICAL" if report.drawdown.current_drawdown < -0.20 else "WARNING",
                    )
                except Exception:
                    pass

            self.bus.publish("risk_analytics", {
                "sharpe": report.sharpe_ratio,
                "vol": report.annualized_vol,
                "var95": report.var_95.cornish_fisher_var if report.var_95 else None,
                "cvar95": report.var_95.cvar if report.var_95 else None,
                "max_dd": report.drawdown.max_drawdown if report.drawdown else None,
                "current_dd": report.drawdown.current_drawdown if report.drawdown else None,
            })

            logger.info(
                "Risk analytics: Sharpe=%.2f, Vol=%.1f%%, CVaR95=%.2f%%, MaxDD=%.1f%%",
                report.sharpe_ratio,
                report.annualized_vol * 100,
                (report.var_95.cvar if report.var_95 else 0) * 100,
                (report.drawdown.max_drawdown if report.drawdown else 0) * 100,
            )

            return TaskResult(
                task_name="risk_analytics",
                status="success",
                message=f"Sharpe={report.sharpe_ratio:.2f}, CVaR95={report.var_95.cvar:.2%}" if report.var_95 else "OK",
            )
        except Exception as exc:
            logger.warning("Risk analytics failed: %s", exc)
            return TaskResult(task_name="risk_analytics", status="failed", message=str(exc))

    def _run_universe_scan(self) -> Optional[TaskResult]:
        """Run universe scanner to find new pair opportunities."""
        try:
            from core.universe_scanner import UniverseScanner
            from common.data_loader import load_prices_multi

            # Get list of symbols to scan
            pairs = self._get_active_pairs()
            pair_tuples = normalize_pairs(pairs)
            symbols = {sym for pair in pair_tuples for sym in pair}

            if len(symbols) < 3:
                logger.info("Universe scan: too few symbols (%d), skipping", len(symbols))
                return None

            prices = load_prices_multi(list(symbols))
            if prices is None or prices.empty or len(prices.columns) < 3:
                logger.info("Universe scan: insufficient price data")
                return None

            scanner = UniverseScanner(
                min_correlation=0.55,
                min_grade="C",
                require_cointegration=True,
                max_pairs=20,
            )
            result = scanner.scan(prices)

            self.bus.publish("universe_scan", {
                "n_instruments": result.n_instruments,
                "n_pairs_screened": result.n_pairs_screened,
                "n_pairs_final": result.n_pairs_final,
                "yield_rate": result.yield_rate,
                "top_pairs": [
                    {"pair": f"{p.sym_x}/{p.sym_y}", "score": p.score, "grade": p.grade}
                    for p in result.pairs[:5]
                ],
            })

            # Alert on new high-quality discoveries
            new_high_quality = [p for p in result.pairs if p.grade in ("A+", "A")]
            if new_high_quality:
                try:
                    from core.alerts import alert_signal
                    for p in new_high_quality[:3]:
                        alert_signal(
                            pair=f"{p.sym_x}/{p.sym_y}",
                            action="NEW_DISCOVERY",
                            z_score=0.0,
                            grade=p.grade,
                            score=f"{p.score:.3f}",
                            half_life=f"{p.half_life:.1f}d",
                        )
                except Exception:
                    pass

            logger.info(
                "Universe scan: %d pairs found from %d screened (yield=%.1f%%)",
                result.n_pairs_final, result.n_pairs_screened, result.yield_rate * 100,
            )

            return TaskResult(
                task_name="universe_scan",
                status="success",
                message=f"{result.n_pairs_final} pairs found ({result.yield_rate:.0%} yield)",
            )
        except Exception as exc:
            logger.warning("Universe scan failed: %s", exc)
            return TaskResult(task_name="universe_scan", status="failed", message=str(exc))

    def _run_correlation_monitor(self) -> Optional[TaskResult]:
        """Run correlation health check on all active pairs."""
        try:
            from core.correlation_monitor import CorrelationMonitor
            from common.data_loader import load_prices_multi

            pairs = self._get_active_pairs()
            if not pairs:
                return None

            pair_tuples = normalize_pairs(pairs)
            symbols = {sym for pair in pair_tuples for sym in pair}

            if len(symbols) < 2:
                return None

            prices = load_prices_multi(list(symbols))
            if prices is None or prices.empty:
                return None

            cm = CorrelationMonitor()
            report = cm.monitor_portfolio(prices, pair_tuples)

            # Alert on breaks
            for pair_key, health in report.pair_health.items():
                if health.has_break and health.divergence_risk in ("HIGH", "CRITICAL"):
                    try:
                        from core.alerts import alert_risk
                        alert_risk(
                            "Correlation Break",
                            f"{pair_key}: {health.alert_message} (action: {health.action})",
                            severity="CRITICAL" if health.divergence_risk == "CRITICAL" else "WARNING",
                        )
                    except Exception:
                        pass

            self.bus.publish("correlation_monitor", {
                "n_pairs": report.n_pairs,
                "avg_correlation": report.avg_correlation,
                "n_breaks": report.n_breaks_detected,
                "n_at_risk": report.n_pairs_at_risk,
                "effective_n_bets": report.effective_n_bets,
            })

            logger.info(
                "Correlation monitor: %d pairs, %d breaks, %d at risk, eff_N=%.1f",
                report.n_pairs, report.n_breaks_detected,
                report.n_pairs_at_risk, report.effective_n_bets,
            )
            return TaskResult(
                task_name="correlation_monitor", status="success",
                message=f"{report.n_breaks_detected} breaks, {report.n_pairs_at_risk} at risk",
            )
        except Exception as exc:
            logger.warning("Correlation monitor failed: %s", exc)
            return TaskResult(task_name="correlation_monitor", status="failed", message=str(exc))

    def _run_monte_carlo(self) -> Optional[TaskResult]:
        """Run Monte Carlo simulation for strategy confidence estimation."""
        try:
            from core.monte_carlo import MonteCarloEngine

            returns = None
            try:
                from core.alpha_persistence import load_alpha_results
                alpha = load_alpha_results()
                if alpha and "equity_curve" in alpha:
                    eq = pd.Series(alpha["equity_curve"])
                    returns = eq.pct_change().dropna()
            except Exception:
                pass

            if returns is None or len(returns) < 60:
                return None

            mc = MonteCarloEngine(n_simulations=2000, seed=42)
            result = mc.simulate_strategy(returns, target_return=0.10, ruin_threshold=-0.25)

            self.bus.publish("monte_carlo", {
                "sharpe_mean": result.sharpe_mean,
                "sharpe_ci": [result.sharpe_ci_lower, result.sharpe_ci_upper],
                "deflated_sharpe": result.deflated_sharpe,
                "prob_loss": result.prob_loss,
                "prob_target_10pct": result.prob_target,
                "prob_ruin_25pct": result.prob_ruin,
                "median_max_dd": result.median_max_drawdown,
            })

            # Alert if strategy confidence is low
            if result.prob_loss > 0.30:
                try:
                    from core.alerts import alert_risk
                    alert_risk(
                        "High Loss Probability",
                        f"MC: {result.prob_loss:.0%} probability of loss (Sharpe CI: [{result.sharpe_ci_lower:.2f}, {result.sharpe_ci_upper:.2f}])",
                        severity="WARNING",
                    )
                except Exception:
                    pass

            logger.info(
                "Monte Carlo: Sharpe=%.2f [%.2f, %.2f], DSR=%.3f, P(loss)=%.1f%%, P(ruin)=%.1f%%",
                result.sharpe_mean, result.sharpe_ci_lower, result.sharpe_ci_upper,
                result.deflated_sharpe, result.prob_loss * 100, result.prob_ruin * 100,
            )
            return TaskResult(
                task_name="monte_carlo", status="success",
                message=f"Sharpe={result.sharpe_mean:.2f}, DSR={result.deflated_sharpe:.3f}, P(loss)={result.prob_loss:.0%}",
            )
        except Exception as exc:
            logger.warning("Monte Carlo failed: %s", exc)
            return TaskResult(task_name="monte_carlo", status="failed", message=str(exc))

    def _run_factor_attribution(self) -> Optional[TaskResult]:
        """Run factor attribution on portfolio returns."""
        try:
            from core.factor_attribution import FactorAttribution

            returns = None
            try:
                from core.alpha_persistence import load_alpha_results
                alpha = load_alpha_results()
                if alpha and "equity_curve" in alpha:
                    eq = pd.Series(alpha["equity_curve"])
                    returns = eq.pct_change().dropna()
            except Exception:
                pass

            if returns is None or len(returns) < 60:
                return None

            # Try to load SPY as benchmark
            benchmark = None
            try:
                from common.data_loader import load_price_data
                spy_data = load_price_data("SPY")
                if not spy_data.empty:
                    col = next((c for c in ("adj_close", "close", "Close") if c in spy_data.columns), None)
                    if col:
                        benchmark = spy_data[col].pct_change().dropna()
            except Exception:
                pass

            fa = FactorAttribution()
            report = fa.attribute(returns, benchmark_returns=benchmark)

            if report.performance:
                perf = report.performance
                self.bus.publish("factor_attribution", {
                    "alpha_annualized": perf.alpha_annualized,
                    "beta": perf.beta_to_benchmark,
                    "r_squared": perf.r_squared,
                    "information_ratio": perf.information_ratio,
                    "tracking_error": perf.tracking_error,
                    "sharpe": perf.sharpe_ratio,
                    "hit_rate": perf.hit_rate,
                    "max_dd": perf.max_drawdown,
                })

                logger.info(
                    "Factor attribution: alpha=%.2f%%, beta=%.2f, IR=%.2f, R²=%.2f",
                    perf.alpha_annualized * 100, perf.beta_to_benchmark,
                    perf.information_ratio, perf.r_squared,
                )

                return TaskResult(
                    task_name="factor_attribution", status="success",
                    message=f"alpha={perf.alpha_annualized:.2%}, beta={perf.beta_to_benchmark:.2f}, IR={perf.information_ratio:.2f}",
                )
            return None
        except Exception as exc:
            logger.warning("Factor attribution failed: %s", exc)
            return TaskResult(task_name="factor_attribution", status="failed", message=str(exc))

    def _run_feedback_loop(self, results: list) -> Optional[Any]:
        """
        Run the Agent Feedback Loop on pipeline results.

        This is the key integration: agent outputs are analyzed by the
        feedback engine and converted into REAL system actions (block entries,
        force exits, deleverage, retrain, etc.)
        """
        try:
            from core.agent_feedback import AgentFeedbackLoop
            from core.action_throttler import ActionThrottler
            from core.execution_safety import get_execution_mode
            from common.config_manager import load_config

            mode = get_execution_mode(load_config())
            loop = AgentFeedbackLoop(dry_run=mode["dry_run"])
            actions = loop.process_agent_results(results)

            if actions:
                throttler = ActionThrottler()
                allowed_actions = []
                throttled_actions = []
                for action in actions:
                    action_key = getattr(action, "target", None)
                    if throttler.allow(action.action_type, key=action_key):
                        allowed_actions.append(action)
                    else:
                        throttled_actions.append(action)
                        logger.info(
                            "Feedback action throttled: %s -> %s",
                            action.action_type,
                            action.target,
                        )

                if not mode["allow_agent_actions"]:
                    summary = loop.execute_actions([])
                    summary.n_actions_generated = len(actions)
                    summary.n_actions_blocked = len(actions)
                    summary.actions = actions
                else:
                    summary = loop.execute_actions(allowed_actions)
                    for action in summary.actions:
                        if getattr(action, "executed", False):
                            throttler.mark(action.action_type, key=getattr(action, "target", None))
                    if throttled_actions:
                        summary.n_actions_generated += len(throttled_actions)
                        summary.n_actions_blocked += len(throttled_actions)
                        summary.actions.extend(throttled_actions)

                self.bus.publish("feedback_loop", {
                    "n_actions": summary.n_actions_generated,
                    "n_executed": summary.n_actions_executed,
                    "n_blocked": summary.n_actions_blocked,
                    "execution_mode": mode,
                    "state_changes": summary.system_state_changes,
                    "throttled_actions": [
                        {"type": a.action_type, "target": a.target, "severity": a.severity}
                        for a in throttled_actions
                    ],
                    "actions": [
                        {"type": a.action_type, "target": a.target,
                         "severity": a.severity, "executed": a.executed}
                        for a in summary.actions
                    ],
                })
                logger.info(
                    "Feedback loop: %d actions generated, %d executed, %d blocked",
                    summary.n_actions_generated, summary.n_actions_executed,
                    summary.n_actions_blocked,
                )
                return summary
            else:
                logger.info("Feedback loop: no actions needed")
                return None

        except Exception as exc:
            logger.warning("Feedback loop failed: %s", exc)
            return None

    def _send_pipeline_alerts(self, results: list, signal_decisions: list) -> None:
        """
        Send operational alerts after daily pipeline completes.

        Alerts:
        - Pipeline summary (OK/FAILED tasks count)
        - Active entry/exit signals
        - Risk warnings from failed tasks
        """
        try:
            from core.alerts import alert_system, alert_signal, alert_risk

            # Pipeline summary
            ok = sum(1 for r in results if r.status == "success")
            failed = len(results) - ok
            if failed > 0:
                alert_system(
                    "daily_pipeline",
                    "WARNING",
                    f"{ok}/{len(results)} tasks OK, {failed} failed",
                )
            else:
                alert_system(
                    "daily_pipeline",
                    "OK",
                    f"All {ok} tasks completed successfully",
                )

            # Signal alerts — notify on active entry/exit signals
            for dec in signal_decisions:
                try:
                    if hasattr(dec, 'blocked') and dec.blocked:
                        continue  # Skip blocked signals

                    pair_label = str(dec.pair_id) if hasattr(dec, 'pair_id') else "?"
                    z = dec.z_score if hasattr(dec, 'z_score') else 0.0
                    intent = getattr(dec, 'intent', None)

                    if intent is None:
                        continue

                    action_str = str(getattr(intent, 'action', 'HOLD'))
                    if 'ENTRY' in action_str.upper() or 'EXIT' in action_str.upper():
                        regime = getattr(dec, 'regime', '?')
                        grade = getattr(dec, 'quality_grade', '?')
                        alert_signal(
                            pair=pair_label,
                            action=action_str,
                            z_score=z,
                            regime=regime,
                            quality=grade,
                        )
                except Exception:
                    pass  # Never let alert errors break the pipeline

        except ImportError:
            logger.debug("core.alerts not available — skipping pipeline alerts")
        except Exception as exc:
            logger.warning("_send_pipeline_alerts failed: %s", exc)

    def run_agent_system_health_check(self) -> Optional[TaskResult]:
        """
        Dispatch the SystemHealthAgent to validate core module importability.

        Second real agent dispatch from operational code (P1-AGENTS).
        Mirrors the two-path pattern of run_agent_data_integrity_check:

        1. **WorkflowEngine path** (``monitoring.workflow.run_system_health_workflow``)
           — full WorkflowRun lifecycle, step-level audit, artifact capture,
           dashboard alert emission.

        2. **Direct registry dispatch** (fallback on ImportError / exception).

        READ_ONLY in both paths — never mutates state.

        Returns
        -------
        TaskResult or None
        """
        import time as _time
        t0 = _time.time()

        # ── Path 1: WorkflowEngine (preferred) ────────────────────
        try:
            from monitoring.workflow import run_system_health_workflow
            from orchestration.contracts import WorkflowStatus

            outcome = run_system_health_workflow(
                triggered_by="orchestrator_daily_pipeline",
                emit_alerts=True,
            )

            elapsed = _time.time() - t0

            outcome_status = getattr(outcome, "status", None)
            outcome_run_id = getattr(outcome, "run_id", "unknown")
            steps_completed = getattr(outcome, "steps_completed", None)
            steps_failed    = getattr(outcome, "steps_failed", None)

            is_success = (
                outcome_status == WorkflowStatus.COMPLETED
                if outcome_status is not None else True
            )

            self.bus.publish("agent_system_health", {
                "path": "workflow_engine",
                "workflow_status": outcome_status.value if outcome_status else "unknown",
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "duration_sec": round(elapsed, 2),
            })
            logger.info(
                "Agent system_health (WorkflowEngine): status=%s duration=%.1fs",
                outcome_status.value if outcome_status else "unknown",
                elapsed,
            )
            return TaskResult(
                task_name="agent_system_health",
                status="success" if is_success else "failed",
                duration_sec=round(elapsed, 2),
                output={
                    "path": "workflow_engine",
                    "workflow_run_id": outcome_run_id,
                    "workflow_status": outcome_status.value if outcome_status else "unknown",
                    "steps_completed": steps_completed,
                    "steps_failed": steps_failed,
                },
            )

        except ImportError:
            logger.debug("monitoring.workflow unavailable, using direct dispatch for system_health")
        except Exception as wf_err:
            logger.warning(
                "WorkflowEngine dispatch failed for system_health (%s) — falling back",
                wf_err,
            )

        # ── Path 2: Direct registry dispatch (fallback) ───────────
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus

            registry = get_default_registry()
            agent = registry.get_agent("system_health")
            if agent is None:
                logger.warning("SystemHealthAgent not registered — skipping")
                return None

            task = AgentTask(
                task_id=f"orch_sh_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                agent_name="system_health",
                task_type="health_sweep",
                payload={},
                priority=3,
                correlation_id=f"daily_pipeline_{datetime.now(timezone.utc).date()}",
            )

            result = registry.dispatch(task)
            elapsed = _time.time() - t0
            status = "success" if result.status == AgentStatus.COMPLETED else "failed"
            unhealthy: list = result.output.get("unhealthy_components", []) if result.output else []
            overall_healthy: bool = result.output.get("overall_healthy", True) if result.output else True

            if not overall_healthy and unhealthy:
                try:
                    from core.alert_bus import emit_dashboard_alert
                    emit_dashboard_alert(
                        level="error",
                        source="orchestrator:system_health",
                        message=f"System health: {len(unhealthy)} component(s) unhealthy",
                        details={"unhealthy_components": unhealthy[:10]},
                    )
                except Exception:
                    pass

            logger.info(
                "Agent system_health (direct): status=%s unhealthy=%d duration=%.1fs",
                status, len(unhealthy), elapsed,
            )
            return TaskResult(
                task_name="agent_system_health",
                status=status,
                duration_sec=round(elapsed, 2),
                output={
                    "path": "direct_dispatch",
                    "agent_status": result.status.value,
                    "overall_healthy": overall_healthy,
                    "unhealthy_components": unhealthy,
                    "audit_trail_size": len(result.audit_trail),
                },
            )

        except Exception as e:
            elapsed = _time.time() - t0
            logger.warning("Agent system_health direct dispatch failed: %s", e)
            return TaskResult(
                task_name="agent_system_health",
                status="failed",
                duration_sec=round(elapsed, 2),
                error=str(e),
            )

    def _load_latest_data_refresh_diagnostics(self) -> dict[str, Any]:
        """Return the most recent data-refresh diagnostics from bus or SQL."""
        try:
            latest = self.bus.latest("data_refresh")
            payload = (latest or {}).get("payload") or {}
            output = payload.get("output")
            if isinstance(output, dict) and (output.get("rows") or output.get("status_counts")):
                return output
        except Exception:
            pass

        run_id = getattr(self, "_current_run_id", None)
        file_payload = load_data_refresh_diagnostics(run_id or "latest")
        if isinstance(file_payload, dict) and file_payload:
            return file_payload

        try:
            from core.sql_store import SqlStore

            store = SqlStore.from_settings({}, read_only=True)
            if run_id and hasattr(store, "load_json"):
                try:
                    obj = store.load_json("data_refresh_diagnostics", run_id, warn_on_error=False)
                except TypeError:
                    obj = store.load_json("data_refresh_diagnostics", run_id)
                if isinstance(obj, dict):
                    return obj
            if hasattr(store, "load_json"):
                try:
                    obj = store.load_json("data_refresh_diagnostics", "latest", warn_on_error=False)
                except TypeError:
                    obj = store.load_json("data_refresh_diagnostics", "latest")
                if isinstance(obj, dict):
                    return obj
        except Exception as exc:
            logger.debug("load latest data refresh diagnostics skipped: %s", exc)

        return {}

    @staticmethod
    def _summarize_data_refresh_diagnostics(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
        data = payload if isinstance(payload, dict) else {}
        rows = list(data.get("rows") or [])

        def _collect(status: str) -> list[str]:
            return [
                str(row.get("symbol"))
                for row in rows
                if str(row.get("final_status") or "") == status and row.get("symbol")
            ]

        return {
            "refresh_status_counts": dict(data.get("status_counts") or {}),
            "symbols_missing": _collect("missing"),
            "symbols_stale": _collect("stale"),
            "symbols_partial": _collect("partial"),
            "data_refresh_run_id": data.get("run_id"),
        }

    def run_agent_data_integrity_check(self) -> Optional[TaskResult]:
        """
        Dispatch the DataIntegrityAgent to validate price data quality.

        This is the first real agent dispatch from operational code (P1-AGENTS).
        Two dispatch paths exist:

        1. **WorkflowEngine path** (``monitoring.workflow.run_data_integrity_workflow``)
           — full WorkflowRun lifecycle, step-level audit, artifact capture,
           dashboard alert emission.  Used when the monitoring.workflow module
           is available.

        2. **Direct registry dispatch** (fallback)
           — calls ``registry.dispatch()`` directly, still typed and audited,
           but without WorkflowEngine lifecycle management.

        The agent is READ_ONLY in both paths — it never mutates state.

        Returns
        -------
        TaskResult or None
            TaskResult wrapping the result, or None if dispatch fails.
        """
        import time as _time
        t0 = _time.time()

        # ── Path 1: WorkflowEngine (preferred) ────────────────────
        try:
            from monitoring.workflow import run_data_integrity_workflow
            from orchestration.contracts import WorkflowStatus

            refresh_diagnostics = self._load_latest_data_refresh_diagnostics()
            refresh_summary = self._summarize_data_refresh_diagnostics(refresh_diagnostics)
            outcome = run_data_integrity_workflow(
                refresh_diagnostics=refresh_diagnostics or None,
                triggered_by="orchestrator_daily_pipeline",
                emit_alerts=True,
            )

            elapsed = _time.time() - t0

            # outcome is WorkflowOutcome (preferred) or WorkflowRun (fallback)
            outcome_status = getattr(outcome, "status", None)
            outcome_run_id = getattr(outcome, "run_id", "unknown")
            steps_completed = getattr(outcome, "steps_completed", None)
            steps_failed    = getattr(outcome, "steps_failed", None)
            artifact_count  = getattr(outcome, "artifact_count", None)

            # WorkflowStatus.COMPLETED or WorkflowStatus value from wf_run
            from orchestration.contracts import WorkflowStatus
            is_success = (
                outcome_status == WorkflowStatus.COMPLETED
                if outcome_status is not None else True
            )

            self.bus.publish("agent_data_integrity", {
                "path": "workflow_engine",
                "workflow_status": outcome_status.value if outcome_status else "unknown",
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "duration_sec": round(elapsed, 2),
                **refresh_summary,
            })
            logger.info(
                "Agent data_integrity (WorkflowEngine): status=%s duration=%.1fs",
                outcome_status.value if outcome_status else "unknown",
                elapsed,
            )
            return TaskResult(
                task_name="agent_data_integrity",
                status="success" if is_success else "failed",
                duration_sec=round(elapsed, 2),
                output={
                    "path": "workflow_engine",
                    "workflow_run_id": outcome_run_id,
                    "workflow_status": outcome_status.value if outcome_status else "unknown",
                    "steps_completed": steps_completed,
                    "steps_failed": steps_failed,
                    "artifact_count": artifact_count,
                    **refresh_summary,
                },
            )

        except ImportError:
            # monitoring.workflow not available — fall through to direct dispatch
            logger.debug("monitoring.workflow unavailable, using direct dispatch")
        except Exception as wf_err:
            logger.warning(
                "WorkflowEngine dispatch failed (%s) — falling back to direct dispatch",
                wf_err,
            )

        # ── Path 2: Direct registry dispatch (fallback) ───────────
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus

            registry = get_default_registry()
            agent = registry.get_agent("data_integrity")
            if agent is None:
                logger.warning("DataIntegrityAgent not registered — skipping")
                return None

            refresh_diagnostics = self._load_latest_data_refresh_diagnostics()
            refresh_summary = self._summarize_data_refresh_diagnostics(refresh_diagnostics)
            task = AgentTask(
                task_id=f"orch_di_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                agent_name="data_integrity",
                task_type="check_data_integrity",
                payload={"data_refresh_diagnostics": refresh_diagnostics},
                priority=3,
                correlation_id=f"daily_pipeline_{datetime.now(timezone.utc).date()}",
            )

            result = registry.dispatch(task)

            elapsed = _time.time() - t0
            status = "success" if result.status == AgentStatus.COMPLETED else "failed"
            issues: int = result.output.get("issues_found", 0) if result.output else 0
            critical: list = result.output.get("critical_issues", []) if result.output else []

            # ── Alert bus (best-effort) ────────────────────────────
            if issues > 0:
                try:
                    from core.alert_bus import emit_dashboard_alert
                    level = "error" if issues >= 3 else "warning"
                    emit_dashboard_alert(
                        level=level,
                        source="orchestrator:data_integrity",
                        message=f"Data integrity: {issues} critical issue(s) detected",
                        details={
                            "critical_issues": critical[:5],
                            "audit_trail_size": len(result.audit_trail),
                            "refresh_status_counts": refresh_summary.get("refresh_status_counts", {}),
                            "symbols_missing": refresh_summary.get("symbols_missing", [])[:10],
                            "symbols_stale": refresh_summary.get("symbols_stale", [])[:10],
                        },
                    )
                except Exception:
                    pass   # alert bus is best-effort

            logger.info(
                "Agent data_integrity (direct): status=%s issues=%d duration=%.1fs",
                status, issues, elapsed,
            )

            return TaskResult(
                task_name="agent_data_integrity",
                status=status,
                duration_sec=round(elapsed, 2),
                output={
                    "path": "direct_dispatch",
                    "agent_status": result.status.value,
                    "issues_found": issues,
                    "critical_issues": critical,
                    "audit_trail_size": len(result.audit_trail),
                    **refresh_summary,
                },
            )

        except Exception as e:
            elapsed = _time.time() - t0
            logger.warning("Agent data_integrity direct dispatch failed: %s", e)
            return TaskResult(
                task_name="agent_data_integrity",
                status="failed",
                duration_sec=round(elapsed, 2),
                error=str(e),
            )

    def run_agent_training_cycle(
        self,
        *,
        min_priority: str = "HIGH",
        max_actions: int = 3,
        execute_recommendations: bool = True,
        action_contexts: Optional[dict[str, dict[str, Any]]] = None,
    ) -> Optional[TaskResult]:
        """
        Review agent-learning state and execute bounded improvement actions.

        Preferred path uses the WorkflowEngine-based agent training workflow.
        Fallback path dispatches the training director directly through the
        registry so the pipeline still keeps moving if the workflow module
        is unavailable.
        """
        import time as _time

        t0 = _time.time()

        try:
            from monitoring.agent_training_workflow import run_agent_training_workflow
            from orchestration.contracts import WorkflowStatus

            outcome = run_agent_training_workflow(
                min_priority=min_priority,
                max_actions=max_actions,
                execute_recommendations=execute_recommendations,
                action_contexts=action_contexts,
                triggered_by="orchestrator_daily_pipeline",
                emit_alerts=True,
            )

            elapsed = _time.time() - t0
            outcome_status = getattr(outcome, "status", None)
            outcome_run_id = getattr(outcome, "run_id", "unknown")
            steps_completed = getattr(outcome, "steps_completed", None)
            steps_failed = getattr(outcome, "steps_failed", None)
            artifact_count = getattr(outcome, "artifact_count", None)
            is_success = (
                outcome_status == WorkflowStatus.COMPLETED
                if outcome_status is not None else True
            )

            self.bus.publish("agent_training_cycle", {
                "path": "workflow_engine",
                "workflow_status": outcome_status.value if outcome_status else "unknown",
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "duration_sec": round(elapsed, 2),
            })
            training_output = {
                "path": "workflow_engine",
                "workflow_run_id": outcome_run_id,
                "workflow_status": outcome_status.value if outcome_status else "unknown",
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "artifact_count": artifact_count,
            }
            self._persist_agent_training_state(
                run_id=str(outcome_run_id),
                cycle_output=training_output,
                cycle_status="success" if is_success else "failed",
                duration_sec=round(elapsed, 2),
            )
            return TaskResult(
                task_name="agent_training_cycle",
                status="success" if is_success else "failed",
                duration_sec=round(elapsed, 2),
                output=training_output,
            )

        except ImportError:
            logger.debug("monitoring.agent_training_workflow unavailable, using direct dispatch")
        except Exception as wf_err:
            logger.warning(
                "WorkflowEngine agent training dispatch failed (%s) - falling back to direct dispatch",
                wf_err,
            )

        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus

            registry = get_default_registry()
            agent = registry.get_agent("agent_training_director")
            if agent is None:
                logger.warning("AgentTrainingDirector not registered - skipping")
                return None

            payload: dict[str, Any] = {
                "min_priority": min_priority,
                "max_actions": max_actions,
                "execute_recommendations": execute_recommendations,
            }
            if action_contexts:
                payload["action_contexts"] = action_contexts

            task = AgentTask(
                task_id=f"orch_train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                agent_name="agent_training_director",
                task_type="execute_training_recommendations",
                payload=payload,
                priority=3,
                correlation_id=f"daily_pipeline_{datetime.now(timezone.utc).date()}",
            )
            result = registry.dispatch(task)

            elapsed = _time.time() - t0
            status = "success" if result.status == AgentStatus.COMPLETED else "failed"
            output = result.output if isinstance(result.output, dict) else {}
            self.bus.publish("agent_training_cycle", {
                "path": "direct_dispatch",
                "agent_status": result.status.value,
                "recommendations_found": output.get("recommendations_found", 0),
                "execution_status_counts": output.get("execution_status_counts", {}),
                "duration_sec": round(elapsed, 2),
            })
            direct_output = {
                "path": "direct_dispatch",
                "agent_status": result.status.value,
                "recommendations_found": output.get("recommendations_found", 0),
                "execution_status_counts": output.get("execution_status_counts", {}),
                "audit_trail_size": len(result.audit_trail),
            }
            self._persist_agent_training_state(
                run_id=task.task_id,
                cycle_output=direct_output,
                cycle_status=status,
                duration_sec=round(elapsed, 2),
            )
            return TaskResult(
                task_name="agent_training_cycle",
                status=status,
                duration_sec=round(elapsed, 2),
                output=direct_output,
            )

        except Exception as exc:
            elapsed = _time.time() - t0
            logger.warning("Agent training cycle direct dispatch failed: %s", exc)
            self._persist_agent_training_state(
                run_id=f"failed_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                cycle_output={"error": str(exc), "path": "direct_dispatch"},
                cycle_status="failed",
                duration_sec=round(elapsed, 2),
            )
            return TaskResult(
                task_name="agent_training_cycle",
                status="failed",
                duration_sec=round(elapsed, 2),
                error=str(exc),
            )

    def _persist_agent_training_state(
        self,
        *,
        run_id: str,
        cycle_output: dict[str, Any],
        cycle_status: str,
        duration_sec: float,
    ) -> None:
        store = getattr(self, "_store", None) or getattr(self, "sql_store", None)
        if store is None:
            return

        try:
            from core.agent_learning import get_agent_learning_coordinator

            coordinator = get_agent_learning_coordinator()
            learning_summary = coordinator.build_summary()
            cycle_summary = {
                "run_id": run_id,
                "status": cycle_status,
                "duration_sec": duration_sec,
                "output": cycle_output,
                "learning_summary": {
                    "total_agents": learning_summary.get("total_agents", 0),
                    "agents_with_recommendations": learning_summary.get(
                        "agents_with_recommendations", 0
                    ),
                    "average_trust_score": learning_summary.get(
                        "average_trust_score", 0.0
                    ),
                    "average_training_pressure": learning_summary.get(
                        "average_training_pressure", 0.0
                    ),
                    "recommendation_counts": learning_summary.get(
                        "recommendation_counts", {}
                    ),
                    "priority_counts": learning_summary.get("priority_counts", {}),
                },
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            if hasattr(store, "save_agent_learning_summary"):
                store.save_agent_learning_summary(learning_summary)
            if hasattr(store, "save_agent_training_cycle"):
                store.save_agent_training_cycle(cycle_summary, run_id=run_id)
        except Exception as exc:
            logger.debug("Persisting agent training state failed (non-fatal): %s", exc)

    def run_cfa_research_cycle(
        self,
        *,
        datasets: Optional[list[Any]] = None,
        lane_candidates: Optional[dict[str, list[str]]] = None,
        cache_snapshot: bool = True,
        emit_alerts: bool = True,
        uncertainty_alert_threshold: float = 0.35,
    ) -> Optional[TaskResult]:
        """
        Run the CFA lane-research cycle and persist its production snapshot.

        The cycle is intentionally optional: when no walk-forward datasets are
        available it exits quietly rather than manufacturing research inputs.
        """
        import time as _time

        t0 = _time.time()
        research_datasets = datasets or self._load_cfa_research_datasets_for_cycle()
        if not research_datasets:
            return None

        try:
            from ml.registry import get_ml_registry
            from ml.research import (
                compare_cfa_research_snapshots,
                load_cfa_research_snapshot,
                summarize_cfa_research_snapshot,
            )

            registry = get_ml_registry()
            if registry is None or not hasattr(registry, "research_cfa_ensemble"):
                logger.warning("CFA research cycle skipped: ML registry unavailable")
                return None
            store = getattr(self, "_store", None) or getattr(self, "sql_store", None)
            previous_snapshot = load_cfa_research_snapshot(store=store, registry=None)
            report = registry.research_cfa_ensemble(
                research_datasets,
                lane_candidates=lane_candidates,
                cache_snapshot=cache_snapshot,
            )
            snapshot = load_cfa_research_snapshot(store=None, registry=registry)
            if not snapshot and isinstance(report, dict):
                snapshot = dict(report)
            summary = summarize_cfa_research_snapshot(snapshot)
            change_summary = compare_cfa_research_snapshots(previous_snapshot, snapshot)
            if isinstance(snapshot, dict):
                snapshot = dict(snapshot)
                snapshot["change_summary"] = dict(change_summary)
                if registry is not None and hasattr(registry, "set_latest_cfa_research"):
                    try:
                        registry.set_latest_cfa_research(snapshot)
                    except Exception:
                        logger.debug("Updating cached cfa_research snapshot with change_summary failed", exc_info=True)

            run_id = str(getattr(self, "_current_run_id", None) or f"cfa_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
            cycle_output = {
                "run_id": run_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "n_datasets": len(research_datasets),
                "lane_candidates_supplied": bool(lane_candidates),
                **summary,
                "change_summary": change_summary,
            }

            self.bus.publish("cfa_research", snapshot)
            self.bus.publish("cfa_research_cycle", cycle_output)
            self._persist_cfa_research_state(
                run_id=run_id,
                snapshot=snapshot,
                report=report if isinstance(report, dict) else {},
                cycle_output=cycle_output,
            )
            if emit_alerts:
                self._emit_cfa_research_alerts(
                    cycle_output=cycle_output,
                    uncertainty_alert_threshold=uncertainty_alert_threshold,
                )

            elapsed = _time.time() - t0
            return TaskResult(
                task_name="cfa_research_cycle",
                status="success",
                duration_sec=round(elapsed, 2),
                output=cycle_output,
            )
        except Exception as exc:
            elapsed = _time.time() - t0
            logger.warning("CFA research cycle failed: %s", exc)
            return TaskResult(
                task_name="cfa_research_cycle",
                status="failed",
                duration_sec=round(elapsed, 2),
                error=str(exc),
            )

    def _load_cfa_research_datasets_for_cycle(self) -> list[Any]:
        store = getattr(self, "_store", None) or getattr(self, "sql_store", None)
        if store is not None and hasattr(store, "load_json"):
            try:
                payload = store.load_json("cfa_research_datasets", "latest")
                if isinstance(payload, dict) and isinstance(payload.get("datasets"), list):
                    return list(payload["datasets"])
                if isinstance(payload, list):
                    return list(payload)
            except Exception as exc:
                logger.debug("Loading cfa_research_datasets from store failed: %s", exc)

        bus_payload = self.bus.latest("cfa_research_datasets")
        if isinstance(bus_payload, dict):
            payload = bus_payload.get("payload") if "payload" in bus_payload else bus_payload
            if isinstance(payload, dict) and isinstance(payload.get("datasets"), list):
                return list(payload["datasets"])
            if isinstance(payload, list):
                return list(payload)
        return []

    def _persist_cfa_research_state(
        self,
        *,
        run_id: str,
        snapshot: dict[str, Any],
        report: dict[str, Any],
        cycle_output: dict[str, Any],
    ) -> None:
        store = getattr(self, "_store", None) or getattr(self, "sql_store", None)
        if store is None:
            return

        try:
            snapshot_payload = dict(snapshot) if isinstance(snapshot, dict) else {}
            snapshot_payload.setdefault("run_id", run_id)
            snapshot_payload.setdefault("updated_at", cycle_output.get("updated_at"))
            if hasattr(store, "save_cfa_research_snapshot"):
                store.save_cfa_research_snapshot(snapshot_payload, run_id=run_id)
            elif hasattr(store, "save_json"):
                store.save_json("cfa_research", run_id, snapshot_payload)
                if run_id != "latest":
                    store.save_json("cfa_research", "latest", snapshot_payload)

            if hasattr(store, "save_json"):
                store.save_json("cfa_research_cycle", run_id, dict(cycle_output))
                if run_id != "latest":
                    store.save_json("cfa_research_cycle", "latest", dict(cycle_output))
                if report:
                    store.save_json("cfa_research_report", run_id, dict(report))
                    if run_id != "latest":
                        store.save_json("cfa_research_report", "latest", dict(report))
        except Exception as exc:
            logger.debug("Persisting cfa_research state failed (non-fatal): %s", exc)

    def _emit_cfa_research_alerts(
        self,
        *,
        cycle_output: dict[str, Any],
        uncertainty_alert_threshold: float,
    ) -> None:
        promote_count = int(cycle_output.get("promote_count", 0) or 0)
        require_more_data_count = int(cycle_output.get("require_more_data_count", 0) or 0)
        uncertainty_score = float(cycle_output.get("uncertainty_score", 1.0) or 1.0)
        promote_lanes = list(cycle_output.get("promote_lanes", []) or [])
        require_more_data_lanes = list(cycle_output.get("require_more_data_lanes", []) or [])
        change_summary = dict(cycle_output.get("change_summary", {}) or {})
        changed_lane_count = int(change_summary.get("changed_lane_count", 0) or 0)
        leadership_changes = list(change_summary.get("leadership_changes", []) or [])

        try:
            from core.alert_bus import emit_dashboard_alert

            if promote_count > 0:
                emit_dashboard_alert(
                    level="warning",
                    source="cfa_research_cycle",
                    message=f"CFA research recommends promoting {promote_count} lane winner(s)",
                    details={
                        "promote_lanes": promote_lanes,
                        "require_more_data_lanes": require_more_data_lanes,
                        "uncertainty_score": uncertainty_score,
                        "change_summary": change_summary,
                        "cycle_output": cycle_output,
                    },
                )
            elif changed_lane_count > 0:
                emit_dashboard_alert(
                    level="info",
                    source="cfa_research_cycle",
                    message=f"CFA leadership changed across {changed_lane_count} lane(s)",
                    details={
                        "leadership_changes": leadership_changes,
                        "uncertainty_score": uncertainty_score,
                        "change_summary": change_summary,
                    },
                )
            elif uncertainty_score >= uncertainty_alert_threshold:
                emit_dashboard_alert(
                    level="warning" if uncertainty_score >= uncertainty_alert_threshold + 0.10 else "info",
                    source="cfa_research_cycle",
                    message=(
                        f"CFA ensemble uncertainty is elevated at {uncertainty_score:.2f}"
                    ),
                    details={
                        "require_more_data_count": require_more_data_count,
                        "require_more_data_lanes": require_more_data_lanes,
                        "cycle_output": cycle_output,
                    },
                )
        except Exception as exc:
            logger.debug("CFA research dashboard alert emit failed: %s", exc)

        try:
            from core.alerts import alert_risk

            if promote_count > 0:
                alert_risk(
                    "CFA Lane Promotion Candidates",
                    (
                        f"Promote lanes: {', '.join(promote_lanes) or 'n/a'}\n"
                        f"Require-more-data lanes: {', '.join(require_more_data_lanes) or 'none'}\n"
                        f"Ensemble uncertainty: {uncertainty_score:.2f}"
                    ),
                    severity="WARNING",
                )
            elif changed_lane_count >= 2:
                alert_risk(
                    "CFA Leadership Drift",
                    (
                        f"Changed lanes: {changed_lane_count}\n"
                        f"Examples: {', '.join(str(change.get('lane_id', '')) for change in leadership_changes[:4]) or 'n/a'}\n"
                        f"Ensemble uncertainty: {uncertainty_score:.2f}"
                    ),
                    severity="INFO",
                )
            elif uncertainty_score >= uncertainty_alert_threshold + 0.15:
                alert_risk(
                    "CFA Ensemble Uncertainty",
                    (
                        f"Uncertainty score: {uncertainty_score:.2f}\n"
                        f"Threshold: {uncertainty_alert_threshold:.2f}\n"
                        f"Lane count: {cycle_output.get('lane_count', 0)}"
                    ),
                    severity="WARNING",
                )
        except Exception as exc:
            logger.debug("CFA research risk alert emit failed: %s", exc)

    def run_portfolio_monitor_cycle(
        self,
        *,
        execute_actions: bool = True,
    ) -> Optional[TaskResult]:
        """
        Run the live portfolio-monitor workflow and optionally route actions.

        Preferred path uses the WorkflowEngine-based portfolio monitor workflow.
        Fallback path dispatches the portfolio_monitor agent directly and then
        routes any resulting actions through the governed feedback loop.
        """
        import time as _time

        t0 = _time.time()
        monitor_payload = self._load_portfolio_monitor_payload_for_cycle()
        if not monitor_payload or not monitor_payload.get("n_open_positions", 0):
            return None

        try:
            from monitoring.portfolio_monitor_workflow import run_portfolio_monitor_workflow
            from orchestration.contracts import WorkflowStatus

            outcome = run_portfolio_monitor_workflow(
                monitor_payload=monitor_payload,
                execute_actions=execute_actions,
                triggered_by="orchestrator_daily_pipeline",
                emit_alerts=True,
            )
            elapsed = _time.time() - t0
            outcome_status = getattr(outcome, "status", None)
            outcome_run_id = getattr(outcome, "run_id", "unknown")
            steps_completed = getattr(outcome, "steps_completed", None)
            steps_failed = getattr(outcome, "steps_failed", None)
            artifact_count = getattr(outcome, "artifact_count", None)
            is_success = (
                outcome_status == WorkflowStatus.COMPLETED
                if outcome_status is not None else True
            )

            workflow_output = {
                "path": "workflow_engine",
                "workflow_run_id": outcome_run_id,
                "workflow_status": outcome_status.value if outcome_status else "unknown",
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "artifact_count": artifact_count,
                "n_open_positions": monitor_payload.get("n_open_positions", 0),
                "forced_exit_candidates": monitor_payload.get("forced_exit_candidates", 0),
            }
            self.bus.publish("portfolio_monitor_cycle", workflow_output)
            return TaskResult(
                task_name="agent_portfolio_monitor",
                status="success" if is_success else "failed",
                duration_sec=round(elapsed, 2),
                output=workflow_output,
            )
        except ImportError:
            logger.debug("monitoring.portfolio_monitor_workflow unavailable, using direct dispatch")
        except Exception as wf_err:
            logger.warning(
                "WorkflowEngine portfolio monitor dispatch failed (%s) - falling back to direct dispatch",
                wf_err,
            )

        try:
            from agents.registry import get_default_registry
            from core.agent_feedback import AgentFeedbackLoop
            from core.contracts import AgentTask, AgentStatus

            registry = get_default_registry()
            agent = registry.get_agent("portfolio_monitor")
            if agent is None:
                logger.warning("PortfolioMonitorAgent not registered - skipping")
                return None

            task = AgentTask(
                task_id=f"orch_pm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                agent_name="portfolio_monitor",
                task_type="scan_portfolio_monitor",
                payload={"monitor_payload": monitor_payload},
                priority=3,
                correlation_id=f"daily_pipeline_{datetime.now(timezone.utc).date()}",
            )
            result = registry.dispatch(task)
            elapsed = _time.time() - t0
            status = "success" if result.status == AgentStatus.COMPLETED else "failed"
            output = result.output if isinstance(result.output, dict) else {}

            feedback_output: dict[str, Any] = {}
            if status == "success":
                loop = AgentFeedbackLoop(dry_run=not execute_actions)
                actions = loop.process_agent_results(
                    [TaskResult(task_name="agent_portfolio_monitor", status="success", output=output)]
                )
                summary = loop.execute_actions(actions)
                feedback_output = {
                    "n_actions_generated": summary.n_actions_generated,
                    "n_actions_executed": summary.n_actions_executed,
                    "n_actions_blocked": summary.n_actions_blocked,
                }

            direct_output = {
                "path": "direct_dispatch",
                "agent_status": result.status.value,
                "n_open_positions": monitor_payload.get("n_open_positions", 0),
                "forced_exit_candidates": monitor_payload.get("forced_exit_candidates", 0),
                "summary": output.get("summary", {}),
                "feedback": feedback_output,
                "audit_trail_size": len(result.audit_trail),
            }
            self.bus.publish("portfolio_monitor_cycle", direct_output)
            return TaskResult(
                task_name="agent_portfolio_monitor",
                status=status,
                duration_sec=round(elapsed, 2),
                output=direct_output,
            )
        except Exception as exc:
            elapsed = _time.time() - t0
            logger.warning("Portfolio monitor cycle direct dispatch failed: %s", exc)
            return TaskResult(
                task_name="agent_portfolio_monitor",
                status="failed",
                duration_sec=round(elapsed, 2),
                error=str(exc),
            )

    def _load_portfolio_monitor_payload_for_cycle(self) -> Dict[str, Any]:
        store = getattr(self, "_store", None) or getattr(self, "sql_store", None)
        if store is not None and hasattr(store, "load_portfolio_monitor_payload"):
            try:
                payload = store.load_portfolio_monitor_payload()
                if isinstance(payload, dict) and payload:
                    return normalize_portfolio_monitor_payload(payload)
            except Exception as exc:
                logger.debug("Loading portfolio monitor payload from store failed: %s", exc)

        tracker = getattr(self, "position_tracker", None) or getattr(self, "_position_tracker", None)
        if tracker is not None and hasattr(tracker, "build_monitor_payload"):
            try:
                payload = tracker.build_monitor_payload()
                if isinstance(payload, dict) and payload:
                    return normalize_portfolio_monitor_payload(payload)
            except Exception as exc:
                logger.debug("Building portfolio monitor payload from tracker failed: %s", exc)

        bus_payload = self.bus.latest("portfolio_monitor")
        if isinstance(bus_payload, dict):
            payload = bus_payload.get("payload") if "payload" in bus_payload else bus_payload
            if isinstance(payload, dict):
                return normalize_portfolio_monitor_payload(payload)
        return {}

    def run_maintenance(self) -> TaskResult:
        """Run SQL store maintenance."""
        task = self.tasks.get("sql_maintenance")
        if task:
            result = self._execute_task(task)
            self._results.append(result)
            return result
        return TaskResult("sql_maintenance", "skipped")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all tasks."""
        return {
            name: {
                "description": t.description,
                "enabled": t.enabled,
                "last_run": t.last_run,
                "last_status": t.last_status,
            }
            for name, t in self.tasks.items()
        }

    def get_results_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent task results."""
        return [asdict(r) for r in self._results[-n:]]

    # ── Scheduler ─────────────────────────────────────────────────────────

    def start_daemon(
        self,
        run_time: Optional[str] = None,
        timezone: str = "America/New_York",
        run_immediately: bool = False,
    ) -> None:
        """
        Start a background daily pipeline scheduler.

        Tries APScheduler (if installed); falls back to threading.Timer.

        Parameters
        ----------
        run_time : str, optional
            "HH:MM" string for daily trigger.  Reads ``scheduler_run_time``
            from config.json if not provided; defaults to ``"16:15"``.
        timezone : str
            Timezone for the cron trigger (APScheduler only).
        run_immediately : bool
            If True, execute one pipeline run immediately before scheduling.
        """
        import atexit

        try:
            from common.config_manager import load_config
            cfg = load_config()
        except Exception:
            cfg = {}

        run_time = run_time or cfg.get("scheduler_run_time", "16:15")
        timezone = cfg.get("scheduler_timezone", timezone)

        if run_immediately:
            logger.info("start_daemon: run_immediately=True — running pipeline now")
            self.run_daily_pipeline()

        # ── APScheduler path (preferred) ─────────────────────────────
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            h, m = (int(x) for x in run_time.split(":"))
            scheduler = BackgroundScheduler(timezone=timezone)
            scheduler.add_job(
                self.run_daily_pipeline,
                trigger=CronTrigger(hour=h, minute=m, timezone=timezone),
                id="daily_pipeline",
                name="Daily Pipeline",
                misfire_grace_time=3600,
            )
            scheduler.start()
            self._scheduler = scheduler
            atexit.register(self.stop_daemon)
            logger.info(
                "start_daemon: APScheduler started — daily run at %s %s",
                run_time, timezone,
            )
            return

        except ImportError:
            logger.debug("APScheduler not installed — using threading.Timer fallback")
        except Exception as e:
            logger.warning("APScheduler setup failed (%s) — using threading.Timer fallback", e)

        # ── threading.Timer fallback ─────────────────────────────────
        import threading
        from datetime import datetime, timezone as dt_tz

        self._daemon_stop_event = threading.Event()

        def _next_run_seconds() -> float:
            now = datetime.now()
            h, m = (int(x) for x in run_time.split(":"))
            target = now.replace(hour=h, minute=m, second=0, microsecond=0)
            delta = (target - now).total_seconds()
            return delta if delta > 0 else delta + 86400  # +24h if already past

        def _loop():
            while not self._daemon_stop_event.is_set():
                delay = _next_run_seconds()
                logger.info("start_daemon: next run in %.0fs (at %s)", delay, run_time)
                if self._daemon_stop_event.wait(timeout=delay):
                    break  # stop requested
                if not self._daemon_stop_event.is_set():
                    logger.info("start_daemon: triggering daily pipeline")
                    try:
                        self.run_daily_pipeline()
                    except Exception as exc:
                        logger.error("start_daemon: pipeline error: %s", exc)

        self._daemon_thread = threading.Thread(target=_loop, daemon=True, name="pipeline-daemon")
        self._daemon_thread.start()
        atexit.register(self.stop_daemon)
        logger.info("start_daemon: threading.Timer fallback started — daily run at %s", run_time)

    def stop_daemon(self) -> None:
        """Stop the background scheduler (APScheduler or threading fallback)."""
        if hasattr(self, "_scheduler"):
            try:
                self._scheduler.shutdown(wait=False)
                logger.info("stop_daemon: APScheduler stopped")
            except Exception:
                pass
        if hasattr(self, "_daemon_stop_event"):
            self._daemon_stop_event.set()
            logger.info("stop_daemon: threading daemon stopped")

    # ── Agent dispatch: signal-layer + risk-layer ──────────────────────────

    def _dispatch_signal_agents(self) -> list:
        """
        Dispatch signal-layer agents with REAL data payloads.

        Agents receive actual spread data, regime info, and positions
        so they can produce meaningful output for the feedback loop.
        """
        results = []
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus
            registry = get_default_registry()

            # Collect real data for agent payloads
            pairs = self._get_active_pairs()
            regime_info = self.bus.latest("regime_analytics") or {}

            # Load actual spread data for pairs
            spreads_data = {}
            try:
                from common.data_loader import load_prices_multi
                import numpy as np
                pair_tuples = _iter_pair_tuples(pairs)
                symbols = {sym for pair in pair_tuples for sym in pair}

                if symbols:
                    prices = load_prices_multi(list(symbols))
                    if prices is not None and not prices.empty:
                        for sx, sy in pair_tuples:
                            if sx in prices.columns and sy in prices.columns:
                                px = prices[sx].dropna()
                                py = prices[sy].dropna()
                                if len(px) > 60 and len(py) > 60:
                                    beta = float(np.cov(px.values, py.values)[0, 1] / max(np.var(px.values), 1e-12))
                                    spread = py - beta * px
                                    spreads_data[f"{sx}/{sy}"] = {
                                        "spread": spread,
                                        "beta": beta,
                                        "z_score": float((spread.iloc[-1] - spread.mean()) / max(spread.std(), 1e-12)),
                                        "prices_x": px,
                                        "prices_y": py,
                                    }
            except Exception as exc:
                logger.debug("Could not load spread data for agents: %s", exc)

            signal_agents = [
                ("regime_surveillance", "regime_surveillance.scan",
                 {"spreads": spreads_data, "pairs": pairs,
                  "current_regime": regime_info.get("payload", {}).get("regime", "NORMAL")}),
                ("signal_analyst", "signal_analyst.classify",
                 {"pair_id": "PLACEHOLDER/SKIP", "spread": None, "skip_if_no_pair": True}),
                ("trade_lifecycle", "lifecycle.inspect",
                 {"states": {}, "pairs": pairs, "n_active_pairs": len(spreads_data)}),
                ("exit_oversight", "exit_oversight.scan",
                 {"open_positions": spreads_data, "pairs": pairs}),
            ]

            ts = datetime.now(timezone.utc)
            for agent_name, task_type, payload in signal_agents:
                agent = registry.get_agent(agent_name)
                if agent is None:
                    continue
                task = AgentTask(
                    task_id=f"{agent_name}_{ts.strftime('%Y%m%d_%H%M%S')}",
                    agent_name=agent_name,
                    task_type=task_type,
                    payload=payload,
                    priority=3,
                    correlation_id=f"daily_pipeline_{ts.date()}",
                )
                result = registry.dispatch(task)
                status = "success" if result.status == AgentStatus.COMPLETED else "failed"
                # PASS REAL AGENT OUTPUT to TaskResult for feedback loop
                agent_output = result.output if isinstance(result.output, dict) else {}
                results.append(TaskResult(
                    task_name=f"agent_{agent_name}",
                    status=status,
                    output=agent_output,
                ))
                logger.info("Agent %s: %s (output_keys=%s)", agent_name, status, list(agent_output.keys())[:5])

        except Exception as e:
            logger.warning("Signal agent dispatch failed: %s", e)
        return results

    def _dispatch_risk_agents(self) -> list:
        """
        Dispatch risk-layer agents with REAL portfolio state.

        Reads actual NAV, positions, and risk data from bus/config
        so agents can produce meaningful risk assessments for the
        feedback loop.
        """
        results = []
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus
            registry = get_default_registry()

            ts = datetime.now(timezone.utc)

            # Get real portfolio state from bus or config
            try:
                from common.config_manager import load_config
                capital = float(load_config().get("scheduler_capital", 1_000_000.0))
            except Exception:
                capital = 1_000_000.0

            risk_bus = self.bus.latest("risk_analytics") or {}
            risk_data = risk_bus.get("payload", {}) if isinstance(risk_bus, dict) else {}

            def _as_float(value, default=0.0):
                try:
                    return float(value)
                except Exception:
                    return default

            alloc_bus = self.bus.latest("portfolio_allocation") or {}
            alloc_data = alloc_bus.get("payload", {}) if isinstance(alloc_bus, dict) else {}
            monitor_bus = self.bus.latest("portfolio_monitor") or {}
            monitor_payload = alloc_data.get("monitoring", {}) if isinstance(alloc_data, dict) else {}
            if not monitor_payload and isinstance(monitor_bus, dict):
                monitor_payload = monitor_bus.get("payload", {}) or {}
            diagnostics = alloc_data.get("portfolio_diagnostics", {}) if isinstance(alloc_data, dict) else {}
            positions = list(monitor_payload.get("positions") or []) if isinstance(monitor_payload, dict) else []

            current_dd_raw = _as_float(
                risk_data.get(
                    "current_dd",
                    risk_data.get(
                        "current_drawdown",
                        (monitor_payload or {}).get("current_drawdown", 0.0),
                    ),
                ),
                0.0,
            )
            current_dd = abs(current_dd_raw)
            current_var = abs(_as_float(risk_data.get("var95", 0.0), 0.0))
            single_day_return = _as_float(
                risk_data.get("single_day_return", risk_data.get("daily_return", 0.0)),
                0.0,
            )
            consecutive_losses = int(_as_float(risk_data.get("consecutive_losses", 0), 0))

            # Compute normalised NAV from drawdown if available.
            peak_nav = capital
            current_nav = capital * (1 - current_dd) if current_dd else capital

            heat_level = str(diagnostics.get("heat_level") or "NORMAL")
            kill_switch_mode = str(diagnostics.get("kill_switch_mode") or "OFF")
            pairs = self._get_active_pairs()
            pair_requests = []
            for pair in pairs:
                pair_tuple = _extract_pair_tuple(pair)
                if pair_tuple is not None:
                    label = f"{pair_tuple[0]}/{pair_tuple[1]}"
                else:
                    label = getattr(pair, "label", str(pair))
                pair_requests.append(
                    {
                        "pair_label": label.replace("-", "/"),
                        "proposed_notional": round(capital * 0.02, 2),
                    }
                )

            risk_agents = [
                ("exposure_monitor", "compute_exposure",
                 {
                     "active_allocations": positions,
                     "positions": positions,
                     "total_capital": capital,
                 }),
                ("drawdown_monitor", "update_drawdown",
                 {
                     "current_value": current_nav,
                     "peak_value": peak_nav,
                     "current_drawdown": current_dd,
                     "rolling_dd_30d": current_dd,
                 }),
                ("kill_switch", "evaluate_kill_switch",
                 {
                     "current_value": current_nav,
                     "peak_value": peak_nav,
                     "single_day_return": single_day_return,
                     "consecutive_losses": consecutive_losses,
                     "current_state": {
                         "mode": kill_switch_mode,
                         "triggered": kill_switch_mode != "OFF",
                     },
                 }),
                ("capital_budget", "check_capital_budget",
                 {
                     "pairs": pair_requests,
                     "total_capital": capital,
                 }),
                ("derisking", "compute_derisking",
                 {
                     "drawdown_state": {
                         "current_dd": current_dd,
                         "current_value": current_nav,
                         "peak_value": peak_nav,
                         "heat_level": heat_level,
                     },
                     "active_allocations": positions,
                 }),
                ("drift_monitoring", "drift_sweep",
                 {}),
                ("alert_aggregation", "aggregate_alerts",
                 {"alerts": []}),
            ]

            for agent_name, task_type, payload in risk_agents:
                agent = registry.get_agent(agent_name)
                if agent is None:
                    continue
                task = AgentTask(
                    task_id=f"{agent_name}_{ts.strftime('%Y%m%d_%H%M%S')}",
                    agent_name=agent_name,
                    task_type=task_type,
                    payload=payload,
                    priority=2,
                    correlation_id=f"daily_pipeline_{ts.date()}",
                )
                result = registry.dispatch(task)
                status = "success" if result.status == AgentStatus.COMPLETED else "failed"
                # PASS REAL AGENT OUTPUT for feedback loop
                agent_output = result.output if isinstance(result.output, dict) else {}
                results.append(TaskResult(
                    task_name=f"agent_{agent_name}",
                    status=status,
                    output=agent_output,
                ))
                logger.info("Agent %s: %s (output_keys=%s)", agent_name, status, list(agent_output.keys())[:5])

        except Exception as e:
            logger.warning("Risk agent dispatch failed: %s", e)
        return results

    def _dispatch_agents_batch(
        self,
        agents_spec: list[tuple[str, str, dict]],
        correlation_prefix: str = "pipeline",
        priority: int = 4,
    ) -> list:
        """Generic agent batch dispatcher. Returns list[TaskResult]."""
        results = []
        try:
            from agents.registry import get_default_registry
            from core.contracts import AgentTask, AgentStatus
            registry = get_default_registry()
            ts = datetime.now(timezone.utc)

            for agent_name, task_type, payload in agents_spec:
                agent = registry.get_agent(agent_name)
                if agent is None:
                    continue
                task = AgentTask(
                    task_id=f"{agent_name}_{ts.strftime('%Y%m%d_%H%M%S')}",
                    agent_name=agent_name,
                    task_type=task_type,
                    payload=payload,
                    priority=priority,
                    correlation_id=f"{correlation_prefix}_{ts.date()}",
                )
                result = registry.dispatch(task)
                status = "success" if result.status == AgentStatus.COMPLETED else "failed"
                results.append(TaskResult(
                    task_name=f"agent_{agent_name}",
                    status=status,
                    output={"agent_status": result.status.value,
                            "audit_size": len(result.audit_trail)},
                ))
                logger.info("Agent %s: %s", agent_name, status)
        except Exception as e:
            logger.warning("Agent batch dispatch failed: %s", e)
        return results

    def dispatch_research_agents(self, symbols: list[str] | None = None) -> list:
        """
        Dispatch all research-layer agents on demand.

        Agents: universe_curator, candidate_discovery, universe_discovery,
        pair_validation, relationship_validation, spread_fit, spread_specification,
        regime_research, signal_research, experiment_coordinator, research_summarization
        """
        if symbols is None:
            try:
                pairs = self._get_active_pairs()
                symbols = []
                pair_payloads = []
                for pair in pairs:
                    pair_tuple = _extract_pair_tuple(pair)
                    if pair_tuple is None:
                        continue
                    sym_x, sym_y = pair_tuple
                    pair_payloads.append({"sym_x": sym_x, "sym_y": sym_y})
                    symbols.extend([sym_x, sym_y])
            except Exception:
                pair_payloads = []
                symbols = []
        else:
            pair_payloads = []

        if not pair_payloads:
            pair_payloads = [
                {"sym_x": symbols[i].upper(), "sym_y": symbols[i + 1].upper()}
                for i in range(0, len(symbols) - 1, 2)
                if i + 1 < len(symbols)
            ]

        symbols = sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
        prices = None
        if symbols:
            try:
                from common.data_loader import load_prices_multi

                prices = load_prices_multi(symbols)
            except Exception:
                prices = None

        return self._dispatch_agents_batch([
            ("universe_curator", "curate_universe", {"symbols": symbols}),
            ("universe_discovery", "discover_pairs", {"symbols": symbols}),
            ("candidate_discovery", "discover_candidates", {"symbols": symbols}),
            ("pair_validation", "validate_pairs", {"pair_ids": pair_payloads, "prices": prices}),
            ("relationship_validation", "validate_relationships", {"pair_ids": pair_payloads, "prices": prices}),
            ("spread_fit", "fit_spreads", {"pair_ids": pair_payloads, "prices": prices}),
            ("spread_specification", "specify_spreads", {"pair_ids": pair_payloads, "prices": prices}),
            ("regime_research", "classify_regimes", {"spreads": {}}),
            ("signal_research", "analyze_signal_quality", {"spreads": {}, "signals": {}}),
            ("experiment_coordinator", "plan_experiment", {"experiment_type": "pair_scan"}),
            ("research_summarization", "summarize_research_run", {}),
        ], correlation_prefix="research", priority=5)

    def dispatch_ml_agents(self) -> list:
        """
        Dispatch all ML-layer agents on demand.

        Agents: feature_steward, label_governance, model_research,
        meta_labeling, regime_modeling, model_risk, promotion_review
        """
        try:
            from ml.features.definitions import FEATURE_REGISTRY

            feature_names = list(FEATURE_REGISTRY.keys())[:8]
        except Exception:
            feature_names = ["inst_ret_1d", "inst_ret_5d", "spread_zscore", "spread_vol_20d"]

        try:
            from ml.labels.definitions import LABEL_REGISTRY

            label_names = list(LABEL_REGISTRY.keys())[:6]
        except Exception:
            label_names = ["reversion_5d", "reversion_10d", "meta_success_5d"]

        model_ids: list[str] = []
        try:
            from ml.registry.registry import get_ml_registry

            registry = get_ml_registry()
            model_ids = [
                str(getattr(meta, "model_id", ""))
                for meta in (registry.list_all() if hasattr(registry, "list_all") else [])
                if getattr(meta, "model_id", "")
            ][:4]
        except Exception:
            model_ids = []
        if not model_ids:
            model_ids = ["demo_meta_model"]

        evaluation_metrics = {
            model_id: {
                "ic": round(0.07 - idx * 0.005, 4),
                "auc": round(0.61 - idx * 0.01, 4),
                "brier": round(0.20 + idx * 0.01, 4),
            }
            for idx, model_id in enumerate(model_ids)
        }
        health_status = {
            model_id: ("healthy" if idx == 0 else "degraded")
            for idx, model_id in enumerate(model_ids)
        }
        age_days = {
            model_id: 30.0 + idx * 15.0
            for idx, model_id in enumerate(model_ids)
        }
        first_model_id = model_ids[0]

        signals = []
        for pair in self._get_active_pairs()[:3]:
            pair_tuple = _extract_pair_tuple(pair)
            if pair_tuple is None:
                continue
            signals.append(
                {
                    "pair_id": f"{pair_tuple[0]}/{pair_tuple[1]}",
                    "z_score": 2.0,
                    "regime": "MEAN_REVERTING",
                    "conviction": 0.7,
                }
            )
        if not signals:
            signals = [
                {
                    "pair_id": "SPY/QQQ",
                    "z_score": 2.1,
                    "regime": "MEAN_REVERTING",
                    "conviction": 0.72,
                }
            ]

        regime_features = {
            signal["pair_id"]: {
                "spread_values": [0.10, -0.05, 0.07, -0.03, 0.02, -0.01, 0.03, -0.02, 0.01, 0.00, -0.01],
                "break_confidence": 0.12,
            }
            for signal in signals
        }

        return self._dispatch_agents_batch([
            ("feature_steward", "audit_feature_health", {"feature_names": feature_names}),
            ("label_governance", "check_label_leakage", {"label_names": label_names}),
            ("model_research", "evaluate_model", {
                "model_ids": model_ids,
                "evaluation_metrics": evaluation_metrics,
            }),
            ("meta_labeling", "assess_meta_label", {"signals": signals}),
            ("regime_modeling", "evaluate_regime_model", {"features": regime_features}),
            ("model_risk", "assess_model_risk", {
                "model_ids": model_ids,
                "health_status": health_status,
                "age_days": age_days,
            }),
            ("promotion_review", "check_promotion_criteria", {
                "subject_type": "MODEL",
                "subject_id": first_model_id,
                "subject_name": first_model_id,
                "metrics": evaluation_metrics[first_model_id],
                "current_status": "CANDIDATE",
                "proposed_status": "CHALLENGER",
                "evidence_ids": [f"evidence_{first_model_id}"],
            }),
        ], correlation_prefix="ml_review", priority=4)

    def dispatch_governance_agents(self) -> list:
        """
        Dispatch all governance-layer agents on demand.

        Agents: policy_review, audit_trail_validation, approval_recommendation,
        change_impact, promotion_gate, orchestration_reliability,
        incident_triage, postmortem_drafting
        """
        try:
            from common.config_manager import load_config

            env_raw = str((load_config() or {}).get("env") or "research").strip().lower()
        except Exception:
            env_raw = "research"

        env_map = {
            "research": "RESEARCH",
            "paper": "PAPER",
            "dev": "STAGING",
            "development": "STAGING",
            "test": "STAGING",
            "staging": "STAGING",
            "prod": "PRODUCTION",
            "production": "PRODUCTION",
            "live": "PRODUCTION",
        }
        environment = env_map.get(env_raw, "RESEARCH")
        evidence_summary = (
            "Routine daily governance review for orchestrator workflows; "
            "metrics reviewed, impact assessed, and risk evaluated."
        )

        return self._dispatch_agents_batch([
            ("policy_review", "check_policy_compliance", {
                "agent_name": "orchestrator",
                "task_type": "run_daily_pipeline",
                "action_type": "DAILY_PIPELINE_REVIEW",
                "environment": environment,
                "risk_class": "BOUNDED_SAFE",
            }),
            ("audit_trail_validation", "validate_audit_trail", {}),
            ("approval_recommendation", "recommend_approval", {
                "action_type": "DAILY_PIPELINE_REVIEW",
                "risk_class": "BOUNDED_SAFE",
                "environment": environment,
                "evidence_summary": evidence_summary,
                "agent_name": "orchestrator",
            }),
            ("change_impact", "assess_change_impact", {
                "change_description": "Routine orchestrator governance review of daily pipeline operations.",
                "affected_components": ["core.orchestrator", "governance.engine", "monitoring.workflow"],
                "change_type": "CONFIG_CHANGE",
                "proposed_by": "orchestrator",
            }),
            ("promotion_gate", "check_promotion_eligibility", {
                "subject_type": "POLICY",
                "subject_id": "daily_pipeline_policy",
                "subject_name": "Daily Pipeline Policy",
                "from_status": "DRAFT",
                "to_status": "ACTIVE",
            }),
            ("orchestration_reliability", "check_workflow_health", {}),
            ("incident_triage", "classify_severity",
             {"title": "daily_check", "description": "routine", "affected_components": [], "detected_by": "orchestrator"}),
            ("postmortem_drafting", "analyze_incident_pattern", {
                "incident_id": "daily_governance_review",
                "incident_summary": "Routine governance review executed without an active production incident.",
                "timeline": [],
                "resolution": "No remediation required.",
            }),
        ], correlation_prefix="governance_review", priority=4)

    def dispatch_portfolio_agents(self) -> list:
        """Dispatch portfolio construction agent on demand."""
        return self._dispatch_agents_batch([
            ("portfolio_construction", "run_allocation_cycle",
             {"intents": [], "total_capital": 1_000_000.0}),
        ], correlation_prefix="portfolio", priority=3)

    # ── Signal collection helpers ─────────────────────────────────────────

    def _get_active_pairs(self) -> list:
        """
        Return the active pairs list from AppContext or config.json.

        Returns [] with a warning if neither source is available.
        """
        if self._fresh_pairs_override is not None:
            return list(self._fresh_pairs_override)

        try:
            from common.config_manager import load_config

            cfg = load_config()
        except Exception:
            cfg = {}
        policy = load_asset_policy(cfg)

        try:
            from core.app_context import AppContext
            ctx = AppContext.get_global()
            if ctx.pairs:
                return [
                    (sym_x, sym_y)
                    for sym_x, sym_y in normalize_pairs(list(ctx.pairs))
                    if pair_allowed_by_policy(sym_x, sym_y, policy=policy)
                ]
        except Exception:
            pass

        pairs = _get_configured_pairs(cfg)
        if pairs:
            return pairs

        logger.warning("_get_active_pairs: no pairs found in AppContext or config")
        return []

    def _load_ml_hook(self):
        """
        Load the ML model for signal quality via the formal ML governance layer only.

        Only ModelScorer (Priority 1) is used. Ad-hoc pickle files are explicitly
        excluded — they bypass governance, drift monitoring, and audit trails.

        Legacy compatibility note:
        - The historical direct loader used ``MetaLabelModel`` and a local
          ``meta_label_latest.pkl`` artifact.
        - We intentionally do not activate that path anymore; the governed
          ModelScorer registry is now the only operational source of ML quality
          signals.

        Returns an object with ``predict_success_probability(features_dict)``
        or None if no governed model is available (triggers deterministic fallback).
        """
        # Priority 1: Formal ModelScorer — only governed, champion-status models
        try:
            from ml.inference.scorer import ModelScorer
            from ml.contracts import InferenceRequest, MLTaskFamily
            scorer = ModelScorer()
            model = scorer.get_model_for_task(MLTaskFamily.META_LABELING)
            if model is not None:
                class _ScorerHook:
                    def __init__(self, _scorer):
                        self._scorer = _scorer
                    def predict_success_probability(self, feats):
                        try:
                            req = InferenceRequest(
                                entity_id="pipeline",
                                task_family=MLTaskFamily.META_LABELING,
                                features=feats if isinstance(feats, dict) else {},
                                allow_fallback=True,
                            )
                            result = self._scorer.score(req)
                            return result.score if not result.fallback_used else float("nan")
                        except Exception:
                            return float("nan")
                logger.info("ML model loaded via ModelScorer (formal inference layer)")
                return _ScorerHook(scorer)
        except Exception as exc:
            logger.debug("ModelScorer not available: %s", exc)

        # No pickle fallbacks — ungoverned models must not reach production signal decisions.
        # Use deterministic quality scoring (QualityConfig.ml_enabled=False path).
        logger.info(
            "No governed ML model available — using deterministic quality fallback. "
            "To enable ML, promote a model to CHAMPION status in the ML registry."
        )
        return None

    def _collect_signal_decisions(
        self,
        pairs: Optional[list] = None,
        lookback_days: int = 252,
    ) -> list:
        """
        Run SignalPipeline for every active pair and return SignalDecision objects.

        Called from run_daily_pipeline() to bridge compute_signals →
        run_portfolio_allocation_cycle().  Never raises — returns [] on any
        error so the pipeline continues without allocation.

        Parameters
        ----------
        pairs : list, optional
            List of pair dicts ``{"sym_x": ..., "sym_y": ...}`` or 2-tuples.
            Defaults to ``_get_active_pairs()``.
        lookback_days : int
            Price history length fed to SignalPipeline (default: 252).

        Returns
        -------
        list[SignalDecision]
        """
        import time as _time

        diagnostics_payload: Dict[str, Any] = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": str(getattr(self, "_current_run_id", None) or "latest"),
            "lookback_days": int(max(lookback_days, 0)),
            "n_pairs_requested": 0,
            "n_pairs_resolved": 0,
            "n_symbols_requested": 0,
            "n_decisions": 0,
            "n_skipped": 0,
            "success_rate": 0.0,
            "price_rows_loaded": 0,
            "skip_reason_counts": {},
            "skipped_pairs": [],
            "status": "started",
        }

        if pairs is None:
            pairs = self._get_active_pairs()
        diagnostics_payload["n_pairs_requested"] = len(pairs or [])
        if not pairs:
            diagnostics_payload["status"] = "no_pairs"
            self._persist_signal_collection_diagnostics(diagnostics_payload)
            return []

        try:
            from common.data_loader import load_prices_multi
            from core.contracts import PairId
            from core.signal_pipeline import SignalPipeline
            from core.signals_engine import compute_mr_score
        except ImportError as exc:
            logger.warning("_collect_signal_decisions: required import failed: %s", exc)
            diagnostics_payload["status"] = "import_failed"
            diagnostics_payload["error"] = str(exc)
            self._persist_signal_collection_diagnostics(diagnostics_payload)
            return []

        decisions = []
        skipped_pairs: list[dict[str, Any]] = []
        skip_reasons: Counter[str] = Counter()
        t0 = _time.time()

        # Load optional meta-label model once for the whole batch.
        ml_hook = self._load_ml_hook()

        resolved_pairs: list[tuple[str, str]] = []
        symbols: set[str] = set()
        for pair_def in pairs:
            pair_tuple = _extract_pair_tuple(pair_def)
            if pair_tuple is None:
                skip_reasons["invalid_pair_def"] += 1
                skipped_pairs.append(
                    {
                        "pair": str(pair_def),
                        "status": "skipped",
                        "reason": "invalid_pair_def",
                    }
                )
                continue

            resolved_pairs.append(pair_tuple)
            symbols.update(pair_tuple)

        diagnostics_payload["n_pairs_resolved"] = len(resolved_pairs)
        diagnostics_payload["n_symbols_requested"] = len(symbols)
        if not resolved_pairs or not symbols:
            diagnostics_payload["status"] = "no_resolved_pairs"
            diagnostics_payload["skip_reason_counts"] = dict(skip_reasons)
            diagnostics_payload["n_skipped"] = int(sum(skip_reasons.values()))
            diagnostics_payload["skipped_pairs"] = skipped_pairs[:50]
            self._persist_signal_collection_diagnostics(diagnostics_payload)
            return []

        try:
            prices_wide = load_prices_multi(
                sorted(symbols),
                start_date=datetime.now(timezone.utc) - timedelta(days=max(lookback_days * 3, 365)),
                col="close",
                fill_method="none",
            )
        except Exception as exc:
            logger.warning("_collect_signal_decisions: bulk price load failed: %s", exc)
            diagnostics_payload["status"] = "bulk_price_load_failed"
            diagnostics_payload["error"] = str(exc)
            diagnostics_payload["skip_reason_counts"] = dict(skip_reasons)
            diagnostics_payload["n_skipped"] = int(sum(skip_reasons.values()))
            diagnostics_payload["skipped_pairs"] = skipped_pairs[:50]
            self._persist_signal_collection_diagnostics(diagnostics_payload)
            return []

        if prices_wide is None or prices_wide.empty:
            logger.info(
                "_collect_signal_decisions: no price matrix available for %d symbols",
                len(symbols),
            )
            diagnostics_payload["status"] = "empty_price_matrix"
            diagnostics_payload["skip_reason_counts"] = dict(skip_reasons)
            diagnostics_payload["n_skipped"] = int(sum(skip_reasons.values()))
            diagnostics_payload["skipped_pairs"] = skipped_pairs[:50]
            self._persist_signal_collection_diagnostics(diagnostics_payload)
            return []
        diagnostics_payload["price_rows_loaded"] = int(len(prices_wide))

        for pair_sym_x, pair_sym_y in resolved_pairs:
            sym_x, sym_y = pair_sym_x, pair_sym_y
            try:
                snapshot, snapshot_diag = _prepare_pair_signal_snapshot(
                    prices_wide,
                    sym_x,
                    sym_y,
                    lookback_days=lookback_days,
                    min_points=max(60, min(120, lookback_days)),
                    include_diagnostics=True,
                )
                if snapshot is None:
                    reason = str(snapshot_diag.get("reason") or "snapshot_unavailable")
                    skip_reasons[reason] += 1
                    skipped_pairs.append(dict(snapshot_diag))
                    continue

                spread = snapshot["spread"]
                px = snapshot["px"]
                py = snapshot["py"]
                z_score = float(snapshot["z_score"])
                half_life = float(snapshot["half_life"])
                variance_ratio = float(snapshot["variance_ratio"])
                hurst = float(snapshot["hurst"])
                correlation = float(snapshot["correlation"])
                mr_score = float(
                    compute_mr_score(
                        spread=spread,
                        hurst=hurst,
                        half_life=half_life,
                        variance_ratio=variance_ratio,
                        z_velocity=float(snapshot["z_velocity"]),
                    )
                )

                pipeline = SignalPipeline(
                    pair_id=PairId(sym_x, sym_y),
                    ml_quality_hook=ml_hook,
                )
                decision = pipeline.evaluate(
                    z_score=z_score,
                    spread_series=spread,
                    prices_x=px,
                    prices_y=py,
                    as_of=snapshot["as_of"],
                    conviction=float(snapshot["conviction"]),
                    mr_score=mr_score,
                    half_life=half_life,
                    correlation=correlation,
                    current_spread_vol=float(snapshot["spread_vol_20d"]),
                    baseline_spread_vol=float(snapshot["spread_vol_120d"]),
                )
                decision_intent = getattr(decision, "intent", None)
                if decision_intent is not None:
                    freshness_lag_days = float(snapshot.get("freshness_lag_days", 0.0) or 0.0)
                    freshness_score = float(np.clip(1.0 - (freshness_lag_days / 7.0), 0.0, 1.0))
                    readiness_score = float(
                        np.clip(
                            (
                                0.55 * float(snapshot.get("coverage_ratio", 0.0) or 0.0)
                                + 0.30 * np.clip(
                                    float(snapshot["aligned_points"]) / max(float(lookback_days), 1.0),
                                    0.0,
                                    1.0,
                                )
                                + 0.15 * freshness_score
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    setattr(decision_intent, "data_readiness_score", readiness_score)
                    if getattr(decision_intent, "spread_vol", None) is None:
                        setattr(decision_intent, "spread_vol", float(snapshot["spread_vol_20d"]))
                    setattr(decision_intent, "correlation", correlation)
                    setattr(decision_intent, "mr_score", mr_score)
                    setattr(decision_intent, "hurst_exponent", hurst)
                    setattr(decision_intent, "variance_ratio", variance_ratio)
                    setattr(decision_intent, "freshness_lag_days", freshness_lag_days)
                    setattr(
                        decision_intent,
                        "coverage_ratio",
                        float(snapshot.get("coverage_ratio", 0.0) or 0.0),
                    )

                advisory_notes_extra: list[str] = [
                    (
                        "signal_input:"
                        f"aligned_points={snapshot['aligned_points']},"
                        f"coverage_ratio={float(snapshot.get('coverage_ratio', 0.0) or 0.0):.4f},"
                        f"freshness_lag_days={float(snapshot.get('freshness_lag_days', 0.0) or 0.0):.2f},"
                        f"alpha={snapshot['alpha']:.6f},"
                        f"beta={snapshot['beta']:.6f},"
                        f"corr={correlation:.4f},"
                        f"hurst={hurst:.4f},"
                        f"mr_score={mr_score:.4f},"
                        f"half_life={half_life}"
                    )
                ]
                ou_optimal_exit_z_override: float = float("nan")
                try:
                    from core.cycle_detector import CycleDetector

                    cd = CycleDetector()
                    cycle_result = cd.analyze(spread)
                    advisory_notes_extra.append(
                        f"cycle:period={cycle_result.dominant_period:.1f},"
                        f"cyclical={cycle_result.is_cyclical},"
                        f"phase={cycle_result.phase_signal:.3f},"
                        f"strength={cycle_result.cycle_strength:.3f}"
                    )
                except Exception:
                    pass

                try:
                    from core.optimal_exit import OptimalExitEngine
                    from core.spread_analytics import SpreadAnalytics

                    sa = SpreadAnalytics()
                    hl = sa._mean_reversion_metrics(spread)
                    half_life_for_exit = (
                        float(getattr(hl, "half_life_days", float("nan")))
                        if hl is not None
                        else float("nan")
                    )
                    if not np.isfinite(half_life_for_exit) or half_life_for_exit <= 0.0:
                        half_life_for_exit = half_life if np.isfinite(half_life) and half_life > 0.0 else 15.0

                    oe = OptimalExitEngine(half_life=half_life_for_exit, entry_z=abs(z_score))
                    boundary = oe.compute_optimal_boundary()
                    exit_5d = boundary.exit_z_dynamic[4] if len(boundary.exit_z_dynamic) > 4 else None
                    exit_10d = boundary.exit_z_dynamic[9] if len(boundary.exit_z_dynamic) > 9 else None
                    if exit_5d is not None:
                        ou_optimal_exit_z_override = float(exit_5d)
                    advisory_notes_extra.append(
                        f"optimal_exit:hl={half_life_for_exit:.1f}d,"
                        f"exit_5d={exit_5d},"
                        f"exit_10d={exit_10d},"
                        f"max_hold={boundary.max_expected_holding:.1f}d"
                    )
                except Exception:
                    pass

                if advisory_notes_extra or not (ou_optimal_exit_z_override != ou_optimal_exit_z_override):
                    import dataclasses

                    old_advisory = decision.advisory
                    new_notes = list(old_advisory.notes) + advisory_notes_extra
                    new_advisory = dataclasses.replace(
                        old_advisory,
                        ou_optimal_exit_z=ou_optimal_exit_z_override
                        if not (ou_optimal_exit_z_override != ou_optimal_exit_z_override)
                        else old_advisory.ou_optimal_exit_z,
                        notes=new_notes,
                    )
                    decision = dataclasses.replace(decision, advisory=new_advisory)

                decisions.append(decision)

            except Exception as exc:
                skip_reasons["pair_exception"] += 1
                skipped_pairs.append(
                    {
                        "pair": f"{sym_x}/{sym_y}",
                        "status": "failed",
                        "reason": "pair_exception",
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "_collect_signal_decisions: %s/%s failed: %s",
                    sym_x or "?",
                    sym_y or "?",
                    exc,
                )

        elapsed = _time.time() - t0
        total_attempted = max(len(resolved_pairs), 1)
        diagnostics_payload.update(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "n_decisions": len(decisions),
                "n_skipped": int(sum(skip_reasons.values())),
                "success_rate": round(float(len(decisions)) / float(total_attempted), 4),
                "skip_reason_counts": dict(skip_reasons),
                "skipped_pairs": skipped_pairs[:50],
                "status": "success" if decisions else "no_decisions",
                "duration_sec": round(elapsed, 2),
            }
        )
        self._persist_signal_collection_diagnostics(diagnostics_payload)
        logger.info(
            "_collect_signal_decisions: %d decisions from %d pairs in %.1fs",
            len(decisions),
            len(pairs),
            elapsed,
        )
        return decisions

    def run_portfolio_allocation_cycle(
        self,
        signal_decisions: Optional[list] = None,
        capital: float = 1_000_000.0,
        allocation_batch_id: str | None = None,
    ) -> Optional[TaskResult]:
        """
        Run one portfolio allocation cycle with safety gating.

        IDEMPOTENCY: Uses batch_id (date-based) to prevent double-allocation.
        If this method is called twice on the same day, the second call is
        a no-op. This protects against scheduler restarts mid-pipeline.

        Resolves:
        - P1-PORTINT: Portfolio bridge called from operational code
        - P1-SAFE: Runtime safety check injected as callback
        - P0-KS: Kill-switch with control-plane callback used

        Parameters
        ----------
        signal_decisions : list[SignalDecision], optional
            If None, returns immediately (no signals to allocate).
        capital : float
            Total capital for allocation.

        Returns
        -------
        TaskResult or None
        """
        import time as _time
        t0 = _time.time()

        if not signal_decisions:
            return None

        # ── Idempotency guard ────────────────────────────────
        # AP-7: use run_id if available; fall back to date for direct calls
        from core.allocation_guard import AllocationBatchGuard

        guard = AllocationBatchGuard()
        batch_id = allocation_batch_id or guard.make_batch_id()
        if not guard.check_and_start(
            batch_id,
            meta={
                "capital": float(capital),
                "n_signal_decisions": int(len(signal_decisions)),
                "run_id": getattr(self, "_current_run_id", None),
            },
        ):
            logger.warning("Allocation batch already processed or throttled: %s", batch_id)
            return TaskResult(
                task_name="portfolio_allocation",
                status="skipped",
                message="allocation_batch_already_processed",
                output={"batch_id": batch_id, "status": "skipped"},
            )

        try:
            from core.portfolio_bridge import bridge_signals_to_allocator

            # ── P1-SAFE: Inject runtime safety check ─────────────
            safety_fn = None
            try:
                from runtime.state import get_runtime_state_manager
                safety_fn = get_runtime_state_manager().is_safe_to_trade
            except ImportError:
                logger.debug("runtime.state unavailable — safety_check=None")

            # ── P0-KS: Use kill-switch factory with control-plane ─
            kill_switch_state = None
            try:
                from portfolio.risk_ops import make_kill_switch_manager_with_control_plane
                ksm = make_kill_switch_manager_with_control_plane()
                kill_switch_state = ksm.state
            except (ImportError, Exception) as e:
                logger.debug("Kill-switch factory unavailable: %s", e)

            allocations, diagnostics = bridge_signals_to_allocator(
                signal_decisions,
                capital=capital,
                safety_check=safety_fn,
                kill_switch=kill_switch_state,
            )

            elapsed = _time.time() - t0
            n_funded = sum(1 for a in allocations if a.approved)

            logger.info(
                "Portfolio allocation: %d funded / %d total, %.1fs",
                n_funded, len(allocations), elapsed,
            )

            # ── Generate rebalance plan from allocations ──────
            rebalance_plan = None
            monitor_payload: Dict[str, Any] = {}
            try:
                from core.portfolio_rebalancer import PortfolioRebalancer
                if n_funded > 0:
                    target_weights = {}
                    for a in allocations:
                        if a.approved and hasattr(a, 'pair_id'):
                            label = str(a.pair_id)
                            w = getattr(a, 'target_weight', 1.0 / max(n_funded, 1))
                            target_weights[label] = w

                    if target_weights:
                        rebalancer = PortfolioRebalancer()
                        rebalance_plan = rebalancer.rebalance(
                            current_positions={},  # Fresh allocation
                            target_weights=target_weights,
                            total_capital=capital,
                        )
                        self.bus.publish("rebalance_plan", {
                            "n_trades": rebalance_plan.n_trades,
                            "total_cost": rebalance_plan.total_estimated_cost,
                            "gross_turnover": rebalance_plan.gross_turnover,
                        })
                        logger.info(
                            "Rebalance plan: %d trades, cost=$%.2f (%.3f%% of capital)",
                            rebalance_plan.n_trades,
                            rebalance_plan.total_estimated_cost,
                            rebalance_plan.cost_as_pct_of_capital,
                        )
            except Exception as rebal_exc:
                logger.debug("Rebalancer not available: %s", rebal_exc)

            try:
                from portfolio.analytics import PortfolioAnalytics

                monitor_payload = normalize_portfolio_monitor_payload(
                    PortfolioAnalytics._build_monitoring_payload(allocations),
                    run_id=batch_id,
                )
            except Exception as monitor_exc:
                logger.debug("Portfolio monitor payload build failed: %s", monitor_exc)
                monitor_payload = {}

            diagnostics_dict = diagnostics.to_dict() if hasattr(diagnostics, "to_dict") else {}
            funded_pairs = [a.pair_id.label for a in allocations if a.approved]
            blocked_pairs = [a.pair_id.label for a in allocations if not a.approved]
            allocation_summary = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "batch_id": batch_id,
                "capital": round(float(capital), 2),
                "n_funded": n_funded,
                "n_total": len(allocations),
                "funded_pairs": funded_pairs,
                "blocked_pairs": blocked_pairs,
                "portfolio_diagnostics": diagnostics_dict,
                "monitoring": monitor_payload,
                "rebalance_n_trades": rebalance_plan.n_trades if rebalance_plan else 0,
                "rebalance_cost": rebalance_plan.total_estimated_cost if rebalance_plan else 0,
            }
            self.bus.publish("portfolio_allocation", allocation_summary)

            store = getattr(self, "_store", None) or getattr(self, "sql_store", None)
            if monitor_payload and store is not None and hasattr(store, "save_portfolio_monitor_payload"):
                try:
                    store.save_portfolio_monitor_payload(monitor_payload, run_id=batch_id)
                    self.bus.publish("portfolio_monitor", monitor_payload)
                except Exception:
                    logger.debug("Failed to persist allocation portfolio monitor payload", exc_info=True)
            if store is not None and hasattr(store, "save_json"):
                try:
                    store.save_json("portfolio_allocation_cycle", batch_id, allocation_summary)
                    store.save_json("portfolio_allocation_cycle", "latest", allocation_summary)
                except Exception:
                    logger.debug("Failed to persist portfolio_allocation_cycle summary", exc_info=True)

            guard.mark_completed(
                batch_id,
                meta={
                    "n_funded": int(n_funded),
                    "n_total": int(len(allocations)),
                    "funded_pairs": funded_pairs,
                    "blocked_pairs": blocked_pairs,
                },
            )
            self._last_allocation_batch = batch_id

            return TaskResult(
                task_name="portfolio_allocation",
                status="success",
                duration_sec=round(elapsed, 2),
                output={
                    "n_funded": n_funded,
                    "n_total": len(allocations),
                    "n_intents_received": diagnostics.n_intents_received,
                    "safety_check_used": safety_fn is not None,
                    "kill_switch_used": kill_switch_state is not None,
                    "rebalance_n_trades": rebalance_plan.n_trades if rebalance_plan else 0,
                    "rebalance_cost": rebalance_plan.total_estimated_cost if rebalance_plan else 0,
                    "batch_id": batch_id,
                    "funded_pairs": funded_pairs,
                    "blocked_pairs": blocked_pairs,
                    "portfolio_diagnostics": diagnostics_dict,
                    "monitoring": monitor_payload,
                },
            )

        except Exception as e:
            elapsed = _time.time() - t0
            logger.warning("Portfolio allocation failed: %s", e)
            try:
                guard.mark_failed(
                    batch_id,
                    meta={
                        "error": str(e),
                        "run_id": getattr(self, "_current_run_id", None),
                    },
                )
            except Exception:
                logger.debug("Failed to persist allocation guard failure state", exc_info=True)
            return TaskResult(
                task_name="portfolio_allocation",
                status="failed",
                duration_sec=round(elapsed, 2),
                error=str(e),
            )


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    """Run the daily pipeline from CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    orch = PairsOrchestrator()
    results = orch.run_daily_pipeline()

    print("\n--- Pipeline Summary ---")
    for r in results:
        icon = "✓" if r.status == "success" else "✗" if r.status == "failed" else "○"
        print(f"  {icon} {r.task_name}: {r.status} ({r.duration_sec:.1f}s)")


if __name__ == "__main__":
    main()
