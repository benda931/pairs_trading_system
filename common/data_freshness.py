from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


class DataFreshnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class FreshnessConfig:
    max_staleness_hours: int = 36
    min_rows: int = 252
    require_datetime_index: bool = True
    require_close_column: bool = True
    max_leg_date_diff_days: int = 3
    allow_weekend_gap: bool = True


def _utc_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _has_weekend_grace(last_ts: pd.Timestamp, now_utc: datetime, cfg: FreshnessConfig) -> bool:
    if not cfg.allow_weekend_gap:
        return False
    if last_ts.weekday() != 4:
        return False
    if now_utc.weekday() in (5, 6):
        return True
    if now_utc.weekday() == 0 and now_utc.hour < 21:
        return True
    return False


def latest_timestamp(df: pd.DataFrame) -> pd.Timestamp | None:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None
    clean_index = df.index.dropna()
    if len(clean_index) == 0:
        return None
    return _to_utc_timestamp(clean_index.max())


def validate_price_frame(
    symbol: str,
    df: pd.DataFrame,
    cfg: FreshnessConfig | None = None,
    now: datetime | None = None,
) -> dict:
    cfg = cfg or FreshnessConfig()
    now_utc = _utc_now(now)
    symbol_txt = str(symbol or "").strip().upper()
    has_close = bool(df is not None and "close" in getattr(df, "columns", []))
    n_rows = int(len(df)) if df is not None else 0

    payload = {
        "symbol": symbol_txt,
        "ok": False,
        "reason": "unknown",
        "last_ts": None,
        "age_hours": None,
        "n_rows": n_rows,
        "has_close": has_close,
    }

    if df is None or df.empty:
        payload["reason"] = "empty_dataframe"
        return payload

    if cfg.require_close_column and not has_close:
        payload["reason"] = "missing_close_column"
        return payload

    if cfg.require_datetime_index and not isinstance(df.index, pd.DatetimeIndex):
        payload["reason"] = "non_datetime_index"
        return payload

    if n_rows < int(cfg.min_rows):
        payload["reason"] = "insufficient_rows"
        return payload

    last_ts = latest_timestamp(df)
    if last_ts is None:
        payload["reason"] = "missing_last_timestamp"
        return payload

    age_hours = max(0.0, (now_utc - last_ts.to_pydatetime()).total_seconds() / 3600.0)
    payload["last_ts"] = last_ts.isoformat()
    payload["age_hours"] = round(float(age_hours), 4)

    if age_hours > float(cfg.max_staleness_hours) and not _has_weekend_grace(last_ts, now_utc, cfg):
        payload["reason"] = "stale_data"
        return payload

    payload["ok"] = True
    payload["reason"] = "ok"
    return payload


def validate_pair_frames(
    sym_x: str,
    df_x: pd.DataFrame,
    sym_y: str,
    df_y: pd.DataFrame,
    cfg: FreshnessConfig | None = None,
    now: datetime | None = None,
) -> dict:
    cfg = cfg or FreshnessConfig()
    x_result = validate_price_frame(sym_x, df_x, cfg=cfg, now=now)
    y_result = validate_price_frame(sym_y, df_y, cfg=cfg, now=now)

    payload = {
        "ok": False,
        "reason": "unknown",
        "x": x_result,
        "y": y_result,
    }

    if not x_result["ok"] or not y_result["ok"]:
        if x_result["ok"] != y_result["ok"]:
            payload["reason"] = "one_leg_invalid"
        else:
            payload["reason"] = "both_legs_invalid"
        return payload

    x_last = _to_utc_timestamp(x_result.get("last_ts"))
    y_last = _to_utc_timestamp(y_result.get("last_ts"))
    if x_last is None or y_last is None:
        payload["reason"] = "missing_last_timestamp"
        return payload

    leg_gap_days = abs((x_last.normalize() - y_last.normalize()).days)
    if leg_gap_days > int(cfg.max_leg_date_diff_days):
        payload["reason"] = "leg_date_mismatch"
        return payload

    payload["ok"] = True
    payload["reason"] = "ok"
    return payload


def assert_pair_fresh(
    sym_x: str,
    df_x: pd.DataFrame,
    sym_y: str,
    df_y: pd.DataFrame,
    cfg: FreshnessConfig | None = None,
    now: datetime | None = None,
) -> None:
    result = validate_pair_frames(sym_x, df_x, sym_y, df_y, cfg=cfg, now=now)
    if not result.get("ok"):
        raise DataFreshnessError(
            f"Freshness check failed for {str(sym_x).upper()}/{str(sym_y).upper()}: "
            f"{result.get('reason')}"
        )


def load_price_data_guarded(
    symbol: str,
    *args,
    freshness: FreshnessConfig | None = None,
    **kwargs,
) -> pd.DataFrame:
    from common.data_loader import load_price_data

    df = load_price_data(symbol, *args, **kwargs)
    result = validate_price_frame(symbol, df, cfg=freshness)
    if not result.get("ok"):
        raise DataFreshnessError(
            f"Freshness check failed for {str(symbol).upper()}: {result.get('reason')}"
        )
    return df
