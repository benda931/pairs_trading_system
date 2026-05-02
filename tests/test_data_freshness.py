from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from common.data_freshness import FreshnessConfig, validate_pair_frames, validate_price_frame


def _price_frame(end: str, periods: int = 260, *, with_close: bool = True) -> pd.DataFrame:
    index = pd.date_range(end=end, periods=periods, freq="B", tz="UTC")
    data = {"close": [100.0 + i * 0.1 for i in range(periods)]}
    if not with_close:
        data = {"open": [100.0 + i * 0.1 for i in range(periods)]}
    return pd.DataFrame(data, index=index)


def test_validate_price_frame_fresh_dataframe_passes():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    df = _price_frame("2026-05-01 10:00:00+00:00")

    result = validate_price_frame("SPY", df, now=now)

    assert result["ok"] is True
    assert result["reason"] == "ok"
    assert result["n_rows"] == 260


def test_validate_price_frame_empty_dataframe_fails():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    result = validate_price_frame("SPY", pd.DataFrame(), now=now)

    assert result["ok"] is False
    assert result["reason"] == "empty_dataframe"


def test_validate_price_frame_missing_close_fails():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    df = _price_frame("2026-05-01 10:00:00+00:00", with_close=False)

    result = validate_price_frame("SPY", df, now=now)

    assert result["ok"] is False
    assert result["reason"] == "missing_close_column"


def test_validate_price_frame_stale_dataframe_fails():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    df = _price_frame("2026-04-20 16:00:00+00:00")

    result = validate_price_frame(
        "SPY",
        df,
        cfg=FreshnessConfig(max_staleness_hours=36),
        now=now,
    )

    assert result["ok"] is False
    assert result["reason"] == "stale_data"


def test_validate_pair_frames_with_different_last_dates_fails():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    df_x = _price_frame("2026-05-01 10:00:00+00:00")
    df_y = _price_frame("2026-04-25 10:00:00+00:00")

    result = validate_pair_frames(
        "SPY",
        df_x,
        "QQQ",
        df_y,
        cfg=FreshnessConfig(max_staleness_hours=240),
        now=now,
    )

    assert result["ok"] is False
    assert result["reason"] == "leg_date_mismatch"


def test_validate_pair_frames_with_same_dates_passes():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    df_x = _price_frame("2026-05-01 10:00:00+00:00")
    df_y = _price_frame("2026-05-01 09:30:00+00:00")

    result = validate_pair_frames("SPY", df_x, "QQQ", df_y, now=now)

    assert result["ok"] is True
    assert result["reason"] == "ok"
