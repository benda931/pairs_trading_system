# -*- coding: utf-8 -*-
"""
data_loader.py — Thin adapter to common.data_loader (legacy compatibility)
=========================================================================

מטרת הקובץ:
-----------
לאפשר לקוד ישן (כמו generate_pairs_universe.py) לעשות:
    from data_loader import load_price_data

אבל בפועל להשתמש במודול החדש:
    common.data_loader
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union
from datetime import datetime, date

import pandas as pd

from common.data_loader import (  # type: ignore[import]
    load_price_data as _core_load_price_data,
    load_prices_multi as _core_load_prices_multi,
    bulk_download as _core_bulk_download,
    DataLoader as _CoreDataLoader,
    settings as settings,
    get_price_metadata as get_price_metadata,
    validate_price_df as validate_price_df,
)


def load_price_data(
    symbol: str,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
    *,
    intraday: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    עטיפה דקה סביב common.data_loader.load_price_data

    תומך גם בחתימה הישנה:
        load_price_data(symbol, start=..., end=..., intraday=True/False)
    """
    if start_date is None and "start" in kwargs:
        start_date = kwargs.pop("start")
    if end_date is None and "end" in kwargs:
        end_date = kwargs.pop("end")

    return _core_load_price_data(
        symbol,
        start_date=start_date,
        end_date=end_date,
        intraday=intraday,
        **kwargs,
    )


def load_prices_multi(
    symbols: Sequence[str],
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
    *,
    intraday: bool = False,
    col: str = "close",
    freq: Optional[str] = None,
    normalize: bool = False,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    return _core_load_prices_multi(
        symbols,
        start_date=start_date,
        end_date=end_date,
        intraday=intraday,
        col=col,
        freq=freq,
        normalize=normalize,
        fill_method=fill_method,
    )


def ensure_prices_cached(
    symbols: Sequence[str],
    *,
    intraday: bool = False,
    force: bool = False,
    days_back: Optional[int] = None,
) -> None:
    _core_bulk_download(
        list(symbols),
        intraday=intraday,
        force=force,
        days_back=days_back,
    )


class DataLoader(_CoreDataLoader):  # type: ignore[misc]
    """
    OOP עטיפה ל-CoreDataLoader — נשמרת לטובת תאימות לאחור.
    """
    pass


__all__ = [
    "load_price_data",
    "load_prices_multi",
    "ensure_prices_cached",
    "DataLoader",
    "settings",
    "get_price_metadata",
    "validate_price_df",
]
