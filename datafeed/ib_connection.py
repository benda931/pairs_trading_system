# core/ib_connection.py
from __future__ import annotations
from ib_insync import IB, Stock, util
import pandas as pd, functools, time, pathlib, pickle, os

IB_HOST, IB_PORT, IB_CLIENT_ID = "127.0.0.1", 7497, 9
CACHE_DIR = pathlib.Path("_ib_cache"); CACHE_DIR.mkdir(exist_ok=True)

_ib: IB | None = None
def ib() -> IB:
    """Singleton IB connection with auto-reconnect."""
    global _ib
    if _ib is None or not _ib.isConnected():
        _ib = IB()
        _ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, readonly=True)
    return _ib

def _cache_path(sym: str, dur: str, bar: str) -> pathlib.Path:
    fname = f"{sym}_{dur}_{bar}.pkl".replace(" ", "")
    return CACHE_DIR/fname

def get_hist(sym="SPY", days="2 Y", bar="1 day",
             use_cache=True, rth=True) -> pd.Series:
    """
    ׳׳—׳–׳™׳¨ ׳¡׳“׳¨׳× Close ׳”׳™׳¡׳˜׳•׳¨׳™׳× ׳¢׳ Cache.
    ג€¢ days: '2 Y', '180 D', '3 M'
    ג€¢ bar:  '1 day', '30 min', ...
    """
    pkl = _cache_path(sym, days, bar)
    if use_cache and pkl.exists():
        return pickle.loads(pkl.read_bytes())

    contract = Stock(sym, "SMART", "USD")
    ibconn = ib()

    # Pacing control ג€“ 1 req/sec
    now = time.monotonic()
    if getattr(ibconn, "_last_req", 0) + 1.1 > now:
        time.sleep( ibconn._last_req + 1.1 - now )
    bars = ibconn.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=days,
        barSizeSetting=bar,
        whatToShow="ADJUSTED_LAST",
        useRTH=rth,
        formatDate=1,
    )
    ibconn._last_req = time.monotonic()

    if not bars:
        raise RuntimeError(f"No data for {sym}. Check market-data subscription.")

    df = util.df(bars).set_index("date")["close"].ffill()
    if use_cache:
        pkl.write_bytes(pickle.dumps(df))
    return df

