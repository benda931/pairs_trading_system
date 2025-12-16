import logging
logging.basicConfig(level=logging.INFO)
# נ“„ volatility.py ג€“ ׳—׳™׳©׳•׳‘ ATR (Average True Range)

import pandas as pd

def calculate_atr(df, window=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

