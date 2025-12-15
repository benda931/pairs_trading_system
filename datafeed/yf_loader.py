import yfinance as yf, pandas as pd
def load_prices(tickers=("SPY","QQQ"), start="2020-01-01"):
    df = yf.download(list(tickers), start=start)["Adj Close"]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    return df.ffill()

