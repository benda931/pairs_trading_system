from ib_insync import IB, Stock
import pandas as pd

def main():
    ib = IB()
    print("Connecting to IBKR...")
    # Paper: 7497, Live: 7496
    ib.connect('127.0.0.1', 7497, clientId=1)

    print("Connected:", ib.isConnected())
    if not ib.isConnected():
        print("❌ Connection failed (check TWS/Gateway/API).")
        return

    # בדיקה: מחירי SPY יומיים לשנתיים אחרונות
    contract = Stock('SPY', 'SMART', 'USD')
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',           # ריק = "עד עכשיו"
        durationStr='2 Y',        # חלון של שנתיים
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1,
    )

    print(f"Bars fetched for SPY: {len(bars)}")
    if bars:
        df = pd.DataFrame(bars)
        print("First bar:", df.iloc[0])
        print("Last bar :", df.iloc[-1])

    ib.disconnect()
    print("Disconnected.")

if __name__ == "__main__":
    main()
