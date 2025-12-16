# scripts/debug_backtester_params.py
from core.optimization_backtester import Backtester

def run_case(z_entry, z_exit, lookback):
    bt = Backtester(
        symbol_a="XLP",
        symbol_b="XLY",
        z_entry=z_entry,
        z_exit=z_exit,
        lookback=lookback,
        # אם אתה יודע איך קוראים לשמות הפנימיים: z_open, z_close וכו', תנסה גם אותם:
        # z_open=z_entry,
        # z_close=z_exit,
        # lookback_window=lookback,
    )
    perf = bt.run()
    print(f"z_entry={z_entry}, z_exit={z_exit}, lookback={lookback} -> {perf}")

if __name__ == "__main__":
    # שתי קיצוניות לחלוטין
    run_case(0.5, 0.2, 30)
    run_case(3.0, 1.5, 250)
