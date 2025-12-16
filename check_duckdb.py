import duckdb
import os

print("CWD:", os.getcwd())

con = duckdb.connect(r"logs\pairs_trading_dev.duckdb")

print("\n=== Tables ===")
print(con.sql("SHOW TABLES").fetchdf())

print("\n=== Prices count ===")
print(con.sql("SELECT COUNT(*) AS n_rows FROM prices").fetchdf())

print("\n=== Rows per symbol/env ===")
print(con.sql("""
    SELECT symbol, env, COUNT(*) AS n_rows
    FROM prices
    GROUP BY symbol, env
    ORDER BY symbol, env
""").fetchdf())
