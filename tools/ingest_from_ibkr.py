from core.ib_data_ingestor import ingest_history_for_universe

if __name__ == "__main__":
    ingest_history_for_universe(
        start="2020-01-01",
        end="2024-12-31",
        max_symbols=20,  # להתחלה, שלא יהיה כבד מדי
    )
