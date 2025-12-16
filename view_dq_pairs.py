# view_dq_pairs.py

from core.sql_store import SqlStore
from types import SimpleNamespace


def main() -> None:
    # סביבה דיפולטיבית כמו ב-dashboard/generate_pairs_universe
    settings_dummy = SimpleNamespace(env="dev")
    store = SqlStore.from_settings(settings_dummy)

    df = store.read_table("dq_pairs")
    if df is None or df.empty:
        print("dq_pairs is empty or not found")
        return

    cols = [c for c in ["sym_x", "sym_y", "corr", "p_value", "half_life", "quality"] if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
