# scripts/sql_store_health_check.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is on sys.path so `import core...` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sql_store import SqlStore  # now this should work


def main() -> None:
    print("=== SqlStore health check ===")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("sys.path[0]:", sys.path[0])

    print("Creating SqlStore.from_settings(read_only=True)...")
    store = SqlStore.from_settings({}, read_only=True)
    print("SqlStore created.")
    print("  engine_url  :", store.engine_url)
    print("  default_env :", store.default_env)

    # Simple connectivity check
    try:
        with store.engine.connect() as conn:
            val = conn.exec_driver_sql("SELECT 1").scalar()
        print("Health check OK, SELECT 1 ->", val)
    except Exception as e:
        print("Health check FAILED:", repr(e))


if __name__ == "__main__":
    main()
