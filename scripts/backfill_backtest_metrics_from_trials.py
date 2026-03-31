# scripts/backfill_backtest_metrics_from_trials.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from core.app_context import AppContext
from core.sql_store import SqlStore


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--only-complete", action="store_true", default=True)
    args = p.parse_args()

    ctx = AppContext.get_global()
    store = SqlStore.from_settings(ctx.settings, read_only=False)

    n = store.backfill_pair_backtest_metrics_from_trials(
        env=args.env,
        profile=args.profile,
        limit_pairs=int(args.limit),
        only_complete=bool(args.only_complete),
    )

    print(f"[backfill] wrote metrics for {n} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
