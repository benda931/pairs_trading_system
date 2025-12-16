# -*- coding: utf-8 -*-
"""
scripts/run_mini_fund_optimize.py — Run Optuna for Mini-Fund pairs
==================================================================

מריץ אופטימיזציה (Optuna) לכל הזוגות במיני-פאנד,
באמצעות api_optimize_pair מתוך root.optimization_tab,
ושומר את התוצאות ל-DuckDB כמו הטאב עצמו.

אחרי הריצה הזו:
- DuckDB יכיל studies/trials לכל זוג.
- run_mini_fund_snapshot.py יחזיר Best_Score/Sharpe אמיתיים.
- run_mini_fund_signals.py יוכל להוציא סיגנלים שמבוססים על פרמטרים אמיתיים.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
import time

# הוספת שורש הפרויקט ל-sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# נשתמש ב-API שכבר כתבת ב-optimization_tab
from root.optimization_tab import api_optimize_pair  # type: ignore


# אותו Universe כמו ב-snapshot
MINI_FUND_PAIRS: List[Tuple[str, str]] = [
    ("SPY", "QQQ"),
    ("XLY", "XLP"),
    ("ITOT", "VTI"),
    ("DIA", "SPY"),
    ("HYG", "SPY"),
]


def main() -> None:
    print("=== Running Optuna for Mini-Fund Pairs ===")
    print(f"Universe size: {len(MINI_FUND_PAIRS)} pairs\n")

    results: List[Dict[str, Any]] = []

    for sym1, sym2 in MINI_FUND_PAIRS:
        pair_label = f"{sym1}-{sym2}"
        print(f"--- Optimizing {pair_label} ---")

        # פרמטרים בסיסיים – אפשר לכוון אותם
        n_trials = 150
        timeout_min = 10
        direction = "maximize"
        sampler_name = "TPE"
        pruner_name = "median"
        profile = "default"

        t0 = time.time()
        df_res, meta = api_optimize_pair(
            sym1,
            sym2,
            ranges=None,
            weights=None,
            n_trials=n_trials,
            timeout_min=timeout_min,
            direction=direction,
            sampler_name=sampler_name,
            pruner_name=pruner_name,
            profile=profile,
            multi_objective=False,
            objective_metrics=None,
            param_mapping=None,
        )
        dt_sec = time.time() - t0

        status = meta.get("status", "unknown")
        best_score = meta.get("best_score")
        study_id = meta.get("study_id")

        print(
            f"Done {pair_label}: status={status}, "
            f"best_score={best_score}, study_id={study_id}, "
            f"duration={dt_sec:.1f}s\n"
        )

        results.append(
            {
                "Pair": pair_label,
                "status": status,
                "best_score": best_score,
                "study_id": study_id,
                "duration_sec": dt_sec,
                "n_trials": n_trials,
                "timeout_min": timeout_min,
            }
        )

    if results:
        df = pd.DataFrame(results)
        out_path = PROJECT_ROOT / "mini_fund_reports" / "mini_fund_opt_runs.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Summary saved to: {out_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
