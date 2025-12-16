# tools/bootstrap_universe.py
# -*- coding: utf-8 -*-

from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

# לוודא ששורש הפרויקט נמצא ב-sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.app_context import AppContext
from core.sql_store import SqlStore
from core.data_quality import data_quality_pairs_to_sql_ready


def main() -> None:
    app_ctx = AppContext.get_global()
    settings = app_ctx.settings
    store = SqlStore.from_settings(settings)
    env = getattr(settings, "env", "dev")

    # כאן אתה מחליף לזוגות שאתה באמת רוצה באוניברס:
    pairs_list = [
        {
            "sym_x": "SPY",
            "sym_y": "QQQ",
            "score": 0.9,
            "z_entry": 2.0,
            "z_exit": 0.5,
            "lookback": 60,
            "hl_bars": 30,
            "corr_min": 0.8,
        },
        {
            "sym_x": "XLY",
            "sym_y": "XLP",
            "score": 0.85,
            "z_entry": 2.2,
            "z_exit": 0.6,
            "lookback": 80,
            "hl_bars": 40,
            "corr_min": 0.75,
        },
        # תוסיף כאן עוד זוגות אמיתיים שאתה רוצה באוניברס...
    ]

    pairs_df = pd.DataFrame(pairs_list)

    # המרה לפורמט איכות זוגות (כולל field-ים נוספים לפי data_quality_pairs_to_sql_ready)
    df_sql = data_quality_pairs_to_sql_ready(pairs_df)
    df_sql["ts_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    df_sql["run_id"] = "universe_bootstrap_v1"
    df_sql["section"] = "data_quality"
    df_sql["env"] = env

    tbl = store._tbl("dq_pairs")
    df_sql.to_sql(tbl, store.engine, if_exists="replace", index=False)

    print("✅ Universe written to table:", tbl)
    print("Rows:", len(df_sql))


if __name__ == "__main__":
    main()
