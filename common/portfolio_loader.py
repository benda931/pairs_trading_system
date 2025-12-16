# -*- coding: utf-8 -*-
"""
portfolio_loader.py — טוען פרוטפוליו חי ו-Eq Curve למערכת
=========================================================

המודול הזה הוא שכבת הגשר בין:
- IBKR / ברוקר / DB,
לבין
- טאב הפרוטפוליו (root/portfolio_tab.py).

המטרה:
- להחזיר DataFrame גולמי שמייצג פוזיציות פתוחות.
- להחזיר Equity Curve היסטורי (אם יש).
- לא *להיות* חכם מדי – רק טעינה ונורמליזציה בסיסית.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# אם אתה משתמש ב-DuckDB
try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None  # type: ignore[misc]

# אליאס טייפ פשוט – לא משתמשים בשם duckdb בתוך טייפים
DuckDBConnection = Any

# אם יש לך חיבור IBKR
try:
    from root.ibkr_connection import get_ib_instance  # type: ignore
except Exception:
    get_ib_instance = None  # type: ignore[misc]


# ============================================================
# עזר: חיבור ל-DB (תתאים את הנתיב/שם הקובץ למה שיש אצלך)
# ============================================================

def _get_duckdb_connection() -> Optional[DuckDBConnection]:
    if duckdb is None:
        return None
    db_path = "data/pairs_trading.duckdb"  # תעדכן לנתיב האמיתי שלך
    return duckdb.connect(db_path)

# ============================================================
# API שטאב הפרוטפוליו משתמש בו
# ============================================================

def load_portfolio_snapshot(
    account_id: Optional[str] = None,
    portfolio_group: Optional[str] = None,
) -> pd.DataFrame:
    """
    מחזיר DataFrame של פוזיציות פתוחות.

    אפשרויות:
    1. למשוך ישירות מ-IBKR (TWS / Gateway) אם אתה רוצה צילום מצב Live.
    2. למשוך מטבלת positions ב-DuckDB/SQLite אם אתה מנהל שם את הפוזיציות.
    3. שילוב של שניהם (למשל DB הוא "אמת", ו-IBKR רק וולידציה).

    הפונקציה מחזירה DataFrame *גולמי* — ה-normalization נעשה בתוך
    root.portfolio_tab._normalize_portfolio_snapshot.
    """
    # ===== דוגמה א': אם הפוזיציות נשמרות ב-DuckDB =====
    con = _get_duckdb_connection()
    if con is not None:
        # דוגמה לגמרי גנרית – תתאים לשמות הטבלה/עמודות אצלך
        query = """
        SELECT *
        FROM positions
        WHERE status = 'OPEN'
        """
        # אם יש לך account_id/portfolio_group בעמודות – אפשר לסנן:
        if account_id:
            query += f" AND account_id = '{account_id}'"
        if portfolio_group:
            query += f" AND portfolio_group = '{portfolio_group}'"

        df = con.execute(query).df()
        return df

    # ===== דוגמה ב': אם אתה מושך ישירות מ-IBKR =====
    if get_ib_instance is not None:
        ib = get_ib_instance()
        # כאן תלוי איך אתה מיישם את זה אצלך:
        # positions = ib.positions()  # לדוגמה
        # ואז להפוך ל-DataFrame:
        # rows = []
        # for p in positions:
        #     rows.append({
        #         "symbol_x": p.contract.symbol,
        #         "qty_x": p.position,
        #         "entry_price_x": p.avgCost,
        #         ...
        #     })
        # return pd.DataFrame(rows)
        # כרגע נחזיר DataFrame ריק אם לא מימשת עוד.
        return pd.DataFrame()

    # ===== ברירת מחדל: אין חיבור – מחזירים ריק =====
    return pd.DataFrame()


def load_equity_curve(
    account_id: Optional[str] = None,
    portfolio_group: Optional[str] = None,
) -> pd.DataFrame:
    """
    מחזיר Equity Curve היסטורי (אם יש).

    מקורות אפשריים:
    - טבלת equity_curve ב-DuckDB.
    - לוג היסטורי של NetLiquidation / NLV על בסיס IBKR.
    - Backtest שמייצר Equity Curve.
    """
    con = _get_duckdb_connection()
    if con is None:
        return pd.DataFrame()

    query = """
    SELECT *
    FROM equity_curve
    ORDER BY timestamp
    """
    if account_id:
        query += f" WHERE account_id = '{account_id}'"
    # אם יש גם portfolio_group בתור שדה – ניתן להוסיף תנאי נוסף

    df = con.execute(query).df()
    return df
