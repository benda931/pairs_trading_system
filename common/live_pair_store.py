# -*- coding: utf-8 -*-
"""
common/live_pair_store.py — Live Pair Profile Store (DuckDB, HF-Grade)
======================================================================

שכבת אחסון מקצועית ל-LivePairProfile מעל DuckDB.

תפקיד
------
הקובץ הזה הוא "מסד הנתונים" של ה־*Live Universe* שלך:
    - מחזיק את כל ה-LivePairProfile (קובץ live_profiles.py).
    - משמש כגשר בין:
        • Research / Optimization / ML  → כתיבה (bulk_upsert)
        • Live Trading Engine           → קריאה (load_for_engine / load_active)
        • Dashboard / Tabs              → קריאה + עדכונים (suspend/activate/priority)

עקרונות תכנון
--------------
1. **Source of Truth אחד** לכל הזוגות החיים:
   - מי מאושר למסחר (is_active),
   - מי מושהה (is_suspended + reason),
   - מה הפרופיל המלא (ז, סטופים, sizing, ML, Regime, QA וכו').

2. **גמישות קדימה**:
   - כל הפרופיל נשמר גם כ-JSON ב-column `profile_json`.
   - אפשר להוסיף שדות ל-LivePairProfile בעתיד בלי לשבור את DB.

3. **שאילתות יעילות**:
   - עמודות יעודיות לסינון/מיון:
     pair_id, sym_x, sym_y,
     is_active, is_suspended, priority_rank,
     score_total, ml_edge_score, ml_confidence, regime_id,
     model_version, last_* timestamps.

4. **API נקי ופשוט**:
   - load_all / load_active / load_for_engine
   - get_by_id / find_by_symbols
   - upsert / bulk_upsert
   - activate / deactivate / suspend / unsuspend
   - summary() לדשבורד

שימוש טיפוסי
-------------
    from common.live_profiles import LivePairProfile
    from common.live_pair_store import LivePairStore

    store = LivePairStore("data/live_pairs.duckdb")

    # כתיבה מאופטימיזציה / ML:
    store.bulk_upsert(profiles_from_opt)

    # קריאה למנוע מסחר חי:
    profiles_for_engine = store.load_for_engine(
        min_score=0.0,
        min_ml_edge=None,
        only_not_suspended=True,
        limit=50,
    )

    # קריאה לדשבורד:
    all_profiles = store.load_all()
    summary = store.summary()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import duckdb

from .live_profiles import LivePairProfile

logger = logging.getLogger(__name__)


class LivePairStore:
    """
    מחסן פרופילי לייב מעל DuckDB.

    Design:
        - טבלה אחת: live_pairs_profile
        - primary key: pair_id
        - profile_json: JSON מלא של LivePairProfile (לגמישות קדימה)
        - עמודות מפתח לקוורי:
            sym_x, sym_y,
            is_active, is_suspended, priority_rank,
            score_total, ml_edge_score, ml_confidence, regime_id,
            model_version, last_optimized_at, last_backtest_at, last_ml_update_at

    הערות:
        - החיבור נשמר פתוח לאורך חיי האובייקט (לתהליך אחד).
        - המימוש **לא** מיועד בו זמנית לכמה תהליכים שכותבים במקביל (זה DuckDB),
          אבל זה לגמרי מספיק למחקר + מנוע חי בתהליך אחד, או Live Engine + Dashboard
          בתצורה שמוגדרת היטב.
    """

    def __init__(
        self,
        db_path: str | Path = "data/live_pairs.duckdb",
        table_name: str = "live_pairs_profile",
    ) -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name

        # נוודא שהתיקייה קיימת (למשל data/)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug("Opening DuckDB connection at %s", self.db_path)
        self._conn = duckdb.connect(str(self.db_path))
        self._ensure_schema()
        self._ensure_indexes()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "LivePairStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """סגירת חיבור ה-DuckDB (אם חי)."""
        if getattr(self, "_conn", None) is not None:
            try:
                self._conn.close()
            except Exception as e:  # pragma: no cover - best-effort cleanup
                logger.warning("Error closing DuckDB connection: %s", e)
            finally:
                self._conn = None  # type: ignore[assignment]

    # ======================================================================
    # סכימה ואינדקסים
    # ======================================================================
    def _ensure_schema(self) -> None:
        """
        מוודא שהטבלה קיימת עם סכימה מינימלית.

        כל התוכן המלא של LivePairProfile נשמר גם ב-profile_json (JSON).
        עמודות נפרדות משמשות לשאילתות יעילות.
        """
        t = self.table_name

        logger.debug("Ensuring schema for table '%s'", t)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t} (
                pair_id TEXT PRIMARY KEY,
                sym_x TEXT NOT NULL,
                sym_y TEXT NOT NULL,

                -- זהות בסיסית
                asset_class   TEXT,
                base_currency TEXT,
                timeframe     TEXT,
                cluster_id    TEXT,

                -- סטטוס
                is_active    BOOLEAN,
                is_suspended BOOLEAN,
                priority_rank BIGINT,

                -- ציונים ו-ML
                score_total      DOUBLE,
                ml_edge_score    DOUBLE,
                ml_confidence    DOUBLE,
                regime_id        TEXT,
                model_version    TEXT,

                -- זמני עדכון
                last_optimized_at TIMESTAMP,
                last_backtest_at  TIMESTAMP,
                last_ml_update_at TIMESTAMP,

                -- הפרופיל המלא כ-JSON
                profile_json TEXT NOT NULL
            );
            """
        )

    def _ensure_indexes(self) -> None:
        """
        מוודא אינדקסים בסיסיים לשאילתות נפוצות.
        DuckDB לא תמיד דורש אינדקסים, אבל זה עוזר ל-ORDER BY / WHERE.
        """
        t = self.table_name
        try:
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{t}_active ON {t}(is_active);")
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{t}_score ON {t}(score_total);")
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{t}_ml_edge ON {t}(ml_edge_score);")
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{t}_regime ON {t}(regime_id);")
        except Exception as e:  # pragma: no cover - קשור לגרסאות DuckDB
            logger.debug("Index creation failed or not supported: %s", e)

    # ======================================================================
    # Helper: המרה Profile <-> Row
    # ======================================================================
    @staticmethod
    def _profile_to_row(profile: LivePairProfile) -> dict:
        """
        ממיר LivePairProfile ל-row dict עבור DB.

        שומר:
            - עמודות מפתח ל-query.
            - JSON מלא בטור profile_json (כולל תאריכים בפורמט ISO).
        """
        data = profile.model_dump(mode="python")

        def _dt(val: Optional[datetime]) -> Optional[str]:
            return val.isoformat() if isinstance(val, datetime) else None

        row = {
            "pair_id": profile.pair_id,
            "sym_x": profile.sym_x,
            "sym_y": profile.sym_y,
            "asset_class": data.get("asset_class"),
            "base_currency": data.get("base_currency"),
            "timeframe": data.get("timeframe"),
            "cluster_id": data.get("cluster_id"),
            "is_active": data.get("is_active"),
            "is_suspended": data.get("is_suspended"),
            "priority_rank": data.get("priority_rank"),
            "score_total": data.get("score_total"),
            "ml_edge_score": data.get("ml_edge_score"),
            "ml_confidence": data.get("ml_confidence"),
            "regime_id": data.get("regime_id"),
            "model_version": data.get("model_version"),
            "last_optimized_at": _dt(data.get("last_optimized_at")),
            "last_backtest_at": _dt(data.get("last_backtest_at")),
            "last_ml_update_at": _dt(data.get("last_ml_update_at")),
            "profile_json": json.dumps(data, default=str),
        }
        return row

    @staticmethod
    def _row_to_profile(row) -> LivePairProfile:
        """
        ממיר Row מה-DB ל-LivePairProfile ע"י טעינת ה-JSON.
        """
        profile_json = row["profile_json"]
        data = json.loads(profile_json)
        # Pydantic יטפל בהמרות datetime וכו'
        return LivePairProfile.model_validate(data)

    # ======================================================================
    # CRUD בסיסי
    # ======================================================================
    def upsert(self, profile: LivePairProfile) -> None:
        """
        הכנסת/עדכון פרופיל בודד.

        Implemenation:
            DELETE + INSERT בתוך אותה טרנזקציה (פשוט ויציב ב-DuckDB).
        """
        t = self.table_name
        row = self._profile_to_row(profile)

        logger.debug("Upserting live profile pair_id=%s", profile.pair_id)
        self._conn.execute("BEGIN TRANSACTION;")
        try:
            self._conn.execute(
                f"DELETE FROM {t} WHERE pair_id = ?;",
                (row["pair_id"],),
            )
            self._conn.execute(
                f"""
                INSERT INTO {t} (
                    pair_id, sym_x, sym_y,
                    asset_class, base_currency, timeframe, cluster_id,
                    is_active, is_suspended, priority_rank,
                    score_total, ml_edge_score, ml_confidence, regime_id, model_version,
                    last_optimized_at, last_backtest_at, last_ml_update_at,
                    profile_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    row["pair_id"],
                    row["sym_x"],
                    row["sym_y"],
                    row["asset_class"],
                    row["base_currency"],
                    row["timeframe"],
                    row["cluster_id"],
                    row["is_active"],
                    row["is_suspended"],
                    row["priority_rank"],
                    row["score_total"],
                    row["ml_edge_score"],
                    row["ml_confidence"],
                    row["regime_id"],
                    row["model_version"],
                    row["last_optimized_at"],
                    row["last_backtest_at"],
                    row["last_ml_update_at"],
                    row["profile_json"],
                ),
            )
            self._conn.execute("COMMIT;")
        except Exception as e:
            logger.error("Upsert failed for pair_id=%s: %s", profile.pair_id, e)
            self._conn.execute("ROLLBACK;")
            raise

    def bulk_upsert(self, profiles: Iterable[LivePairProfile]) -> None:
        """
        הכנסת/עדכון של אוסף פרופילים.

        מיועד לשימוש מה-Optimization/ML:
            - מריצים Optuna/ML → בונים רשימת LivePairProfile → bulk_upsert.

        לא מוחק זוגות שלא הופיעו פה – זה נשלט ע"י deactivate_missing().
        """
        profiles = list(profiles)
        if not profiles:
            return

        t = self.table_name
        rows = [self._profile_to_row(p) for p in profiles]

        logger.info("Bulk upsert of %d live profiles", len(rows))
        self._conn.execute("BEGIN TRANSACTION;")
        try:
            for row in rows:
                self._conn.execute(
                    f"DELETE FROM {t} WHERE pair_id = ?;",
                    (row["pair_id"],),
                )
                self._conn.execute(
                    f"""
                    INSERT INTO {t} (
                        pair_id, sym_x, sym_y,
                        asset_class, base_currency, timeframe, cluster_id,
                        is_active, is_suspended, priority_rank,
                        score_total, ml_edge_score, ml_confidence, regime_id, model_version,
                        last_optimized_at, last_backtest_at, last_ml_update_at,
                        profile_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        row["pair_id"],
                        row["sym_x"],
                        row["sym_y"],
                        row["asset_class"],
                        row["base_currency"],
                        row["timeframe"],
                        row["cluster_id"],
                        row["is_active"],
                        row["is_suspended"],
                        row["priority_rank"],
                        row["score_total"],
                        row["ml_edge_score"],
                        row["ml_confidence"],
                        row["regime_id"],
                        row["model_version"],
                        row["last_optimized_at"],
                        row["last_backtest_at"],
                        row["last_ml_update_at"],
                        row["profile_json"],
                    ),
                )
            self._conn.execute("COMMIT;")
        except Exception as e:
            logger.error("Bulk upsert failed: %s", e)
            self._conn.execute("ROLLBACK;")
            raise

    def deactivate_missing(self, existing_pair_ids: Sequence[str]) -> int:
        """
        מסמן כלא-פעילים זוגות שלא הופיעו ברשימת pair_id שניתנה.

        שימושי אחרי bulk_upsert:
            - universe חדש הגיע מאופטימיזציה,
            - מה שלא בפנים → is_active=False.

        מחזיר: מספר השורות שעודכנו.
        """
        t = self.table_name
        if not existing_pair_ids:
            return 0

        # DuckDB לא תומך בקלות ב-NOT IN עם רשימה גדולה מאוד, אבל כאן זה סביר.
        placeholders = ", ".join(["?"] * len(existing_pair_ids))
        query = f"""
            UPDATE {t}
            SET is_active = FALSE
            WHERE pair_id NOT IN ({placeholders});
        """
        logger.info("Deactivating pairs missing from new universe (%d ids)", len(existing_pair_ids))
        res = self._conn.execute(query, existing_pair_ids)
        return res.rowcount or 0

    # ======================================================================
    # GET / FIND / LOAD
    # ======================================================================
    def get_by_id(self, pair_id: str) -> Optional[LivePairProfile]:
        """מביא פרופיל לזוג לפי pair_id (או None אם לא קיים)."""
        t = self.table_name
        res = self._conn.execute(
            f"SELECT * FROM {t} WHERE pair_id = ?;",
            (pair_id,),
        ).fetchone()
        if res is None:
            return None
        return self._row_to_profile(res)

    def find_by_symbols(self, sym_x: str, sym_y: str) -> List[LivePairProfile]:
        """
        מחזיר כל הפרופילים שמתאימים ל-sym_x/sym_y (תיאורטית יכולות להיות כמה גרסאות).
        """
        t = self.table_name
        res = self._conn.execute(
            f"SELECT * FROM {t} WHERE sym_x = ? AND sym_y = ?;",
            (sym_x, sym_y),
        ).fetchall()
        return [self._row_to_profile(r) for r in res]

    def load_all(self, order_by: str = "pair_id") -> List[LivePairProfile]:
        """
        טעינת כל הפרופילים (לשימוש Dashboard/ניהול).

        order_by:
            'pair_id', 'score_total', 'priority_rank', 'ml_edge_score' וכו'.
        """
        t = self.table_name
        allowed = {
            "pair_id",
            "sym_x",
            "sym_y",
            "score_total",
            "priority_rank",
            "ml_edge_score",
            "regime_id",
        }
        if order_by not in allowed:
            order_by = "pair_id"

        res = self._conn.execute(
            f"SELECT * FROM {t} ORDER BY {order_by};"
        ).fetchall()
        return [self._row_to_profile(r) for r in res]

    def load_active(
        self,
        min_score: Optional[float] = None,
        min_ml_edge: Optional[float] = None,
        only_not_suspended: bool = True,
        limit: Optional[int] = None,
    ) -> List[LivePairProfile]:
        """
        טעינת פרופילים פעילים (למשל ל-Dashboard או לניתוח).

        פילטרים:
            - is_active = TRUE
            - אם only_not_suspended: is_suspended = FALSE
            - min_score: גילוח לפי score_total
            - min_ml_edge: גילוח לפי ml_edge_score
        """
        t = self.table_name
        conditions = ["is_active = TRUE"]
        params: list = []

        if only_not_suspended:
            conditions.append("COALESCE(is_suspended, FALSE) = FALSE")

        if min_score is not None:
            conditions.append("score_total >= ?")
            params.append(min_score)

        if min_ml_edge is not None:
            conditions.append("ml_edge_score IS NOT NULL AND ml_edge_score >= ?")
            params.append(min_ml_edge)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

        query = f"""
            SELECT * FROM {t}
            WHERE {where_clause}
            ORDER BY
                COALESCE(priority_rank, 999999),
                score_total DESC,
                ml_edge_score DESC NULLS LAST
            {limit_clause};
        """

        res = self._conn.execute(query, params).fetchall()
        return [self._row_to_profile(r) for r in res]

    def load_for_engine(
        self,
        min_score: Optional[float] = None,
        min_ml_edge: Optional[float] = None,
        max_priority_rank: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[LivePairProfile]:
        """
        טעינת universe שהמנוע החי אמור לעבוד עליו.

        לוגיקה:
            - is_active = TRUE
            - is_suspended = FALSE
            - score_total >= min_score (אם קיים)
            - ml_edge_score >= min_ml_edge (אם קיים)
            - priority_rank <= max_priority_rank (אם קיים)
        """
        t = self.table_name
        conditions = [
            "is_active = TRUE",
            "COALESCE(is_suspended, FALSE) = FALSE",
        ]
        params: list = []

        if min_score is not None:
            conditions.append("score_total >= ?")
            params.append(min_score)

        if min_ml_edge is not None:
            conditions.append("ml_edge_score IS NOT NULL AND ml_edge_score >= ?")
            params.append(min_ml_edge)

        if max_priority_rank is not None:
            conditions.append("priority_rank IS NOT NULL AND priority_rank <= ?")
            params.append(max_priority_rank)

        where_clause = " AND ".join(conditions)
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

        query = f"""
            SELECT * FROM {t}
            WHERE {where_clause}
            ORDER BY
                COALESCE(priority_rank, 999999),
                score_total DESC,
                ml_edge_score DESC NULLS LAST
            {limit_clause};
        """

        logger.debug(
            "Loading profiles for engine (min_score=%s, min_ml_edge=%s, max_priority=%s, limit=%s)",
            min_score,
            min_ml_edge,
            max_priority_rank,
            limit,
        )
        res = self._conn.execute(query, params).fetchall()
        return [self._row_to_profile(r) for r in res]

    # ======================================================================
    # פעולות סטטוס / שליטה (לשימוש Dashboard / Ops)
    # ======================================================================
    def set_active(self, pair_id: str, is_active: bool) -> None:
        """עדכון is_active לזוג מסוים."""
        t = self.table_name
        logger.info("Set is_active=%s for pair_id=%s", is_active, pair_id)
        self._conn.execute(
            f"UPDATE {t} SET is_active = ? WHERE pair_id = ?;",
            (is_active, pair_id),
        )

    def suspend(self, pair_id: str, reason: Optional[str] = None) -> None:
        """מסמן זוג כמושעה (is_suspended=True) עם סיבת הקפאה."""
        t = self.table_name
        logger.info("Suspending pair_id=%s reason=%s", pair_id, reason)
        # נעדכן גם בתוך JSON כדי לשמור עקביות
        profile = self.get_by_id(pair_id)
        if profile is not None:
            profile.is_suspended = True
            profile.suspend_reason = reason
            self.upsert(profile)
        else:
            self._conn.execute(
                f"UPDATE {t} SET is_suspended = TRUE WHERE pair_id = ?;",
                (pair_id,),
            )

    def unsuspend(self, pair_id: str) -> None:
        """מסיר הקפאה (is_suspended=False, suspend_reason=None)."""
        t = self.table_name
        logger.info("Unsuspending pair_id=%s", pair_id)
        profile = self.get_by_id(pair_id)
        if profile is not None:
            profile.is_suspended = False
            profile.suspend_reason = None
            self.upsert(profile)
        else:
            self._conn.execute(
                f"UPDATE {t} SET is_suspended = FALSE, suspend_reason = NULL WHERE pair_id = ?;",
                (pair_id,),
            )

    def update_priority(self, pair_id: str, priority_rank: Optional[int]) -> None:
        """עדכון priority_rank (למשל מתוך Dashboard)."""
        t = self.table_name
        logger.info("Updating priority_rank=%s for pair_id=%s", priority_rank, pair_id)
        profile = self.get_by_id(pair_id)
        if profile is not None:
            # LivePairProfile לא מחייב field priority_rank, אבל הוא קיים בדאטה
            data = profile.model_dump(mode="python")
            data["priority_rank"] = priority_rank
            updated = LivePairProfile.model_validate(data)
            self.upsert(updated)
        else:
            self._conn.execute(
                f"UPDATE {t} SET priority_rank = ? WHERE pair_id = ?;",
                (priority_rank, pair_id),
            )

    # ======================================================================
    # Summary / Analytics (לשימוש Dashboard)
    # ======================================================================
    def summary(self) -> dict:
        """
        מחזיר סיכום מהיר ל-Dashboard:
            - כמה זוגות יש.
            - כמה פעילים / מושהים.
            - סטטיסטיקות בסיסיות על score_total / ml_edge_score.
        """
        t = self.table_name
        res = self._conn.execute(
            f"""
            SELECT
                COUNT(*)                            AS total_pairs,
                SUM(CASE WHEN is_active    THEN 1 ELSE 0 END) AS active_pairs,
                SUM(CASE WHEN is_suspended THEN 1 ELSE 0 END) AS suspended_pairs,
                AVG(score_total)                   AS avg_score,
                MIN(score_total)                   AS min_score,
                MAX(score_total)                   AS max_score,
                AVG(ml_edge_score)                 AS avg_ml_edge,
                MIN(ml_edge_score)                 AS min_ml_edge,
                MAX(ml_edge_score)                 AS max_ml_edge
            FROM {t};
            """
        ).fetchone()

        if res is None:
            return {
                "total_pairs": 0,
                "active_pairs": 0,
                "suspended_pairs": 0,
                "avg_score": None,
                "min_score": None,
                "max_score": None,
                "avg_ml_edge": None,
                "min_ml_edge": None,
                "max_ml_edge": None,
            }

        return {
            "total_pairs": res["total_pairs"],
            "active_pairs": res["active_pairs"],
            "suspended_pairs": res["suspended_pairs"],
            "avg_score": res["avg_score"],
            "min_score": res["min_score"],
            "max_score": res["max_score"],
            "avg_ml_edge": res["avg_ml_edge"],
            "min_ml_edge": res["min_ml_edge"],
            "max_ml_edge": res["max_ml_edge"],
        }
