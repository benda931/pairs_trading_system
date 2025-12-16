# -*- coding: utf-8 -*-
"""
scripts/maintain_duckdb_cache.py — HF-grade maintenance for cache.duckdb
=======================================================================

תפקידים:
---------
1. verify  — בדיקת מצב בסיס הנתונים + ה-WAL:
   - האם הקובץ קיים.
   - האם WAL קיים.
   - האם אפשר לפתוח חיבור.
   - אילו טבלאות קיימות ב-main schema.

2. repair  — תיקון "חכם":
   - ניסיון לפתוח את ה-DB כמו שהוא.
   - אם נכשל:
       * גיבוי DB + WAL עם חותמת זמן.
       * ניתוק ה-WAL (rename).
       * ניסיון פתיחה מחדש ללא WAL.
       * אם עדיין נכשל → reset מלא.

3. reset   — ריסט יזום:
   - גיבוי DB + WAL (אלא אם כבר גובו).
   - מחיקת DB + WAL.
   - יצירת DB חדש ונקי עם טבלת cache_meta.

התאמה למערכת:
--------------
- ברירת המחדל לנתיב ה-DB נקראת מתוך config.json:
    paths.duckdb_cache_path
  או מתוך:
    data.sql_store.url  (duckdb:///path/to/file.duckdb)

- אם אין config.json או שאין שדות מתאימים:
    - ננסה משתנה סביבה PAIRS_TRADING_CACHE_DB.
    - אחרת ניפול לנתיב LOCALAPPDATA/pairs_trading_system/cache.duckdb.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb


# ========= Project / config helpers =========


def _project_root() -> Path:
    """
    מניח שהסקריפט יושב ב-scripts/ בתוך פרויקט pairs_trading_system
    ומחזיר את השורש (תיקייה אחת מעל scripts).
    """
    return Path(__file__).resolve().parent.parent


def _load_config(config_path: Optional[Path] = None) -> Optional[dict]:
    """
    טוען את config.json אם קיים, אחרת מחזיר None.
    """
    if config_path is None:
        config_path = _project_root() / "config.json"

    if not config_path.exists():
        print(f"[Config] config.json not found at: {config_path}")
        return None

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[Config] Loaded config from: {config_path}")
        return cfg
    except Exception as exc:  # noqa: BLE001
        print(f"[Config] Failed to load config.json: {repr(exc)}")
        return None


def _db_path_from_config(cfg: dict) -> Optional[Path]:
    """
    מנסה להפיק את נתיב ה-DuckDB מתוך config.json.

    עדיפות:
    1. cfg["paths"]["duckdb_cache_path"]
    2. cfg["data"]["sql_store"]["url"] אם הוא duckdb:///...
    """
    # 1) paths.duckdb_cache_path
    try:
        paths = cfg.get("paths") or {}
        cache_path = paths.get("duckdb_cache_path")
        if cache_path:
            return Path(cache_path).expanduser().resolve()
    except Exception:
        pass

    # 2) data.sql_store.url (duckdb:///C:/.../cache.duckdb)
    try:
        data_cfg = cfg.get("data") or {}
        sql_store = data_cfg.get("sql_store") or {}
        url = sql_store.get("url") or ""
        prefix = "duckdb:///"
        if url.startswith(prefix):
            path_str = url[len(prefix) :]
            return Path(path_str).expanduser().resolve()
    except Exception:
        pass

    return None


# ========= Paths & small helpers =========


def _default_db_path() -> Path:
    """
    מחזיר את הנתיב הדיפולטי ל-cache.duckdb.

    סדר עדיפויות:
    1. --db-path מה-CLI (מטופל ב-main).
    2. config.json (paths.duckdb_cache_path או data.sql_store.url).
    3. משתנה סביבה PAIRS_TRADING_CACHE_DB.
    4. LOCALAPPDATA/pairs_trading_system/cache.duckdb.
    """
    # נסה config.json
    cfg = _load_config()
    if cfg is not None:
        cfg_path = _db_path_from_config(cfg)
        if cfg_path is not None:
            print(f"[Path] Using DB path from config.json: {cfg_path}")
            return cfg_path

    # משתנה סביבה
    env_path = os.getenv("PAIRS_TRADING_CACHE_DB")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        print(f"[Path] Using DB path from PAIRS_TRADING_CACHE_DB: {p}")
        return p

    # ברירת מחדל לפי LOCALAPPDATA
    local_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    p = (local_dir / "pairs_trading_system" / "cache.duckdb").resolve()
    print(f"[Path] Using default LOCALAPPDATA DB path: {p}")
    return p


def _wal_path_for(db_path: Path) -> Path:
    return db_path.with_suffix(db_path.suffix + ".wal")


def _timestamp() -> str:
    # משתמש ב-UTC timezone-aware כדי לא לקבל התראות Deprecation
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _backup_file(src: Path, suffix: str) -> Optional[Path]:
    if not src.exists():
        return None
    backup_path = src.with_name(f"{src.name}.{suffix}")
    shutil.copy2(src, backup_path)
    print(f"[Backup] {src.name} -> {backup_path.name}")
    return backup_path


def _backup_db_and_wal(db_path: Path) -> None:
    ts = _timestamp()
    print(f"[Backup] Creating backups with suffix: {ts}")
    _backup_file(db_path, f"broken-{ts}")
    _backup_file(_wal_path_for(db_path), f"broken-{ts}")


# ========= Core operations =========


def verify_db(db_path: Path) -> int:
    """
    בודק אם אפשר לפתוח את ה-DB, האם יש WAL, ומה מצב הטבלאות.
    """
    print(f"[Verify] Using DB path: {db_path}")

    wal_path = _wal_path_for(db_path)
    if wal_path.exists():
        print(f"[Verify] WAL file exists: {wal_path}")
    else:
        print("[Verify] WAL file does not exist.")

    if not db_path.exists():
        print("[Verify] DB file does not exist.")
        return 1

    try:
        conn = duckdb.connect(str(db_path))
    except Exception as e:  # noqa: BLE001
        print(f"[Verify] ERROR: failed to open DB: {repr(e)}")
        return 2

    try:
        print("[Verify] Connected successfully. Running basic checks...")

        try:
            tables = conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                ORDER BY table_name
                """
            ).fetchall()
        except Exception as e_tab:  # noqa: BLE001
            print(f"[Verify] WARNING: failed to list tables: {repr(e_tab)}")
            tables = []

        if tables:
            print("[Verify] Tables in DB (schema=main):")
            for (name,) in tables:
                print(f"   - {name}")
        else:
            print("[Verify] DB is empty (no tables in main schema).")

        # אפשר להוסיף כאן בדיקות נוספות בעתיד (verify_database וכד')
        conn.close()
        print("[Verify] DONE.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[Verify] ERROR during inspection: {repr(e)}")
        try:
            conn.close()
        except Exception:
            pass
        return 3


def reset_db(db_path: Path, already_backed_up: bool = False) -> int:
    """
    ריסט מלא ל-cache.duckdb:

    - גיבוי DB + WAL (אם already_backed_up=False).
    - מחיקת DB + WAL קיימים.
    - יצירת DB חדש ונקי עם טבלת cache_meta.

    שים לב: כיון שמדובר ב-*cache* ולא ב-SqlStore הראשי, מותר לנו
    לעשות rebuild מלא, בתנאי שדואגים ל-ingestion ול-universe מבחוץ.
    """
    print(f"[Reset] Using DB path: {db_path}")
    wal_path = _wal_path_for(db_path)

    if not already_backed_up:
        print("[Reset] Backing up DB + WAL before reset...")
        _backup_db_and_wal(db_path)

    # מחיקה פיזית של הקבצים הישנים
    if db_path.exists():
        db_path.unlink()
        print(f"[Reset] Removed old DB file: {db_path.name}")
    if wal_path.exists():
        wal_path.unlink()
        print(f"[Reset] Removed old WAL file: {wal_path.name}")

    # יצירת DB חדש ונקי
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_meta (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO cache_meta (key, value) VALUES (?, ?)",
        ("created_utc", datetime.now(timezone.utc).isoformat()),
    )
    conn.close()

    print("[Reset] New empty cache.duckdb created.")
    print(
        "[Reset] IMPORTANT: you will need to repopulate dq_pairs / prices via "
        "your ingestion & universe scripts (e.g., build_dq_pairs_universe.py)."
    )
    return 0


def repair_db(db_path: Path) -> int:
    """
    תיקון "חכם":

    1. אם אין DB → אין מה לתקן.
    2. ניסיון לפתוח את ה-DB כמו שהוא.
       - אם מצליח → אין מה לתקן.
       - אם נכשל:
           א. גיבוי DB + WAL.
           ב. ניתוק ה-WAL (rename).
           ג. ניסיון פתיחה ללא WAL.
           ד. אם עדיין נכשל → reset מלא.
    """
    print(f"[Repair] Using DB path: {db_path}")
    wal_path = _wal_path_for(db_path)

    if not db_path.exists():
        print("[Repair] DB file does not exist. Nothing to repair.")
        return 1

    # שלב 1: ניסיון פתיחה רגיל
    print("[Repair] Step 1: try opening DB as-is...")
    try:
        conn = duckdb.connect(str(db_path))
        conn.close()
        print("[Repair] DB opened successfully; nothing to repair.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[Repair] Opening DB failed: {repr(e)}")

    # שלב 2: גיבוי DB + WAL
    print("[Repair] Step 2: backing up DB + WAL before modifications...")
    _backup_db_and_wal(db_path)

    # שלב 3: אם יש WAL – ננתק אותו (rename)
    if wal_path.exists():
        new_wal_name = wal_path.with_name(f"{wal_path.name}.disabled-{_timestamp()}")
        wal_path.rename(new_wal_name)
        print(f"[Repair] Renamed WAL -> {new_wal_name.name}")
    else:
        print("[Repair] No WAL file to detach.")

    # שלב 4: ניסיון פתיחה בלי WAL
    print("[Repair] Step 3: try opening DB again without WAL...")
    try:
        conn = duckdb.connect(str(db_path))
        print("[Repair] DB opened successfully without WAL.")
        # אפשר להוסיף כאן בדיקת טבלאות בסיסית:
        try:
            tables = conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                ORDER BY table_name
                """
            ).fetchall()
            print(f"[Repair] Tables after repair: {[t[0] for t in tables]}")
        except Exception as e_tab:  # noqa: BLE001
            print(f"[Repair] WARNING: failed to list tables after repair: {repr(e_tab)}")

        conn.close()
        print("[Repair] DONE. DB is usable again (WAL detached).")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[Repair] Still cannot open DB after detaching WAL: {repr(e)}")

    # שלב 5: Rebuild מלא
    print("[Repair] Step 4: rebuilding DB from scratch (cache DB semantics).")
    return reset_db(db_path, already_backed_up=True)


# ========= CLI =========


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Maintain DuckDB cache (verify / repair / reset)."
    )
    parser.add_argument(
        "action",
        choices=["verify", "repair", "reset"],
        help="Action to perform on cache.duckdb",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help=(
            "Override DB path. "
            "If not provided, will try config.json, then PAIRS_TRADING_CACHE_DB, "
            "then LOCALAPPDATA/pairs_trading_system/cache.duckdb."
        ),
    )

    args = parser.parse_args()

    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()
        print(f"[Main] DB path overridden via CLI: {db_path}")
    else:
        db_path = _default_db_path()

    print(f"[Main] Action: {args.action}")
    print(f"[Main] DB path: {db_path}")

    if args.action == "verify":
        return verify_db(db_path)
    if args.action == "repair":
        return repair_db(db_path)
    if args.action == "reset":
        return reset_db(db_path)

    # לא אמור להגיע לכאן
    print("[Main] Unknown action.")
    return 99


if __name__ == "__main__":
    raise SystemExit(main())
