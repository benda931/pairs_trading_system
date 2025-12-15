# -*- coding: utf-8 -*-
"""
scripts/save_zoom_best_params.py — Extract & persist Zoom best params (JSON + SqlStore)
=======================================================================================

תפקיד:
------
1. למצוא את ה-zoom study האחרון לזוג נתון (למשל "zoom::BITO-BKCH::stage1").
2. לשלוף ממנו את ה-best_trial:
   - best_trial.value (Score)
   - best_trial.params (dict מלא)
   - user_attrs (perf / meta אם קיימים).
3. לשמור:
   a. JSON בקובץ data/zoom_best_params/<PAIR>.json
   b. שורה בטבלת zoom_best_params ב-SqlStore (DuckDB).

הרצה לדוגמה:
-------------
    python scripts/save_zoom_best_params.py --pair BITO-BKCH
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
from sqlalchemy import text

from common.zoom_storage import resolve_zoom_storage
from core.sql_store import SqlStore


# =============================================================================
# Dataclasses קטנים לניקיון
# =============================================================================


@dataclass
class ZoomBestRecord:
    pair: str
    study_name: str
    stage: int
    score: float
    n_trials: int
    created_at_utc: datetime
    sampler: str
    direction: str
    params: Dict[str, Any]
    perf: Dict[str, Any]


# =============================================================================
# Helpers
# =============================================================================


def _guess_stage_from_name(study_name: str) -> int:
    """
    מנסה לחלץ את ה-stage מתוך השם, למשל:
    "zoom::BITO-BKCH::stage1" → 1
    אם נכשל → מחזיר 0.
    """
    try:
        parts = study_name.split("::")
        for p in parts:
            if p.startswith("stage"):
                return int(p.replace("stage", "").strip())
    except Exception:
        pass
    return 0


def _ensure_zoom_best_dir(project_root: Path) -> Path:
    out_dir = project_root / "data" / "zoom_best_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_json(record: ZoomBestRecord, project_root: Path) -> Path:
    out_dir = _ensure_zoom_best_dir(project_root)
    out_path = out_dir / f"{record.pair}.json"

    payload = {
        "pair": record.pair,
        "study_name": record.study_name,
        "stage": record.stage,
        "score": record.score,
        "n_trials": record.n_trials,
        "created_at_utc": record.created_at_utc.isoformat(),
        "sampler": record.sampler,
        "direction": record.direction,
        "params": record.params,
        "perf": record.perf,
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _get_default_engine_url() -> str:
    """
    מנבא engine_url של DuckDB, בדומה למה שהטאב משתמש:
    duckdb:///C:/Users/<user>/AppData/Local/pairs_trading_system/cache.duckdb
    """
    home = Path.home()
    duck_path = home / "AppData" / "Local" / "pairs_trading_system" / "cache.duckdb"
    return f"duckdb:///{duck_path}"


def _save_to_sql(record: ZoomBestRecord, engine_url: Optional[str] = None) -> None:
    """
    שומר את ה-zoom-best לטבלה zoom_best_params בתוך SqlStore.

    אם engine_url=None → משתמש בנתיב DuckDB הדיפולטי.
    """
    if engine_url is None:
        engine_url = _get_default_engine_url()

    # נקים SqlStore ב-write mode
    store = SqlStore.from_settings({"engine_url": engine_url}, read_only=False)

    ddl = """
    CREATE TABLE IF NOT EXISTS zoom_best_params (
        pair TEXT,
        study_name TEXT,
        stage INTEGER,
        score DOUBLE,
        n_trials INTEGER,
        created_at_utc TIMESTAMP,
        sampler TEXT,
        direction TEXT,
        params_json TEXT,
        perf_json TEXT
    );
    """

    insert_sql = """
    INSERT INTO zoom_best_params (
        pair, study_name, stage, score, n_trials,
        created_at_utc, sampler, direction,
        params_json, perf_json
    )
    VALUES (
        :pair, :study_name, :stage, :score, :n_trials,
        :created_at_utc, :sampler, :direction,
        :params_json, :perf_json
    );
    """

    params_json = json.dumps(record.params, ensure_ascii=False)
    perf_json = json.dumps(record.perf, ensure_ascii=False)

    with store.engine.begin() as conn:
        conn.exec_driver_sql(ddl)
        conn.execute(
            text(insert_sql),
            {
                "pair": record.pair,
                "study_name": record.study_name,
                "stage": record.stage,
                "score": record.score,
                "n_trials": record.n_trials,
                "created_at_utc": record.created_at_utc,
                "sampler": record.sampler,
                "direction": record.direction,
                "params_json": params_json,
                "perf_json": perf_json,
            },
        )


# =============================================================================
# Main logic
# =============================================================================


def extract_zoom_best(pair: str, project_root: Path) -> ZoomBestRecord:
    """
    מוצא את ה-study האחרון עבור זוג (stage1 וכו'), מוציא את ה-best trial,
    ומחזיר ZoomBestRecord.
    """
    cfg = resolve_zoom_storage(project_root)
    print(f"Storage: {cfg.storage_url}")

    summaries = optuna.study.get_all_study_summaries(storage=cfg.storage_url)
    pair_name = pair.upper()

    candidates = [s for s in summaries if pair_name in s.study_name]
    if not candidates:
        raise RuntimeError(f"לא נמצאו studies שמכילים '{pair_name}' בשם.")

    print(f"\nנמצאו studies עבור {pair_name}:")
    for i, s in enumerate(candidates):
        best_val = getattr(getattr(s, "best_trial", None), "value", None)
        n_trials = getattr(s, "n_trials", None)
        print(f"[{i}] {s.study_name} | best_value={best_val} | n_trials={n_trials}")

    chosen = candidates[-1]
    print(f"\n✅ נשתמש ב-study: {chosen.study_name}")

    study = optuna.load_study(study_name=chosen.study_name, storage=cfg.storage_url)
    best = study.best_trial

    # score
    score = float(best.value)

    # basic metadata
    stage = _guess_stage_from_name(chosen.study_name)
    n_trials = len(study.trials)

    # sampler + direction
    sampler_name = study.sampler.__class__.__name__
    direction = getattr(study, "direction", None)
    if direction is not None:
        direction_str = getattr(direction, "name", str(direction))
    else:
        direction_str = "unknown"

    # params & perf
    params = dict(best.params)
    perf = best.user_attrs.get("perf", {})
    if not isinstance(perf, dict):
        perf = {}

    # created_at
    dt_str = best.user_attrs.get("datetime_complete")
    if isinstance(dt_str, str):
        try:
            created_at = datetime.fromisoformat(dt_str)
        except Exception:
            created_at = datetime.now(tz=timezone.utc)
    else:
        created_at = datetime.now(tz=timezone.utc)

    return ZoomBestRecord(
        pair=pair_name,
        study_name=chosen.study_name,
        stage=stage,
        score=score,
        n_trials=n_trials,
        created_at_utc=created_at,
        sampler=sampler_name,
        direction=direction_str,
        params=params,
        perf=perf,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract zoom best params for a pair and save to JSON + SqlStore."
    )
    parser.add_argument(
        "--pair",
        required=True,
        help="Pair name, למשל 'BITO-BKCH'",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Root של הפרויקט (ברירת מחדל: הנוכחי).",
    )
    parser.add_argument(
        "--engine-url",
        default=None,
        help="SqlStore engine_url (אם לא מצוין → DuckDB דיפולטי ב-AppData/Local/pairs_trading_system/cache.duckdb).",
    )

    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()

    record = extract_zoom_best(args.pair, project_root)

    # JSON
    out_path = _save_json(record, project_root)
    print(f"\n💾 נשמר קובץ פרמטרים ל-{record.pair}:")
    print(f"   {out_path}")

    # SQL
    try:
        _save_to_sql(record, engine_url=args.engine_url)
        print("\n💾 נשמרו גם ב-SqlStore (טבלה: zoom_best_params).")
    except Exception as e:
        print(f"\n⚠️ שמירה ל-SqlStore נכשלה: {e}")
        print("   (ה-JSON נשמר כרגיל, אז תמיד אפשר להשתמש בו כגיבוי.)")


if __name__ == "__main__":
    main()
