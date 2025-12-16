# -*- coding: utf-8 -*-
"""
scripts/build_dq_pairs_universe.py — Build dq_pairs universe into DuckDB cache
==============================================================================

תפקידים:
---------
1. לקרוא יוניברס זוגות מ-config.json:
   - אחד מהשדות:
       * dq_pairs
       * pairs
       * ranked_pairs
       * pairs_universe
   - תומך בפורמטים:
       * רשימת dict-ים (עם sym_x/sym_y או שמות שדות חלופיים).
       * רשימת רשימות/tuple-ים: ["SPY", "QQQ"].
       * רשימת מחרוזות: "SPY-QQQ" / "SPY/QQQ" / "SPY:QQQ".

2. אם בקונפיג אין יוניברס שימושי:
   - טוען מקובץ CSV שמוגדר תחת:
       * pairs_file (למשל "pairs_universe.csv"),
     עם גמישות בשמות העמודות.

3. בונה טבלת dq_pairs בתוך DuckDB cache:
   - שדות בסיס:
       sym_x, sym_y
   - שדות משלימים:
       source        ("config" / "csv")
       raw_score     (אם הגיע מהקונפיג/CSV)
       active        (ברירת מחדל: TRUE)
       created_ts_utc

4. Drop & recreate:
   - מוחק טבלת dq_pairs אם קיימת (אלא אם --append).
   - יוצר מחדש עם סכמה נקייה וטעינת כל הזוגות.

ברירת מחדל לנתיב DB:
---------------------
1. אם עבר --sql-url:
   - אם מתחיל ב-"duckdb:///" → מפענחים את הנתיב.
   - אחרת, ננסה לפתוח אותו ישירות.

2. אם לא עבר --sql-url:
   - ננסה לקרוא את config.json (בשורש הפרויקט).
   - משם ננסה:
       * paths.duckdb_cache_path
       * data.sql_store.url אם הוא duckdb:///...

3. אם גם זה לא:
   - משתנה סביבה PAIRS_TRADING_CACHE_DB.

4. אם אין גם זה:
   - LOCALAPPDATA/pairs_trading_system/cache.duckdb.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd

LOGGER_NAME = "BuildDqPairs"
logger = logging.getLogger(LOGGER_NAME)


# ========= Project / config helpers =========


def _project_root() -> Path:
    """Assume scripts/ is directly under project root."""
    return Path(__file__).resolve().parent.parent


def _load_config(config_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load config.json if exists, otherwise return None."""
    if config_path is None:
        config_path = _project_root() / "config.json"

    if not config_path.exists():
        logger.warning("config.json not found at: %s", config_path)
        return None

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = json.load(f)
        logger.info("Loaded config from: %s", config_path)
        return cfg
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load config.json: %r", exc)
        return None


def _db_path_from_config(cfg: Dict[str, Any]) -> Optional[Path]:
    """
    Try to infer DuckDB path from config.

    Priority:
    1. cfg["paths"]["duckdb_cache_path"]
    2. cfg["data"]["sql_store"]["url"] if it starts with duckdb:///...
    """
    try:
        paths = cfg.get("paths") or {}
        cache_path = paths.get("duckdb_cache_path")
        if cache_path:
            return Path(cache_path).expanduser().resolve()
    except Exception:
        pass

    try:
        data_cfg = cfg.get("data") or {}
        sql_store = data_cfg.get("sql_store") or {}
        url = sql_store.get("url") or ""
        prefix = "duckdb:///"
        if url.startswith(prefix):
            return Path(url[len(prefix) :]).expanduser().resolve()
    except Exception:
        pass

    return None


def _resolve_db_path(sql_url: Optional[str], cfg: Optional[Dict[str, Any]]) -> Path:
    """
    Resolve DB path using:
    1. --sql-url if provided.
    2. config.json (paths.duckdb_cache_path or data.sql_store.url).
    3. PAIRS_TRADING_CACHE_DB.
    4. LOCALAPPDATA/pairs_trading_system/cache.duckdb.
    """
    # 1) Explicit sql_url from CLI
    if sql_url:
        if sql_url.startswith("duckdb:///"):
            path_str = sql_url[len("duckdb:///") :]
            p = Path(path_str).expanduser().resolve()
            logger.info("Using DB path from --sql-url (duckdb:///): %s", p)
            return p
        else:
            # non-standard: we try to treat it as a file path
            p = Path(sql_url).expanduser().resolve()
            logger.info("Using DB path from --sql-url (file path): %s", p)
            return p

    # 2) From config.json
    if cfg is not None:
        cfg_path = _db_path_from_config(cfg)
        if cfg_path is not None:
            logger.info("Using DB path from config.json: %s", cfg_path)
            return cfg_path

    # 3) Environment variable
    env_path = os.getenv("PAIRS_TRADING_CACHE_DB")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        logger.info("Using DB path from PAIRS_TRADING_CACHE_DB: %s", p)
        return p

    # 4) LOCALAPPDATA default
    local_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    p = (local_dir / "pairs_trading_system" / "cache.duckdb").resolve()
    logger.info("Using default LOCALAPPDATA DB path: %s", p)
    return p


# ========= Pairs extraction helpers =========

_PAIR_KEY_CANDIDATES: Sequence[Tuple[str, str]] = (
    ("sym_x", "sym_y"),
    ("sym1", "sym2"),
    ("x", "y"),
    ("lhs", "rhs"),
    ("asset_x", "asset_y"),
    ("ticker_x", "ticker_y"),
    ("base", "quote"),
)


@dataclass
class PairRecord:
    sym_x: str
    sym_y: str
    source: str
    raw_score: Optional[float] = None


def _normalize_pair_from_dict(d: Dict[str, Any], source: str) -> Optional[PairRecord]:
    # Try to find a key-pair for the two symbols
    lowered = {str(k).lower().strip(): v for k, v in d.items()}

    sym_x_val: Optional[str] = None
    sym_y_val: Optional[str] = None
    for kx, ky in _PAIR_KEY_CANDIDATES:
        if kx in lowered and ky in lowered:
            sym_x_val = lowered[kx]
            sym_y_val = lowered[ky]
            break

    if sym_x_val is None or sym_y_val is None:
        return None

    sym_x = str(sym_x_val).strip()
    sym_y = str(sym_y_val).strip()
    if not sym_x or not sym_y:
        return None

    # Try to capture any score-like field
    score_keys = ("score", "quality", "quality_score", "edge")
    raw_score = None
    for sk in score_keys:
        if sk in lowered:
            try:
                raw_score = float(lowered[sk])
            except Exception:
                raw_score = None
            break

    return PairRecord(sym_x=sym_x, sym_y=sym_y, source=source, raw_score=raw_score)


def _normalize_pair_from_seq(seq: Sequence[Any], source: str) -> Optional[PairRecord]:
    if len(seq) < 2:
        return None
    sym_x = str(seq[0]).strip()
    sym_y = str(seq[1]).strip()
    if not sym_x or not sym_y:
        return None
    return PairRecord(sym_x=sym_x, sym_y=sym_y, source=source)


def _normalize_pair_from_str(s: str, source: str) -> Optional[PairRecord]:
    s = s.strip()
    if not s:
        return None
    for sep in ("-", "/", ":"):
        if sep in s:
            left, right = s.split(sep, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return PairRecord(sym_x=left, sym_y=right, source=source)
    return None


def _iter_candidate_lists_from_config(cfg: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    """
    Generates (name, value) pairs for lists that might represent pairs universe.
    Priority names:
        dq_pairs, pairs, ranked_pairs, pairs_universe
    """
    preferred_keys = ("dq_pairs", "pairs", "ranked_pairs", "pairs_universe")

    for key in preferred_keys:
        if key in cfg:
            yield key, cfg[key]

    # Fallback: scan all keys for any list that looks like pairs
    for key, value in cfg.items():
        if isinstance(value, list) and key not in preferred_keys:
            yield key, value


def _extract_pairs_from_config(cfg: Dict[str, Any]) -> List[PairRecord]:
    """
    Try to extract pairs from config.json using multiple heuristics.
    """
    pairs: List[PairRecord] = []

    for name, val in _iter_candidate_lists_from_config(cfg):
        if not isinstance(val, list) or not val:
            continue

        logger.info("[Config] Trying list '%s' with %d elements as pairs universe", name, len(val))
        local_pairs: List[PairRecord] = []

        for item in val:
            rec: Optional[PairRecord] = None
            if isinstance(item, dict):
                rec = _normalize_pair_from_dict(item, source=f"config:{name}")
            elif isinstance(item, (list, tuple)):
                rec = _normalize_pair_from_seq(item, source=f"config:{name}")
            elif isinstance(item, str):
                rec = _normalize_pair_from_str(item, source=f"config:{name}")

            if rec is not None:
                local_pairs.append(rec)

        unique_local = {(p.sym_x, p.sym_y) for p in local_pairs}
        logger.info(
            "[Config] Parsed %d pairs from field '%s' (after duplicates: %d).",
            len(local_pairs),
            name,
            len(unique_local),
        )
        pairs.extend(local_pairs)

        # אם מצאנו משהו משמעותי (>0) בשדה "עדיף" – אפשר לעצור כאן.
        if name in ("dq_pairs", "pairs", "ranked_pairs", "pairs_universe") and local_pairs:
            break

    # Deduplicate global
    unique: Dict[Tuple[str, str], PairRecord] = {}
    for p in pairs:
        key = (p.sym_x, p.sym_y)
        if key not in unique:
            unique[key] = p

    return list(unique.values())


def _load_pairs_from_csv(cfg: Optional[Dict[str, Any]], project_root: Path) -> List[PairRecord]:
    """
    Try to load pairs universe from CSV pointed by config["pairs_file"].

    Supported formats:
    ------------------
    1. Columns (any of the following pairs):
       - sym_x, sym_y
       - sym1, sym2
       - x, y
       - lhs, rhs
       - asset_x, asset_y
       - ticker_x, ticker_y
       - base, quote
    2. Or a single 'pair' column with strings like:
       - "XLY-XLP"
       - "SPY/QQQ"
       - "BTC:IBIT"
    """
    if cfg is None:
        logger.info("No config loaded, cannot infer pairs_file.")
        return []

    pairs_file = cfg.get("pairs_file")
    if not pairs_file:
        logger.info("config.json has no 'pairs_file' key; skipping CSV source.")
        return []

    candidates = [
        Path(pairs_file),
        project_root / pairs_file,
        project_root / "data" / pairs_file,
    ]

    resolved: Optional[Path] = None
    for c in candidates:
        if c.exists():
            resolved = c
            break

    if resolved is None:
        logger.warning(
            "pairs_file='%s' not found in any common location (root or data/).",
            pairs_file,
        )
        return []

    logger.info("Loading pairs universe from CSV: %s", resolved)
    try:
        df = pd.read_csv(resolved)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read pairs CSV '%s': %r", resolved, exc)
        return []

    if df.empty:
        logger.warning("CSV '%s' is empty; no pairs inside.", resolved)
        return []

    cols = {str(c).lower().strip(): c for c in df.columns}
    csv_pairs: List[PairRecord] = []

    # Case 1: two-column variants
    used_two_cols = False
    for lx, ly in _PAIR_KEY_CANDIDATES:
        if lx in cols and ly in cols:
            cx, cy = cols[lx], cols[ly]
            logger.info("Parsing CSV using columns '%s', '%s'.", cx, cy)
            for _, row in df.iterrows():
                sym_x = str(row[cx]).strip()
                sym_y = str(row[cy]).strip()
                if not sym_x or not sym_y:
                    continue

                score_val: Optional[float] = None
                for sk in ("score", "quality", "quality_score", "edge"):
                    if sk in cols:
                        try:
                            score_val = float(row[cols[sk]])
                        except Exception:
                            score_val = None
                        break

                csv_pairs.append(
                    PairRecord(
                        sym_x=sym_x,
                        sym_y=sym_y,
                        source="csv",
                        raw_score=score_val,
                    )
                )
            used_two_cols = True
            break

    # Case 2: single 'pair' column
    if not used_two_cols and "pair" in cols:
        c_pair = cols["pair"]
        logger.info("Parsing CSV using single 'pair' column: %s", c_pair)
        for _, row in df.iterrows():
            raw = str(row[c_pair]).strip()
            if not raw:
                continue
            rec = _normalize_pair_from_str(raw, source="csv")
            if rec is not None:
                csv_pairs.append(rec)

    # Deduplicate
    unique: Dict[Tuple[str, str], PairRecord] = {}
    for p in csv_pairs:
        key = (p.sym_x, p.sym_y)
        if key not in unique:
            unique[key] = p

    logger.info(
        "Loaded %d unique pairs from CSV '%s'.",
        len(unique),
        resolved,
    )
    return list(unique.values())


# ========= Build dq_pairs into DuckDB =========


def _build_dq_pairs_table(
    db_path: Path,
    pairs: List[PairRecord],
    append: bool = False,
) -> None:
    """
    Create / replace dq_pairs table in DuckDB cache from the given pairs list.
    """
    if not pairs:
        raise ValueError("No pairs provided to build dq_pairs table.")

    logger.info("Connecting to DuckDB at: %s", db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    try:
        if not append:
            logger.info("Dropping existing dq_pairs table (if any).")
            conn.execute("DROP TABLE IF EXISTS dq_pairs")

        logger.info("Creating dq_pairs table.")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dq_pairs (
                sym_x TEXT,
                sym_y TEXT,
                source TEXT,
                raw_score DOUBLE,
                active BOOLEAN,
                created_ts_utc TIMESTAMP
            )
            """
        )

        now_utc = datetime.now(timezone.utc)

        records = [
            (
                p.sym_x,
                p.sym_y,
                p.source,
                p.raw_score,
                True,
                now_utc,
            )
            for p in pairs
        ]

        logger.info("Inserting %d pairs into dq_pairs.", len(records))
        conn.executemany(
            """
            INSERT INTO dq_pairs (
                sym_x,
                sym_y,
                source,
                raw_score,
                active,
                created_ts_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )

        # quick sanity check
        cnt = conn.execute("SELECT COUNT(*) FROM dq_pairs").fetchone()[0]
        logger.info("dq_pairs now contains %d rows.", cnt)
    finally:
        conn.close()
        logger.info("Closed DuckDB connection.")


# ========= CLI =========


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build dq_pairs universe into DuckDB cache from config.json / CSV."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json (default: project_root/config.json).",
    )
    parser.add_argument(
        "--sql-url",
        type=str,
        default=None,
        help=(
            "Optional DuckDB URL or file path. "
            "If starts with 'duckdb:///', the path after it is used as file path."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing dq_pairs instead of dropping it.",
    )

    args = parser.parse_args()

    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    logger.info("===== Build dq_pairs universe started =====")

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    cfg = _load_config(config_path)

    db_path = _resolve_db_path(args.sql_url, cfg)
    logger.info("Using DuckDB path: %s", db_path)

    project_root = _project_root()

    # 1) Try from config
    pairs_from_config = _extract_pairs_from_config(cfg) if cfg is not None else []
    logger.info("Found %d unique pairs from config.", len(pairs_from_config))

    # 2) If needed, try from CSV
    pairs_from_csv: List[PairRecord] = []
    if not pairs_from_config:
        pairs_from_csv = _load_pairs_from_csv(cfg, project_root)
        logger.info("Found %d unique pairs from CSV.", len(pairs_from_csv))

    all_pairs = pairs_from_config or pairs_from_csv

    if not all_pairs:
        msg = (
            "Could not find any pairs universe. "
            "Expected non-empty list under one of: dq_pairs / pairs / ranked_pairs / pairs_universe, "
            "or a valid pairs_file CSV. Please update config.json or CSV and try again."
        )
        logger.error(msg)
        raise SystemExit(1)

    # Deduplicate globally once more
    unique: Dict[Tuple[str, str], PairRecord] = {}
    for p in all_pairs:
        key = (p.sym_x, p.sym_y)
        if key not in unique:
            unique[key] = p

    final_pairs = list(unique.values())
    logger.info("Final unique pairs count: %d", len(final_pairs))

    _build_dq_pairs_table(db_path=db_path, pairs=final_pairs, append=args.append)

    logger.info("===== Build dq_pairs universe finished successfully =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
