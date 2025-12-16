# -*- coding: utf-8 -*-
"""
scripts/export_zoom_best_params_for_dq_pairs.py
===============================================

Export best zoom-campaign parameters for all pairs in dq_pairs.

Pipeline:
---------
1. Load pairs from DuckDB table dq_pairs (sym_x, sym_y).
2. For each pair, try to load one or more Optuna studies that belong to the
   zoom-campaign (e.g. stages 1..3).
3. For each study, extract:
   - best_value (objective)
   - best_trial params
   - n_trials, direction, study_name
4. Build a consolidated DataFrame:
   - sym_x, sym_y, stage, study_name, best_value, n_trials, is_finished, ...
     + all params as columns.
5. Save to:
   - zoom_best_params.csv
   - (optional) write DuckDB table zoom_best_params.

NOTE:
-----
Because I don't see your exact optimization_tab.py code here,
this script is intentionally flexible:

- You can provide:
    --studies-dir      (directory where Optuna DB / SQLite lives)
    --storage-url      (e.g. sqlite:///studies/optuna_zoom.db)
    --study-prefix     (e.g. "zoom")
    --stages           (e.g. 3 -> stage1, stage2, stage3)
- We assume study_name pattern like:
    f"{study_prefix}:{pair_key}:stage{stage}"
  where pair_key could be "IBIT-BITO" (exactly as in zoom-campaign),
  but you can adjust pattern logic in _build_study_name if needed.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import optuna
import pandas as pd

logger = logging.getLogger("ZoomExport")


# =========================
# Config dataclass
# =========================


@dataclass
class ExportConfig:
    duckdb_path: Path
    dq_table: str = "dq_pairs"

    # Optuna storage
    studies_dir: Path = Path("studies")
    storage_url: Optional[str] = None  # if None -> infer from studies_dir

    study_prefix: str = "zoom"  # prefix used by zoom-campaign
    n_stages: int = 3          # how many zoom stages to look for

    # Output
    output_csv: Path = Path("zoom_best_params.csv")
    output_duckdb_table: Optional[str] = None  # e.g. "zoom_best_params"


# =========================
# Helpers
# =========================


def _default_duckdb_path() -> Path:
    """
    Default path for cache.duckdb, matching your current system layout.
    """
    import os

    env_path = os.getenv("PAIRS_TRADING_CACHE_DB")
    if env_path:
        return Path(env_path).expanduser().resolve()

    local_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return (local_dir / "pairs_trading_system" / "cache.duckdb").resolve()


def _infer_storage_url(studies_dir: Path) -> str:
    """
    Try to infer a reasonable Optuna storage URL from studies_dir.

    Common pattern:
        studies/optuna_zoom.db -> sqlite:///studies/optuna_zoom.db

    If you already know your exact URL from optimization_tab.py,
    pass it explicitly via --storage-url.
    """
    # This is a heuristic – feel free to adjust.
    # We look for a single *.db file inside studies_dir.
    candidates = list(studies_dir.glob("*.db"))
    if not candidates:
        # fallback: assume sqlite file named optuna.db in studies_dir
        db_path = studies_dir / "optuna.db"
    else:
        # pick first db file found
        db_path = candidates[0]

    return f"sqlite:///{db_path.as_posix()}"


def _build_pair_key(sym_x: str, sym_y: str) -> str:
    """
    Build the key used in study_name for a given pair.

    zoom-campaign is called with:
        --pair SYM1-SYM2

    So by default we assume pair_key = "SYM1-SYM2".
    If your optimization_tab uses a different naming, adjust here.
    """
    return f"{sym_x}-{sym_y}"


def _build_study_name(prefix: str, pair_key: str, stage: int) -> str:
    """
    Build Optuna study_name for a stage of zoom-campaign.

    Example pattern:
        "zoom:IBIT-BITO:stage1"
    """
    return f"{prefix}:{pair_key}:stage{stage}"


def _connect_duckdb(path: Path) -> duckdb.DuckDBPyConnection:
    logger.info("[ZoomExport] Connecting to DuckDB: %s", path)
    return duckdb.connect(str(path))


def _load_pairs(conn: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    """
    Load sym_x, sym_y universe from dq_pairs.
    """
    sql = f"SELECT sym_x, sym_y FROM {table} ORDER BY sym_x, sym_y"
    df = conn.execute(sql).fetchdf()
    if df.empty:
        raise RuntimeError(f"{table} is empty — nothing to export.")
    df["sym_x"] = df["sym_x"].astype(str).str.strip()
    df["sym_y"] = df["sym_y"].astype(str).str.strip()
    logger.info("[ZoomExport] Loaded %d pairs from %s.", len(df), table)
    return df


# =========================
# Core logic: Export best params
# =========================


def _extract_best_trial_dict(study: optuna.Study) -> Dict[str, object]:
    """
    Flatten best_trial params + some study metadata into a dict.
    """
    bt = study.best_trial
    out: Dict[str, object] = {}

    out["study_name"] = study.study_name
    out["direction"] = study.direction.name
    out["n_trials"] = len(study.trials)
    out["best_value"] = bt.value
    out["best_trial_number"] = bt.number
    out["best_state"] = bt.state.name

    # params
    for k, v in bt.params.items():
        out[f"param_{k}"] = v

    # user attrs (if any)
    for k, v in bt.user_attrs.items():
        out[f"attr_{k}"] = v

    return out


def export_zoom_best_params(cfg: ExportConfig) -> pd.DataFrame:
    """
    Main export pipeline:

    1. Load pairs from dq_pairs.
    2. For each pair and each stage, attempt to load Optuna study.
    3. If found and has trials, record best_trial info + params.
    4. Build DataFrame and save to CSV (+ optional DuckDB table).
    """
    studies_dir = cfg.studies_dir.resolve()
    studies_dir.mkdir(parents=True, exist_ok=True)

    storage_url = cfg.storage_url or _infer_storage_url(studies_dir)
    logger.info("[ZoomExport] Using Optuna storage: %s", storage_url)

    # 1) Load pairs from dq_pairs
    conn = _connect_duckdb(cfg.duckdb_path)
    try:
        pairs_df = _load_pairs(conn, cfg.dq_table)
    finally:
        conn.close()
        logger.info("[ZoomExport] Closed DuckDB connection after loading pairs.")

    rows: List[Dict[str, object]] = []
    n_pairs = len(pairs_df)
    n_found = 0

    for idx, row in pairs_df.iterrows():
        sym_x = str(row["sym_x"])
        sym_y = str(row["sym_y"])
        pair_key = _build_pair_key(sym_x, sym_y)

        logger.info(
            "[ZoomExport] (%d/%d) Processing pair %s-%s ...",
            idx + 1,
            n_pairs,
            sym_x,
            sym_y,
        )

        for stage in range(1, cfg.n_stages + 1):
            study_name = _build_study_name(cfg.study_prefix, pair_key, stage)

            try:
                study = optuna.load_study(study_name=study_name, storage=storage_url)
            except KeyError:
                # Study not found in this storage
                logger.debug(
                    "[ZoomExport] Study not found: %s (pair=%s, stage=%d)",
                    study_name,
                    pair_key,
                    stage,
                )
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[ZoomExport] Failed to load study '%s': %r",
                    study_name,
                    exc,
                )
                continue

            if not study.trials:
                logger.debug(
                    "[ZoomExport] Study '%s' has no trials; skipping.", study_name
                )
                continue

            best = _extract_best_trial_dict(study)
            best["sym_x"] = sym_x
            best["sym_y"] = sym_y
            best["pair_key"] = pair_key
            best["zoom_stage"] = stage

            rows.append(best)
            n_found += 1

    if not rows:
        logger.warning("[ZoomExport] No studies/trials found for any pair.")
        df = pd.DataFrame(columns=["sym_x", "sym_y", "pair_key", "zoom_stage"])
    else:
        df = pd.DataFrame(rows)
        # sort by pair + stage + best_value (descending)
        df = df.sort_values(
            ["sym_x", "sym_y", "zoom_stage", "best_value"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

    # Save to CSV
    out_path = cfg.output_csv.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("[ZoomExport] Saved zoom best params CSV to: %s", out_path)

    # Optional: write DuckDB table
    if cfg.output_duckdb_table:
        conn = _connect_duckdb(cfg.duckdb_path)
        try:
            conn.register("df_zoom", df)
            tmp = f"_{cfg.output_duckdb_table}_tmp"
            conn.execute(f"DROP TABLE IF EXISTS {tmp}")
            conn.execute(f"CREATE TABLE {tmp} AS SELECT * FROM df_zoom")
            conn.execute(f"DROP TABLE IF EXISTS {cfg.output_duckdb_table}")
            conn.execute(f"ALTER TABLE {tmp} RENAME TO {cfg.output_duckdb_table}")
            logger.info(
                "[ZoomExport] Wrote DuckDB table '%s' into %s",
                cfg.output_duckdb_table,
                cfg.duckdb_path,
            )
        finally:
            conn.close()
            logger.info("[ZoomExport] Closed DuckDB connection after write.")

    logger.info(
        "[ZoomExport] DONE. total_pairs=%d, studies_found=%d, rows=%d",
        n_pairs,
        n_found,
        len(df),
    )
    return df


# =========================
# CLI
# =========================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export best zoom-campaign params for all pairs in dq_pairs."
    )
    parser.add_argument(
        "--duckdb-path",
        type=str,
        default=None,
        help="Path to DuckDB cache (default: LOCALAPPDATA\\pairs_trading_system\\cache.duckdb or PAIRS_TRADING_CACHE_DB).",
    )
    parser.add_argument(
        "--dq-table",
        type=str,
        default="dq_pairs",
        help="Table name for pairs universe (default: dq_pairs).",
    )
    parser.add_argument(
        "--studies-dir",
        type=str,
        default="studies",
        help="Directory where Optuna DB files live (default: studies).",
    )
    parser.add_argument(
        "--storage-url",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///studies/optuna_zoom.db). "
             "If omitted, will try to infer from studies-dir.",
    )
    parser.add_argument(
        "--study-prefix",
        type=str,
        default="zoom",
        help="Study prefix used by zoom-campaign (default: zoom).",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=3,
        help="How many zoom stages to look for (default: 3).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="zoom_best_params.csv",
        help="Output CSV path (default: zoom_best_params.csv).",
    )
    parser.add_argument(
        "--output-duckdb-table",
        type=str,
        default="",
        help="If non-empty, write results into this DuckDB table as well (e.g. zoom_best_params).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if args.duckdb_path:
        duckdb_path = Path(args.duckdb_path).expanduser().resolve()
    else:
        duckdb_path = _default_duckdb_path()

    cfg = ExportConfig(
        duckdb_path=duckdb_path,
        dq_table=args.dq_table,
        studies_dir=Path(args.studies_dir),
        storage_url=args.storage_url,
        study_prefix=args.study_prefix,
        n_stages=args.stages,
        output_csv=Path(args.output_csv),
        output_duckdb_table=args.output_duckdb_table or None,
    )

    logger.info("===== Zoom Best Params Export started =====")
    logger.info("DuckDB: %s | dq_table: %s", cfg.duckdb_path, cfg.dq_table)
    logger.info(
        "studies_dir: %s | storage_url: %s | study_prefix: %s | stages: %d",
        cfg.studies_dir,
        cfg.storage_url or "<auto>",
        cfg.study_prefix,
        cfg.n_stages,
    )

    try:
        export_zoom_best_params(cfg)
        logger.info("===== Zoom Best Params Export finished successfully =====")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Zoom best-params export failed: %r", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
