# -*- coding: utf-8 -*-
from __future__ import annotations

r"""
root/run_zoom_campaign.py — Zoom Campaign CLI (HF-grade)
=======================================================

תפקיד:
-------
סקריפט מקצועי להרצת Zoom Campaign על זוג אחד או יותר, מעל
api_zoom_campaign_for_pair מ-root.optimization_tab.

מאפיינים:
----------
- תמיכה במספר זוגות (XLY-XLP,QQQ-IWM,...) דרך CLI.
- שליטה ב:
    * n_stages
    * n_trials_per_stage
    * profile (default/defensive/aggressive/...)
    * elite_frac
    * cleanup_strategy (keep_last / keep_all)
    * זמן timeout בדקות
- לוגים ברמת קרן גידור לקובץ logs/zoom_campaign.log.
- שמירת תוצאות:
    * zoom_results_<PAIR>.csv לכל זוג.
    * zoom_campaign_summary.json (meta על כל הזוגות).
- מצב Dry-run: מדפיס מה *היה* רץ בלי להריץ בפועל.

הרצה לדוגמה:
-------------
    python -m root.run_zoom_campaign
    python -m root.run_zoom_campaign --pairs XLY-XLP,QQQ-IWM --stages 4 --trials 80
    python -m root.run_zoom_campaign --pairs XLY-XLP --profile defensive --elite-frac 0.15
    python -m root.run_zoom_campaign --pairs XLY-XLP --dry-run

חשוב:
------
הסקריפט מניח שאתה מריץ אותו משורש הפרויקט
(התיקייה שבה נמצאת תיקיית root/, core/, common/ וכו'):
    C:\Users\omrib\OneDrive\Desktop\pairs_trading_system
"""


import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from root.optimization_tab import api_zoom_campaign_for_pair


# =========================
# Logging configuration
# =========================

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ZoomCampaign")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # project root = תיקייה מעל root/
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "zoom_campaign.log"

    fh = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,   # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    ch = logging.StreamHandler()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("===== ZoomCampaign CLI startup =====")
    logger.info("Project root: %s", project_root)
    logger.info("Logs path: %s", log_path)

    return logger


LOGGER = _setup_logger()


# =========================
# CLI argument parsing
# =========================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive Zoom Campaign optimisation over one or more pairs.",
    )

    parser.add_argument(
        "--pairs",
        type=str,
        default="XLY-XLP",
        help=(
            "Comma-separated list of pairs in SYM1-SYM2 format. "
            "לדוגמה: XLY-XLP,QQQ-IWM,SPY-QQQ"
        ),
    )

    parser.add_argument(
        "--stages",
        type=int,
        default=3,
        help="Number of zoom stages (>=1). כל שלב מריץ Optuna ואז מעדכן טווחים.",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials per stage (n_trials_per_stage).",
    )

    parser.add_argument(
        "--timeout-min",
        type=int,
        default=10,
        help="Timeout in minutes per stage (עובר ל-run_optuna_for_pair).",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Profile name for param ranges (default/defensive/aggressive/...).",
    )

    parser.add_argument(
        "--elite-frac",
        type=float,
        default=0.2,
        help="Fraction of top trials used as 'elite' for zoom (0.05–0.5 טיפוסי).",
    )

    parser.add_argument(
        "--cleanup",
        type=str,
        choices=["keep_last", "keep_all"],
        default="keep_last",
        help=(
            "DuckDB cleanup strategy: "
            "'keep_last' מוחק את כל ה-studies הקודמים ומשאיר רק את האחרון, "
            "'keep_all' לא מוחק כלום."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="zoom_results",
        help="Directory to save CSV/JSON outputs (יחסי לשורש הפרויקט).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run בלבד: מדפיס מה היה רץ (pairs/stages/trials וכו') בלי להריץ אופטימיזציה.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Level=DEBUG ל-logger (יותר פלט למסך/לוג).",
    )

    return parser.parse_args()


# =========================
# Helpers
# =========================

def _resolve_project_root() -> Path:
    """שורש הפרויקט (פשוט: תיקייה מעל root/)."""
    return Path(__file__).resolve().parent.parent


def _parse_pairs_arg(pairs_arg: str) -> List[Tuple[str, str]]:
    """
    הופך string כמו "XLY-XLP,QQQ-IWM" לרשימת טאפלים:
        [("XLY","XLP"), ("QQQ","IWM")]
    מדלג על ערכים לא חוקיים.
    """
    out: List[Tuple[str, str]] = []
    for raw in pairs_arg.split(","):
        raw = raw.strip()
        if not raw:
            continue
        sep = None
        # מחפשים מפריד אפשרי (ברירת מחדל '-')
        for cand in ("-", "/", "|", ":"):
            if cand in raw:
                sep = cand
                break
        if sep is None:
            LOGGER.warning("Skipping invalid pair format (no separator): %r", raw)
            continue
        a, b = raw.split(sep, 1)
        a, b = a.strip(), b.strip()
        if not a or not b:
            LOGGER.warning("Skipping invalid pair (empty side): %r", raw)
            continue
        out.append((a, b))
    return out


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# =========================
# Main run logic
# =========================

def run_zoom_campaign_cli() -> None:
    args = _parse_args()

    # עדכון log level
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.debug("Verbose mode ON (DEBUG level).")

    project_root = _resolve_project_root()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory: %s", out_dir)

    # פירוק pairs
    pairs = _parse_pairs_arg(args.pairs)
    if not pairs:
        LOGGER.error("No valid pairs provided. Use --pairs XLY-XLP,QQQ-IWM,.... Exiting.")
        return

    LOGGER.info(
        "Zoom Campaign config: pairs=%s | stages=%d | trials_per_stage=%d | profile=%s | elite_frac=%.3f | cleanup=%s | timeout_min=%d",
        ",".join([f"{a}-{b}" for a, b in pairs]),
        args.stages,
        args.trials,
        args.profile,
        args.elite_frac,
        args.cleanup,
        args.timeout_min,
    )

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Project root: {project_root}")
        print(f"Output dir  : {out_dir}")
        print(f"Pairs       : {', '.join([f'{a}-{b}' for a, b in pairs])}")
        print(f"Stages      : {args.stages}")
        print(f"Trials/stage: {args.trials}")
        print(f"Profile     : {args.profile}")
        print(f"Elite frac  : {args.elite_frac}")
        print(f"Cleanup     : {args.cleanup}")
        print(f"Timeout (m) : {args.timeout_min}")
        print("\nDry-run בלבד — לא מריץ api_zoom_campaign_for_pair.")
        return

    # תוצאות לכל זוג
    all_meta: Dict[str, Any] = {
        "pairs": [],
        "summary": {},
    }

    for sym1, sym2 in pairs:
        pair_label = f"{sym1}-{sym2}"
        LOGGER.info("===== ZoomCampaign START for %s =====", pair_label)

        try:
            df_zoom, meta = api_zoom_campaign_for_pair(
                sym1,
                sym2,
                n_stages=args.stages,
                n_trials_per_stage=args.trials,
                timeout_min=args.timeout_min,
                profile=args.profile,
                elite_frac=args.elite_frac,
                cleanup_strategy=args.cleanup,
                # כרגע dsr_min / wf_min = None; אפשר להקשיח בעתיד
                dsr_min=None,
                wf_min=None,
            )
        except Exception as e:
            LOGGER.exception("ZoomCampaign failed for %s: %s", pair_label, e)
            # נמשיך לשאר הזוגות, אבל נרשום שגיאה במטא
            all_meta["summary"][pair_label] = {
                "status": "error",
                "error_message": str(e),
            }
            continue

        # שמירת CSV תוצאות
        try:
            csv_path = out_dir / f"zoom_results_{pair_label.replace('/', '_')}.csv"
            df_zoom.to_csv(csv_path, index=False)
            LOGGER.info(
                "Saved zoom results for %s to %s (rows=%d)",
                pair_label,
                csv_path,
                len(df_zoom),
            )
        except Exception as e:
            LOGGER.warning("Failed to save CSV for %s: %s", pair_label, e)

        # הדפסת סיכום קצר למסך
        final_stage = meta.get("final_stage")
        final_best_score = _safe_float(meta.get("final_best_score"))
        status = meta.get("status", "unknown")
        print("\n----------------------------------------")
        print(f"PAIR          : {pair_label}")
        print(f"Status        : {status}")
        print(f"Final stage   : {final_stage}")
        print(f"Best Score    : {final_best_score}")
        print(f"Duration (sec): {meta.get('duration_sec')}")
        print("----------------------------------------\n")

        # שמירה ב-all_meta
        all_meta["pairs"].append(pair_label)
        all_meta["summary"][pair_label] = meta

        LOGGER.info(
            "ZoomCampaign DONE for %s | status=%s | final_stage=%s | final_best_score=%s",
            pair_label,
            status,
            final_stage,
            f"{final_best_score:.4f}" if final_best_score is not None else "n/a",
        )

    # שמירת summary JSON לכל הריצה
    try:
        summary_path = out_dir / "zoom_campaign_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_meta, f, ensure_ascii=False, indent=2)
        LOGGER.info("Saved zoom campaign summary JSON to %s", summary_path)
    except Exception as e:
        LOGGER.warning("Failed to save zoom_campaign_summary.json: %s", e)

    LOGGER.info("===== ZoomCampaign CLI finished =====")


def main() -> None:
    run_zoom_campaign_cli()


if __name__ == "__main__":
    main()
