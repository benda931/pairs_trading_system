# -*- coding: utf-8 -*-
"""
core/refactor_load_price_data.py — Structural refactor tool for load_price_data (HF-grade)
==========================================================================================

מטרה:
------
לעדכן בצורה בטוחה *רק* קריאות לפונקציה load_price_data בכל הפרויקט:

1. המרת keyword-ים ישנים:
   - start=    -> start_date=
   - end=      -> end_date=

2. אופציונלי: המרת קריאות positional:
   - load_price_data(sym, start_date, end_date)
     → load_price_data(symbol=sym, start_date=start_date, end_date=end_date) (לקריאות ברורות)

הכל נעשה בצורה סטרוקטורלית עם libcst:
- לא מחפשים "start=" בטקסט גולמי.
- נוגעים *רק* בקריאות לפונקציה load_price_data (Name או Attribute).
- שומרים על כל הפורמט, הערות, רווחים וכו'.

שימוש בסיסי:
-------------
    cd <project_root>
    python core/refactor_load_price_data.py --dry-run
    python core/refactor_load_price_data.py --apply

פלגים עיקריים:
---------------
    --dry-run        (ברירת מחדל) — מדפיס אילו קבצים היו משתנים.
    --apply          — באמת כותב את השינויים לקבצים.
    --include common core root    — להגביל לתיקיות ספציפיות.
    --exclude-dirs .venv .git ... — לדלג על תיקיות מסוימות.
    --no-positional  — לא לגעת בקריאות positional (רק start/end → start_date/end_date).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import libcst as cst


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ../ -> שורש הפרויקט (pairs_trading_system)


# ======================================================================
# Stats / bookkeeping
# ======================================================================


@dataclass
class RefactorStats:
    files_touched: int = 0
    calls_seen: int = 0
    calls_modified_keywords: int = 0
    calls_modified_positional: int = 0


# ======================================================================
# CST Transformer
# ======================================================================


class LoadPriceDataTransformer(cst.CSTTransformer):
    """
    Transformer לריפקטור של load_price_data:

    - מתמקד רק בקריאות לפונקציה load_price_data.
    - keywords:
        start=    → start_date=
        end=      → end_date=
    - positional (אופציונלי):
        load_price_data(sym, start_dt, end_dt)
            → load_price_data(sym, start_date=start_dt, end_date=end_dt)
      (רק אם אין כבר keywords, ורק אם יש 2–3 ארגומנטים אחרי הסימבול).
    """

    def __init__(self, *, transform_positional: bool, stats: RefactorStats) -> None:
        super().__init__()
        self.transform_positional = transform_positional
        self.stats = stats

    # Helper: לזהות שהפונקציה נקראת load_price_data (Name או Attribute.attr)
    def _is_load_price_data_call(self, node: cst.Call) -> bool:
        func_name: Optional[str] = None

        if isinstance(node.func, cst.Name):
            func_name = node.func.value
        elif isinstance(node.func, cst.Attribute):
            attr = node.func.attr
            if isinstance(attr, cst.Name):
                func_name = attr.value

        return func_name == "load_price_data"

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if not self._is_load_price_data_call(updated_node):
            return updated_node

        self.stats.calls_seen += 1

        args = list(updated_node.args)
        changed_keywords = False
        changed_positional = False

        # 1) keyword refactor: start= -> start_date=, end= -> end_date=
        new_args: List[cst.Arg] = []
        for arg in args:
            if arg.keyword is not None and isinstance(arg.keyword, cst.Name):
                kw = arg.keyword.value
                if kw == "start":
                    arg = arg.with_changes(keyword=cst.Name("start_date"))
                    changed_keywords = True
                elif kw == "end":
                    arg = arg.with_changes(keyword=cst.Name("end_date"))
                    changed_keywords = True
            new_args.append(arg)

        args = new_args

        # 2) positional -> keyword (opt-in)
        if self.transform_positional:
            # תנאים:
            # - אין keywords בכלל (או שכבר טיפלנו ב-"start"/"end")
            # - יש לפחות 3 ארגומנטים
            # - הארגומנט הראשון הוא הסימבול (נשאיר positional)
            # - השני והשלישי לא ממותגים → נוסיף להם keyword לפי מיקום
            has_keywords = any(arg.keyword is not None for arg in args)
            if not has_keywords and len(args) >= 3:
                # args[0] -> symbol (נשאיר)
                new_args_pos: List[cst.Arg] = [args[0]]

                # second -> start_date, third -> end_date
                second = args[1]
                third = args[2]

                if second.keyword is None:
                    second = second.with_changes(keyword=cst.Name("start_date"))
                    changed_positional = True
                if third.keyword is None:
                    third = third.with_changes(keyword=cst.Name("end_date"))
                    changed_positional = True

                new_args_pos.append(second)
                new_args_pos.append(third)

                # שאר הארגומנטים (אם יש) נשארים כפי שהם
                if len(args) > 3:
                    new_args_pos.extend(args[3:])

                args = new_args_pos

        if changed_keywords:
            self.stats.calls_modified_keywords += 1
        if changed_positional:
            self.stats.calls_modified_positional += 1

        if changed_keywords or changed_positional:
            return updated_node.with_changes(args=args)

        return updated_node


# ======================================================================
# File processing
# ======================================================================


SKIP_DIR_NAMES: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}


def iter_python_files(
    root: Path,
    include_dirs: Optional[Iterable[str]] = None,
    exclude_dirs: Optional[Iterable[str]] = None,
) -> Iterable[Path]:
    """
    הולך על כל קבצי ה-.py מתחת ל-root, עם אפשרות include/exclude.

    include_dirs: שמות תיקיות יחסיות (למשל ["common","core","root"]) – אם None → הכול.
    exclude_dirs: שמות תיקיות שיש לדלג עליהן (בנוסף ל-SKIP_DIR_NAMES).
    """
    include_set: Optional[Set[str]] = set(include_dirs) if include_dirs else None
    exclude_set: Set[str] = set(exclude_dirs or [])

    for dirpath, dirnames, filenames in os.walk(root):
        # סינון תיקיות
        parts = Path(dirpath).relative_to(root).parts
        # אם יש חלק בתיקיה שנמצא ב-SKIP_DIR_NAMES או exclude_set → דילוג על התיקיה
        if any(p in SKIP_DIR_NAMES or p in exclude_set for p in parts):
            dirnames[:] = []  # לא לרדת פנימה
            continue

        if include_set is not None:
            # אם אף חלק מהנתיב לא ברשימת include → דלג
            if not any(p in include_set for p in parts):
                continue

        for fname in filenames:
            if fname.endswith(".py"):
                yield Path(dirpath) / fname


def process_file(
    path: Path,
    *,
    transform_positional: bool,
    dry_run: bool,
    stats: RefactorStats,
    verbose: bool,
) -> None:
    code = path.read_text(encoding="utf-8")
    try:
        module = cst.parse_module(code)
    except Exception:
        if verbose:
            print(f"[SKIP syntax] {path}")
        return

    transformer = LoadPriceDataTransformer(
        transform_positional=transform_positional,
        stats=stats,
    )
    new_module = module.visit(transformer)

    if new_module.code != code:
        if dry_run:
            print(f"[DRY] Would update: {path}")
        else:
            path.write_text(new_module.code, encoding="utf-8")
            print(f"[UPDATED] {path}")
        stats.files_touched += 1
    else:
        if verbose:
            print(f"[NO CHANGE] {path}")


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Structural refactor for load_price_data calls (start/end → start_date/end_date).",
    )
    p.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="שורש הפרויקט לחיפוש קבצי Python (ברירת מחדל: הורה של core/).",
    )
    p.add_argument(
        "--include",
        nargs="*",
        default=["common", "core", "root"],
        help="שמות תיקיות (יחסית ל-root) לכלול בסריקה. ריק = הכל.",
    )
    p.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=[],
        help="שמות תיקיות להחריג (בנוסף ל-.git, .venv וכו').",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="לכתוב את השינויים לקבצים (ברירת מחדל: dry-run בלבד).",
    )
    p.add_argument(
        "--no-positional",
        action="store_true",
        help="לא לגעת בקריאות positional (רק start/end → start_date/end_date).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="להדפיס מידע גם על קבצים שלא השתנו / דולגים.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    include_dirs = args.include or None
    exclude_dirs = args.exclude_dirs or []

    dry_run = not args.apply
    transform_positional = not args.no_positional

    stats = RefactorStats()

    print(f"Project root  : {root}")
    print(f"Include dirs  : {include_dirs if include_dirs else 'ALL'}")
    print(f"Exclude dirs  : {exclude_dirs} + {sorted(SKIP_DIR_NAMES)}")
    print(f"Dry run       : {dry_run}")
    print(f"Positional -> kw: {transform_positional}")
    print()

    for path in iter_python_files(root, include_dirs=include_dirs, exclude_dirs=exclude_dirs):
        process_file(
            path,
            transform_positional=transform_positional,
            dry_run=dry_run,
            stats=stats,
            verbose=args.verbose,
        )

    print("\n=== Refactor summary ===")
    print(f"Files touched            : {stats.files_touched}")
    print(f"load_price_data calls    : {stats.calls_seen}")
    print(f"  keyword-updated calls  : {stats.calls_modified_keywords}")
    print(f"  positional-updated     : {stats.calls_modified_positional}")
    if dry_run:
        print("\n* זה היה DRY-RUN בלבד — לא נכתב שום קובץ.")
        print("  כדי ליישם בפועל, הרץ שוב עם --apply.")


if __name__ == "__main__":
    main()
