# -*- coding: utf-8 -*-
"""
core/refactor_streamlit_query_params.py — Update st.experimental_get_query_params → st.query_params
==================================================================================================

Streamlit הכריז על deprecation של st.experimental_get_query_params.
ה-API החדש הוא st.query_params (אטריביוט, לא פונקציה).

המטרה כאן:
-----------
להמיר בצורה סטרוקטורלית (libcst) רק שימושים מהצורה:

    x = st.experimental_get_query_params()
    foo(st.experimental_get_query_params())

ל:

    x = st.query_params
    foo(st.query_params)

בלי לגעת בקוד אחר ובלי לעשות replace טקסטואלי.

שימוש:
------
    cd <project_root>
    python core/refactor_streamlit_query_params.py          # DRY-RUN
    python core/refactor_streamlit_query_params.py --apply  # ליישם בפועל
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set

import libcst as cst


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_NAMES: Set[str] = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
}


@dataclass
class RefactorStats:
    files_touched: int = 0
    calls_seen: int = 0
    calls_rewritten: int = 0


class QueryParamsTransformer(cst.CSTTransformer):
    """
    ממיר st.experimental_get_query_params() → st.query_params
    """

    def __init__(self, stats: RefactorStats) -> None:
        super().__init__()
        self.stats = stats
        self.changed_this_file: bool = False

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        func = updated_node.func

        # מחפשים st.experimental_get_query_params(...)
        if isinstance(func, cst.Attribute):
            value = func.value
            attr = func.attr
            if (
                isinstance(value, cst.Name)
                and value.value in ("st", "streamlit")
                and isinstance(attr, cst.Name)
                and attr.value == "experimental_get_query_params"
            ):
                self.stats.calls_seen += 1

                # מחליפים את הקריאה כולה ב-Attribute: st.query_params
                replacement = cst.Attribute(
                    value=value,
                    attr=cst.Name("query_params"),
                )
                self.changed_this_file = True
                self.stats.calls_rewritten += 1
                return replacement

        return updated_node


def iter_python_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        parts = Path(dirpath).relative_to(root).parts
        if any(p in SKIP_DIR_NAMES for p in parts):
            dirnames[:] = []
            continue
        for fname in filenames:
            if fname.endswith(".py"):
                yield Path(dirpath) / fname


def process_file(path: Path, *, dry_run: bool, stats: RefactorStats, verbose: bool) -> None:
    code = path.read_text(encoding="utf-8")
    try:
        module = cst.parse_module(code)
    except Exception:
        if verbose:
            print(f"[SKIP syntax] {path}")
        return

    transformer = QueryParamsTransformer(stats=stats)
    new_module = module.visit(transformer)

    if not transformer.changed_this_file:
        if verbose:
            print(f"[NO CHANGE] {path}")
        return

    if dry_run:
        print(f"[DRY] Would update: {path}")
    else:
        path.write_text(new_module.code, encoding="utf-8")
        print(f"[UPDATED] {path}")
    stats.files_touched += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refactor st.experimental_get_query_params() → st.query_params",
    )
    p.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="שורש הפרויקט (ברירת מחדל: הורה של core/).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="לכתוב את השינויים לקבצים (ברירת מחדל: DRY-RUN בלבד).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="להדפיס גם קבצים שלא שונו.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    dry_run = not args.apply

    stats = RefactorStats()

    print(f"Project root : {root}")
    print(f"Dry run      : {dry_run}")
    print(f"Skip dirs    : {sorted(SKIP_DIR_NAMES)}")
    print()

    for path in iter_python_files(root):
        process_file(path=path, dry_run=dry_run, stats=stats, verbose=args.verbose)

    print("\n=== Refactor summary ===")
    print(f"Files touched        : {stats.files_touched}")
    print(f"Calls seen           : {stats.calls_seen}")
    print(f"Calls rewritten      : {stats.calls_rewritten}")
    if dry_run:
        print("\n* DRY-RUN בלבד — לא נכתב שום קובץ.")
        print("  כדי ליישם בפועל, הרץ שוב עם --apply.")


if __name__ == "__main__":
    main()
