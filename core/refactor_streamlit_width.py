# -*- coding: utf-8 -*-
"""
core/refactor_streamlit_width.py — Update Streamlit use_container_width → width (HF-grade)
=========================================================================================

מטרה:
------
להחליף בצורה סטרוקטורלית (libcst) את keyword-ים use_container_width= בפונקציות של Streamlit:

- use_container_width=True  → width="stretch"
- use_container_width=False → width="content"

רק בקריאות לפונקציות של st.* (dataframe, plotly_chart, table, וכו'),
בלי לחפש טקסט גולמי ובלי להפיל קוד אחר.

שימוש:
------
    cd <project_root>
    python core/refactor_streamlit_width.py          # DRY-RUN (לא נוגע בקבצים)
    python core/refactor_streamlit_width.py --apply  # כותב את השינויים

אפשר להוסיף --verbose כדי לראות גם קבצים שלא השתנו.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set

import libcst as cst


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ../ -> pairs_trading_system root


@dataclass
class RefactorStats:
    files_touched: int = 0
    calls_seen: int = 0
    args_modified: int = 0


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


class UseContainerWidthTransformer(cst.CSTTransformer):
    """
    Transformer שמעדכן רק קריאות ל-st.XXX(... use_container_width=...).
    """

    def __init__(self, stats: RefactorStats) -> None:
        super().__init__()
        self.stats = stats
        self.changed_this_file: bool = False

    def _is_streamlit_call(self, node: cst.Call) -> bool:
        """
        מזהה st.something(...) או streamlit.something(...).
        לא מחייב רשימת פונקציות ספציפית – כל עוד האובייקט הוא st או streamlit.
        """
        func = node.func
        if isinstance(func, cst.Attribute):
            # st.dataframe(...), streamlit.dataframe(...), וכו'
            value = func.value
            if isinstance(value, cst.Name) and value.value in ("st", "streamlit"):
                return True
        return False

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if not self._is_streamlit_call(updated_node):
            return updated_node

        self.stats.calls_seen += 1

        new_args: List[cst.Arg] = []
        changed = False

        for arg in updated_node.args:
            if arg.keyword is not None and isinstance(arg.keyword, cst.Name):
                kw = arg.keyword.value
                if kw == "use_container_width":
                    # מחפש ערך True/False
                    val = arg.value
                    if isinstance(val, cst.Name):
                        if val.value == "True":
                            # use_container_width=True → width="stretch"
                            arg = cst.Arg(
                                keyword=cst.Name("width"),
                                value=cst.SimpleString('"stretch"'),
                            )
                            changed = True
                            self.stats.args_modified += 1
                        elif val.value == "False":
                            # use_container_width=False → width="content"
                            arg = cst.Arg(
                                keyword=cst.Name("width"),
                                value=cst.SimpleString('"content"'),
                            )
                            changed = True
                            self.stats.args_modified += 1
                        else:
                            # use_container_width=<expression> – נשאיר ונזהיר בלוג (תיאורטית)
                            pass
            new_args.append(arg)

        if changed:
            self.changed_this_file = True
            return updated_node.with_changes(args=new_args)
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


def process_file(
    path: Path,
    *,
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

    transformer = UseContainerWidthTransformer(stats=stats)
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
        description="Refactor Streamlit use_container_width → width='stretch'/'content'.",
    )
    p.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="שורש הפרויקט לחיפוש קבצי Python.",
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
        process_file(
            path=path,
            dry_run=dry_run,
            stats=stats,
            verbose=args.verbose,
        )

    print("\n=== Refactor summary ===")
    print(f"Files touched      : {stats.files_touched}")
    print(f"st.* calls seen    : {stats.calls_seen}")
    print(f"use_container_width updated → width : {stats.args_modified}")
    if dry_run:
        print("\n* DRY-RUN בלבד — לא נכתב שום קובץ.")
        print("  כדי ליישם בפועל, הרץ שוב עם --apply.")


if __name__ == "__main__":
    main()
