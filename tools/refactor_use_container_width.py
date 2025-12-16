# tools/refactor_use_container_width.py
# -*- coding: utf-8 -*-
"""
Refactor use_container_width → width (HF-grade, v2)
===================================================

סקריפט חכם לניקוי כל השימושים ב-use_container_width בקוד Streamlit:

- use_container_width=True  → width="stretch"
- use_container_width=False → width="content"

תכונות:
--------
- Dry-run כברירת מחדל (לא נוגע בקבצים עד שתוסיף --apply).
- מציג סיכום: כמה קבצים עודכנו, כמה החלפות בכל קובץ.
- מדלג על:
    * venv / .venv
    * תיקיות שמתחילות ב-"."
- לא מחליף בתוך שורות שהן תגובה בלבד.
- מאפשר:
    * --include path1 path2  → טיפול רק בתיקיות מסוימות.
    * --exclude path1 path2  → הוספת תיקיות לדילוג.
    * --backup               → יצירת קובץ .bak לפני כתיבה.
    * --ext .py .pyi         → בחירת סיומות לניתוח (ברירת מחדל: .py בלבד).

הרצה טיפוסית:
--------------
    cd /path/to/pairs_trading_system

    # לראות מה הוא *היה* משנה (בלי לגעת בקבצים)
    python tools/refactor_use_container_width.py

    # לבצע בפועל + ליצור גיבוי .bak
    python tools/refactor_use_container_width.py --apply --backup
"""

from __future__ import annotations

import argparse
import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# קבצים שלא נרצה לגעת בהם (סקריפטים/אייג'נטים שעוסקים בעצמם ב-use_container_width)
SKIP_EXACT = {
    (PROJECT_ROOT / "core" / "refactor_streamlit_width.py").resolve(),
    (PROJECT_ROOT / "root" / "system_upgrader_agent.py").resolve(),
    (PROJECT_ROOT / "tools" / "refactor_use_container_width.py").resolve(),
}
# תבניות להחלפה
RE_TRUE = re.compile(r"(?<!\.)use_container_width\s*=\s*True")
RE_FALSE = re.compile(r"(?<!\.)use_container_width\s*=\s*False")
RE_ANY = re.compile(r"(?<!\.)use_container_width\s*=")



@dataclass
class FileRefactorResult:
    path: Path
    replacements_true: int = 0
    replacements_false: int = 0
    total_before: int = 0
    total_after: int = 0
    changed: bool = False


def should_skip_dir(path: Path, extra_excludes: Sequence[Path]) -> bool:
    """החלטה אם לדלג על תיקייה."""
    parts = {p.lower() for p in path.parts}
    if "venv" in parts or ".venv" in parts:
        return True
    if any(part.startswith(".") for part in path.parts):
        return True
    for ex in extra_excludes:
        try:
            if path.is_relative_to(ex):
                return True
        except AttributeError:
            # Python<3.9 fallback
            if str(path).startswith(str(ex)):
                return True
    return False


def collect_files(
    root: Path,
    exts: Sequence[str],
    includes: Sequence[Path],
    excludes: Sequence[Path],
) -> List[Path]:
    """איסוף כל הקבצים הרלוונטיים (לפי include/exclude והסיומות)."""
    files: List[Path] = []

    # אם אין includes – נעבוד על כל הפרויקט
    roots = includes or [root]

    norm_exts = {e if e.startswith(".") else f".{e}" for e in exts}

    for base in roots:
        base = base.resolve()
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if should_skip_dir(p.parent, excludes):
                continue
            if p.suffix not in norm_exts:
                continue
            files.append(p)

    return sorted(files)


def refactor_text(text: str) -> tuple[str, int, int, int, int]:
    """
    מחליף use_container_width בקוד טקסטואלי.

    מחזיר:
        (new_text, replacements_true, replacements_false, total_before, total_after)
    """
    lines = text.splitlines(keepends=True)

    replacements_true = 0
    replacements_false = 0
    total_before = 0
    total_after = 0

    new_lines: List[str] = []

    for line in lines:
        # נספור כמה instances יש בשורה לפני
        before_matches = RE_ANY.findall(line)
        total_before += len(before_matches)

        stripped = line.lstrip()
        # אם זו שורת תגובה מלאה (# ...) – לא ניגע בה
        if stripped.startswith("#"):
            new_lines.append(line)
            continue

        # מחליפים True / False
        line, count_true = RE_TRUE.subn('width="stretch"', line)
        line, count_false = RE_FALSE.subn('width="content"', line)

        replacements_true += count_true
        replacements_false += count_false

        after_matches = RE_ANY.findall(line)
        total_after += len(after_matches)

        new_lines.append(line)

    new_text = "".join(new_lines)
    return new_text, replacements_true, replacements_false, total_before, total_after


def show_diff(original: str, modified: str, rel_path: Path) -> None:
    """מדפיס diff קטן לקריאה אנושית (לפי קובץ)."""
    diff = difflib.unified_diff(
        original.splitlines(),
        modified.splitlines(),
        fromfile=f"{rel_path} (before)",
        tofile=f"{rel_path} (after)",
        lineterm="",
    )
    printed = False
    for line in diff:
        if not printed:
            print("---------- diff ----------")
            printed = True
        print(line)
    if printed:
        print("--------------------------\n")


def refactor_file(path: Path, apply: bool, backup: bool) -> FileRefactorResult:
    """Refactor יחיד לקובץ אחד."""
    text = path.read_text(encoding="utf-8")
    new_text, c_true, c_false, total_before, total_after = refactor_text(text)

    changed = new_text != text

    result = FileRefactorResult(
        path=path,
        replacements_true=c_true,
        replacements_false=c_false,
        total_before=total_before,
        total_after=total_after,
        changed=changed,
    )

    if not changed:
        return result

    rel = path.relative_to(PROJECT_ROOT)
    print(f"[CHANGED] {rel}")
    print(f"  - True  → stretch : {c_true}")
    print(f"  - False → content : {c_false}")

    # השאריות (אם total_after>0) – נציין אותן
    if total_after > 0:
        print(f"  ! WARNING: still {total_after} occurrences of 'use_container_width=' in this file")

    # diff קטן
    show_diff(text, new_text, rel)

    if apply:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            backup_path.write_text(text, encoding="utf-8")
            print(f"  Backup written → {backup_path.relative_to(PROJECT_ROOT)}")
        path.write_text(new_text, encoding="utf-8")
        print("  ✅ File updated on disk.\n")
    else:
        print("  (dry-run: no changes written)\n")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refactor use_container_width → width (stretch/content) across the project."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="לכתוב את השינויים לקבצים בפועל (ברירת מחדל: dry-run בלבד).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="ליצור קובץ .bak לפני כתיבה (רלוונטי רק עם --apply).",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="תיקיות לכלול (יחסי ל-Project root). אם לא מצוין – כל הפרויקט.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="תיקיות לדילוג (יחסי ל-Project root). כברירת מחדל מדלג על venv/.venv/.*",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=[".py"],
        help="סיומות קבצים לטיפול (ברירת מחדל: .py). למשל: --ext .py .pyi",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Mode        : {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Extensions  : {args.ext or ['.py']}\n")

    # עיבוד include/exclude ל-Path
    includes = [PROJECT_ROOT / p for p in args.include] if args.include else []
    excludes = [(PROJECT_ROOT / p).resolve() for p in args.exclude]

    files = collect_files(PROJECT_ROOT, args.ext, includes, excludes)
    print(f"Found {len(files)} file(s) to scan.\n")

    results: List[FileRefactorResult] = []
    for path in files:
        # דילוג על קבצים מיוחדים
        if path.resolve() in SKIP_EXACT:
            print(f"[SKIP]    {path.relative_to(PROJECT_ROOT)} (special refactor/agent/tool file)")
            continue

        res = refactor_file(path, apply=args.apply, backup=args.backup)
        results.append(res)


    # סיכום
    total_changed = sum(1 for r in results if r.changed)
    total_true = sum(r.replacements_true for r in results)
    total_false = sum(r.replacements_false for r in results)
    total_before = sum(r.total_before for r in results)
    total_after = sum(r.total_after for r in results)

    print("========== SUMMARY ==========")
    print(f"Files scanned         : {len(files)}")
    print(f"Files changed         : {total_changed}")
    print(f"Replacements True→str : {total_true}")
    print(f"Replacements False→ct : {total_false}")
    print(f"Occurrences before    : {total_before}")
    print(f"Occurrences after     : {total_after}")
    if not args.apply:
        print("\nNOTE: Dry-run only. Add --apply to actually write changes.")
    print("================================")


if __name__ == "__main__":
    main()
