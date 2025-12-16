#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
system_upgrader_agent.py — Log-Driven Code Upgrader (HF-Grade)
==============================================================

סוכן רפקטורינג למערכת ה-Pairs Trading שלך:

תפקיד:
-------
- לקרוא לוג (Streamlit output / dashboard_audit.log / כל לוג שתבחר).
- לזהות Patterns של בעיות מוכרות:
    * Deprecation: use_container_width → width="stretch"/"content"
    * ArrowInvalid על DataFrame עם עמודות types מעורבים (בעיקר ב-Config diff)
    * FutureWarning של pandas groupby (observed=False)
    * IBKR connection issues / yfinance errors / מודולים חסרים
- לבצע תיקונים אוטומטיים לקבצי Python (בזהירות, עם גיבוי .bak).
- לתת דוח מסודר: מה תוקן, מה לא תוקן אבל זוהה כבעיה (TODO).

שימוש:
-------
    cd C:\\Users\\omrib\\OneDrive\\Desktop\\pairs_trading_system

    # דוגמא בסיסית (סורק את כל הפרויקט, בלי לוג):
    python system_upgrader_agent.py --repo-root .

    # דוגמא עם לוג (סורק טעויות ואזהרות ומתאים את התיקונים):
    python system_upgrader_agent.py --repo-root . --log-path logs/dashboard_audit.log

    # Dry-run: רק מציג מה *היה* משתנה, בלי לכתוב לקבצים:
    python system_upgrader_agent.py --repo-root . --log-path logs/dashboard_audit.log --dry-run
"""

from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# ----------------------------------------------------------------------
# Data Classes for Issues & Fix Reports
# ----------------------------------------------------------------------


@dataclass
class DetectedIssues:
    """מה זוהה מתוך הלוג/קוד – מה כדאי לנסות לתקן."""
    use_container_width: bool = False
    arrow_invalid_config_df: bool = False
    pandas_groupby_observed: bool = False
    ibkr_connection_issues: bool = False
    yfinance_symbol_errors: bool = False
    risk_parity_missing: bool = False


@dataclass
class FileChange:
    path: Path
    description: str


@dataclass
class FixReport:
    """סיכום פעולת הסוכן."""
    changes: List[FileChange] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add_change(self, path: Path, desc: str) -> None:
        self.changes.append(FileChange(path=path, description=desc))

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def render(self) -> str:
        lines: List[str] = []
        lines.append("=== System Upgrader Agent – Report ===")
        lines.append("")
        if self.changes:
            lines.append("Applied changes:")
            for ch in self.changes:
                rel = ch.path
                lines.append(f"  - {rel}: {ch.description}")
            lines.append("")
        else:
            lines.append("No code changes were applied.")
            lines.append("")

        if self.notes:
            lines.append("Notes / TODOs:")
            for n in self.notes:
                lines.append(f"  - {n}")
            lines.append("")
        else:
            lines.append("No additional notes.")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Utility Helpers
# ----------------------------------------------------------------------


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def write_text_safe(path: Path, content: str, *, dry_run: bool = False) -> None:
    if dry_run:
        return
    path.write_text(content, encoding="utf-8")


def backup_file(path: Path, *, dry_run: bool = False) -> Path:
    """
    יוצר קובץ גיבוי לפני תיקון (אם עוד לא קיים).
    """
    backup = path.with_suffix(path.suffix + ".bak")
    if not dry_run and not backup.exists():
        backup.write_bytes(path.read_bytes())
    return backup


# ----------------------------------------------------------------------
# Log Parsing – Identify Issues from Log File
# ----------------------------------------------------------------------


def parse_log_for_issues(log_path: Optional[Path]) -> DetectedIssues:
    issues = DetectedIssues()
    if not log_path or not log_path.exists():
        return issues

    text = read_text_safe(log_path)

    if "Please replace `use_container_width` with `width`" in text:
        issues.use_container_width = True

    if "ArrowInvalid" in text and "Conversion failed for column default with type object" in text:
        issues.arrow_invalid_config_df = True

    if "FutureWarning" in text and "The default of observed=False is deprecated" in text:
        issues.pandas_groupby_observed = True

    if "ConnectionRefusedError" in text and "Connect call failed" in text:
        issues.ibkr_connection_issues = True

    if "YahooProvider: empty data" in text or "YFPricesMissingError" in text:
        issues.yfinance_symbol_errors = True

    if "risk_parity helper unavailable" in text or "cannot import name 'apply_risk_parity'" in text:
        issues.risk_parity_missing = True

    return issues


# ----------------------------------------------------------------------
# 1) Fix use_container_width Deprecation
# ----------------------------------------------------------------------


def fix_use_container_width_in_text(text: str) -> str:
    """
    מחליף שימושים ב-use_container_width ב-width="stretch"/"content".

    לוגיקה:
    - use_container_width=True  → width="stretch"
    - use_container_width=False → width="content"
    """
    text = text.replace('width="stretch"', 'width="stretch"')
    text = text.replace("use_container_width=False", 'width="content"')
    return text


def fix_use_container_width(repo_root: Path, report: FixReport, dry_run: bool) -> None:
    """
    עובר על כל קבצי ה-Python תחת repo_root ומתקן use_container_width.
    """
    for path in repo_root.rglob("*.py"):
        original = read_text_safe(path)
        upgraded = fix_use_container_width_in_text(original)
        if upgraded != original:
            backup_file(path, dry_run=dry_run)
            write_text_safe(path, upgraded, dry_run=dry_run)
            report.add_change(path, "Replaced use_container_width with width='stretch'/'content'")


# ----------------------------------------------------------------------
# 2) Fix ArrowInvalid in Config diff (dashboard.py)
# ----------------------------------------------------------------------


def fix_config_diff_dataframe_in_dashboard(dashboard_path: Path, report: FixReport, dry_run: bool) -> None:
    """
    מטפל ספציפית בבעיה שנראתה בלוג:

    ArrowInvalid: Could not convert 100000.0 with type float: tried to convert to boolean
    Conversion failed for column default with type object

    זה קורה ב-Developer Tools כשאנחנו יוצרים DataFrame עם עמודות (default/raw/effective)
    שיש בהן טיפוסים מעורבים. כדי שלא נשבור Arrow, נהפוך אותם ל-strings (repr).

    אנחנו מחפשים את הבלוק שבו df_cfg נבנה ומחליפים אותו בגרסה יציבה:
        "default": repr(default_cfg.get(k)), ...
    """
    if not dashboard_path.exists():
        return

    text = read_text_safe(dashboard_path)

    pattern = r"""
        df_cfg\s*=\s*pd\.DataFrame\(
        \s*\[
        \s*\{
        \s*"key":\s*k,\s*
        "default":\s*default_cfg\.get\(k\),\s*
        "raw":\s*raw_cfg\.get\(k\),\s*
        "effective":\s*merged\.get\(k\)
        \s*\}
        .*?
        \]
        \s*\)
    """

    regex = re.compile(pattern, re.DOTALL | re.VERBOSE)
    if not regex.search(text):
        return

    replacement = textwrap.dedent(
        """
        df_cfg = pd.DataFrame(
            [
                {
                    "key": k,
                    "default": repr(default_cfg.get(k)),
                    "raw": repr(raw_cfg.get(k)),
                    "effective": repr(merged.get(k)),
                }
                for k in sorted(set(default_cfg.keys()) | set(raw_cfg.keys()) | set(merged.keys()))
            ]
        )
        """
    ).strip()

    new_text = regex.sub(replacement, text)
    if new_text != text:
        backup_file(dashboard_path, dry_run=dry_run)
        write_text_safe(dashboard_path, new_text, dry_run=dry_run)
        report.add_change(
            dashboard_path,
            "Refactored df_cfg DataFrame to use repr(...) for default/raw/effective (Arrow-safe).",
        )


# ----------------------------------------------------------------------
# 3) Fix pandas FutureWarning (groupby observed=False)
# ----------------------------------------------------------------------


def fix_pandas_groupby_observed(repo_root: Path, report: FixReport, dry_run: bool) -> None:
    """
    מחפש מקומות שבהם יש groupby בלי observed=... ומוסיף observed=False.

    זה פוטנציאלית שביר, אז אנחנו עושים תיקון מאוד ממוקד:
    - רק במקומות שמזכירים .groupby( ... )["..."] עם agg/mean וכו' בקבצים ידועים,
      לדוגמה insights.py.
    """
    for path in repo_root.rglob("insights.py"):
        original = read_text_safe(path)
        text = original

        # דוגמה לתיקון פשוט:
        # df.groupby("some_cat")["x"].mean()  -> df.groupby("some_cat", observed=False)["x"].mean()
        # לא ננסה להיות חכמים מדי – רק groupby("...") → groupby("...", observed=False) אם אין observed.
        pattern = r"groupby\((?P<inner>[^)]*)\)"
        regex = re.compile(pattern)

        def _replace(m: re.Match) -> str:
            inner = m.group("inner")
            if "observed=" in inner:
                return m.group(0)
            return f'groupby({inner}, observed=False)'

        new_text = regex.sub(_replace, text)

        if new_text != original:
            backup_file(path, dry_run=dry_run)
            write_text_safe(path, new_text, dry_run=dry_run)
            report.add_change(
                path,
                "Added observed=False to pandas groupby calls to avoid FutureWarning.",
            )


# ----------------------------------------------------------------------
# 4) Optional: Notes bug auto-fix (session_state conflict pattern)
# ----------------------------------------------------------------------


def fix_notes_session_state_pattern(dashboard_path: Path, report: FixReport, dry_run: bool) -> None:
    """
    מזהה pattern של:

        notes_key = ...
        existing_note = st.session_state.get(notes_key, "")
        note = st.text_area(..., key=notes_key, ...)
        st.session_state[notes_key] = note

    ומחליף אותו לגרסה בטוחה:
        if notes_key not in st.session_state: st.session_state[notes_key] = ""
        note = st.text_area(..., key=notes_key, ...)
        # בלי כתיבה ידנית ל-session_state אחרי זה.

    התיקון מבוסס regex – זה לא מושלם, אבל אמור לעבוד על הבלוק שיצרנו.
    """
    if not dashboard_path.exists():
        return

    text = read_text_safe(dashboard_path)

    pattern = r"""
        notes_key\s*=\s*keygen\("notes",\s*str\(TODAY\)\)\s*\n
        \s*existing_note\s*=\s*st\.session_state\.get\(notes_key,\s*""\)\s*\n
        \s*note\s*=\s*st\.text_area\(
        (?P<args>.*?)
        key=notes_key,
        .*?
        \)\s*\n
        \s*st\.session_state\[notes_key\]\s*=\s*note
    """

    regex = re.compile(pattern, re.DOTALL | re.VERBOSE)
    if not regex.search(text):
        return

    # בונים replacement עדין: נבטל את existing_note + השורה של הכתיבה
    replacement = textwrap.dedent(
        """
        notes_key = keygen("notes", str(TODAY))

        if notes_key not in st.session_state:
            st.session_state[notes_key] = ""

        note = st.text_area(
        \\g<args>key=notes_key,
        height=120,
        )

        # הערך נשמר אוטומטית ב-st.session_state[notes_key] ע"י ה-widget.
        """
    ).strip()

    # בגלל ה-\\g<args> נשתמש ב-sub עם פונקציה
    def _repl(m: re.Match) -> str:
        args = m.group("args")
        rep = (
            'notes_key = keygen("notes", str(TODAY))\n\n'
            "if notes_key not in st.session_state:\n"
            "    st.session_state[notes_key] = \"\"\n\n"
            "note = st.text_area(\n"
            f"{args}key=notes_key,\n"
            "height=120,\n"
            ")\n\n"
            "# הערך נשמר אוטומטית ב-st.session_state[notes_key] ע\"י ה-widget.\n"
        )
        return rep

    new_text = regex.sub(_repl, text)
    if new_text != text:
        backup_file(dashboard_path, dry_run=dry_run)
        write_text_safe(dashboard_path, new_text, dry_run=dry_run)
        report.add_change(
            dashboard_path,
            "Fixed notes session_state pattern (avoids StreamlitAPIException on widget key conflict).",
        )


# ----------------------------------------------------------------------
# 5) High-level run: Decide what to fix, and apply
# ----------------------------------------------------------------------


def run_upgrader(repo_root: Path, log_path: Optional[Path], dry_run: bool) -> FixReport:
    report = FixReport()

    issues = parse_log_for_issues(log_path)
    if log_path and log_path.exists():
        report.add_note(f"Parsed log file: {log_path}")
    else:
        report.add_note("No log file provided or not found – running generic upgrades.")

    dashboard_path = repo_root / "root" / "dashboard.py"
    insights_path = repo_root / "root" / "insights.py"

    # 1) use_container_width → width="stretch"/"content"
    if issues.use_container_width:
        fix_use_container_width(repo_root, report, dry_run)
    else:
        report.add_note("use_container_width deprecation not detected in log – skip automatic replacement (can force with --force-width-fix).")

    # 2) ArrowInvalid DataFrame in dashboard (config diff panel)
    if issues.arrow_invalid_config_df:
        fix_config_diff_dataframe_in_dashboard(dashboard_path, report, dry_run)
    else:
        report.add_note("ArrowInvalid on config DataFrame not detected – skip DataFrame fix.")

    # 3) pandas groupby observed=False
    if issues.pandas_groupby_observed:
        fix_pandas_groupby_observed(repo_root, report, dry_run)
    else:
        report.add_note("pandas FutureWarning (observed=False) not detected – skip groupby patch.")

    # 4) Fix notes session_state pattern (זה פשוט טוב שיהיה תמיד תקין)
    fix_notes_session_state_pattern(dashboard_path, report, dry_run)

    # 5) Issues שלא מתקנים אוטומטית – רק הערות
    if issues.ibkr_connection_issues:
        report.add_note(
            "IBKR connection issues detected (ConnectionRefusedError). "
            "הקוד עצמו בסדר – יש לו fallback ל-Yahoo. "
            "כדי לסדר זאת צריך לוודא ש-TWS/IB Gateway פתוח ו-API Port תואם (7497/7496)."
        )

    if issues.yfinance_symbol_errors:
        report.add_note(
            "Detected yfinance/Yahoo errors (empty data / invalid symbols). "
            "ככל הנראה יש ב-universe סימבולים במבנה מוזר (\"{'SYMBOLS': ...}\"). "
            "מומלץ לנקות את קובץ ה-pairs/config או להרחיב עוד את normalize_symbol_input."
        )

    if issues.risk_parity_missing:
        report.add_note(
            "risk_parity helper unavailable – apply_risk_parity חסר ב-core/risk_parity.py. "
            "אם אתה רוצה שהתכונה תעבוד, צריך להשלים את הפונקציה במודול או לסמן את הפיצ'ר כלא זמין באופטימיזציה."
        )

    return report


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="System Upgrader Agent – Log-Driven Code Upgrader")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="נתיב לשורש הרפו (ברירת מחדל: .)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="",
        help="נתיב לקובץ לוג לניתוח (למשל logs/dashboard_audit.log או stdout של Streamlit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="אם מוגדר – לא שומר שינויים, רק מדווח מה היה משתנה.",
    )

    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    log_path = Path(args.log_path).resolve() if args.log_path else None

    report = run_upgrader(repo_root, log_path, dry_run=args.dry_run)
    print(report.render())


if __name__ == "__main__":
    main()
