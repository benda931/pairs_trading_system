# -*- coding: utf-8 -*-
"""
migrate_utcnow_to_timezone.py — Refactor datetime.utcnow → datetime.now(timezone.utc)
====================================================================================

מה הסקריפט עושה:
----------------
1. סורק את עץ התיקיות (ברירת מחדל: התיקייה שממנה מריצים את הסקריפט).
2. בכל קובץ .py:
   - מחפש מופעים של datetime.now(timezone.utc)(...)
   - מחליף אותם ב-datetime.now(timezone.utc)(...)
     (כלומר פשוט מחליף את 'datetime.now(timezone.utc)(' ב-'datetime.now(timezone.utc)(')
   - מוודא שיש import ל-timezone:
       * אם יש כבר `from datetime import datetime` → הופך ל-`from datetime import datetime, timezone`
       * אם יש `from datetime import ...` בלי timezone → מוסיף timezone לרשימה
       * אם אין בכלל import של timezone → מוסיף `from datetime import timezone` אחרי `import datetime`
         או בתחילת הקובץ אם אין `import datetime`

שימוש:
------
    python tools/migrate_utcnow_to_timezone.py               # על התיקייה הנוכחית
    python tools/migrate_utcnow_to_timezone.py /path/to/project/root

הסקריפט מדפיס אילו קבצים שונו.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Optional, Tuple

import logging

logger = logging.getLogger("utcnow_migrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


UTCNOW_PATTERN = re.compile(r"datetime\.utcnow\s*\(")
FROM_DATETIME_IMPORT_RE = re.compile(
    r"^from\s+datetime\s+import\s+([^\n]+)$",
    re.MULTILINE,
)
IMPORT_DATETIME_RE = re.compile(
    r"^import\s+datetime\b.*$",
    re.MULTILINE,
)


def _ensure_timezone_import(content: str) -> str:
    """
    מוודא שבקובץ יש import ל-timezone:
    1) אם כבר יש 'timezone' בקובץ → לא נוגעים.
    2) אם יש שורה 'from datetime import ...' → מוסיפים timezone לרשימה.
    3) אם יש 'import datetime' → מוסיפים אחרי זה 'from datetime import timezone'.
    4) אחרת → מוסיפים 'from datetime import timezone' בתחילת הקובץ.
    """
    if "timezone" in content:
        # בהנחה שאם timezone מופיע בקובץ, כבר יש import מתאים.
        return content

    # 1) נסה לעדכן "from datetime import ..."
    m = FROM_DATETIME_IMPORT_RE.search(content)
    if m:
        imports_str = m.group(1)
        parts = [p.strip() for p in imports_str.split(",")]
        if "timezone" not in parts:
            parts.append("timezone")
            new_imports_str = ", ".join(sorted(set(parts)))
            new_line = f"from datetime import {new_imports_str}"
            start, end = m.span()
            content = content[:start] + new_line + content[end:]
        return content

    # 2) אם אין "from datetime import ...", אבל יש "import datetime"
    m2 = IMPORT_DATETIME_RE.search(content)
    if m2:
        # נכניס שורה חדשה אחרי import datetime
        line = m2.group(0)
        insert_pos = m2.end()
        insert_text = f"{line}\nfrom datetime import timezone"
        content = content[:m2.start()] + insert_text + content[m2.end():]
        return content

    # 3) אין שום import datetime → נוסיף בראש הקובץ
    return f"from datetime import timezone\n{content}"


def process_file(path: Path) -> bool:
    """
    מעבד קובץ יחיד:
    - מחליף datetime.now(timezone.utc)(...) ב-datetime.now(timezone.utc)(...)
    - דואג ל-import נכון ל-timezone
    מחזיר True אם הקובץ השתנה.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return False

    if "datetime.utcnow" not in text:
        return False

    original_text = text

    # 1) החלפת datetime.now(timezone.utc)( ב-datetime.now(timezone.utc)(
    text = text.replace("datetime.now(timezone.utc)(", "datetime.now(timezone.utc)(")

    # 2) לוודא import ל-timezone
    if "datetime.now(timezone.utc)" in text:
        text = _ensure_timezone_import(text)

    if text != original_text:
        try:
            path.write_text(text, encoding="utf-8")
            logger.info("Updated %s", path)
            return True
        except Exception as exc:
            logger.error("Failed to write %s: %s", path, exc)
            return False

    return False


def iter_python_files(root: Path):
    """
    מחזיר generator של כל קבצי .py תחת root,
    מדלג על תיקיות וירטואל־env נפוצות.
    """
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".idea",
        ".vscode",
    }

    for p in root.rglob("*.py"):
        # דילוג על תיקיות לא רלוונטיות
        if any(part in skip_dirs for part in p.parts):
            continue
        yield p


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        root = Path(argv[1]).resolve()
    else:
        root = Path.cwd().resolve()

    if not root.exists():
        logger.error("Root path does not exist: %s", root)
        return 1

    logger.info("Scanning Python files under: %s", root)

    changed_files = 0
    for py_file in iter_python_files(root):
        if process_file(py_file):
            changed_files += 1

    logger.info("Done. Files updated: %d", changed_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
