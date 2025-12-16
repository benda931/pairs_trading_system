#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hedge_fund_upgrade_agent.py
===========================

סוכן GPT לשדרוג מערכת pairs-trading לרמת קרן גידור, כולל:

- סריקה מלאה של כל התקייה (קבצי Python)
- בניית תוכנית שדרוג (phases) + טאסקים לכל קובץ
- שדרוג קבצים בודדים או כל הפרויקט בעזרת OpenAI Responses API
- גיבויים אוטומטיים לכל קובץ לפני כתיבה
- Snapshot דטרמיניסטי של הפרויקט לפני שדרוגים (אופציונלי)
- הרצת health_check_full_system.py בין פייזים (אופציונלי)

שימוש בסיסי:

    # 0. התקנת חבילה והגדרת מפתח
    pip install openai
    export OPENAI_API_KEY="sk-..."

    # 1. יצירת תוכנית שדרוג + טאסקים לכל קובץ
    python hedge_fund_upgrade_agent.py plan --root .

    # 2. שדרוג קובץ ספציפי לפי הטאסקים מהתוכנית
    python hedge_fund_upgrade_agent.py upgrade-file \
        --root . \
        --file core/sql_store.py \
        --apply --snapshot

    # 3. שדרוג כל הקבצים לפי התוכנית (בהתחלה בלי --apply, כדי לראות)
    python hedge_fund_upgrade_agent.py upgrade-all --root .
    python hedge_fund_upgrade_agent.py upgrade-all --root . --apply --snapshot --health-check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import textwrap
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI  # pip install openai


# לטעון את קובץ .env כדי שה-OPENAI_API_KEY ייכנס ל-Environment
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")  # 👈 כאן הוא קורא את .env שלך

# תיקיית snapshots לשדרוגי קוד
SNAPSHOT_DIR = PROJECT_ROOT / "upgrade_snapshots"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Make sure you have a .env file with 'OPENAI_API_KEY=sk-...' "
        "in the project root or set the environment variable."
    )

# ======================================================================
# קונפיגורציה כללית
# ======================================================================

# מודל ברירת מחדל – אפשר לשנות ל-gpt-5, gpt-4.1 וכן הלאה
DEFAULT_MODEL = os.getenv("HEDGE_FUND_UPGRADER_MODEL", "gpt-5.1")
# רמת reasoning למודלים תומכים (gpt-5 / gpt-5.1)
DEFAULT_REASONING = os.getenv("HEDGE_FUND_UPGRADER_REASONING", "medium")  # none/low/medium/high

# קובץ פרומט מאסטר (אם יש לך כבר "פורמט פרומט.txt")
DEFAULT_MASTER_PROMPT = Path("פרומטים") / "פורמט פרומט.txt"

# קובץ התוכנית (JSON) + גרסת Markdown אנושית
DEFAULT_PLAN_JSON = "hedge_fund_upgrade_plan.json"
DEFAULT_PLAN_MD = "hedge_fund_upgrade_plan.md"

# תיקיות להתעלם מהן בסריקה
IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".eggs",
}

# ======================================================================
# מודל נתונים
# ======================================================================


@dataclass
class FileInfo:
    rel_path: str
    size_bytes: int
    n_lines: int
    header: str


@dataclass
class RepoSnapshot:
    root: Path
    files: List[FileInfo]


@dataclass
class FilePlan:
    """
    תיאור ה-"תפקיד" של קובץ במערכת + טאסקים קשורים.
    """

    rel_path: str
    role: str
    priority: str  # high / medium / low
    categories: List[str]
    tasks: List[str]
    phase_ids: List[str]


@dataclass
class Phase:
    id: str
    name: str
    order: int
    description: str
    milestones: List[str]


@dataclass
class UpgradePlan:
    project_root: str
    phases: List[Phase]
    files: List[FilePlan]


# ======================================================================
# Static phase & per-file metadata (deterministic, hand-curated)
# ======================================================================

STATIC_PHASES: Dict[str, Dict[str, Any]] = {
    "phase_architecture_env": {
        "id": "phase_architecture_env",
        "name": "Architecture & Environment Modes",
        "order": 1,
        "description": "Introduce explicit environment modes (dev/research/paper/live) and centralize upgrade orchestration.",
        "milestones": [
            "Define global upgrade phases and stable ids",
            "Make upgrade commands deterministic and idempotent across runs",
        ],
    },
    "phase_observability_ops": {
        "id": "phase_observability_ops",
        "name": "Observability, Operations & Code Quality",
        "order": 6,
        "description": "Add observability, health checks, and safety rails around automated upgrades.",
        "milestones": [
            "Introduce health checks after major upgrade phases",
            "Add snapshotting of code/configs prior to applying upgrades",
        ],
    },
}

STATIC_FILE_TASKS: Dict[str, Dict[str, Any]] = {
    "hedge_fund_upgrade_agent.py": {
        "role": "Central coordinator for stepwise, phase-based upgrades of the codebase toward hedge-fund-grade standards.",
        "priority": "medium",
        "categories": ["infra", "other"],
        "phase_ids": ["phase_architecture_env", "phase_observability_ops"],
        "tasks": [
            "Model upgrade phases and per-file tasks as explicit data structures tied to the global phase ids.",
            "Add deterministic execution order and idempotency checks so rerunning the agent does not corrupt files.",
            "Introduce dry-run and snapshot modes that back up modified files and configs before applying changes.",
            "Integrate with health_check_full_system.py to validate the system after each major upgrade phase.",
        ],
    },
}

# ======================================================================
# Utilities
# ======================================================================


def debug(msg: str) -> None:
    print(f"[hedge-fund-upgrader] {msg}")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def backup_file(path: Path) -> Path:
    """
    יוצר גיבוי לקובץ לפני שכתבת עליו.
    example: core/sql_store.py -> core/sql_store.py.gpt_backup
    """
    backup = path.with_suffix(path.suffix + ".gpt_backup")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def create_project_snapshot(root: Path, label: str) -> Path:
    """
    Create a deterministic zip snapshot of relevant project files (code + configs).

    The snapshot name is derived from the project directory name and the provided label.
    If a snapshot with the same name already exists, it is reused (idempotent).
    """
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label.strip()) or "snapshot"
    archive_path = SNAPSHOT_DIR / f"{root.name}_{safe_label}.zip"
    if archive_path.is_file():
        debug(f"Snapshot already exists, reusing: {archive_path}")
        return archive_path

    files_to_include: List[Path] = []
    for p in root.rglob("*"):
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".toml", ".ini"}:
            files_to_include.append(p)

    files_to_include.sort(key=lambda p: str(p.relative_to(root)))

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_include:
            zf.write(file_path, arcname=str(file_path.relative_to(root)))

    debug(f"Created snapshot archive: {archive_path}")
    return archive_path


def extract_header_from_source(source: str) -> str:
    """
    מחלץ שורה "מתארת" לקובץ: docstring מודולרי או הערה/שורה ראשונה.
    """
    import ast

    header = ""
    try:
        mod = ast.parse(source)
        doc = ast.get_docstring(mod)
        if doc:
            header = doc.strip().splitlines()[0].strip()
    except Exception:
        header = ""

    if not header:
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                stripped = stripped.lstrip("#").strip()
            header = stripped
            break

    return header[:200]


def iter_python_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.py"):
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        files.append(p)
    files.sort()
    return files


def scan_repo(root: Path) -> RepoSnapshot:
    root = root.resolve()
    file_infos: List[FileInfo] = []
    for p in iter_python_files(root):
        try:
            src = p.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = str(p.relative_to(root))
        header = extract_header_from_source(src)
        size_bytes = len(src.encode("utf-8"))
        n_lines = src.count("\n") + 1
        file_infos.append(
            FileInfo(
                rel_path=rel,
                size_bytes=size_bytes,
                n_lines=n_lines,
                header=header,
            )
        )
    return RepoSnapshot(root=root, files=file_infos)


def build_repo_summary_text(snapshot: RepoSnapshot, max_files: int = 250) -> str:
    lines: List[str] = []
    lines.append(f"Project root: {snapshot.root}")
    lines.append("")
    lines.append("Python files overview (trimmed):")
    lines.append("")

    for i, info in enumerate(snapshot.files):
        prefix = f"{i+1:03d}. {info.rel_path}  (lines={info.n_lines}, bytes={info.size_bytes})"
        if info.header:
            prefix += f"  —  {info.header}"
        lines.append(prefix)
        if i + 1 >= max_files:
            remaining = len(snapshot.files) - max_files
            if remaining > 0:
                lines.append(f"... ({remaining} more files omitted from summary)")
            break

    return "\n".join(lines)


def extract_code_block_from_markdown(text: str) -> Optional[str]:
    """
    מחפש את הבלוק הראשון של ```python ...``` או ``` ... ``` ומחזיר את הקוד.
    פחות רגיש לבעיות בפורמט (ואפילו מסתדר אם אין סוגר ``` בסוף).
    """
    # קודם ננסה למצוא ```python
    start = text.find("```python")
    if start == -1:
        # אם אין, ננסה כל ``` ראשון
        start = text.find("```")
    if start == -1:
        return None

    # נזוז לשורה שאחרי ה-```...`
    newline_after = text.find("\n", start)
    if newline_after == -1:
        return None

    # נחפש את ה-``` הסוגר אחרי תחילת הקוד
    end = text.find("```", newline_after)
    if end == -1:
        # אם אין סוגר, ניקח עד סוף הטקסט
        code = text[newline_after:].strip()
    else:
        code = text[newline_after:end].strip()

    return code or None


def apply_unified_diff(original: str, diff_text: str) -> str:
    """
    מקבל תוכן קובץ מקורי + unified diff (כמו מה שהמודל מחזיר ב-```diff)
    ומחזיר את התוכן החדש אחרי החלת הדיפ.

    מניח diff עבור קובץ אחד, בפורמט @@ -l,s +l2,s2 @@ ...
    """
    orig_lines = original.splitlines(keepends=True)
    diff_lines = diff_text.splitlines(keepends=False)

    result: List[str] = []
    i = 0  # אינדקס בשורות המקור
    idx = 0  # אינדקס בשורות הדיפ

    def parse_hunk_header(line: str) -> Optional[tuple[int, int, int, int]]:
        # דוגמה: @@ -12,5 +12,7 @@ ...
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if not m:
            return None
        l1 = int(m.group(1))
        s1 = int(m.group(2) or "1")
        l2 = int(m.group(3))
        s2 = int(m.group(4) or "1")
        return l1, s1, l2, s2

    while idx < len(diff_lines):
        line = diff_lines[idx]
        # דילוג על headerים כלליים
        if line.startswith("diff ") or line.startswith("index "):
            idx += 1
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            idx += 1
            continue
        if line.startswith("@@"):
            hunk = parse_hunk_header(line)
            if not hunk:
                idx += 1
                continue
            old_start, old_count, new_start, new_count = hunk

            # להוסיף את כל השורות שלא שייכות ל-hunk הזה (unchanged prefix)
            target_index = old_start - 1  # 1-based -> 0-based
            while i < target_index and i < len(orig_lines):
                result.append(orig_lines[i])
                i += 1

            # לעבד את ה-hunk
            idx += 1
            while idx < len(diff_lines) and not diff_lines[idx].startswith("@@"):
                l = diff_lines[idx]
                if not l:
                    idx += 1
                    continue
                tag = l[0]
                content = l[1:]
                if tag == " ":
                    # שורה ללא שינוי – נלקחת מהמקור
                    if i < len(orig_lines):
                        result.append(orig_lines[i])
                        i += 1
                    else:
                        # fallback – אין מספיק שורות מקור
                        result.append(content + "\n")
                elif tag == "-":
                    # שורה שנמחקה – מדלגים על שורת המקור
                    i += 1
                elif tag == "+":
                    # שורה חדשה – מוסיפים ל-result
                    result.append(content + "\n")
                idx += 1
            continue

        # אם לא בתוך hunk ולא header – מתקדמים
        idx += 1

    # להוסיף כל מה שנשאר בסוף המקור
    while i < len(orig_lines):
        result.append(orig_lines[i])
        i += 1

    return "".join(result)


# ======================================================================
# OpenAI client wrapper
# ======================================================================

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def call_gpt_raw(
    *,
    model: str,
    instructions: str,
    user_input: str,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: int = 8192,
) -> str:
    """
    קריאה בסיסית ל-Responses API, מחזירה טקסט חופשי (output_text).
    """
    client = get_client()
    kwargs: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    resp = client.responses.create(**kwargs)
    # ספריית Python מספקת output_text שמחבר את כל קטעי הטקסט יחד.
    return resp.output_text  # type: ignore[attr-defined]


def call_gpt_json(
    *,
    model: str,
    instructions: str,
    user_input: str,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: int = 8192,
) -> Dict[str, Any]:
    """
    מבקש מהמודל JSON בלבד ומנסה לפענח ל-dict.
    אם JSON לא תקין – זורק חריגה עם התשובה הגולמית לקבלת debug.
    """
    text = call_gpt_raw(
        model=model,
        instructions=instructions,
        user_input=user_input,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    text_stripped = text.strip()

    # לפעמים המודל מחזיר JSON עטוף ב-```json ...```
    if text_stripped.startswith("```"):
        m = re.search(r"```json(.*?)(```)", text_stripped, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            m = re.search(r"```(.*?)(```)", text_stripped, flags=re.DOTALL)
        if m:
            text_stripped = m.group(1).strip()

    try:
        return json.loads(text_stripped)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse JSON from model. Error: {exc}\n\nRaw output:\n{text_stripped[:4000]}"
        )


def load_master_prompt(path: Path) -> str:
    """
    טוען קובץ פרומט מאסטר (system/developer message).
    אם לא קיים – יחזיר הנחיות ברירת מחדל שמתמקדות בשדרוג לרמת קרן גידור.
    """
    if path.is_file():
        return load_text(path)

    default = textwrap.dedent(
        """
        You are my senior quant + platform architect for a pairs-trading system
        that must be upgraded to **hedge-fund / live-trading quality**.

        Always optimize for:
        - Capital safety and risk controls
        - Deterministic behavior and reproducibility
        - Clean architecture (separation: data ingest / research / execution / web UI / infra)
        - Type hints, docstrings, logging, and error handling
        - Minimal but high-impact diff (avoid rewriting everything at once)
        """
    ).strip()
    return default


# ======================================================================
# Brief: מה זה "רמת קרן גידור" מבחינת המודל
# ======================================================================

HEDGE_FUND_BRIEF = """
Goal: Upgrade this pairs-trading project into a **hedge-fund-grade, live-trading Web application**
with maximum **system-wide utilization** — a single, coherent engine from data → research →
decisions → execution → monitoring, with minimal dead code and fragmented flows.

Global principles:
- Capital safety first (risk-aware by construction).
- Deterministic, reproducible behaviour across research/backtest/paper/live.
- One shared infrastructure (AppContext, SqlStore, risk engine, broker router, dashboard/agents).
- Incremental but compounding improvements: small, safe steps that move the whole system forward.

Key dimensions:

1) Web / App Layer & UX
   - Provide a single coherent user entry point (dashboard or web app) for all workflows:
     research, optimization, monitoring, live operations, and agents.
   - Separate research UI from live trading & monitoring screens while reusing the same core
     services underneath.
   - Make it trivial to switch between environments (dev / research / paper / live) via
     explicit profiles, not ad-hoc flags.
   - Keep the UI "fund-grade": clear states, no fake/dummy data in live contexts,
     and obvious indicators of environment and risk status.

2) Data Layer & Backfills
   - Robust ingestion from brokers and/or market data vendors (e.g. IBKR, Yahoo, etc.).
   - Centralized storage (SQL/Parquet/DuckDB/SQLite/Postgres) with explicit schemas,
     indices, and retention policies.
   - Clear policies for:
       * history length,
       * intraday vs end-of-day handling,
       * gaps detection, anomaly logging, and data quality scoring.
   - One "source of truth" for prices, signals, risk state, and experiment metadata
     so that backtests, research, and live trading all see consistent data.

3) Signal & Portfolio Layer
   - Clean separation between:
       * signal generation (fair values, spreads, z-scores, regimes),
       * portfolio construction and sizing,
       * risk overlays (stops, max leverage, exposure constraints, kill-switches).
   - Same parameters and risk rules shared between backtest and live, with environment-aware
     overrides only where explicitly justified.
   - Signals and portfolios are **persisted and explainable** (metadata, configs, git_rev)
     so that any live decision can be traced back to data, parameters, and code version.

4) Execution / Broker Layer
   - Single broker abstraction with:
       * configuration for paper vs live,
       * retry/backoff, idempotent order submission, and circuit breakers,
       * safe order construction (sizes, limits, slippage controls).
   - Clear distinction between:
       * desired target position,
       * current fills, slippage, rejected orders, and pending state.
   - Execution logic is separated from signal logic; signals express *what* to do,
     execution modules express *how* to do it safely.

5) Observability, Telemetry & Operations
   - Structured logs per component (ingest, signals, optimization, execution, web, agents).
   - Simple health endpoints or checks for all critical services (SqlStore, broker, data, risk engine).
   - System-level health views: latency, data freshness, error rates, and risk status.
   - Self-diagnostics for:
       * failing data sources,
       * misaligned configs or environments,
       * stuck positions or unbalanced portfolios.
   - Make it easy to plug this into alerting/monitoring (e.g. dashboards, Slack/Email/SMS hooks).

6) Code Quality & Safety
   - Strong type hints, well named modules, minimal global state, and consistent project layout.
   - Avoid duplication and dead code; prefer refactoring to shared utilities over copy-paste.
   - Mark TODOs clearly and avoid half-finished experimental code in production-critical paths.
   - Keep changes as small and localized as possible while moving toward the target architecture,
     but do not hesitate to perform coherent refactors when they clearly reduce complexity.
   - Dependencies are explicit and minimal; new third-party libraries are added only when
     they clearly improve robustness or capability.

7) System-wide Cohesion & Utilization
   - Treat the project as **one coherent engine**, not a collection of isolated scripts.
   - Make sure modules plug into shared infrastructure (AppContext, SqlStore, risk engine,
     broker router, dashboard/agents) instead of re-inventing ad-hoc flows.
   - Prefer improvements that increase **effective system utilization**:
       * more of the codebase is actually used in live/research pipelines,
       * less dead code and fewer fragmented entry points,
       * fewer one-off CLI tools that bypass the central infrastructure.
   - When editing a file, always ask:
       * "How does this change help the rest of the system?"
       * "Can this logic be expressed via existing shared components instead of new custom paths?"

8) Configuration, Environments & Governance
   - Centralize configuration (env/profile/settings) via a small number of well-defined models,
     not scattered globals and ad-hoc env var reads.
   - Make environment (dev/research/paper/live) **explicit and inspectable** everywhere:
     logs, dashboards, SqlStore labels, and context snapshots.
   - Track configuration and code versions (git_rev, config snapshots) together, so any run
     can be reproduced and audited.
   - Keep dangerous actions (live trading, capital allocation, overrides of risk limits)
     clearly guarded and logged.

9) Testing, Simulation & Deployment
   - Provide fast, realistic tests for critical components: SqlStore, signal generation,
     risk engine, execution router, and key dashboards.
   - Support simulation and replay modes (e.g. run live-like flows on historical data)
     to validate end-to-end behaviour before enabling in live.
   - Make it easy to deploy parameter or code changes gradually (dev → paper → live),
     with clear checkpoints and health checks at each step.

10) AI, Automation & Upgrade Agent
    - Treat the upgrade agent and other AI agents as **first-class tools** in the development
      and operations pipeline, not as gimmicks:
        * They should propose focused, incremental improvements,
        * Respect safety and compatibility constraints,
        * And always aim to increase system cohesion and utilization.
    - Prefer automation that:
        * Reduces manual, error-prone workflows,
        * Connects disconnected pieces of the system (e.g. SqlStore ↔ dashboards ↔ agents),
        * Improves the ability to reason about the state of the fund in real time.
"""



# ======================================================================
# בניית פרומפט לתוכנית שדרוג (plan)
# ======================================================================


def build_upgrade_file_prompt(
    *,
    snapshot: RepoSnapshot,
    plan: UpgradePlan,
    target_file: FilePlan,
    current_source: str,
) -> str:
    """
    מייצר פרומפט מפורט עבור קובץ יחיד: תיאור תפקיד הקובץ + הטאסקים + קונטקסט.
    המטרה: גם לשפר את הקובץ עצמו, וגם לחזק את החיבור של המערכת כמנוע אחד
    (Data → Research → Decisions → Execution → Monitoring).
    """
    # סיכום קצר של כל הפרויקט
    summary = build_repo_summary_text(snapshot, max_files=80)

    # סיכום קצר של הטאסקים לקובץ
    tasks_bullets = "\n".join(f"- {t}" for t in target_file.tasks) or "- (no tasks in plan)"

    # באילו phases הקובץ משתתף
    phases_by_id = {ph.id: ph for ph in plan.phases}
    phase_lines: List[str] = []
    for pid in target_file.phase_ids:
        ph = phases_by_id.get(pid)
        if not ph:
            continue
        phase_lines.append(f"- {pid}: {ph.name} (order={ph.order})")
    phases_text = "\n".join(phase_lines) or "- (file not mapped to any phase explicitly)"

    # קבצים חשובים אחרים, לצורך קונטקסט של "מערכת"
    other_high_priority_files = [
        f for f in plan.files if f.priority == "high" and f.rel_path != target_file.rel_path
    ]
    other_high_lines: List[str] = []
    for f in other_high_priority_files[:15]:
        cats = ",".join(f.categories)
        other_high_lines.append(f"- {f.rel_path}  [{cats}]: {f.role}")
    others_text = "\n".join(other_high_lines) or "- (no other high-priority files listed)"

    # הפרומפט המלא למודל
    return textwrap.dedent(
        f"""
        You are editing ONE file inside a pairs-trading project that must be upgraded to
        **hedge-fund-grade live trading**, as part of a single, coherent end-to-end system.

        GLOBAL CONTEXT
        ==============
        {HEDGE_FUND_BRIEF}

        Project snapshot (trimmed)
        --------------------------
        {summary}

        Target file (from the upgrade plan)
        ===================================
        - path: {target_file.rel_path}
        - role: {target_file.role}
        - priority: {target_file.priority}
        - categories: {", ".join(target_file.categories) if target_file.categories else "(none)"}

        This file participates in phases:
        ---------------------------------
        {phases_text}

        Concrete tasks for THIS file
        ============================
        {tasks_bullets}

        Other high-priority files for reference
        =======================================
        {others_text}

        SYSTEM GOALS
        ============
        - Think of the whole codebase as a single, integrated engine:
          * Data ingest                → SqlStore / data layer (DuckDB/SQL, prices, signals, risk state, experiments)
          * Research / backtests / optimization → signals, regimes, risk overlays, metrics
          * Execution / broker layer   → paper/live trading via a unified order/broker router
          * Dashboard / agents         → monitoring, decisions, automation, upgrade/ops agents.
        - Your changes to this file should:
          * Reduce fragmentation (fewer ad-hoc flows, more use of shared abstractions).
          * Increase **effective system utilization** (more of the project’s modules being used end-to-end).
          * Move the system closer to “one coherent pipeline” from data → research → decisions → trades → monitoring.
          * Prefer integration with shared infrastructure (AppContext, SqlStore, risk engine, broker router,
            dashboard/agents) over inventing new custom paths.

        CURRENT FILE CONTENT
        ====================
        ```python
        {current_source}
        ```

        YOUR JOB
        ========
        - Bring this file to **production / hedge-fund-grade** quality while:
          * Respecting the tasks listed for this file.
          * Keeping backwards compatibility as much as reasonable (avoid breaking imports/APIs silently).
          * Preferring coherent, well-structured refactors over tiny patches when they clearly simplify the
            architecture and improve integration, even if the diff is non-trivial.
          * Improving type hints, docstrings, logging, and risk-safety checks where relevant.
          * Improving integration with data layer, live trading, risk, and web/agents layers where appropriate.
          * Increasing system-wide cohesion: when safe, replace ad-hoc logic with shared infrastructure
            (SqlStore, AppContext, risk engine, broker router, dashboard/agents) so the project behaves
            more like one unified system and less like isolated scripts.
          * Avoiding unnecessary new dependencies or “micro-frameworks” unless they clearly improve robustness
            or observability.

        - If you detect obvious bugs, fix them.
        - If the file is obsolete, add a clear comment at the top explaining how it should be replaced or which
          shared component should be used instead.

         OUTPUT FORMAT (STRICT)
        ======================
        1. A short section "Summary of changes" as markdown bullets (max ~10 bullets).
        2. A single ```python``` block containing the **FULL updated file**. This block is MANDATORY.
           - If you are unsure or cannot improve the file, STILL return the original file inside the ```python``` block.
           - Do NOT omit the ```python``` block under any circumstances.
        3. A "Sanity checklist" section with manual tests I should run (pytest, manual run, etc.).


        Do NOT modify any other files in this response; only this one file’s contents are allowed in the code block.
        """
    ).strip()




def build_phases_prompt(snapshot: RepoSnapshot) -> str:
    """
    פרומפט שמבקש מהמודל להחזיר *רק* את רשימת ה-phases (בלי files).
    """
    summary = build_repo_summary_text(snapshot, max_files=80)
    return textwrap.dedent(
        f"""
        You are a senior quant + platform architect.

        {HEDGE_FUND_BRIEF}

        Below is a snapshot of the project, listing key Python files:

        ---
        {summary}
        ---

        TASK
        ====
        Design the high-level upgrade phases to move this project to hedge-fund-grade.

        The output MUST be **pure JSON** of the form:

        {{
          "phases": [
            {{
              "id": "phase_architecture",
              "name": "Architecture & Environment Modes",
              "order": 1,
              "description": "Short 1-3 sentence description",
              "milestones": [
                "Short bullet milestone",
                "Another milestone"
              ]
            }},
            ...
          ]
        }}

        RULES
        =====
        - Keep 4–8 phases.
        - Keep descriptions and milestones concise.
        - Do NOT include any "files" array here.
        - Do NOT use markdown. Return valid JSON only.
        """
    ).strip()


def build_file_chunk_prompt(
    snapshot: RepoSnapshot,
    phases: List[Phase],
    files_chunk: List[FileInfo],
    chunk_index: int,
    total_chunks: int,
) -> str:
    """
    פרומפט שמבקש מהמודל להחזיר FilePlan רק עבור קבוצת קבצים אחת (chunk).
    """
    # נסכם קצת את השורש, אבל נתרכז ברשימת הקבצים של ה-chunk
    root_summary = build_repo_summary_text(snapshot, max_files=40)

    files_lines = []
    for fi in files_chunk:
        files_lines.append(
            f"- {fi.rel_path} (lines={fi.n_lines}, bytes={fi.size_bytes}) — {fi.header}"
        )
    files_text = "\n".join(files_lines)

    phases_lines = []
    for ph in sorted(phases, key=lambda p: p.order):
        phases_lines.append(f"- {ph.id}: {ph.name} (order={ph.order})")
    phases_text = "\n".join(phases_lines)

    return textwrap.dedent(
        f"""
        You are designing per-file upgrade tasks for a hedge-fund-grade pairs-trading system.

        GLOBAL CONTEXT
        ==============
        {HEDGE_FUND_BRIEF}

        Project snapshot (trimmed)
        --------------------------
        {root_summary}

        Existing upgrade phases
        -----------------------
        {phases_text}

        FILE CHUNK {chunk_index+1}/{total_chunks}
        ==========================
        Below is the list of files you must cover in THIS chunk only:

        {files_text}

        TASK
        ====
        For **each file listed above**, produce a JSON object describing its role and concrete upgrade tasks.

        The output MUST be **pure JSON** of the form:

        {{
          "files": [
            {{
              "rel_path": "relative/path/to/file.py",
              "role": "short description of what this file should represent in a hedge-fund architecture",
              "priority": "high|medium|low",
              "categories": [
                "one or more of: 'web_ui', 'api', 'data_ingest', 'research', 'backtest', 'live_trading', 'risk', 'infra', 'utils', 'tests', 'other'"
              ],
              "tasks": [
                "Concrete todo item for this specific file (short)",
                "Another todo item (short)",
                "Use 2–6 tasks depending on priority"
              ],
              "phase_ids": [
                "ids of phases where this file is primarily modified, e.g. ['phase_architecture', 'phase_live_trading']"
              ]
            }},
            ...
          ]
        }}

        RULES
        =====
        - You MUST include an entry in "files" for EVERY file in this chunk.
        - You MUST NOT include files that are not listed in this chunk.
        - Keep text concise and avoid repetition.
        - Use reasonable priorities:
          * 'high' for core/ root/ scripts/ pieces in architecture, data, execution, web, risk.
          * 'medium' for supporting modules.
          * 'low' for small helpers/utilities.
        - Make sure phase_ids refer only to the existing phase ids: {", ".join(p.id for p in phases)}.
        - Do NOT include a top-level "phases" key here.
        - Do NOT use markdown. Return valid JSON only.
        """
    ).strip()


# ======================================================================
# בניית פרומפט לשדרוג קובץ יחיד
# ======================================================================


def build_upgrade_file_prompt(
    *,
    snapshot: RepoSnapshot,
    plan: UpgradePlan,
    target_file: FilePlan,
    current_source: str,
) -> str:
    """
    מייצר פרומפט מפורט עבור קובץ יחיד: תיאור תפקיד הקובץ + הטאסקים + קונטקסט.
    """
    summary = build_repo_summary_text(snapshot, max_files=80)

    # סיכום קצר של הטאסקים לקובץ
    tasks_bullets = "\n".join(f"- {t}" for t in target_file.tasks) or "- (no tasks in plan)"

    # באילו phases הקובץ משתתף
    phases_by_id = {ph.id: ph for ph in plan.phases}
    phase_lines: List[str] = []
    for pid in target_file.phase_ids:
        ph = phases_by_id.get(pid)
        if not ph:
            continue
        phase_lines.append(f"- {pid}: {ph.name} (order={ph.order})")
    phases_text = "\n".join(phase_lines) or "- (file not mapped to any phase explicitly)"

    # קונטקסט קצר על פרויקטים נוספים – במיוחד על כיווני קרן גידור
    other_high_priority_files = [
        f for f in plan.files if f.priority == "high" and f.rel_path != target_file.rel_path
    ]
    other_high_lines: List[str] = []
    for f in other_high_priority_files[:15]:
        cats = ",".join(f.categories)
        other_high_lines.append(f"- {f.rel_path}  [{cats}]: {f.role}")
    others_text = "\n".join(other_high_lines) or "- (no other high-priority files listed)"

    return textwrap.dedent(
        f"""
        You are editing ONE file inside a pairs-trading project that must be upgraded to
        **hedge-fund-grade live trading**.

        GLOBAL CONTEXT
        ==============
        {HEDGE_FUND_BRIEF}

        Project snapshot (trimmed)
        --------------------------
        {summary}

        Target file (from the upgrade plan)
        ===================================
        - path: {target_file.rel_path}
        - role: {target_file.role}
        - priority: {target_file.priority}
        - categories: {", ".join(target_file.categories) if target_file.categories else "(none)"}

        This file participates in phases:
        ---------------------------------
        {phases_text}

        Concrete tasks for THIS file
        ============================
        {tasks_bullets}

        Other high-priority files for reference
        =======================================
        {others_text}

        CURRENT FILE CONTENT
        ====================
        ```python
        {current_source}
        ```

        YOUR JOB
        ========
        - Bring this file to **production / hedge-fund-grade** quality while:
          * Respecting the tasks listed for this file.
          * Keeping backwards compatibility as much as reasonable (avoid breaking imports/APIs silently).
          * Minimizing diff size: prefer focused refactors over rewrites.
          * Improving type hints, docstrings, logging, and risk-safety checks where relevant.
          * Improving integration with data layer, live trading, risk, and web layers where appropriate.

        - If you detect obvious bugs, fix them.
        - If the file is obsolete, add a clear comment at the top explaining how it should be replaced.

         OUTPUT FORMAT (STRICT)
        ======================
        1. A short section "Summary of changes" as markdown bullets (max ~10 bullets).
        2. A single ```python``` block containing the **FULL updated file**. This block is MANDATORY.
           - If you are unsure or cannot improve the file, STILL return the original file inside the ```python``` block.
           - Do NOT omit the ```python``` block under any circumstances.
        3. A "Sanity checklist" section with manual tests I should run (pytest, manual run, etc.).


        Do NOT modify any other files in this response; only this one file’s contents are allowed in the code block.
        """
    ).strip()


# ======================================================================
# המרה מ-JSON dict ל-UpgradePlan
# ======================================================================


def plan_from_dict(d: Dict[str, Any]) -> UpgradePlan:
    phases: List[Phase] = []
    for ph in d.get("phases", []):
        phases.append(
            Phase(
                id=ph["id"],
                name=ph["name"],
                order=int(ph.get("order", 0)),
                description=ph.get("description", ""),
                milestones=list(ph.get("milestones", []) or []),
            )
        )

    files: List[FilePlan] = []
    for f in d.get("files", []):
        files.append(
            FilePlan(
                rel_path=f["rel_path"],
                role=f.get("role", ""),
                priority=f.get("priority", "medium"),
                categories=list(f.get("categories", []) or []),
                tasks=list(f.get("tasks", []) or []),
                phase_ids=list(f.get("phase_ids", []) or []),
            )
        )

    return UpgradePlan(
        project_root=d.get("project_root", ""),
        phases=phases,
        files=files,
    )


def plan_to_dict(plan: UpgradePlan) -> Dict[str, Any]:
    return {
        "project_root": plan.project_root,
        "phases": [asdict(ph) for ph in plan.phases],
        "files": [asdict(f) for f in plan.files],
    }


def apply_static_overrides_to_plan(plan: UpgradePlan) -> UpgradePlan:
    """
    Apply deterministic phase and per-file overrides to an UpgradePlan.

    Ensures that critical phases and this agent file have stable, explicitly
    modeled metadata regardless of GPT variability. Idempotent by design.
    """
    # Phases: merge/override by id
    phases_by_id: Dict[str, Phase] = {p.id: p for p in plan.phases}
    for pid, meta in STATIC_PHASES.items():
        if pid in phases_by_id:
            p = phases_by_id[pid]
            p.name = meta["name"]
            p.order = int(meta.get("order", p.order))
            p.description = meta.get("description", p.description)
            p.milestones = list(meta.get("milestones", []) or [])
        else:
            phases_by_id[pid] = Phase(
                id=meta["id"],
                name=meta["name"],
                order=int(meta.get("order", 0)),
                description=meta.get("description", ""),
                milestones=list(meta.get("milestones", []) or []),
            )
    plan.phases = sorted(phases_by_id.values(), key=lambda p: p.order)

    # Files: override this agent file's metadata/tasks if present, or create it
    files_by_rel: Dict[str, FilePlan] = {
        f.rel_path.replace("\\", "/"): f for f in plan.files
    }
    for rel_key, meta in STATIC_FILE_TASKS.items():
        rel_norm = rel_key.replace("\\", "/")
        fp = files_by_rel.get(rel_norm)
        if fp is None:
            fp = FilePlan(
                rel_path=rel_norm,
                role=meta.get("role", ""),
                priority=meta.get("priority", "medium"),
                categories=list(meta.get("categories", []) or []),
                tasks=list(meta.get("tasks", []) or []),
                phase_ids=list(meta.get("phase_ids", []) or []),
            )
            plan.files.append(fp)
            files_by_rel[rel_norm] = fp
        else:
            if "role" in meta:
                fp.role = meta["role"]
            if "priority" in meta:
                fp.priority = meta["priority"]
            if "categories" in meta:
                fp.categories = list(meta["categories"])
            if "tasks" in meta:
                fp.tasks = list(meta["tasks"])
            if "phase_ids" in meta:
                fp.phase_ids = list(meta["phase_ids"])

    return plan


# ======================================================================
# פקודות CLI – לוגיקה של יצירת תוכנית, שדרוגים וכו'
# ======================================================================


def load_existing_plan(root: Path, json_name: str) -> UpgradePlan:
    path = (root / json_name).resolve()
    if not path.is_file():
        raise SystemExit(f"Plan JSON not found: {path}. Run 'plan' first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    plan = plan_from_dict(data)
    return apply_static_overrides_to_plan(plan)


def find_file_plan(plan: UpgradePlan, rel_path: str) -> Optional[FilePlan]:
    rel_norm = rel_path.replace("\\", "/")
    for f in plan.files:
        if f.rel_path.replace("\\", "/") == rel_norm:
            return f
    return None


def cmd_plan(args: argparse.Namespace) -> None:
    root: Path = args.root.resolve()
    debug(f"Scanning repo at: {root}")
    snapshot = scan_repo(root)

    if not snapshot.files:
        raise SystemExit("No Python files found under root.")

    system_instructions = load_master_prompt(args.master_prompt)

    # ============================
    # שלב 1: לבנות phases בלבד
    # ============================
    debug("Requesting phases (no files yet)...")
    phases_prompt = build_phases_prompt(snapshot)
    phases_json = call_gpt_json(
        model=args.model,
        instructions=system_instructions,
        user_input=phases_prompt,
        reasoning_effort=args.reasoning,
        max_output_tokens=args.max_tokens,
    )

    phases_raw = phases_json.get("phases", [])
    if not isinstance(phases_raw, list) or not phases_raw:
        raise RuntimeError(f"Model did not return a valid 'phases' array. Got: {phases_json}")

    phases: List[Phase] = []
    for ph in phases_raw:
        phases.append(
            Phase(
                id=ph["id"],
                name=ph["name"],
                order=int(ph.get("order", 0)),
                description=ph.get("description", ""),
                milestones=list(ph.get("milestones", []) or []),
            )
        )

    debug(f"Received {len(phases)} phases.")

    # ============================
    # שלב 2: לולאה על קבצים (chunks)
    # ============================
    chunk_size = args.chunk_size
    files_plans: List[FilePlan] = []

    total_files = len(snapshot.files)
    total_chunks = (total_files + chunk_size - 1) // chunk_size

    debug(
        f"Building file plans in {total_chunks} chunks (chunk_size={chunk_size}, total_files={total_files})."
    )

    for ci in range(total_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, total_files)
        chunk = snapshot.files[start:end]

        debug(
            f"Requesting file plans for chunk {ci+1}/{total_chunks} (files {start+1}-{end})..."
        )
        chunk_prompt = build_file_chunk_prompt(
            snapshot=snapshot,
            phases=phases,
            files_chunk=chunk,
            chunk_index=ci,
            total_chunks=total_chunks,
        )

        chunk_json = call_gpt_json(
            model=args.model,
            instructions=system_instructions,
            user_input=chunk_prompt,
            reasoning_effort=args.reasoning,
            max_output_tokens=args.max_tokens,
        )

        chunk_files_raw = chunk_json.get("files", [])
        if not isinstance(chunk_files_raw, list) or not chunk_files_raw:
            debug(f"[WARN] Empty or invalid 'files' in chunk {ci+1}; skipping.")
            continue

        for f in chunk_files_raw:
            try:
                fp = FilePlan(
                    rel_path=f["rel_path"],
                    role=f.get("role", ""),
                    priority=f.get("priority", "medium"),
                    categories=list(f.get("categories", []) or []),
                    tasks=list(f.get("tasks", []) or []),
                    phase_ids=list(f.get("phase_ids", []) or []),
                )
                files_plans.append(fp)
            except Exception as exc:
                debug(f"[WARN] Failed to parse FilePlan in chunk {ci+1}: {exc}")

    debug(f"Collected {len(files_plans)} file plans from all chunks.")

    # ============================
    # בניית UpgradePlan ושמירה
    # ============================
    plan = UpgradePlan(
        project_root=str(root),
        phases=phases,
        files=files_plans,
    )

    # להחיל overrides דטרמיניסטיים לפייזים ולקבצים קריטיים
    plan = apply_static_overrides_to_plan(plan)

    json_path = root / args.plan_json
    md_path = root / args.plan_md

    save_text(json_path, json.dumps(plan_to_dict(plan), indent=2, ensure_ascii=False))
    save_text(md_path, render_plan_markdown(plan))

    debug(f"Wrote JSON plan: {json_path}")
    debug(f"Wrote Markdown plan: {md_path}")
    print()
    print(
        f"Plan created with {len(plan.phases)} phases and {len(plan.files)} files (chunk_size={chunk_size})."
    )
    print(f"- JSON : {json_path}")
    print(f"- Markdown : {md_path}")


# ======================================================================
# יצירת Markdown אנושי מהתוכנית
# ======================================================================


def render_plan_markdown(plan: UpgradePlan) -> str:
    lines: List[str] = []
    lines.append(f"# Hedge-Fund Upgrade Plan")
    if plan.project_root:
        lines.append(f"_Project root: {plan.project_root}_")
    lines.append("")

    # Phases
    lines.append("## Phases")
    for ph in sorted(plan.phases, key=lambda p: p.order):
        lines.append(f"### {ph.order}. {ph.name}  (`{ph.id}`)")
        if ph.description:
            lines.append("")
            lines.append(ph.description)
        if ph.milestones:
            lines.append("")
            lines.append("**Milestones:**")
            for m in ph.milestones:
                lines.append(f"- {m}")
        lines.append("")

    # Files
    lines.append("## Files & Tasks")
    for f in sorted(plan.files, key=lambda x: x.rel_path):
        lines.append(f"### `{f.rel_path}`")
        lines.append(f"- Role: {f.role or '(n/a)'}")
        lines.append(f"- Priority: `{f.priority}`")
        if f.categories:
            lines.append(f"- Categories: {', '.join(f.categories)}")
        if f.phase_ids:
            lines.append(f"- Phases: {', '.join(f.phase_ids)}")
        if f.tasks:
            lines.append("")
            lines.append("**Tasks:**")
            for t in f.tasks:
                lines.append(f"- {t}")
        lines.append("")

    return "\n".join(lines)


# ======================================================================
# Upgrade-file – שדרוג קובץ יחיד לפי ה-Plan
# ======================================================================


def cmd_upgrade_file(args: argparse.Namespace) -> None:
    """
    משדרג קובץ יחיד לפי ה-Plan.

    לוגיקה:
    -------
    - always: Dry-run (apply=False) → קורא ל-GPT, שומר את כל התשובה ל-dump, מחלץ קוד ומדפיס Summary.
    - apply=True:
        1) אם יש dump קיים (logs/upgrade_<safe_rel>_dump.md) → משתמש בו ולא קורא שוב ל-GPT.
        2) אם אין dump → קורא ל-GPT, שומר dump, מחלץ קוד ומיישם.
    - לא כותב לקובץ אם אין שינוי אמיתי (idempotent).
    - אם diff → משתמש ב-apply_unified_diff כדי לייצר new_source לפני כתיבה.
    """
    root: Path = args.root.resolve()
    plan = load_existing_plan(root, args.plan_json)

    target_path = (root / args.file).resolve()
    if not target_path.is_file():
        raise SystemExit(f"Target file not found: {target_path}")

    rel = str(target_path.relative_to(root))
    file_plan = find_file_plan(plan, rel)
    if not file_plan:
        raise SystemExit(f"File `{rel}` not found in plan. Re-run 'plan' or check JSON.")

    debug(f"Upgrading file according to plan: {rel}")

    snapshot = scan_repo(root)
    current_source = load_text(target_path)
    system_instructions = load_master_prompt(args.master_prompt)

    # נתיבי לוג ודאמפ
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    safe_rel = rel.replace("\\", "_").replace("/", "_")
    dump_path = logs_dir / f"upgrade_{safe_rel}_dump.md"

    # Snapshot (optional, idempotent)
    snapshot_path: Optional[Path] = None
    if args.apply and getattr(args, "snapshot", False):
        snapshot_path = create_project_snapshot(root, f"upgrade-file_{safe_rel}")
        debug(f"Project snapshot for {rel}: {snapshot_path}")

    response_text: str

    # אם apply=True ויש כבר dump → נשתמש בו ולא נקרא שוב ל-GPT
    if args.apply and dump_path.is_file():
        debug(f"Using existing dump for {rel}: {dump_path}")
        response_text = load_text(dump_path)
    else:
        # קוראים ל-GPT ומייצרים dump חדש
        user_input = build_upgrade_file_prompt(
            snapshot=snapshot,
            plan=plan,
            target_file=file_plan,
            current_source=current_source,
        )

        debug(f"Calling model={args.model} for file upgrade (apply={args.apply})")
        response_text = call_gpt_raw(
            model=args.model,
            instructions=system_instructions,
            user_input=user_input,
            reasoning_effort=args.reasoning,
            max_output_tokens=args.max_tokens,
        )

        # שומרים את כל התשובה ל-dump
        save_text(dump_path, response_text)
        debug(f"Full model response saved to: {dump_path}")

    # מחלץ קוד מתוך ה-dump / התשובה
    code = extract_code_block_from_markdown(response_text)
    if code is None:
        debug("WARNING: Could not find code/diff block in response; not modifying file.")
        print(response_text)
        return

    # נזהה אם זה diff או קובץ מלא:
    is_diff = (
        "```diff" in response_text
        or code.lstrip().startswith("@@")
        or code.lstrip().startswith("--- ")
        or code.lstrip().startswith("diff ")
    )

    if is_diff:
        debug("Detected unified diff in response; applying patch to current source.")
        new_source = apply_unified_diff(current_source, code)
    else:
        # מניח שזה קובץ Python מלא
        new_source = code

    # אם Apply: נכתוב לקובץ (אם יש שינוי אמיתי)
    if args.apply:
        if new_source.strip() == current_source.strip():
            debug(
                "No changes detected between generated code and current file; skipping write."
            )
            return
        backup = backup_file(target_path)
        debug(f"Backup written: {backup}")
        save_text(target_path, new_source)
        debug(f"File updated: {target_path}")
    else:
        debug("Dry run only: not writing file. Use --apply to save changes.")
        print("\n===== MODEL RESPONSE (truncated) =====\n")
        print(
            textwrap.shorten(
                response_text, width=5000, placeholder="\n... [truncated] ..."
            )
        )


# ======================================================================
# Health-check integration
# ======================================================================


def run_health_check(root: Path) -> bool:
    """
    Run health_check_full_system.py if present.

    Returns:
        True if the script existed and exited with code 0, False otherwise.
    """
    script = (root / "health_check_full_system.py").resolve()
    if not script.is_file():
        debug("[health-check] health_check_full_system.py not found; skipping.")
        return False

    debug(f"[health-check] Running health check script: {script}")
    try:
        result = subprocess.run(
            [os.sys.executable, str(script)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        debug(f"[health-check] ERROR invoking health check: {exc}")
        return False

    debug(f"[health-check] returncode={result.returncode}")
    if result.stdout:
        debug(f"[health-check][stdout]\n{result.stdout.strip()}")
    if result.stderr:
        debug(f"[health-check][stderr]\n{result.stderr.strip()}")

    if result.returncode != 0:
        debug("[health-check] Health check FAILED.")
        return False

    debug("[health-check] Health check passed.")
    return True


# ======================================================================
# Upgrade-all – שדרוג כל הקבצים לפי התוכנית
# ======================================================================


def cmd_upgrade_all(args: argparse.Namespace) -> None:
    root: Path = args.root.resolve()
    plan = load_existing_plan(root, args.plan_json)
    snapshot = scan_repo(root)
    system_instructions = load_master_prompt(args.master_prompt)

    # Optional project-wide snapshot before any modifications
    snapshot_path: Optional[Path] = None
    if args.apply and getattr(args, "snapshot", False):
        snapshot_path = create_project_snapshot(root, "upgrade-all")
        debug(f"Project snapshot for upgrade-all: {snapshot_path}")

    # Deterministic ordering: by primary phase order, then priority, then path
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    phase_order_by_id: Dict[str, int] = {ph.id: ph.order for ph in plan.phases}

    def primary_phase_order(fp: FilePlan) -> int:
        orders = [phase_order_by_id.get(pid, 9999) for pid in fp.phase_ids]
        return min(orders) if orders else 9999

    files_with_meta = [(fp, primary_phase_order(fp)) for fp in plan.files]
    files_with_meta.sort(
        key=lambda item: (
            item[1],
            priority_rank.get(item[0].priority, 1),
            item[0].rel_path,
        )
    )

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    last_phase_order: Optional[int] = None

    for fp, phase_order in files_with_meta:
        target_path = (root / fp.rel_path).resolve()
        if not target_path.is_file():
            debug(f"[SKIP] Missing file listed in plan: {fp.rel_path}")
            continue

        # אם סיימנו פייז (phase_order השתנה) – אפשר להריץ health-check
        if (
            args.apply
            and getattr(args, "health_check", False)
            and last_phase_order is not None
            and last_phase_order != 9999
            and phase_order != last_phase_order
        ):
            debug(f"Completed phase order {last_phase_order}; running health check.")
            ok = run_health_check(root)
            if not ok:
                raise SystemExit(
                    f"Health check failed after completing phase order {last_phase_order}; aborting further upgrades."
                )

        last_phase_order = phase_order

        debug("=" * 80)
        debug(
            f"Upgrading {fp.rel_path} (phase_order={phase_order}, priority={fp.priority}, apply={args.apply})"
        )

        current_source = load_text(target_path)

        # per-file dump path
        safe_rel = fp.rel_path.replace("\\", "_").replace("/", "_")
        dump_path = logs_dir / f"upgrade_{safe_rel}_dump.md"

        # אם apply=True ויש dump קיים – נשתמש בו, אחרת נקרא ל-GPT ונשמור
        if args.apply and dump_path.is_file():
            debug(f"[{fp.rel_path}] Using existing dump: {dump_path}")
            response_text = load_text(dump_path)
        else:
            user_input = build_upgrade_file_prompt(
                snapshot=snapshot,
                plan=plan,
                target_file=fp,
                current_source=current_source,
            )

            try:
                response_text = call_gpt_raw(
                    model=args.model,
                    instructions=system_instructions,
                    user_input=user_input,
                    reasoning_effort=args.reasoning,
                    max_output_tokens=args.max_tokens,
                )
            except Exception as exc:
                debug(f"ERROR calling model for {fp.rel_path}: {exc}")
                continue

            save_text(dump_path, response_text)
            debug(f"[{fp.rel_path}] Full model response saved to: {dump_path}")

        code = extract_code_block_from_markdown(response_text)
        if code is None:
            debug(f"[WARN] No python/diff code block for {fp.rel_path}; skipping file update.")
            continue

        is_diff = (
            "```diff" in response_text
            or code.lstrip().startswith("@@")
            or code.lstrip().startswith("--- ")
            or code.lstrip().startswith("diff ")
        )

        if is_diff:
            debug(f"[{fp.rel_path}] Detected unified diff; applying patch.")
            new_source = apply_unified_diff(current_source, code)
        else:
            new_source = code

        if args.apply:
            if new_source.strip() == current_source.strip():
                debug(
                    f"[{fp.rel_path}] No changes between generated code and current file; skipping write."
                )
                continue
            backup = backup_file(target_path)
            debug(f"[{fp.rel_path}] Backup: {backup}")
            save_text(target_path, new_source)
            debug(f"[{fp.rel_path}] Updated.")
        else:
            debug(f"[{fp.rel_path}] Dry run only (no write).")

    # Health-check בסיום כל הריצות (אם ביקשו)
    if args.apply and getattr(args, "health_check", False):
        debug("Completed all upgrades; running final health check.")
        ok = run_health_check(root)
        if not ok:
            raise SystemExit("Final health check failed after upgrade-all.")

    debug("Finished upgrade-all.")


# ======================================================================
# show-plan – תצוגת סיכום מהתוכנית
# ======================================================================


def cmd_show_plan(args: argparse.Namespace) -> None:
    root: Path = args.root.resolve()
    plan = load_existing_plan(root, args.plan_json)

    print(f"Hedge-fund upgrade plan for project: {plan.project_root}")
    print(f"- Phases: {len(plan.phases)}")
    print(f"- Files : {len(plan.files)}")
    print("")

    print("PHASES:")
    for ph in sorted(plan.phases, key=lambda p: p.order):
        print(f"  {ph.order}. {ph.name}  ({ph.id})")
    print("")

    if args.verbose:
        print("FILES:")
        for f in sorted(plan.files, key=lambda x: x.rel_path):
            cats = ", ".join(f.categories)
            print(f"  - {f.rel_path}  [{f.priority}]  ({cats})")
            if args.verbose >= 2:
                for t in f.tasks:
                    print(f"      * {t}")


# ======================================================================
# CLI setup
# ======================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GPT-based hedge-fund upgrade agent for your pairs-trading project.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model id (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--reasoning",
        type=str,
        default=DEFAULT_REASONING,
        help="Reasoning effort for GPT-5/GPT-5.1 models (none/low/medium/high).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=20000,  # במקום 8192
        help="max_output_tokens for the Responses API.",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=40,
        help="Number of files per GPT chunk when building the plan (to avoid huge JSON).",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # plan
    sp = sub.add_parser(
        "plan",
        help="Scan repo and generate hedge-fund upgrade plan + tasks for each file.",
    )
    sp.add_argument("--root", type=Path, default=Path("."), help="Project root directory.")
    sp.add_argument(
        "--master-prompt",
        type=Path,
        default=DEFAULT_MASTER_PROMPT,
        help="Path to your master prompt (system instructions).",
    )
    sp.add_argument(
        "--plan-json",
        type=str,
        default=DEFAULT_PLAN_JSON,
        help="Output JSON plan filename.",
    )
    sp.add_argument(
        "--plan-md",
        type=str,
        default=DEFAULT_PLAN_MD,
        help="Output Markdown plan filename.",
    )
    sp.set_defaults(func=cmd_plan)

    # upgrade-file
    sf = sub.add_parser("upgrade-file", help="Upgrade a single file using the existing plan.")
    sf.add_argument("--root", type=Path, default=Path("."), help="Project root directory.")
    sf.add_argument("--file", type=str, required=True, help="Relative path of file to upgrade.")
    sf.add_argument(
        "--master-prompt",
        type=Path,
        default=DEFAULT_MASTER_PROMPT,
        help="Path to your master prompt (system instructions).",
    )
    sf.add_argument(
        "--plan-json",
        type=str,
        default=DEFAULT_PLAN_JSON,
        help="Plan JSON filename (must exist).",
    )
    sf.add_argument(
        "--apply",
        action="store_true",
        help="Actually overwrite the file (with .gpt_backup). Default: dry run.",
    )
    sf.add_argument(
        "--snapshot",
        action="store_true",
        help="Create a project snapshot before applying this file's upgrade.",
    )
    sf.set_defaults(func=cmd_upgrade_file)

    # upgrade-all
    sa = sub.add_parser(
        "upgrade-all",
        help="Upgrade all files from the plan, ordered by phase, priority, and path.",
    )
    sa.add_argument("--root", type=Path, default=Path("."), help="Project root directory.")
    sa.add_argument(
        "--master-prompt",
        type=Path,
        default=DEFAULT_MASTER_PROMPT,
        help="Path to your master prompt (system instructions).",
    )
    sa.add_argument(
        "--plan-json",
        type=str,
        default=DEFAULT_PLAN_JSON,
        help="Plan JSON filename (must exist).",
    )
    sa.add_argument(
        "--apply",
        action="store_true",
        help="If set, writes changes to disk (with backups). Otherwise dry-run.",
    )
    sa.add_argument(
        "--snapshot",
        action="store_true",
        help="Create a project-wide snapshot archive before applying upgrades.",
    )
    sa.add_argument(
        "--health-check",
        action="store_true",
        help="Run health_check_full_system.py after each completed phase (and at the end) and abort on failure.",
    )
    sa.set_defaults(func=cmd_upgrade_all)

    # show-plan
    ss = sub.add_parser("show-plan", help="Print a quick summary of the existing plan.")
    ss.add_argument("--root", type=Path, default=Path("."), help="Project root directory.")
    ss.add_argument(
        "--plan-json",
        type=str,
        default=DEFAULT_PLAN_JSON,
        help="Plan JSON filename (must exist).",
    )
    ss.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Show more detail (e.g. tasks for each file with -vv).",
    )
    ss.set_defaults(func=cmd_show_plan)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
