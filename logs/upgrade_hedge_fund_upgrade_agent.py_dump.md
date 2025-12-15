Plan
- Add static, hand-curated phase and per-file metadata (including this agent’s tasks) and a helper to overlay them onto any loaded/created `UpgradePlan`.
- Make `load_existing_plan` and `cmd_plan` apply these static overrides so the plan is deterministic for critical phases/files.
- Introduce a project-wide snapshot facility (zip archive of code/configs) with idempotent naming, and wire it into `upgrade-file` and `upgrade-all` via a `--snapshot` flag.
- Extend `upgrade-all` to:
  - Sort files deterministically by primary phase order, then priority, then path.
  - Reuse/record per-file GPT dumps similar to `upgrade-file`.
  - Skip writes when generated code is unchanged (idempotent).
- Add a `--health-check` flag to `upgrade-all` and a `run_health_check` helper that runs `health_check_full_system.py` after each completed phase and at the end, aborting on failure.
- Minor import/utility cleanups where touched (add `zipfile`, `subprocess`, remove unused `Iterable`).

---

## Unified Diff

```diff
@@
-import argparse
-import json
-import os
-import re
-import textwrap
-from dataclasses import dataclass, asdict
-from pathlib import Path
-from typing import Any, Dict, Iterable, List, Optional
-
-from dotenv import load_dotenv
-from openai import OpenAI  # pip install openai
+import argparse
+import json
+import os
+import re
+import subprocess
+import textwrap
+import zipfile
+from dataclasses import dataclass, asdict
+from pathlib import Path
+from typing import Any, Dict, List, Optional
+
+from dotenv import load_dotenv
+from openai import OpenAI  # pip install openai
@@
-PROJECT_ROOT = Path(__file__).resolve().parent
-load_dotenv(PROJECT_ROOT / ".env")  # 👈 כאן הוא קורא את .env שלך
-
-
-OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
+PROJECT_ROOT = Path(__file__).resolve().parent
+load_dotenv(PROJECT_ROOT / ".env")  # 👈 כאן הוא קורא את .env שלך
+
+SNAPSHOT_DIR = PROJECT_ROOT / "upgrade_snapshots"
+
+
+OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
@@
 @dataclass
 class UpgradePlan:
     project_root: str
     phases: List[Phase]
     files: List[FilePlan]
+
+
+# ======================================================================
+# Static phase & per-file metadata (deterministic, hand-curated)
+# ======================================================================
+
+STATIC_PHASES: Dict[str, Dict[str, Any]] = {
+    "phase_architecture_env": {
+        "id": "phase_architecture_env",
+        "name": "Architecture & Environment Modes",
+        "order": 1,
+        "description": "Introduce explicit environment modes (dev/research/paper/live) and centralize upgrade orchestration.",
+        "milestones": [
+            "Define global upgrade phases and stable ids",
+            "Make upgrade commands deterministic and idempotent across runs",
+        ],
+    },
+    "phase_observability_ops": {
+        "id": "phase_observability_ops",
+        "name": "Observability, Operations & Code Quality",
+        "order": 6,
+        "description": "Add observability, health checks, and safety rails around automated upgrades.",
+        "milestones": [
+            "Introduce health checks after major upgrade phases",
+            "Add snapshotting of code/configs prior to applying upgrades",
+        ],
+    },
+}
+
+STATIC_FILE_TASKS: Dict[str, Dict[str, Any]] = {
+    "hedge_fund_upgrade_agent.py": {
+        "role": "Central coordinator for stepwise, phase-based upgrades of the codebase toward hedge-fund-grade standards.",
+        "priority": "medium",
+        "categories": ["infra", "other"],
+        "phase_ids": ["phase_architecture_env", "phase_observability_ops"],
+        "tasks": [
+            "Model upgrade phases and per-file tasks as explicit data structures tied to the global phase ids.",
+            "Add deterministic execution order and idempotency checks so rerunning the agent does not corrupt files.",
+            "Introduce dry-run and snapshot modes that back up modified files and configs before applying changes.",
+            "Integrate with health_check_full_system.py to validate the system after each major upgrade phase.",
+        ],
+    },
+}
@@
 def debug(msg: str) -> None:
     print(f"[hedge-fund-upgrader] {msg}")
@@
 def backup_file(path: Path) -> Path:
@@
     backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
     return backup
+
+
+def create_project_snapshot(root: Path, label: str) -> Path:
+    """
+    Create a deterministic zip snapshot of relevant project files (code + configs).
+
+    The snapshot name is derived from the project directory name and the provided label.
+    If a snapshot with the same name already exists, it is reused (idempotent).
+    """
+    SNAPSHOT_DIR.mkdir(exist_ok=True)
+    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label.strip()) or "snapshot"
+    archive_path = SNAPSHOT_DIR / f"{root.name}_{safe_label}.zip"
+    if archive_path.is_file():
+        debug(f"Snapshot already exists, reusing: {archive_path}")
+        return archive_path
+
+    files_to_include: List[Path] = []
+    for p in root.rglob("*"):
+        if any(part in IGNORE_DIRS for part in p.parts):
+            continue
+        if not p.is_file():
+            continue
+        if p.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".toml", ".ini"}:
+            files_to_include.append(p)
+
+    files_to_include.sort(key=lambda p: str(p.relative_to(root)))
+
+    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
+        for file_path in files_to_include:
+            zf.write(file_path, arcname=str(file_path.relative_to(root)))
+
+    debug(f"Created snapshot archive: {archive_path}")
+    return archive_path
@@
 def extract_code_block_from_markdown(text: str) -> Optional[str]:
@@
-    return code or None
+    return code or None
@@
 def plan_to_dict(plan: UpgradePlan) -> Dict[str, Any]:
     return {
         "project_root": plan.project_root,
         "phases": [asdict(ph) for ph in plan.phases],
         "files": [asdict(f) for f in plan.files],
     }
+
+
+def apply_static_overrides_to_plan(plan: UpgradePlan) -> UpgradePlan:
+    """
+    Apply deterministic phase and per-file overrides to an UpgradePlan.
+
+    Ensures that critical phases and this agent file have stable, explicitly
+    modeled metadata regardless of GPT variability. Idempotent by design.
+    """
+    # Phases: merge/override by id
+    phases_by_id: Dict[str, Phase] = {p.id: p for p in plan.phases}
+    for pid, meta in STATIC_PHASES.items():
+        if pid in phases_by_id:
+            p = phases_by_id[pid]
+            p.name = meta["name"]
+            p.order = int(meta.get("order", p.order))
+            p.description = meta.get("description", p.description)
+            p.milestones = list(meta.get("milestones", []) or [])
+        else:
+            phases_by_id[pid] = Phase(
+                id=meta["id"],
+                name=meta["name"],
+                order=int(meta.get("order", 0)),
+                description=meta.get("description", ""),
+                milestones=list(meta.get("milestones", []) or []),
+            )
+    plan.phases = sorted(phases_by_id.values(), key=lambda p: p.order)
+
+    # Files: override this agent file's metadata/tasks if present, or create it
+    files_by_rel: Dict[str, FilePlan] = {
+        f.rel_path.replace("\\", "/"): f for f in plan.files
+    }
+    for rel_key, meta in STATIC_FILE_TASKS.items():
+        rel_norm = rel_key.replace("\\", "/")
+        fp = files_by_rel.get(rel_norm)
+        if fp is None:
+            fp = FilePlan(
+                rel_path=rel_norm,
+                role=meta.get("role", ""),
+                priority=meta.get("priority", "medium"),
+                categories=list(meta.get("categories", []) or []),
+                tasks=list(meta.get("tasks", []) or []),
+                phase_ids=list(meta.get("phase_ids", []) or []),
+            )
+            plan.files.append(fp)
+            files_by_rel[rel_norm] = fp
+        else:
+            if "role" in meta:
+                fp.role = meta["role"]
+            if "priority" in meta:
+                fp.priority = meta["priority"]
+            if "categories" in meta:
+                fp.categories = list(meta["categories"])
+            if "tasks" in meta:
+                fp.tasks = list(meta["tasks"])
+            if "phase_ids" in meta:
+                fp.phase_ids = list(meta["phase_ids"])
+
+    return plan
@@
 def load_existing_plan(root: Path, json_name: str) -> UpgradePlan:
     path = (root / json_name).resolve()
     if not path.is_file():
         raise SystemExit(f"Plan JSON not found: {path}. Run 'plan' first.")
     data = json.loads(path.read_text(encoding="utf-8"))
-    return plan_from_dict(data)
+    plan = plan_from_dict(data)
+    return apply_static_overrides_to_plan(plan)
@@
 def cmd_plan(args: argparse.Namespace) -> None:
@@
-    plan = UpgradePlan(
+    plan = UpgradePlan(
         project_root=str(root),
         phases=phases,
         files=files_plans,
     )
+
+    # Apply deterministic, hand-curated overrides for critical phases/files
+    plan = apply_static_overrides_to_plan(plan)
@@
 def cmd_upgrade_file(args: argparse.Namespace) -> None:
@@
-    # נתיבי לוג ודאמפ
-    logs_dir = PROJECT_ROOT / "logs"
-    logs_dir.mkdir(exist_ok=True)
-    safe_rel = rel.replace("\\", "_").replace("/", "_")
-    dump_path = logs_dir / f"upgrade_{safe_rel}_dump.md"
+    # נתיבי לוג ודאמפ
+    logs_dir = PROJECT_ROOT / "logs"
+    logs_dir.mkdir(exist_ok=True)
+    safe_rel = rel.replace("\\", "_").replace("/", "_")
+    dump_path = logs_dir / f"upgrade_{safe_rel}_dump.md"
+
+    # Snapshot (optional, idempotent)
+    snapshot_path: Optional[Path] = None
+    if args.apply and getattr(args, "snapshot", False):
+        snapshot_path = create_project_snapshot(root, f"upgrade-file_{safe_rel}")
+        debug(f"Project snapshot for {rel}: {snapshot_path}")
@@
-    # אם apply=True ויש כבר dump → נשתמש בו ולא נקרא שוב ל-GPT
+    # אם apply=True ויש כבר dump → נשתמש בו ולא נקרא שוב ל-GPT
@@
-    else:
+    else:
@@
-    # מחלץ קוד מתוך ה-dump / התשובה
+    # מחלץ קוד מתוך ה-dump / התשובה
@@
-    # אם Apply: נכתוב לקובץ (אם יש שינוי אמיתי)
+    # אם Apply: נכתוב לקובץ (אם יש שינוי אמיתי)
@@
-        backup = backup_file(target_path)
-        debug(f"Backup written: {backup}")
-        save_text(target_path, code)
-        debug(f"File updated: {target_path}")
+        backup = backup_file(target_path)
+        debug(f"Backup written: {backup}")
+        save_text(target_path, code)
+        debug(f"File updated: {target_path}")
@@
-        debug("Dry run only: not writing file. Use --apply to save changes.")
+        debug("Dry run only: not writing file. Use --apply to save changes.")
@@
-def cmd_upgrade_all(args: argparse.Namespace) -> None:
-    root: Path = args.root.resolve()
-    plan = load_existing_plan(root, args.plan_json)
-    snapshot = scan_repo(root)
-    system_instructions = load_master_prompt(args.master_prompt)
-
-    # נעבור לפי סדר חשיבות ואז לפי שם, כדי שקוד קריטי יקבל תשומת לב קודם.
-    file_plans = sorted(
-        plan.files,
-        key=lambda f: ({"high": 0, "medium": 1, "low": 2}.get(f.priority, 1), f.rel_path),
-    )
-
-    for fp in file_plans:
-        target_path = (root / fp.rel_path).resolve()
-        if not target_path.is_file():
-            debug(f"[SKIP] Missing file listed in plan: {fp.rel_path}")
-            continue
-
-        debug("=" * 80)
-        debug(f"Upgrading {fp.rel_path} (priority={fp.priority}, apply={args.apply})")
-        current_source = load_text(target_path)
-        user_input = build_upgrade_file_prompt(
-            snapshot=snapshot,
-            plan=plan,
-            target_file=fp,
-            current_source=current_source,
-        )
-
-        try:
-            response_text = call_gpt_raw(
-                model=args.model,
-                instructions=system_instructions,
-                user_input=user_input,
-                reasoning_effort=args.reasoning,
-                max_output_tokens=args.max_tokens,
-            )
-        except Exception as exc:
-            debug(f"ERROR calling model for {fp.rel_path}: {exc}")
-            continue
-
-        code = extract_code_block_from_markdown(response_text)
-        if code is None:
-            debug(f"[WARN] No python code block for {fp.rel_path}; skipping file update.")
-            continue
-
-        if args.apply:
-            backup = backup_file(target_path)
-            debug(f"[{fp.rel_path}] Backup: {backup}")
-            save_text(target_path, code)
-            debug(f"[{fp.rel_path}] Updated.")
-        else:
-            debug(f"[{fp.rel_path}] Dry run only (no write).")
-
-    debug("Finished upgrade-all.")
+def run_health_check(root: Path) -> bool:
+    """
+    Run health_check_full_system.py if present.
+
+    Returns:
+        True if the script existed and exited with code 0, False otherwise.
+    """
+    script = (root / "health_check_full_system.py").resolve()
+    if not script.is_file():
+        debug("[health-check] health_check_full_system.py not found; skipping.")
+        return False
+
+    debug(f"[health-check] Running health check script: {script}")
+    try:
+        result = subprocess.run(
+            [os.sys.executable, str(script)],
+            cwd=root,
+            capture_output=True,
+            text=True,
+            check=False,
+        )
+    except Exception as exc:
+        debug(f"[health-check] ERROR invoking health check: {exc}")
+        return False
+
+    debug(f"[health-check] returncode={result.returncode}")
+    if result.stdout:
+        debug(f"[health-check][stdout]\n{result.stdout.strip()}")
+    if result.stderr:
+        debug(f"[health-check][stderr]\n{result.stderr.strip()}")
+
+    if result.returncode != 0:
+        debug("[health-check] Health check FAILED.")
+        return False
+
+    debug("[health-check] Health check passed.")
+    return True
+
+
+def cmd_upgrade_all(args: argparse.Namespace) -> None:
+    root: Path = args.root.resolve()
+    plan = load_existing_plan(root, args.plan_json)
+    snapshot = scan_repo(root)
+    system_instructions = load_master_prompt(args.master_prompt)
+
+    # Optional project-wide snapshot before any modifications
+    snapshot_path: Optional[Path] = None
+    if args.apply and getattr(args, "snapshot", False):
+        snapshot_path = create_project_snapshot(root, "upgrade-all")
+        debug(f"Project snapshot for upgrade-all: {snapshot_path}")
+
+    # Deterministic ordering: by primary phase order, then priority, then path
+    priority_rank = {"high": 0, "medium": 1, "low": 2}
+    phase_order_by_id: Dict[str, int] = {ph.id: ph.order for ph in plan.phases}
+
+    def primary_phase_order(fp: FilePlan) -> int:
+        orders = [phase_order_by_id.get(pid, 9999) for pid in fp.phase_ids]
+        return min(orders) if orders else 9999
+
+    files_with_meta = [
+        (fp, primary_phase_order(fp)) for fp in plan.files
+    ]
+    files_with_meta.sort(
+        key=lambda item: (
+            item[1],
+            priority_rank.get(item[0].priority, 1),
+            item[0].rel_path,
+        )
+    )
+
+    logs_dir = PROJECT_ROOT / "logs"
+    logs_dir.mkdir(exist_ok=True)
+
+    last_phase_order: Optional[int] = None
+
+    for fp, phase_order in files_with_meta:
+        target_path = (root / fp.rel_path).resolve()
+        if not target_path.is_file():
+            debug(f"[SKIP] Missing file listed in plan: {fp.rel_path}")
+            continue
+
+        # If we have just completed a phase (order change), optionally run health check
+        if (
+            args.apply
+            and getattr(args, "health_check", False)
+            and last_phase_order is not None
+            and last_phase_order != 9999
+            and phase_order != last_phase_order
+        ):
+            debug(f"Completed phase order {last_phase_order}; running health check.")
+            ok = run_health_check(root)
+            if not ok:
+                raise SystemExit(
+                    f"Health check failed after completing phase order {last_phase_order}; aborting further upgrades."
+                )
+
+        last_phase_order = phase_order
+
+        debug("=" * 80)
+        debug(
+            f"Upgrading {fp.rel_path} (phase_order={phase_order}, priority={fp.priority}, apply={args.apply})"
+        )
