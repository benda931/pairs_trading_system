from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.json"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "python-package-conda.yml"
AGENT_FEEDBACK_PATH = REPO_ROOT / "core" / "agent_feedback.py"
ACTION_THROTTLER_PATH = REPO_ROOT / "core" / "action_throttler.py"
ORCHESTRATOR_PATH = REPO_ROOT / "core" / "orchestrator.py"
ALLOCATION_GUARD_PATH = REPO_ROOT / "core" / "allocation_guard.py"
CONFIG_MANAGER_PATH = REPO_ROOT / "common" / "config_manager.py"
PROMOTION_SCRIPT_PATH = REPO_ROOT / "scripts" / "promote_pairs_to_production.py"
SELECTION_SCRIPT_PATH = REPO_ROOT / "scripts" / "select_top_pairs_from_ranked_csv.py"
STATE_PROVIDER_PATH = REPO_ROOT / "core" / "state_provider.py"
APP_CONTEXT_PATH = REPO_ROOT / "core" / "app_context.py"
DATA_FRESHNESS_PATH = REPO_ROOT / "common" / "data_freshness.py"
FRESHNESS_GATE_TEST_PATH = REPO_ROOT / "tests" / "test_orchestrator_freshness_gate.py"
NEEDLE = "datetime.now(timezone.utc)" + "()"
EXCLUDED_DIRS = {".git", "__pycache__", ".venv", "venv", "logs"}
BLOCKED_CRYPTO_SYMBOLS = {
    "IBIT",
    "ETHA",
    "BITO",
    "BLOK",
    "BKCH",
    "WGMI",
    "MSTR",
    "COIN",
    "RIOT",
    "MARA",
    "GBTC",
    "ETHE",
    "BTC",
    "ETH",
    "BTC-USD",
    "ETH-USD",
}
REQUIRED_THROTTLED_ACTIONS = {
    "KILL_SWITCH",
    "DELEVERAGE",
    "FORCE_EXIT",
    "BLOCK_ENTRY",
    "RETRAIN_MODEL",
    "OPTIMIZE_PARAMS",
    "UPDATE_CONFIG",
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.pair_utils import load_asset_policy, normalize_pairs, pair_allowed_by_policy, parse_pair_record
from core.execution_safety import get_execution_mode


def _iter_repo_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        yield path


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_datetime_callable_bug(root: Path) -> list[str]:
    matches: list[str] = []
    for path in _iter_repo_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if NEEDLE not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if NEEDLE in line:
                matches.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    return matches


def _find_top_level_streamlit_imports(core_root: Path) -> list[str]:
    matches: list[str] = []
    for path in core_root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            stripped = line.lstrip()
            if stripped != line:
                continue
            if line.startswith("import streamlit") or line.startswith("from streamlit"):
                matches.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    return matches


def _validate_config(cfg: dict) -> list[str]:
    errors: list[str] = []
    strategy = cfg.get("strategy", {})
    execution = cfg.get("execution", {})
    asset_policy = cfg.get("asset_policy", {})
    production_pairs = cfg.get("production_pairs", [])

    if strategy.get("dry_run") is not True:
        errors.append("config.json: strategy.dry_run must be true")
    if execution.get("dry_run") is not True:
        errors.append("config.json: execution.dry_run must be true")
    if execution.get("allow_live_orders") is not False:
        errors.append("config.json: execution.allow_live_orders must be false")
    if execution.get("allow_agent_actions") is not False:
        errors.append("config.json: execution.allow_agent_actions must be false")
    if execution.get("paper_only") is not True:
        errors.append("config.json: execution.paper_only must be true")
    if asset_policy.get("allow_crypto") is not False:
        errors.append("config.json: asset_policy.allow_crypto must be false")
    if cfg.get("use_production_pairs") is not True:
        errors.append("config.json: use_production_pairs must be true")
    if cfg.get("use_production_pairs") is True and asset_policy.get("enforce_etf_like_in_production") is not True:
        errors.append(
            "config.json: asset_policy.enforce_etf_like_in_production must be true when use_production_pairs is enabled"
        )
    if not isinstance(production_pairs, list) or not production_pairs:
        errors.append("config.json: production_pairs must exist and be non-empty")
    else:
        for pair in production_pairs:
            parsed = parse_pair_record(pair)
            if not parsed:
                errors.append(f"config.json: invalid production pair {pair!r}")
                continue
            for symbol in parsed:
                if symbol in BLOCKED_CRYPTO_SYMBOLS:
                    errors.append(
                        f"config.json: production pair {pair!r} contains blocked crypto symbol {symbol}"
                    )
    return errors


def _validate_pair_policy(cfg: dict) -> list[str]:
    errors: list[str] = []
    asset_policy = load_asset_policy(cfg)

    if parse_pair_record("SPY/QQQ") != ("SPY", "QQQ"):
        errors.append('pair_utils: parse_pair_record("SPY/QQQ") must equal ("SPY", "QQQ")')
    if parse_pair_record("SPY-QQQ") != ("SPY", "QQQ"):
        errors.append('pair_utils: parse_pair_record("SPY-QQQ") must equal ("SPY", "QQQ")')
    if pair_allowed_by_policy("IBIT", "ETHA", policy=asset_policy) is not False:
        errors.append("pair_utils: IBIT/ETHA must be blocked by asset policy")
    if pair_allowed_by_policy("SPY", "QQQ", policy=asset_policy) is not True:
        errors.append("pair_utils: SPY/QQQ must be allowed by asset policy")

    if asset_policy.get("enforce_etf_like_in_production"):
        etf_like = {str(s).upper() for s in asset_policy.get("etf_like_symbols", [])}
        for pair in cfg.get("production_pairs", []):
            parsed = parse_pair_record(pair)
            if not parsed:
                continue
            missing = [symbol for symbol in parsed if symbol not in etf_like]
            if missing:
                errors.append(
                    f"asset_policy: production pair {pair!r} has non-ETF-like symbols {', '.join(missing)}"
                )
    return errors


def _validate_orchestrator_contract() -> list[str]:
    errors: list[str] = []
    from core.orchestrator import PairsOrchestrator

    original_reconciliation = PairsOrchestrator._run_startup_reconciliation
    original_health = PairsOrchestrator._run_startup_health_check
    try:
        PairsOrchestrator._run_startup_reconciliation = lambda self: None
        PairsOrchestrator._run_startup_health_check = lambda self: None
        contract = PairsOrchestrator().validate_pipeline_contract()
    finally:
        PairsOrchestrator._run_startup_reconciliation = original_reconciliation
        PairsOrchestrator._run_startup_health_check = original_health

    if not contract.get("ok", False):
        for err in contract.get("errors", []):
            errors.append(f"orchestrator contract: {err}")
    return errors


def _validate_production_pair_runtime(cfg: dict) -> list[str]:
    errors: list[str] = []
    production_pairs = list(cfg.get("production_pairs", []) or [])
    if not production_pairs:
        return errors

    normalized_pairs = normalize_pairs(production_pairs, dedupe=True)
    if len(normalized_pairs) != len(production_pairs):
        errors.append("config.json: production_pairs contains invalid or duplicate unordered pairs")

    try:
        from core.orchestrator import _get_configured_pairs

        resolved_pairs = _get_configured_pairs(cfg)
    except Exception as exc:
        return [f"production pair runtime: unable to resolve configured pairs: {exc}"]

    if resolved_pairs != normalized_pairs:
        errors.append(
            "production pair runtime: _get_configured_pairs(config) does not match normalized production_pairs"
        )
    if cfg.get("use_production_pairs") is True and not resolved_pairs:
        errors.append("production pair runtime: use_production_pairs is enabled but no active production pairs resolved")
    return errors


def _validate_execution_mode(cfg: dict) -> list[str]:
    errors: list[str] = []
    mode = get_execution_mode(cfg)

    if mode.get("dry_run") is not True:
        errors.append("execution mode: effective dry_run must remain true")
    if mode.get("allow_live_orders") is not False:
        errors.append("execution mode: effective allow_live_orders must remain false")
    if mode.get("allow_agent_actions") is not False:
        errors.append("execution mode: effective allow_agent_actions must remain false")
    if mode.get("paper_only") is not True:
        errors.append("execution mode: effective paper_only must remain true")
    return errors


def _validate_feedback_throttling() -> list[str]:
    errors: list[str] = []

    try:
        feedback_text = AGENT_FEEDBACK_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"feedback throttling: unable to read {AGENT_FEEDBACK_PATH}: {exc}"]

    try:
        throttler_text = ACTION_THROTTLER_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"feedback throttling: unable to read {ACTION_THROTTLER_PATH}: {exc}"]

    required_feedback_snippets = {
        "persistent throttler import": "from core.action_throttler import ActionThrottler as PersistentActionThrottler",
        "legacy throttler renamed": "class LegacyInMemoryActionThrottler",
        "persistent throttler default": "self._throttle = throttler or PersistentActionThrottler()",
        "summary throttled count": "n_actions_throttled",
        "summary throttled actions": "throttled_actions",
    }
    for label, snippet in required_feedback_snippets.items():
        if snippet not in feedback_text:
            errors.append(f"feedback throttling: missing {label} ({snippet})")

    if "class ActionThrottler" in feedback_text:
        errors.append(
            "feedback throttling: legacy local ActionThrottler class must not exist in core/agent_feedback.py"
        )

    missing_action_types = [
        action_type
        for action_type in sorted(REQUIRED_THROTTLED_ACTIONS)
        if f'"{action_type}":' not in throttler_text and f"'{action_type}':" not in throttler_text
    ]
    if missing_action_types:
        errors.append(
            "feedback throttling: core/action_throttler.py missing cooldowns for "
            + ", ".join(missing_action_types)
        )

    return errors


def _validate_allocation_wiring() -> list[str]:
    errors: list[str] = []

    try:
        orchestrator_text = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"allocation wiring: unable to read {ORCHESTRATOR_PATH}: {exc}"]

    try:
        guard_text = ALLOCATION_GUARD_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"allocation wiring: unable to read {ALLOCATION_GUARD_PATH}: {exc}"]

    required_orchestrator_snippets = {
        "batch id passed from daily pipeline": 'allocation_batch_id=f"daily:{date.today():%Y%m%d}"',
        "allocation guard import": "from core.allocation_guard import AllocationBatchGuard",
        "guard constructed": "guard = AllocationBatchGuard()",
        "guard check": "guard.check_and_start(",
        "already processed skip": 'message="allocation_batch_already_processed"',
        "mark completed": "guard.mark_completed(",
        "mark failed": "guard.mark_failed(",
    }
    for label, snippet in required_orchestrator_snippets.items():
        if snippet not in orchestrator_text:
            errors.append(f"allocation wiring: missing {label} ({snippet})")

    required_guard_snippets = {
        "batch id factory": 'return f"{strategy}:{trading_day:%Y%m%d}"',
        "failed status recorded": 'batch["status"] = "failed"',
        "completed status recorded": 'batch["status"] = "completed"',
        "started status recorded": '"status": "started"',
    }
    for label, snippet in required_guard_snippets.items():
        if snippet not in guard_text:
            errors.append(f"allocation wiring: missing {label} in core/allocation_guard.py ({snippet})")

    return errors


def _validate_promotion_workflow() -> list[str]:
    errors: list[str] = []

    try:
        promotion_text = PROMOTION_SCRIPT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"promotion workflow: unable to read {PROMOTION_SCRIPT_PATH}: {exc}"]

    try:
        selection_text = SELECTION_SCRIPT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"promotion workflow: unable to read {SELECTION_SCRIPT_PATH}: {exc}"]

    required_promotion_snippets = {
        "asset policy load": "load_asset_policy",
        "asset policy gating": "pair_allowed_by_policy",
        "walk forward gate": "run_walk_forward",
        "enforce etf flag": '--enforce-etf-like',
        "no crypto flag": '--no-crypto',
        "require viable flag": '--require-viable',
        "config mutation": "mutate_config(",
        "config backup": "backup=True",
        "unordered dedupe": 'canonical_pair_id(*sorted(',
        "wf dsr rejection": "wf_dsr_gate_failed",
        "wf overfit rejection": "wf_prob_overfit_above_threshold",
    }
    for label, snippet in required_promotion_snippets.items():
        if snippet not in promotion_text:
            errors.append(f"promotion workflow: missing {label} ({snippet})")

    required_selection_snippets = {
        "production mode flag": '--production-mode',
        "no crypto flag": '--no-crypto',
        "enforce etf flag": '--enforce-etf-like',
        "require viable flag": '--require-viable',
        "asset policy gating": "pair_allowed_by_policy",
    }
    for label, snippet in required_selection_snippets.items():
        if snippet not in selection_text:
            errors.append(f"promotion workflow: selection script missing {label} ({snippet})")

    return errors


def _validate_state_provider_wiring() -> list[str]:
    errors: list[str] = []

    try:
        provider_text = STATE_PROVIDER_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"state provider wiring: unable to read {STATE_PROVIDER_PATH}: {exc}"]

    try:
        app_context_text = APP_CONTEXT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"state provider wiring: unable to read {APP_CONTEXT_PATH}: {exc}"]

    required_provider_snippets = {
        "state provider protocol": "class StateProvider(Protocol):",
        "in-memory provider": "class InMemoryStateProvider:",
        "streamlit provider": "class StreamlitStateProvider:",
        "lazy streamlit import": "import streamlit as st",
        "default provider resolver": "def get_default_state_provider() -> StateProvider:",
    }
    for label, snippet in required_provider_snippets.items():
        if snippet not in provider_text:
            errors.append(f"state provider wiring: missing {label} ({snippet})")

    required_app_context_snippets = {
        "provider import": "from core.state_provider import StateProvider, get_default_state_provider",
        "provider getter": "def get_state_provider() -> StateProvider:",
        "state getter": "def _state_get(",
        "state setter": "def _state_set(",
    }
    for label, snippet in required_app_context_snippets.items():
        if snippet not in app_context_text:
            errors.append(f"state provider wiring: missing {label} in core/app_context.py ({snippet})")

    try:
        app_context_ast = ast.parse(app_context_text, filename=str(APP_CONTEXT_PATH))
    except SyntaxError as exc:
        errors.append(f"state provider wiring: unable to parse core/app_context.py: {exc}")
        return errors

    direct_session_state_lines: list[str] = []
    direct_streamlit_import_lines: list[str] = []
    for node in ast.walk(app_context_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "streamlit":
                    direct_streamlit_import_lines.append("import streamlit")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "streamlit":
                direct_streamlit_import_lines.append(f"from {node.module} import ...")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "session_state"
            and isinstance(node.value, ast.Name)
            and node.value.id == "st"
        ):
            direct_session_state_lines.append(f"line {getattr(node, 'lineno', '?')}")

    if direct_session_state_lines:
        errors.append("state provider wiring: core/app_context.py must not access st.session_state directly")
    if direct_streamlit_import_lines:
        errors.append("state provider wiring: core/app_context.py must not import streamlit directly")

    return errors


def _validate_data_freshness_wiring() -> list[str]:
    errors: list[str] = []

    try:
        freshness_text = DATA_FRESHNESS_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"data freshness wiring: unable to read {DATA_FRESHNESS_PATH}: {exc}"]

    try:
        orchestrator_text = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"data freshness wiring: unable to read {ORCHESTRATOR_PATH}: {exc}"]

    try:
        gate_test_text = FRESHNESS_GATE_TEST_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"data freshness wiring: unable to read {FRESHNESS_GATE_TEST_PATH}: {exc}"]

    required_freshness_snippets = {
        "freshness error": "class DataFreshnessError(RuntimeError):",
        "freshness config": "class FreshnessConfig:",
        "frame validator": "def validate_price_frame(",
        "pair validator": "def validate_pair_frames(",
        "guarded loader": "def load_price_data_guarded(",
        "all stale reason": '"reason"] = "stale_data"',
        "leg mismatch reason": '"reason"] = "leg_date_mismatch"',
    }
    for label, snippet in required_freshness_snippets.items():
        if snippet not in freshness_text:
            errors.append(f"data freshness wiring: missing {label} in common/data_freshness.py ({snippet})")

    required_orchestrator_snippets = {
        "freshness import": "from common.data_freshness import FreshnessConfig, validate_pair_frames",
        "freshness task": 'name="data_freshness_check"',
        "freshness dependency": 'depends_on=["data_refresh"]',
        "compute dependency": 'depends_on=["data_freshness_check"]',
        "failed reason": 'reason = "all_pairs_stale_or_invalid"',
        "fresh pairs override reset": "self._fresh_pairs_override = None",
        "fresh pairs override update": "orchestrator._fresh_pairs_override = list(passed_pairs)",
        "freshness bus publish": 'orchestrator.bus.publish("data_freshness", payload)',
        "compute override use": 'getattr(orchestrator, "_fresh_pairs_override", None) is not None',
        "pipeline order": 'order = ["health_check", "data_refresh", "data_freshness_check", "compute_signals", "risk_check"]',
    }
    for label, snippet in required_orchestrator_snippets.items():
        if snippet not in orchestrator_text:
            errors.append(f"data freshness wiring: missing {label} in core/orchestrator.py ({snippet})")

    required_gate_test_snippets = {
        "failed blocks compute": "test_data_freshness_check_fails_and_blocks_compute_signals",
        "all stale blocks allocation": "test_run_daily_pipeline_all_pairs_stale_skips_compute_signals_and_allocation",
        "partial pass uses only passed pairs": "test_run_daily_pipeline_partial_freshness_uses_only_passed_pairs",
        "all pass continues": "test_run_daily_pipeline_all_pass_continues_normally",
    }
    for label, snippet in required_gate_test_snippets.items():
        if snippet not in gate_test_text:
            errors.append(f"data freshness wiring: missing {label} coverage in tests/test_orchestrator_freshness_gate.py ({snippet})")

    return errors


def _validate_config_governance_wiring() -> list[str]:
    errors: list[str] = []

    try:
        config_manager_text = CONFIG_MANAGER_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"config governance wiring: unable to read {CONFIG_MANAGER_PATH}: {exc}"]

    try:
        promotion_text = PROMOTION_SCRIPT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"config governance wiring: unable to read {PROMOTION_SCRIPT_PATH}: {exc}"]

    required_config_manager_snippets = {
        "repo-root project path": "PROJECT_ROOT = Path(__file__).resolve().parent.parent",
        "repo-root config path": 'CONFIG_PATH = PROJECT_ROOT / "config.json"',
        "repo-root config dir": 'CONFIG_DIR = PROJECT_ROOT / "configs"',
        "config path resolver": "def resolve_config_path(",
        "config backup helper": "def backup_config(",
        "config mutate helper": "def mutate_config(",
    }
    for label, snippet in required_config_manager_snippets.items():
        if snippet not in config_manager_text:
            errors.append(f"config governance wiring: missing {label} in common/config_manager.py ({snippet})")

    required_promotion_snippets = {
        "mutate config usage": "mutate_config(",
        "promotion backup enabled": "backup=True",
        "promotion backup label": 'backup_label="pre_production_update"',
    }
    for label, snippet in required_promotion_snippets.items():
        if snippet not in promotion_text:
            errors.append(f"config governance wiring: promotion script missing {label} ({snippet})")

    if 'CONFIG_PATH = PROJECT_ROOT / "common" / "config.json"' in config_manager_text:
        errors.append("config governance wiring: config manager must not point at common/config.json")

    return errors


def _validate_ci_workflow(workflow_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        workflow_text = workflow_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"ci workflow: unable to read {workflow_path}: {exc}"]

    required_snippets = {
        "compile job": "compile:",
        "production safety job": "production-safety-tests:",
        "compileall command": "python -m compileall .",
        "validator command": "python scripts/validate_production_safety.py",
        "pair utils test": "tests/test_pair_utils.py",
        "config manager paths test": "tests/test_config_manager_paths.py",
        "config manager mutation test": "tests/test_config_manager_mutation.py",
        "orchestrator freshness gate test": "tests/test_orchestrator_freshness_gate.py",
        "orchestrator contract test": "tests/test_orchestrator_contract.py",
        "allocation guard test": "tests/test_allocation_guard.py",
        "execution safety test": "tests/test_execution_safety.py",
        "feedback dry run test": "tests/test_feedback_dry_run.py",
        "promotion workflow test": "tests/test_promote_pairs_to_production.py",
        "state provider test": "tests/test_state_provider.py",
        "agent feedback test": "tests/test_agent_feedback.py",
        "validator test": "tests/test_validate_production_safety.py",
    }
    for label, snippet in required_snippets.items():
        if snippet not in workflow_text:
            errors.append(f"ci workflow: missing {label} ({snippet})")
    return errors


def main() -> int:
    failures: list[str] = []

    cfg = _load_config()

    datetime_matches = _find_datetime_callable_bug(REPO_ROOT)
    if datetime_matches:
        failures.append("Found legacy datetime callable bug:")
        failures.extend(f"  - {match}" for match in datetime_matches)

    streamlit_matches = _find_top_level_streamlit_imports(REPO_ROOT / "core")
    if streamlit_matches:
        failures.append("Found top-level Streamlit imports under core/:")
        failures.extend(f"  - {match}" for match in streamlit_matches)

    failures.extend(_validate_config(cfg))
    failures.extend(_validate_pair_policy(cfg))
    failures.extend(_validate_orchestrator_contract())
    failures.extend(_validate_production_pair_runtime(cfg))
    failures.extend(_validate_execution_mode(cfg))
    failures.extend(_validate_feedback_throttling())
    failures.extend(_validate_allocation_wiring())
    failures.extend(_validate_promotion_workflow())
    failures.extend(_validate_state_provider_wiring())
    failures.extend(_validate_data_freshness_wiring())
    failures.extend(_validate_config_governance_wiring())
    failures.extend(_validate_ci_workflow(WORKFLOW_PATH))

    if failures:
        print("PRODUCTION SAFETY VALIDATION: FAIL")
        for failure in failures:
            print(failure)
        return 1

    print("PRODUCTION SAFETY VALIDATION: PASS")
    print("- no legacy datetime callable bug found")
    print("- no top-level Streamlit imports found under core/")
    print("- config.json safety defaults validated")
    print("- pair parser and policy sanity checks validated")
    print("- orchestrator pipeline contract validated")
    print("- production pair runtime resolution validated")
    print("- effective execution mode validated")
    print("- feedback throttling wiring validated")
    print("- allocation idempotency wiring validated")
    print("- promotion workflow gates validated")
    print("- state-provider decoupling wiring validated")
    print("- data freshness gating wiring validated")
    print("- config governance wiring validated")
    print("- CI workflow production-safety gates validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
