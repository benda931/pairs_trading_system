from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.json"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "python-package-conda.yml"
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

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.pair_utils import load_asset_policy, normalize_pairs, pair_allowed_by_policy, parse_pair_record


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
        "orchestrator contract test": "tests/test_orchestrator_contract.py",
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
    print("- CI workflow production-safety gates validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
