from __future__ import annotations

from pathlib import Path

from scripts import validate_production_safety as validator

_LEGACY_BUG = "datetime.now(timezone.utc)" + "()"


def _base_config() -> dict:
    return {
        "strategy": {"dry_run": True},
        "execution": {
            "dry_run": True,
            "allow_live_orders": False,
            "allow_agent_actions": False,
            "paper_only": True,
        },
        "use_production_pairs": True,
        "production_pairs": ["SPY/QQQ"],
        "asset_policy": {
            "allow_crypto": False,
            "enforce_etf_like_in_production": True,
            "etf_like_symbols": ["SPY", "QQQ"],
        },
    }


def test_find_datetime_callable_bug_detects_real_match_and_skips_logs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "bad.py").write_text(f"x = {_LEGACY_BUG}\n", encoding="utf-8")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "ignored.py").write_text(f"x = {_LEGACY_BUG}\n", encoding="utf-8")

    matches = validator._find_datetime_callable_bug(tmp_path)

    assert any("src" in match and "bad.py" in match for match in matches)
    assert not any("logs" in match for match in matches)


def test_find_top_level_streamlit_imports_flags_only_top_level(tmp_path):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "bad.py").write_text("import streamlit as st\n", encoding="utf-8")
    (core_dir / "ok.py").write_text("def f():\n    import streamlit as st\n", encoding="utf-8")

    original_repo_root = validator.REPO_ROOT
    validator.REPO_ROOT = tmp_path
    try:
        matches = validator._find_top_level_streamlit_imports(core_dir)
    finally:
        validator.REPO_ROOT = original_repo_root

    assert any("bad.py" in match for match in matches)
    assert not any("ok.py" in match for match in matches)


def test_validate_config_rejects_unsafe_defaults():
    cfg = _base_config()
    cfg["execution"]["allow_live_orders"] = True
    cfg["production_pairs"] = ["IBIT/ETHA"]

    errors = validator._validate_config(cfg)

    assert any("execution.allow_live_orders must be false" in err for err in errors)
    assert any("contains blocked crypto symbol IBIT" in err for err in errors)


def test_validate_pair_policy_rejects_non_etf_like_production_pair():
    cfg = _base_config()
    cfg["production_pairs"] = ["SPY/AAPL"]

    errors = validator._validate_pair_policy(cfg)

    assert any("non-ETF-like symbols AAPL" in err for err in errors)


def test_validate_pair_policy_accepts_valid_production_pair():
    cfg = _base_config()

    errors = validator._validate_pair_policy(cfg)

    assert errors == []


def test_validate_ci_workflow_accepts_expected_gates(tmp_path):
    workflow = tmp_path / "ci.yml"
    workflow.write_text(
        "\n".join(
            [
                "jobs:",
                "  compile:",
                "    steps:",
                "      - run: python -m compileall .",
                "  production-safety-tests:",
                "    steps:",
                "      - run: python scripts/validate_production_safety.py",
                "      - run: python -m pytest tests/test_orchestrator_contract.py tests/test_validate_production_safety.py -q",
            ]
        ),
        encoding="utf-8",
    )

    errors = validator._validate_ci_workflow(workflow)

    assert errors == []


def test_validate_ci_workflow_rejects_missing_validator_gate(tmp_path):
    workflow = tmp_path / "ci.yml"
    workflow.write_text(
        "\n".join(
            [
                "jobs:",
                "  compile:",
                "    steps:",
                "      - run: python -m compileall .",
                "  production-safety-tests:",
                "    steps:",
                "      - run: python -m pytest tests/test_pair_utils.py -q",
            ]
        ),
        encoding="utf-8",
    )

    errors = validator._validate_ci_workflow(workflow)

    assert any("missing validator command" in err for err in errors)


def test_validate_production_pair_runtime_accepts_consistent_config(monkeypatch):
    cfg = _base_config()
    cfg["production_pairs"] = ["SPY/QQQ", "QQQ/XLK"]

    monkeypatch.setattr(
        "core.orchestrator._get_configured_pairs",
        lambda supplied_cfg=None: [("SPY", "QQQ"), ("QQQ", "XLK")],
    )

    errors = validator._validate_production_pair_runtime(cfg)

    assert errors == []


def test_validate_production_pair_runtime_rejects_duplicate_unordered_pairs(monkeypatch):
    cfg = _base_config()
    cfg["production_pairs"] = ["SPY/QQQ", "QQQ/SPY"]

    monkeypatch.setattr(
        "core.orchestrator._get_configured_pairs",
        lambda supplied_cfg=None: [("SPY", "QQQ")],
    )

    errors = validator._validate_production_pair_runtime(cfg)

    assert any("invalid or duplicate unordered pairs" in err for err in errors)
