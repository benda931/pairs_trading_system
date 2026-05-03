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


def test_validate_execution_mode_accepts_safe_defaults():
    cfg = _base_config()

    errors = validator._validate_execution_mode(cfg)

    assert errors == []


def test_validate_execution_mode_rejects_effective_live_mode():
    cfg = _base_config()
    cfg["strategy"]["dry_run"] = False
    cfg["execution"]["allow_live_orders"] = True
    cfg["execution"]["allow_agent_actions"] = True
    cfg["execution"]["paper_only"] = False
    cfg["ib_enable"] = True

    errors = validator._validate_execution_mode(cfg)

    assert any("effective dry_run must remain true" in err for err in errors)
    assert any("effective allow_live_orders must remain false" in err for err in errors)
    assert any("effective allow_agent_actions must remain false" in err for err in errors)


def test_validate_feedback_throttling_accepts_current_wiring():
    errors = validator._validate_feedback_throttling()

    assert errors == []


def test_validate_feedback_throttling_rejects_missing_persistent_guard(monkeypatch, tmp_path):
    feedback_path = tmp_path / "agent_feedback.py"
    throttler_path = tmp_path / "action_throttler.py"
    feedback_path.write_text(
        "\n".join(
            [
                "from core.action_throttler import ActionThrottler",
                "class LegacyInMemoryActionThrottler: pass",
                "self._throttle = throttler",
            ]
        ),
        encoding="utf-8",
    )
    throttler_path.write_text(
        "DEFAULT_COOLDOWNS = {'FORCE_EXIT': 300}",
        encoding="utf-8",
    )

    monkeypatch.setattr(validator, "AGENT_FEEDBACK_PATH", feedback_path)
    monkeypatch.setattr(validator, "ACTION_THROTTLER_PATH", throttler_path)

    errors = validator._validate_feedback_throttling()

    assert any("missing persistent throttler import" in err for err in errors)
    assert any("missing persistent throttler default" in err for err in errors)
    assert any("missing cooldowns" in err for err in errors)


def test_validate_allocation_wiring_accepts_current_setup():
    errors = validator._validate_allocation_wiring()

    assert errors == []


def test_validate_allocation_wiring_rejects_missing_guard(monkeypatch, tmp_path):
    orchestrator_path = tmp_path / "orchestrator.py"
    guard_path = tmp_path / "allocation_guard.py"
    orchestrator_path.write_text(
        "\n".join(
            [
                "def run_daily_pipeline():",
                "    pass",
                "def run_portfolio_allocation_cycle():",
                "    return None",
            ]
        ),
        encoding="utf-8",
    )
    guard_path.write_text(
        'def make_batch_id():\n    return "daily:20260502"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(validator, "ORCHESTRATOR_PATH", orchestrator_path)
    monkeypatch.setattr(validator, "ALLOCATION_GUARD_PATH", guard_path)

    errors = validator._validate_allocation_wiring()

    assert any("missing allocation guard import" in err for err in errors)
    assert any("missing guard check" in err for err in errors)


def test_validate_promotion_workflow_accepts_current_setup():
    errors = validator._validate_promotion_workflow()

    assert errors == []


def test_validate_promotion_workflow_rejects_missing_wf_gate(monkeypatch, tmp_path):
    promotion_path = tmp_path / "promote_pairs_to_production.py"
    selection_path = tmp_path / "select_top_pairs_from_ranked_csv.py"
    promotion_path.write_text(
        "\n".join(
            [
                "def run():",
                "    return []",
                'FLAG = "--no-crypto"',
            ]
        ),
        encoding="utf-8",
    )
    selection_path.write_text(
        "\n".join(
            [
                "def main():",
                "    return 0",
                'FLAG = "--production-mode"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(validator, "PROMOTION_SCRIPT_PATH", promotion_path)
    monkeypatch.setattr(validator, "SELECTION_SCRIPT_PATH", selection_path)

    errors = validator._validate_promotion_workflow()

    assert any("missing walk forward gate" in err for err in errors)
    assert any("selection script missing asset policy gating" in err for err in errors)


def test_validate_state_provider_wiring_accepts_current_setup():
    errors = validator._validate_state_provider_wiring()

    assert errors == []


def test_validate_state_provider_wiring_rejects_direct_streamlit(monkeypatch, tmp_path):
    provider_path = tmp_path / "state_provider.py"
    app_context_path = tmp_path / "app_context.py"
    provider_path.write_text(
        "\n".join(
            [
                "class StateProvider: pass",
                "class InMemoryStateProvider: pass",
                "class StreamlitStateProvider: pass",
                "def get_default_state_provider(): return InMemoryStateProvider()",
                "import streamlit as st",
            ]
        ),
        encoding="utf-8",
    )
    app_context_path.write_text(
        "\n".join(
            [
                "import streamlit as st",
                "def get_state_provider():",
                "    return None",
                "x = st.session_state",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(validator, "STATE_PROVIDER_PATH", provider_path)
    monkeypatch.setattr(validator, "APP_CONTEXT_PATH", app_context_path)

    errors = validator._validate_state_provider_wiring()

    assert any("must not access st.session_state directly" in err for err in errors)
    assert any("must not import streamlit directly" in err for err in errors)


def test_validate_data_freshness_wiring_accepts_current_setup():
    errors = validator._validate_data_freshness_wiring()

    assert errors == []


def test_validate_data_freshness_wiring_rejects_missing_pipeline_gate(monkeypatch, tmp_path):
    freshness_path = tmp_path / "data_freshness.py"
    orchestrator_path = tmp_path / "orchestrator.py"
    gate_test_path = tmp_path / "test_orchestrator_freshness_gate.py"

    freshness_path.write_text(
        "\n".join(
            [
                "class DataFreshnessError(RuntimeError): pass",
                "class FreshnessConfig: pass",
                "def validate_price_frame(*args, **kwargs): return {}",
                "def validate_pair_frames(*args, **kwargs): return {}",
                "def load_price_data_guarded(*args, **kwargs): return None",
            ]
        ),
        encoding="utf-8",
    )
    orchestrator_path.write_text(
        "\n".join(
            [
                "from common.data_freshness import FreshnessConfig",
                'order = ["health_check", "data_refresh", "compute_signals", "risk_check"]',
            ]
        ),
        encoding="utf-8",
    )
    gate_test_path.write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(validator, "DATA_FRESHNESS_PATH", freshness_path)
    monkeypatch.setattr(validator, "ORCHESTRATOR_PATH", orchestrator_path)
    monkeypatch.setattr(validator, "FRESHNESS_GATE_TEST_PATH", gate_test_path)

    errors = validator._validate_data_freshness_wiring()

    assert any("missing freshness import" in err or "missing freshness task" in err for err in errors)
    assert any("missing failed blocks compute" in err for err in errors)
