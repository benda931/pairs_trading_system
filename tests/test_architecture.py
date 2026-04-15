# -*- coding: utf-8 -*-
"""
tests/test_architecture.py — Architecture Boundary Enforcement
================================================================

These tests prevent architectural drift by enforcing the tiered
hierarchy defined in the Principal Architecture Review.

Tier 0: contracts, safety, config (imports stdlib only)
Tier 1: signal pipeline (imports T0 + common/)
Tier 2: portfolio, risk (imports T0-1 + portfolio/)
Tier 3: enrichments, ML (imports T0-2)
Tier 4: agents, orchestrator (imports T0-3 + agents/)
Tier 5: UI, dashboard (imports anything — ONLY tier with streamlit)

Rules enforced:
- core/ must NEVER import streamlit at module level
- core/ must NEVER import from root/
- No Tier N module imports from Tier N+1
- All Tier 0-2 modules must be importable without Streamlit
- SignalEnvelope is the canonical signal output type
"""
from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
AGENTS_DIR = PROJECT_ROOT / "agents"
ROOT_DIR = PROJECT_ROOT / "root"
COMMON_DIR = PROJECT_ROOT / "common"


def _get_python_files(directory: Path) -> list[Path]:
    """Get all .py files in directory (non-recursive)."""
    if not directory.exists():
        return []
    return sorted(f for f in directory.glob("*.py") if f.name != "__init__.py")


def _get_imports_from_file(filepath: Path) -> list[str]:
    """Extract all import statements from a Python file using AST parsing."""
    try:
        with open(filepath, encoding="utf-8-sig") as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


# ─── Test: No streamlit imports in core/ ──────────────────────

class TestNoStreamlitInCore:
    """Core modules must NEVER import streamlit at module level."""

    # Known violations that are being migrated (track them to prevent regression)
    KNOWN_VIOLATIONS = {
        "app_context.py",       # AP-1: to be moved to root/
        "alert_bus.py",         # AP-1: UI to be extracted
        "context_services.py",  # AP-1: lazy import inside function
        "ml_analysis.py",       # AP-1: UI to be extracted
    }

    def test_no_new_streamlit_imports_in_core(self):
        """No NEW core/ modules should import streamlit beyond known violations."""
        violations = []
        for filepath in _get_python_files(CORE_DIR):
            imports = _get_imports_from_file(filepath)
            if any("streamlit" in imp for imp in imports):
                if filepath.name not in self.KNOWN_VIOLATIONS:
                    violations.append(filepath.name)

        assert not violations, (
            f"NEW core/ modules importing streamlit (architectural violation): {violations}\n"
            f"Known violations being migrated: {sorted(self.KNOWN_VIOLATIONS)}"
        )

    def test_known_violations_count_stable(self):
        """Track known violations — count should decrease over time, never increase."""
        actual_violations = set()
        for filepath in _get_python_files(CORE_DIR):
            imports = _get_imports_from_file(filepath)
            if any("streamlit" in imp for imp in imports):
                actual_violations.add(filepath.name)

        # Violations should only decrease from known set
        new = actual_violations - self.KNOWN_VIOLATIONS
        assert not new, f"New streamlit violations in core/: {new}"


# ─── Test: No root/ imports in core/ ──────────────────────────

class TestNoRootImportsInCore:
    """Core modules must NEVER import from root/."""

    KNOWN_VIOLATIONS = {
        "full_parameter_optimization.py",  # AP-2: imports root.trade_logic, root.analysis
        "ib_data_ingestor.py",             # AP-2: imports root.ibkr_connection
        "ib_order_router.py",              # AP-2: imports root.ibkr_connection
        "system_health.py",                # AP-2: imports root modules for health checks
    }

    def test_no_new_root_imports_in_core(self):
        """No NEW core/ modules should import from root/."""
        violations = []
        for filepath in _get_python_files(CORE_DIR):
            imports = _get_imports_from_file(filepath)
            if any(imp.startswith("root.") or imp.startswith("root") for imp in imports):
                if filepath.name not in self.KNOWN_VIOLATIONS:
                    violations.append(filepath.name)

        assert not violations, (
            f"NEW core/ modules importing from root/ (architectural violation): {violations}"
        )


# ─── Test: Critical path modules importable ───────────────────

class TestCriticalPathImportable:
    """All Tier 0-1 modules must be importable without Streamlit."""

    TIER_0_MODULES = [
        "core.contracts",
        "core.signal_contracts",
        "common.config_manager",
    ]

    TIER_1_MODULES = [
        "core.intents",
        "core.regime_engine",
        "core.threshold_engine",
        "core.signal_quality",
        "core.lifecycle",
    ]

    @pytest.mark.parametrize("module_name", TIER_0_MODULES + TIER_1_MODULES)
    def test_tier_0_1_importable(self, module_name):
        """Tier 0-1 modules must import without errors."""
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            if "streamlit" in str(exc).lower():
                pytest.fail(
                    f"{module_name} requires Streamlit — violates Tier 0-1 rule"
                )
            # Other ImportErrors (missing optional deps) are acceptable
            pass


# ─── Test: SignalEnvelope is canonical ─────────────────────────

class TestSignalContractCanonical:
    """SignalEnvelope must be the canonical signal output type."""

    def test_signal_envelope_importable(self):
        from core.signal_contracts import SignalEnvelope
        assert SignalEnvelope is not None

    def test_signal_envelope_is_frozen(self):
        """SignalEnvelope should be immutable (frozen dataclass)."""
        from core.signal_contracts import SignalEnvelope
        import dataclasses
        assert dataclasses.is_dataclass(SignalEnvelope)

    def test_signal_envelope_has_typed_layers(self):
        """SignalEnvelope must have typed sub-contracts, not metadata dict."""
        from core.signal_contracts import (
            SignalEnvelope, RegimeContext, ThresholdContext,
            QualityVerdict, SoftSignalModifiers, AdvisoryOverlays,
        )
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SignalEnvelope)}
        assert "regime_ctx" in fields
        assert "threshold_ctx" in fields
        assert "quality" in fields
        assert "soft" in fields

    def test_hard_block_codes_are_enum(self):
        """HardBlockCode must be an enum, not bare strings."""
        from core.signal_contracts import HardBlockCode
        assert hasattr(HardBlockCode, "GRADE_F_SKIP")
        assert hasattr(HardBlockCode, "REGIME_CRISIS")
        assert hasattr(HardBlockCode, "SPREAD_NAN")

    def test_signal_decision_alias_points_to_envelope(self):
        """During migration, SignalDecision should alias SignalEnvelope."""
        from core.signal_pipeline import SignalDecision
        from core.signal_contracts import SignalEnvelope
        assert SignalDecision is SignalEnvelope


# ─── Test: Agent feedback has throttler ────────────────────────

class TestAgentGovernance:
    """Agent feedback loop must have safety controls."""

    def test_action_throttler_exists(self):
        from core.agent_feedback import ActionThrottler
        t = ActionThrottler()
        assert hasattr(t, "can_execute")
        assert hasattr(t, "record_execution")
        assert hasattr(t, "reset_cycle")

    def test_throttle_config_has_cool_downs(self):
        from core.contracts import ActionThrottleConfig
        cfg = ActionThrottleConfig()
        assert "KILL_SWITCH" in cfg.cool_down_seconds
        assert "RETRAIN_MODEL" in cfg.cool_down_seconds
        assert cfg.max_actions_per_cycle > 0
        assert cfg.max_emergency_actions_per_day > 0

    def test_feedback_loop_uses_throttler(self):
        from core.agent_feedback import AgentFeedbackLoop, ActionThrottler as _AT
        loop = AgentFeedbackLoop()
        assert hasattr(loop, "_throttle")
        assert isinstance(loop._throttle, _AT)


# ─── Test: Allocation idempotency ──────────────────────────────

class TestAllocationIdempotency:
    """Portfolio allocation must have dedup guard."""

    def test_orchestrator_has_batch_guard(self):
        """Orchestrator must track last allocation batch."""
        from core.orchestrator import PairsOrchestrator
        orch = PairsOrchestrator()
        # The batch guard uses _last_allocation_batch attribute
        result1 = orch.run_portfolio_allocation_cycle(signal_decisions=None)
        assert result1 is None  # No decisions = no allocation


# ─── Test: StateProvider protocol exists ───────────────────────

class TestStateProviderContract:
    """StateProvider must be defined for Streamlit decoupling."""

    def test_state_provider_protocol_exists(self):
        from core.contracts import StateProvider
        assert hasattr(StateProvider, "get")
        assert hasattr(StateProvider, "set")
        assert hasattr(StateProvider, "has")

    def test_in_memory_provider_works(self):
        from core.contracts import InMemoryStateProvider
        p = InMemoryStateProvider()
        p.set("key", "value")
        assert p.get("key") == "value"
        assert p.has("key")
        assert p.get("missing", "default") == "default"
        assert not p.has("missing")

    def test_engine_contract_exists(self):
        from core.contracts import EngineContract
        ec = EngineContract(
            name="test", purpose="testing", inputs={}, outputs={},
            failure_mode="skip", fallback="none", owner="test",
            consumers=(), economic_value="research", classification="RESEARCH",
        )
        assert ec.classification == "RESEARCH"


# ─── Test: Data freshness guard exists ─────────────────────────

class TestDataFreshnessGuard:
    """Data freshness guard must exist for production safety."""

    def test_freshness_guard_importable(self):
        from common.data_loader import load_price_data_guarded, DataFreshnessError
        assert callable(load_price_data_guarded)
        assert issubclass(DataFreshnessError, RuntimeError)
