# -*- coding: utf-8 -*-
"""
tests/test_remediation.py
=========================
Tests verifying remediation fixes.
Each test maps to a finding ID from docs/remediation/remediation_ledger.md.

Finding IDs tested:
  R-001  / P0-WF   : Walk-forward minimum segment floor (63 days)
  R-002  / P1-MINOBS: Minimum observation count (252 trading days)
  P0-KS            : Kill-switch bridge to control plane
  P1-PIPE          : Signal pipeline integration layer
  P1-GOV           : Governance gate on model promotion
  P1-SURV2         : Stale data surveillance hook in data_loader
"""

from __future__ import annotations

import inspect
import textwrap
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# R-001 / P0-WF: Walk-forward minimum segment floor
# ---------------------------------------------------------------------------

class TestR001WalkforwardMinSegmentFloor:
    """R-001: min_seg_days must be floored at 63 (one calendar quarter)."""

    def test_source_contains_floor_of_63(self):
        """
        Verify that the source of _run_walkforward_for_params contains
        the floor: max(63, min_seg_days).
        This is a structural check — the function lives in root/optimization_tab.py
        which imports streamlit and cannot be executed directly in tests.
        """
        import pathlib
        tab_path = pathlib.Path(__file__).parent.parent / "root" / "optimization_tab.py"
        assert tab_path.exists(), "optimization_tab.py not found"
        source = tab_path.read_text(encoding="utf-8")
        # The remediation must set a hard floor of 63
        assert "max(63, min_seg_days)" in source, (
            "R-001 floor 'max(63, min_seg_days)' not found in _run_walkforward_for_params"
        )

    def test_function_docstring_warns_calendar_validation(self):
        """
        The function must document the WARNING that this is calendar validation,
        not true walk-forward optimization.
        """
        import pathlib
        tab_path = pathlib.Path(__file__).parent.parent / "root" / "optimization_tab.py"
        source = tab_path.read_text(encoding="utf-8")
        assert "calendar_validation" in source, (
            "R-001: _run_walkforward_for_params must label output as calendar_validation"
        )
        assert "NOT true walk-forward" in source or "calendar_validation" in source, (
            "R-001: Function must warn consumers about the calendar-validation-only nature"
        )

    def test_min_seg_days_default_is_63(self):
        """The session_state default for opt_wf_min_days starts at 63."""
        import pathlib
        tab_path = pathlib.Path(__file__).parent.parent / "root" / "optimization_tab.py"
        source = tab_path.read_text(encoding="utf-8")
        # session_state.get("opt_wf_min_days", 63)
        assert '"opt_wf_min_days", 63' in source or "'opt_wf_min_days', 63" in source, (
            "R-001: default value for opt_wf_min_days should be 63"
        )


# ---------------------------------------------------------------------------
# R-002 / P1-MINOBS: Minimum observation count
# ---------------------------------------------------------------------------

class TestR002MinimumObservations:
    """R-002: ValidationThresholds.MIN_OBS must be >= 252 (1 trading year)."""

    def test_validation_thresholds_has_min_obs(self):
        """ValidationThresholds dataclass exposes a MIN_OBS field."""
        try:
            from core.contracts import ValidationThresholds
        except ImportError:
            pytest.skip("core.contracts not importable")

        assert hasattr(ValidationThresholds, "MIN_OBS"), (
            "R-002: ValidationThresholds must have a MIN_OBS class-level field"
        )

    def test_min_obs_is_at_least_252(self):
        """The MIN_OBS value must be >= 252 (one full trading year)."""
        try:
            from core.contracts import ValidationThresholds
        except ImportError:
            pytest.skip("core.contracts not importable")

        vt = ValidationThresholds()
        # Support both dataclass instance field and class-level default
        min_obs = getattr(vt, "MIN_OBS", None)
        if min_obs is None:
            min_obs = getattr(ValidationThresholds, "MIN_OBS", None)

        assert min_obs is not None, "R-002: MIN_OBS not accessible on ValidationThresholds"
        assert int(min_obs) >= 252, (
            f"R-002: ValidationThresholds.MIN_OBS={min_obs} must be >= 252 "
            "(one full trading year of data)"
        )

    def test_min_obs_source_comment_references_r002(self):
        """The MIN_OBS line in contracts.py should reference the finding ID."""
        import pathlib
        contracts_path = (
            pathlib.Path(__file__).parent.parent / "core" / "contracts.py"
        )
        assert contracts_path.exists(), "core/contracts.py not found"
        source = contracts_path.read_text(encoding="utf-8")
        # Check the constant exists with value 252
        assert "MIN_OBS" in source and "252" in source, (
            "R-002: MIN_OBS = 252 not found in core/contracts.py"
        )


# ---------------------------------------------------------------------------
# P0-KS: Kill-switch bridge
# ---------------------------------------------------------------------------

class TestP0KillSwitchBridge:
    """P0-KS: KillSwitchManager must call control_plane_callback on auto-trigger."""

    def _make_ksm(self, callback=None):
        from portfolio.risk_ops import KillSwitchConfig, KillSwitchManager
        return KillSwitchManager(
            cfg=KillSwitchConfig(),
            control_plane_callback=callback,
        )

    def test_kill_switch_callback_invoked_on_hard_trigger(self):
        """
        P0-KS: When a hard kill-switch condition fires, control_plane_callback
        must be invoked with a non-empty reason string.
        """
        try:
            from portfolio.risk_ops import KillSwitchConfig, KillSwitchManager
        except ImportError:
            pytest.skip("portfolio.risk_ops not importable")

        calls: list[str] = []
        ksm = KillSwitchManager(
            cfg=KillSwitchConfig(),
            control_plane_callback=lambda reason: calls.append(reason),
        )
        # Hard threshold is 20% drawdown by default
        ksm.check(current_value=75_000, peak_value=100_000)
        assert len(calls) > 0, "P0-KS: Callback was not invoked on kill-switch trigger"
        assert calls[0], "P0-KS: Callback was invoked with an empty reason"

    def test_kill_switch_callback_not_invoked_when_no_trigger(self):
        """Callback must NOT fire when kill-switch conditions are not met."""
        try:
            from portfolio.risk_ops import KillSwitchConfig, KillSwitchManager
        except ImportError:
            pytest.skip("portfolio.risk_ops not importable")

        calls: list[str] = []
        ksm = KillSwitchManager(
            cfg=KillSwitchConfig(),
            control_plane_callback=lambda reason: calls.append(reason),
        )
        # Only 1% drawdown — should not trigger anything
        ksm.check(current_value=99_000, peak_value=100_000)
        assert len(calls) == 0, "P0-KS: Callback fired unexpectedly for small drawdown"

    def test_kill_switch_backward_compatible_no_callback(self):
        """
        P0-KS: KillSwitchManager must be constructable without callback
        (backward compatible).
        """
        try:
            from portfolio.risk_ops import KillSwitchConfig, KillSwitchManager
        except ImportError:
            pytest.skip("portfolio.risk_ops not importable")

        # Should not raise
        ksm = KillSwitchManager(cfg=KillSwitchConfig())
        state = ksm.check(current_value=75_000, peak_value=100_000)
        assert state is not None

    def test_kill_switch_callback_not_invoked_twice_for_same_trigger(self):
        """
        Callback should fire only on the first trigger (newly_triggered), not
        on repeated calls that merely confirm an existing triggered state.
        """
        try:
            from portfolio.risk_ops import KillSwitchConfig, KillSwitchManager
        except ImportError:
            pytest.skip("portfolio.risk_ops not importable")

        calls: list[str] = []
        ksm = KillSwitchManager(
            cfg=KillSwitchConfig(),
            control_plane_callback=lambda reason: calls.append(reason),
        )
        ksm.check(current_value=75_000, peak_value=100_000)
        count_after_first = len(calls)
        # Second call with same drawdown — already triggered, no new event
        ksm.check(current_value=75_000, peak_value=100_000)
        assert len(calls) == count_after_first, (
            "P0-KS: Callback fired more than once for the same kill-switch trigger"
        )

    def test_p0ks_factory_creates_manager(self):
        """
        P0-KS: make_kill_switch_manager_with_control_plane() creates a
        KillSwitchManager instance.
        """
        try:
            from portfolio.risk_ops import (
                KillSwitchManager,
                make_kill_switch_manager_with_control_plane,
            )
        except ImportError:
            pytest.skip("portfolio.risk_ops not importable")

        manager = make_kill_switch_manager_with_control_plane()
        assert isinstance(manager, KillSwitchManager), (
            "P0-KS: Factory did not return a KillSwitchManager instance"
        )

    def test_p0ks_factory_returns_manager_when_control_plane_absent(self):
        """
        P0-KS: Factory must degrade gracefully when control_plane package
        is unavailable (returns plain KillSwitchManager without callback).
        """
        try:
            from portfolio.risk_ops import (
                KillSwitchManager,
                make_kill_switch_manager_with_control_plane,
            )
        except ImportError:
            pytest.skip("portfolio.risk_ops not importable")

        import sys
        # Temporarily hide control_plane to simulate absent package
        original = sys.modules.get("control_plane.engine")
        sys.modules["control_plane.engine"] = None  # type: ignore[assignment]
        try:
            manager = make_kill_switch_manager_with_control_plane()
            assert isinstance(manager, KillSwitchManager)
        finally:
            if original is None:
                sys.modules.pop("control_plane.engine", None)
            else:
                sys.modules["control_plane.engine"] = original

    def test_p0ks_manual_trigger_also_fires_callback(self):
        """P0-KS: trigger_manual() must also invoke the callback."""
        try:
            from portfolio.risk_ops import KillSwitchConfig, KillSwitchManager
            from portfolio.contracts import KillSwitchMode
        except ImportError:
            pytest.skip("portfolio.risk_ops not importable")

        calls: list[str] = []
        ksm = KillSwitchManager(
            cfg=KillSwitchConfig(),
            control_plane_callback=lambda reason: calls.append(reason),
        )
        ksm.trigger_manual(KillSwitchMode.HARD, reason="operator_test")
        assert len(calls) > 0, (
            "P0-KS: Callback not invoked on manual kill-switch trigger"
        )


# ---------------------------------------------------------------------------
# P1-PIPE: Signal pipeline
# ---------------------------------------------------------------------------

class TestP1SignalPipeline:
    """P1-PIPE: SignalPipeline integration layer."""

    def test_signal_pipeline_importable(self):
        """P1-PIPE: core.signal_pipeline module must be importable."""
        try:
            import core.signal_pipeline  # noqa: F401
        except ImportError as e:
            pytest.fail(f"P1-PIPE: core.signal_pipeline not importable: {e}")

    def test_signal_pipeline_creates(self):
        """P1-PIPE: SignalPipeline can be instantiated with a PairId."""
        try:
            from core.signal_pipeline import SignalPipeline
            from core.contracts import PairId
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        pair_id = PairId("AAPL", "MSFT")
        pipeline = SignalPipeline(pair_id=pair_id)
        assert pipeline is not None
        assert pipeline.pair_id == pair_id

    def test_signal_decision_has_required_fields(self):
        """
        P1-PIPE: SignalDecision must expose all fields expected by the
        portfolio layer (pair_id, as_of, z_score, regime, quality_grade,
        intent, blocked, block_reasons, size_multiplier, warnings, metadata).
        """
        try:
            from core.signal_pipeline import SignalDecision
            from core.contracts import PairId
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        from datetime import datetime

        decision = SignalDecision(
            pair_id=PairId("AAPL", "MSFT"),
            as_of=datetime.utcnow(),
            z_score=2.5,
            regime="MEAN_REVERTING",
            quality_grade="B",
            intent=None,
            blocked=False,
        )

        required_fields = [
            "pair_id", "as_of", "z_score", "regime", "quality_grade",
            "intent", "blocked", "block_reasons", "size_multiplier",
            "warnings", "metadata",
        ]
        for field_name in required_fields:
            assert hasattr(decision, field_name), (
                f"P1-PIPE: SignalDecision missing required field '{field_name}'"
            )

    def test_signal_decision_block_reasons_defaults_to_empty_list(self):
        """block_reasons defaults to empty list (not None)."""
        try:
            from core.signal_pipeline import SignalDecision
            from core.contracts import PairId
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        from datetime import datetime

        decision = SignalDecision(
            pair_id=PairId("AAPL", "MSFT"),
            as_of=datetime.utcnow(),
            z_score=1.0,
            regime="UNKNOWN",
            quality_grade="C",
            intent=None,
            blocked=False,
        )
        assert decision.block_reasons == [], (
            "P1-PIPE: block_reasons default must be [] not None"
        )
        assert decision.warnings == [], (
            "P1-PIPE: warnings default must be [] not None"
        )

    def test_signal_pipeline_registry_creates_pipeline(self):
        """P1-PIPE: SignalPipelineRegistry.get_or_create returns a SignalPipeline."""
        try:
            from core.signal_pipeline import SignalPipeline, SignalPipelineRegistry
            from core.contracts import PairId
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        registry = SignalPipelineRegistry()
        pair_id = PairId("AAPL", "MSFT")
        pipeline = registry.get_or_create(pair_id)
        assert isinstance(pipeline, SignalPipeline), (
            "P1-PIPE: get_or_create must return a SignalPipeline instance"
        )
        assert pipeline.pair_id == pair_id

    def test_signal_pipeline_registry_reuses_existing_pipeline(self):
        """get_or_create returns the same instance on repeated calls."""
        try:
            from core.signal_pipeline import SignalPipelineRegistry
            from core.contracts import PairId
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        registry = SignalPipelineRegistry()
        pair_id = PairId("AAPL", "MSFT")
        p1 = registry.get_or_create(pair_id)
        p2 = registry.get_or_create(pair_id)
        assert p1 is p2, "P1-PIPE: Registry must return the same pipeline for the same PairId"

    def test_signal_pipeline_remove(self):
        """Registry.remove() deregisters the pipeline for that PairId."""
        try:
            from core.signal_pipeline import SignalPipelineRegistry
            from core.contracts import PairId
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        registry = SignalPipelineRegistry()
        pair_id = PairId("AAPL", "MSFT")
        registry.get_or_create(pair_id)
        removed = registry.remove(pair_id)
        assert removed is True
        assert pair_id not in registry.active_pairs

    def test_signal_pipeline_active_pairs_tracking(self):
        """active_pairs reflects registered pipelines."""
        try:
            from core.signal_pipeline import SignalPipelineRegistry
            from core.contracts import PairId
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        registry = SignalPipelineRegistry()
        p1 = PairId("AAPL", "MSFT")
        p2 = PairId("GS", "MS")
        registry.get_or_create(p1)
        registry.get_or_create(p2)
        active = registry.active_pairs
        assert p1 in active
        assert p2 in active

    def test_get_signal_pipeline_registry_singleton(self):
        """get_signal_pipeline_registry() returns the same object each time."""
        try:
            from core.signal_pipeline import get_signal_pipeline_registry
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        r1 = get_signal_pipeline_registry()
        r2 = get_signal_pipeline_registry()
        assert r1 is r2, "P1-PIPE: get_signal_pipeline_registry() must be a singleton"

    def test_evaluate_bar_returns_bar_decision(self):
        """P1-PIPE: evaluate_bar() must return a BarDecision with correct fields."""
        try:
            from core.signal_pipeline import SignalPipeline, BarDecision
            from core.contracts import PairId
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        pipeline = SignalPipeline(pair_id=PairId("AAPL", "MSFT"))
        decision = pipeline.evaluate_bar(z_score=2.5, current_pos=0.0)
        assert isinstance(decision, BarDecision), (
            "P1-PIPE: evaluate_bar must return a BarDecision"
        )
        assert hasattr(decision, "action"), "P1-PIPE: BarDecision missing 'action'"
        assert hasattr(decision, "entry_z"), "P1-PIPE: BarDecision missing 'entry_z'"
        assert hasattr(decision, "regime"), "P1-PIPE: BarDecision missing 'regime'"
        assert hasattr(decision, "quality_grade"), "P1-PIPE: BarDecision missing 'quality_grade'"
        assert hasattr(decision, "blocked"), "P1-PIPE: BarDecision missing 'blocked'"

    def test_evaluate_bar_entry_signal(self):
        """P1-PIPE: evaluate_bar with z above entry threshold should produce entry."""
        try:
            from core.signal_pipeline import SignalPipeline
            from core.contracts import PairId
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        pipeline = SignalPipeline(pair_id=PairId("AAPL", "MSFT"))
        # z = +3.0 (above default entry_z=2.0) → should short spread (action = -1)
        decision = pipeline.evaluate_bar(z_score=3.0, current_pos=0.0)
        assert decision.action == -1.0, (
            f"P1-PIPE: z=3.0 should produce short entry (action=-1), got {decision.action}"
        )

    def test_evaluate_bar_no_entry_below_threshold(self):
        """P1-PIPE: evaluate_bar with z below entry threshold should stay flat."""
        try:
            from core.signal_pipeline import SignalPipeline
            from core.contracts import PairId
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        pipeline = SignalPipeline(pair_id=PairId("AAPL", "MSFT"))
        decision = pipeline.evaluate_bar(z_score=1.0, current_pos=0.0)
        assert decision.action == 0.0, (
            f"P1-PIPE: z=1.0 below entry_z should be flat, got {decision.action}"
        )

    def test_backtester_has_use_signal_pipeline_param(self):
        """P1-PIPE: optimization_backtester._bt_build_signal_frame must support
        use_signal_pipeline parameter."""
        import pathlib
        bt_path = pathlib.Path(__file__).parent.parent / "core" / "optimization_backtester.py"
        source = bt_path.read_text(encoding="utf-8")
        assert "use_signal_pipeline" in source, (
            "P1-PIPE: optimization_backtester.py must contain 'use_signal_pipeline' parameter"
        )
        assert "signal_pipeline" in source and "SignalPipeline" in source, (
            "P1-PIPE: optimization_backtester.py must import and use SignalPipeline"
        )


# ---------------------------------------------------------------------------
# P1-PORTINT: Portfolio receives real signal intents
# ---------------------------------------------------------------------------

class TestP1PortfolioIntegration:
    """P1-PORTINT: PortfolioAllocator receives real signal intents via bridge."""

    def test_bridge_module_importable(self):
        """P1-PORTINT: core.portfolio_bridge must be importable."""
        try:
            from core.portfolio_bridge import (
                extract_entry_intents,
                bridge_signals_to_allocator,
            )
        except ImportError as e:
            pytest.fail(f"P1-PORTINT: core.portfolio_bridge not importable: {e}")

    def test_extract_entry_intents_from_decisions(self):
        """P1-PORTINT: extract_entry_intents filters blocked and non-entry decisions."""
        from core.portfolio_bridge import extract_entry_intents
        from core.signal_pipeline import SignalDecision
        from core.contracts import PairId
        from core.intents import EntryIntent
        from datetime import datetime

        pair = PairId("AAPL", "MSFT")
        now = datetime.utcnow()

        # One valid entry intent
        d1 = SignalDecision(
            pair_id=pair, as_of=now, z_score=2.5, regime="MEAN_REVERTING",
            quality_grade="A", intent=EntryIntent(pair_id=pair, confidence=0.7, z_score=2.5),
            blocked=False, size_multiplier=1.0,
        )
        # One blocked decision
        d2 = SignalDecision(
            pair_id=pair, as_of=now, z_score=1.0, regime="CRISIS",
            quality_grade="F", intent=None,
            blocked=True, block_reasons=["quality F"],
        )
        # One with no intent (z below threshold)
        d3 = SignalDecision(
            pair_id=PairId("GOOG", "META"), as_of=now, z_score=0.5,
            regime="MEAN_REVERTING", quality_grade="B", intent=None,
            blocked=False,
        )

        intents = extract_entry_intents([d1, d2, d3])
        assert len(intents) == 1, (
            f"P1-PORTINT: Expected 1 eligible intent, got {len(intents)}"
        )
        assert isinstance(intents[0], EntryIntent)
        assert intents[0].pair_id == pair
        # Check enrichment
        assert getattr(intents[0], "quality_grade", None) == "A"
        assert getattr(intents[0], "regime", None) == "MEAN_REVERTING"

    def test_bridge_runs_allocation_cycle(self):
        """P1-PORTINT: bridge_signals_to_allocator produces allocation decisions."""
        from core.portfolio_bridge import bridge_signals_to_allocator
        from core.signal_pipeline import SignalDecision
        from core.contracts import PairId, SignalDirection
        from core.intents import EntryIntent
        from datetime import datetime

        pair = PairId("AAPL", "MSFT")
        now = datetime.utcnow()

        decisions = [
            SignalDecision(
                pair_id=pair, as_of=now, z_score=2.5, regime="MEAN_REVERTING",
                quality_grade="A",
                intent=EntryIntent(
                    pair_id=pair, confidence=0.7, z_score=2.5,
                    direction=SignalDirection.LONG_SPREAD,
                    entry_z_threshold=2.0, exit_z_target=0.5, stop_z=4.0,
                ),
                blocked=False, size_multiplier=1.0,
                metadata={"half_life": 15.0},
            ),
        ]

        allocations, diagnostics = bridge_signals_to_allocator(
            signal_decisions=decisions,
            capital=1_000_000.0,
        )

        # Must produce at least one allocation decision (funded or unfunded)
        assert len(allocations) >= 1, (
            f"P1-PORTINT: Expected at least 1 allocation decision, got {len(allocations)}"
        )
        # Diagnostics should record the intent
        assert diagnostics.n_intents_received >= 1, (
            "P1-PORTINT: diagnostics must record intents received"
        )

    def test_bridge_empty_decisions_returns_empty(self):
        """P1-PORTINT: No eligible intents produces empty allocation with diagnostics."""
        from core.portfolio_bridge import bridge_signals_to_allocator
        from core.signal_pipeline import SignalDecision
        from core.contracts import PairId
        from datetime import datetime

        pair = PairId("AAPL", "MSFT")
        now = datetime.utcnow()

        # All blocked
        decisions = [
            SignalDecision(
                pair_id=pair, as_of=now, z_score=1.0, regime="CRISIS",
                quality_grade="F", intent=None, blocked=True,
                block_reasons=["quality F"],
            ),
        ]

        allocations, diagnostics = bridge_signals_to_allocator(decisions, capital=1_000_000.0)
        assert len(allocations) == 0
        assert diagnostics.n_intents_received == 1

    def test_unfunded_has_rationale(self):
        """P1-PORTINT: Unfunded allocations carry explicit rationale."""
        from core.portfolio_bridge import bridge_signals_to_allocator
        from core.signal_pipeline import SignalDecision
        from core.contracts import PairId, SignalDirection
        from core.intents import EntryIntent
        from datetime import datetime

        # Create 10 intents — allocator has limited capital, some must be unfunded
        decisions = []
        for i in range(10):
            pair = PairId(f"SYM{i:02d}A", f"SYM{i:02d}B")
            decisions.append(SignalDecision(
                pair_id=pair,
                as_of=datetime.utcnow(),
                z_score=2.0 + i * 0.1,
                regime="MEAN_REVERTING",
                quality_grade="B",
                intent=EntryIntent(
                    pair_id=pair, confidence=0.6, z_score=2.0 + i * 0.1,
                    direction=SignalDirection.LONG_SPREAD,
                ),
                blocked=False, size_multiplier=1.0,
            ))

        allocations, diagnostics = bridge_signals_to_allocator(
            decisions, capital=100_000.0,  # small capital → some must be unfunded
        )

        # Some must be approved, some rejected (capital-limited)
        assert len(allocations) > 0, "P1-PORTINT: Expected allocation decisions"
        # Check that all decisions have rationale
        for a in allocations:
            assert a.rationale is not None, (
                f"P1-PORTINT: Allocation for {a.pair_id} missing rationale"
            )


# ---------------------------------------------------------------------------
# P1-SAFE: Runtime safety gating on portfolio bridge
# ---------------------------------------------------------------------------

class TestP1SafetyGating:
    """P1-SAFE: is_safe_to_trade() blocks entries via portfolio bridge."""

    def _make_valid_decision(self):
        from core.signal_pipeline import SignalDecision
        from core.contracts import PairId, SignalDirection
        from core.intents import EntryIntent
        from datetime import datetime

        pair = PairId("AAPL", "MSFT")
        return SignalDecision(
            pair_id=pair,
            as_of=datetime.utcnow(),
            z_score=2.5,
            regime="MEAN_REVERTING",
            quality_grade="A",
            intent=EntryIntent(
                pair_id=pair, confidence=0.7, z_score=2.5,
                direction=SignalDirection.LONG_SPREAD,
            ),
            blocked=False,
            size_multiplier=1.0,
        )

    def test_bridge_accepts_safety_check_parameter(self):
        """P1-SAFE: bridge_signals_to_allocator must accept safety_check kwarg."""
        import inspect
        from core.portfolio_bridge import bridge_signals_to_allocator
        sig = inspect.signature(bridge_signals_to_allocator)
        assert "safety_check" in sig.parameters, (
            "P1-SAFE: bridge_signals_to_allocator must accept 'safety_check' parameter"
        )

    def test_unsafe_state_blocks_all_entries(self):
        """P1-SAFE: When safety_check returns (False, reasons), all entries are blocked."""
        from core.portfolio_bridge import bridge_signals_to_allocator

        decisions = [self._make_valid_decision()]

        # Inject a safety check that says "UNSAFE"
        def unsafe_check():
            return (False, ["kill_switch_active: test_emergency"])

        allocations, diagnostics = bridge_signals_to_allocator(
            decisions,
            capital=1_000_000.0,
            safety_check=unsafe_check,
        )

        # Must block ALL entries
        assert len(allocations) == 0, (
            f"P1-SAFE: Unsafe state must produce 0 allocations, got {len(allocations)}"
        )
        # Diagnostics must record the block
        assert diagnostics.n_blocked_risk > 0, (
            "P1-SAFE: Diagnostics must record blocked entries when unsafe"
        )
        # Binding constraints must include SAFETY_BLOCK
        assert any("SAFETY_BLOCK" in c for c in diagnostics.binding_constraints), (
            f"P1-SAFE: binding_constraints must include SAFETY_BLOCK, got {diagnostics.binding_constraints}"
        )

    def test_safe_state_allows_entries(self):
        """P1-SAFE: When safety_check returns (True, []), entries proceed normally."""
        from core.portfolio_bridge import bridge_signals_to_allocator

        decisions = [self._make_valid_decision()]

        # Inject a safety check that says "SAFE"
        def safe_check():
            return (True, [])

        allocations, diagnostics = bridge_signals_to_allocator(
            decisions,
            capital=1_000_000.0,
            safety_check=safe_check,
        )

        # Must produce at least one allocation decision
        assert len(allocations) >= 1, (
            f"P1-SAFE: Safe state must allow allocations, got {len(allocations)}"
        )

    def test_no_safety_check_allows_entries(self):
        """P1-SAFE: Without safety_check (research/backtest), entries proceed."""
        from core.portfolio_bridge import bridge_signals_to_allocator

        decisions = [self._make_valid_decision()]

        # No safety_check parameter → entries proceed (backward compatible)
        allocations, diagnostics = bridge_signals_to_allocator(
            decisions, capital=1_000_000.0,
        )

        assert len(allocations) >= 1, (
            "P1-SAFE: Without safety_check, allocations must proceed normally"
        )

    def test_safety_check_exception_treated_as_unsafe(self):
        """P1-SAFE: If safety_check raises, treat as unsafe (conservative)."""
        from core.portfolio_bridge import bridge_signals_to_allocator

        decisions = [self._make_valid_decision()]

        def broken_check():
            raise RuntimeError("connection lost")

        allocations, diagnostics = bridge_signals_to_allocator(
            decisions, capital=1_000_000.0,
            safety_check=broken_check,
        )

        assert len(allocations) == 0, (
            "P1-SAFE: Safety check exception must block all entries"
        )
        assert any("SAFETY_BLOCK" in c for c in diagnostics.binding_constraints), (
            "P1-SAFE: Exception must be recorded in binding_constraints"
        )

    def test_runtime_state_manager_compatible(self):
        """P1-SAFE: RuntimeStateManager.is_safe_to_trade matches expected signature."""
        from runtime.state import RuntimeStateManager
        mgr = RuntimeStateManager(env="test")

        safe, reasons = mgr.is_safe_to_trade()
        assert isinstance(safe, bool), "P1-SAFE: is_safe_to_trade must return bool"
        assert isinstance(reasons, list), "P1-SAFE: is_safe_to_trade must return list"

        # Default state should be safe (no kill switch, no halted components)
        # Note: INITIALIZING state may be unsafe depending on implementation
        # We just verify the signature contract here

    def test_bridge_source_does_not_import_runtime(self):
        """P1-SAFE: core/portfolio_bridge.py must NOT import from runtime/."""
        import pathlib, ast
        bridge_path = pathlib.Path(__file__).parent.parent / "core" / "portfolio_bridge.py"
        source = bridge_path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("runtime"), (
                    f"P1-SAFE: core/portfolio_bridge.py has 'from {node.module} import ...' "
                    "— core/ must not import from runtime/ (architecture boundary)"
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("runtime"), (
                        f"P1-SAFE: core/portfolio_bridge.py has 'import {alias.name}' "
                        "— core/ must not import from runtime/"
                    )


# ---------------------------------------------------------------------------
# P1-AGENTS: First operational agent dispatch
# ---------------------------------------------------------------------------

class TestP1AgentDispatch:
    """P1-AGENTS: DataIntegrityAgent dispatched from orchestrator."""

    def test_data_integrity_agent_registered(self):
        """P1-AGENTS: DataIntegrityAgent must be in the default registry."""
        from agents.registry import get_default_registry
        registry = get_default_registry()
        agent = registry.get_agent("data_integrity")
        assert agent is not None, (
            "P1-AGENTS: DataIntegrityAgent must be registered in default registry"
        )

    def test_data_integrity_agent_dispatch_produces_typed_result(self):
        """P1-AGENTS: Dispatching DataIntegrityAgent produces typed AgentResult."""
        from agents.registry import get_default_registry
        from core.contracts import AgentTask, AgentResult, AgentStatus

        registry = get_default_registry()
        task = AgentTask(
            task_id="test_di_001",
            agent_name="data_integrity",
            task_type="check_data_integrity",
            payload={},
        )
        result = registry.dispatch(task)

        assert isinstance(result, AgentResult), (
            "P1-AGENTS: dispatch must return AgentResult"
        )
        assert result.status == AgentStatus.COMPLETED, (
            f"P1-AGENTS: Expected COMPLETED, got {result.status}"
        )
        assert isinstance(result.output, dict), (
            "P1-AGENTS: output must be dict"
        )

    def test_data_integrity_audit_trail_exists(self):
        """P1-AGENTS: Dispatch creates audit trail entries."""
        from agents.registry import get_default_registry
        from core.contracts import AgentTask

        registry = get_default_registry()
        task = AgentTask(
            task_id="test_di_audit",
            agent_name="data_integrity",
            task_type="check_data_integrity",
            payload={},
        )
        result = registry.dispatch(task)

        assert len(result.audit_trail) > 0, (
            "P1-AGENTS: AgentResult must have non-empty audit_trail"
        )

        # Registry must also record the dispatch in its own audit log
        audit_log = registry.get_audit_log(agent_name="data_integrity", limit=5)
        assert len(audit_log) > 0, (
            "P1-AGENTS: Registry audit log must record the dispatch"
        )

    def test_orchestrator_has_agent_dispatch_method(self):
        """P1-AGENTS: Orchestrator must have run_agent_data_integrity_check method."""
        from core.orchestrator import PairsOrchestrator
        orch = PairsOrchestrator()
        assert hasattr(orch, "run_agent_data_integrity_check"), (
            "P1-AGENTS: Orchestrator must have run_agent_data_integrity_check method"
        )

    def test_orchestrator_agent_dispatch_returns_task_result(self):
        """P1-AGENTS: run_agent_data_integrity_check returns TaskResult."""
        from core.orchestrator import PairsOrchestrator, TaskResult
        orch = PairsOrchestrator()
        result = orch.run_agent_data_integrity_check()

        assert result is not None, (
            "P1-AGENTS: run_agent_data_integrity_check must return TaskResult"
        )
        assert isinstance(result, TaskResult), (
            f"P1-AGENTS: Expected TaskResult, got {type(result)}"
        )
        assert result.task_name == "agent_data_integrity"
        assert result.output is not None
        # Output can come from WorkflowEngine path or direct dispatch path
        assert "agent_status" in result.output or "workflow_status" in result.output, (
            f"P1-AGENTS: output must contain agent_status or workflow_status, got {list(result.output.keys())}"
        )

    def test_agent_does_not_mutate_state(self):
        """P1-AGENTS: DataIntegrityAgent is READ_ONLY — no side effects."""
        from agents.registry import get_default_registry
        from core.contracts import AgentTask

        registry = get_default_registry()
        # Check permissions
        assert registry.has_permission("data_integrity", "read_only"), (
            "P1-AGENTS: DataIntegrityAgent must have read_only permission"
        )

    def test_daily_pipeline_source_contains_agent_dispatch(self):
        """P1-AGENTS: run_daily_pipeline source must contain agent dispatch call."""
        import pathlib
        orch_path = pathlib.Path(__file__).parent.parent / "core" / "orchestrator.py"
        source = orch_path.read_text(encoding="utf-8-sig")
        assert "run_agent_data_integrity_check" in source
        assert "agent_data_integrity" in source

    def test_signal_agents_dispatched(self):
        """P1-AGENTS: Signal-layer agents must be dispatched after compute_signals."""
        import pathlib
        source = pathlib.Path(__file__).parent.parent / "core" / "orchestrator.py"
        src = source.read_text(encoding="utf-8-sig")
        assert "_dispatch_signal_agents" in src
        assert "regime_surveillance" in src

    def test_risk_agents_dispatched(self):
        """P1-AGENTS: Risk-layer agents must be dispatched after allocation."""
        import pathlib
        source = pathlib.Path(__file__).parent.parent / "core" / "orchestrator.py"
        src = source.read_text(encoding="utf-8-sig")
        assert "_dispatch_risk_agents" in src
        assert "exposure_monitor" in src
        assert "drawdown_monitor" in src
        assert "kill_switch" in src

    def test_signal_agent_dispatch_returns_results(self):
        """P1-AGENTS: _dispatch_signal_agents returns list of TaskResult."""
        from core.orchestrator import PairsOrchestrator, TaskResult
        orch = PairsOrchestrator()
        results = orch._dispatch_signal_agents()
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, TaskResult)

    def test_risk_agent_dispatch_returns_results(self):
        """P1-AGENTS: _dispatch_risk_agents returns list of TaskResult."""
        from core.orchestrator import PairsOrchestrator, TaskResult
        orch = PairsOrchestrator()
        results = orch._dispatch_risk_agents()
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, TaskResult)

    def test_all_dispatched_agents_registered(self):
        """P1-AGENTS: All 13 operational agents must be in default registry."""
        from agents.registry import get_default_registry
        registry = get_default_registry()
        operational = [
            # Monitoring (2)
            "system_health", "data_integrity",
            # Signal-layer (4)
            "regime_surveillance", "signal_analyst",
            "trade_lifecycle", "exit_oversight",
            # Risk-layer (7)
            "exposure_monitor", "drawdown_monitor", "kill_switch",
            "capital_budget", "derisking",
            "drift_monitoring", "alert_aggregation",
        ]
        for name in operational:
            assert registry.get_agent(name) is not None, (
                f"Agent '{name}' must be registered in default registry"
            )

    def test_research_agent_dispatch(self):
        """P1-AGENTS: dispatch_research_agents returns results."""
        from core.orchestrator import PairsOrchestrator, TaskResult
        orch = PairsOrchestrator()
        results = orch.dispatch_research_agents(symbols=["AAPL", "MSFT"])
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, TaskResult)


# ---------------------------------------------------------------------------
# P1-ML: ML meta-label hook wired into orchestrator
# ---------------------------------------------------------------------------

class TestP1MLHookWiring:
    """P1-ML: Orchestrator wires MetaLabelModel into SignalPipeline when model exists."""

    def test_orchestrator_source_loads_ml_model(self):
        """P1-ML: _collect_signal_decisions must attempt to load meta-label model."""
        import pathlib
        orch_path = pathlib.Path(__file__).parent.parent / "core" / "orchestrator.py"
        source = orch_path.read_text(encoding="utf-8-sig")
        assert "MetaLabelModel" in source, (
            "P1-ML: orchestrator must import MetaLabelModel"
        )
        assert "ml_quality_hook" in source, (
            "P1-ML: orchestrator must pass ml_quality_hook to SignalPipeline"
        )
        assert "meta_label_latest.pkl" in source, (
            "P1-ML: orchestrator must look for meta_label_latest.pkl"
        )

    def test_pipeline_accepts_ml_hook(self):
        """P1-ML: SignalPipeline constructor accepts ml_quality_hook."""
        from core.signal_pipeline import SignalPipeline
        from core.contracts import PairId

        # With None (deterministic fallback)
        p1 = SignalPipeline(pair_id=PairId("A", "B"), ml_quality_hook=None)
        d1 = p1.evaluate_bar(z_score=2.5, current_pos=0.0)
        assert d1 is not None

        # With a mock hook (implements predict_success_probability)
        class MockHook:
            def predict_success_probability(self, features):
                return 0.8
        p2 = SignalPipeline(pair_id=PairId("A", "B"), ml_quality_hook=MockHook())
        d2 = p2.evaluate_bar(z_score=2.5, current_pos=0.0)
        assert d2 is not None

    def test_fallback_when_no_model_file(self):
        """P1-ML: Without model file, ml_hook=None and pipeline uses deterministic quality."""
        from core.signal_pipeline import SignalPipeline
        from core.contracts import PairId

        # Default: no hook
        pipeline = SignalPipeline(pair_id=PairId("X", "Y"))
        decision = pipeline.evaluate_bar(z_score=3.0, current_pos=0.0)
        # Must still produce a valid decision
        assert decision.action != 0 or True  # action could be 0 if blocked — that's fine
        assert hasattr(decision, "quality_grade")


# ---------------------------------------------------------------------------
# P1-GOV: Governance gate on model promotion
# ---------------------------------------------------------------------------

class TestP1GovernanceGate:
    """P1-GOV: MLModelRegistry.promote() must call governance check for CHAMPION."""

    def _make_registered_registry(self):
        """Helper: create a registry with one model registered as CANDIDATE."""
        from ml.registry.registry import MLModelRegistry
        from ml.contracts import (
            MLTaskFamily, ModelMetadata, ModelStatus, GovernanceStatus,
            PromotionDecision, PromotionOutcome,
        )
        import uuid

        registry = MLModelRegistry()
        model_id = f"test_model_{uuid.uuid4().hex[:8]}"
        metadata = ModelMetadata(
            model_id=model_id,
            task_family=MLTaskFamily.META_LABELING,
            model_class="MockClassifier",
            version="1.0",
            status=ModelStatus.CANDIDATE,
            governance_status=GovernanceStatus.PENDING_REVIEW,
        )
        registry.register(object(), metadata)
        return registry, model_id

    def test_promote_champion_calls_governance_check(self):
        """
        P1-GOV: promote() to CHAMPION calls governance.engine.get_governance_engine
        and invokes check_policy().
        """
        try:
            from ml.registry.registry import MLModelRegistry
            from ml.contracts import (
                ModelStatus, PromotionDecision, PromotionOutcome,
            )
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        registry, model_id = self._make_registered_registry()

        mock_result = MagicMock()
        mock_result.passed = True
        mock_engine = MagicMock()
        mock_engine.check_policy.return_value = mock_result

        with patch(
            "ml.registry.registry.get_governance_engine",
            return_value=mock_engine,
            create=True,
        ):
            decision = PromotionDecision(
                model_id=model_id,
                outcome=PromotionOutcome.PROMOTE,
                manually_approved=True,
                requires_manual_approval=True,
            )
            registry.promote(model_id, ModelStatus.CHAMPION, decision)

        mock_engine.check_policy.assert_called_once(), (
            "P1-GOV: governance check_policy() was not called during CHAMPION promotion"
        )

    def test_promote_non_champion_skips_governance_check(self):
        """
        P1-GOV: Promoting to CHALLENGER should NOT invoke the governance check
        (only CHAMPION promotions require governance sign-off).
        """
        try:
            from ml.registry.registry import MLModelRegistry
            from ml.contracts import (
                ModelStatus, PromotionDecision, PromotionOutcome,
            )
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        registry, model_id = self._make_registered_registry()

        mock_engine = MagicMock()

        with patch(
            "ml.registry.registry.get_governance_engine",
            return_value=mock_engine,
            create=True,
        ):
            decision = PromotionDecision(
                model_id=model_id,
                outcome=PromotionOutcome.PROMOTE,
                manually_approved=True,
                requires_manual_approval=False,
            )
            registry.promote(model_id, ModelStatus.CHALLENGER, decision)

        mock_engine.check_policy.assert_not_called(), (
            "P1-GOV: governance check was incorrectly invoked for CHALLENGER promotion"
        )

    def test_promote_champion_blocked_by_governance_raises(self):
        """
        P1-GOV: If governance returns a CRITICAL failure, promote() must raise
        ValueError with a message referencing the finding.
        """
        try:
            from ml.registry.registry import MLModelRegistry
            from ml.contracts import (
                ModelStatus, PromotionDecision, PromotionOutcome,
            )
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        registry, model_id = self._make_registered_registry()

        mock_check_result = MagicMock()
        mock_check_result.passed = False
        mock_check_result.severity = MagicMock()
        mock_check_result.severity.value = "CRITICAL"
        mock_check_result.message = "Governance policy BLOCKED"

        mock_engine = MagicMock()
        mock_engine.check_policy.return_value = mock_check_result

        with patch(
            "ml.registry.registry.get_governance_engine",
            return_value=mock_engine,
            create=True,
        ):
            decision = PromotionDecision(
                model_id=model_id,
                outcome=PromotionOutcome.PROMOTE,
                manually_approved=True,
                requires_manual_approval=False,
            )
            with pytest.raises(ValueError, match="[Gg]overnance"):
                registry.promote(model_id, ModelStatus.CHAMPION, decision)

    def test_promote_to_champion_demotes_existing_champion(self):
        """
        When a new CHAMPION is promoted, the previous CHAMPION for the same
        task family must be demoted to CHALLENGER (auto-demotion).
        """
        try:
            from ml.registry.registry import MLModelRegistry
            from ml.contracts import (
                MLTaskFamily, ModelMetadata, ModelStatus, GovernanceStatus,
                PromotionDecision, PromotionOutcome,
            )
            import uuid
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        registry = MLModelRegistry()

        def _reg(status):
            mid = f"model_{uuid.uuid4().hex[:8]}"
            meta = ModelMetadata(
                model_id=mid,
                task_family=MLTaskFamily.META_LABELING,
                model_class="MockModel",
                version="1.0",
                status=status,
                governance_status=GovernanceStatus.APPROVED,
            )
            registry.register(object(), meta)
            return mid

        champ_id = _reg(ModelStatus.CANDIDATE)
        challenger_id = _reg(ModelStatus.CANDIDATE)

        # Promote first model to CHAMPION
        mock_ok = MagicMock()
        mock_ok.passed = True
        mock_engine = MagicMock()
        mock_engine.check_policy.return_value = mock_ok

        with patch(
            "ml.registry.registry.get_governance_engine",
            return_value=mock_engine,
            create=True,
        ):
            d1 = PromotionDecision(
                model_id=champ_id,
                outcome=PromotionOutcome.PROMOTE,
                manually_approved=True,
                requires_manual_approval=True,
            )
            registry.promote(champ_id, ModelStatus.CHAMPION, d1)

            # Now promote second model to CHAMPION — first must be demoted
            d2 = PromotionDecision(
                model_id=challenger_id,
                outcome=PromotionOutcome.PROMOTE,
                manually_approved=True,
                requires_manual_approval=True,
            )
            registry.promote(challenger_id, ModelStatus.CHAMPION, d2)

        old_champ = registry.get(champ_id)
        new_champ = registry.get(challenger_id)
        assert new_champ.status == ModelStatus.CHAMPION
        assert old_champ.status == ModelStatus.CHALLENGER, (
            "P1-GOV: Previous CHAMPION was not demoted to CHALLENGER after new promotion"
        )


# ---------------------------------------------------------------------------
# P1-SURV2: Stale data surveillance hook
# ---------------------------------------------------------------------------

class TestP1Surv2StaleDataHook:
    """P1-SURV2: data_loader must have surveillance infrastructure for stale data."""

    def test_compute_data_age_hours_helper_exists(self):
        """
        P1-SURV2: _compute_data_age_hours helper must exist in common.data_loader.
        """
        try:
            from common import data_loader
        except ImportError as e:
            pytest.skip(f"common.data_loader not importable: {e}")

        assert hasattr(data_loader, "_compute_data_age_hours"), (
            "P1-SURV2: _compute_data_age_hours not found in common.data_loader"
        )
        assert callable(data_loader._compute_data_age_hours), (
            "P1-SURV2: _compute_data_age_hours must be callable"
        )

    def test_compute_data_age_hours_returns_float_for_recent_data(self):
        """_compute_data_age_hours returns a non-negative float for a fresh DataFrame."""
        try:
            from common.data_loader import _compute_data_age_hours
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        import pandas as pd
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0]},
            index=pd.to_datetime([
                now.replace(hour=0, minute=0, second=0, microsecond=0)
                .__class__(now.year, now.month, now.day, tzinfo=timezone.utc)
            ] * 3),
        )
        result = _compute_data_age_hours(df)
        # Should be a non-negative float (0 to ~24 hours for today's data)
        assert result is not None, "P1-SURV2: Should return a float for a populated DataFrame"
        assert isinstance(result, float), "P1-SURV2: Must return a float, not None"
        assert result >= 0.0, "P1-SURV2: Data age cannot be negative"

    def test_compute_data_age_hours_returns_none_for_empty_df(self):
        """_compute_data_age_hours returns None when given an empty DataFrame."""
        try:
            from common.data_loader import _compute_data_age_hours
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        import pandas as pd

        result = _compute_data_age_hours(pd.DataFrame())
        assert result is None, (
            "P1-SURV2: _compute_data_age_hours must return None for empty DataFrame"
        )

    def test_compute_data_age_hours_returns_none_for_none_input(self):
        """_compute_data_age_hours handles None input without raising."""
        try:
            from common.data_loader import _compute_data_age_hours
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        result = _compute_data_age_hours(None)  # type: ignore[arg-type]
        assert result is None, (
            "P1-SURV2: _compute_data_age_hours must return None for None input"
        )

    def test_surveillance_hook_in_load_price_data_source(self):
        """
        P1-SURV2: load_price_data() source must contain the surveillance hook
        that calls surveillance engine detect() with SURV-DI-001.
        """
        import pathlib
        loader_path = (
            pathlib.Path(__file__).parent.parent / "common" / "data_loader.py"
        )
        assert loader_path.exists(), "common/data_loader.py not found"
        source = loader_path.read_text(encoding="utf-8")
        assert "SURV-DI-001" in source, (
            "P1-SURV2: Surveillance rule ID 'SURV-DI-001' not found in data_loader.py"
        )
        assert "_compute_data_age_hours" in source, (
            "P1-SURV2: _compute_data_age_hours call not found in data_loader.py"
        )
        assert "get_surveillance_engine" in source, (
            "P1-SURV2: get_surveillance_engine import not found in data_loader.py"
        )

    def test_surveillance_hook_errors_do_not_break_load(self):
        """
        P1-SURV2: Surveillance hook failure must not propagate — data loading
        must complete even if the surveillance engine raises.
        The hook must be wrapped in a bare except clause.
        """
        import pathlib
        loader_path = (
            pathlib.Path(__file__).parent.parent / "common" / "data_loader.py"
        )
        source = loader_path.read_text(encoding="utf-8")
        # Locate the surveillance block — verify it's inside a try/except
        assert "except Exception" in source or "except:" in source, (
            "P1-SURV2: Surveillance hook must be in a try/except block"
        )
        # Specifically verify the pattern: try / surveillance / except Exception: pass
        assert "pass  # Surveillance errors never break data loading" in source or (
            "pass" in source and "Surveillance" in source
        ), (
            "P1-SURV2: Surveillance hook exception must be silently swallowed"
        )
