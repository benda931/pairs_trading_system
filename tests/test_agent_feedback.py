# -*- coding: utf-8 -*-
"""Tests for core.agent_feedback — Agent Feedback Loop."""
from __future__ import annotations
from types import SimpleNamespace


class TestFeedbackActionGeneration:
    """Test that agent outputs produce correct feedback actions."""

    def test_regime_crisis_generates_block_and_deleverage(self):
        from core.agent_feedback import SignalFeedbackRules
        result = {"status": "success", "output": {"regime_map": {"XLI/XLB": "CRISIS"}, "current_regime": "CRISIS"}}
        actions = SignalFeedbackRules.process("regime_surveillance", result)
        types = [a.action_type for a in actions]
        assert "BLOCK_ENTRY" in types
        assert "DELEVERAGE" in types
        assert any(a.severity == "CRITICAL" for a in actions)

    def test_regime_normal_generates_no_actions(self):
        from core.agent_feedback import SignalFeedbackRules
        result = {"status": "success", "output": {"current_regime": "NORMAL"}}
        actions = SignalFeedbackRules.process("regime_surveillance", result)
        assert len(actions) == 0

    def test_regime_tension_generates_threshold_adjust(self):
        from core.agent_feedback import SignalFeedbackRules
        result = {"status": "success", "output": {"current_regime": "TENSION"}}
        actions = SignalFeedbackRules.process("regime_surveillance", result)
        assert len(actions) == 1
        assert actions[0].action_type == "ADJUST_THRESHOLD"

    def test_drawdown_10pct_triggers_deleverage(self):
        from core.agent_feedback import RiskFeedbackRules
        result = {"status": "success", "output": {"current_dd_pct": -0.12}}
        actions = RiskFeedbackRules.process("drawdown_monitor", result)
        assert len(actions) == 1
        assert actions[0].action_type == "DELEVERAGE"
        assert actions[0].severity == "CRITICAL"

    def test_drawdown_20pct_triggers_kill_switch(self):
        from core.agent_feedback import RiskFeedbackRules
        result = {"status": "success", "output": {"current_drawdown": -0.22}}
        actions = RiskFeedbackRules.process("drawdown_monitor", result)
        types = [a.action_type for a in actions]
        assert "KILL_SWITCH" in types
        assert any(a.severity == "EMERGENCY" for a in actions)

    def test_kill_switch_triggered(self):
        from core.agent_feedback import RiskFeedbackRules
        result = {"status": "success", "output": {"triggered": True, "mode": "EXITS_ONLY"}}
        actions = RiskFeedbackRules.process("kill_switch", result)
        assert len(actions) == 1
        assert actions[0].action_type == "KILL_SWITCH"

    def test_exit_oversight_generates_force_exits(self):
        from core.agent_feedback import SignalFeedbackRules
        result = {"status": "success", "output": {
            "exit_signals": [
                {"pair_id": "XLI/XLB", "reason": "spread_diverged"},
                {"pair_id": "XLY/XLC", "reason": "stale_position"},
            ]
        }}
        actions = SignalFeedbackRules.process("exit_oversight", result)
        assert len(actions) == 2
        assert all(a.action_type == "FORCE_EXIT" for a in actions)

    def test_gpt_retrain_recommendation(self):
        from core.agent_feedback import ImprovementFeedbackRules
        result = {"status": "success", "output": {
            "recommendations": [{"action": "RETRAIN", "pair": "XLI/XLB", "reason": "AUC degraded"}]
        }}
        actions = ImprovementFeedbackRules.process("gpt_signal_advisor", result)
        assert len(actions) == 1
        assert actions[0].action_type == "RETRAIN_MODEL"

    def test_quality_F_blocks_entry(self):
        from core.agent_feedback import SignalFeedbackRules
        result = {"status": "success", "output": {"quality_grade": "F", "pair_id": "XLI/XLB"}}
        actions = SignalFeedbackRules.process("signal_analyst", result)
        assert len(actions) == 1
        assert actions[0].action_type == "BLOCK_ENTRY"

    def test_empty_output_no_actions(self):
        from core.agent_feedback import SignalFeedbackRules, RiskFeedbackRules
        for agent in ["regime_surveillance", "signal_analyst", "exit_oversight"]:
            actions = SignalFeedbackRules.process(agent, {"status": "success", "output": {}})
            assert isinstance(actions, list)
        for agent in ["drawdown_monitor", "kill_switch", "exposure_monitor"]:
            actions = RiskFeedbackRules.process(agent, {"status": "success", "output": {}})
            assert isinstance(actions, list)


class TestFeedbackLoopEngine:
    """Test the AgentFeedbackLoop orchestrator."""

    def _make_result(self, name, status="success", output=None):
        return SimpleNamespace(task_name=name, status=status, output=output or {})

    def test_process_empty_results(self):
        from core.agent_feedback import AgentFeedbackLoop
        loop = AgentFeedbackLoop()
        actions = loop.process_agent_results([])
        assert actions == []

    def test_process_mixed_results(self):
        from core.agent_feedback import AgentFeedbackLoop
        loop = AgentFeedbackLoop()
        results = [
            self._make_result("agent_regime_surveillance", output={"current_regime": "CRISIS"}),
            self._make_result("agent_drawdown_monitor", output={"current_dd_pct": -0.15}),
            self._make_result("health_check"),  # Not an agent — should be ignored
        ]
        actions = loop.process_agent_results(results)
        assert len(actions) >= 3  # CRISIS → 2 actions + DD → 1 action

    def test_actions_sorted_by_severity(self):
        from core.agent_feedback import AgentFeedbackLoop
        loop = AgentFeedbackLoop()
        results = [
            self._make_result("agent_regime_surveillance", output={"current_regime": "TENSION"}),
            self._make_result("agent_drawdown_monitor", output={"current_drawdown": -0.25}),
        ]
        actions = loop.process_agent_results(results)
        severities = [a.severity for a in actions]
        # EMERGENCY should come before WARNING
        if "EMERGENCY" in severities and "WARNING" in severities:
            assert severities.index("EMERGENCY") < severities.index("WARNING")

    def test_dry_run_blocks_execution(self):
        from core.agent_feedback import AgentFeedbackLoop, FeedbackAction
        loop = AgentFeedbackLoop(dry_run=True)
        actions = [FeedbackAction(
            action_id="test", source_agent="test", action_type="KILL_SWITCH",
            severity="EMERGENCY", target="system",
        )]
        summary = loop.execute_actions(actions)
        assert summary.n_actions_executed == 0
        assert summary.n_actions_blocked == 1

    def test_execute_actions_returns_summary(self):
        from core.agent_feedback import AgentFeedbackLoop
        loop = AgentFeedbackLoop(dry_run=True)
        summary = loop.execute_actions([])
        assert summary.n_actions_generated == 0
        assert summary.n_actions_executed == 0


class TestCycleDetector:
    """Test cycle_detector integration."""

    def test_import(self):
        from core.cycle_detector import CycleDetector
        assert CycleDetector is not None

    def test_detects_known_cycle(self):
        import numpy as np, pandas as pd
        from core.cycle_detector import CycleDetector
        rng = np.random.default_rng(42)
        t = np.arange(500)
        spread = np.sin(2 * np.pi * t / 20) + 0.2 * rng.normal(0, 1, 500)
        s = pd.Series(spread, index=pd.date_range("2020-01-01", periods=500, freq="B"))
        result = CycleDetector().analyze(s)
        assert result.is_cyclical
        assert 15 < result.dominant_period < 30  # Should detect ~20 day cycle


class TestOptimalExit:
    """Test optimal_exit integration."""

    def test_import(self):
        from core.optimal_exit import OptimalExitEngine
        assert OptimalExitEngine is not None

    def test_stop_loss_triggers(self):
        from core.optimal_exit import OptimalExitEngine
        engine = OptimalExitEngine(half_life=15, entry_z=2.0, stop_z=4.0)
        engine.compute_optimal_boundary()
        signal = engine.compute_exit_signal(current_z=5.0, holding_days=3)
        assert signal.should_exit
        assert "STOP_LOSS" in signal.reason

    def test_mean_reverted_triggers_exit(self):
        from core.optimal_exit import OptimalExitEngine
        engine = OptimalExitEngine(half_life=15, entry_z=2.0, exit_z=0.5)
        engine.compute_optimal_boundary()
        signal = engine.compute_exit_signal(current_z=0.3, holding_days=10)
        assert signal.should_exit
        assert "MEAN_REVERTED" in signal.reason

    def test_crisis_regime_more_aggressive(self):
        from core.optimal_exit import OptimalExitEngine
        engine = OptimalExitEngine(half_life=15, entry_z=2.0)
        engine.compute_optimal_boundary()
        normal = engine.compute_exit_signal(current_z=1.0, holding_days=5, regime="NORMAL")
        crisis = engine.compute_exit_signal(current_z=1.0, holding_days=5, regime="CRISIS")
        assert crisis.exit_score >= normal.exit_score
        assert crisis.regime_factor > normal.regime_factor
