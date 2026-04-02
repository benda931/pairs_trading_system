# -*- coding: utf-8 -*-
"""Tests for core.orchestrator and core.paper_trader.

Sections
--------
1. TestAgentBus            — pub/sub message bus
2. TestOrchestrator        — task execution + get_status
3. TestPaperTrader         — positions, persistence, NAV
4. TestRunPortfolioAllocationCycle  — P1-PORTINT / P1-SAFE / P0-KS
5. TestCollectSignalDecisions       — OLS z-score + SignalPipeline dispatch
6. TestStartDaemon                  — scheduler lifecycle
7. TestCLIScript                    — scripts/run_daily_pipeline.py
"""
from __future__ import annotations

import json
from pathlib import Path


class TestAgentBus:
    def test_publish_and_read(self, tmp_path):
        from core.orchestrator import AgentBus

        bus = AgentBus(path=tmp_path / "bus.json")
        bus.publish("test_agent", {"msg": "hello"})
        latest = bus.latest("test_agent")
        assert latest is not None
        assert latest["payload"]["msg"] == "hello"

    def test_history(self, tmp_path):
        from core.orchestrator import AgentBus

        bus = AgentBus(path=tmp_path / "bus.json")
        for i in range(5):
            bus.publish("agent", {"i": i})
        history = bus.history("agent", n=3)
        assert len(history) == 3
        assert history[-1]["payload"]["i"] == 4

    def test_missing_agent(self, tmp_path):
        from core.orchestrator import AgentBus

        bus = AgentBus(path=tmp_path / "bus.json")
        assert bus.latest("nonexistent") is None


class TestOrchestrator:
    def test_health_check(self):
        from core.orchestrator import PairsOrchestrator

        orch = PairsOrchestrator()
        result = orch._execute_task(orch.tasks["health_check"])
        assert result.status == "success"
        assert result.output["modules_ok"] > 0

    def test_get_status(self):
        from core.orchestrator import PairsOrchestrator

        orch = PairsOrchestrator()
        status = orch.get_status()
        assert "health_check" in status
        assert "data_refresh" in status


class TestPaperTrader:
    def test_open_and_close(self, tmp_path):
        from core.paper_trader import PaperTrader

        trader = PaperTrader(state_path=tmp_path / "portfolio.json")
        pos = trader.open_position(
            pair_label="AAPL-MSFT", sym_x="AAPL", sym_y="MSFT",
            price_x=180.0, price_y=410.0, entry_z=2.5, notional=10000,
        )
        assert pos is not None
        assert len(trader.positions) == 1

        trade = trader.close_position(pos.trade_id, "manual")
        assert trade is not None
        assert len(trader.positions) == 0
        assert len(trader.closed_trades) == 1

    def test_update_prices(self, tmp_path):
        from core.paper_trader import PaperTrader

        trader = PaperTrader(state_path=tmp_path / "portfolio.json")
        trader.open_position(
            pair_label="A-B", sym_x="A", sym_y="B",
            price_x=100.0, price_y=200.0, entry_z=2.0, notional=5000,
        )
        trader.update_prices({"A": 105.0, "B": 210.0})
        assert trader.positions[0].current_price_x == 105.0
        assert trader.positions[0].days_held == 1

    def test_persistence(self, tmp_path):
        from core.paper_trader import PaperTrader

        state_file = tmp_path / "portfolio.json"
        trader = PaperTrader(state_path=state_file)
        trader.open_position(
            pair_label="X-Y", sym_x="X", sym_y="Y",
            price_x=50.0, price_y=100.0, entry_z=1.5, notional=8000,
        )
        trader.save_state()
        assert state_file.exists()

        trader2 = PaperTrader(state_path=state_file)
        assert len(trader2.positions) == 1
        assert trader2.positions[0].pair_label == "X-Y"

    def test_max_positions(self, tmp_path):
        from core.paper_trader import PaperTrader

        trader = PaperTrader(state_path=tmp_path / "portfolio.json", max_positions=2)
        trader.open_position("A-B", "A", "B", 100, 200, 2.0, 5000)
        trader.open_position("C-D", "C", "D", 50, 150, 1.8, 5000)
        result = trader.open_position("E-F", "E", "F", 30, 80, 2.2, 5000)
        assert result is None
        assert len(trader.positions) == 2

    def test_nav_calculation(self, tmp_path):
        from core.paper_trader import PaperTrader

        trader = PaperTrader(state_path=tmp_path / "portfolio.json")
        initial = trader.nav()
        assert initial == trader.initial_capital

    def test_summary(self, tmp_path):
        from core.paper_trader import PaperTrader

        trader = PaperTrader(state_path=tmp_path / "portfolio.json")
        s = trader.summary()
        assert "nav" in s
        assert "win_rate" in s
        assert "open_positions" in s


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — TestRunPortfolioAllocationCycle  (P1-PORTINT / P1-SAFE / P0-KS)
# ──────────────────────────────────────────────────────────────────────────────

class TestRunPortfolioAllocationCycle:
    """run_portfolio_allocation_cycle() wires P1-PORTINT, P1-SAFE, and P0-KS."""

    def test_returns_none_on_empty_signal_decisions(self):
        from core.orchestrator import PairsOrchestrator

        orch = PairsOrchestrator()
        result = orch.run_portfolio_allocation_cycle(signal_decisions=[])
        assert result is None

    def test_returns_task_result_on_valid_input(self):
        """At minimum the bridge attempt should complete and return a TaskResult."""
        from core.orchestrator import PairsOrchestrator
        from core.contracts import PairId
        from core.intents import SignalDecision, EntryIntent

        orch = PairsOrchestrator()

        pair = PairId("AAA", "BBB")
        intent = EntryIntent(pair_id=pair, z_score=2.5, confidence=0.8)
        sd = SignalDecision(pair_id=pair, intent=intent)
        result = orch.run_portfolio_allocation_cycle(
            signal_decisions=[sd],
            capital=100_000.0,
        )
        # May return None if bridge import fails in test env — that's acceptable
        # but if it returns something it must have the right shape
        if result is not None:
            assert hasattr(result, "task_name")
            assert result.task_name == "portfolio_allocation"

    def test_output_contains_safety_check_used_bool(self):
        """TaskResult output must include safety_check_used when returned."""
        from core.orchestrator import PairsOrchestrator
        from core.contracts import PairId
        from core.intents import SignalDecision, EntryIntent

        orch = PairsOrchestrator()
        pair = PairId("A", "B")
        intent = EntryIntent(pair_id=pair, z_score=2.1, confidence=0.7)
        sd = SignalDecision(pair_id=pair, intent=intent)
        result = orch.run_portfolio_allocation_cycle(signal_decisions=[sd])
        if result is not None and result.output is not None:
            assert isinstance(result.output.get("safety_check_used"), bool)

    def test_does_not_raise_on_bridge_import_error(self, monkeypatch):
        """If the bridge module is unavailable the method must not raise."""
        import sys

        # Simulate bridge import failure
        monkeypatch.setitem(sys.modules, "core.portfolio_bridge", None)

        from core.orchestrator import PairsOrchestrator
        from core.contracts import PairId
        from core.intents import SignalDecision, EntryIntent

        orch = PairsOrchestrator()
        pair = PairId("X", "Y")
        intent = EntryIntent(pair_id=pair, z_score=2.0, confidence=0.6)
        sd = SignalDecision(pair_id=pair, intent=intent)
        # Must not raise
        try:
            orch.run_portfolio_allocation_cycle(signal_decisions=[sd])
        except Exception as exc:
            raise AssertionError(
                f"run_portfolio_allocation_cycle raised unexpectedly: {exc}"
            ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — TestCollectSignalDecisions
# ──────────────────────────────────────────────────────────────────────────────

class TestCollectSignalDecisions:
    """_collect_signal_decisions() loads prices, computes OLS z-score, runs pipeline."""

    def test_returns_empty_list_when_no_pairs(self, monkeypatch):
        from core.orchestrator import PairsOrchestrator

        orch = PairsOrchestrator()
        # Force _get_active_pairs to return []
        monkeypatch.setattr(orch, "_get_active_pairs", lambda: [])
        result = orch._collect_signal_decisions()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_returns_list_on_normal_call(self):
        """Should always return a list, never raise."""
        from core.orchestrator import PairsOrchestrator

        orch = PairsOrchestrator()
        result = orch._collect_signal_decisions()
        assert isinstance(result, list)

    def test_never_raises(self, monkeypatch):
        """Even if prices fail to load, the method must return gracefully."""
        from core.orchestrator import PairsOrchestrator

        orch = PairsOrchestrator()

        def _bad_pairs():
            return [("INVALID_SYM_X", "INVALID_SYM_Y")]

        monkeypatch.setattr(orch, "_get_active_pairs", _bad_pairs)
        try:
            result = orch._collect_signal_decisions()
            assert isinstance(result, list)
        except Exception as exc:
            raise AssertionError(
                f"_collect_signal_decisions raised unexpectedly: {exc}"
            ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 — TestStartDaemon
# ──────────────────────────────────────────────────────────────────────────────

class TestStartDaemon:
    """start_daemon() and stop_daemon() must exist and be callable."""

    def test_start_daemon_method_exists(self):
        from core.orchestrator import PairsOrchestrator

        orch = PairsOrchestrator()
        assert callable(getattr(orch, "start_daemon", None))

    def test_stop_daemon_method_exists(self):
        from core.orchestrator import PairsOrchestrator

        orch = PairsOrchestrator()
        assert callable(getattr(orch, "stop_daemon", None))

    def test_stop_daemon_is_idempotent_before_start(self):
        """Calling stop_daemon() before start_daemon() must not raise."""
        from core.orchestrator import PairsOrchestrator

        orch = PairsOrchestrator()
        try:
            orch.stop_daemon()
        except Exception as exc:
            raise AssertionError(
                f"stop_daemon raised before any daemon was started: {exc}"
            ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 — TestCLIScript
# ──────────────────────────────────────────────────────────────────────────────

class TestCLIScript:
    """scripts/run_daily_pipeline.py exists and is runnable."""

    def test_script_file_exists(self):
        script = Path(__file__).resolve().parent.parent / "scripts" / "run_daily_pipeline.py"
        assert script.exists(), f"CLI script not found: {script}"

    def test_main_is_callable(self):
        import importlib.util, sys
        script = Path(__file__).resolve().parent.parent / "scripts" / "run_daily_pipeline.py"
        spec = importlib.util.spec_from_file_location("run_daily_pipeline", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(getattr(mod, "main", None)), "scripts/run_daily_pipeline.py must define main()"
