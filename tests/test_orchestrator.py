# -*- coding: utf-8 -*-
"""Tests for core.orchestrator and core.paper_trader."""
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
