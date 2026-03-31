# -*- coding: utf-8 -*-
"""Tests for core.risk_engine."""
from __future__ import annotations


def test_risk_limits_defaults():
    from core.risk_engine import RiskLimits

    limits = RiskLimits()
    assert limits.max_net_leverage > 0
    assert 0 < limits.max_drawdown_pct <= 1


def test_risk_state_creation():
    from core.risk_engine import RiskState

    state = RiskState()
    assert state is not None


def test_evaluate_kill_switch():
    from core.risk_engine import evaluate_kill_switch, RiskLimits, RiskState

    state = RiskState()
    limits = RiskLimits()
    decision = evaluate_kill_switch(state, limits)
    assert decision is not None


def test_compute_overall_risk_score():
    from core.risk_engine import compute_overall_risk_score, RiskLimits, RiskState

    state = RiskState()
    limits = RiskLimits()
    score = compute_overall_risk_score(state, limits)
    assert isinstance(score, (int, float))


def test_risk_limits_to_dict():
    from core.risk_engine import RiskLimits

    limits = RiskLimits()
    d = limits.to_dict()
    assert isinstance(d, dict)
    assert "max_net_leverage" in d
