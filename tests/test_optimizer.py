# -*- coding: utf-8 -*-
"""Tests for core.optimizer."""
from __future__ import annotations


def test_compute_weighted_score():
    from core.optimizer import compute_weighted_score

    norm = {"sharpe": 0.8, "sortino": 0.7, "calmar": 0.6}
    weights = {"sharpe": 0.5, "sortino": 0.3, "calmar": 0.2}
    score = compute_weighted_score(norm, weights)
    assert isinstance(score, (int, float))
    assert score > 0


def test_normalize_metrics():
    from core.optimizer import normalize_metrics

    raw = {"sharpe": 1.5, "sortino": 2.0, "calmar": 3.0}
    normalized = normalize_metrics(raw)
    assert isinstance(normalized, dict)


def test_rank_trials():
    from core.optimizer import rank_trials

    assert callable(rank_trials)
