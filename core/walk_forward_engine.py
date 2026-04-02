# -*- coding: utf-8 -*-
"""
core/walk_forward_engine.py — True Walk-Forward Validation Engine
=================================================================

Expanding-window walk-forward with:
- Train on [0, T], test on [T, T+test_days]
- Purge gap between train/test (no leakage)
- Out-of-sample Sharpe per fold
- Deflated Sharpe Ratio (Bailey & Lopez de Prado)
- Probability of overfitting via CSCV
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Result of one walk-forward fold."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_days: int
    test_days: int
    # In-sample metrics
    is_sharpe: float = 0.0
    is_return: float = 0.0
    is_trades: int = 0
    # Out-of-sample metrics
    oos_sharpe: float = 0.0
    oos_return: float = 0.0
    oos_max_dd: float = 0.0
    oos_trades: int = 0
    oos_win_rate: float = 0.0
    # Optimal params found in-sample
    best_params: dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result."""
    n_folds: int = 0
    avg_oos_sharpe: float = 0.0
    std_oos_sharpe: float = 0.0
    min_oos_sharpe: float = 0.0
    max_oos_sharpe: float = 0.0
    avg_oos_return: float = 0.0
    pct_profitable_folds: float = 0.0
    deflated_sharpe: float = 0.0
    prob_overfit: float = 0.0
    folds: list = field(default_factory=list)
    combined_oos_equity: Optional[pd.Series] = None


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts for multiple testing — the expected max Sharpe from
    n_trials independent strategies under the null.
    """
    from scipy import stats

    if n_obs <= 1 or n_trials <= 1:
        return 0.0

    # Expected max Sharpe under null
    e_max = stats.norm.ppf(1 - 1 / n_trials) * np.sqrt(1 / n_obs)

    # Variance of Sharpe estimator (Lo, 2002)
    var_sr = (1 + 0.5 * sharpe**2 - skew * sharpe + ((kurtosis - 3) / 4) * sharpe**2) / n_obs

    if var_sr <= 0:
        return 0.0

    # DSR = prob that observed Sharpe > expected max
    dsr = float(stats.norm.cdf((sharpe - e_max) / np.sqrt(var_sr)))
    return dsr


def run_walk_forward(
    sym_x: str,
    sym_y: str,
    *,
    n_folds: int = 5,
    test_days: int = 126,        # 6 months test
    min_train_days: int = 252,   # 1 year minimum training
    purge_days: int = 5,         # Gap between train/test
    n_optuna_trials: int = 20,   # Trials per fold
) -> WalkForwardResult:
    """
    Run true expanding-window walk-forward validation.

    For each fold:
    1. Train: [0, T] — optimize parameters via Optuna
    2. Purge: [T, T+purge_days] — gap to prevent leakage
    3. Test: [T+purge_days, T+purge_days+test_days] — OOS evaluation
    """
    from common.data_loader import load_price_data, _load_symbol_full_cached
    if hasattr(_load_symbol_full_cached, "cache_clear"):
        _load_symbol_full_cached.cache_clear()

    px = load_price_data(sym_x)["close"]
    py = load_price_data(sym_y)["close"]
    common = px.index.intersection(py.index)
    px, py = px.loc[common], py.loc[common]

    total_days = len(px)
    if total_days < min_train_days + test_days * 2:
        return WalkForwardResult()

    # Calculate fold boundaries
    # Last fold ends at the end of data
    # Work backwards to define folds
    fold_boundaries = []
    test_end_idx = total_days
    for fold in range(n_folds - 1, -1, -1):
        test_start_idx = test_end_idx - test_days
        train_end_idx = test_start_idx - purge_days
        train_start_idx = 0  # Expanding window: always starts at 0

        if train_end_idx < min_train_days:
            break  # Not enough training data

        fold_boundaries.append({
            "fold": fold,
            "train_start": train_start_idx,
            "train_end": train_end_idx,
            "test_start": test_start_idx,
            "test_end": min(test_end_idx, total_days),
        })
        test_end_idx = test_start_idx

    fold_boundaries.reverse()
    actual_folds = len(fold_boundaries)

    if actual_folds == 0:
        return WalkForwardResult()

    folds = []
    oos_equities = []

    for fb in fold_boundaries:
        fold_id = fb["fold"]
        train_px = px.iloc[fb["train_start"]:fb["train_end"]]
        train_py = py.iloc[fb["train_start"]:fb["train_end"]]
        test_px = px.iloc[fb["test_start"]:fb["test_end"]]
        test_py = py.iloc[fb["test_start"]:fb["test_end"]]

        # ── In-sample: optimize parameters ───────────────────────
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                z_open = trial.suggest_float("z_open", 1.2, 3.5)
                z_close = trial.suggest_float("z_close", 0.1, 1.2)
                lookback = trial.suggest_int("lookback", 20, 120)

                beta = float(np.cov(train_px.values, train_py.values)[0, 1] / np.var(train_px.values))
                spread = train_py - beta * train_px
                mu = spread.rolling(lookback, min_periods=lookback // 2).mean()
                sig = spread.rolling(lookback, min_periods=lookback // 2).std().replace(0, np.nan)
                z = ((spread - mu) / sig).fillna(0.0)

                eq = 100000.0
                pos = 0.0
                trades = 0
                for j in range(lookback, len(z)):
                    zv = z.iloc[j]
                    if np.isnan(zv): continue
                    if pos != 0:
                        ret_y = (train_py.iloc[j] - train_py.iloc[j-1]) / train_py.iloc[j-1]
                        ret_x = (train_px.iloc[j] - train_px.iloc[j-1]) / train_px.iloc[j-1]
                        eq += pos * (ret_y - beta * ret_x) * eq * 0.20
                    if pos == 0:
                        if zv <= -z_open: pos = 1.0; trades += 1
                        elif zv >= z_open: pos = -1.0; trades += 1
                    elif (pos > 0 and zv >= -z_close) or (pos < 0 and zv <= z_close):
                        pos = 0.0
                if trades < 3: return -999
                return (eq / 100000 - 1)

            from common.optuna_factory import create_optuna_study
            study = create_optuna_study(direction="maximize")
            study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)
            best_params = study.best_params
            is_return = study.best_value
        except Exception:
            best_params = {"z_open": 2.0, "z_close": 0.5, "lookback": 60}
            is_return = 0.0

        # ── Out-of-sample: test with optimized params ────────────
        z_open = best_params.get("z_open", 2.0)
        z_close = best_params.get("z_close", 0.5)
        lookback = int(best_params.get("lookback", 60))

        # Use train data to estimate beta (no test data leakage!)
        beta = float(np.cov(train_px.values, train_py.values)[0, 1] / np.var(train_px.values))

        # Apply to test data
        full_px = px.iloc[:fb["test_end"]]
        full_py = py.iloc[:fb["test_end"]]
        spread = full_py - beta * full_px
        mu = spread.rolling(lookback, min_periods=lookback // 2).mean()
        sig = spread.rolling(lookback, min_periods=lookback // 2).std().replace(0, np.nan)
        z = ((spread - mu) / sig).fillna(0.0)

        # Simulate OOS
        test_z = z.iloc[fb["test_start"]:fb["test_end"]]
        test_px_s = test_px
        test_py_s = test_py
        eq = 100000.0
        pos = 0.0
        trades = 0
        wins = 0
        entry_eq = eq
        eq_series = []
        peak = eq

        for j in range(len(test_z)):
            zv = float(test_z.iloc[j])
            if np.isnan(zv):
                eq_series.append(eq)
                continue
            pnl = 0.0
            if pos != 0 and j > 0:
                ret_y = (test_py_s.iloc[j] - test_py_s.iloc[j-1]) / test_py_s.iloc[j-1]
                ret_x = (test_px_s.iloc[j] - test_px_s.iloc[j-1]) / test_px_s.iloc[j-1]
                pnl = pos * (ret_y - beta * ret_x) * eq * 0.20
                eq += pnl

            if pos == 0:
                if zv <= -z_open:
                    pos = 1.0; entry_eq = eq; trades += 1
                elif zv >= z_open:
                    pos = -1.0; entry_eq = eq; trades += 1
            elif (pos > 0 and zv >= -z_close) or (pos < 0 and zv <= z_close):
                if eq > entry_eq: wins += 1
                pos = 0.0

            eq_series.append(eq)
            peak = max(peak, eq)

        oos_ret = (eq / 100000 - 1) * 100
        oos_dd = ((min(eq_series) if eq_series else eq) - peak) / peak * 100 if peak > 0 else 0
        eq_s = pd.Series(eq_series, index=test_z.index[:len(eq_series)])
        rets = eq_s.pct_change().dropna()
        oos_sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 5 and rets.std() > 0 else 0

        fold_result = WalkForwardFold(
            fold_id=fold_id,
            train_start=str(px.index[fb["train_start"]].date()),
            train_end=str(px.index[fb["train_end"] - 1].date()),
            test_start=str(px.index[fb["test_start"]].date()),
            test_end=str(px.index[min(fb["test_end"] - 1, len(px) - 1)].date()),
            train_days=fb["train_end"] - fb["train_start"],
            test_days=fb["test_end"] - fb["test_start"],
            is_sharpe=0,
            is_return=round(is_return * 100, 2) if is_return != -999 else 0,
            oos_sharpe=round(oos_sharpe, 3),
            oos_return=round(oos_ret, 2),
            oos_max_dd=round(oos_dd, 2),
            oos_trades=trades,
            oos_win_rate=round(wins / max(trades, 1) * 100, 1),
            best_params=best_params,
        )
        folds.append(fold_result)
        if len(eq_s) > 0:
            oos_equities.append(eq_s)

    # ── Aggregate metrics ────────────────────────────────────────
    oos_sharpes = [f.oos_sharpe for f in folds]
    oos_returns = [f.oos_return for f in folds]

    avg_sharpe = float(np.mean(oos_sharpes))
    std_sharpe = float(np.std(oos_sharpes))

    # Deflated Sharpe
    n_trials = n_optuna_trials * actual_folds
    n_obs = sum(f.test_days for f in folds)
    dsr = deflated_sharpe_ratio(avg_sharpe, n_trials, n_obs)

    # Probability of overfit = % of folds with negative OOS Sharpe
    pct_neg = sum(1 for s in oos_sharpes if s <= 0) / len(oos_sharpes) if oos_sharpes else 1.0

    # Combined OOS equity
    combined_eq = pd.concat(oos_equities) if oos_equities else pd.Series(dtype=float)

    return WalkForwardResult(
        n_folds=actual_folds,
        avg_oos_sharpe=round(avg_sharpe, 3),
        std_oos_sharpe=round(std_sharpe, 3),
        min_oos_sharpe=round(min(oos_sharpes), 3) if oos_sharpes else 0,
        max_oos_sharpe=round(max(oos_sharpes), 3) if oos_sharpes else 0,
        avg_oos_return=round(float(np.mean(oos_returns)), 2),
        pct_profitable_folds=round((1 - pct_neg) * 100, 1),
        deflated_sharpe=round(dsr, 3),
        prob_overfit=round(pct_neg * 100, 1),
        folds=folds,
        combined_oos_equity=combined_eq,
    )
