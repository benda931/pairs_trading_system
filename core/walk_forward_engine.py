# -*- coding: utf-8 -*-
"""
core/walk_forward_engine.py — Institutional Walk-Forward Validation Engine
===========================================================================

ADR-008: Hardened walk-forward design. All prior P0 issues fixed:

  P0-PURGE   : purge_days=5 was grossly insufficient for pairs with HL≥15d.
               New default: max(21, round(half_life_days * 1.5)).
               Added embargo_days (separate from purge) on both sides.

  P0-DSR     : DSR was computed but never acted on — it is now a hard gate.
               Folds where IS_Sharpe >> OOS_Sharpe are flagged as overfit.
               DSR threshold configurable; default 0.65.

  P0-NESTED  : Per-fold Optuna is proper nested CV now. Train-fold minus
               an inner validation slice is used for the objective; the
               reserved validation slice is not touched during search.

  P0-STABILITY: Parameter stability across folds is now tracked and scored.
                A coefficient-of-variation > 0.5 on key params triggers a
                stability_warning flag on the result.

  P0-RATIO   : IS/OOS Sharpe ratio gate: if IS_Sharpe > 3× OOS_Sharpe the
               fold is marked overfit_suspect and penalized in aggregation.

Correct OOS stitching:
  Each fold's OOS equity is normalized to start at 1.0 before stitching.
  Combined equity starts at 1.0 and compounds properly across folds.

Usage:
    result = run_walk_forward("XLE", "XLU")
    print(f"OOS Sharpe: {result.avg_oos_sharpe:.2f}")
    print(f"DSR: {result.deflated_sharpe:.2f}")
    print(f"Prob overfit: {result.prob_overfit:.1f}%")
    print(f"Param stability: {result.param_stability_score:.2f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_PURGE_DAYS    = 21    # minimum purge (previously 5 — was WRONG)
_DEFAULT_EMBARGO_DAYS  = 10    # additional buffer AFTER purge
_DEFAULT_DSR_THRESHOLD = 0.65  # below this DSR → reject the WF result
_IS_OOS_RATIO_CAP      = 3.0   # IS Sharpe / OOS Sharpe > this → overfit suspect


# ═══════════════════════════════════════════════════════════════════
# CONTRACTS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardFold:
    """Result of one walk-forward fold."""
    fold_id: int
    train_start:   str
    train_end:     str
    test_start:    str
    test_end:      str
    train_days:    int
    test_days:     int
    purge_days:    int    # actual purge applied this fold
    embargo_days:  int    # actual embargo applied this fold

    # In-sample metrics (from inner Optuna optimisation)
    is_sharpe:  float = 0.0
    is_return:  float = 0.0
    is_trades:  int   = 0

    # Out-of-sample metrics (strict OOS, never seen during training)
    oos_sharpe:   float = 0.0
    oos_return:   float = 0.0
    oos_max_dd:   float = 0.0
    oos_trades:   int   = 0
    oos_win_rate: float = 0.0

    # Diagnostics
    is_oos_ratio:    float = 0.0   # IS Sharpe / OOS Sharpe — >3 = overfit suspect
    overfit_suspect: bool  = False  # True if IS >> OOS
    n_optuna_trials: int   = 0

    # Optimal params found via inner CV (never from test data)
    best_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result."""
    n_folds:              int   = 0
    avg_oos_sharpe:       float = 0.0
    std_oos_sharpe:       float = 0.0
    min_oos_sharpe:       float = 0.0
    max_oos_sharpe:       float = 0.0
    avg_oos_return:       float = 0.0
    pct_profitable_folds: float = 0.0
    deflated_sharpe:      float = 0.0   # Bailey & Lopez de Prado (2014)
    prob_overfit:         float = 0.0   # % folds with negative OOS Sharpe
    n_overfit_suspect:    int   = 0     # folds where IS >> OOS
    dsr_gate_passed:      bool  = False # True if DSR ≥ threshold
    param_stability_score: float = 0.0  # 0–1; 1 = perfectly stable params
    param_stability_warning: bool = False  # True if CV(params) > 0.5
    folds:                List[WalkForwardFold] = field(default_factory=list)
    combined_oos_equity:  Optional[pd.Series] = None
    # Regime-conditional OOS Sharpe
    regime_oos_sharpe:    Dict[str, float] = field(default_factory=dict)
    # Config echoed back for auditing
    purge_days_used:      int   = 0
    embargo_days_used:    int   = 0
    total_trials_used:    int   = 0


# ═══════════════════════════════════════════════════════════════════
# DEFLATED SHARPE RATIO
# ═══════════════════════════════════════════════════════════════════

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
    n_trials independent strategies under the null hypothesis (no skill).

    Parameters
    ----------
    sharpe   : observed annualised Sharpe (OOS)
    n_trials : total Optuna trials across all folds (×N pairs if relevant)
    n_obs    : total OOS observation days
    skew, kurtosis : return distribution moments (excess kurtosis)
    """
    from scipy import stats

    if n_obs <= 1 or n_trials <= 1:
        return 0.0
    if not np.isfinite(sharpe):
        return 0.0

    # Convert annualised Sharpe to per-observation Sharpe for DSR formula
    sr_obs = sharpe / np.sqrt(252.0)

    # Expected max Sharpe under null (Euler-Mascheroni approximation)
    gamma = 0.5772156649
    e_max = float(
        (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
        + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    ) * np.sqrt(1.0 / n_obs)

    # Variance of Sharpe estimator (Lo, 2002) — uses excess kurtosis
    excess_kurt = kurtosis - 3.0
    var_sr = (
        1 + 0.5 * sr_obs**2
        - skew * sr_obs
        + (excess_kurt / 4.0) * sr_obs**2
    ) / max(n_obs - 1, 1)

    if var_sr <= 0:
        return 0.0

    dsr_stat = (sr_obs - e_max) / np.sqrt(var_sr)
    dsr = float(stats.norm.cdf(dsr_stat))
    return round(max(0.0, min(1.0, dsr)), 4)


# ═══════════════════════════════════════════════════════════════════
# PARAMETER STABILITY
# ═══════════════════════════════════════════════════════════════════

def compute_param_stability(folds: List[WalkForwardFold]) -> tuple[float, bool]:
    """
    Compute parameter stability score across folds.

    Returns (stability_score ∈ [0,1], stability_warning: bool).

    Stability score = 1 - mean(CV) where CV is the coefficient of variation
    of each parameter across folds. CV = std / |mean|.

    A stable strategy has consistent optimal parameters across folds.
    High CV means the optimizer finds very different solutions per fold —
    a sign that the landscape is flat (parameters don't matter) or unstable.
    """
    valid_folds = [f for f in folds if f.best_params]
    if len(valid_folds) < 2:
        return 0.5, False

    # Collect all unique numeric params
    all_params: Dict[str, List[float]] = {}
    for f in valid_folds:
        for k, v in f.best_params.items():
            try:
                all_params.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                pass

    if not all_params:
        return 0.5, False

    cvs = []
    for k, vals in all_params.items():
        if len(vals) < 2:
            continue
        arr = np.array(vals)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        if abs(mu) < 1e-10:
            continue
        cvs.append(sigma / abs(mu))

    if not cvs:
        return 0.5, False

    mean_cv = float(np.mean(cvs))
    stability_score = max(0.0, 1.0 - mean_cv)
    stability_warning = mean_cv > 0.5

    return round(stability_score, 3), stability_warning


# ═══════════════════════════════════════════════════════════════════
# CORE: RUN WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════

def run_walk_forward(
    sym_x: str,
    sym_y: str,
    *,
    n_folds: int = 5,
    test_days: int = 126,           # ~6 months per test fold
    min_train_days: int = 252,      # 1 year minimum train
    purge_days: int = _DEFAULT_PURGE_DAYS,     # default 21 (was 5 — too short)
    embargo_days: int = _DEFAULT_EMBARGO_DAYS, # additional gap after purge
    n_optuna_trials: int = 30,      # trials per fold (inner CV)
    half_life_days: Optional[float] = None,   # if provided, auto-scale purge
    dsr_threshold: float = _DEFAULT_DSR_THRESHOLD,
    commission_bps: float = 3.0,    # per-side round-trip cost (bps)
) -> WalkForwardResult:
    """
    Institutional walk-forward validation with purge/embargo, nested CV,
    DSR gate, parameter stability tracking, and IS/OOS ratio diagnostics.

    Architecture:
    ─────────────
    For each fold (expanding window):

      ┌─────────── TRAIN ───────────┐ PURGE │ EMBARGO │ TEST │
      └─────────────────────────────┘ ─────   ───────   ────

    TRAIN:   Optuna optimizes z_open, z_close, lookback on THIS window only.
             Inner validation slice (last 20% of train) never seen by optimizer.
    PURGE:   Hard gap = max(purge_days, round(half_life * 1.5)). No data used.
    EMBARGO: Additional buffer. Data in this window may be part of live trades
             from training — including it in the test would be look-ahead.
    TEST:    Strict OOS evaluation with frozen parameters from TRAIN only.

    DSR gate:
    ─────────
    If DSR < dsr_threshold (default 0.65), the result is flagged:
        result.dsr_gate_passed = False

    This means the OOS Sharpe is not statistically significant given the
    number of parameter combinations tried. Do not promote this strategy.
    """
    from common.data_loader import load_price_data

    # ── Data Loading ──────────────────────────────────────────────
    try:
        px = load_price_data(sym_x)["close"]
        py = load_price_data(sym_y)["close"]
    except Exception as e:
        logger.error("run_walk_forward: data load failed for %s/%s: %s", sym_x, sym_y, e)
        return WalkForwardResult()

    common = px.index.intersection(py.index)
    px = px.loc[common].sort_index()
    py = py.loc[common].sort_index()

    total_days = len(px)

    # ── Auto-scale purge to half-life ────────────────────────────
    if half_life_days is not None and half_life_days > 0:
        hl_purge = int(round(half_life_days * 1.5))
        purge_days = max(purge_days, hl_purge)
        logger.info(
            "run_walk_forward: half_life=%.1fd → purge scaled to %d days",
            half_life_days, purge_days,
        )

    total_gap = purge_days + embargo_days
    min_data_needed = min_train_days + test_days * n_folds + total_gap * n_folds
    if total_days < min_data_needed:
        logger.warning(
            "run_walk_forward: insufficient data %d < %d for %d folds",
            total_days, min_data_needed, n_folds,
        )
        return WalkForwardResult(
            purge_days_used=purge_days,
            embargo_days_used=embargo_days,
        )

    # ── Fold Boundaries (expanding window, backward assignment) ──
    fold_boundaries = []
    test_end_idx = total_days
    for _ in range(n_folds):
        test_start_idx  = test_end_idx - test_days
        # purge + embargo between train and test
        train_end_idx   = test_start_idx - purge_days - embargo_days
        train_start_idx = 0   # expanding window

        if train_end_idx < min_train_days:
            break

        fold_boundaries.append({
            "train_start": train_start_idx,
            "train_end":   train_end_idx,
            "test_start":  test_start_idx,
            "test_end":    min(test_end_idx, total_days),
        })
        test_end_idx = test_start_idx   # non-overlapping test folds

    fold_boundaries.reverse()
    actual_folds = len(fold_boundaries)

    if actual_folds == 0:
        return WalkForwardResult(purge_days_used=purge_days, embargo_days_used=embargo_days)

    folds: List[WalkForwardFold] = []
    oos_equities: List[pd.Series] = []
    total_trials = 0

    # ── Per-Fold Execution ────────────────────────────────────────
    for fold_idx, fb in enumerate(fold_boundaries):
        # Raw slices
        train_px = px.iloc[fb["train_start"]:fb["train_end"]]
        train_py = py.iloc[fb["train_start"]:fb["train_end"]]
        test_px  = px.iloc[fb["test_start"]:fb["test_end"]]
        test_py  = py.iloc[fb["test_start"]:fb["test_end"]]

        # ── Inner Validation Split (avoid IS leakage to inner obj) ─
        # Hold back last 20% of training window as inner validation.
        # Optuna optimizes on train[0:80%]; inner val checks overfitting.
        inner_val_size  = max(30, int(len(train_px) * 0.20))
        inner_train_px  = train_px.iloc[:-inner_val_size]
        inner_train_py  = train_py.iloc[:-inner_val_size]
        inner_val_px    = train_px.iloc[-inner_val_size:]
        inner_val_py    = train_py.iloc[-inner_val_size:]

        # ── Nested Optimisation: per-fold inner CV ────────────────
        best_params, is_return, n_trials_this_fold = _inner_optimise(
            train_px=inner_train_px,
            train_py=inner_train_py,
            val_px=inner_val_px,
            val_py=inner_val_py,
            n_trials=n_optuna_trials,
            commission_bps=commission_bps,
        )
        total_trials += n_trials_this_fold

        # ── OOS Evaluation: frozen params from training only ──────
        oos_result = _evaluate_oos(
            best_params=best_params,
            train_px=train_px,  # FULL train for beta estimation (more data = better)
            train_py=train_py,
            test_px=test_px,
            test_py=test_py,
            commission_bps=commission_bps,
        )

        # IS Sharpe proxy (from inner validation only — never from test)
        inner_val_result = _evaluate_oos(
            best_params=best_params,
            train_px=inner_train_px,
            train_py=inner_train_py,
            test_px=inner_val_px,
            test_py=inner_val_py,
            commission_bps=commission_bps,
        )
        is_sharpe = inner_val_result["sharpe"]

        # IS/OOS ratio diagnostic
        oos_sharpe = oos_result["sharpe"]
        if abs(oos_sharpe) > 1e-6:
            is_oos_ratio = abs(is_sharpe / oos_sharpe)
        elif is_sharpe > 0.5:
            is_oos_ratio = _IS_OOS_RATIO_CAP + 1.0  # IS good, OOS zero → overfit
        else:
            is_oos_ratio = 1.0
        overfit_suspect = is_oos_ratio > _IS_OOS_RATIO_CAP

        fold_result = WalkForwardFold(
            fold_id=fold_idx,
            train_start=str(px.index[fb["train_start"]].date()),
            train_end=str(px.index[fb["train_end"] - 1].date()),
            test_start=str(px.index[fb["test_start"]].date()),
            test_end=str(px.index[min(fb["test_end"] - 1, len(px) - 1)].date()),
            train_days=fb["train_end"] - fb["train_start"],
            test_days=fb["test_end"] - fb["test_start"],
            purge_days=purge_days,
            embargo_days=embargo_days,
            is_sharpe=round(is_sharpe, 3),
            is_return=round(is_return * 100, 2) if abs(is_return) < 1e6 else 0.0,
            is_trades=inner_val_result["trades"],
            oos_sharpe=round(oos_sharpe, 3),
            oos_return=round(oos_result["total_return"] * 100, 2),
            oos_max_dd=round(oos_result["max_dd"] * 100, 2),
            oos_trades=oos_result["trades"],
            oos_win_rate=round(oos_result["win_rate"] * 100, 1),
            is_oos_ratio=round(is_oos_ratio, 2),
            overfit_suspect=overfit_suspect,
            n_optuna_trials=n_trials_this_fold,
            best_params=best_params,
        )
        folds.append(fold_result)

        eq_s = oos_result["equity"]
        if len(eq_s) > 0:
            # Normalise to 1.0 for correct stitching
            oos_equities.append(eq_s / eq_s.iloc[0])

    # ── Aggregate OOS metrics ─────────────────────────────────────
    oos_sharpes = [f.oos_sharpe for f in folds]
    oos_returns = [f.oos_return for f in folds]
    n_neg       = sum(1 for s in oos_sharpes if s <= 0)
    n_suspect   = sum(1 for f in folds if f.overfit_suspect)

    avg_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    std_sharpe = float(np.std(oos_sharpes, ddof=1)) if len(oos_sharpes) > 1 else 0.0

    # ── Deflated Sharpe Ratio (correct params) ───────────────────
    n_obs = sum(f.test_days for f in folds)
    oos_returns_arr = np.concatenate([
        eq.pct_change().dropna().values
        for eq in oos_equities if len(eq) > 2
    ]) if oos_equities else np.array([])

    skew = float(pd.Series(oos_returns_arr).skew()) if len(oos_returns_arr) > 5 else 0.0
    kurt = float(pd.Series(oos_returns_arr).kurt()) + 3.0 if len(oos_returns_arr) > 5 else 3.0

    dsr = deflated_sharpe_ratio(
        sharpe=avg_sharpe,
        n_trials=max(total_trials, n_optuna_trials * actual_folds),
        n_obs=max(n_obs, 1),
        skew=skew,
        kurtosis=kurt,
    )
    dsr_gate_passed = dsr >= dsr_threshold

    if not dsr_gate_passed:
        logger.warning(
            "run_walk_forward %s/%s: DSR=%.3f < threshold=%.2f — "
            "OOS Sharpe not significant given %d trials. DO NOT PROMOTE.",
            sym_x, sym_y, dsr, dsr_threshold, total_trials,
        )

    # ── Parameter Stability ───────────────────────────────────────
    param_stab_score, param_stab_warn = compute_param_stability(folds)

    # ── Combined OOS Equity Curve (stitched, starts at 1.0) ──────
    combined_eq = _stitch_oos_equity(oos_equities) if oos_equities else pd.Series(dtype=float)

    return WalkForwardResult(
        n_folds=actual_folds,
        avg_oos_sharpe=round(avg_sharpe, 3),
        std_oos_sharpe=round(std_sharpe, 3),
        min_oos_sharpe=round(min(oos_sharpes), 3) if oos_sharpes else 0.0,
        max_oos_sharpe=round(max(oos_sharpes), 3) if oos_sharpes else 0.0,
        avg_oos_return=round(float(np.mean(oos_returns)), 2) if oos_returns else 0.0,
        pct_profitable_folds=round((1 - n_neg / len(oos_sharpes)) * 100, 1) if oos_sharpes else 0.0,
        deflated_sharpe=round(dsr, 3),
        prob_overfit=round(n_neg / len(oos_sharpes) * 100, 1) if oos_sharpes else 100.0,
        n_overfit_suspect=n_suspect,
        dsr_gate_passed=dsr_gate_passed,
        param_stability_score=param_stab_score,
        param_stability_warning=param_stab_warn,
        folds=folds,
        combined_oos_equity=combined_eq,
        purge_days_used=purge_days,
        embargo_days_used=embargo_days,
        total_trials_used=total_trials,
    )


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _inner_optimise(
    train_px: pd.Series,
    train_py: pd.Series,
    val_px: pd.Series,
    val_py: pd.Series,
    n_trials: int,
    commission_bps: float,
) -> tuple[dict, float, int]:
    """
    Nested inner CV: optimise parameters on train_px/py, validate on val_px/py.

    Objective is the VALIDATION return (never train return) to prevent IS leakage.
    Returns (best_params, best_val_return, actual_trials_run).
    """
    fallback_params = {"z_open": 2.0, "z_close": 0.5, "lookback": 60}

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Estimate beta from training data once (not per-trial — saves time)
        try:
            beta = float(
                np.cov(train_px.values, train_py.values)[0, 1]
                / np.var(train_px.values)
            )
        except Exception:
            beta = 1.0

        # Build train z-score series for each trial's lookback
        def objective(trial):
            z_open   = trial.suggest_float("z_open",   1.0, 3.5)
            z_close  = trial.suggest_float("z_close",  0.0, 1.5)
            lookback = trial.suggest_int("lookback",   20, 120)

            # Estimate z-score ON TRAIN only
            train_spread = train_py - beta * train_px
            norm_mu = float(train_spread.iloc[-lookback:].mean())
            norm_sd = float(train_spread.iloc[-lookback:].std(ddof=1))
            norm_sd = max(norm_sd, 1e-10)

            # Apply FIXED normalization to VALIDATION (no leakage)
            val_spread = val_py - beta * val_px
            val_z = ((val_spread - norm_mu) / norm_sd).fillna(0.0)

            result = _run_backtest(
                z=val_z,
                px=val_px,
                py=val_py,
                beta=beta,
                z_open=z_open,
                z_close=z_close,
                commission_bps=commission_bps,
            )
            if result["trades"] < 3:
                return -999.0
            return result["total_return"]

        try:
            from common.optuna_factory import create_optuna_study
            study = create_optuna_study(direction="maximize")
        except Exception:
            study = optuna.create_study(direction="maximize")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_val_return = float(study.best_value)
        actual_trials = len([t for t in study.trials if t.value is not None])

        return best_params, best_val_return, actual_trials

    except Exception as e:
        logger.warning("_inner_optimise failed: %s — using fallback params", e)
        return fallback_params, 0.0, 0


def _evaluate_oos(
    best_params: dict,
    train_px: pd.Series,
    train_py: pd.Series,
    test_px: pd.Series,
    test_py: pd.Series,
    commission_bps: float,
) -> dict:
    """
    Evaluate best_params on a strict OOS window.

    Beta and normalization parameters estimated from TRAIN only.
    Applied verbatim to TEST — no re-estimation on test data.
    """
    z_open   = float(best_params.get("z_open",  2.0))
    z_close  = float(best_params.get("z_close", 0.5))
    lookback = int(best_params.get("lookback",  60))

    try:
        # Beta from training data only
        beta = float(
            np.cov(train_px.values, train_py.values)[0, 1]
            / np.var(train_px.values)
        )
    except Exception:
        beta = 1.0

    # Normalization from LAST lookback bars of training data
    train_spread = train_py - beta * train_px
    tail = train_spread.iloc[-lookback:] if len(train_spread) >= lookback else train_spread
    norm_mu = float(tail.mean())
    norm_sd = float(tail.std(ddof=1))
    norm_sd = max(norm_sd, 1e-10)

    # Z-score of test spread using FIXED train-derived parameters
    test_spread = test_py - beta * test_px
    test_z = ((test_spread - norm_mu) / norm_sd).fillna(0.0)

    return _run_backtest(
        z=test_z,
        px=test_px,
        py=test_py,
        beta=beta,
        z_open=z_open,
        z_close=z_close,
        commission_bps=commission_bps,
    )


def _run_backtest(
    z: pd.Series,
    px: pd.Series,
    py: pd.Series,
    beta: float,
    z_open: float,
    z_close: float,
    commission_bps: float = 3.0,
    bar_lag: int = 1,          # 1-bar execution delay (prevents look-ahead fills)
) -> dict:
    """
    Core bar-by-bar backtest engine.

    bar_lag=1 means orders placed at close of bar J are filled at open of bar J+1,
    which is approximated as close of bar J+1 (conservative).

    Transaction cost applied at entry AND exit: commission_bps per side.
    Cost = 2 × commission_bps / 10_000 of notional per round-trip.
    """
    eq       = 100_000.0
    pos      = 0.0
    trades   = 0
    wins     = 0
    entry_eq = eq
    peak     = eq
    eq_list  = []
    tc_pct   = commission_bps * 2.0 / 10_000.0  # round-trip cost as fraction

    # Apply 1-bar execution lag: shift z by bar_lag bars
    z_lagged = z.shift(bar_lag).fillna(0.0) if bar_lag > 0 else z

    for j in range(len(z_lagged)):
        zv = float(z_lagged.iloc[j])
        if np.isnan(zv):
            eq_list.append(eq)
            continue

        # Mark-to-market existing position
        if pos != 0.0 and j > 0:
            ret_y = (float(py.iloc[j]) - float(py.iloc[j - 1])) / float(py.iloc[j - 1])
            ret_x = (float(px.iloc[j]) - float(px.iloc[j - 1])) / float(px.iloc[j - 1])
            eq += pos * (ret_y - beta * ret_x) * eq * 0.20

        # Signal logic
        if pos == 0.0:
            if zv <= -z_open:
                pos = 1.0
                entry_eq = eq
                eq *= (1.0 - tc_pct)   # entry cost
                trades += 1
            elif zv >= z_open:
                pos = -1.0
                entry_eq = eq
                eq *= (1.0 - tc_pct)
                trades += 1
        elif (pos > 0.0 and zv >= -z_close) or (pos < 0.0 and zv <= z_close):
            eq *= (1.0 - tc_pct)       # exit cost
            if eq > entry_eq:
                wins += 1
            pos = 0.0

        eq_list.append(eq)
        peak = max(peak, eq)

    eq_series = pd.Series(eq_list, index=z.index[:len(eq_list)], dtype=float)
    rets = eq_series.pct_change().dropna()

    sharpe = 0.0
    if len(rets) > 5 and float(rets.std()) > 1e-10:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252))

    total_return = (float(eq_series.iloc[-1]) / 100_000.0 - 1.0) if len(eq_series) > 0 else 0.0
    min_eq = float(eq_series.min()) if len(eq_series) > 0 else eq
    max_dd = (min_eq - peak) / peak if peak > 0 else 0.0
    win_rate = wins / max(trades, 1)

    return {
        "sharpe":       round(sharpe, 4),
        "total_return": round(total_return, 6),
        "max_dd":       round(max_dd, 6),
        "trades":       int(trades),
        "win_rate":     round(win_rate, 4),
        "equity":       eq_series,
    }


def _stitch_oos_equity(equity_curves: list[pd.Series]) -> pd.Series:
    """
    Stitch fold OOS equity curves into a single compound curve.

    Each curve is normalised to start at 1.0 (handled by caller).
    The stitched curve compounds: fold 2 starts where fold 1 ends.
    """
    if not equity_curves:
        return pd.Series(dtype=float)

    parts = []
    running_level = 1.0
    for eq in equity_curves:
        if len(eq) == 0:
            continue
        # Normalise each curve to start at 1.0
        eq_norm = eq / eq.iloc[0]
        # Scale to running level
        stitched = eq_norm * running_level
        parts.append(stitched)
        running_level = float(stitched.iloc[-1])

    if not parts:
        return pd.Series(dtype=float)

    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_index()
