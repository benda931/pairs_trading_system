#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full_parameter_optimization.py — Hedge-Fund Grade Optimizer
===========================================================

CLI entrypoint for two modes:

  • direct : single-study full-data optimization
  • wf     : walk-forward cross-validation + meta-optimization

Outputs:
  - best parameters
  - near-optimal ranges around best parameters
  - parameter importances (Optuna)
  - walk-forward window results (if mode=wf)

This script is designed to be used both as:
  - a CLI tool (python full_parameter_optimization.py --mode=direct --config=cfg.yaml)
  - an importable module (call run_direct_optimization() / run_walk_forward()).
"""

from __future__ import annotations

import os
import argparse
import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler
from optuna.importance import get_param_importances

from params import PARAM_SPECS, freeze
from common.data_loader import DataLoader
from ranges import RangeManager
from common.advanced_metrics import (
    dynamic_time_warping,
    rolling_covariance,
    distance_correlation,
    beta_market_dynamic,
    drawdown_half_life,  # not used yet, kept for extension
)

from common.signal_generator import RSIConfig, rsi_signals, BollingerConfig, bollinger_signals
from root.trade_logic import apply_transaction_costs
from root.analysis import get_walk_forward_splits
from meta_optimizer import meta_optimize

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("FullParamOpt")


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full parameter optimization")
    parser.add_argument(
        "--mode",
        choices=["direct", "wf"],
        default="direct",
        help="Operation mode: 'direct' (single run) or 'wf' (walk-forward)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )
    return parser


# ---------------------------------------------------------------------------
# Global-ish state (loaded once, reused)
# ---------------------------------------------------------------------------

_dl = DataLoader()
_rm = RangeManager()

_user_cfg: Dict[str, Any] = {}
_ranges_cfg = None  # type: ignore[assignment]
_filtered_ranges: Dict[str, Tuple[float, float, Optional[float]]] = {}
DEFAULT_VALUES: Dict[str, float] = {}
ADV_SHARPE_CONFIG: Dict[str, Any] = {}
N_TRIALS: int = 100
OUTPUT_FILE: str = "optimal_params.yaml"
SAMPLER: Optional[TPESampler] = None


# ---------------------------------------------------------------------------
# Checkpoint callback for Optuna
# ---------------------------------------------------------------------------

def checkpoint_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Save best parameters at each trial into a YAML checkpoint."""
    try:
        data = study.best_params
        checkpoint_file = _user_cfg.get("checkpoint_file", "checkpoint_best_params.yaml")
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        logger.info("Checkpoint saved at trial %s → %s", trial.number, checkpoint_file)
    except Exception as exc:
        logger.error("Checkpoint failed at trial %s: %s", trial.number, exc)


# ---------------------------------------------------------------------------
# Configuration loading & setup
# ---------------------------------------------------------------------------

def load_user_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg_path = config_path or os.getenv("OPT_CONFIG_PATH", "default.yaml")
    cfg: Dict[str, Any] = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as cf:
                cfg = yaml.safe_load(cf) or {}
            logger.info("Loaded config from %s", cfg_path)
        except Exception as exc:
            logger.warning("Failed to load config %s: %s (using defaults)", cfg_path, exc)
    else:
        logger.warning("Config file %s not found, using defaults", cfg_path)
    return cfg


def setup_ranges_and_defaults() -> None:
    """
    Load RangeManager config, build filtered ranges, default values,
    advanced Sharpe config, and global N_TRIALS / OUTPUT_FILE / SAMPLER.
    """
    global _ranges_cfg, _filtered_ranges, DEFAULT_VALUES, ADV_SHARPE_CONFIG, N_TRIALS, OUTPUT_FILE, SAMPLER

    _ranges_cfg = _rm.load_config()

    # base ranges from RangeManager config
    base_ranges = {k: (v.low, v.high, v.step) for k, v in _ranges_cfg.default_ranges.items()}

    # apply feature toggles; in this version we enable everything by default
    _filtered_ranges = _rm.filter(
        base_ranges,
        use_ecm=True,
        use_kalman=True,
        use_ml=True,
        use_meta=True,
    )

    # Optional random seed for reproducibility
    tmp_seed = _user_cfg.get("general", {}).get("seed", None)
    SAMPLER = TPESampler(seed=tmp_seed) if tmp_seed is not None else None

    # Trials & output file
    general_cfg = _user_cfg.get("general", {}) or {}
    N_TRIALS = int(general_cfg.get("n_trials", _ranges_cfg.general.n_trials or 100))
    OUTPUT_FILE = general_cfg.get(
        "output_file",
        _ranges_cfg.general.output_file or "optimal_params.yaml",
    )

    # Advanced Sharpe config
    adv_cfg = _user_cfg.get("advanced_sharpe", {}) or {}
    ADV_SHARPE_CONFIG = {
        "freq": adv_cfg.get("freq", _ranges_cfg.advanced.freq or 252),
        "rolling_window": adv_cfg.get(
            "rolling_window",
            _ranges_cfg.advanced.rolling_window or 63,
        ),
        "vol_target": adv_cfg.get("vol_target", _ranges_cfg.advanced.vol_target or 0.15),
        "downside": adv_cfg.get("downside", _ranges_cfg.advanced.downside or False),
        "dd_penalty": adv_cfg.get("dd_penalty", _ranges_cfg.advanced.dd_penalty),
        "skew_kurt_corr": adv_cfg.get(
            "skew_kurt_corr",
            _ranges_cfg.advanced.skew_kurt_corr or False,
        ),
    }

    # Build default param values (mid-range or first categorical choice)
    DEFAULT_VALUES = {
        p.name: (
            p.choices[0]
            if getattr(p, "is_categorical", False)
            else (
                (
                    _filtered_ranges.get(p.name, (p.lo, p.hi, None))[0]
                    + _filtered_ranges.get(p.name, (p.lo, p.hi, None))[1]
                )
                / 2.0
            )
        )
        for p in PARAM_SPECS
    }

    logger.info(
        "Setup ranges: %d params, N_TRIALS=%d, OUTPUT_FILE=%s",
        len(DEFAULT_VALUES),
        N_TRIALS,
        OUTPUT_FILE,
    )


# ---------------------------------------------------------------------------
# Advanced Sharpe Ratio
# ---------------------------------------------------------------------------

def compute_advanced_sharpe(
    returns: pd.Series,
    rf_rates: Optional[pd.Series] = None,
    freq: int = 252,
    rolling_window: Optional[int] = None,
    vol_target: Optional[float] = None,
    downside: bool = False,
    dd_penalty: Optional[float] = None,
    skew_kurt_corr: bool = False,
) -> float:
    """
    Compute an advanced Sharpe ratio with optional volatility targeting and penalties.

    - rf_rates: risk-free rate series aligned with returns (if not None).
    - downside: if True, use downside deviation instead of full std.
    - dd_penalty: if provided, subtract dd_penalty * |max_drawdown| from Sharpe.
    - skew_kurt_corr: skew/kurtosis correction factor on top of Sharpe.
    """
    if returns.empty:
        return 0.0

    # Align risk-free rates if provided
    if rf_rates is not None:
        rf = rf_rates.reindex(returns.index).fillna(method="ffill")
        excess = returns - rf
    else:
        excess = returns.copy()

    # Volatility targeting on rolling window
    if rolling_window and vol_target:
        rv = excess.rolling(rolling_window).std() * np.sqrt(freq)
        rv = rv.replace(0.0, np.nan)
        scaling = vol_target / rv
        scaling = scaling.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        excess = excess * scaling

    # Downside or standard deviation
    if downside:
        sigma = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    else:
        sigma = excess.std()

    if sigma == 0 or np.isnan(sigma):
        return 0.0

    mu = excess.mean()
    base = (mu / sigma) * np.sqrt(freq)

    # Optional skew/kurtosis correction
    if skew_kurt_corr:
        s = excess.skew()
        k = excess.kurtosis()
        cf = 1.0 + (s / 6.0) * (mu / sigma) - ((k - 3.0) / 24.0) * (mu / sigma) ** 2
        base *= cf

    # Optional drawdown penalty
    if dd_penalty:
        eq = (1.0 + returns).cumprod()
        dd = ((eq / eq.cummax()) - 1.0).min()
        base -= dd_penalty * abs(dd)

    return float(base)


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def generate_matrix_signals(params: Dict[str, Any], matrices: Dict[str, Any]) -> pd.Series:
    """Generate matrix-based signals from various distance/correlation metrics."""
    price = matrices["price_matrices"]

    dtw = dynamic_time_warping(price, window=int(params.get("dtw_window", 10)))
    cov = rolling_covariance(
        price,
        vol=matrices.get("vol_matrices"),
        window=int(params.get("cov_window", 20)),
    )
    dcor = distance_correlation(price)
    beta = beta_market_dynamic(
        price,
        market=matrices.get("market_matrices"),
        window=int(params.get("beta_window", 20)),
    )

    sigs = {"dtw": dtw, "cov": cov, "dcor": dcor, "beta": beta}
    # Z-score normalization per signal
    z = {k: (v - v.mean()) / (v.std() or 1.0) for k, v in sigs.items()}

    w = np.array(
        [
            params.get("w_dtw", 1.0),
            params.get("w_cov", 1.0),
            params.get("w_dcor", 1.0),
            params.get("w_beta", 1.0),
        ]
    )

    comp = w @ np.vstack([z["dtw"], z["cov"], z["dcor"], z["beta"]])
    thr = params.get("signal_threshold", 0.0)
    sig = np.where(comp > thr, 1.0, np.where(comp < -thr, -1.0, 0.0))
    return pd.Series(sig, index=price.index)


def generate_price_signals(params: Dict[str, Any], matrices: Dict[str, Any]) -> pd.Series:
    """Generate price-based signals using RSI + Bollinger combination."""
    ps = matrices["price_matrices"].apply(lambda M: M[-1, 0])
    rsi_cfg = RSIConfig(
        window=int(params.get("rsi_window", 14)),
        lower=params.get("rsi_lower", 30),
        upper=params.get("rsi_upper", 70),
    )
    rsi = rsi_signals(ps, rsi_cfg)["position"]

    bb_cfg = BollingerConfig(
        window=int(params.get("bb_window", 20)),
        num_std=params.get("bb_std", 2.0),
    )
    bb = bollinger_signals(ps, bb_cfg)["position"]

    comp = params.get("w_rsi", 1.0) * rsi + params.get("w_bb", 1.0) * bb
    thr = params.get("signal_threshold", 0.0)
    sig = np.where(comp > thr, 1.0, np.where(comp < -thr, -1.0, 0.0))
    return pd.Series(sig, index=ps.index)


# ---------------------------------------------------------------------------
# Optuna objective functions
# ---------------------------------------------------------------------------

def _build_params_for_trial(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Freeze defaults and ask trial for each parameter spec."""
    specs = freeze(PARAM_SPECS, **DEFAULT_VALUES)
    return {p.name: p.suggest(trial) for p in specs}


def objective(trial: optuna.trial.Trial) -> float:
    """Main objective: full-data advanced Sharpe of combined matrix + price signals."""
    params = _build_params_for_trial(trial)

    mats = _dl.load_matrices()
    mp = generate_matrix_signals(params, mats)
    pp = generate_price_signals(params, mats)
    pc = mats["price_matrices"].apply(lambda M: M[-1, 0] / M[-2, 0] - 1.0)

    # Raw returns
    mr = (mp.shift(1) * pc).fillna(0.0)
    pr = (pp.shift(1) * pc).fillna(0.0)

    # Apply transaction costs if available
    try:
        mr = apply_transaction_costs(mr, mp, pc, params)
        pr = apply_transaction_costs(pr, pp, pc, params)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("apply_transaction_costs failed in objective: %s", exc)

    # Risk-free rates if provided
    rf = _dl.load_rf_series() if hasattr(_dl, "load_rf_series") else None

    # Separate advanced Sharpe scores
    sm = compute_advanced_sharpe(mr, rf, **ADV_SHARPE_CONFIG)
    sp = compute_advanced_sharpe(pr, rf, **ADV_SHARPE_CONFIG)

    wm = float(params.get("w_mat", 1.0))
    wp = float(params.get("w_price", 1.0))
    if wm + wp == 0.0:
        cs = 0.0
    else:
        cs = (wm * sm + wp * sp) / (wm + wp)

    # Log per-trial Sharpe components
    trial.set_user_attr("sharpe_mat", sm)
    trial.set_user_attr("sharpe_price", sp)

    return float(cs)


def objective_on_subset(trial: optuna.trial.Trial, mats: Dict[str, Any]) -> float:
    """Objective restricted to a subset of matrices (for walk-forward windows)."""
    params = _build_params_for_trial(trial)

    mp = generate_matrix_signals(params, mats)
    pp = generate_price_signals(params, mats)
    pc = mats["price_matrices"].apply(lambda M: M[-1, 0] / M[-2, 0] - 1.0)

    wm = float(params.get("w_mat", 1.0))
    wp = float(params.get("w_price", 1.0))

    combined_sig = (wm * mp + wp * pp)
    returns = (combined_sig.shift(1) * pc).fillna(0.0)

    try:
        returns = apply_transaction_costs(returns, mp, pc, params)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("apply_transaction_costs failed in objective_on_subset: %s", exc)

    rf = _dl.load_rf_series() if hasattr(_dl, "load_rf_series") else None

    return compute_advanced_sharpe(returns, rf, **ADV_SHARPE_CONFIG)


# ---------------------------------------------------------------------------
# Direct optimization (single study)
# ---------------------------------------------------------------------------

def run_direct_optimization() -> None:
    """Run a single full-data optimization study and write results to OUTPUT_FILE."""
    study = optuna.create_study(
        study_name=_ranges_cfg.general.study_name or "direct",
        storage=_user_cfg.get("storage", "sqlite:///opt.db"),
        load_if_exists=True,
        direction="maximize",
        sampler=SAMPLER,
    )

    logger.info("Starting direct optimization: %d trials", N_TRIALS)

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=_ranges_cfg.general.n_jobs or 1,
        callbacks=[checkpoint_callback],
    )

    logger.info("Direct optimization done. Best value = %.6f", study.best_value)

    # Collect trials DataFrame
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    logger.info("Trials collected: %d rows", len(df))

    # Save full trials for post-analysis (optional parquet)
    try:
        trials_output = _user_cfg.get("trials_output", "trials_direct.parquet")
        df.to_parquet(trials_output)
        logger.info("Saved trials DataFrame to %s", trials_output)
    except Exception as exc:
        logger.warning("Could not save trials DataFrame: %s", exc)

    # Build near-optimal ranges around best trial
    best = study.best_params
    thr = 0.95 * float(study.best_value)
    logger.info("Building near-optimal ranges for params (value ≥ %.4f)", thr)

    df_val = df[df["value"] >= thr].copy()

    ranges: Dict[str, List[float]] = {}
    for name in best.keys():
        col = f"params_{name}"
        if col not in df.columns:
            continue
        try:
            low = float(pd.to_numeric(df_val[col], errors="coerce").min())
            high = float(pd.to_numeric(df_val[col], errors="coerce").max())
            ranges[name] = [low, high]
        except Exception:
            continue

    importances = get_param_importances(study)

    output: Dict[str, Any] = {
        "best": best,
        "ranges": ranges,
        "importances": importances,
        "best_value": float(study.best_value),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(output, f)

    logger.info("Saved direct optimization summary to %s", OUTPUT_FILE)


# ---------------------------------------------------------------------------
# Walk-forward optimization
# ---------------------------------------------------------------------------

def run_walk_forward() -> None:
    """Walk-forward optimization with meta-optimization over window results."""
    mats = _dl.load_matrices()
    dates = mats["price_matrices"].index

    wf_cfg = _user_cfg.get("walk_forward", {}) or {}
    train_window = int(wf_cfg.get("train_window", 252 * 3))
    test_window = int(wf_cfg.get("test_window", 252))
    step = int(wf_cfg.get("step", 252))

    splits = get_walk_forward_splits(
        dates,
        train_window=train_window,
        test_window=test_window,
        step=step,
    )

    logger.info("Walk-forward splits: %d windows", len(splits))

    window_results: List[Dict[str, Any]] = []

    for i, (ti, te) in enumerate(splits, start=1):
        logger.info("Starting WF window %d", i)
        train_mats = {k: v.iloc[ti] for k, v in mats.items()}
        test_mats = {k: v.iloc[te] for k, v in mats.items()}

        study_name = (_ranges_cfg.general.study_name or "wf") + f"_wf{i}"
        study = optuna.create_study(
            study_name=study_name,
            storage=_user_cfg.get("storage", "sqlite:///opt.db"),
            load_if_exists=True,
            direction="maximize",
            sampler=SAMPLER,
        )

        study.optimize(
            lambda tr: objective_on_subset(tr, train_mats),
            n_trials=N_TRIALS,
            n_jobs=_ranges_cfg.general.n_jobs or 1,
        )

        best = study.best_params
        logger.info("WF window %d best value = %.6f", i, study.best_value)

        # Evaluate best params on test window
        mp = generate_matrix_signals(best, test_mats)
        pp = generate_price_signals(best, test_mats)
        pc = test_mats["price_matrices"].apply(lambda M: M[-1, 0] / M[-2, 0] - 1.0)

        wm = float(best.get("w_mat", 1.0))
        wp = float(best.get("w_price", 1.0))
        combined_sig = (wm * mp + wp * pp)
        returns = (combined_sig.shift(1) * pc).fillna(0.0)

        try:
            returns = apply_transaction_costs(returns, mp, pc, best)
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("apply_transaction_costs failed in WF test: %s", exc)

        rf = _dl.load_rf_series() if hasattr(_dl, "load_rf_series") else None
        test_sharpe = compute_advanced_sharpe(returns, rf, **ADV_SHARPE_CONFIG)

        window_results.append(
            {
                "window": i,
                "params": best,
                "test_sharpe": test_sharpe,
            }
        )

    # Meta-optimization over window results (user-defined logic)
    final_params = meta_optimize(window_results)

    output: Dict[str, Any] = {
        "final_params": final_params,
        "window_results": window_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(output, f)

    logger.info("Saved walk-forward results to %s", OUTPUT_FILE)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    global _user_cfg
    _user_cfg = load_user_config(args.config)

    setup_ranges_and_defaults()

    if args.mode == "direct":
        run_direct_optimization()
    else:
        run_walk_forward()


if __name__ == "__main__":
    main()
