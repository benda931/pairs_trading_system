# -*- coding: utf-8 -*-
"""
core/fair_value_optimizer.py — Hedge-Fund-Grade Optuna Optimizer (v3)
=====================================================================

מה הקובץ נותן לך:
-----------------
1. חיבור מלא ל־params.py:
   - שימוש ב-PARAM_SPECS + tags כדי לבנות את מרחב החיפוש.
   - clamp לערכים חוקיים לפי ה-Spec.
   - ניתן לחפש רק על חלק מהפרמטרים (signal / execution / macro / diagnostics וכו').

2. חיבור למנוע Fair-Value:
   - core.fair_value_engine._load_config_from_params(params) → cfg
   - FairValueEngine(cfg).run(prices_wide=df, pairs=[(...), ...])

3. Walk-forward CV:
   - n_folds, test_frac, purge_frac.
   - כל fold נותן מטריקות; האופטימיזציה מבוססת על ממוצע/חציון/trimmed mean.

4. Scoring מקצועי:
   - target: "dsr_net" → fallback ל-"psr_net" → "sr_net".
   - penalties:
       * penalty_turnover * turnover_est
       * min_avg_hold → penalize אם זמן אחזקת סופי קצר מדי
       * complexity_penalty על מספר פרמטרים פעילים.

5. Persist & Repro:
   - שמירת best params כ-JSON.
   - שמירת per-trial metrics כ-Parquet (או CSV אם צריך).
   - seed, logging levels, שם study, storage אופציונלי.

הנחת עבודה:
-----------
- מנוע Fair-Value:
    from core.fair_value_engine import _load_config_from_params, FairValueEngine

    cfg = _load_config_from_params(params_dict)
    engine = FairValueEngine(cfg)
    res = engine.run(prices_wide=df, pairs=[(...), ...])

- Data:
    prices_wide: DataFrame עם index = DateTimeIndex, columns = tickers.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------- Optuna imports -----------------
try:
    import optuna  # type: ignore
except Exception as e:  # pragma: no cover
    # כאן באמת אין Optuna בכלל
    raise RuntimeError("optuna is required for optimization") from e

# עכשיו מנסים את הסמפלרים
try:
    from optuna.samplers import TPESampler, CMAESampler  # type: ignore
    _HAS_CMAES = True
except Exception:
    # גרסת Optuna שלך לא מכילה CMAESampler – נשתמש רק ב-TPE
    from optuna.samplers import TPESampler  # type: ignore
    CMAESampler = None  # type: ignore
    _HAS_CMAES = False

from optuna.pruners import MedianPruner, SuccessiveHalvingPruner  # type: ignore

# ----------------- params.py imports -----------------
try:
    from core.params import (  # type: ignore
        PARAM_SPECS,
        filter_by_tags,
        clamp_params_to_specs,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError("core.params (PARAM_SPECS, filter_by_tags, clamp_params_to_specs) is required") from e


# ----------------- Fair-Value engine imports -----------------
_FVE_MOD = None
for _mod in ("core.fair_value_engine", "fair_value_engine"):
    try:
        _FVE_MOD = __import__(_mod, fromlist=["*"])
        break
    except Exception:
        continue

if _FVE_MOD is None:  # pragma: no cover
    raise RuntimeError("Could not import Fair Value Engine module (core.fair_value_engine)")

try:
    _load_config_from_params = _FVE_MOD._load_config_from_params  # type: ignore[attr-defined]
    FairValueEngine = _FVE_MOD.FairValueEngine                    # type: ignore[attr-defined]
except Exception as e:  # pragma: no cover
    raise RuntimeError("Fair Value Engine must expose _load_config_from_params & FairValueEngine") from e


# =====================================================================
# Logging
# =====================================================================

_logger = logging.getLogger("FairValueOptimizer")


def _setup_logger(level: str = "INFO") -> None:
    """Setup module logger level and simple StreamHandler."""
    if not _logger.handlers:
        _logger.addHandler(logging.StreamHandler())
    _logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# =====================================================================
# Configuration
# =====================================================================


@dataclass
class OptConfig:
    """High-level configuration for Fair-Value optimization."""

    # Search
    n_trials: int = 100
    timeout_sec: Optional[int] = None
    sampler: str = "tpe"            # "tpe" | "cmaes"
    seed: Optional[int] = 42
    pruner: str = "median"          # "none" | "median" | "sha"

    # Space control via PARAM_SPECS tags
    include_tags: Tuple[str, ...] = ("signal", "window", "volatility")
    exclude_tags: Tuple[str, ...] = tuple()

    # Metric & scoring
    target: str = "dsr_net"          # fallback to psr_net -> sr_net
    agg: str = "median"              # "mean" | "median" | "trimmed_mean"
    trim_alpha: float = 0.1          # for trimmed_mean
    use_ensemble: bool = False       # if True: rows with window == -1 when available

    # Penalties (subtract from score)
    penalty_turnover: float = 0.0    # weight * turnover_est
    min_avg_hold: float = 0.0        # if >0: penalize (min_avg_hold - avg_hold_days)+
    complexity_penalty: float = 0.0  # penalty * (# active params / total params)

    # CV settings (purged walk-forward)
    n_folds: int = 3
    test_frac: float = 0.2
    purge_frac: float = 0.02

    # Study & output
    study_name: str = "fv_opt"
    storage: Optional[str] = None    # e.g. "sqlite:///fv_opt.db"
    direction: str = "maximize"
    save_dir: str = "opt_results"
    save_folds: bool = True          # save per-trial, per-fold CSV/Parquet
    log_level: str = "INFO"


# =====================================================================
# Utils
# =====================================================================


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    return df.sort_index()


def _build_study(cfg: OptConfig) -> optuna.Study:
    sampler_name = (cfg.sampler or "tpe").lower()

    if sampler_name == "cmaes" and _HAS_CMAES and CMAESampler is not None:
        sampler = CMAESampler(seed=cfg.seed)  # type: ignore[misc]
    else:
        # אם ביקשו cmaes אבל אין CMAES בגרסת Optuna → נ fallback ל-TPE
        sampler = TPESampler(seed=cfg.seed)

    if cfg.pruner.lower() == "median":
        pruner = MedianPruner(n_warmup_steps=max(5, cfg.n_trials // 10))
    elif cfg.pruner.lower() == "sha":
        pruner = SuccessiveHalvingPruner()
    else:
        pruner = None

    return optuna.create_study(
        study_name=cfg.study_name,
        storage=cfg.storage,
        direction=cfg.direction,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )


def _time_folds(index: pd.DatetimeIndex, n_folds: int, test_frac: float, purge_frac: float):
    """Generate (train_idx, test_idx) pairs for purged walk-forward CV."""
    n = len(index)
    if n < 50:
        # fallback: single split
        split = int(n * (1 - test_frac))
        yield (index[:split], index[split:])
        return

    test_len = max(10, int(n * test_frac))
    step = max(test_len // 2, 10)
    purge = max(0, int(n * purge_frac))
    start = 0
    for _ in range(n_folds):
        train_end = max(0, min(n - test_len - 1, start + test_len * 2))
        test_start = min(n - test_len, train_end + purge)
        test_end = min(n, test_start + test_len)
        train_idx = index[:train_end]
        test_idx = index[test_start:test_end]
        if len(test_idx) < 10 or len(train_idx) < 30:
            break
        yield (train_idx, test_idx)
        start = test_start + step


def _select_metric(df: pd.DataFrame, target: str) -> pd.Series:
    """בחר מטריקת בסיס לפי עדיפות: target → psr_net → sr_net."""
    s = df.get(target)
    if s is None or s.isna().all():
        if target != "psr_net":
            s = df.get("psr_net")
    if (s is None or s.isna().all()) and target != "sr_net":
        s = df.get("sr_net")
    if s is None:
        return pd.Series(dtype=float)
    return s


def _aggregate(values: pd.Series, how: str = "median", trim_alpha: float = 0.1) -> float:
    v = values.dropna().astype(float)
    if v.empty:
        return float("nan")
    if how == "mean":
        return float(v.mean())
    if how == "trimmed_mean":
        a = max(0.0, min(0.49, float(trim_alpha)))
        k = int(math.floor(len(v) * a))
        if k > 0:
            v = v.sort_values().iloc[k: len(v) - k]
        return float(v.mean()) if len(v) else float("nan")
    return float(v.median())


def _composite_score(
    df: pd.DataFrame,
    target: str,
    how: str,
    trim_alpha: float,
    penalty_turnover: float,
    min_avg_hold: float,
    complexity_penalty: float,
    n_params_active: int,
    n_params_total: int,
) -> float:
    """Compute composite score with penalties for turnover / hold / complexity."""
    s = _select_metric(df, target)
    base = _aggregate(s, how, trim_alpha)
    if not np.isfinite(base):
        return float("nan")

    pen = 0.0
    if penalty_turnover > 0 and "turnover_est" in df.columns:
        pen += float(penalty_turnover) * _aggregate(df["turnover_est"], how, trim_alpha)

    if min_avg_hold > 0 and "avg_hold_days" in df.columns:
        deficit = (min_avg_hold - df["avg_hold_days"]).clip(lower=0)
        pen += _aggregate(deficit, how, trim_alpha)

    if complexity_penalty > 0 and n_params_total > 0:
        complexity = n_params_active / float(n_params_total)
        pen += complexity_penalty * complexity

    return float(base - pen)


def _save_trial_rows(save_dir: str, trial_number: int, rows: List[pd.DataFrame]) -> None:
    if not rows:
        return
    os.makedirs(save_dir, exist_ok=True)
    df = pd.concat(rows, axis=0, ignore_index=True)
    path_parquet = os.path.join(save_dir, f"trial_{trial_number:05d}.parquet")
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception:
        path_csv = os.path.join(save_dir, f"trial_{trial_number:05d}.csv")
        df.to_csv(path_csv, index=False)


def _make_json_safe(obj: Any) -> Any:
    """Convert numpy/pandas scalars to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# =====================================================================
# Objective builder
# =====================================================================


def make_objective(
    prices_wide: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    cfg: OptConfig,
    include_specs: Optional[Iterable] = None,
):
    """Build Optuna objective function for Fair-Value optimization."""

    _setup_logger(cfg.log_level)
    prices_wide = _ensure_datetime_index(prices_wide)

    # Build search space
    specs = include_specs if include_specs is not None else filter_by_tags(
        PARAM_SPECS,
        include=cfg.include_tags,
        exclude=cfg.exclude_tags,
    )
    if not specs:
        raise RuntimeError("No ParamSpecs selected for optimization (check include_tags/exclude_tags)")

    n_params_total = len(specs)

    def objective(trial: optuna.Trial) -> float:
        # 1) sample params
        params: Dict[str, Any] = {}
        for p in specs:
            params[p.name] = p.suggest(trial)

        # 2) clamp to legal ranges (safety)
        params = clamp_params_to_specs(params)

        # 3) build engine config via engine's own helper
        cfg_obj = _load_config_from_params(params)
        engine = FairValueEngine(cfg_obj)

        # 4) CV loop
        fold_scores: List[float] = []
        trial_rows: List[pd.DataFrame] = []

        for i, (train_idx, test_idx) in enumerate(
            _time_folds(prices_wide.index, cfg.n_folds, cfg.test_frac, cfg.purge_frac)
        ):
            test_df = prices_wide.loc[test_idx.min(): test_idx.max()]

            res = engine.run(prices_wide=test_df, pairs=pairs)
            if res is None or res.empty:
                fold_scores.append(float("nan"))
                continue

            # select ensemble / primary window rows
            if cfg.use_ensemble and "window" in res.columns and (res["window"] == -1).any():
                rows = res[res["window"] == -1]
            else:
                rows = res

            n_params_active = sum(1 for v in params.values() if v is not None)
            score_fold = _composite_score(
                rows,
                target=cfg.target,
                how=cfg.agg,
                trim_alpha=cfg.trim_alpha,
                penalty_turnover=cfg.penalty_turnover,
                min_avg_hold=cfg.min_avg_hold,
                complexity_penalty=cfg.complexity_penalty,
                n_params_active=n_params_active,
                n_params_total=n_params_total,
            )

            if not np.isfinite(score_fold):
                score_fold = -1e9

            fold_scores.append(score_fold)

            if cfg.save_folds:
                tmp = rows.copy()
                tmp["trial"] = trial.number
                tmp["fold"] = i
                trial_rows.append(tmp)

            # pruning
            trial.report(float(np.mean(fold_scores)), step=i + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores)) if fold_scores else float("nan")

        if cfg.save_folds:
            _save_trial_rows(cfg.save_dir, trial.number, trial_rows)

        return mean_score

    return objective


# =====================================================================
# Runner
# =====================================================================


def optimize_fair_value(
    prices_wide: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    cfg: Optional[OptConfig] = None,
    include_specs: Optional[Iterable] = None,
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """Run Fair-Value optimization with Optuna and return (study, best_params)."""

    cfg = cfg or OptConfig()
    _setup_logger(cfg.log_level)

    prices_wide = _ensure_datetime_index(prices_wide)
    os.makedirs(cfg.save_dir, exist_ok=True)

    study = _build_study(cfg)
    objective = make_objective(prices_wide, pairs, cfg, include_specs)

    _logger.info(
        "Starting optimization: trials=%d timeout=%s tags(include=%s, exclude=%s)",
        cfg.n_trials,
        cfg.timeout_sec,
        cfg.include_tags,
        cfg.exclude_tags,
    )

    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        timeout=cfg.timeout_sec,
        gc_after_trial=True,
    )

    best: Dict[str, Any] = study.best_trial.params if study.trials else {}

    # Persist best
    best_path = os.path.join(cfg.save_dir, f"{cfg.study_name}_best.json")
    with open(best_path, "w", encoding="utf8") as f:
        json.dump(_make_json_safe(best), f, indent=2, ensure_ascii=False)

    _logger.info("Best params saved to %s", best_path)

    return study, best


# =====================================================================
# CLI entry (optional)
# =====================================================================

if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Optimize Fair Value Engine with Optuna (HF-grade)")
    p.add_argument("--csv", required=True, help="Path to wide CSV with Date index column")
    p.add_argument("--pairs", nargs="*", help="Pairs as Y:X entries (e.g. XLY:XLC)")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--sampler", default="tpe", choices=["tpe", "cmaes"])
    p.add_argument("--pruner", default="median", choices=["none", "median", "sha"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study", default="fv_opt")
    p.add_argument("--save-dir", default="opt_results")
    p.add_argument("--use-ensemble", action="store_true")
    p.add_argument("--agg", default="median", choices=["mean", "median", "trimmed_mean"])
    p.add_argument("--trim-alpha", type=float, default=0.1)
    p.add_argument("--penalty-turnover", type=float, default=0.0)
    p.add_argument("--min-avg-hold", type=float, default=0.0)
    p.add_argument("--complexity-penalty", type=float, default=0.0)
    args = p.parse_args()

    _setup_logger("INFO")

    df = pd.read_csv(args.csv)
    # autodetect date in first column or 'Date'
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.set_index(df.columns[0])
    df = df.select_dtypes(include=[float, int]).dropna(how="all", axis=1)

    pairs_list: List[Tuple[str, str]] = []
    if args.pairs:
        for s in args.pairs:
            if ":" in s:
                y, x = s.split(":", 1)
                pairs_list.append((y.strip(), x.strip()))

    cfg = OptConfig(
        n_trials=args.trials,
        sampler=args.sampler,
        pruner=args.pruner,
        seed=args.seed,
        study_name=args.study,
        save_dir=args.save_dir,
        use_ensemble=bool(args.use_ensemble),
        agg=args.agg,
        trim_alpha=float(args.trim_alpha),
        penalty_turnover=float(args.penalty_turnover),
        min_avg_hold=float(args.min_avg_hold),
        complexity_penalty=float(args.complexity_penalty),
    )

    study, best = optimize_fair_value(df, pairs_list, cfg)
    print("Best params:")
    print(json.dumps(_make_json_safe(best), indent=2, ensure_ascii=False))
