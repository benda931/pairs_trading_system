# -*- coding: utf-8 -*-
"""
core/meta_optimization.py
========================

HF-grade Meta-Optimization (Range Shrink)
-----------------------------------------

Purpose
-------
Iteratively shrink a base search space around best-performing parameters.
This is useful when you have a large param space and you want to "zoom" in
around promising regions before running heavier campaigns.

Key Properties
--------------
- Deterministic when random_seed is provided (sampler seeds are derived per round).
- Uses only public Optuna APIs (no trial._suggest).
- Best-effort but debuggable: trials get rich user_attrs (params/perf/raw/norm/score).
- Safe range shrink with guards against collapsing / invalid ranges.

Inputs
------
- base_ranges: {param_name: (low, high, step)}
- pair: (sym1, sym2)
- weights: weights for compute_weighted_score(normalize_metrics(perf))

Outputs
-------
- Updated ranges of the same shape as base_ranges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import optuna
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution

from core.distributions import build_distributions
from core.metrics import compute_weighted_score, normalize_metrics
from core.optimization_backtester import OptimizationBacktester

logger = logging.getLogger(__name__)

ParamRange = Tuple[float, float, float]  # (low, high, step)
ParamRanges = Dict[str, ParamRange]
Metrics = Mapping[str, Any]


# ----------------------------- Sampler Selection -----------------------------


def _create_sampler(name: str, seed: Optional[int]) -> optuna.samplers.BaseSampler:
    """
    Create an Optuna sampler by name. If seed is provided, behavior is deterministic.
    """
    name_up = (name or "TPE").strip().upper()

    if name_up == "TPE":
        return optuna.samplers.TPESampler(seed=seed)
    if name_up in ("CMAES", "CMA-ES"):
        return optuna.samplers.CmaEsSampler(seed=seed)
    if name_up == "RANDOM":
        return optuna.samplers.RandomSampler(seed=seed)

    logger.warning("Unknown sampler '%s' - falling back to TPE", name)
    return optuna.samplers.TPESampler(seed=seed)


def _suggest_from_distribution(
    trial: optuna.Trial,
    name: str,
    dist: BaseDistribution,
) -> Any:
    """
    Suggest a parameter using *public* Optuna APIs only.
    Supports FloatDistribution / IntDistribution / CategoricalDistribution.
    """
    if isinstance(dist, FloatDistribution):
        return trial.suggest_float(
            name=name,
            low=dist.low,
            high=dist.high,
            step=dist.step,
            log=dist.log,
        )

    if isinstance(dist, IntDistribution):
        return trial.suggest_int(
            name=name,
            low=dist.low,
            high=dist.high,
            step=dist.step,
            log=dist.log,
        )

    if isinstance(dist, CategoricalDistribution):
        return trial.suggest_categorical(name, list(dist.choices))

    raise TypeError(f"Unsupported distribution type for {name}: {type(dist)!r}")


# ----------------------------- Range Utilities ------------------------------


def _validate_ranges(ranges: ParamRanges) -> None:
    """
    Validate basic invariants for ranges.
    """
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError("base_ranges must be a non-empty dict[str, (low, high, step)].")

    for k, v in ranges.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Invalid param name: {k!r}")
        if not (isinstance(v, tuple) and len(v) == 3):
            raise ValueError(f"Range for {k} must be (low, high, step). Got: {v!r}")
        lo, hi, step = v
        lo_f, hi_f, step_f = float(lo), float(hi), float(step)
        if hi_f <= lo_f:
            raise ValueError(f"Invalid range for {k}: high<=low ({lo_f}, {hi_f})")
        if step_f <= 0:
            raise ValueError(f"Invalid step for {k}: step<=0 ({step_f})")


def _shrink_range(
    base_lo: float,
    base_hi: float,
    best_val: float,
    step: float,
    shrink_factor: float,
) -> Tuple[float, float]:
    """
    Shrink [base_lo, base_hi] around best_val using shrink_factor.

    shrink_factor controls half-width relative to base span.
    Example: shrink_factor=0.2 -> new range half-width = 20% * (base_hi-base_lo)
    """
    best_val = max(base_lo, min(base_hi, best_val))
    span = (base_hi - base_lo) * float(shrink_factor)

    if span <= 0:
        return base_lo, base_hi

    new_lo = max(base_lo, best_val - span)
    new_hi = min(base_hi, best_val + span)

    # Guard collapse
    if new_hi <= new_lo:
        return base_lo, base_hi

    # Optional: enforce at least one step of room
    if (new_hi - new_lo) < step:
        # widen minimally around best
        half = step / 2.0
        new_lo = max(base_lo, best_val - half)
        new_hi = min(base_hi, best_val + half)
        if new_hi <= new_lo:
            return base_lo, base_hi

    return new_lo, new_hi


def _derive_round_seed(seed: Optional[int], round_idx: int) -> Optional[int]:
    """
    Make per-round deterministic seeds while keeping full-run determinism.
    """
    if seed is None:
        return None
    # deterministic and stable
    return int(seed) + int(round_idx) * 1009


@dataclass(frozen=True)
class _ObjectiveResult:
    score: float
    perf_raw: Dict[str, Any]
    perf_norm: Dict[str, Any]
    params: Dict[str, Any]


# --------------------------- Meta Optimization API ---------------------------


def meta_optimization_sampling(
    base_ranges: ParamRanges,
    pair: Tuple[str, str],
    n_outer: int = 3,
    n_inner: int = 50,
    sampler_name: str = "TPE",
    weights: Optional[Dict[str, float]] = None,
    shrink_factor: float = 0.2,
    *,
    random_seed: Optional[int] = 42,
    storage: Optional[str] = None,
    study_prefix: str = "meta_opt",
    n_jobs: int = 1,
    on_trial_end: Optional[Callable[[optuna.Trial, Metrics, float], None]] = None,
) -> ParamRanges:
    """
    Iteratively shrink parameter ranges around best-performing parameters.

    Returns:
        Updated ParamRanges after n_outer rounds.

    Notes:
        - If storage is provided, studies are persisted and can be resumed.
        - Determinism is achieved via sampler seeding (derived per round).
        - The backtest engine is OptimizationBacktester(sym1, sym2, **params).
          It is expected that bt.run() returns Dict[str, float] (metrics).
    """
    _validate_ranges(base_ranges)

    if n_outer <= 0:
        raise ValueError("n_outer must be >= 1")
    if n_inner <= 0:
        raise ValueError("n_inner must be >= 1")
    if not (0 < float(shrink_factor) <= 1.0):
        raise ValueError("shrink_factor must be in (0, 1].")

    sym1, sym2 = pair

    if weights is None:
        # Sensible default; override via config in Phase 1/2 later
        weights = {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2}

    current_ranges: ParamRanges = dict(base_ranges)

    global_best_score = float("-inf")
    global_best_params: Dict[str, Any] = {}

    logger.info(
        "Meta-optimization start: pair=(%s,%s), n_outer=%d, n_inner=%d, sampler=%s, seed=%s",
        sym1,
        sym2,
        n_outer,
        n_inner,
        sampler_name,
        random_seed,
    )

    for round_idx in range(1, n_outer + 1):
        logger.info(
            "Meta-opt round %d/%d for pair (%s,%s). Current ranges keys=%d",
            round_idx,
            n_outer,
            sym1,
            sym2,
            len(current_ranges),
        )

        # Build distributions from current ranges
        distributions = build_distributions(current_ranges)

        # Round-specific deterministic seed
        round_seed = _derive_round_seed(random_seed, round_idx)
        sampler = _create_sampler(sampler_name, seed=round_seed)

        study_name = f"{study_prefix}_{sym1}_{sym2}_round{round_idx}"
        storage_arg = storage if storage else None

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=study_name,
            storage=storage_arg,
            load_if_exists=bool(storage_arg),
        )

        def _run_objective(trial: optuna.Trial) -> float:
            # 1) Sample params
            params: Dict[str, Any] = {}
            for p_name, dist in distributions.items():
                params[p_name] = _suggest_from_distribution(trial, p_name, dist)

            # 2) Backtest
            bt = OptimizationBacktester(sym1, sym2, **params)
            perf_raw = bt.run()  # expected: Dict[str, float]-like

            # 3) Normalize & score
            perf_norm = normalize_metrics(perf_raw)
            score = float(compute_weighted_score(perf_norm, weights or {}))

            # 4) Persist rich attrs for post-analysis
            trial.set_user_attr("params", params)
            trial.set_user_attr("perf_raw", dict(perf_raw))
            trial.set_user_attr("perf_norm", dict(perf_norm))
            trial.set_user_attr("score", score)

            # 5) Optional callback
            if on_trial_end is not None:
                try:
                    on_trial_end(trial, perf_raw, score)
                except Exception as cb_exc:  # pragma: no cover
                    logger.debug("on_trial_end callback failed: %s", cb_exc)

            return score

        study.optimize(
            _run_objective,
            n_trials=n_inner,
            n_jobs=max(1, int(n_jobs)),
            show_progress_bar=False,
        )

        best_trial = study.best_trial if getattr(study, "best_trial", None) is not None else None
        if best_trial is None:
            logger.warning("No successful trials in round %d – leaving ranges unchanged", round_idx)
            continue

        best_params = dict(best_trial.params)
        best_score = float(best_trial.value) if best_trial.value is not None else float("-inf")

        logger.info("Round %d best: score=%.6f params=%s", round_idx, best_score, best_params)

        if best_score > global_best_score:
            global_best_score = best_score
            global_best_params = dict(best_params)

        # Shrink ranges around best params
        updated_ranges: ParamRanges = {}
        for key, (lo, hi, step) in current_ranges.items():
            base_lo, base_hi, step_f = float(lo), float(hi), float(step)
            best_val = best_params.get(key, (base_lo + base_hi) / 2.0)

            try:
                best_val_f = float(best_val)
            except Exception:
                best_val_f = (base_lo + base_hi) / 2.0

            new_lo, new_hi = _shrink_range(
                base_lo=base_lo,
                base_hi=base_hi,
                best_val=best_val_f,
                step=step_f,
                shrink_factor=shrink_factor,
            )
            updated_ranges[key] = (float(new_lo), float(new_hi), float(step_f))

        current_ranges = updated_ranges

    logger.info(
        "Meta-optimization complete for (%s,%s). best_score=%.6f",
        sym1,
        sym2,
        global_best_score,
    )
    if global_best_params:
        logger.info("Global best params encountered: %s", global_best_params)

    return current_ranges


def batch_meta_optimize(
    pairs: List[Tuple[str, str]],
    base_ranges: ParamRanges,
    n_outer: int = 3,
    n_inner: int = 50,
    sampler_name: str = "TPE",
    weights: Optional[Dict[str, float]] = None,
    shrink_factor: float = 0.2,
    *,
    random_seed: Optional[int] = 42,
    storage: Optional[str] = None,
    study_prefix: str = "meta_opt",
    n_jobs: int = 1,
) -> Dict[Tuple[str, str], ParamRanges]:
    """
    Batch meta-optimization across multiple pairs.

    Returns:
        {(sym1, sym2): optimized_ranges}
    """
    _validate_ranges(base_ranges)

    results: Dict[Tuple[str, str], ParamRanges] = {}
    for i, pair in enumerate(pairs, start=1):
        logger.info("[%d/%d] Batch meta-opt start for pair=%s", i, len(pairs), pair)

        # Derive a stable per-pair seed (keeps determinism, reduces identical sequences)
        pair_seed = None
        if random_seed is not None:
            # deterministic hash-like mixing
            pair_seed = int(random_seed) + (hash(pair) % 100_000)

        optimized_ranges = meta_optimization_sampling(
            base_ranges=base_ranges,
            pair=pair,
            n_outer=n_outer,
            n_inner=n_inner,
            sampler_name=sampler_name,
            weights=weights,
            shrink_factor=shrink_factor,
            random_seed=pair_seed,
            storage=storage,
            study_prefix=study_prefix,
            n_jobs=n_jobs,
            on_trial_end=None,
        )
        results[pair] = optimized_ranges

    logger.info("Batch meta-opt complete for %d pairs", len(pairs))
    return results
