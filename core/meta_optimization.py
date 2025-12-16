import optuna
import logging
from typing import Dict, Tuple, Any, List, Optional

from core.distributions import build_distributions
from core.optimization_backtester import Backtester
from core.metrics import normalize_metrics, compute_weighted_score

logger = logging.getLogger(__name__)


def meta_optimization_sampling(
    
    base_ranges: Dict[str, Tuple[float, float, float]],
    pair: Tuple[str, str],
    n_outer: int = 3,
    n_inner: int = 50,
    sampler_name: str = "TPE",
    weights: Optional[Dict[str, float]] = None,
    shrink_factor: float = 0.2,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Perform meta-optimization by iteratively tuning parameter search ranges.

    Args:
        base_ranges: Initial parameter ranges as {param: (low, high, step)}.
        pair: Tuple of symbols to backtest (e.g., ("EURUSD", "GBPUSD")).
        n_outer: Number of outer iterations (range refinements).
        n_inner: Number of trials per outer iteration.
        sampler_name: "TPE" or "CMAES" sampler.
        weights: Optional weights for metrics when computing score.
        shrink_factor: Fraction to shrink range around best param each iteration.

    Returns:
        Updated parameter ranges after meta-optimization.
    """
    if weights is None:
        weights = {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2}

    current_ranges = base_ranges.copy()
    sym1, sym2 = pair

    for round_idx in range(1, n_outer + 1):
        logger.info(f"Meta-optimize round {round_idx}/{n_outer} with ranges: {current_ranges}")
        # Build distributions
        distributions = build_distributions(current_ranges)

        # Choose sampler
        sampler = (
            optuna.samplers.TPESampler()
            if sampler_name.upper() == "TPE"
            else optuna.samplers.CmaEsSampler()
        )
        study = optuna.create_study(
            direction="maximize", sampler=sampler,
            study_name=f"meta_opt_{sym1}_{sym2}_round{round_idx}"
        )

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, dist in distributions.items():
                # use suggest based on distribution type
                if hasattr(dist, 'low') and hasattr(dist, 'high') and hasattr(dist, 'step'):
                    lo, hi, step = current_ranges[name]
                    if isinstance(step, int):
                        params[name] = trial.suggest_int(name, int(lo), int(hi), step=int(step))
                    else:
                        params[name] = trial.suggest_float(name, lo, hi, step=step)
                else:
                    params[name] = trial._suggest(name, dist)  # fallback
            # Run backtest
            bt = Backtester(sym1, sym2, **params)
            perf = bt.run().to_dict()
            # Normalize and score
            norm = normalize_metrics(perf)
            score = compute_weighted_score(norm, weights)
            trial.set_user_attr("perf", perf)
            return score

        # Optimize inner trials
        study.optimize(objective, n_trials=n_inner)

        best_params = study.best_trial.params
        logger.info(f"Best params in round {round_idx}: {best_params}")

        # Update ranges around best parameters
        updated_ranges: Dict[str, Tuple[float, float, float]] = {}
        for key, (lo, hi, step) in current_ranges.items():
            best_val = best_params.get(key, (lo + hi) / 2)
            span = (hi - lo) * shrink_factor
            new_lo = max(lo, best_val - span)
            new_hi = min(hi, best_val + span)
            updated_ranges[key] = (new_lo, new_hi, step)
        current_ranges = updated_ranges

    logger.info(f"Final meta-optimized ranges: {current_ranges}")
    return current_ranges


def batch_meta_optimize(
    pairs: List[Tuple[str, str]],
    base_ranges: Dict[str, Tuple[float, float, float]],
    n_outer: int = 3,
    n_inner: int = 50,
    sampler_name: str = "TPE",
    weights: Optional[Dict[str, float]] = None,
    shrink_factor: float = 0.2,
) -> Dict[Tuple[str, str], Dict[str, Tuple[float, float, float]]]:
    """
    Apply meta-optimization across multiple pairs in batch.

    Returns a mapping of pair to its optimized ranges.
    """
    results = {}
    for pair in pairs:
        logger.info(f"Starting batch meta-opt for pair {pair}")
        optimized = meta_optimization_sampling(
            base_ranges, pair, n_outer, n_inner, sampler_name, weights, shrink_factor
        )
        results[pair] = optimized
    return results

