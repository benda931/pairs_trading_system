# -*- coding: utf-8 -*-
# core/optimizer.py
"""Pairs Trading Optimizer (HF-grade, v2)
========================================

Optuna-powered hyperparameter search for Omri's pairs-trading system.

Highlights
----------
- Hedge-fund grade optimization loop around :class:`Backtester` /
  :class:`OptimizationBacktester`.
- Multi-metric evaluation (Sharpe, return, drawdown, tail-risk, stability,
  trades, turnover, exposure).
- Pluggable scoring modes:
    * "classic"  – backwards compatible weighted Sharpe/return/DD.
    * "hf"       – hedge-fund grade score via HFObjectiveConfig + compute_hf_objective.
    * "hybrid"   – convex combination of classic & hf (default).
- Optional *multi-objective* Optuna study (score, Sharpe, drawdown).
- Robust error handling and logging (no silent failures, no print spam).
- Flexible config structure with sensible defaults.
- Rich results DataFrame: params + metrics + scores + diagnostics per trial.

Public API (backwards compatible)
---------------------------------
- :func:`run_optimization(candidates, config, n_trials=50, study_name="pair_opt_study")`
    *Returns a pandas.DataFrame of all evaluated trials.*

`config` is a dict with at least:

- ``config['data']``
    - ``price_csv``: path to CSV with "Close" prices for leg A.
    - ``hedge_csv``: path to CSV with "Close" prices for leg B.
    - OR pre-loaded series/dataframes under keys ``price`` / ``hedge``.
- ``config['ranges']``
    - mapping ``param -> (low, high)`` or ``(low, high, step)``.
- ``config['signals']['generator']``
    - callable that the Backtester will use for signal generation.
- ``config['backtest_params']``
    - kwargs forwarded to :class:`Backtester`.

Optional keys:

- ``config['seed']`` (int)
- ``config['score_weights']``: weights for the *classic* multi-metric score.
- ``config['score']``:
    - ``mode``: "classic" | "hf" | "hybrid" (default "hybrid")
    - ``hybrid_alpha``: float in [0, 1], weight on HF score (default 0.7)
    - ``hf_objective``: dict overriding HFObjectiveConfig fields
- ``config['optuna']``: dict with keys
    - ``direction``: "maximize" / "minimize" (default "maximize")
    - ``sampler``: "TPE" / "CMAES"
    - ``pruner``: "median" / "none"
    - ``timeout_sec``: global timeout in seconds
    - ``n_trials``: override of n_trials argument
    - ``multi_objective``: bool, if True use (score, sharpe, drawdown)
- ``config['seed_from_candidates']``: bool – if True, first |candidates|
  trials are seeded from `candidates` (if provided).

Advanced:
- Downstream helpers: :func:`summarize_results`, :func:`compute_pareto_front`,
  :func:`rank_trials`, :func:`tag_best_trials`, :func:`run_optimization_with_study`.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)
import logging
import time
import traceback

import numpy as np
import pandas as pd

# Optuna is optional at runtime; for typing we use TYPE_CHECKING.
try:  # pragma: no cover - optional dependency
    import optuna  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from optuna.study import Study  # type: ignore[import]
    from optuna.trial import Trial  # type: ignore[import]
else:  # at runtime, when optuna might be missing – fall back to Any
    Study = Any  # type: ignore[valid-type]
    Trial = Any  # type: ignore[valid-type]

# Backtester / HF objective from optimization_backtester
try:
    from core.optimization_backtester import (
        Backtester,
        BacktestResult,
        HFObjectiveConfig,
        compute_hf_objective,
    )
except Exception:  # pragma: no cover - optional
    Backtester = None  # type: ignore
    BacktestResult = Any  # type: ignore[assignment]
    HFObjectiveConfig = None  # type: ignore[assignment]
    compute_hf_objective = None  # type: ignore[assignment]

# Optional metrics module (used if available for classic score)
try:  # type: ignore[import]
    from core.metrics import normalize_metrics, compute_weighted_score  # type: ignore
except Exception:  # pragma: no cover - optional
    normalize_metrics = None  # type: ignore
    compute_weighted_score = None  # type: ignore


logger = logging.getLogger(__name__)

ParamRangeTuple = Tuple[float, float, Optional[float]]
ParamRanges = Mapping[str, ParamRangeTuple]


# ---------------------------------------------------------------------------
# Helpers: data loading & ranges
# ---------------------------------------------------------------------------


def _load_price_series(config: Mapping[str, Any]) -> Tuple[pd.Series, pd.Series]:
    """Load price & hedge series from CSVs or preloaded data.

    Supports two modes:
    - CSV paths: config['data']['price_csv'] / ['hedge_csv'].
    - Preloaded series/dataframes: config['data']['price'] / ['hedge'].
    """

    data_cfg = config.get("data", {}) or {}

    # Preloaded series/dataframe – higher priority
    price_obj = data_cfg.get("price")
    hedge_obj = data_cfg.get("hedge")

    def _to_series(obj: Any) -> Optional[pd.Series]:
        if obj is None:
            return None
        if isinstance(obj, pd.Series):
            return obj.sort_index()
        if isinstance(obj, pd.DataFrame):
            # heuristic: prefer "Close" column if exists
            col = "Close" if "Close" in obj.columns else obj.columns[0]
            return obj[col].sort_index()
        return None

    price = _to_series(price_obj)
    hedge = _to_series(hedge_obj)

    # If not preloaded, try CSV paths
    if price is None:
        price_csv = data_cfg.get("price_csv")
        if not price_csv:
            raise ValueError("config['data']['price_csv'] is required if no preloaded price provided")
        df_p = pd.read_csv(price_csv, parse_dates=True, index_col=0)
        col = "Close" if "Close" in df_p.columns else df_p.columns[0]
        price = df_p[col].sort_index()

    if hedge is None:
        hedge_csv = data_cfg.get("hedge_csv")
        if not hedge_csv:
            raise ValueError("config['data']['hedge_csv'] is required if no preloaded hedge provided")
        df_h = pd.read_csv(hedge_csv, parse_dates=True, index_col=0)
        col = "Close" if "Close" in df_h.columns else df_h.columns[0]
        hedge = df_h[col].sort_index()

    if len(price) != len(hedge):
        # Align on common index
        df = pd.concat({"price": price, "hedge": hedge}, axis=1).dropna()
        price, hedge = df["price"], df["hedge"]

    return price, hedge


def _coerce_ranges(ranges_cfg: Mapping[str, Any]) -> ParamRanges:
    """Coerce config['ranges'] into a normalized mapping.

    Accepts:
    - (low, high)
    - (low, high, step)
    - dict with keys low/high[/step]
    """

    out: Dict[str, ParamRangeTuple] = {}
    for name, spec in ranges_cfg.items():
        try:
            if isinstance(spec, Mapping):
                lo = float(spec.get("low"))
                hi = float(spec.get("high"))
                step_val = spec.get("step")
                step = float(step_val) if step_val is not None else None
            elif isinstance(spec, (list, tuple)) and len(spec) >= 2:
                lo = float(spec[0])
                hi = float(spec[1])
                step = float(spec[2]) if len(spec) >= 3 and spec[2] is not None else None
            else:
                raise TypeError(f"Unsupported range spec for {name!r}: {spec!r}")
            if hi <= lo:
                hi = lo + 1e-9
            out[name] = (lo, hi, step)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Bad range for %s: %r (%s)", name, spec, exc)
    return out


# ---------------------------------------------------------------------------
# Scoring helpers: classic + HF + hybrid
# ---------------------------------------------------------------------------


def _build_hf_cfg_from_config(config: Mapping[str, Any]) -> Optional[HFObjectiveConfig]:
    """Build HFObjectiveConfig from config['score']['hf_objective'] if available."""
    if HFObjectiveConfig is None:
        return None

    score_cfg = config.get("score", {}) or {}
    hf_cfg_dict = score_cfg.get("hf_objective", {}) or {}
    if not isinstance(hf_cfg_dict, Mapping):
        hf_cfg_dict = {}

    base = HFObjectiveConfig()
    for k, v in hf_cfg_dict.items():
        if hasattr(base, k):
            try:
                setattr(base, k, v)
            except Exception:
                logger.debug("Failed to override HFObjectiveConfig.%s=%r", k, v)

    return base


def _resolve_score_mode(config: Mapping[str, Any]) -> Tuple[str, float]:
    """Resolve score mode and hybrid alpha from config['score'] block.

    Returns
    -------
    mode : {"classic", "hf", "hybrid"}
    hybrid_alpha : float
        weight on HF score when mode == "hybrid".
    """
    score_cfg = config.get("score", {}) or {}
    mode = str(score_cfg.get("mode", "hybrid")).lower()
    if mode not in {"classic", "hf", "hybrid"}:
        mode = "hybrid"

    try:
        alpha = float(score_cfg.get("hybrid_alpha", 0.7))
    except Exception:
        alpha = 0.7
    alpha = float(min(max(alpha, 0.0), 1.0))

    return mode, alpha


def _score_classic_from_metrics(metrics: Mapping[str, float], config: Mapping[str, Any]) -> float:
    """Compute a scalar *classic* score from backtest metrics.

    Preference hierarchy:
    1. If `normalize_metrics` + `compute_weighted_score` are available, use them.
    2. Otherwise, use a simple bounded transform on Sharpe/return/drawdown.
    """

    # Explicit custom scorer function
    custom_scorer = config.get("scorer")
    if callable(custom_scorer):
        try:
            return float(custom_scorer(metrics))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("custom scorer failed: %s", exc)

    weights_cfg = config.get("score_weights") or {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2}

    if normalize_metrics is not None and compute_weighted_score is not None:
        try:
            norm = normalize_metrics(metrics)  # type: ignore[arg-type]
            return float(compute_weighted_score(norm, weights_cfg))  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("metric module scoring failed, falling back: %s", exc)

    # Fallback: smooth transforms to [0, 1]
    sharpe = float(metrics.get("Sharpe", metrics.get("sharpe", 0.0)))
    profit = float(metrics.get("Profit", metrics.get("total_return", 0.0)))
    dd = float(metrics.get("Drawdown", metrics.get("max_drawdown", 0.0)))

    s = 0.5 + float(np.tanh(sharpe / 3.0)) / 2.0
    p = 0.5 + float(np.tanh(profit / 1e4)) / 2.0
    d = 1.0 - float(min(max(dd, 0.0), 1.0))

    w_sh = float(weights_cfg.get("Sharpe", 0.5))
    w_pf = float(weights_cfg.get("Profit", 0.3))
    w_dd = float(weights_cfg.get("Drawdown", 0.2))
    z = abs(w_sh) + abs(w_pf) + abs(w_dd) or 1.0
    w_sh, w_pf, w_dd = w_sh / z, w_pf / z, w_dd / z

    return w_sh * s + w_pf * p + w_dd * d


def _combined_score(
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
    hf_result: Optional[BacktestResult],
    hf_cfg: Optional[HFObjectiveConfig],
) -> Tuple[float, Dict[str, Any]]:
    """Compute (final_score, extras) using classic + HF + hybrid modes.

    extras contains diagnostic components (classic_score, hf_score, mode, alpha, etc.)
    for logging into trial user_attrs and the results dataframe.
    """
    mode, alpha = _resolve_score_mode(config)
    classic_score = _score_classic_from_metrics(metrics, config)

    hf_score: Optional[float] = None
    if hf_result is not None and HFObjectiveConfig is not None and compute_hf_objective is not None:
        try:
            cfg = hf_cfg or HFObjectiveConfig()
            hf_score = float(compute_hf_objective(hf_result, cfg))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("HF objective failed, falling back to classic: %s", exc)
            hf_score = None

    # Decide final score
    if mode == "classic" or hf_score is None:
        final_score = classic_score
    elif mode == "hf":
        final_score = hf_score
    else:  # hybrid
        final_score = alpha * hf_score + (1.0 - alpha) * classic_score

    extras: Dict[str, Any] = {
        "score_mode": mode,
        "hybrid_alpha": alpha,
        "classic_score": float(classic_score),
        "hf_score": float(hf_score) if hf_score is not None else None,
        "final_score": float(final_score),
    }

    return float(final_score), extras


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------


def _build_optuna_objects(
    config: Mapping[str, Any],
    n_trials: int,
    study_name: str,
) -> Tuple[Study, Dict[str, Any]]:
    """Build Optuna Study + runtime options from config.

    Returns
    -------
    study : optuna.study.Study
    runtime_opts : dict
        Keys:
            - ``timeout_sec``: global timeout.
            - ``seed``: random seed.
            - ``multi_objective``: bool flag.
    """

    if optuna is None:  # pragma: no cover - explicit error
        raise RuntimeError("optuna is not installed. Run `pip install optuna`.")

    opt_cfg = config.get("optuna", {}) or {}
    seed = int(config.get("seed", opt_cfg.get("seed", 42)))
    direction = opt_cfg.get("direction", "maximize")
    sampler_name = str(opt_cfg.get("sampler", "TPE")).upper()
    pruner_name = str(opt_cfg.get("pruner", "median")).lower()
    timeout_sec = opt_cfg.get("timeout_sec")
    multi_objective = bool(opt_cfg.get("multi_objective", False))

    # Sampler
    if sampler_name == "CMAES" and hasattr(optuna.samplers, "CmaEsSampler"):
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    # Pruner
    if pruner_name.startswith("median"):
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    if multi_objective:
        directions = opt_cfg.get("directions") or ["maximize", "maximize", "minimize"]
        study = optuna.create_study(
            study_name=study_name,
            directions=list(directions),
            sampler=sampler,
            pruner=pruner,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

    return study, {"timeout_sec": timeout_sec, "seed": seed, "multi_objective": multi_objective}


# ---------------------------------------------------------------------------
# Public API: run_optimization
# ---------------------------------------------------------------------------


def run_optimization(
    candidates: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Any]]]],
    config: Mapping[str, Any],
    n_trials: int = 50,
    study_name: str = "pair_opt_study",
) -> pd.DataFrame:
    """Run Optuna-based hyperparameter search over a pairs-trading config.

    Parameters
    ----------
    candidates:
        Optional initial grid / warm-start candidates. If provided, they are:
        - Optionally used as seeded trials (see config['seed_from_candidates']).
        - Appended to the final results DataFrame with ``_source="candidate_grid"``.
    config:
        Dictionary with data/ranges/signals/backtest_params and optional
        optuna/score settings (see module docstring).
    n_trials:
        Maximum number of trials (overridden if ``config['optuna']['n_trials']``
        is provided).
    study_name:
        Name for the Optuna study.

    Returns
    -------
    pandas.DataFrame
        Each row is a trial: parameters + metrics + trial metadata.
    """

    # Resolve trial count from config (config wins over argument)
    opt_cfg = config.get("optuna", {}) or {}
    n_trials = int(opt_cfg.get("n_trials", n_trials))

    # Load price & hedge once
    price, hedge = _load_price_series(config)

    # Normalize ranges
    ranges_cfg = config.get("ranges", {}) or {}
    if not ranges_cfg:
        raise ValueError("config['ranges'] must be provided and non-empty")
    ranges: ParamRanges = _coerce_ranges(ranges_cfg)

    if Backtester is None:
        raise RuntimeError("Backtester is not available. Ensure core.optimization_backtester is importable.")

    logger.info("run_optimization: n_trials=%s, n_params=%s", n_trials, len(ranges))

    # Build Optuna objects
    study, rt_opts = _build_optuna_objects(config, n_trials=n_trials, study_name=study_name)
    timeout_sec = rt_opts.get("timeout_sec")
    multi_objective = bool(rt_opts.get("multi_objective", False))

    # External callbacks (per-trial hooks)
    callbacks: Sequence[Callable[[Trial, Dict[str, Any]], None]] = tuple(
        config.get("callbacks", []) or []
    )

    # Signals and backtest configuration
    signals_cfg = config.get("signals", {}) or {}
    signal_generator = signals_cfg.get("generator")
    if not callable(signal_generator):
        raise ValueError("config['signals']['generator'] must be callable")

    backtest_params = config.get("backtest_params", {}) or {}

    # HF objective config (if available)
    hf_cfg = _build_hf_cfg_from_config(config)

    # Candidate warm-start: we keep only the param cols that intersect ranges
    seed_from_candidates = bool(config.get("seed_from_candidates", False))
    candidate_param_dicts: List[Dict[str, Any]] = []
    if candidates is not None and seed_from_candidates:
        try:
            if isinstance(candidates, pd.DataFrame):
                cdf = candidates
            else:
                cdf = pd.DataFrame.from_records(list(candidates))
            for _, row in cdf.iterrows():
                pdict: Dict[str, Any] = {}
                for name in ranges.keys():
                    if name in row:
                        val = row[name]
                        if isinstance(val, (int, float)):
                            pdict[name] = float(val)
                if pdict:
                    candidate_param_dicts.append(pdict)
        except Exception as exc:
            logger.debug("Failed to extract seed params from candidates: %s", exc)
            candidate_param_dicts = []

    results: List[Dict[str, Any]] = []

    def _sample_params_from_trial(trial: Trial, ranges_: ParamRanges) -> Dict[str, Any]:
        """Sample params from optuna Trial given numeric ranges."""
        params_: Dict[str, Any] = {}
        for name, (low, high, step) in ranges_.items():
            lo, hi = float(low), float(high)
            if hi <= lo:
                hi = lo + 1e-9
            if step is None:
                params_[name] = trial.suggest_float(name, lo, hi)
            else:
                params_[name] = trial.suggest_float(name, lo, hi, step=float(step))
        return params_

    def objective(trial: Trial) -> Union[float, Tuple[float, float, float]]:  # type: ignore[valid-type]
        # Optional: warm-start from candidates for first trials
        if seed_from_candidates and candidate_param_dicts and trial.number < len(candidate_param_dicts):
            params = dict(candidate_param_dicts[trial.number])
        else:
            params = _sample_params_from_trial(trial, ranges)

        t0 = time.time()

        try:
            # Backtester invocation
            bt = Backtester(
                price=price,
                hedge=hedge,
                signals=signal_generator,
                signal_params=params,
                **backtest_params,
            )
            bt_res = bt.run()

            # Metrics extraction – be robust to different result shapes
            metrics_raw: Dict[str, Any] = {}
            if hasattr(bt_res, "metrics") and isinstance(bt_res.metrics, Mapping):  # type: ignore[attr-defined]
                metrics_raw.update(bt_res.metrics)  # type: ignore[arg-type]
            elif isinstance(bt_res, Mapping):
                metrics_raw.update(bt_res)  # type: ignore[arg-type]

            # Normalize base metrics
            sharpe = float(metrics_raw.get("Sharpe", metrics_raw.get("sharpe", np.nan)))
            total_ret = float(metrics_raw.get("Profit", metrics_raw.get("total_return", np.nan)))
            max_dd = float(metrics_raw.get("Drawdown", metrics_raw.get("max_drawdown", np.nan)))

            win_rate = np.nan
            if hasattr(bt_res, "win_rate") and callable(getattr(bt_res, "win_rate", None)):  # type: ignore[truthy-function]
                try:
                    win_rate = float(bt_res.win_rate())  # type: ignore[call-arg]
                except Exception:
                    win_rate = np.nan

            metrics: Dict[str, float] = {
                "Sharpe": sharpe,
                "Profit": total_ret,
                "Drawdown": max_dd,
                "WinRate": float(win_rate) if np.isfinite(win_rate) else np.nan,
            }

            # Try to obtain HF-grade BacktestResult (if available)
            hf_result: Optional[BacktestResult] = None
            full_res = getattr(bt, "_last_result", None)
            if full_res is not None:
                hf_result = full_res  # type: ignore[assignment]

                # enrich metrics with tail / behavior if available
                try:
                    metrics["ES_95"] = float(getattr(hf_result, "es_95", np.nan))
                    metrics["VaR_95"] = float(getattr(hf_result, "var_95", np.nan))
                    metrics["Trades"] = float(getattr(hf_result, "n_trades", np.nan))
                    metrics["Turnover"] = float(getattr(hf_result, "turnover", np.nan))
                    metrics["Exposure"] = float(getattr(hf_result, "exposure", np.nan))
                except Exception:
                    pass

            # Compute final score (classic / hf / hybrid)
            score, score_extras = _combined_score(metrics, config, hf_result, hf_cfg)

            runtime_sec = float(time.time() - t0)

            # Attach rich info to trial for Optuna dashboards
            try:
                trial.set_user_attr("params", params)
                trial.set_user_attr("metrics", metrics)
                trial.set_user_attr("score", score_extras.get("final_score", score))
                trial.set_user_attr("score_mode", score_extras.get("score_mode"))
                trial.set_user_attr("classic_score", score_extras.get("classic_score"))
                trial.set_user_attr("hf_score", score_extras.get("hf_score"))
                trial.set_user_attr("runtime_sec", runtime_sec)
            except Exception:
                pass

            row: Dict[str, Any] = {
                **params,
                "trial_number": int(trial.number),
                "sharpe": sharpe,
                "return": total_ret,
                "drawdown": max_dd,
                "win_rate": win_rate,
                "score": score_extras.get("final_score", score),
                "score_mode": score_extras.get("score_mode"),
                "classic_score": score_extras.get("classic_score"),
                "hf_score": score_extras.get("hf_score"),
                "runtime_sec": runtime_sec,
            }

            # enrich row with HF stats if available
            if hf_result is not None:
                try:
                    stats = getattr(hf_result, "stats", {}) or {}
                    if isinstance(stats, Mapping):
                        for k in (
                            "hf_base_score",
                            "hf_final_score",
                            "hf_dd_soft_penalty",
                            "hf_tail_penalty",
                            "hf_behavior_penalty",
                            "hf_wf_sharpe_mean",
                            "hf_wf_sharpe_std",
                        ):
                            if k in stats:
                                row[k] = stats[k]
                except Exception:
                    pass

            results.append(row)

            # External callbacks (e.g. Streamlit progress, logging to DuckDB/SQL, etc.)
            for cb in callbacks:
                try:
                    cb(trial, row)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("callback failed: %s", exc)

            # Return value to Optuna
            if multi_objective:
                # score → maximize, Sharpe → maximize, drawdown → minimize
                dd_for_obj = max_dd if np.isfinite(max_dd) else np.inf
                return float(score), float(sharpe), float(dd_for_obj)
            return float(score)

        except Exception as exc:
            # Mark trial as failed but keep it in logs
            err_msg = f"Backtest failed: {exc}"
            logger.warning("[Optuna] Trial failed: %s", err_msg)
            logger.debug("Traceback:\n%s", traceback.format_exc())

            try:
                trial.set_user_attr("error", str(exc))
            except Exception:
                pass
            # Let Optuna see a terrible value so it can move on
            if multi_objective:
                return -1e9, -1e9, 1e6
            return float("nan")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)

    # Build DataFrame from collected results
    df = pd.DataFrame(results)

    # Attach trial_id and order
    if not df.empty:
        if "trial_number" in df.columns:
            df.insert(0, "trial_id", df["trial_number"].astype(int))
        else:
            df.insert(0, "trial_id", range(len(df)))

        # Sort by Sharpe primarily, then final score as tiebreaker
        if "sharpe" in df.columns:
            df = df.sort_values(["sharpe", "score"], ascending=[False, False])

    # Attach study to config for callers that want deeper Optuna use
    try:
        if isinstance(config, dict):  # type: ignore[redundant-cast]
            config.setdefault("_optuna", {})["study"] = study  # type: ignore[index]
    except Exception:
        pass

    # Optionally append candidates for downstream comparison
    try:
        if candidates is not None:
            if not isinstance(candidates, pd.DataFrame):
                candidates_df = pd.DataFrame.from_records(list(candidates))
            else:
                candidates_df = candidates.copy()
            candidates_df["_source"] = "candidate_grid"
            df["_source"] = "optuna"  # type: ignore[index]
            df = pd.concat([df, candidates_df], ignore_index=True, sort=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to append candidates to results: %s", exc)

    return df


# ---------------------------------------------------------------------------
# Extra helpers for downstream code (analytics / deployment)
# ---------------------------------------------------------------------------


def summarize_results(df: pd.DataFrame) -> Dict[str, float]:
    """Summarize an optimization results DataFrame into a small KPI dict.

    Useful עבור דאשבורד, לוגים, או בדיקות סף.
    הפונקציה דפנסיבית ולא תזרוק שגיאה אם חלק מהעמודות חסרות.
    """

    if df is None or df.empty:
        return {
            "rows": 0.0,
            "best_sharpe": float("nan"),
            "avg_sharpe": float("nan"),
            "best_score": float("nan"),
        }

    out: Dict[str, float] = {"rows": float(len(df))}

    try:
        if "sharpe" in df.columns:
            s = pd.to_numeric(df["sharpe"], errors="coerce")
            out["best_sharpe"] = float(s.max())
            out["avg_sharpe"] = float(s.mean())
        if "score" in df.columns:
            sc = pd.to_numeric(df["score"], errors="coerce")
            out["best_score"] = float(sc.max())
        if "return" in df.columns:
            r = pd.to_numeric(df["return"], errors="coerce")
            out["max_return"] = float(r.max())
        if "drawdown" in df.columns:
            d = pd.to_numeric(df["drawdown"], errors="coerce")
            out["min_drawdown"] = float(d.min())
        if "runtime_sec" in df.columns:
            rt = pd.to_numeric(df["runtime_sec"], errors="coerce")
            out["avg_runtime_sec"] = float(rt.mean())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("summarize_results failed: %s", exc)

    return out


def extract_best_params(df: pd.DataFrame, metric: str = "sharpe") -> Dict[str, Any]:
    """Extract the parameter dict for the best row in ``df``.

    Parameters
    ----------
    df:
        Results DataFrame from :func:`run_optimization`.
    metric:
        Column to rank by (default ``'sharpe'``). If the metric is missing,
        the function falls back to the first numeric column.
    """

    if df is None or df.empty:
        return {}

    metric = str(metric)
    cols = list(df.columns)

    if metric not in cols:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric = num_cols[0] if num_cols else cols[0]

    try:
        best_idx = int(pd.to_numeric(df[metric], errors="coerce").idxmax())
    except Exception:
        best_idx = int(df.index[0])

    row = df.loc[best_idx]

    # Exclude known metric/metadata columns
    exclude = {
        "trial_id",
        "trial_number",
        "sharpe",
        "return",
        "drawdown",
        "win_rate",
        "score",
        "score_mode",
        "classic_score",
        "hf_score",
        "runtime_sec",
        metric,
    }
    params: Dict[str, Any] = {}
    for col in cols:
        if col in exclude:
            continue
        val = row.get(col)
        if isinstance(val, (float, int, str, bool)) or val is None:
            params[col] = val

    return params


def build_param_grid(ranges: ParamRanges, points: int = 5) -> pd.DataFrame:
    """Create a simple grid over the provided parameter ranges.

    לא מריץ בק-טסטים – רק מחזיר DataFrame של קומבינציות פרמטרים, שאפשר
    אחר-כך להזין ל-Backtester או להשתמש כ-seed למחקר.
    """

    if points <= 1:
        points = 2

    max_dims = 8  # hard cap to avoid combinatorial explosion
    items = list(ranges.items())[:max_dims]

    grids: List[np.ndarray] = []
    names: List[str] = []
    for name, (lo, hi, step) in items:
        names.append(name)
        if step is not None and step > 0:
            arr = np.arange(float(lo), float(hi) + 1e-9, float(step))
            if len(arr) > points:
                idx = np.linspace(0, len(arr) - 1, num=points).round().astype(int)
                arr = arr[idx]
        else:
            arr = np.linspace(float(lo), float(hi), num=points)
        grids.append(arr)

    mesh = np.meshgrid(*grids)
    stacked = np.array(mesh).T.reshape(-1, len(names))

    combos: List[Dict[str, float]] = []
    for vals in stacked:
        combos.append({k: float(v) for k, v in zip(names, vals)})

    return pd.DataFrame(combos)


def run_optimization_with_study(
    candidates: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Any]]]],
    config: Mapping[str, Any],
    n_trials: int = 50,
    study_name: str = "pair_opt_study",
) -> Tuple[pd.DataFrame, Optional[Study]]:
    """Wrapper around :func:`run_optimization` that also returns the Study.

    Keeps the original API (``run_optimization`` still returns only DataFrame),
    but allows callers to inspect the Optuna Study for deeper analytics.
    """

    df = run_optimization(
        candidates=candidates,
        config=config,
        n_trials=n_trials,
        study_name=study_name,
    )

    study: Optional[Study] = None
    try:
        if isinstance(config, Mapping):
            opt_block = config.get("_optuna") or {}
            if isinstance(opt_block, Mapping):
                study_obj = opt_block.get("study")
                if optuna is not None and isinstance(study_obj, optuna.study.Study):  # type: ignore[attr-defined]
                    study = study_obj  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("run_optimization_with_study: failed to extract Study: %s", exc)

    return df, study


# ---------------------------------------------------------------------------
# Advanced analytics helpers (Pareto, ranking, tagging)
# ---------------------------------------------------------------------------


def rank_trials(
    df: pd.DataFrame,
    by: str = "score",
    *,
    ascending: bool = False,
    top: Optional[int] = None,
) -> pd.DataFrame:
    """Return a ranked view of the trials DataFrame.

    Parameters
    ----------
    df:
        Results DataFrame returned by :func:`run_optimization`.
    by:
        Column to sort by (default ``'score'``). If missing, falls back to the
        first numeric column.
    ascending:
        Sort direction (default ``False`` for "best on top").
    top:
        Optional cap on number of rows to return.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    by = str(by)
    cols = df.columns.tolist()
    if by not in cols:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return df.copy()
        by = num_cols[0]

    ranked = df.sort_values(by, ascending=ascending)
    if top is not None:
        ranked = ranked.head(int(top))
    return ranked.reset_index(drop=True)


def compute_pareto_front(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    sharpe_col: str = "sharpe",
    dd_col: str = "drawdown",
) -> pd.DataFrame:
    """Compute a simple Pareto frontier over (score, sharpe, drawdown).

    Non-dominated points are those for which no other point is strictly
    better in all dimensions (higher score, higher Sharpe, lower drawdown).
    """

    if df is None or df.empty:
        return pd.DataFrame()

    cols = []
    for c in (score_col, sharpe_col, dd_col):
        if c in df.columns:
            cols.append(c)
    if len(cols) < 2:
        # Not enough information for a meaningful Pareto set
        return pd.DataFrame()

    dff = df.copy()
    for c in cols:
        dff[c] = pd.to_numeric(dff[c], errors="coerce")

    # Drop rows with all-NaN metrics
    dff = dff.dropna(subset=cols, how="all")
    if dff.empty:
        return pd.DataFrame()

    # Build matrix with consistent sign conventions
    # score, sharpe -> maximize; drawdown -> minimize
    score_vals = dff.get(score_col, pd.Series(index=dff.index, dtype=float)).fillna(-np.inf)
    sharpe_vals = dff.get(sharpe_col, pd.Series(index=dff.index, dtype=float)).fillna(-np.inf)
    dd_vals = dff.get(dd_col, pd.Series(index=dff.index, dtype=float)).fillna(np.inf)

    M = np.vstack([score_vals.values, sharpe_vals.values, dd_vals.values]).T

    def _is_dominated(idx: int, mat: np.ndarray) -> bool:
        s, sh, draw = mat[idx]
        better_or_equal = (mat[:, 0] >= s) & (mat[:, 1] >= sh) & (mat[:, 2] <= draw)
        strictly_better = (mat[:, 0] > s) | (mat[:, 1] > sh) | (mat[:, 2] < draw)
        return bool(np.any(better_or_equal & strictly_better))

    mask = np.array([not _is_dominated(i, M) for i in range(M.shape[0])])
    pareto = dff.loc[mask].copy()
    return pareto.reset_index(drop=True)


def tag_best_trials(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    top_k: int = 10,
) -> pd.DataFrame:
    """Add a boolean column ``is_best`` marking top-k trials by score.

    This is handy for visualizations (color-coding best points) or for
    filtering when exporting to execution profiles.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if score_col not in df.columns:
        return df.copy()

    dff = df.copy()
    s = pd.to_numeric(dff[score_col], errors="coerce")
    order = s.sort_values(ascending=False).index.to_list()
    top_k = max(1, int(top_k))
    top_idx = set(order[:top_k])
    dff["is_best"] = dff.index.isin(top_idx)
    return dff
