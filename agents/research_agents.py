# -*- coding: utf-8 -*-
"""
agents/research_agents.py — Research and Discovery Agent Implementations
=========================================================================

Eight agent classes covering the full research pipeline from universe curation
through candidate discovery, relationship validation, spread specification,
regime research, signal research, experiment coordination, and summarization.

All agents:
  - Subclass BaseAgent (from agents.base)
  - Handle ImportError gracefully with lightweight fallbacks
  - Return a proper dict from _execute() — never None
  - Use uuid.uuid4() for generated IDs
  - Use datetime.utcnow().isoformat() + "Z" for timestamps
  - Are fully type-annotated
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from agents.base import AgentAuditLogger, BaseAgent
from core.contracts import AgentTask


def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _new_id() -> str:
    return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════
# 1. UniverseCuratorAgent
# ══════════════════════════════════════════════════════════════════


class UniverseCuratorAgent(BaseAgent):
    """
    Curates a tradeable universe from a raw symbol list.

    Filters symbols based on price-data history length, average daily volume
    (if provided), and optional sector grouping.

    Task types
    ----------
    curate_universe
        Full curation pass: history check, vol computation, sector grouping.
    refresh_universe
        Lightweight re-check of an existing universe for staleness.

    Required payload keys
    ---------------------
    symbols : list[str]

    Optional payload keys
    ---------------------
    prices : pd.DataFrame
        Daily close prices (columns = symbols).
    min_history_days : int
        Minimum number of trading days required (default 252).
    min_avg_volume_M : float
        Minimum average daily volume in millions (default 1.0, ignored if
        volume data not present in prices).
    liquidity_filter : bool
        Whether to apply the volume filter (default True).
    sector_map : dict[str, str]
        Optional mapping of symbol → sector for grouping.

    Output keys
    -----------
    eligible_symbols : list[str]
    excluded_symbols : list[str]
    exclusion_reasons : dict[str, str]
    universe_size : int
    staleness_warnings : list[str]
    sector_groups : dict[str, list[str]]
    annualized_vols : dict[str, float]
    """

    NAME = "universe_curator"
    ALLOWED_TASK_TYPES = {"curate_universe", "refresh_universe"}
    REQUIRED_PAYLOAD_KEYS = {"symbols"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        import math

        try:
            import pandas as pd
            import numpy as np
            _has_pandas = True
        except ImportError:
            _has_pandas = False

        symbols: list[str] = task.payload["symbols"]
        prices = task.payload.get("prices")
        min_history: int = int(task.payload.get("min_history_days", 252))
        min_vol_M: float = float(task.payload.get("min_avg_volume_M", 1.0))
        liquidity_filter: bool = bool(task.payload.get("liquidity_filter", True))
        sector_map: dict[str, str] = task.payload.get("sector_map") or {}

        audit.log(
            f"Curating universe: {len(symbols)} symbols, "
            f"min_history={min_history}, liquidity_filter={liquidity_filter}"
        )

        eligible: list[str] = []
        excluded: list[str] = []
        exclusion_reasons: dict[str, str] = {}
        staleness_warnings: list[str] = []
        annualized_vols: dict[str, float] = {}

        if not _has_pandas or prices is None:
            if prices is None:
                audit.warn("No prices DataFrame provided — accepting all symbols as eligible")
            else:
                audit.warn("pandas unavailable — skipping price-based filters")
            eligible = list(symbols)
        else:
            try:
                import pandas as pd
                import numpy as np

                # Ensure we operate on a DataFrame
                if not hasattr(prices, "columns"):
                    raise ValueError("prices must be a pandas DataFrame")

                available_cols = set(prices.columns)

                for sym in symbols:
                    if sym not in available_cols:
                        excluded.append(sym)
                        exclusion_reasons[sym] = "no_price_data"
                        continue

                    series = prices[sym].dropna()
                    n_obs = len(series)

                    if n_obs < min_history:
                        excluded.append(sym)
                        exclusion_reasons[sym] = (
                            f"insufficient_history:{n_obs}<{min_history}"
                        )
                        continue

                    # Compute annualised vol
                    rets = series.pct_change().dropna()
                    if len(rets) >= 2:
                        ann_vol = float(rets.std() * math.sqrt(252))
                        annualized_vols[sym] = ann_vol
                    else:
                        annualized_vols[sym] = float("nan")

                    # Staleness check: last date within 7 calendar days
                    last_date = series.index[-1]
                    if hasattr(last_date, "date"):
                        last_date = last_date.date()
                    try:
                        from datetime import date
                        today = date.today()
                        days_stale = (today - last_date).days
                        if days_stale > 7:
                            staleness_warnings.append(
                                f"{sym}: last price {last_date} is {days_stale}d old"
                            )
                    except Exception:
                        pass

                    eligible.append(sym)

                audit.log(
                    f"After price filter: {len(eligible)} eligible, "
                    f"{len(excluded)} excluded"
                )
            except Exception as exc:
                audit.warn(f"Price filtering failed ({exc}) — returning all symbols as eligible")
                eligible = list(symbols)

        # Sector grouping
        sector_groups: dict[str, list[str]] = {}
        for sym in eligible:
            sector = sector_map.get(sym, "UNKNOWN")
            sector_groups.setdefault(sector, []).append(sym)

        audit.log(
            f"Universe curation complete: {len(eligible)} eligible symbols, "
            f"{len(sector_groups)} sectors, {len(staleness_warnings)} staleness warnings"
        )

        return {
            "eligible_symbols": eligible,
            "excluded_symbols": excluded,
            "exclusion_reasons": exclusion_reasons,
            "universe_size": len(eligible),
            "staleness_warnings": staleness_warnings,
            "sector_groups": sector_groups,
            "annualized_vols": annualized_vols,
        }


# ══════════════════════════════════════════════════════════════════
# 2. CandidateDiscoveryAgent
# ══════════════════════════════════════════════════════════════════


class CandidateDiscoveryAgent(BaseAgent):
    """
    Discovers candidate pairs from a universe using correlation and distance.

    Correlation is a discovery primitive, NOT tradability proof. Every candidate
    produced here must pass through RelationshipValidationAgent before routing
    to the portfolio layer.

    Task types
    ----------
    discover_candidates
        Full discovery scan across all symbol pairs.
    rank_candidates
        Re-rank an existing candidate list by composite score.

    Required payload keys
    ---------------------
    symbols : list[str]

    Optional payload keys
    ---------------------
    prices : pd.DataFrame
    max_pairs : int  (default 200)
    min_correlation : float  (default 0.60)
    discovery_families : list[str]  (default ["correlation"])

    Output keys
    -----------
    candidates : list[dict]
    n_candidates : int
    yield_rate : float
    discovery_summary : dict
    """

    NAME = "candidate_discovery"
    ALLOWED_TASK_TYPES = {"discover_candidates", "rank_candidates"}
    REQUIRED_PAYLOAD_KEYS = {"symbols"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        from itertools import combinations

        symbols: list[str] = task.payload["symbols"]
        prices = task.payload.get("prices")
        max_pairs: int = int(task.payload.get("max_pairs", 200))
        min_corr: float = float(task.payload.get("min_correlation", 0.60))
        families: list[str] = task.payload.get("discovery_families") or ["correlation"]

        audit.log(
            f"Discovering candidates: {len(symbols)} symbols, "
            f"min_corr={min_corr}, max_pairs={max_pairs}, families={families}"
        )

        n_possible = len(symbols) * (len(symbols) - 1) // 2 if len(symbols) >= 2 else 0
        candidates: list[dict[str, Any]] = []

        if prices is None or n_possible == 0:
            audit.warn("No prices or fewer than 2 symbols — returning empty candidate list")
            return {
                "candidates": [],
                "n_candidates": 0,
                "yield_rate": 0.0,
                "discovery_summary": {"n_possible": n_possible, "families_used": families},
            }

        try:
            import numpy as np
            import pandas as pd

            available = [s for s in symbols if s in prices.columns]
            audit.log(f"{len(available)}/{len(symbols)} symbols have price data")

            if len(available) < 2:
                return {
                    "candidates": [],
                    "n_candidates": 0,
                    "yield_rate": 0.0,
                    "discovery_summary": {"n_possible": 0, "families_used": families},
                }

            returns = prices[available].pct_change().dropna()
            corr = returns.corr()

            # Normalised prices for distance calculation
            price_sub = prices[available].dropna()
            normed = (price_sub - price_sub.mean()) / (price_sub.std() + 1e-12)

            for sx, sy in combinations(available, 2):
                c = float(corr.loc[sx, sy]) if (sx in corr.index and sy in corr.columns) else 0.0
                if abs(c) < min_corr:
                    continue

                # Euclidean distance on normalised prices
                try:
                    dist = float(np.linalg.norm(normed[sx].values - normed[sy].values))
                except Exception:
                    dist = float("nan")

                # Composite score: high corr + low distance is better
                dist_score = 1.0 / (1.0 + max(dist, 0.0)) if not (dist != dist) else 0.5
                score = 0.6 * abs(c) + 0.4 * dist_score

                candidates.append(
                    {
                        "sym_x": min(sx, sy),
                        "sym_y": max(sx, sy),
                        "correlation": round(c, 6),
                        "distance": round(dist, 4) if dist == dist else None,
                        "score": round(score, 6),
                    }
                )

            candidates.sort(key=lambda x: -(x["score"] or 0.0))
            candidates = candidates[:max_pairs]

        except Exception as exc:
            audit.warn(f"Candidate discovery computation failed: {exc}")
            candidates = []

        n_possible_with_data = len(list(combinations(
            [s for s in symbols if prices is not None and s in prices.columns], 2
        ))) if prices is not None else n_possible
        yield_rate = (
            float(len(candidates)) / float(n_possible_with_data)
            if n_possible_with_data > 0
            else 0.0
        )

        audit.log(
            f"Discovery complete: {len(candidates)} candidates "
            f"(yield_rate={yield_rate:.2%}) from {n_possible_with_data} possible pairs"
        )

        return {
            "candidates": candidates,
            "n_candidates": len(candidates),
            "yield_rate": round(yield_rate, 6),
            "discovery_summary": {
                "n_possible": n_possible_with_data,
                "n_above_threshold": len(candidates),
                "families_used": families,
                "min_correlation": min_corr,
            },
        }


# ══════════════════════════════════════════════════════════════════
# 3. RelationshipValidationAgent
# ══════════════════════════════════════════════════════════════════


class RelationshipValidationAgent(BaseAgent):
    """
    Validates candidate pairs using cointegration and stationarity tests.

    Attempts to use ``research.pair_validator.PairValidator`` if available;
    falls back to a lightweight ADF test via ``statsmodels`` or ``scipy``.

    Task types
    ----------
    validate_relationships
        Batch validation of a list of pair IDs.
    revalidate_pair
        Single-pair re-validation.

    Required payload keys
    ---------------------
    pair_ids : list[dict]  (each: {"sym_x": str, "sym_y": str})

    Optional payload keys
    ---------------------
    prices : pd.DataFrame
    train_end : str  (ISO date/datetime)

    Output keys
    -----------
    validation_results : list[dict]
    n_passed : int
    n_failed : int
    n_warned : int
    pass_rate : float
    """

    NAME = "relationship_validation"
    ALLOWED_TASK_TYPES = {"validate_relationships", "revalidate_pair"}
    REQUIRED_PAYLOAD_KEYS = {"pair_ids"}

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _adf_test(series: Any) -> tuple[float, str]:
        """Return (p_value, method_used)."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series.dropna(), autolag="AIC")
            return float(result[1]), "statsmodels_adf"
        except ImportError:
            pass
        try:
            import numpy as np
            # Approximate via OLS AR(1) residuals — very rough
            y = series.dropna().values.astype(float)
            if len(y) < 10:
                return 0.5, "insufficient_data"
            dy = y[1:] - y[:-1]
            y_lag = y[:-1]
            cov = float(np.cov(dy, y_lag)[0, 1])
            var_lag = float(np.var(y_lag))
            rho = cov / (var_lag + 1e-12)
            # Very rough heuristic
            if rho < -0.15:
                return 0.04, "heuristic_ar1"
            elif rho < 0.0:
                return 0.12, "heuristic_ar1"
            return 0.50, "heuristic_ar1"
        except Exception:
            return 0.5, "fallback"

    def _validate_pair_lightweight(
        self,
        sym_x: str,
        sym_y: str,
        prices: Any,
        train_end: Any,
        audit: AgentAuditLogger,
    ) -> dict[str, Any]:
        """Lightweight fallback validator using ADF on OLS residuals."""
        pair_label = f"{min(sym_x, sym_y)}/{max(sym_x, sym_y)}"
        try:
            import numpy as np
            import pandas as pd

            px = prices[sym_x].dropna()
            py = prices[sym_y].dropna()

            if train_end is not None:
                px = px.loc[px.index <= train_end]
                py = py.loc[py.index <= train_end]

            common = px.index.intersection(py.index)
            if len(common) < 60:
                return {
                    "pair_label": pair_label,
                    "result": "FAIL",
                    "rejection_reasons": [f"insufficient_overlap:{len(common)}<60"],
                    "warnings": [],
                    "method": "lightweight",
                }

            px_c = px.loc[common].values.astype(float)
            py_c = py.loc[common].values.astype(float)

            # OLS: py = beta * px + alpha + residual
            beta, alpha = float(np.polyfit(px_c, py_c, 1))
            residuals = pd.Series(py_c - beta * px_c - alpha)

            p_val, method = self._adf_test(residuals)
            audit.log(f"{pair_label}: ADF p={p_val:.4f} ({method}), beta={beta:.4f}")

            if p_val <= 0.10:
                return {
                    "pair_label": pair_label,
                    "result": "PASS",
                    "rejection_reasons": [],
                    "warnings": [],
                    "adf_p_value": round(p_val, 6),
                    "hedge_ratio": round(beta, 6),
                    "method": method,
                }
            elif p_val <= 0.20:
                return {
                    "pair_label": pair_label,
                    "result": "WARN",
                    "rejection_reasons": [],
                    "warnings": [f"marginal_adf_p:{p_val:.4f}"],
                    "adf_p_value": round(p_val, 6),
                    "hedge_ratio": round(beta, 6),
                    "method": method,
                }
            else:
                return {
                    "pair_label": pair_label,
                    "result": "FAIL",
                    "rejection_reasons": [f"adf_p_too_high:{p_val:.4f}>0.10"],
                    "warnings": [],
                    "adf_p_value": round(p_val, 6),
                    "hedge_ratio": round(beta, 6),
                    "method": method,
                }
        except Exception as exc:
            audit.warn(f"{pair_label}: lightweight validation error — {exc}")
            return {
                "pair_label": pair_label,
                "result": "FAIL",
                "rejection_reasons": [f"validation_error:{exc}"],
                "warnings": [],
                "method": "error",
            }

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        raw_pairs: list[dict[str, str]] = task.payload["pair_ids"]
        prices = task.payload.get("prices")
        train_end = task.payload.get("train_end")

        audit.log(
            f"Validating {len(raw_pairs)} pairs, "
            f"train_end={train_end}, prices={'provided' if prices is not None else 'missing'}"
        )

        # Try full PairValidator first
        _full_validator = None
        try:
            from core.contracts import PairId
            from research.pair_validator import PairValidator
            _full_validator = PairValidator()
            audit.log("Using research.pair_validator.PairValidator")
        except ImportError:
            audit.warn("research.pair_validator unavailable — using lightweight ADF fallback")

        results: list[dict[str, Any]] = []

        for p in raw_pairs:
            sym_x = p.get("sym_x", "")
            sym_y = p.get("sym_y", "")
            pair_label = f"{min(sym_x, sym_y)}/{max(sym_x, sym_y)}"

            if _full_validator is not None and prices is not None:
                try:
                    from core.contracts import PairId
                    pid = PairId(sym_x, sym_y)
                    report = _full_validator.validate(pid, prices, train_end=train_end)
                    results.append(
                        {
                            "pair_label": pair_label,
                            "result": report.result.value,
                            "rejection_reasons": [r.value if hasattr(r, "value") else str(r) for r in report.rejection_reasons],
                            "warnings": [w.value if hasattr(w, "value") else str(w) for w in getattr(report, "warnings", [])],
                            "method": "PairValidator",
                        }
                    )
                except Exception as exc:
                    audit.warn(f"{pair_label}: PairValidator error ({exc}), falling back")
                    if prices is not None:
                        results.append(
                            self._validate_pair_lightweight(sym_x, sym_y, prices, train_end, audit)
                        )
                    else:
                        results.append(
                            {
                                "pair_label": pair_label,
                                "result": "FAIL",
                                "rejection_reasons": ["no_prices_and_validator_failed"],
                                "warnings": [],
                                "method": "error",
                            }
                        )
            elif prices is not None:
                results.append(
                    self._validate_pair_lightweight(sym_x, sym_y, prices, train_end, audit)
                )
            else:
                audit.warn(f"{pair_label}: no prices available for validation")
                results.append(
                    {
                        "pair_label": pair_label,
                        "result": "FAIL",
                        "rejection_reasons": ["no_price_data"],
                        "warnings": [],
                        "method": "no_data",
                    }
                )

        n_passed = sum(1 for r in results if r["result"] == "PASS")
        n_failed = sum(1 for r in results if r["result"] == "FAIL")
        n_warned = sum(1 for r in results if r["result"] == "WARN")
        n_total = len(results)
        pass_rate = float(n_passed) / float(n_total) if n_total > 0 else 0.0

        audit.log(
            f"Validation complete: {n_passed} PASS, {n_warned} WARN, {n_failed} FAIL "
            f"(pass_rate={pass_rate:.2%})"
        )

        return {
            "validation_results": results,
            "n_passed": n_passed,
            "n_failed": n_failed,
            "n_warned": n_warned,
            "pass_rate": round(pass_rate, 6),
        }


# ══════════════════════════════════════════════════════════════════
# 4. SpreadSpecificationAgent
# ══════════════════════════════════════════════════════════════════


class SpreadSpecificationAgent(BaseAgent):
    """
    Fits spread definitions for validated pairs.

    Attempts to use ``research.spread_constructor.build_spread`` if available;
    falls back to a manual numpy-based OLS.

    Task types
    ----------
    specify_spreads
        Batch spread fitting for a list of pair IDs.
    refit_spread
        Single-pair re-fit with possibly updated parameters.

    Required payload keys
    ---------------------
    pair_ids : list[dict]  (each: {"sym_x": str, "sym_y": str})
    prices : pd.DataFrame

    Optional payload keys
    ---------------------
    model : str  ("STATIC_OLS" | "ROLLING_OLS" | "KALMAN", default "STATIC_OLS")
    window : int  (default 60)
    train_end : str  (ISO date)

    Output keys
    -----------
    spread_specs : list[dict]
    n_fitted : int
    n_failed : int
    """

    NAME = "spread_specification"
    ALLOWED_TASK_TYPES = {"specify_spreads", "refit_spread"}
    REQUIRED_PAYLOAD_KEYS = {"pair_ids", "prices"}

    def _ols_fallback(
        self,
        sym_x: str,
        sym_y: str,
        prices: Any,
        train_end: Any,
        window: int,
        audit: AgentAuditLogger,
    ) -> dict[str, Any]:
        """Manual OLS spread fitting via numpy."""
        pair_label = f"{min(sym_x, sym_y)}/{max(sym_x, sym_y)}"
        try:
            import numpy as np

            px = prices[sym_x].dropna()
            py = prices[sym_y].dropna()

            if train_end is not None:
                px = px.loc[px.index <= train_end]
                py = py.loc[py.index <= train_end]

            common = px.index.intersection(py.index)
            if len(common) < 20:
                raise ValueError(f"insufficient_overlap: {len(common)}<20")

            px_c = px.loc[common].values.astype(float)
            py_c = py.loc[common].values.astype(float)

            beta, alpha = float(np.polyfit(px_c, py_c, 1))
            spread = py_c - beta * px_c - alpha

            return {
                "pair_label": pair_label,
                "model": "STATIC_OLS_FALLBACK",
                "hedge_ratio": round(beta, 6),
                "intercept": round(alpha, 6),
                "spread_mean": round(float(spread.mean()), 6),
                "spread_std": round(float(spread.std()), 6),
                "spread_length": int(len(spread)),
                "status": "ok",
            }
        except Exception as exc:
            audit.warn(f"{pair_label}: OLS fallback failed — {exc}")
            return {
                "pair_label": pair_label,
                "model": "STATIC_OLS_FALLBACK",
                "hedge_ratio": None,
                "intercept": None,
                "spread_mean": None,
                "spread_std": None,
                "spread_length": 0,
                "status": f"error:{exc}",
            }

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        raw_pairs: list[dict[str, str]] = task.payload["pair_ids"]
        prices = task.payload["prices"]
        model_str: str = str(task.payload.get("model", "STATIC_OLS")).upper()
        window: int = int(task.payload.get("window", 60))
        train_end = task.payload.get("train_end")

        audit.log(
            f"Fitting spreads for {len(raw_pairs)} pairs, "
            f"model={model_str}, window={window}, train_end={train_end}"
        )

        # Try full spread constructor first
        _use_full = False
        try:
            from core.contracts import PairId, SpreadModel
            from research.spread_constructor import build_spread
            _use_full = True
            audit.log("Using research.spread_constructor.build_spread")
        except ImportError:
            audit.warn("research.spread_constructor unavailable — using OLS fallback")

        spread_specs: list[dict[str, Any]] = []
        n_failed = 0

        for p in raw_pairs:
            sym_x = p.get("sym_x", "")
            sym_y = p.get("sym_y", "")
            pair_label = f"{min(sym_x, sym_y)}/{max(sym_x, sym_y)}"

            if _use_full:
                try:
                    from core.contracts import PairId, SpreadModel
                    from research.spread_constructor import build_spread
                    try:
                        mdl = SpreadModel(model_str)
                    except ValueError:
                        mdl = SpreadModel("STATIC_OLS")
                        audit.warn(f"Unknown model '{model_str}', using STATIC_OLS")
                    pid = PairId(sym_x, sym_y)
                    defn, series = build_spread(
                        pid, prices, model=mdl, train_end=train_end, window=window
                    )
                    spread = series.dropna()
                    spread_specs.append(
                        {
                            "pair_label": pair_label,
                            "model": mdl.value,
                            "hedge_ratio": round(float(defn.hedge_ratio), 6),
                            "intercept": round(float(defn.intercept), 6),
                            "spread_mean": round(float(spread.mean()), 6) if len(spread) else None,
                            "spread_std": round(float(spread.std()), 6) if len(spread) > 1 else None,
                            "spread_length": int(len(spread)),
                            "status": "ok",
                        }
                    )
                    audit.log(f"{pair_label}: fitted, beta={defn.hedge_ratio:.4f}")
                except Exception as exc:
                    audit.warn(f"{pair_label}: build_spread failed ({exc}), using OLS fallback")
                    spec = self._ols_fallback(sym_x, sym_y, prices, train_end, window, audit)
                    if spec["status"] == "ok":
                        spread_specs.append(spec)
                    else:
                        n_failed += 1
            else:
                spec = self._ols_fallback(sym_x, sym_y, prices, train_end, window, audit)
                if spec["status"] == "ok":
                    spread_specs.append(spec)
                else:
                    n_failed += 1

        audit.log(f"Spread fitting complete: {len(spread_specs)} fitted, {n_failed} failed")
        return {
            "spread_specs": spread_specs,
            "n_fitted": len(spread_specs),
            "n_failed": n_failed,
        }


# ══════════════════════════════════════════════════════════════════
# 5. RegimeResearchAgent
# ══════════════════════════════════════════════════════════════════


class RegimeResearchAgent(BaseAgent):
    """
    Classifies market regimes for each pair spread.

    Attempts to use ``core.regime_engine.RegimeEngine``. Falls back to a
    Hurst-exponent heuristic if the engine is unavailable.

    Task types
    ----------
    classify_regimes
        Classify the current regime for each provided spread.
    regime_transition_study
        Detect transitions relative to a prior regime map.

    Required payload keys
    ---------------------
    spreads : dict[str, list | pd.Series]
        Mapping of pair_label → spread values.

    Optional payload keys
    ---------------------
    prices : pd.DataFrame
    as_of : str  (ISO date)
    lookback_days : int  (default 252)
    prior_regimes : dict[str, str]  (for transition detection)

    Output keys
    -----------
    regimes : dict[str, str]
    regime_distribution : dict[str, int]
    mean_reverting_fraction : float
    crisis_detected : bool
    regime_transitions : list[dict]
    """

    NAME = "regime_research"
    ALLOWED_TASK_TYPES = {"classify_regimes", "regime_transition_study"}
    REQUIRED_PAYLOAD_KEYS = {"spreads"}

    @staticmethod
    def _hurst_exponent(series: Any) -> float:
        """Compute a rough Hurst exponent estimate via R/S analysis."""
        try:
            import numpy as np
            ts = series[-512:] if len(series) > 512 else series
            n = len(ts)
            if n < 20:
                return 0.5
            lags = [int(n * f) for f in [0.1, 0.25, 0.5] if int(n * f) >= 4]
            if not lags:
                return 0.5
            rs_vals = []
            lag_vals = []
            for lag in lags:
                sub = ts[:lag]
                mean = float(sub.mean())
                std = float(sub.std())
                if std < 1e-12:
                    continue
                deviation = sub - mean
                cumdev = deviation.cumsum()
                r = float(cumdev.max() - cumdev.min())
                rs = r / std
                rs_vals.append(rs)
                lag_vals.append(lag)
            if len(rs_vals) < 2:
                return 0.5
            log_rs = np.log(rs_vals)
            log_lag = np.log(lag_vals)
            hurst = float(np.polyfit(log_lag, log_rs, 1)[0])
            return max(0.0, min(1.0, hurst))
        except Exception:
            return 0.5

    @staticmethod
    def _classify_from_hurst(hurst: float) -> str:
        """Convert Hurst exponent to regime label."""
        if hurst < 0.40:
            return "MEAN_REVERTING"
        elif hurst < 0.55:
            return "TRANSITIONAL"
        elif hurst < 0.75:
            return "TRENDING"
        else:
            return "TRENDING"

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        spreads: dict[str, Any] = task.payload["spreads"]
        as_of = task.payload.get("as_of")
        lookback: int = int(task.payload.get("lookback_days", 252))
        prior_regimes: dict[str, str] = task.payload.get("prior_regimes") or {}

        audit.log(
            f"Classifying regimes for {len(spreads)} pairs, "
            f"as_of={as_of}, lookback={lookback}"
        )

        # Try full RegimeEngine
        _use_engine = False
        try:
            from core.regime_engine import RegimeEngine
            _engine = RegimeEngine()
            _use_engine = True
            audit.log("Using core.regime_engine.RegimeEngine")
        except (ImportError, Exception) as exc:
            audit.warn(f"RegimeEngine unavailable ({exc}) — using Hurst exponent fallback")
            _engine = None

        regimes: dict[str, str] = {}
        crisis_detected = False

        for label, spread_data in spreads.items():
            try:
                import numpy as np

                # Coerce spread to numpy array
                if hasattr(spread_data, "values"):
                    arr = spread_data.values.astype(float)
                else:
                    arr = np.array(spread_data, dtype=float)

                arr = arr[~np.isnan(arr)]
                arr = arr[-lookback:] if len(arr) > lookback else arr

                if _use_engine and _engine is not None:
                    try:
                        import pandas as pd
                        from core.regime_engine import build_regime_features
                        prices = task.payload.get("prices")
                        spread_series = pd.Series(arr)
                        features = build_regime_features(
                            spread=spread_series,
                            prices_x=None,
                            prices_y=None,
                            as_of=None,
                        )
                        label_obj, _conf = _engine.classify(features)
                        regime_str = label_obj.value if hasattr(label_obj, "value") else str(label_obj)
                    except Exception as exc2:
                        audit.warn(f"{label}: RegimeEngine.classify failed ({exc2}), using heuristic")
                        hurst = self._hurst_exponent(arr)
                        regime_str = self._classify_from_hurst(hurst)
                else:
                    hurst = self._hurst_exponent(arr)
                    regime_str = self._classify_from_hurst(hurst)

                regimes[label] = regime_str
                if regime_str in ("CRISIS", "BROKEN"):
                    crisis_detected = True

            except Exception as exc:
                audit.warn(f"{label}: regime classification error — {exc}")
                regimes[label] = "UNKNOWN"

        # Compute distribution
        regime_distribution: dict[str, int] = {}
        for r in regimes.values():
            regime_distribution[r] = regime_distribution.get(r, 0) + 1

        n_total = len(regimes)
        n_mr = regime_distribution.get("MEAN_REVERTING", 0)
        mean_reverting_fraction = float(n_mr) / float(n_total) if n_total > 0 else 0.0

        # Transition detection
        transitions: list[dict[str, str]] = []
        for label, current_regime in regimes.items():
            prior = prior_regimes.get(label)
            if prior and prior != current_regime:
                transitions.append(
                    {"label": label, "from": prior, "to": current_regime}
                )

        audit.log(
            f"Regime classification complete: {regime_distribution}, "
            f"mean_reverting_fraction={mean_reverting_fraction:.2%}, "
            f"crisis_detected={crisis_detected}, transitions={len(transitions)}"
        )

        return {
            "regimes": regimes,
            "regime_distribution": regime_distribution,
            "mean_reverting_fraction": round(mean_reverting_fraction, 6),
            "crisis_detected": crisis_detected,
            "regime_transitions": transitions,
        }


# ══════════════════════════════════════════════════════════════════
# 6. SignalResearchAgent
# ══════════════════════════════════════════════════════════════════


class SignalResearchAgent(BaseAgent):
    """
    Analyses signal quality across a set of pair z-scores.

    For each pair, classifies the current action (WATCH/ENTER/HOLD/EXIT),
    grades signal quality, and optionally computes a historical hit rate.

    Task types
    ----------
    analyze_signal_quality
        Assess current signal quality across all provided z-scores.
    compare_signal_families
        Compare quality metrics across different signal families.

    Required payload keys
    ---------------------
    spreads : dict[str, list | pd.Series]
    signals : dict[str, float]  (pair_label → current z-score)

    Optional payload keys
    ---------------------
    regimes : dict[str, str]  (pair_label → regime string)
    lookback_vol : int  (default 20)

    Output keys
    -----------
    signal_quality : dict[str, dict]
    portfolio_summary : dict
    """

    NAME = "signal_research"
    ALLOWED_TASK_TYPES = {"analyze_signal_quality", "compare_signal_families"}
    REQUIRED_PAYLOAD_KEYS = {"spreads", "signals"}

    @staticmethod
    def _action_from_z(z: float, regime: str) -> str:
        """Classify trading action from z-score and regime."""
        # Hard veto for unsafe regimes
        if regime in ("CRISIS", "BROKEN", "TRENDING"):
            return "SKIP"
        abs_z = abs(z)
        if abs_z >= 2.0:
            return "ENTER"
        elif abs_z >= 1.5:
            return "WATCH"
        elif abs_z <= 0.5:
            return "EXIT"
        else:
            return "HOLD"

    @staticmethod
    def _quality_grade(z: float, regime: str, conviction: float) -> str:
        """Return a quality grade A+/A/B/C/D/F."""
        if regime in ("CRISIS", "BROKEN", "TRENDING"):
            return "F"
        abs_z = abs(z)
        if conviction >= 0.80 and abs_z >= 2.0 and regime == "MEAN_REVERTING":
            return "A+"
        elif conviction >= 0.65 and abs_z >= 1.8:
            return "A"
        elif conviction >= 0.50 and abs_z >= 1.5:
            return "B"
        elif conviction >= 0.35 and abs_z >= 1.2:
            return "C"
        elif abs_z >= 0.8:
            return "D"
        return "F"

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        spreads: dict[str, Any] = task.payload["spreads"]
        signals: dict[str, float] = task.payload["signals"]
        regimes: dict[str, str] = task.payload.get("regimes") or {}
        lookback_vol: int = int(task.payload.get("lookback_vol", 20))

        audit.log(
            f"Analysing signal quality for {len(signals)} pairs, "
            f"lookback_vol={lookback_vol}"
        )

        signal_quality: dict[str, dict[str, Any]] = {}

        for label, z_score in signals.items():
            z = float(z_score) if z_score is not None else 0.0
            regime = regimes.get(label, "UNKNOWN")

            # Compute rolling vol from spread if available
            vol = None
            hit_rate = None
            spread_data = spreads.get(label)
            if spread_data is not None:
                try:
                    import numpy as np
                    if hasattr(spread_data, "values"):
                        arr = spread_data.values.astype(float)
                    else:
                        arr = np.array(spread_data, dtype=float)
                    arr = arr[~np.isnan(arr)]
                    if len(arr) >= lookback_vol:
                        recent = arr[-lookback_vol:]
                        vol = float(recent.std())
                    # Hit-rate: fraction of periods where spread reverted toward mean
                    if len(arr) >= 5:
                        spread_mean = float(arr.mean())
                        above = arr > spread_mean
                        next_down = (arr[1:] < arr[:-1])[above[:-1]]
                        below_arr = arr <= spread_mean
                        next_up = (arr[1:] > arr[:-1])[below_arr[:-1]]
                        hits = int(next_down.sum()) + int(next_up.sum())
                        total = int(above[:-1].sum()) + int(below_arr[:-1].sum())
                        hit_rate = float(hits) / float(total) if total > 0 else None
                except Exception:
                    pass

            # Conviction from |z| capped at 1.0
            conviction = min(abs(z) / 3.0, 1.0)
            action = self._action_from_z(z, regime)
            grade = self._quality_grade(z, regime, conviction)

            notes: list[str] = []
            if abs(z) > 4.0:
                notes.append("extreme_z_score:verify_data")
            if regime in ("CRISIS", "BROKEN"):
                notes.append(f"regime_veto:{regime}")
            if vol is not None and vol < 1e-6:
                notes.append("near_zero_vol:stale_spread")

            signal_quality[label] = {
                "z_score": round(z, 6),
                "action": action,
                "quality_grade": grade,
                "conviction": round(conviction, 4),
                "regime": regime,
                "rolling_vol": round(vol, 6) if vol is not None else None,
                "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
                "notes": notes,
            }

        # Portfolio summary
        active_signals = [
            lbl for lbl, sq in signal_quality.items()
            if sq["action"] in ("ENTER", "WATCH")
        ]
        convictions = [sq["conviction"] for sq in signal_quality.values()]
        avg_conviction = (
            float(sum(convictions)) / float(len(convictions)) if convictions else 0.0
        )
        grade_dist: dict[str, int] = {}
        for sq in signal_quality.values():
            g = sq["quality_grade"]
            grade_dist[g] = grade_dist.get(g, 0) + 1

        audit.log(
            f"Signal analysis complete: {len(active_signals)} active signals, "
            f"avg_conviction={avg_conviction:.3f}, grade_distribution={grade_dist}"
        )

        return {
            "signal_quality": signal_quality,
            "portfolio_summary": {
                "active_signals": len(active_signals),
                "active_signal_labels": active_signals,
                "avg_conviction": round(avg_conviction, 6),
                "quality_distribution": grade_dist,
                "total_pairs": len(signal_quality),
            },
        }


# ══════════════════════════════════════════════════════════════════
# 7. ExperimentCoordinatorAgent
# ══════════════════════════════════════════════════════════════════


class ExperimentCoordinatorAgent(BaseAgent):
    """
    Plans and tracks research experiments.

    Produces structured experiment manifests with reproducibility seeds
    and parameter validation.

    Task types
    ----------
    plan_experiment
        Generate an experiment plan from type and parameters.
    summarize_experiment_batch
        Aggregate results across multiple experiment runs.

    Required payload keys
    ---------------------
    experiment_type : str

    Optional payload keys
    ---------------------
    parameters : dict
    n_trials : int  (default 10)
    pair_ids : list[dict]

    Output keys
    -----------
    experiment_id : str
    experiment_type : str
    plan : dict
    estimated_duration_minutes : float
    reproducibility_seed : int
    warnings : list[str]
    """

    NAME = "experiment_coordinator"
    ALLOWED_TASK_TYPES = {"plan_experiment", "summarize_experiment_batch"}
    REQUIRED_PAYLOAD_KEYS = {"experiment_type"}

    # Approximate duration (minutes) per trial by experiment type
    _DURATION_PER_TRIAL: dict[str, float] = {
        "WALK_FORWARD_OPTIMIZATION": 5.0,
        "REGIME_BACKTEST": 2.0,
        "HYPERPARAMETER_SEARCH": 3.0,
        "UNIVERSE_SWEEP": 1.0,
        "CORRELATION_STUDY": 0.5,
        "COINTEGRATION_STUDY": 1.0,
        "SIGNAL_QUALITY_STUDY": 0.5,
        "DEFAULT": 2.0,
    }

    # Valid parameter ranges by type
    _PARAM_RANGES: dict[str, dict[str, tuple]] = {
        "min_correlation": (0.0, 1.0),
        "max_pairs": (1, 10_000),
        "window": (5, 1000),
        "lookback_days": (20, 2520),
        "n_splits": (2, 20),
        "test_days": (20, 1260),
        "min_train_days": (100, 5040),
        "embargo_days": (0, 100),
    }

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        import random

        experiment_type: str = str(task.payload["experiment_type"]).upper()
        parameters: dict[str, Any] = task.payload.get("parameters") or {}
        n_trials: int = int(task.payload.get("n_trials", 10))
        pair_ids: list[dict[str, str]] = task.payload.get("pair_ids") or []

        audit.log(
            f"Planning experiment: type={experiment_type}, n_trials={n_trials}, "
            f"n_pairs={len(pair_ids)}, n_params={len(parameters)}"
        )

        warnings: list[str] = []

        # Validate parameter ranges
        validated_params: dict[str, Any] = {}
        for key, val in parameters.items():
            if key in self._PARAM_RANGES:
                lo, hi = self._PARAM_RANGES[key]
                try:
                    fval = float(val)
                    if not (lo <= fval <= hi):
                        warnings.append(
                            f"parameter '{key}'={val} outside expected range [{lo}, {hi}]"
                        )
                    validated_params[key] = val
                except (TypeError, ValueError):
                    warnings.append(f"parameter '{key}'={val!r} could not be validated as numeric")
                    validated_params[key] = val
            else:
                validated_params[key] = val

        # Reproducibility seed
        seed = int(abs(hash(f"{experiment_type}:{n_trials}:{sorted(validated_params.items())}")) % (2**31))

        # Duration estimate
        per_trial = self._DURATION_PER_TRIAL.get(experiment_type, self._DURATION_PER_TRIAL["DEFAULT"])
        estimated_minutes = per_trial * n_trials

        # Build plan
        plan = {
            "experiment_type": experiment_type,
            "n_trials": n_trials,
            "parameters": validated_params,
            "pair_ids": [f"{p.get('sym_x','')}/{p.get('sym_y','')}" for p in pair_ids],
            "n_pairs": len(pair_ids),
            "step_sequence": [
                "universe_curation",
                "candidate_discovery",
                "relationship_validation",
                "spread_fitting",
                "regime_classification",
                "signal_quality_assessment",
                "summarization",
            ],
            "reproducibility_seed": seed,
            "created_at": _utcnow(),
        }

        if estimated_minutes > 120:
            warnings.append(f"estimated_duration={estimated_minutes:.1f}min may exceed 2h — consider reducing n_trials")

        experiment_id = _new_id()
        audit.log(
            f"Experiment plan created: id={experiment_id}, "
            f"est_duration={estimated_minutes:.1f}min, warnings={len(warnings)}"
        )

        return {
            "experiment_id": experiment_id,
            "experiment_type": experiment_type,
            "plan": plan,
            "estimated_duration_minutes": round(estimated_minutes, 2),
            "reproducibility_seed": seed,
            "warnings": warnings,
        }


# ══════════════════════════════════════════════════════════════════
# 8. ResearchSummarizationAgent
# ══════════════════════════════════════════════════════════════════


class ResearchSummarizationAgent(BaseAgent):
    """
    Aggregates research pipeline results into a structured summary.

    Produces a human-readable research summary with key findings, warnings,
    and actionable recommendations.

    Task types
    ----------
    summarize_research_run
        Aggregate a completed discovery pipeline run.
    generate_universe_report
        Summarize universe eligibility statistics.

    Optional payload keys
    ---------------------
    validation_results : list[dict]
    regime_map : dict[str, str]
    candidate_count : int
    pass_rate : float
    run_metadata : dict

    Output keys
    -----------
    summary_id : str
    universe_size : int
    candidates_generated : int
    pairs_validated : int
    pairs_passed : int
    pass_rate : float
    regime_distribution : dict
    key_findings : list[str]
    warnings : list[str]
    recommendations : list[str]
    """

    NAME = "research_summarization"
    ALLOWED_TASK_TYPES = {"summarize_research_run", "generate_universe_report"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        validation_results: list[dict] = task.payload.get("validation_results") or []
        regime_map: dict[str, str] = task.payload.get("regime_map") or {}
        candidate_count: int = int(task.payload.get("candidate_count") or len(validation_results))
        pass_rate_override = task.payload.get("pass_rate")
        run_metadata: dict[str, Any] = task.payload.get("run_metadata") or {}

        audit.log(
            f"Summarizing research run: {len(validation_results)} validation results, "
            f"{len(regime_map)} regime classifications"
        )

        # Compute aggregate stats
        n_validated = len(validation_results)
        n_passed = sum(1 for r in validation_results if r.get("result") == "PASS")
        n_warned = sum(1 for r in validation_results if r.get("result") == "WARN")
        n_failed = sum(1 for r in validation_results if r.get("result") == "FAIL")
        pass_rate = (
            float(pass_rate_override)
            if pass_rate_override is not None
            else (float(n_passed) / float(n_validated) if n_validated > 0 else 0.0)
        )

        # Regime distribution
        regime_distribution: dict[str, int] = {}
        for regime in regime_map.values():
            regime_distribution[regime] = regime_distribution.get(regime, 0) + 1

        # Universe size from metadata or fallback
        universe_size: int = int(
            run_metadata.get("universe_size", 0)
            or run_metadata.get("n_symbols", 0)
            or 0
        )

        # Key findings
        key_findings: list[str] = []
        warnings: list[str] = []
        recommendations: list[str] = []

        if n_validated > 0:
            key_findings.append(
                f"Validated {n_validated} pairs: {n_passed} PASS, {n_warned} WARN, {n_failed} FAIL"
            )
            key_findings.append(f"Overall pass rate: {pass_rate:.1%}")

        if pass_rate > 0.20:
            warnings.append(
                f"Pass rate {pass_rate:.1%} exceeds 20% — potential false discovery inflation; "
                "verify statistical thresholds"
            )

        if candidate_count > 0:
            yield_rate = float(n_passed) / float(candidate_count) if candidate_count else 0.0
            key_findings.append(f"Candidate yield rate: {yield_rate:.1%} ({n_passed}/{candidate_count})")

        n_mr = regime_distribution.get("MEAN_REVERTING", 0)
        n_total_regimes = sum(regime_distribution.values())
        if n_total_regimes > 0:
            mr_frac = float(n_mr) / float(n_total_regimes)
            key_findings.append(
                f"Mean-reverting regime fraction: {mr_frac:.1%} ({n_mr}/{n_total_regimes})"
            )
            if mr_frac < 0.30:
                warnings.append(
                    f"Only {mr_frac:.1%} of pairs in MEAN_REVERTING regime — "
                    "market conditions may not be favourable for pairs trading"
                )

        n_crisis = regime_distribution.get("CRISIS", 0) + regime_distribution.get("BROKEN", 0)
        if n_crisis > 0:
            warnings.append(
                f"{n_crisis} pairs in CRISIS/BROKEN regime — do not route to portfolio layer"
            )

        if n_passed == 0:
            recommendations.append("No pairs passed validation — review universe selection and thresholds")
        elif n_passed < 5:
            recommendations.append("Fewer than 5 pairs passed — consider relaxing validation thresholds")
        else:
            recommendations.append(
                f"Route {n_passed} validated pairs to portfolio ranking layer"
            )

        if n_warned > 0:
            recommendations.append(
                f"Review {n_warned} pairs with WARN status — some may become viable with more data"
            )

        # Rejection breakdown
        rejection_breakdown: dict[str, int] = {}
        for r in validation_results:
            for reason in r.get("rejection_reasons", []):
                rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + 1
        if rejection_breakdown:
            top_reason = max(rejection_breakdown, key=lambda k: rejection_breakdown[k])
            key_findings.append(
                f"Most common rejection reason: '{top_reason}' ({rejection_breakdown[top_reason]} pairs)"
            )

        summary_id = _new_id()
        audit.log(
            f"Research summary generated: id={summary_id}, "
            f"{n_passed}/{n_validated} passed, pass_rate={pass_rate:.2%}"
        )

        return {
            "summary_id": summary_id,
            "generated_at": _utcnow(),
            "universe_size": universe_size,
            "candidates_generated": candidate_count,
            "pairs_validated": n_validated,
            "pairs_passed": n_passed,
            "pairs_warned": n_warned,
            "pairs_failed": n_failed,
            "pass_rate": round(pass_rate, 6),
            "regime_distribution": regime_distribution,
            "rejection_breakdown": rejection_breakdown,
            "key_findings": key_findings,
            "warnings": warnings,
            "recommendations": recommendations,
            "run_metadata": run_metadata,
        }
