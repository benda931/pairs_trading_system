# -*- coding: utf-8 -*-
"""
scripts/optuna_backtest_search.py — Optuna search על פרמטרי Backtest
====================================================================

מטרה:
------
1. להריץ Backtest על זוג/ות עם סט פרמטרים מוגדר היטב מתוך core.params.PARAM_SPECS.
2. להשתמש ב-Optuna + ParamSpec.suggest כדי לחקור טווחים **רחבים** אבל נשלטים.
3. לחווט נכון ל-root.backtest.run_backtest (רק kwargs חוקיים).
4. לשמור תוצאות ל-CSV ולתת לך best trial אמיתי (Sharpe / PnL / DD).

שימוש לדוגמה
-------------

# אופטימיזציה על זוג אחד (מצב core — רק הפרמטרים המרכזיים ל-backtest):
python -m scripts.optuna_backtest_search ^
  --pair XLP XLY ^
  --start 2018-01-01 ^
  --end 2025-12-05 ^
  --n-trials 50 ^
  --direction maximize ^
  --objective sharpe ^
  --param-mode core ^
  --study-name "XLP-XLY_sharpe_opt" ^
  --output "results/optuna_XLP_XLY_sharpe_core.csv"

# אופטימיזציה על זוג אחד (מצב tagged — בחירה אוטומטית לפי tags+importance):
python -m scripts.optuna_backtest_search ^
  --pair XLP XLY ^
  --start 2018-01-01 ^
  --end 2025-12-05 ^
  --n-trials 100 ^
  --direction maximize ^
  --objective sharpe ^
  --param-mode tagged ^
  --max-params 40 ^
  --study-name "XLP-XLY_sharpe_opt_tagged" ^
  --output "results/optuna_XLP_XLY_sharpe_tagged.csv"

# כמה זוגות ראשונים מה-universe (עדיין single-pair בפועל, אבל אפשר להרחיב ללופ):
python -m scripts.optuna_backtest_search ^
  --from-universe dq_pairs ^
  --limit-pairs 20 ^
  --start 2018-01-01 ^
  --end 2025-12-05 ^
  --n-trials 80 ^
  --direction maximize ^
  --objective sharpe ^
  --param-mode core ^
  --study-name "dq_pairs_sharpe_opt" ^
  --output "results/optuna_dq_pairs_core.csv"
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import is_dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import optuna
import pandas as pd

from core.params import (  # type: ignore[import]
    PARAM_SPECS,
    PARAM_INDEX,
    ParamSpec,
    filter_by_tags,
    get_param_importance_level,
)
from root import backtest as backtest_mod  # type: ignore[import]


logger = logging.getLogger("OptunaBacktest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ============================================================
#   1. סט "Core" של פרמטרים ל-backtest.run_backtest
# ============================================================

# אלה הפרמטרים שאנחנו *רוצים* ש־Optuna יחפש עליהם כרגע בבקטסט:
# (את הטווחים/step/log אתה מגדיר ב-core.params.ParamSpec)
BACKTEST_PARAM_NAMES_CORE: List[str] = [
    # Signal & windows
    "z_entry",
    "z_exit",
    "lookback",
    "atr_window",
    "entry_decay",
    "exit_decay",

    # Gates איכות / פילטרים לזוג
    "edge_min",
    "atr_max",
    "corr_min",
    "beta_lo",
    "beta_hi",
    "coint_pmax",
    "half_life_max",

    # Execution & costs
    "notional",
    "slippage_bps",
    "slippage_mode",
    "slippage_atr_frac",
    "transaction_cost_per_trade",
    "bar_lag",
    "max_bars_held",

    # Stops
    "z_stop",
    "run_dd_stop_pct",
]

# אילו kwargs מותר להעביר ל-run_backtest (חתימה שלך):
ALLOWED_KWARGS_FOR_RUN_BACKTEST = {
    "z_entry",
    "z_exit",
    "lookback",
    "atr_window",
    "entry_conditions",
    "exit_conditions",
    "edge_min",
    "atr_max",
    "corr_min",
    "coint_pmax",
    "half_life_max",
    "notional",
    "slippage_bps",
    "slippage_mode",
    "slippage_atr_frac",
    "transaction_cost_per_trade",
    "bar_lag",
    "max_bars_held",
    "z_stop",
    "run_dd_stop_pct",
    # אם תוסיף בעתיד עוד פרמטרים לחתימה של run_backtest –
    # תוסיף את שמם גם כאן.
}


# ============================================================
#   2. עזרים כלליים
# ============================================================

def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _sql_url_from_env_or_default() -> str:
    env_url = os.getenv("SQL_STORE_URL")
    if env_url:
        return env_url
    local = Path(os.getenv("LOCALAPPDATA", ".")) / "pairs_trading_system" / "cache.duckdb"
    return f"duckdb:///{local}"


def _load_pairs_from_universe(table: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    sql_url = _sql_url_from_env_or_default()
    logger.info("Loading pairs from DuckDB universe table '%s' (url=%s)", table, sql_url)
    con = duckdb.connect(sql_url.replace("duckdb:///", ""))
    try:
        df = con.execute(f"SELECT sym_x, sym_y FROM {table}").df()
    finally:
        con.close()
    if limit is not None:
        df = df.head(limit)
    pairs = list(df.itertuples(index=False, name=None))
    logger.info("Universe %s → %d pairs (used=%d)", table, len(df), len(pairs))
    return [(str(a), str(b)) for a, b in pairs]


def _extract_metrics(result: Any) -> Dict[str, Any]:
    if is_dataclass(result):
        data = asdict(result)
    elif isinstance(result, dict):
        data = dict(result)
    else:
        data = getattr(result, "__dict__", {}) or {}

    # אם יש dict פנימי של metrics – נפרוס אותו על data (בלי לדרוס)
    nested_candidates = []
    for key in ("metrics", "summary", "stats"):
        if key in data and isinstance(data[key], dict):
            nested_candidates.append(key)

    for key in nested_candidates:
        for k, v in data[key].items():
            data.setdefault(k, v)

    def _get(*keys: str, default: float = np.nan) -> float:
        for k in keys:
            if k in data and data[k] is not None:
                try:
                    return float(data[k])
                except Exception:
                    continue
        return float(default)

    def _get_int(*keys: str, default: int = 0) -> int:
        for k in keys:
            if k in data and data[k] is not None:
                try:
                    return int(data[k])
                except Exception:
                    continue
        return int(default)

    metrics: Dict[str, Any] = {
        "sharpe": _get("sharpe", "Sharpe"),
        "total_pnl": _get("total_pnl", "TotalPnL", "pnl", "Profit"),
        "max_drawdown": _get("max_drawdown", "MaxDD", "max_dd", "Drawdown"),
        "cagr": _get("cagr", "CAGR"),
        "win_rate": _get("win_rate", "WinRate"),
        "avg_trade_pnl": _get("avg_trade_pnl", "AvgTradePnL", "AvgPnL"),
        "n_trades": _get_int("n_trades", "NumTrades", "num_trades", "Trades"),
    }


    # לשמור גם את שאר המפתחות
    for k, v in data.items():
        if k not in metrics:
            metrics[k] = v

    return metrics



def _to_datetime(d: date | datetime) -> datetime:
    """ממיר date ל-datetime בחצות (או מחזיר כמו שהוא אם כבר datetime)."""
    if isinstance(d, datetime):
        return d
    return datetime(d.year, d.month, d.day)

def _build_run_kwargs_from_params(
    sym1: str,
    sym2: str,
    start: date,
    end: date,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    מתרגם dict של פרמטרים מ-Optuna ל-kwargs מסודרים עבור run_backtest.

    כאן אנחנו נותנים משמעות מקצועית לכל הפרמטרים:
    - 0 / ערכים שליליים ב-gates → כיבוי gate (None).
    - beta_lo/beta_hi → beta_range מסודר (lo <= hi).
    - טיפול ב-slippage_mode / slippage_atr_frac.
    """

    def _none_if_le(val: Any, threshold: float) -> Optional[float]:
        if val is None:
            return None
        v = float(val)
        return None if v <= threshold else v

    def _int_or_none(val: Any, min_val: int = 1) -> Optional[int]:
        if val is None:
            return None
        v = int(val)
        return max(min_val, v)

    # --- ליבת סיגנל / חלונות ---
    z_entry = float(params.get("z_entry", 2.0))
    z_exit = float(params.get("z_exit", 0.5))
    lookback = int(params.get("lookback", 60))
    atr_window = int(params.get("atr_window", 20))

    # --- Gates סטטיסטיים / איכות זוג ---
    edge_min = _none_if_le(params.get("edge_min"), 0.0)
    atr_max = _none_if_le(params.get("atr_max"), 0.0)

    corr_min = params.get("corr_min")
    if corr_min is not None:
        corr_min = float(corr_min)
        corr_min = max(-1.0, min(1.0, corr_min))

    coint_pmax = _none_if_le(params.get("coint_pmax"), 0.0)

    half_life_max = params.get("half_life_max")
    if half_life_max is not None:
        half_life_max = max(1, int(half_life_max))

    # beta_lo / beta_hi → beta_range
    beta_lo = params.get("beta_lo")
    beta_hi = params.get("beta_hi")
    beta_range: Optional[Tuple[float, float]] = None
    if beta_lo is not None and beta_hi is not None:
        lo = float(beta_lo)
        hi = float(beta_hi)
        if hi < lo:
            lo, hi = hi, lo
        beta_range = (lo, hi)

    # --- Execution & Costs ---
    notional = float(params.get("notional", 10_000.0))

    slippage_bps = float(params.get("slippage_bps", 1.0))
    slippage_mode = str(params.get("slippage_mode", "bps"))
    if slippage_mode not in ("bps", "atr_frac"):
        slippage_mode = "bps"

    slippage_atr_frac = params.get("slippage_atr_frac")
    if slippage_mode != "atr_frac":
        slippage_atr_frac = None
    elif slippage_atr_frac is not None:
        slippage_atr_frac = float(slippage_atr_frac)

    transaction_cost_per_trade = float(params.get("transaction_cost_per_trade", 1.0))

    bar_lag = int(params.get("bar_lag", 0))
    if bar_lag < 0:
        bar_lag = 0

    max_bars_held = _int_or_none(params.get("max_bars_held"), min_val=1)

    # --- Stops ---
    z_stop = params.get("z_stop")
    if z_stop is not None:
        z_stop = float(z_stop)

    run_dd_stop_pct = params.get("run_dd_stop_pct")
    if run_dd_stop_pct is not None:
        run_dd_stop_pct = float(run_dd_stop_pct)

    kwargs: Dict[str, Any] = dict(
        sym_x=sym1,
        sym_y=sym2,
        start_date=_to_datetime(start),
        end_date=_to_datetime(end),
        # ליבה
        z_entry=z_entry,
        z_exit=z_exit,
        lookback=lookback,
        atr_window=atr_window,
        # gates
        edge_min=edge_min,
        atr_max=atr_max,
        corr_min=corr_min,
        beta_range=beta_range,
        coint_pmax=coint_pmax,
        half_life_max=half_life_max,
        # execution
        notional=notional,
        slippage_bps=slippage_bps,
        slippage_mode=slippage_mode,
        slippage_atr_frac=slippage_atr_frac,
        transaction_cost_per_trade=transaction_cost_per_trade,
        bar_lag=bar_lag,
        max_bars_held=max_bars_held,
        # stops
        z_stop=z_stop,
        run_dd_stop_pct=run_dd_stop_pct,
        # תנאי כניסה/יציאה – כרגע ברירת מחדל פנימית
        entry_conditions=None,
        exit_conditions=None,
    )

    return kwargs

# ============================================================
#   3. בחירת ParamSpecs לחיפוש
# ============================================================

def _get_core_search_specs() -> List[ParamSpec]:
    """
    מחזיר את סט הפרמטרים "הידני" (BACKTEST_PARAM_NAMES_CORE)
    מתוך PARAM_INDEX. אם משהו חסר ב-PARAM_INDEX → לוג אזהרה.
    """
    specs: List[ParamSpec] = []
    missing: List[str] = []

    for name in BACKTEST_PARAM_NAMES_CORE:
        spec = PARAM_INDEX.get(name)
        if spec is None:
            missing.append(name)
        else:
            specs.append(spec)

    if missing:
        logger.warning(
            "Missing ParamSpec for core params: %s",
            ", ".join(missing),
        )

    logger.info(
        "Using %d CORE ParamSpecs for search: %s",
        len(specs),
        ", ".join(p.name for p in specs),
    )
    return specs


def _choose_tagged_search_specs(
    max_params: int = 30,
) -> List[ParamSpec]:
    """
    בוחר סט פרמטרים חשובים מתוך PARAM_SPECS לפי tags+importance:

    • include tags: signal, spread, mean_reversion, risk, stops, execution,
      volatility, regime, macro, hedge, correlation, portfolio.
    • importance_level ∈ {high, medium}
    • מקסימום max_params פרמטרים.
    """
    include_tags = [
        "signal",
        "spread",
        "mean_reversion",
        "risk",
        "stops",
        "execution",
        "volatility",
        "regime",
        "macro",
        "hedge",
        "correlation",
        "portfolio",
    ]
    candidates = filter_by_tags(PARAM_SPECS, include=include_tags)
    filtered: List[ParamSpec] = []
    for spec in candidates:
        level = get_param_importance_level(spec.name)
        if level in ("high", "medium"):
            filtered.append(spec)

    # high קודם, אחר כך medium, ואז לפי שם
    filtered.sort(
        key=lambda p: (
            0 if get_param_importance_level(p.name) == "high" else 1,
            p.name,
        )
    )

    if len(filtered) > max_params:
        filtered = filtered[:max_params]

    logger.info(
        "Using %d TAGGED ParamSpecs for search: %s",
        len(filtered),
        ", ".join(p.name for p in filtered),
    )
    return filtered


def _suggest_params_for_trial(trial: optuna.Trial, specs: Sequence[ParamSpec]) -> Dict[str, Any]:
    """
    משתמש ב-ParamSpec.suggest כדי להציע dict פרמטרים מלא ל-trial.
    """
    params: Dict[str, Any] = {}
    for spec in specs:
        val = spec.suggest(trial)
        params[spec.name] = val
    return params


# ============================================================
#   4. חיווט ל-run_backtest
# ============================================================

def _split_backtest_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    מסנן מתוך params רק kwargs חוקיים ל-run_backtest,
    ומטפל ב-beta_range שנבנה מ-beta_lo/beta_hi.
    """
    bt_kwargs: Dict[str, Any] = {}

    # חיווט מיוחד ל-beta_range=(beta_lo, beta_hi)
    beta_lo = params.get("beta_lo")
    beta_hi = params.get("beta_hi")
    if beta_lo is not None and beta_hi is not None:
        bt_kwargs["beta_range"] = (float(beta_lo), float(beta_hi))

    for name, value in params.items():
        if name in ALLOWED_KWARGS_FOR_RUN_BACKTEST:
            bt_kwargs[name] = value

    return bt_kwargs


def _run_single_backtest(
    sym1: str,
    sym2: str,
    start: date,
    end: date,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    עטיפה ל-backtest.run_backtest.

    כאן עושים את כל המיפוי מפרמטרים של Optuna ל-kwargs אמיתיים.
    """
    logger.debug(
        "run_backtest(%s-%s, %s→%s) with %s",
        sym1,
        sym2,
        start,
        end,
        params,
    )

    kwargs = _build_run_kwargs_from_params(sym1, sym2, start, end, params)

    try:
        result = backtest_mod.run_backtest(**kwargs)
    except TypeError as exc:
        logger.error("run_backtest TypeError: %s", exc)
        logger.error("kwargs used: %r", kwargs)
        raise

    metrics = _extract_metrics(result)
    return metrics



# ============================================================
#   5. Optuna objective
# ============================================================

class BacktestObjective:
    """
    אובייקט שניתן להעביר ל-Optuna כ-objective פונקציה.

    • ב-__call__ הוא:
      1. דוגם params לפי ParamSpec.suggest.
      2. מריץ _run_single_backtest.
      3. מחלץ metrics ומחזיר את המטריקה שבחרת (Sharpe / PnL / MaxDD).
      4. שומר את כל המידע ב-trial.user_attrs לצורך CSV / ניתוח.
    """

    def __init__(
        self,
        sym1: str,
        sym2: str,
        start: date,
        end: date,
        specs: Sequence[ParamSpec],
        objective: str = "sharpe",
    ) -> None:
        self.sym1 = sym1
        self.sym2 = sym2
        self.start = start
        self.end = end
        self.specs = list(specs)
        self.objective = objective

    def __call__(self, trial: optuna.Trial) -> float:
        params = _suggest_params_for_trial(trial, self.specs)
        metrics = _run_single_backtest(self.sym1, self.sym2, self.start, self.end, params)

        sharpe = float(metrics.get("sharpe", np.nan))
        max_dd = float(metrics.get("max_drawdown", np.nan))
        total_pnl = float(metrics.get("total_pnl", np.nan))

        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params_used", params)

        if self.objective == "sharpe":
            if not np.isfinite(sharpe):
                return -1e6
            return sharpe

        if self.objective == "pnl":
            if not np.isfinite(total_pnl):
                return -1e6
            return total_pnl

        if self.objective == "max_dd":
            if not np.isfinite(max_dd):
                return 1e6
            # Optuna מגדיל → כדי למזער DD נחזיר -DD
            return -max_dd

        # ברירת מחדל: Sharpe
        if not np.isfinite(sharpe):
            return -1e6
        return sharpe


# ============================================================
#   6. CLI & main
# ============================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Optuna backtest search using core.params.PARAM_SPECS",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pair",
        nargs=2,
        metavar=("SYM1", "SYM2"),
        help="Specific pair to optimize (e.g. XLP XLY).",
    )
    group.add_argument(
        "--from-universe",
        type=str,
        help="DuckDB table name to load pairs from (e.g. dq_pairs).",
    )

    parser.add_argument("--limit-pairs", type=int, default=1, help="Limit number of pairs from universe (currently only the first is used).")
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)

    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--direction",
        type=str,
        choices=("maximize", "minimize"),
        default="maximize",
        help="Optuna direction (for sharpe/pnl usually 'maximize').",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=("sharpe", "pnl", "max_dd"),
        default="sharpe",
        help="Which metric to use as objective.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="backtest_param_search",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///optuna.db). If not set, uses in-memory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path for all trials.",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=30,
        help="Maximum number of parameters (only used in param-mode=tagged).",
    )
    parser.add_argument(
        "--param-mode",
        type=str,
        choices=("core", "tagged"),
        default="core",
        help=(
            "core   = שימוש בסט ידני של פרמטרים שחווטו ל-run_backtest (BACKTEST_PARAM_NAMES_CORE).\n"
            "tagged = בחירה אוטומטית לפי tags+importance מתוך PARAM_SPECS (עד max-params)."
        ),
    )

    args = parser.parse_args(argv)

    # בחירת סט הפרמטרים לפי מצב
    if args.param_mode == "core":
        specs = _get_core_search_specs()
    else:
        specs = _choose_tagged_search_specs(max_params=args.max_params)

    if not specs:
        raise RuntimeError("No ParamSpecs were selected for search — check BACKTEST_PARAM_NAMES_CORE / tags / PARAM_SPECS.")

    # בחירת זוג/ות
    if args.pair:
        pairs = [(args.pair[0], args.pair[1])]
    else:
        pairs = _load_pairs_from_universe(args.from_universe, limit=args.limit_pairs)

    # כרגע: לוקחים את הזוג הראשון – אפשר בעתיד לולאה על pairs
    sym1, sym2 = pairs[0]
    logger.info("Running Optuna search for pair %s-%s (param_mode=%s)", sym1, sym2, args.param_mode)

    objective_fn = BacktestObjective(
        sym1=sym1,
        sym2=sym2,
        start=args.start,
        end=args.end,
        specs=specs,
        objective=args.objective,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        load_if_exists=True,
    )

    study.optimize(objective_fn, n_trials=args.n_trials)

    logger.info("Study finished: best_value=%s", study.best_value)
    logger.info("Best params:")
    for k, v in study.best_trial.params.items():
        logger.info("  %s = %s", k, v)

    # פלט CSV של כל ה-trials (עם המטריקות והפרמטרים המלאים)
    if args.output:
        rows: List[Dict[str, Any]] = []
        for t in study.trials:
            metrics = t.user_attrs.get("metrics", {}) or {}
            params_used = t.user_attrs.get("params_used", {}) or {}
            row: Dict[str, Any] = {
                "trial_id": t.number,
                "state": t.state.name,
                "value": t.value,
            }
            # params שנדגמו
            for k, v in params_used.items():
                row[f"param_{k}"] = v
            # metrics שחולצו מה-backtest
            for k, v in metrics.items():
                row[f"metric_{k}"] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Saved all trials to %s (rows=%d)", out_path, len(df))


if __name__ == "__main__":
    main()
