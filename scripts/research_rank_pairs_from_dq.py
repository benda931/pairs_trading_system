# -*- coding: utf-8 -*-
"""
scripts/research_rank_pairs_from_dq.py — Pair Research & Ranking (HF-grade v2)
==============================================================================

תפקידים:
--------
1. לקרוא את יוניברס הזוגות מטבלת dq_pairs ב-cache.duckdb.
2. למשוך היסטוריית מחירים לכל הסימבולים הרלוונטיים (מטבלת prices).
3. לחשב מטריקות זמן-סדרה לכל זוג (corr, p-value, half-life, Sharpe, DD, Vol, ...).
4. להרכיב DataFrame 'מחקר' ולקרוא ל-core.pair_ranking.rank_pairs_df.
5. לשמור קובץ pairs_universe_ranked.csv לשימוש בכל המערכת (zoom, optimizers, dashboard).

מאפיינים "קרן גידור":
----------------------
- עובד מול מקור אמת אחד (DuckDB cache עם prices + dq_pairs).
- ציון אחיד דרך core.pair_ranking (profile + קונפיג מה-config.json אם קיים).
- שמירת גם מטריקות גולמיות (אופציונלי) וגם יוניברס מדורג.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import duckdb
import numpy as np
import pandas as pd

# מוסיפים את שורש הפרויקט ל-sys.path כדי ש-core/ ו-common/ יזוהו כמודולים
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pair_ranking import PairScoreConfig, PairScoreProfile, rank_pairs_df
from common.config_manager import load_config

# ננסה להשתמש ב-statsmodels ל-ADF אם קיים, אחרת נשרוד בלי p_value אמיתי
try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # noqa: BLE001
    adfuller = None  # type: ignore[misc]

logger = logging.getLogger("PairResearch")


# ======================================================================
# Config dataclass
# ======================================================================


@dataclass
class ResearchConfig:
    duckdb_path: Path
    dq_table: str = "dq_pairs"

    # טבלת מחירים
    prices_table: str = "prices"
    col_ts: str = "ts"
    col_symbol: str = "symbol"
    col_close: str = "close"

    # טווח זמן אופציונלי
    start_date: Optional[str] = None  # למשל "2020-01-01"
    end_date: Optional[str] = None  # למשל "2025-12-31"

    # פרופיל ציון (כמו ב-PairScoreConfig)
    profile: PairScoreProfile = PairScoreProfile.RESEARCH

    # מינימום תצפיות בשביל למדוד משהו
    min_obs: int = 250

    # קובץ seed לזיהוי קטגוריה (sector / commodity / crypto / index / factor וכו')
    pairs_seed_csv: Path = Path("pairs_universe.csv")

    # קובץ יצוא סופי (מדורג)
    output_csv: Path = Path("pairs_universe_ranked.csv")

    # קובץ יצוא מטריקות גולמיות (אופציונלי)
    raw_metrics_csv: Optional[Path] = None


# ======================================================================
# Low-level helpers
# ======================================================================


def _default_duckdb_path() -> Path:
    """
    מחזיר את הנתיב הדיפולטי ל-cache.duckdb,
    במבנה שבו שאר המערכת כבר משתמשת.
    """
    env_path = os.getenv("PAIRS_TRADING_CACHE_DB")
    if env_path:
        return Path(env_path).expanduser().resolve()

    local_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return (local_dir / "pairs_trading_system" / "cache.duckdb").resolve()


def _connect_duckdb(path: Path) -> duckdb.DuckDBPyConnection:
    logger.info("[Research] Connecting to DuckDB: %s", path)
    return duckdb.connect(str(path))


def _load_dq_pairs(conn: duckdb.DuckDBPyConnection, cfg: ResearchConfig) -> pd.DataFrame:
    logger.info("[Research] Loading dq_pairs from table '%s'...", cfg.dq_table)
    df = conn.execute(
        f"SELECT sym_x, sym_y FROM {cfg.dq_table} ORDER BY sym_x, sym_y"
    ).fetchdf()

    if df.empty:
        raise RuntimeError("dq_pairs table is empty — nothing to research.")

    logger.info("[Research] Loaded %d pairs from dq_pairs.", len(df))
    return df


def _load_seed_categories(cfg: ResearchConfig, project_root: Path) -> pd.DataFrame:
    """
    טוען את pairs_universe.csv (אם קיים) כדי להוסיף seed_category / metadata
    (לדוגמה: sector / commodity / crypto / index / factor וכו').
    """
    csv_candidates = [
        cfg.pairs_seed_csv,
        project_root / cfg.pairs_seed_csv,
        project_root / "data" / cfg.pairs_seed_csv,
    ]

    for p in csv_candidates:
        if p.exists():
            logger.info("[Research] Loading seed CSV for categories: %s", p)
            df = pd.read_csv(p)
            if df.empty:
                logger.warning(
                    "[Research] Seed CSV '%s' is empty, skipping categories merge.", p
                )
                return pd.DataFrame()

            cols = {str(c).lower().strip(): c for c in df.columns}

            if "sym_x" in cols and "sym_y" in cols:
                sx = cols["sym_x"]
                sy = cols["sym_y"]
            elif "symbol_1" in cols and "symbol_2" in cols:
                sx = cols["symbol_1"]
                sy = cols["symbol_2"]
            else:
                logger.warning(
                    "[Research] Seed CSV '%s' does not contain sym_x/sym_y or symbol_1/symbol_2; "
                    "categories will not be merged.",
                    p,
                )
                return pd.DataFrame()

            # seed_category (אם קיים)
            if "seed_category" in cols:
                cat_col = cols["seed_category"]
                out = df[[sx, sy, cat_col]].rename(
                    columns={sx: "sym_x", sy: "sym_y", cat_col: "seed_category"}
                )
                return out

            out = df[[sx, sy]].rename(columns={sx: "sym_x", sy: "sym_y"})
            return out

    logger.info("[Research] No seed CSV found for categories; continuing without categories.")
    return pd.DataFrame()


def _load_prices_for_symbols(
    conn: duckdb.DuckDBPyConnection,
    cfg: ResearchConfig,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """
    טוען מחירי סגירה לטווח תאריכים עבור רשימת סמלים,
    ומחזיר DataFrame בפורמט צר (symbol, ts, dt, close).

    dt = תאריך יומי (ללא שעה) – עליו עושים pivot בהמשך.
    """
    if not symbols:
        raise ValueError("[Research] No symbols provided to _load_prices_for_symbols")

    sym_list = ",".join(f"'{s}'" for s in sorted(set(symbols)))

    where_clauses = [f"{cfg.col_symbol} IN ({sym_list})"]
    if cfg.start_date:
        where_clauses.append(
            f"{cfg.col_ts} >= TIMESTAMP '{cfg.start_date}'"
        )
    if cfg.end_date:
        where_clauses.append(
            f"{cfg.col_ts} <= TIMESTAMP '{cfg.end_date}'"
        )

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT
            {cfg.col_symbol} AS symbol,
            {cfg.col_ts} AS ts,
            {cfg.col_close} AS close
        FROM {cfg.prices_table}
        WHERE {where_sql}
        ORDER BY symbol, ts
    """

    logger.info(
        "[Research] Loading prices for %d symbols from table '%s' (date filter: %s → %s)...",
        len(set(symbols)),
        cfg.prices_table,
        cfg.start_date or "MIN",
        cfg.end_date or "MAX",
    )

    df = conn.execute(sql).fetchdf()

    if df.empty:
        raise RuntimeError(
            "[Research] No prices returned from DuckDB for requested symbols/date range"
        )

    expected_cols = {"symbol", "ts", "close"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"[Research] prices query missing columns {missing}, got {list(df.columns)!r}"
        )

    df["ts"] = pd.to_datetime(df["ts"])
    df["dt"] = df["ts"].dt.date
    df = df.dropna(subset=["dt", "close"])
    df["symbol"] = df["symbol"].astype(str)
    df = df.drop_duplicates(subset=["symbol", "dt"]).sort_values(["symbol", "dt"])

    return df


# ======================================================================
# Time-series metrics per pair
# ======================================================================


def _compute_max_drawdown(equity: pd.Series) -> float:
    """
    Max drawdown (ביחידות spread) על בסיס cumulative sum של spread_ret.
    """
    if equity.empty:
        return 0.0
    cum = equity.cumsum()
    running_max = cum.cummax()
    dd = running_max - cum
    max_dd = float(dd.max())
    return max_dd


def _compute_adf_pvalue(spread: pd.Series) -> float:
    """
    מחזיר p-value של ADF על ה-spread אם statsmodels זמין, אחרת NaN.
    """
    if adfuller is None:
        return float("nan")

    s = spread.dropna()
    if len(s) < 50:
        return float("nan")

    try:
        res = adfuller(s, maxlag=1, autolag="AIC")
        return float(res[1])
    except Exception:  # noqa: BLE001
        return float("nan")


def _compute_half_life(spread: pd.Series) -> float:
    """
    חישוב half-life קלאסי של OU ע"י רגרסיה:
        Δy_t = a + b * y_{t-1} + ε_t
        hl = -ln(2) / b  (אם b<0, אחרת inf)
    """
    s = spread.dropna()
    if len(s) < 50:
        return float("inf")

    y_lag = s.shift(1).dropna()
    dy = s.diff().dropna()

    y_lag = y_lag.loc[dy.index]
    if len(y_lag) != len(dy):
        return float("inf")

    x = y_lag.values
    y = dy.values

    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return float("inf")

    b = ((x - x_mean) * (y - y_mean)).sum() / denom
    if b >= 0:
        return float("inf")

    hl = -np.log(2.0) / b
    return float(max(hl, 0.0))


def _compute_pair_metrics_from_series(
    s_x: pd.Series,
    s_y: pd.Series,
    min_obs: int,
) -> Dict[str, float]:
    """
    מקבל שתי סדרות מחירים (close) אחרי חיתוך לתאריכים משותפים,
    ומחזיר dict של כל המטריקות הדרושות.
    """
    df = pd.concat(
        [s_x.rename("px_x"), s_y.rename("px_y")],
        axis=1,
    ).dropna()

    n_obs = int(len(df))
    if n_obs < min_obs:
        return {
            "corr": float("nan"),
            "p_value": float("nan"),
            "half_life": float("nan"),
            "spread_sharpe": float("nan"),
            "spread_sortino": float("nan"),
            "spread_vol": float("nan"),
            "spread_max_dd": float("nan"),
            "spread_pos_ratio": float("nan"),
            "spread_neg_ratio": float("nan"),
            "n_obs": float(n_obs),
            "beta": float("nan"),
            "intercept": float("nan"),
        }

    with np.errstate(divide="ignore", invalid="ignore"):
        log_x = np.log(df["px_x"])
        log_y = np.log(df["px_y"])

    valid_mask = ~(log_x.isna() | log_y.isna())
    lx = log_x[valid_mask].values
    ly = log_y[valid_mask].values

    if len(lx) < min_obs:
        return {
            "corr": float("nan"),
            "p_value": float("nan"),
            "half_life": float("nan"),
            "spread_sharpe": float("nan"),
            "spread_sortino": float("nan"),
            "spread_vol": float("nan"),
            "spread_max_dd": float("nan"),
            "spread_pos_ratio": float("nan"),
            "spread_neg_ratio": float("nan"),
            "n_obs": float(len(lx)),
            "beta": float("nan"),
            "intercept": float("nan"),
        }

    x_mean = lx.mean()
    y_mean = ly.mean()
    denom = ((ly - y_mean) ** 2).sum()

    if denom == 0:
        beta = 1.0
        intercept = x_mean - beta * y_mean
    else:
        beta = ((ly - y_mean) * (lx - x_mean)).sum() / denom
        intercept = x_mean - beta * y_mean

    spread = lx - (beta * ly + intercept)
    spread = pd.Series(spread, index=log_x[valid_mask].index).dropna()

    ret_x = np.diff(lx)
    ret_y = np.diff(ly)
    if len(ret_x) > 1 and np.isfinite(ret_x).all() and np.isfinite(ret_y).all():
        corr = float(np.corrcoef(ret_x, ret_y)[0, 1])
    else:
        corr = float("nan")

    p_value = _compute_adf_pvalue(spread)
    half_life = _compute_half_life(spread)

    spread_ret = spread.diff().dropna()
    if spread_ret.empty or spread_ret.std() == 0:
        spread_sharpe = float("nan")
        spread_sortino = float("nan")
        spread_vol = float("nan")
        spread_max_dd = float("nan")
        spread_pos_ratio = float("nan")
        spread_neg_ratio = float("nan")
    else:
        mu = float(spread_ret.mean())
        sigma = float(spread_ret.std())
        ann_factor = sqrt(252.0)
        spread_sharpe = (mu / sigma) * ann_factor

        downside = spread_ret[spread_ret < 0.0]
        if len(downside) > 0 and downside.std() > 0:
            downside_std = float(downside.std())
            spread_sortino = (mu / downside_std) * ann_factor
        else:
            spread_sortino = float("nan")

        spread_vol = sigma * ann_factor
        spread_max_dd = _compute_max_drawdown(spread_ret)

        spread_pos_ratio = float((spread_ret > 0.0).mean())
        spread_neg_ratio = float((spread_ret < 0.0).mean())

    metrics = {
        "corr": float(corr),
        "p_value": float(p_value),
        "half_life": float(half_life),
        "spread_sharpe": float(spread_sharpe),
        "spread_sortino": float(spread_sortino),
        "spread_vol": float(spread_vol),
        "spread_max_dd": float(spread_max_dd),
        "spread_pos_ratio": float(spread_pos_ratio),
        "spread_neg_ratio": float(spread_neg_ratio),
        "n_obs": float(n_obs),
        "beta": float(beta),
        "intercept": float(intercept),
    }
    return metrics


# ======================================================================
# Pair-scoring config integration
# ======================================================================


def _load_global_settings() -> Dict[str, Any]:
    """
    מנסה לטעון config.json מהשורש ולהוציא ממנו section רלוונטי.

    אם אין קובץ / יש שגיאה → מחזיר dict ריק.
    """
    cfg_path = PROJECT_ROOT / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load config.json for pair scoring: %r", exc)
        return {}


def _build_pair_score_config(research_cfg: ResearchConfig) -> PairScoreConfig:
    """
    בונה PairScoreConfig מתוך config.json (אם יש section pair_scoring),
    ומעדכן אותו לפי ה-ResearchConfig (profile, min_obs).
    """
    settings = _load_global_settings()
    pair_section = settings.get("pair_scoring") or settings.get("pair_score") or {}

    # אם ל-PairScoreConfig יש from_dict (בגרסאות החדשות) – נשתמש בו.
    if hasattr(PairScoreConfig, "from_dict") and isinstance(pair_section, dict):
        try:
            score_cfg = PairScoreConfig.from_dict(pair_section)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("PairScoreConfig.from_dict failed (%r), falling back to defaults.", exc)
            score_cfg = PairScoreConfig(profile=research_cfg.profile)
    else:
        score_cfg = PairScoreConfig(profile=research_cfg.profile)

    # override לפי המחקר הנוכחי
    score_cfg.profile = research_cfg.profile
    # נשמור אחריות על מינימום תצפיות בציון (viability)
    if hasattr(score_cfg, "min_n_obs"):
        score_cfg.min_n_obs = research_cfg.min_obs  # type: ignore[assignment]
    if hasattr(score_cfg, "soft_min_n_obs"):
        # לא נוריד soft_min מתחת למינימום הקשה
        current_soft = getattr(score_cfg, "soft_min_n_obs")
        score_cfg.soft_min_n_obs = max(current_soft, research_cfg.min_obs)  # type: ignore[assignment]

    return score_cfg


# ======================================================================
# Main research pipeline
# ======================================================================


def run_research(cfg: ResearchConfig) -> pd.DataFrame:
    """
    צינור המחקר המלא:
    1. טעינת dq_pairs.
    2. טעינת seed_category (אם יש).
    3. טעינת מחירים לכל הסימבולים.
    4. חישוב מטריקות לכל זוג.
    5. קריאה ל-core.pair_ranking.rank_pairs_df.
    6. שמירה ל-CSV והחזרה.
    """
    project_root = PROJECT_ROOT

    conn = _connect_duckdb(cfg.duckdb_path)
    try:
        pairs_df = _load_dq_pairs(conn, cfg)

        seed_df = _load_seed_categories(cfg, project_root)
        if not seed_df.empty:
            pairs_df = pairs_df.merge(
                seed_df,
                on=["sym_x", "sym_y"],
                how="left",
            )

        symbols = sorted(
            set(pairs_df["sym_x"].astype(str)) | set(pairs_df["sym_y"].astype(str))
        )

        prices_df = _load_prices_for_symbols(conn, cfg, symbols)

        wide = prices_df.pivot(index="dt", columns="symbol", values="close").sort_index()

        out_rows: List[Dict[str, Any]] = []

        logger.info("[Research] Computing metrics for %d pairs...", len(pairs_df))

        for _, row in pairs_df.iterrows():
            sym_x = str(row["sym_x"])
            sym_y = str(row["sym_y"])

            if sym_x not in wide.columns or sym_y not in wide.columns:
                logger.warning(
                    "[Research] Missing prices for pair (%s, %s); skipping metrics.",
                    sym_x,
                    sym_y,
                )
                metrics = {
                    "corr": float("nan"),
                    "p_value": float("nan"),
                    "half_life": float("nan"),
                    "spread_sharpe": float("nan"),
                    "spread_sortino": float("nan"),
                    "spread_vol": float("nan"),
                    "spread_max_dd": float("nan"),
                    "spread_pos_ratio": float("nan"),
                    "spread_neg_ratio": float("nan"),
                    "n_obs": float(0),
                    "beta": float("nan"),
                    "intercept": float("nan"),
                }
            else:
                s_x = wide[sym_x]
                s_y = wide[sym_y]
                metrics = _compute_pair_metrics_from_series(
                    s_x,
                    s_y,
                    min_obs=cfg.min_obs,
                )

            rec: Dict[str, Any] = {
                "sym_x": sym_x,
                "sym_y": sym_y,
            }
            if "seed_category" in pairs_df.columns:
                rec["seed_category"] = str(row.get("seed_category") or "")

            rec.update(metrics)
            out_rows.append(rec)

        base_df = pd.DataFrame(out_rows)
        logger.info(
            "[Research] Finished computing raw metrics for %d pairs.", len(base_df)
        )

        # לשימוש מחקרי אפשר לשמור גם את המטריקות הגולמיות (לפני ציון)
        if cfg.raw_metrics_csv:
            raw_path = cfg.raw_metrics_csv
            if not raw_path.is_absolute():
                raw_path = (project_root / raw_path).resolve()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            base_df.to_csv(raw_path, index=False)
            logger.info("[Research] Saved raw metrics to: %s", raw_path)

        # ציון ודירוג — לא זורקים זוגות במחקר, רק מסמנים is_viable
        logger.info("[Research] Ranking pairs via core.pair_ranking (no hard drop)...")
        score_cfg = PairScoreConfig(profile=cfg.profile)

        ranked_df = rank_pairs_df(
            base_df,
            cfg=score_cfg,
            top=None,
            enforce_viability=False,  # <<< השורה הקריטית
        )

        n_total = len(ranked_df)
        n_viable = int(ranked_df["is_viable"].sum()) if "is_viable" in ranked_df.columns else n_total
        logger.info(
            "[Research] Ranking complete. total_pairs=%d, viable_pairs=%d",
            n_total,
            n_viable,
        )


        # נשמור יוניברס מדורג ל-CSV
        out_path = cfg.output_csv
        if not out_path.is_absolute():
            out_path = (project_root / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ranked_df.to_csv(out_path, index=False)
        logger.info("[Research] Saved ranked universe to: %s", out_path)

        if not ranked_df.empty:
            preview_cols = [
                "sym_x",
                "sym_y",
                "pair_score",
                "pair_label",
                "corr",
                "half_life",
                "spread_sharpe",
                "spread_sortino",
                "spread_vol",
                "spread_max_dd",
                "n_obs",
                "seed_category",
            ]
            existing_cols = [c for c in preview_cols if c in ranked_df.columns]
            preview = ranked_df[existing_cols].head(10)
            logger.info("Top 10 ranked pairs:\n%s", preview.to_string(index=False))

        return ranked_df

    finally:
        conn.close()
        logger.info("[Research] Closed DuckDB connection.")


# ======================================================================
# CLI
# ======================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Research & rank pairs from dq_pairs and prices (HF-grade)."
    )
    parser.add_argument(
        "--duckdb-path",
        type=str,
        default=None,
        help=(
            "Path to DuckDB cache (default: LOCALAPPDATA\\pairs_trading_system\\cache.duckdb "
            "or PAIRS_TRADING_CACHE_DB)."
        ),
    )
    parser.add_argument(
        "--dq-table",
        type=str,
        default="dq_pairs",
        help="Table name for pairs universe (default: dq_pairs).",
    )
    parser.add_argument(
        "--prices-table",
        type=str,
        default="prices",
        help="Table name for prices (default: prices).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date filter, e.g. 2020-01-01.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date filter, e.g. 2025-12-31.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=[p.value for p in PairScoreProfile],
        default=PairScoreProfile.RESEARCH.value,
        help="Scoring profile: research / live / conservative (default: research).",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=250,
        help="Minimum observations per pair to compute metrics (default: 250).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="pairs_universe_ranked.csv",
        help="Output CSV path for ranked universe (default: pairs_universe_ranked.csv).",
    )
    parser.add_argument(
        "--raw-metrics-csv",
        type=str,
        default="",
        help="Optional CSV path for raw pair metrics before scoring.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if args.duckdb_path:
        duckdb_path = Path(args.duckdb_path).expanduser().resolve()
    else:
        duckdb_path = _default_duckdb_path()

    raw_metrics_path: Optional[Path]
    if args.raw_metrics_csv:
        raw_metrics_path = Path(args.raw_metrics_csv)
    else:
        raw_metrics_path = None

    cfg = ResearchConfig(
        duckdb_path=duckdb_path,
        dq_table=args.dq_table,
        prices_table=args.prices_table,
        start_date=args.start_date,
        end_date=args.end_date,
        profile=PairScoreProfile(args.profile),
        min_obs=args.min_obs,
        output_csv=Path(args.output_csv),
        raw_metrics_csv=raw_metrics_path,
    )

    logger.info("===== Pair Research & Ranking started =====")
    logger.info("DuckDB: %s", cfg.duckdb_path)
    logger.info(
        "dq_table: %s | prices_table: %s | date range: %s → %s",
        cfg.dq_table,
        cfg.prices_table,
        cfg.start_date or "MIN",
        cfg.end_date or "MAX",
    )
    logger.info("profile: %s | min_obs: %d", cfg.profile.value, cfg.min_obs)

    try:
        run_research(cfg)
        logger.info("===== Pair Research & Ranking finished successfully =====")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pair research failed: %r", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
