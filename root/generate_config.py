# -*- coding: utf-8 -*-
"""
generate_config.py — יצירת config.json מקצועי ל־Pairs Trading System
=====================================================================

מטרות
------
- לבנות יקום סמלים מגוון לפי תחומים (טכנולוגיה, פיננסים, אנרגיה, ETF, קריפטו, מדדים).
- ליצור universe של זוגות למסחר זוגי (pairs) בצורה מבוקרת:
  * עדיפות לזוגות "הגיוניים" (intra-sector) אך גם ערבוב בין סקטורים (inter-sector).
  * מגבלה על מספר הפעמים שכל סימבול יכול להופיע (max_pairs_per_symbol).
- להצמיד לכל זוג:
  * score בסיסי (לדרוג ראשוני לפי היגיון סקטוריאלי).
  * metadata על הסקטורים של שני הסימבולים + סוג זוג (intra/inter).
- לבנות config מלא:
  * strategy / filters / volatility_profiles / optimization / ib.
  * pairs כולל metadata.
  * universe + summary סטטיסטי של התפלגות הזוגות.

הקובץ תומך בהרצה כ־CLI:

    python generate_config.py --n-pairs 250 --seed 42 --risk-profile balanced \
        --max-pairs-per-symbol 15 --intra-ratio 0.6 --output config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("generate_config")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ============================================================================
# Symbol Universe Definition
# ============================================================================

@dataclass(frozen=True)
class SymbolGroup:
    name: str
    symbols: List[str]


SYMBOL_GROUPS: List[SymbolGroup] = [
    SymbolGroup(
        name="tech",
        symbols=["AAPL", "MSFT", "GOOG", "META", "NVDA", "AMZN", "TSLA", "CRM", "ADBE", "ORCL"],
    ),
    SymbolGroup(
        name="finance",
        symbols=["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "TD", "RY"],
    ),
    SymbolGroup(
        name="energy",
        symbols=["XOM", "CVX", "COP", "EOG", "PSX", "MPC", "VLO", "OXY", "KMI", "HAL"],
    ),
    SymbolGroup(
        name="etf",
        symbols=["XLY", "XLP", "XLC", "XLI", "XLB", "XLK", "XLV", "XBI", "XLE", "XLU"],
    ),
    SymbolGroup(
        name="crypto_proxy",
        symbols=["IBIT", "MSTR", "RIOT", "MARA", "COIN", "BTC", "ETH", "GBTC", "BITO", "SQ"],
    ),
    SymbolGroup(
        name="indices",
        symbols=["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "IVV", "VT", "XLF", "XLK"],
    ),
]


# מיפוי סימבול -> סקטור
SYMBOL_TO_SECTOR: Dict[str, str] = {}
for group in SYMBOL_GROUPS:
    for sym in group.symbols:
        SYMBOL_TO_SECTOR[sym] = group.name


def build_universe() -> List[str]:
    """
    Build the full symbol universe (unique symbols) based on SYMBOL_GROUPS.
    """
    universe = sorted(set(sym for g in SYMBOL_GROUPS for sym in g.symbols))
    logger.info("Universe built with %d unique symbols.", len(universe))
    return universe


# ============================================================================
# Pair Metadata & Scoring
# ============================================================================

@dataclass
class PairMetadata:
    symbols: Tuple[str, str]
    sector_x: str
    sector_y: str
    sector_pair_type: str  # "intra" / "inter"
    score: float


def _sector_pair_type(sym_x: str, sym_y: str) -> str:
    """Return 'intra' if both in same sector, else 'inter'."""
    sx = SYMBOL_TO_SECTOR.get(sym_x, "unknown")
    sy = SYMBOL_TO_SECTOR.get(sym_y, "unknown")
    return "intra" if sx == sy and sx != "unknown" else "inter"


def _base_score(sym_x: str, sym_y: str) -> float:
    """
    Base score heuristic:

    - intra-sector pairs: בסיס גבוה יותר.
    - ETF / indices pairs: בונוס נוסף (מאד "מסחר זוגי" קלאסי).
    - crypto proxies: מעט קנס בגלל תנודתיות גבוהה / סיכון צד.
    """
    sector_x = SYMBOL_TO_SECTOR.get(sym_x, "unknown")
    sector_y = SYMBOL_TO_SECTOR.get(sym_y, "unknown")

    same_sector = sector_x == sector_y and sector_x != "unknown"
    etf_like = sector_x == "etf" and sector_y == "etf"
    index_like = sector_x == "indices" and sector_y == "indices"
    crypto_involved = (
        "crypto" in sector_x
        or "crypto" in sector_y
        or "crypto_proxy" in sector_x
        or "crypto_proxy" in sector_y
    )

    score = 0.5
    if same_sector:
        score += 0.25
    if etf_like or index_like:
        score += 0.10
    if crypto_involved:
        score -= 0.05

    # clamp לצורך סניטציה
    return max(0.3, min(0.95, score))


def generate_pairs(
    universe: List[str],
    n_pairs: int = 250,
    rng: random.Random | None = None,
    *,
    max_pairs_per_symbol: int = 15,
    intra_ratio: float = 0.6,
) -> List[PairMetadata]:
    """
    Generate a list of candidate pairs with heuristic scoring + מגבלות per-symbol.

    האלגוריתם:

    1. מייצרים את כל הקומבינציות האפשריות (unordered) ב־universe.
    2. מסווגים אותן ל-intra / inter לפי סקטור.
    3. מחשבים base_score לכל זוג.
    4. ממיינים בתוך כל קבוצה לפי score יורד.
    5. Greedy:
        - בוחרים קודם intra עד n_intra_target (n_pairs * intra_ratio),
          תוך שמירה על max_pairs_per_symbol.
        - אח"כ ממלאים inter עד n_pairs.
    6. מחזירים רשימת PairMetadata ממוינת לפי score.

    Parameters
    ----------
    universe : list[str]
        Full list of unique symbols.
    n_pairs : int, default 250
        Number of pairs to select.
    rng : random.Random or None
        Optional random generator for reproducible results.
    max_pairs_per_symbol : int, default 15
        מגבלה על מספר הזוגות שכל סימבול יכול להופיע בהם.
    intra_ratio : float, default 0.6
        יחס היעד בין intra-sector pairs (0..1).

    Returns
    -------
    list[PairMetadata]
    """
    if rng is None:
        rng = random.Random()

    all_combos = list(combinations(universe, 2))
    rng.shuffle(all_combos)

    intra_candidates: List[Tuple[str, str]] = []
    inter_candidates: List[Tuple[str, str]] = []

    for sym_x, sym_y in all_combos:
        if _sector_pair_type(sym_x, sym_y) == "intra":
            intra_candidates.append((sym_x, sym_y))
        else:
            inter_candidates.append((sym_x, sym_y))

    # חישוב score עבור כל מועמד
    def _score_and_wrap(pairs: List[Tuple[str, str]]) -> List[Tuple[float, Tuple[str, str]]]:
        out: List[Tuple[float, Tuple[str, str]]] = []
        for sym_x, sym_y in pairs:
            base = _base_score(sym_x, sym_y)
            # מוסיפים "רעש" קטן כדי לשבור שוויון אך לשמור על סדר
            jitter = rng.uniform(-0.01, 0.01)
            out.append((base + jitter, (sym_x, sym_y)))
        # מיון לפי score יורד
        out.sort(key=lambda t: t[0], reverse=True)
        return out

    intra_scored = _score_and_wrap(intra_candidates)
    inter_scored = _score_and_wrap(inter_candidates)

    symbol_usage: Dict[str, int] = {sym: 0 for sym in universe}
    selected_pairs: List[Tuple[str, str]] = []

    n_intra_target = int(n_pairs * intra_ratio)

    def _try_add(candidates: List[Tuple[float, Tuple[str, str]]], target: int):
        for score, (sym_x, sym_y) in candidates:
            if len(selected_pairs) >= target:
                break
            if symbol_usage[sym_x] >= max_pairs_per_symbol:
                continue
            if symbol_usage[sym_y] >= max_pairs_per_symbol:
                continue
            selected_pairs.append((sym_x, sym_y))
            symbol_usage[sym_x] += 1
            symbol_usage[sym_y] += 1

    # שלב 1: intra עד n_intra_target
    _try_add(intra_scored, n_intra_target)

    # שלב 2: מילוי inter עד n_pairs
    if len(selected_pairs) < n_pairs:
        _try_add(inter_scored, n_pairs)

    if len(selected_pairs) < n_pairs:
        logger.warning(
            "Not enough pairs to reach requested n_pairs=%d (got %d) under max_pairs_per_symbol=%d.",
            n_pairs,
            len(selected_pairs),
            max_pairs_per_symbol,
        )

    # בניית PairMetadata
    pairs_metadata: List[PairMetadata] = []
    for sym_x, sym_y in selected_pairs:
        sx = SYMBOL_TO_SECTOR.get(sym_x, "unknown")
        sy = SYMBOL_TO_SECTOR.get(sym_y, "unknown")
        sector_type = _sector_pair_type(sym_x, sym_y)
        base = _base_score(sym_x, sym_y)
        pairs_metadata.append(
            PairMetadata(
                symbols=(sym_x, sym_y),
                sector_x=sx,
                sector_y=sy,
                sector_pair_type=sector_type,
                score=base,
            )
        )

    # מיון סופי לפי score יורד
    pairs_metadata.sort(key=lambda p: p.score, reverse=True)

    intra_count = sum(1 for p in pairs_metadata if p.sector_pair_type == "intra")
    inter_count = sum(1 for p in pairs_metadata if p.sector_pair_type == "inter")
    logger.info(
        "Generated %d pairs (intra=%d, inter=%d, intra_ratio=%.2f).",
        len(pairs_metadata),
        intra_count,
        inter_count,
        intra_ratio,
    )
    return pairs_metadata


# ============================================================================
# Risk Profiles Templates
# ============================================================================

RISK_PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "conservative": {
        "strategy": {
            "z_open": 2.5,
            "z_close": 0.7,
            "max_exposure_per_trade": 0.05,
            "rolling_window": 90,
        },
        "filters": {
            "min_correlation": 0.8,
            "max_drawdown_threshold": 0.10,
        },
    },
    "balanced": {
        "strategy": {
            "z_open": 2.0,
            "z_close": 0.5,
            "max_exposure_per_trade": 0.10,
            "rolling_window": 60,
        },
        "filters": {
            "min_correlation": 0.7,
            "max_drawdown_threshold": 0.15,
        },
    },
    "aggressive": {
        "strategy": {
            "z_open": 1.5,
            "z_close": 0.3,
            "max_exposure_per_trade": 0.15,
            "rolling_window": 40,
        },
        "filters": {
            "min_correlation": 0.6,
            "max_drawdown_threshold": 0.20,
        },
    },
}


def _apply_risk_profile(base: Dict[str, Any], profile: str) -> Dict[str, Any]:
    """
    Merge overrides from RISK_PROFILES[profile] מעל base config dict קטן.

    base הוא dict עם keys חלקיים (strategy, filters). overrides מכיל רק שדות
    שרוצים לשנות לפי רמת סיכון.
    """
    profile_lower = profile.lower()
    overrides = RISK_PROFILES.get(profile_lower)
    if overrides is None:
        logger.warning("Unknown risk_profile=%r, using 'balanced' defaults.", profile)
        overrides = RISK_PROFILES["balanced"]

    merged = base.copy()
    for section, section_over in overrides.items():
        merged.setdefault(section, {})
        merged[section].update(section_over)
    return merged


# ============================================================================
# Config Builder
# ============================================================================

def build_config_dict(
    pairs_meta: List[PairMetadata],
    *,
    ib_host: str = "127.0.0.1",
    ib_port: int = 6083,
    ib_client_id: int = 1,
    version: str = "3.4",
    author: str = "Omri",
    risk_profile: str = "balanced",
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Build the full config dict to be serialized as JSON.

    Parameters
    ----------
    pairs_meta : list[PairMetadata]
    ib_host, ib_port, ib_client_id : IBKR connection params
    version : str
    author : str
    risk_profile : {"conservative","balanced","aggressive"}
    seed : int or None
        Seed שנשתמש בו במטאדאטה לצורך reproducibility.

    Returns
    -------
    dict
    """
    pairs_json = [
        {
            "symbols": [pm.symbols[0], pm.symbols[1]],
            "score": round(pm.score, 3),
            "sector_x": pm.sector_x,
            "sector_y": pm.sector_y,
            "sector_pair_type": pm.sector_pair_type,
        }
        for pm in pairs_meta
    ]

    # בסיס ל-strategy / filters – לפני פרופיל סיכון
    base_core: Dict[str, Any] = {
        "strategy": {
            "z_open": 2.0,
            "z_close": 0.5,
            "max_exposure_per_trade": 0.10,
            "rolling_window": 60,
            "atr_window": 14,
            "use_volatility_adjustment": True,
            "auto_rebalance": True,
            "dry_run": False,
            "log_trades": True,
        },
        "filters": {
            "min_correlation": 0.7,
            "min_edge": 1.2,
            "min_half_life": 1,
            "max_half_life": 200,
            "min_volatility_zscore": -1.5,
            "max_volatility_zscore": 1.5,
        },
    }

    core = _apply_risk_profile(base_core, risk_profile)

    # סיכומים
    intra_count = sum(1 for p in pairs_meta if p.sector_pair_type == "intra")
    inter_count = sum(1 for p in pairs_meta if p.sector_pair_type == "inter")

    symbol_usage: Dict[str, int] = {}
    for pm in pairs_meta:
        for sym in pm.symbols:
            symbol_usage[sym] = symbol_usage.get(sym, 0) + 1

    config: Dict[str, Any] = {
        "strategy": core["strategy"],
        "filters": core["filters"],
        "volatility_profiles": {
            "default": {"min_z": -1.5, "max_z": 1.5},
            "tech": {"min_z": -2.0, "max_z": 2.0},
            "etf": {"min_z": -1.0, "max_z": 1.0},
            "crypto": {"min_z": -2.5, "max_z": 2.5},
        },
        "optimization": {
            "enabled": True,
            "min_quality_score": 0.5,
            "max_drawdown_threshold": core["filters"].get("max_drawdown_threshold", 0.15),
            "std_threshold": 0.01,
            "median_diff_threshold": 0.02,
            "z_of_atr": 1.0,
            "price_ratio_dev": 0.03,
            "volatility_spike": 0.2,
            "percentile_rank_spread": 85,
            "rolling_corr_drop": 0.1,
            "price_ratio_z": 1.5,
            "log_price_ratio_z": 1.2,
            "standardized_residual": 1.0,
            "kalman_residual": 1.1,
            "cointegration_deviation": 0.05,
        },
        "ib": {
            "host": ib_host,
            "port": ib_port,
            "client_id": ib_client_id,
        },
        "pairs": pairs_json,
        "universe": {
            "groups": [asdict(g) for g in SYMBOL_GROUPS],
            "symbol_to_sector": SYMBOL_TO_SECTOR,
        },
        "metadata": {
            "version": version,
            "created": "2025-06-12",  # אפשר לעדכן לדינאמי לפי datetime.now
            "author": author,
            "note": "Auto-generated pairs universe with sector metadata",
            "risk_profile": risk_profile,
            "seed": seed,
        },
        "summary": {
            "n_pairs": len(pairs_meta),
            "n_symbols": len(SYMBOL_TO_SECTOR),
            "intra_pairs": intra_count,
            "inter_pairs": inter_count,
            "symbol_usage": symbol_usage,
        },
    }

    return config


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Serialize config dict to JSON file (UTF-8, pretty-printed).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info("✅ נוצר %s עם %d זוגות מדורגים.", output_path, len(config.get("pairs", [])))


# ============================================================================
# CLI / main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate config.json with ranked pairs universe."
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=250,
        help="מספר זוגות ליצירה (default: 250)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed ל-random לצורך שחזור (default: None = רנדומי כל פעם)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config.json",
        help="שם/נתיב קובץ ה-output (default: config.json בתיקייה הנוכחית)",
    )
    parser.add_argument(
        "--ib-host",
        type=str,
        default="127.0.0.1",
        help="IBKR host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--ib-port",
        type=int,
        default=6083,
        help="IBKR port (default: 6083 – ליישר מול ibkr_connection במערכת שלך)",
    )
    parser.add_argument(
        "--ib-client-id",
        type=int,
        default=1,
        help="IBKR client_id (default: 1)",
    )
    parser.add_argument(
        "--max-pairs-per-symbol",
        type=int,
        default=15,
        help="מקסימום זוגות שסימבול אחד יכול להשתתף בהם (default: 15)",
    )
    parser.add_argument(
        "--intra-ratio",
        type=float,
        default=0.6,
        help="יחס יעד של pairs intra-sector (0..1), default=0.6",
    )
    parser.add_argument(
        "--risk-profile",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
        help="פרופיל סיכון ל-strategy/filters (default: balanced)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    universe = build_universe()
    pairs_meta = generate_pairs(
        universe,
        n_pairs=args.n_pairs,
        rng=rng,
        max_pairs_per_symbol=args.max_pairs_per_symbol,
        intra_ratio=args.intra_ratio,
    )

    config = build_config_dict(
        pairs_meta,
        ib_host=args.ib_host,
        ib_port=args.ib_port,
        ib_client_id=args.ib_client_id,
        risk_profile=args.risk_profile,
        seed=args.seed,
    )

    output_path = Path(args.output).resolve()
    save_config(config, output_path)
    print(f"✅ נוצר {output_path.name} עם {len(config['pairs'])} זוגות מדורגים.")


if __name__ == "__main__":
    main()
