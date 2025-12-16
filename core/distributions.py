# -*- coding: utf-8 -*-
"""
core/distributions.py — Central Optuna distribution factory (HF-grade, v3)
=========================================================================

תפקיד המודול
-------------
שכבה אחידה, חכמה ועשירה לניהול Optuna distributions מעל ParamRangeModel:

- מקור האמת לפרמטרים: `core.ranges.load_ranges_config().default_ranges`
  (Dict[str, ParamRangeModel]).
- בניית אובייקטי Distribution לכל פרמטר:
    * UniformDistribution / IntUniformDistribution / LogUniformDistribution /
      CategoricalDistribution.
- אינטרוספקציה עמוקה:
    * סוג פרמטר (continuous/log/int/categorical).
    * קבוצות / tags (למשל "risk", "signal", "macro").
    * "Search-space complexity score" להערכת עומק החיפוש.
- המלצות:
    * בחירת sampler (TPE / CMA-ES) לפי מאפייני מרחב הפרמטרים.
    * קירוב ל-n_trials מומלץ.

תכונות מרכזיות
---------------
1. Cache חכם:
   - `get_param_distributions()` עטוף ב־lru_cache כדי למנוע טעינה חוזרת מהקונפיג.

2. Discovery & Introspection:
   - `list_parameters()` — רשימת שמות.
   - `get_distribution(name)` — דיסטריביושן בודד.
   - `describe_distributions()` — תיאור מפורט (type + kwargs + meta + kind).
   - `group_parameters_by_type()` — חלוקה לפי continuous/int/log/categorical.
   - `get_parameters_by_group()` / `get_parameters_by_tag()`.

3. Export & Validation:
   - `export_distributions_to_json()` — ייצוא מלא ל־JSON.
   - `validate_distributions()` — כל range קיבל דיסטריביושן.
   - `validate_int_steps()` — ולידציית step.
   - `validate_log_ranges()` — ולידציית log (low>0 וכו').

4. Search-space analytics:
   - `summarize_search_space()`:
       * מספר פרמטרים.
       * פירוק לפי types.
       * "volume" מקורב של space.
       * מדד מורכבות.

5. המלצות אופטימיזציה:
   - `recommend_sampler()` — TPE מול CMA-ES לפי סוגי פרמטרים.
   - `recommend_n_trials()` — קירוב מבוסס dimension וקטגוריות.

6. CLI עשיר:
   - `python -m core.distributions --list`
   - `python -m core.distributions --get PARAM`
   - `python -m core.distributions --describe`
   - `python -m core.distributions --summary`
   - `python -m core.distributions --group GROUP`
   - `python -m core.distributions --tag TAG`

הערות עיצוב
-----------
- לא נוגעים ב-Optuna Study, רק ב־distributions.
- המודול סובלני: אם ParamRangeModel התרחב (attributes חדשים), הוא ניצל אותם רק אם קיימים.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import json
import logging
import math
import sys
import argparse

from optuna.distributions import (  # type: ignore
    BaseDistribution,
    UniformDistribution,
    IntUniformDistribution,
    LogUniformDistribution,
    CategoricalDistribution,
)

from core.ranges import load_ranges_config

__version__ = "3.0.0"

__all__ = [
    "get_param_distributions",
    "clear_param_distributions_cache",
    "get_distribution",
    "list_parameters",
    "export_distributions_to_json",
    "validate_distributions",
    "validate_int_steps",
    "validate_log_ranges",
    "describe_distributions",
    "summarize_search_space",
    "group_parameters_by_type",
    "get_parameters_by_group",
    "get_parameters_by_tag",
    "recommend_sampler",
    "recommend_n_trials",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def make_json_safe(obj: Any) -> Any:
    """
    הפיכת אובייקט לכלי JSON-friendly:
    - מספרים/מחרוזות/בוליאנים/None → כמות שהם.
    - dict → dict עם המרה רקורסיבית.
    - list/tuple/set → list.
    - כל אובייקט אחר → str(obj).

    המטרה:
    - לאפשר ייצוא דיסטריביושנים (עם _kwargs וכו') ללא שבירת json.dumps.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Core factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_param_distributions() -> Dict[str, BaseDistribution]:
    """
    בונה ומקאש את כל Optuna distributions לכל הפרמטרים ב־ranges.

    Discovery rules (תמצית):
    ------------------------
    1. אם ParamRangeModel.distribution קיים:
       - "log"/"loguniform"       → LogUniformDistribution
       - "int"/"intuniform"       → IntUniformDistribution
       - "categorical"            → CategoricalDistribution (מחייב choices)
       - "uniform"/אחר            → UniformDistribution

    2. אם distribution לא מוגדר:
       - אם name ברשימת log_uniform_params && low>0 → LogUniformDistribution.
       - אם low/high/step כולם שלמים → IntUniformDistribution.
       - אחרת → UniformDistribution.

    3. אם prm.choices קיים (לא ריק) → CategoricalDistribution.

    Notes
    -----
    - זהו *single source of truth* ל-search space של Optuna בכל המודולים.
    - אם רוצים לשנות צורת sample של פרמטר – עושים זאת ב־ranges, לא כאן.
    """
    cfg = load_ranges_config()
    ranges_config = cfg.default_ranges  # Dict[str, ParamRangeModel]
    distributions: Dict[str, BaseDistribution] = {}

    # Legacy list עבור log-uniform (ניתן להעלים בעתיד כשכל ה-ranges מגדירים distribution/log)
    log_uniform_params = {
        "cointegration_deviation",
        "vix_level",
        "interest_rate_diff",
        "fx_rate_volatility",
    }

    for name, prm in ranges_config.items():
        low = getattr(prm, "low", None)
        high = getattr(prm, "high", None)
        step = getattr(prm, "step", 1.0)
        dist_type = getattr(prm, "distribution", None)
        choices = getattr(prm, "choices", None)

        # Validate range
        if low is None or high is None:
            raise ValueError(f"Parameter '{name}' missing low/high in ranges config.")

        if high <= low:
            raise ValueError(
                f"Invalid range for parameter '{name}': low ({low}) >= high ({high})"
            )

        # Handle categorical early if choices defined
        if choices:
            dist = CategoricalDistribution(choices=list(choices))
            distributions[name] = dist
            logger.debug(
                "Param '%s': CategoricalDistribution(choices=%s)",
                name,
                choices,
            )
            continue

        # Normalize step
        try:
            step_val = float(step)
        except Exception:
            step_val = 1.0

        # Determine distribution type if not explicitly set
        if not dist_type:
            if name in log_uniform_params and low > 0:
                dist_type = "log"
            elif float(step_val).is_integer() and float(low).is_integer() and float(high).is_integer():
                dist_type = "int"
            else:
                dist_type = "uniform"
        else:
            dist_type = str(dist_type).lower()

        # Select distribution based on type
        if dist_type in {"log", "loguniform", "log_uniform"}:
            if low <= 0:
                raise ValueError(
                    f"Log distribution requires low>0 for parameter '{name}', got low={low}"
                )
            dist: BaseDistribution = LogUniformDistribution(low=float(low), high=float(high))
        elif dist_type in {"int", "intuniform", "int_uniform"}:
            int_low = int(low)
            int_high = int(high)
            int_step = int(step_val) if step_val >= 1 and float(step_val).is_integer() else 1
            dist = IntUniformDistribution(low=int_low, high=int_high, step=int_step)
        elif dist_type in {"categorical"}:
            # אמור היה להיתפס קודם, אבל לשם בטיחות:
            if not choices:
                raise ValueError(
                    f"Categorical distribution requires non-empty 'choices' for parameter '{name}'"
                )
            dist = CategoricalDistribution(choices=list(choices))
        else:  # uniform or default
            dist = UniformDistribution(low=float(low), high=float(high))

        distributions[name] = dist
        logger.debug(
            "Param '%s': using %s(low=%s, high=%s, step=%s, type='%s')",
            name,
            type(dist).__name__,
            low,
            high,
            step_val,
            dist_type,
        )

    logger.info("Built %d parameter distributions from ranges config.", len(distributions))
    return distributions


# ---------------------------------------------------------------------------
# Introspection utilities
# ---------------------------------------------------------------------------

def list_parameters() -> List[str]:
    """
    Return a sorted list of all available parameter names.

    Returns
    -------
    List[str]
        כל מפתחות הפרמטרים, ממויינים.
    """
    return sorted(get_param_distributions().keys())


def get_distribution(name: str) -> BaseDistribution:
    """
    Retrieve a single parameter distribution by name.

    Parameters
    ----------
    name : str
        שם הפרמטר.

    Returns
    -------
    BaseDistribution
        אובייקט ה-Distribution של Optuna.

    Raises
    ------
    KeyError
        אם אין פרמטר כזה במיפוי.
    """
    dists = get_param_distributions()
    try:
        return dists[name]
    except KeyError as exc:
        raise KeyError(f"Distribution for parameter '{name}' not found.") from exc


def _classify_kind(dist: BaseDistribution) -> str:
    """
    סיווג סוג דיסטריביושן לרמת meta:
    - "log" / "continuous" / "int" / "categorical".
    """
    if isinstance(dist, LogUniformDistribution):
        return "log"
    if isinstance(dist, IntUniformDistribution):
        return "int"
    if isinstance(dist, CategoricalDistribution):
        return "categorical"
    return "continuous"  # Uniform/default


def describe_distributions() -> Dict[str, Dict[str, Any]]:
    """
    מחזיר תיאור עשיר לכל דיסטריביושן:
    - type (שם המחלקה באופטונה)
    - kind (continuous/int/log/categorical)
    - kwargs (low/high/step/choices וכו')
    - meta מ-ParamRangeModel אם קיימים (distribution, step, tags, group, param_type).

    שימושים:
    - לוגים.
    - דשבורדים (טאב Inspect Parameters).
    - Agents שמבינים את המרחב ומעצבים קונפיגים חכמים.
    """
    cfg = load_ranges_config()
    ranges_config = cfg.default_ranges
    dists = get_param_distributions()

    out: Dict[str, Dict[str, Any]] = {}
    for name, dist in dists.items():
        prm = ranges_config.get(name)
        kwargs = getattr(dist, "_kwargs", {})

        meta: Dict[str, Any] = {}
        if prm is not None:
            for attr in ("distribution", "step", "tags", "group", "param_type"):
                if hasattr(prm, attr):
                    meta[attr] = getattr(prm, attr)

        out[name] = {
            "type": type(dist).__name__,
            "kind": _classify_kind(dist),
            "kwargs": make_json_safe(kwargs),
            "meta": make_json_safe(meta),
        }

    return out


def group_parameters_by_type() -> Dict[str, List[str]]:
    """
    מחזיר מיפוי סוג → רשימת פרמטרים:
    - "continuous" (Uniform)
    - "log"
    - "int"
    - "categorical"

    זה מאפשר להבין כמה ממד רציף/אינט/קטגורי יש לנו.
    """
    info = describe_distributions()
    groups: Dict[str, List[str]] = {"continuous": [], "log": [], "int": [], "categorical": []}
    for name, meta in info.items():
        kind = meta.get("kind", "continuous")
        if kind not in groups:
            groups[kind] = []
        groups[kind].append(name)
    return groups


def get_parameters_by_group(group: str) -> List[str]:
    """
    מחזיר את כל הפרמטרים ששייכים ל-ParamRangeModel.group == group.

    זה מאפשר לעבוד, למשל, רק עם:
    - group="risk"
    - group="signal"
    - group="macro"
    וכו'.
    """
    cfg = load_ranges_config()
    ranges_config = cfg.default_ranges
    out: List[str] = []
    for name, prm in ranges_config.items():
        if getattr(prm, "group", None) == group:
            out.append(name)
    return sorted(out)


def get_parameters_by_tag(tag: str) -> List[str]:
    """
    מחזיר פרמטרים שמכילים tag מסוים ב-ParamRangeModel.tags (אם קיים).

    Useful:
    - להפריד פרמטרים tag="execution" / tag="slippage" / tag="macro".
    """
    cfg = load_ranges_config()
    ranges_config = cfg.default_ranges
    out: List[str] = []
    for name, prm in ranges_config.items():
        tags = getattr(prm, "tags", None)
        if not tags:
            continue
        try:
            if isinstance(tags, (list, tuple, set)) and tag in tags:
                out.append(name)
        except Exception:
            continue
    return sorted(out)


# ---------------------------------------------------------------------------
# Cache & validation
# ---------------------------------------------------------------------------

def clear_param_distributions_cache() -> None:
    """
    Clear the cached parameter distributions to force a reload from configuration
    on next call.
    """
    get_param_distributions.cache_clear()
    logger.info("Parameter distributions cache cleared.")


def export_distributions_to_json(filepath: str) -> None:
    """
    Export all parameter distributions to a JSON file, including type, kind,
    init args, and basic meta.

    Parameters
    ----------
    filepath : str
        הנתיב לקובץ ה-JSON לייצוא.
    """
    dists_info = describe_distributions()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(dists_info), f, indent=2, ensure_ascii=False)
    logger.info("Exported %d distributions to %s", len(dists_info), filepath)


def validate_distributions() -> None:
    """
    Validate that every range in configuration has a corresponding distribution.

    Raises
    ------
    KeyError
        אם יש פרמטר בקונפיג ללא דיסטריביושן.
    """
    ranges = load_ranges_config().default_ranges
    dists = get_param_distributions()
    missing = [name for name in ranges if name not in dists]
    if missing:
        raise KeyError(f"No distributions found for parameters: {missing}")
    logger.info("All configured parameters have valid distributions.")


def validate_int_steps() -> List[str]:
    """
    Validate that all IntUniformDistribution have coherent steps.

    Checks:
    - step>=1
    - (high-low) divisible by step (בערך, עם טולרנס קטן).

    Returns
    -------
    List[str]
        רשימת פרמטרים עם בעיה (אם יש).
    """
    issues: List[str] = []
    dists = get_param_distributions()
    for name, dist in dists.items():
        if not isinstance(dist, IntUniformDistribution):
            continue
        kw = getattr(dist, "_kwargs", {})
        low = int(kw.get("low", 0))
        high = int(kw.get("high", 0))
        step = int(kw.get("step", 1))
        if step < 1:
            issues.append(name)
            continue
        span = high - low
        if span > 0 and span % step != 0:
            issues.append(name)
    if issues:
        logger.warning("Found int-step inconsistencies for params: %s", issues)
    else:
        logger.info("All int distributions have coherent step/low/high.")
    return issues


def validate_log_ranges() -> List[str]:
    """
    Validate that all LogUniformDistribution have low>0 and reasonable span.

    Checks:
    - low>0
    - high>low
    - log10(high/low) לא קיצוני מדי (למשל >6 → range של 10^6).

    Returns
    -------
    List[str]
        רשימת פרמטרים עם בעיה (אם יש).
    """
    issues: List[str] = []
    dists = get_param_distributions()
    for name, dist in dists.items():
        if not isinstance(dist, LogUniformDistribution):
            continue
        kw = getattr(dist, "_kwargs", {})
        low = float(kw.get("low", 0.0))
        high = float(kw.get("high", 0.0))
        if low <= 0 or high <= low:
            issues.append(name)
            continue
        span_order = math.log10(high / low)
        if span_order > 6:
            issues.append(name)
    if issues:
        logger.warning("Found log-range issues for params: %s", issues)
    else:
        logger.info("All log distributions have reasonable ranges.")
    return issues


# ---------------------------------------------------------------------------
# Search-space analytics & recommendations
# ---------------------------------------------------------------------------

def summarize_search_space() -> Dict[str, Any]:
    """
    מסכם את מרחב הפרמטרים לצרכי ניתוח:

    Returns
    -------
    Dict[str, Any] עם שדות לדוגמה:
    - n_params                      — מספר פרמטרים.
    - counts_by_kind                — {continuous, log, int, categorical}.
    - approx_volume_log10           — לוג10 של "נפח" חיפוש (הערכה).
    - effective_dimensionality      — מספר ממדים רציפים+אינט+קטגוריים.
    - suggestions                   — רשימת הערות כלליות.
    """
    info = describe_distributions()
    groups = group_parameters_by_type()

    n_params = len(info)
    counts_by_kind = {k: len(v) for k, v in groups.items()}

    # ננסה להעריך "נפח" חיפוש: continuous/log → high-low; int → מס' נקודות; cat → #choices
    approx_volume = 1.0
    for name, meta in info.items():
        kind = meta.get("kind", "continuous")
        kwargs = meta.get("kwargs", {})
        try:
            if kind in {"continuous", "log"}:
                low = float(kwargs.get("low", 0.0))
                high = float(kwargs.get("high", 0.0))
                span = max(high - low, 1e-9)
                approx_volume *= span
            elif kind == "int":
                low = int(kwargs.get("low", 0))
                high = int(kwargs.get("high", 0))
                step = int(kwargs.get("step", 1))
                pts = max(((high - low) // max(step, 1)) + 1, 1)
                approx_volume *= float(pts)
            elif kind == "categorical":
                choices = kwargs.get("choices", []) or []
                approx_volume *= max(len(choices), 1)
        except Exception:
            # אם משהו נראה מוזר — פשוט לא נתרום אותו לנפח
            continue

    approx_volume_log10 = math.log10(approx_volume) if approx_volume > 0 else float("-inf")

    # אפקטיב דימנשיונליטי — פשוט ספירה של כל הפרמטרים הלא-דגנרטיביים
    effective_dim = 0
    for name, meta in info.items():
        kind = meta.get("kind", "continuous")
        kw = meta.get("kwargs", {})
        if kind in {"continuous", "log"}:
            low = kw.get("low")
            high = kw.get("high")
            if low is not None and high is not None and high > low:
                effective_dim += 1
        elif kind in {"int", "categorical"}:
            effective_dim += 1

    suggestions: List[str] = []
    if effective_dim <= 5:
        suggestions.append("מרחב הפרמטרים קטן יחסית — אפשר להשתמש ב-CMA-ES או Grid מקומי.")
    elif effective_dim <= 15:
        suggestions.append("מרחב הפרמטרים בינוני — TPE או CMA-ES מתאימים, עם 200–500 ניסויים.")
    else:
        suggestions.append("מרחב הפרמטרים גבוה-ממד — TPE/Random + pruning אגרסיבי רצויים.")

    if counts_by_kind.get("categorical", 0) > 0:
        suggestions.append("יש פרמטרים קטגוריים — TPE בדרך כלל מתמודד טוב יותר מ-CMA-ES.")

    if approx_volume_log10 > 6:
        suggestions.append("נפח החיפוש גדול מאוד (10^6+) — כדאי להגביל ranges או לבצע coarse→fine search.")

    return {
        "n_params": n_params,
        "counts_by_kind": counts_by_kind,
        "approx_volume_log10": approx_volume_log10,
        "effective_dimensionality": effective_dim,
        "suggestions": suggestions,
    }


def recommend_sampler() -> str:
    """
    המלצה בסיסית על sampler (TPE / CMA-ES) לפי מאפייני מרחב הפרמטרים.

    Heuristics:
    - אם יש categorical → TPE.
    - אם n_params <= 5 וכל הפרמטרים continuous/log → CMA-ES.
    - אחרת → TPE.
    """
    info = describe_distributions()
    kinds = {meta.get("kind", "continuous") for meta in info.values()}
    n_params = len(info)

    if "categorical" in kinds:
        return "tpe"
    if n_params <= 5 and kinds.issubset({"continuous", "log"}):
        return "cmaes"
    return "tpe"


def recommend_n_trials(target: str = "balanced") -> int:
    """
    המלצה גסה על n_trials לפי dimensionality והעדפה.

    target:
    -------
    - "small"     — לכיול מהיר מאוד / debug.
    - "balanced"  — trade-off בין זמן לתוצאה (ברירת מחדל).
    - "deep"      — חיפוש עמוק יותר.

    Heuristic:
    ----------
    n_trials ≈ factor * effective_dimensionality^2
    כאשר:
    - factor=5  ל-small
    - factor=10 ל-balanced
    - factor=20 ל-deep
    """
    smry = summarize_search_space()
    dim = smry.get("effective_dimensionality", 0) or 0

    if dim <= 0:
        return 50  # ברירת מחדל לבטא
    if target == "small":
        factor = 5
    elif target == "deep":
        factor = 20
    else:
        factor = 10

    n_trials = max(int(factor * dim * dim), 20)
    return n_trials


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_list() -> None:
    """הדפסת רשימת פרמטרים בפורמט JSON."""
    params = list_parameters()
    print(json.dumps(make_json_safe(params), indent=2, ensure_ascii=False))


def _cli_get(param: str) -> None:
    """הדפסת דיסטריביושן בודד ל-stdout כ-JSON."""
    try:
        dist = get_distribution(param)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)

    output = {
        "name": param,
        "type": type(dist).__name__,
        "kind": _classify_kind(dist),
        "kwargs": make_json_safe(getattr(dist, "_kwargs", {})),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _cli_describe() -> None:
    """הדפסת תיאור מלא של כל הדיסטריביושנים (type + kind + kwargs + meta)."""
    info = describe_distributions()
    print(json.dumps(make_json_safe(info), indent=2, ensure_ascii=False))


def _cli_summary() -> None:
    """הדפסת summary של מרחב הפרמטרים."""
    smry = summarize_search_space()
    print(json.dumps(make_json_safe(smry), indent=2, ensure_ascii=False))


def _cli_group(group: str) -> None:
    """הדפסת פרמטרים לפי group (למשל 'risk', 'signal', וכו')."""
    params = get_parameters_by_group(group)
    print(json.dumps(make_json_safe(params), indent=2, ensure_ascii=False))


def _cli_tag(tag: str) -> None:
    """הדפסת פרמטרים לפי tag ב-ParamRangeModel.tags."""
    params = get_parameters_by_tag(tag)
    print(json.dumps(make_json_safe(params), indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Manage and query parameter distributions for optimization."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all parameter names")
    group.add_argument("--get", metavar="PARAM", help="Get distribution for PARAM as JSON")
    group.add_argument(
        "--describe",
        action="store_true",
        help="Describe all distributions (type + kwargs + meta) as JSON",
    )
    group.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of search space (dimensionality, volume, suggestions)",
    )
    group.add_argument(
        "--group",
        metavar="GROUP",
        help="List parameters for a given ParamRangeModel.group",
    )
    group.add_argument(
        "--tag",
        metavar="TAG",
        help="List parameters that contain a given tag in ParamRangeModel.tags",
    )

    args = parser.parse_args()

    if args.list:
        _cli_list()
        sys.exit(0)

    if args.describe:
        _cli_describe()
        sys.exit(0)

    if args.summary:
        _cli_summary()
        sys.exit(0)

    if args.group:
        _cli_group(args.group)
        sys.exit(0)

    if args.tag:
        _cli_tag(args.tag)
        sys.exit(0)

    if args.get:
        _cli_get(args.get)
        sys.exit(0)
