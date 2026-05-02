# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


DEFAULT_BLOCKED_SYMBOLS = [
    "IBIT",
    "ETHA",
    "BITO",
    "BLOK",
    "BKCH",
    "WGMI",
    "MSTR",
    "COIN",
    "RIOT",
    "MARA",
    "GBTC",
    "ETHE",
    "BTC",
    "ETH",
    "BTC-USD",
    "ETH-USD",
]

DEFAULT_BLOCKED_CATEGORIES = ["crypto", "bitcoin", "blockchain", "btc", "eth"]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except Exception:
        pass
    text = str(value).strip().upper()
    return text in {"", "NAN", "NONE", "<NA>", "NULL"}


def _coerce_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _coerce_bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return None


def _default_asset_policy() -> dict[str, Any]:
    return {
        "enabled": True,
        "prefer_etf_like": True,
        "allow_crypto": False,
        "blocked_symbols": list(DEFAULT_BLOCKED_SYMBOLS),
        "blocked_categories": list(DEFAULT_BLOCKED_CATEGORIES),
        "require_is_viable_for_production": True,
        "min_n_obs": 252,
        "max_half_life": 200,
        "max_corr_clone": 0.995,
    }


def canonical_symbol(symbol: Any) -> str:
    if _is_missing(symbol):
        return ""
    return str(symbol).strip().upper()


def parse_pair_record(pair: Any) -> tuple[str, str] | None:
    sym_x: Any = None
    sym_y: Any = None

    if isinstance(pair, str):
        raw = pair.strip()
        if "/" in raw:
            sym_x, sym_y = raw.split("/", 1)
        elif "-" in raw:
            sym_x, sym_y = raw.split("-", 1)
        elif "_" in raw:
            sym_x, sym_y = raw.split("_", 1)
    elif isinstance(pair, Mapping):
        symbols = pair.get("symbols")
        if isinstance(symbols, (list, tuple)) and len(symbols) >= 2:
            sym_x, sym_y = symbols[0], symbols[1]
        else:
            sym_x = pair.get("sym_x") or pair.get("symbol_x") or pair.get("symbol_1")
            sym_y = pair.get("sym_y") or pair.get("symbol_y") or pair.get("symbol_2")
            if _is_missing(sym_x) or _is_missing(sym_y):
                pair_text = pair.get("pair") or pair.get("pair_id")
                if isinstance(pair_text, str):
                    return parse_pair_record(pair_text)
    elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
        sym_x, sym_y = pair[0], pair[1]

    left = canonical_symbol(sym_x)
    right = canonical_symbol(sym_y)
    if not left or not right or left == right:
        return None
    return left, right


def canonical_pair_id(sym_x: str, sym_y: str, sep: str = "/") -> str:
    left = canonical_symbol(sym_x)
    right = canonical_symbol(sym_y)
    return f"{left}{sep}{right}" if left and right else ""


def normalize_pairs(pairs: Iterable[Any], dedupe: bool = True) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs or []:
        parsed = parse_pair_record(pair)
        if parsed is None:
            continue
        sym_x, sym_y = parsed
        if not sym_x or not sym_y or sym_x == sym_y:
            continue
        key = tuple(sorted((sym_x, sym_y)))
        if dedupe and key in seen:
            continue
        seen.add(key)
        normalized.append((sym_x, sym_y))
    return normalized


def extract_symbols_from_pairs(pairs: Iterable[Any]) -> list[str]:
    symbols = {sym for pair in normalize_pairs(pairs) for sym in pair}
    return sorted(symbols)


def is_crypto_symbol(symbol: str, policy: Mapping[str, Any] | None = None) -> bool:
    sym = canonical_symbol(symbol)
    if not sym:
        return False
    effective_policy = load_asset_policy({"asset_policy": dict(policy or {})}) if policy else _default_asset_policy()
    blocked_symbols = {
        canonical_symbol(item)
        for item in list(effective_policy.get("blocked_symbols", []) or [])
        if canonical_symbol(item)
    }
    if sym in blocked_symbols:
        return True
    if "BTC" in sym or "ETH" in sym:
        return True
    return False


def is_crypto_category(category: Any) -> bool:
    if _is_missing(category):
        return False
    text = str(category).strip().lower()
    return any(token in text for token in DEFAULT_BLOCKED_CATEGORIES)


def is_crypto_related_pair(
    sym_x: Any,
    sym_y: Any,
    seed_category: Any = None,
    policy: Mapping[str, Any] | None = None,
) -> bool:
    return (
        is_crypto_symbol(canonical_symbol(sym_x), policy=policy)
        or is_crypto_symbol(canonical_symbol(sym_y), policy=policy)
        or is_crypto_category(seed_category)
    )


def load_asset_policy(config: Mapping[str, Any] | None = None) -> dict:
    policy = _default_asset_policy()
    overrides = {}
    if isinstance(config, Mapping):
        candidate = config.get("asset_policy")
        if isinstance(candidate, Mapping):
            overrides = dict(candidate)
    policy.update(overrides)
    policy["blocked_symbols"] = [
        sym
        for sym in (canonical_symbol(item) for item in list(policy.get("blocked_symbols", []) or []))
        if sym
    ]
    policy["blocked_categories"] = [
        str(item).strip().lower()
        for item in list(policy.get("blocked_categories", []) or [])
        if not _is_missing(item)
    ]
    return policy


def pair_allowed_by_policy(
    sym_x,
    sym_y,
    *,
    seed_category=None,
    row=None,
    policy=None,
) -> bool:
    effective_policy = load_asset_policy({"asset_policy": dict(policy or {})}) if policy else _default_asset_policy()
    if not bool(effective_policy.get("enabled", True)):
        return True

    row_map = dict(row) if isinstance(row, Mapping) else {}
    category = seed_category
    if _is_missing(category):
        for key in ("seed_category", "category", "asset_category", "universe_category"):
            if key in row_map and not _is_missing(row_map.get(key)):
                category = row_map.get(key)
                break

    if not bool(effective_policy.get("allow_crypto", False)):
        if is_crypto_related_pair(sym_x, sym_y, seed_category=category, policy=effective_policy):
            return False

    if bool(effective_policy.get("require_is_viable_for_production", True)):
        viable = None
        for key in ("is_viable", "is_viable_for_production"):
            if key in row_map:
                viable = _coerce_bool(row_map.get(key))
                if viable is not None:
                    break
        if viable is False:
            return False

    min_n_obs = _coerce_float(effective_policy.get("min_n_obs"))
    n_obs = _coerce_float(row_map.get("n_obs"))
    if min_n_obs is not None and n_obs is not None and n_obs < min_n_obs:
        return False

    max_half_life = _coerce_float(effective_policy.get("max_half_life"))
    half_life = _coerce_float(row_map.get("half_life"))
    if max_half_life is not None and half_life is not None and half_life > max_half_life:
        return False

    max_corr_clone = _coerce_float(effective_policy.get("max_corr_clone"))
    corr = _coerce_float(row_map.get("corr"))
    is_clone_like = _coerce_bool(row_map.get("is_clone_like"))
    if (
        max_corr_clone is not None
        and corr is not None
        and is_clone_like is True
        and corr >= max_corr_clone
    ):
        return False

    return True


__all__ = [
    "canonical_symbol",
    "parse_pair_record",
    "canonical_pair_id",
    "normalize_pairs",
    "extract_symbols_from_pairs",
    "is_crypto_symbol",
    "is_crypto_category",
    "is_crypto_related_pair",
    "load_asset_policy",
    "pair_allowed_by_policy",
]
