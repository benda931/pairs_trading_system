from __future__ import annotations

import json
from pathlib import Path

from common.pair_utils import (
    extract_symbols_from_pairs,
    load_asset_policy,
    normalize_pairs,
    pair_allowed_by_policy,
    parse_pair_record,
)


def test_parse_pair_record_supports_slash():
    assert parse_pair_record("SPY/QQQ") == ("SPY", "QQQ")


def test_parse_pair_record_supports_dash():
    assert parse_pair_record("SPY-QQQ") == ("SPY", "QQQ")


def test_parse_pair_record_supports_symbols_array_dict():
    assert parse_pair_record({"symbols": ["SPY", "QQQ"]}) == ("SPY", "QQQ")


def test_parse_pair_record_supports_sym_x_sym_y():
    assert parse_pair_record({"sym_x": "SPY", "sym_y": "QQQ"}) == ("SPY", "QQQ")


def test_parse_pair_record_supports_symbol_1_symbol_2():
    assert parse_pair_record({"symbol_1": "SPY", "symbol_2": "QQQ"}) == ("SPY", "QQQ")


def test_normalize_pairs_dedupes_reversed_pairs():
    pairs = normalize_pairs(["SPY/QQQ", "QQQ/SPY", ("IWM", "SPY")])
    assert pairs == [("SPY", "QQQ"), ("IWM", "SPY")]


def test_pair_allowed_by_policy_blocks_ibit_etha():
    assert pair_allowed_by_policy("IBIT", "ETHA", policy={"allow_crypto": False}) is False


def test_pair_allowed_by_policy_blocks_blok_bkch():
    assert pair_allowed_by_policy("BLOK", "BKCH", policy={"allow_crypto": False}) is False


def test_pair_allowed_by_policy_allows_spy_qqq():
    assert pair_allowed_by_policy("SPY", "QQQ", policy={"allow_crypto": False}) is True


def test_pair_allowed_by_policy_enforces_etf_like_allowlist():
    policy = {
        "allow_crypto": False,
        "enforce_etf_like_in_production": True,
        "etf_like_symbols": ["SPY", "QQQ"],
    }
    assert pair_allowed_by_policy("SPY", "QQQ", policy=policy) is True


def test_pair_allowed_by_policy_blocks_non_allowlisted_symbol_when_enforced():
    policy = {
        "allow_crypto": False,
        "enforce_etf_like_in_production": True,
        "etf_like_symbols": ["SPY", "QQQ"],
    }
    assert pair_allowed_by_policy("SPY", "AAPL", policy=policy) is False


def test_pair_allowed_by_policy_allows_non_etf_symbol_when_not_enforced():
    policy = {
        "allow_crypto": False,
        "enforce_etf_like_in_production": False,
        "etf_like_symbols": ["SPY", "QQQ"],
    }
    assert pair_allowed_by_policy("SPY", "AAPL", policy=policy) is True


def test_configured_production_pairs_are_allowed_under_enforced_etf_like_policy():
    repo_root = Path(__file__).resolve().parent.parent
    cfg = json.loads((repo_root / "config.json").read_text(encoding="utf-8"))
    policy = load_asset_policy(cfg)

    assert policy["enforce_etf_like_in_production"] is True
    for pair in cfg["production_pairs"]:
        sym_x, sym_y = parse_pair_record(pair)
        assert pair_allowed_by_policy(sym_x, sym_y, policy=policy) is True


def test_extract_symbols_from_config_style_string_pairs():
    symbols = extract_symbols_from_pairs(["SPY/QQQ", "IWM/SPY"])
    assert symbols == ["IWM", "QQQ", "SPY"]
