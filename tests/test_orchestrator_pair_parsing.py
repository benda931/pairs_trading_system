from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_get_configured_pairs_filters_crypto_production_pairs(monkeypatch):
    import common.config_manager as config_manager_module
    from core.orchestrator import _get_configured_pairs

    monkeypatch.setattr(
        config_manager_module,
        "load_config",
        lambda: {
            "use_production_pairs": True,
            "production_pairs": ["SPY/QQQ", "IBIT/ETHA"],
            "asset_policy": {"allow_crypto": False},
        },
    )

    assert _get_configured_pairs() == [("SPY", "QQQ")]


def test_task_data_refresh_uses_string_pairs_from_config(monkeypatch):
    import common.config_manager as config_manager_module
    import common.data_loader as data_loader_module
    from core.orchestrator import task_data_refresh

    captured: dict[str, list[str]] = {"symbols": []}

    class _StubStore:
        default_env = "dev"

        def load_prices_coverage_summary(self, env=None, warn_on_error=False):
            return pd.DataFrame()

        def save_json(self, namespace, key, payload, warn_on_error=False):
            return None

    def _fake_load_price_data(symbol, *args, **kwargs):
        return pd.DataFrame()

    def _fake_get_price_metadata(df, *, symbol=""):
        return {
            "symbol": symbol,
            "n_rows": 0,
            "n_cols": 0,
            "start_date": None,
            "end_date": None,
            "has_nans": False,
            "freq": None,
            "intraday": False,
            "data_source": None,
            "age_hours": None,
        }

    def _fake_bulk_download(symbols, *args, **kwargs):
        captured["symbols"] = sorted(symbols)
        return [Path(f"data/daily/{sym}.csv") for sym in symbols]

    monkeypatch.setattr(
        config_manager_module,
        "load_config",
        lambda: {
            "use_production_pairs": True,
            "production_pairs": ["SPY/QQQ", "IBIT/ETHA"],
            "asset_policy": {"allow_crypto": False},
            "scheduler_price_backfill_days": 30,
            "scheduler_price_refresh_days": 7,
        },
    )
    monkeypatch.setattr(data_loader_module, "load_price_data", _fake_load_price_data)
    monkeypatch.setattr(data_loader_module, "get_price_metadata", _fake_get_price_metadata)
    monkeypatch.setattr(data_loader_module, "bulk_download", _fake_bulk_download)
    monkeypatch.setattr("core.orchestrator.is_external_scheduler_daemon_active", lambda: False)
    monkeypatch.setattr(
        "core.sql_store.SqlStore.from_settings",
        lambda *args, **kwargs: _StubStore(),
    )

    payload = task_data_refresh(run_id="refresh_prod_pairs")

    assert payload["status"] in {"ok", "partial"}
    assert payload["symbols_requested"] == 2
    assert captured["symbols"] == ["QQQ", "SPY"]
