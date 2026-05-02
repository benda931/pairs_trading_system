from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _write_ranked_csv(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "sym_x": "SPY",
                "sym_y": "QQQ",
                "pair_score": 0.92,
                "pair_label": "A",
                "corr": 0.88,
                "p_value": 0.01,
                "half_life": 30,
                "n_obs": 500,
                "is_viable": True,
                "is_clone_like": False,
                "seed_category": "etf",
            },
            {
                "sym_x": "IBIT",
                "sym_y": "ETHA",
                "pair_score": 0.95,
                "pair_label": "A+",
                "corr": 0.90,
                "p_value": 0.01,
                "half_life": 25,
                "n_obs": 500,
                "is_viable": True,
                "is_clone_like": False,
                "seed_category": "crypto",
            },
            {
                "sym_x": "VOO",
                "sym_y": "SPY",
                "pair_score": 0.89,
                "pair_label": "A",
                "corr": 0.999,
                "p_value": 0.01,
                "half_life": 10,
                "n_obs": 500,
                "is_viable": True,
                "is_clone_like": True,
                "seed_category": "etf",
            },
            {
                "sym_x": "IBB",
                "sym_y": "XBI",
                "pair_score": 0.87,
                "pair_label": "B+",
                "corr": 0.84,
                "p_value": 0.02,
                "half_life": 45,
                "n_obs": 520,
                "is_viable": True,
                "is_clone_like": False,
                "seed_category": "etf",
            },
            {
                "sym_x": "SPY",
                "sym_y": "AAPL",
                "pair_score": 0.86,
                "pair_label": "B+",
                "corr": 0.82,
                "p_value": 0.02,
                "half_life": 40,
                "n_obs": 510,
                "is_viable": True,
                "is_clone_like": False,
                "seed_category": "mixed",
            },
        ]
    )
    df.to_csv(path, index=False)


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "pairs": ["SPY/QQQ"],
                "production_pairs": [],
                "use_production_pairs": False,
                "asset_policy": {
                    "enabled": True,
                    "allow_crypto": False,
                    "require_is_viable_for_production": True,
                    "enforce_etf_like_in_production": False,
                    "etf_like_symbols": ["SPY", "QQQ", "IBB", "XBI"],
                    "min_n_obs": 252,
                    "max_half_life": 200,
                    "max_corr_clone": 0.995,
                    "blocked_symbols": ["IBIT", "ETHA"],
                    "blocked_categories": ["crypto", "bitcoin", "blockchain", "btc", "eth"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_promote_pairs_to_production_without_wf_filters_crypto_and_writes_report(tmp_path):
    from scripts.promote_pairs_to_production import main

    ranked_csv = tmp_path / "pairs_universe_ranked.csv"
    config_path = tmp_path / "config.json"
    output_json = tmp_path / "production_pairs_approved.json"
    output_csv = tmp_path / "production_pairs_approved.csv"
    _write_ranked_csv(ranked_csv)
    _write_config(config_path)

    rc = main(
        [
            "--ranked-csv",
            str(ranked_csv),
            "--config",
            str(config_path),
            "--top",
            "30",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--min-score",
            "0.3",
            "--min-label",
            "B-",
            "--min-n-obs",
            "252",
            "--max-half-life",
            "200",
            "--no-crypto",
            "--require-viable",
        ]
    )

    assert rc == 0
    approved = json.loads(output_json.read_text(encoding="utf-8"))
    assert "SPY/QQQ" in approved
    assert "IBB/XBI" in approved
    assert "IBIT/ETHA" not in approved
    report = pd.read_csv(output_csv)
    blocked = report.set_index(["sym_x", "sym_y"])["rejection_reason"].to_dict()
    assert blocked[("IBIT", "ETHA")] == "crypto_blocked"
    assert blocked[("VOO", "SPY")] == "clone_like_blocked"


def test_promote_pairs_to_production_with_wf_gate_and_config_backup(tmp_path, monkeypatch):
    import scripts.promote_pairs_to_production as promotion_module

    ranked_csv = tmp_path / "pairs_universe_ranked.csv"
    config_path = tmp_path / "config.json"
    output_json = tmp_path / "production_pairs_approved.json"
    output_csv = tmp_path / "production_pairs_approved.csv"
    _write_ranked_csv(ranked_csv)
    _write_config(config_path)

    def _fake_run_walk_forward(sym_x, sym_y, **kwargs):
        if (sym_x, sym_y) == ("SPY", "QQQ"):
            folds = [SimpleNamespace(oos_trades=6), SimpleNamespace(oos_trades=5)]
            return SimpleNamespace(
                dsr_gate_passed=True,
                deflated_sharpe=0.75,
                avg_oos_sharpe=0.42,
                prob_overfit=20.0,
                param_stability_score=0.81,
                param_stability_warning=False,
                folds=folds,
            )
        folds = [SimpleNamespace(oos_trades=9)]
        return SimpleNamespace(
            dsr_gate_passed=False,
            deflated_sharpe=0.41,
            avg_oos_sharpe=0.22,
            prob_overfit=65.0,
            param_stability_score=0.77,
            param_stability_warning=False,
            folds=folds,
        )

    monkeypatch.setattr(promotion_module, "run_walk_forward", _fake_run_walk_forward)

    rc = promotion_module.main(
        [
            "--ranked-csv",
            str(ranked_csv),
            "--config",
            str(config_path),
            "--top",
            "30",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--min-score",
            "0.3",
            "--min-label",
            "B-",
            "--min-n-obs",
            "252",
            "--max-half-life",
            "200",
            "--no-crypto",
            "--require-viable",
            "--run-walk-forward",
            "--update-config",
        ]
    )

    assert rc == 0
    approved = json.loads(output_json.read_text(encoding="utf-8"))
    assert approved == ["SPY/QQQ"]
    updated_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert updated_cfg["production_pairs"] == ["SPY/QQQ"]
    assert updated_cfg["use_production_pairs"] is True
    backups = list((tmp_path / "configs").glob("config.pre_production_update.*.json"))
    assert backups, "expected backup config file to be created"

    report = pd.read_csv(output_csv)
    row = report[(report["sym_x"] == "IBB") & (report["sym_y"] == "XBI")].iloc[0]
    assert row["rejection_reason"] in {
        "wf_dsr_gate_failed;wf_dsr_below_threshold;wf_oos_sharpe_below_threshold;wf_prob_overfit_above_threshold",
        "wf_dsr_gate_failed;wf_dsr_below_threshold;wf_oos_sharpe_below_threshold;wf_prob_overfit_above_threshold",
    }


def test_promote_pairs_to_production_enforces_etf_like_allowlist(tmp_path):
    from scripts.promote_pairs_to_production import main

    ranked_csv = tmp_path / "pairs_universe_ranked.csv"
    config_path = tmp_path / "config.json"
    output_json = tmp_path / "production_pairs_approved.json"
    output_csv = tmp_path / "production_pairs_approved.csv"
    _write_ranked_csv(ranked_csv)
    _write_config(config_path)

    rc = main(
        [
            "--ranked-csv",
            str(ranked_csv),
            "--config",
            str(config_path),
            "--top",
            "30",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--min-score",
            "0.3",
            "--min-label",
            "B-",
            "--min-n-obs",
            "252",
            "--max-half-life",
            "200",
            "--no-crypto",
            "--enforce-etf-like",
            "--require-viable",
        ]
    )

    assert rc == 0
    approved = json.loads(output_json.read_text(encoding="utf-8"))
    assert "SPY/QQQ" in approved
    assert "IBB/XBI" in approved
    assert "SPY/AAPL" not in approved
    report = pd.read_csv(output_csv)
    blocked = report.set_index(["sym_x", "sym_y"])["rejection_reason"].to_dict()
    assert blocked[("SPY", "AAPL")] == "not_etf_like"


def test_promote_pairs_to_production_dedupes_reversed_pairs(tmp_path):
    from scripts.promote_pairs_to_production import main

    ranked_csv = tmp_path / "pairs_universe_ranked.csv"
    config_path = tmp_path / "config.json"
    output_json = tmp_path / "production_pairs_approved.json"
    output_csv = tmp_path / "production_pairs_approved.csv"
    _write_ranked_csv(ranked_csv)
    _write_config(config_path)

    df = pd.read_csv(ranked_csv)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "sym_x": "QQQ",
                        "sym_y": "SPY",
                        "pair_score": 0.91,
                        "pair_label": "A",
                        "corr": 0.88,
                        "p_value": 0.01,
                        "half_life": 30,
                        "n_obs": 500,
                        "is_viable": True,
                        "is_clone_like": False,
                        "seed_category": "etf",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df.to_csv(ranked_csv, index=False)

    rc = main(
        [
            "--ranked-csv",
            str(ranked_csv),
            "--config",
            str(config_path),
            "--top",
            "30",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--min-score",
            "0.3",
            "--min-label",
            "B-",
            "--min-n-obs",
            "252",
            "--max-half-life",
            "200",
            "--no-crypto",
            "--require-viable",
        ]
    )

    assert rc == 0
    approved = json.loads(output_json.read_text(encoding="utf-8"))
    assert "SPY/QQQ" in approved
    assert "QQQ/SPY" not in approved
    report = pd.read_csv(output_csv)
    blocked = report.set_index(["sym_x", "sym_y"])["rejection_reason"].to_dict()
    assert blocked[("QQQ", "SPY")] == "duplicate_unordered_pair"
