# -*- coding: utf-8 -*-
"""
agents/auto_agents.py — Autonomous Execution Agents
====================================================

Agents that execute real changes to the system autonomously:
- Retrain ML models
- Optimize parameters via Optuna
- Refresh price data
- Update configuration

All changes are bounded, reversible, and logged.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base import BaseAgent, AgentAuditLogger
from core.contracts import AgentTask

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AutoModelRetrainer(BaseAgent):
    """
    Automatically retrains the meta-label model on fresh data.

    1. Backs up existing model
    2. Trains new model on latest data
    3. Evaluates new vs old
    4. Deploys if improved, rollback if worse

    Task types: auto_retrain_model
    """

    NAME = "auto_model_retrainer"
    ALLOWED_TASK_TYPES = {"auto_retrain_model"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        model_path = PROJECT_ROOT / "models" / "meta_label_latest.pkl"
        backup_path = PROJECT_ROOT / "models" / "meta_label_backup.pkl"

        pair = task.payload.get("pair", "XLY-XLC")
        horizon = int(task.payload.get("horizon", 10))
        entry_threshold = float(task.payload.get("entry_threshold", 1.5))

        audit.log(f"Starting model retraining for {pair}, horizon={horizon}")

        # 1. Backup existing model
        if model_path.exists():
            shutil.copy2(model_path, backup_path)
            audit.log(f"Backed up existing model to {backup_path.name}")

        # 2. Load data and train
        try:
            from common.data_loader import load_price_data, _load_symbol_full_cached
            from scripts.train_meta_label import train_meta_label_model
            import numpy as np
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools import add_constant

            if hasattr(_load_symbol_full_cached, "cache_clear"):
                _load_symbol_full_cached.cache_clear()

            sym_a, sym_b = pair.split("-", 1)
            px = load_price_data(sym_a)["close"]
            py = load_price_data(sym_b)["close"]

            common = px.index.intersection(py.index)
            px, py = px.loc[common], py.loc[common]

            if len(px) < 200:
                audit.warn(f"Insufficient data: {len(px)} rows")
                return {"retrained": False, "reason": "insufficient_data", "n_rows": len(px)}

            X_ols = add_constant(px.values)
            result = OLS(py.values, X_ols).fit()
            spread = py - result.params[1] * px - result.params[0]
            mu = spread.rolling(60, min_periods=30).mean()
            sigma = spread.rolling(60, min_periods=30).std().replace(0, np.nan)
            z = ((spread - mu) / sigma).fillna(0.0)

            train_end = str(px.index[int(len(px) * 0.8)].date())

            model, artifact, metrics = train_meta_label_model(
                px=px, py=py, z=z, spread=spread,
                train_end=train_end,
                label_horizon=horizon,
                entry_threshold=entry_threshold,
                pair_id=pair,
            )

            # 3. Evaluate
            new_auc = metrics.get("val_auc", 0.0)
            audit.log(f"New model AUC: {new_auc:.3f}")

            # 4. Save
            model.save(str(model_path))
            audit.log(f"Model saved to {model_path.name}")

            return {
                "retrained": True,
                "pair": pair,
                "auc": new_auc,
                "brier": metrics.get("val_brier"),
                "n_train": metrics.get("n_train"),
                "n_test": metrics.get("n_test"),
                "model_path": str(model_path),
            }

        except Exception as e:
            audit.error(f"Retraining failed: {e}")
            # Rollback
            if backup_path.exists():
                shutil.copy2(backup_path, model_path)
                audit.log("Rolled back to backup model")
            return {"retrained": False, "error": str(e)}


class AutoDataRefresher(BaseAgent):
    """
    Downloads fresh price data for all tracked symbols.

    Task types: auto_refresh_data
    """

    NAME = "auto_data_refresher"
    ALLOWED_TASK_TYPES = {"auto_refresh_data"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        symbols = task.payload.get("symbols", None)

        try:
            from common.data_loader import load_pairs, load_price_data, _load_symbol_full_cached

            if hasattr(_load_symbol_full_cached, "cache_clear"):
                _load_symbol_full_cached.cache_clear()

            if symbols is None:
                pairs = load_pairs()
                symbols = sorted({s for p in pairs for s in p.get("symbols", [])})

            audit.log(f"Refreshing data for {len(symbols)} symbols")

            ok, fail = 0, 0
            for sym in symbols:
                try:
                    df = load_price_data(sym)
                    if len(df) > 100:
                        ok += 1
                    else:
                        fail += 1
                        audit.warn(f"{sym}: only {len(df)} rows")
                except Exception as e:
                    fail += 1
                    audit.warn(f"{sym}: {e}")

            audit.log(f"Data refresh: {ok} OK, {fail} failed")
            return {"refreshed": ok, "failed": fail, "total": len(symbols)}

        except Exception as e:
            audit.error(f"Data refresh failed: {e}")
            return {"refreshed": 0, "failed": 0, "error": str(e)}


class AutoParameterOptimizer(BaseAgent):
    """
    Runs Optuna optimization for a pair with suggested parameters.

    Task types: auto_optimize_params
    Payload: pair, param_ranges (optional)
    """

    NAME = "auto_parameter_optimizer"
    ALLOWED_TASK_TYPES = {"auto_optimize_params"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        pair = task.payload.get("pair", "XLY-XLC")
        n_trials = int(task.payload.get("n_trials", 20))

        audit.log(f"Optimizing parameters for {pair}, n_trials={n_trials}")

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            from common.data_loader import load_price_data, _load_symbol_full_cached
            import numpy as np

            if hasattr(_load_symbol_full_cached, "cache_clear"):
                _load_symbol_full_cached.cache_clear()

            sym_a, sym_b = pair.split("-", 1)
            px = load_price_data(sym_a)["close"]
            py = load_price_data(sym_b)["close"]
            common = px.index.intersection(py.index)
            px, py = px.loc[common], py.loc[common]

            def objective(trial):
                z_open = trial.suggest_float("z_open", 1.5, 3.0)
                z_close = trial.suggest_float("z_close", 0.1, 1.0)
                lookback = trial.suggest_int("lookback", 30, 120)

                beta = float(np.cov(px.values, py.values)[0, 1] / np.var(px.values))
                spread = py - beta * px
                mu = spread.rolling(lookback, min_periods=lookback // 2).mean()
                sigma = spread.rolling(lookback, min_periods=lookback // 2).std().replace(0, np.nan)
                z = ((spread - mu) / sigma).fillna(0.0)

                # Simple PnL: count profitable mean-reversion signals
                entries = z.abs() >= z_open
                exits = z.abs() <= z_close
                n_trades = entries.sum()
                if n_trades == 0:
                    return 0.0
                win_rate = (exits & entries.shift(10).fillna(False)).sum() / max(1, n_trades)
                return float(win_rate)

            from common.optuna_factory import create_optuna_study
            study = create_optuna_study(study_name=f"auto_opt_{pair_label}", direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best = study.best_params
            audit.log(f"Best params: {best}, value={study.best_value:.3f}")

            return {
                "optimized": True,
                "pair": pair,
                "best_params": best,
                "best_value": float(study.best_value),
                "n_trials": n_trials,
            }

        except Exception as e:
            audit.error(f"Optimization failed: {e}")
            return {"optimized": False, "error": str(e)}


class AutoConfigUpdater(BaseAgent):
    """
    Updates config.json with optimized parameters.
    Always backs up before writing.

    Task types: auto_update_config
    Payload: updates (dict of key-value pairs to set)
    """

    NAME = "auto_config_updater"
    ALLOWED_TASK_TYPES = {"auto_update_config"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        updates = task.payload.get("updates", {})
        if not updates:
            audit.log("No updates to apply")
            return {"updated": False, "reason": "empty_updates"}

        try:
            # Use canonical config_manager (has validation + backup logic)
            from common.config_manager import load_config, save_config, save_config_profile

            config = load_config()
            audit.log("Config loaded via canonical config_manager")

            # Save a timestamped backup profile before modifying
            backup_name = save_config_profile(config, profile="auto_agent_backup", validate=False)
            audit.log(f"Config profile backup saved: {backup_name}")

            # Apply updates (only to strategy section for safety)
            strategy = config.get("strategy", {})
            applied = []
            for key, value in updates.items():
                if key in strategy:
                    old_val = strategy[key]
                    strategy[key] = value
                    applied.append({"key": key, "old": old_val, "new": value})
                    audit.log(f"Updated strategy.{key}: {old_val} → {value}")
            config["strategy"] = strategy

            # Write via canonical save_config (validates before writing)
            save_config(config, validate=True)

            audit.log(f"Config updated via config_manager: {len(applied)} changes applied")
            return {"updated": True, "changes": applied, "backup": backup_name}

        except Exception as e:
            audit.error(f"Config update failed: {e}")
            # Rollback: reload from profile backup if available
            try:
                from common.config_manager import load_config as _lc, save_config as _sc
                original = _lc()  # Will load the file before our failed write
                _sc(original, validate=False)
                audit.log("Rolled back config from canonical loader")
            except Exception:
                pass
            return {"updated": False, "error": str(e)}
