# -*- coding: utf-8 -*-
from __future__ import annotations

"""
core/ml_analysis.py — Full-stack ML & AutoML Dashboard (v6.0, Part 1/6)
=======================================================================

This module provides a *hedge-fund grade* ML analysis engine for
Optuna results in the pairs-trading system.

Design (6 parts)
----------------
1. (THIS PART) Core infrastructure:
   - Imports & optional heavy deps (guarded)
   - Global config & logging
   - MLConfig / MLArtifacts dataclasses
   - Metrics helpers (RMSE, MAPE, Sharpe-like for regression)
   - Model & CV registries
   - Preprocessing builder (numeric / categorical)
   - Baseline `render_ml_analysis` that runs a solid ML workflow
     (model selection, CV, holdout metrics, simple residuals plot)

2. Feature-rich training & evaluation:
   - Extended metrics (MAPE, R^2, custom "Sharpe-style")
   - Better CV schemes (PurgedKFold when available)
   - Model comparison table across multiple models

3. Explainability:
   - SHAP (if installed), permutation importance heatmap, PDPs
   - Residual diagnostics & calibration

4. Scenario grid:
   - `st.data_editor` “what-if” panel
   - Multi-row prediction scenarios, with SHAP force plots

5. AutoML & Hyper-param Tuning:
   - Optuna-based HPO
   - PyCaret / H2O (if available)
   - Export pipeline (joblib), dataset → Parquet, Optuna study → pickle

6. Tracking & Reporting:
   - MLflow / Weights&Biases / Neptune toggles
   - Model card Markdown + optional PDF export
   - Compact API for programmatic usage (no Streamlit)

Part 1 is self-contained and production-ready. Later parts will
extend functionality by adding helpers and gently wrapping the
core behavior without breaking its public interface.
"""

###############################################################################
# Imports                                                                     #
###############################################################################
import hashlib
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt  # noqa: WPS433

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor  # type: ignore

# Optional heavy deps (lazy / guarded)
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:
    CatBoostRegressor = None  # type: ignore

try:
    import shap  # type: ignore
except Exception:
    shap = None  # type: ignore

try:
    import optuna  # type: ignore
except Exception:
    optuna = None  # type: ignore

try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None  # type: ignore

try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore

try:
    import neptune.new as neptune  # type: ignore
except Exception:
    neptune = None  # type: ignore

try:
    from mlfinlab.cross_validation import PurgedKFold  # type: ignore
except Exception:
    PurgedKFold = None  # type: ignore

# AutoML wrappers (optional but expected in this project)
try:
    from common.automl_tools import (  # type: ignore
        fairness_report,
        permutation_importance,
        run_pycaret_regression,
        shap_summary_plot,
    )
except Exception:
    fairness_report = None  # type: ignore
    permutation_importance = None  # type: ignore
    run_pycaret_regression = None  # type: ignore
    shap_summary_plot = None  # type: ignore

###############################################################################
# Globals & Config                                                            #
###############################################################################
logger = logging.getLogger("MLAnalysis")
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("ML_ANALYSIS_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

CUDA = False
try:
    import torch  # type: ignore

    CUDA = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
except Exception:
    CUDA = False

SEED = int(os.environ.get("ML_ANALYSIS_SEED", "42"))


###############################################################################
# Dataclasses & Type Helpers                                                  #
###############################################################################

@dataclass
class MLConfig:
    """
    High-level configuration for a single ML analysis run.

    This is populated from the Streamlit sidebar but can also be
    constructed programmatically (for tests and scripts).
    """

    model_name: str = "XGBoost"
    target_var: str = "Score"
    feature_cols: List[str] = field(default_factory=list)

    cv_type: str = "KFold"        # KFold / RepeatedKFold / TimeSeries / PurgedKFold
    cv_folds: int = 5
    holdout_pct: float = 0.2

    use_optuna_hpo: bool = False
    use_mlflow: bool = False
    use_wandb: bool = False
    use_neptune: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MLArtifacts:
    """
    Container for all key artifacts of a run:

        - config:   MLConfig used.
        - pipeline: fitted sklearn Pipeline (preprocess + model).
        - X_train / X_test / y_train / y_test.
        - preds:   predictions on holdout set.
        - metrics: primary metrics on holdout.
        - cv_scores: CV R^2 scores (per fold).
    """

    config: MLConfig
    pipeline: Optional[Pipeline] = None

    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None

    preds: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    cv_scores: Optional[np.ndarray] = None

    def to_perf_dict(self) -> Dict[str, Any]:
        out = {
            "config": self.config.to_dict(),
            "metrics": dict(self.metrics),
        }
        if self.cv_scores is not None:
            out["cv_mean_r2"] = float(self.cv_scores.mean())
            out["cv_std_r2"] = float(self.cv_scores.std())
        return out


###############################################################################
# Metrics helpers                                                             #
###############################################################################

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # noqa: WPS110
    """Root Mean Square Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # noqa: WPS110
    """Mean Absolute Percentage Error (ignoring inf/NaN)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs((y_true - y_pred) / y_true)
        out[np.isinf(out)] = np.nan
    return float(np.nanmean(out))


def regression_sharpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    A simple "Sharpe-like" metric for regression:

        s = mean(error) / std(error)

    This is *not* portfolio Sharpe, but can be helpful as a
    normalized error shape measure when comparing models.
    """
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    if len(err) < 2:
        return 0.0
    mu = float(np.mean(err))
    sd = float(np.std(err, ddof=0))
    if sd == 0.0 or np.isnan(sd):
        return 0.0
    return float(mu / sd)


def display_metrics_block(
    r2: float,
    rmse_val: float,
    mae_val: float,
    mape_val: float,
    sharpe_like: float,
) -> None:
    """Render a small metrics block in Streamlit (Hebrew-friendly)."""
    st.write("### 📊 Hold-out Metrics")
    st.markdown(
        f"- **R²:** `{r2:.3f}`\n"
        f"- **RMSE:** `{rmse_val:.4f}`\n"
        f"- **MAE:** `{mae_val:.4f}`\n"
        f"- **MAPE:** `{mape_val:.2%}`\n"
        f"- **Sharpe-like (errors):** `{sharpe_like:.3f}`"
    )


###############################################################################
# Model builders & registries                                                 #
###############################################################################

def build_ridge(**kw: Any) -> Ridge:
    return Ridge(random_state=SEED, **kw)


def build_elastic(**kw: Any) -> ElasticNet:
    return ElasticNet(random_state=SEED, **kw)


def build_rf(**kw: Any) -> RandomForestRegressor:
    base = dict(n_estimators=500, max_depth=8, random_state=SEED, n_jobs=-1)
    base.update(kw)
    return RandomForestRegressor(**base)


def build_gb(**kw: Any) -> GradientBoostingRegressor:
    base = dict(n_estimators=600, learning_rate=0.05, random_state=SEED)
    base.update(kw)
    return GradientBoostingRegressor(**base)


def build_xgb(**kw: Any) -> XGBRegressor:
    tm = "gpu_hist" if CUDA else "hist"
    base = dict(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        random_state=SEED,
        n_jobs=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method=tm,
        predictor="gpu_predictor" if CUDA else "auto",
        early_stopping_rounds=50,
    )
    base.update(kw)
    return XGBRegressor(**base)  # type: ignore[arg-type]


def build_lgbm(**kw: Any):
    if lgb is None:
        raise ModuleNotFoundError("lightgbm not installed")
    base = dict(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        objective="regression",
        random_state=SEED,
        n_jobs=-1,
    )
    base.update(kw)
    return lgb.LGBMRegressor(**base)  # type: ignore[arg-type]


def build_cat(**kw: Any):
    if CatBoostRegressor is None:
        raise ModuleNotFoundError("catboost not installed")
    base = dict(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        random_state=SEED,
        loss_function="RMSE",
        verbose=False,
    )
    base.update(kw)
    return CatBoostRegressor(**base)  # type: ignore[arg-type]


def build_meta_stack() -> StackingRegressor:
    estimators: List[Tuple[str, Any]] = []
    if lgb is not None:
        estimators.append(("lgb", build_lgbm()))
    if CatBoostRegressor is not None:
        estimators.append(("cat", build_cat()))
    estimators.append(("xgb", build_xgb()))
    final_est = build_elastic(alpha=0.05, l1_ratio=0.7)
    return StackingRegressor(estimators=estimators, final_estimator=final_est, n_jobs=-1)


MODEL_REGISTRY: Dict[str, Any] = {
    "Ridge": build_ridge,
    "ElasticNet": build_elastic,
    "RandomForest": build_rf,
    "GradientBoosting": build_gb,
    "XGBoost": build_xgb,
    "LightGBM": build_lgbm if lgb else None,
    "CatBoost": build_cat if CatBoostRegressor else None,
    "MetaStack": build_meta_stack,
    "SGD": lambda **_: SGDRegressor(random_state=SEED, max_iter=1000, tol=1e-3),
}

CV_REGISTRY: Dict[str, Any] = {
    "KFold": lambda k: KFold(n_splits=k, shuffle=True, random_state=SEED),
    "RepeatedKFold": lambda k: RepeatedKFold(n_splits=k, n_repeats=3, random_state=SEED),
    "TimeSeries": lambda k: TimeSeriesSplit(n_splits=k),
    "PurgedKFold": lambda k: PurgedKFold(n_splits=k) if PurgedKFold else KFold(n_splits=k),
}


###############################################################################
# Preprocessing & data validation                                             #
###############################################################################

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that standardizes numeric features and
    one-hot encodes categorical features.

    Returns:
        (transformer, num_cols, cat_cols)
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
    )
    return pre, num_cols, cat_cols


def _validate_and_prepare_data(
    df_results: pd.DataFrame,
    cfg: MLConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Ensure df_results contains target & selected features, apply basic
    NA cleaning (ffill/bfill) and return (X, y).
    """
    if df_results is None or df_results.empty:
        raise ValueError("df_results is empty — run optimisation first.")

    missing = [c for c in cfg.feature_cols if c not in df_results.columns]
    if missing:
        raise ValueError(f"Missing feature columns in df_results: {missing}")

    if cfg.target_var not in df_results.columns:
        raise ValueError(f"Target variable '{cfg.target_var}' not found in df_results.")

    X = df_results[cfg.feature_cols].copy()
    y = df_results[cfg.target_var].copy()

    # Simple but robust NA handling
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(method="ffill").fillna(method="bfill")

    return X, y


###############################################################################
# Core training routine (baseline, to be extended in later parts)             #
###############################################################################

def _run_core_ml_pipeline(
    df_results: pd.DataFrame,
    cfg: MLConfig,
) -> MLArtifacts:
    """
    Core ML flow used by Part 1:

        1. Validate & prepare X, y.
        2. Build preprocessing transformer.
        3. Train/test split.
        4. CV evaluation (R²).
        5. Fit model on train, evaluate on holdout.
        6. Return MLArtifacts container.
    """
    X_full, y_full = _validate_and_prepare_data(df_results, cfg)
    pre, num_cols, cat_cols = build_preprocessor(X_full)

    shuffle = cfg.cv_type not in {"TimeSeries", "PurgedKFold"}
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=cfg.holdout_pct,
        random_state=SEED,
        shuffle=shuffle,
    )

    builder = MODEL_REGISTRY.get(cfg.model_name)
    if builder is None:
        raise ValueError(f"Model '{cfg.model_name}' unavailable — missing dependency?")

    model = builder()
    pipe = Pipeline([("pre", pre), ("mdl", model)])

    cv_factory = CV_REGISTRY.get(cfg.cv_type)
    if cv_factory is None:
        raise ValueError(f"CV type '{cfg.cv_type}' not supported.")
    cv = cv_factory(cfg.cv_folds)

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    y_true = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(preds, dtype=float)

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "reg_sharpe": regression_sharpe(y_true, y_pred),
    }

    artifacts = MLArtifacts(
        config=cfg,
        pipeline=pipe,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preds=y_pred,
        metrics=metrics,
        cv_scores=cv_scores,
    )
    return artifacts


###############################################################################
# Streamlit UI entrypoint (baseline)                                          #
###############################################################################

def render_ml_analysis(df_results: pd.DataFrame, sel_params: List[str]) -> None:  # noqa: C901
    """
    Streamlit UI entry point (Part 1/6).

    Part 1 provides:
        - Sidebar configuration (model / target / CV / holdout %).
        - Basic ML run (single model).
        - CV R² + holdout metrics.
        - Residuals vs Predicted scatter.

    Later parts will extend this with:
        - Explainability panels (SHAP / permutation).
        - AutoML (PyCaret / H2O).
        - Scenario editor & advanced tracking.
    """
    st.header("🧠 ML & AutoML Analysis — Core (v6.0, Part 1/6)")

    if df_results is None or df_results.empty:
        st.warning("df_results is empty — run optimisation first.")
        return

    # ---------------- Sidebar: configuration ----------------
    st.sidebar.header("ML Settings")

    # Features come from sel_params (checked against df_results)
    if not sel_params:
        # fallback: all numeric params except obvious metrics
        blacklist = {"Score", "Sharpe", "Profit", "Drawdown"}
        sel_params = [
            c
            for c in df_results.columns
            if c not in blacklist and pd.api.types.is_numeric_dtype(df_results[c])
        ]

    target_var = st.sidebar.selectbox(
        "Target variable",
        options=[c for c in ("Score", "Profit", "Drawdown", "Sharpe") if c in df_results.columns],
        index=0,
    )

    model_choice = st.sidebar.selectbox("Model", [m for m, fn in MODEL_REGISTRY.items() if fn is not None], index=4)
    cv_type = st.sidebar.selectbox("CV type", list(CV_REGISTRY.keys()), index=0)
    cv_folds = int(st.sidebar.number_input("Folds", 3, 10, value=5))
    holdout_pct = float(st.sidebar.number_input("Hold-out %", 0.1, 0.4, value=0.2, step=0.05))

    # Build MLConfig
    cfg = MLConfig(
        model_name=model_choice,
        target_var=target_var,
        feature_cols=sel_params,
        cv_type=cv_type,
        cv_folds=cv_folds,
        holdout_pct=holdout_pct,
        use_optuna_hpo=False,
        use_mlflow=False,
        use_wandb=False,
        use_neptune=False,
    )

    # ---------------- Core ML run ----------------
    try:
        artifacts = _run_core_ml_pipeline(df_results, cfg)
    except Exception as e:
        st.error(f"ML run failed: {e}")
        logger.exception("ML run failed")
        return

    # CV scores
    if artifacts.cv_scores is not None:
        st.write(
            f"**CV R²:** {artifacts.cv_scores.mean():.3f} ± {artifacts.cv_scores.std():.3f} "
            f"({cfg.cv_type}, k={cfg.cv_folds})"
        )

    # Metrics block
    m = artifacts.metrics
    display_metrics_block(
        r2=m.get("r2", 0.0),
        rmse_val=m.get("rmse", 0.0),
        mae_val=m.get("mae", 0.0),
        mape_val=m.get("mape", 0.0),
        sharpe_like=m.get("reg_sharpe", 0.0),
    )

    # ---------------- Residuals vs Predicted ----------------
    st.subheader("Residuals Diagnostics")
    try:
        y_test = np.asarray(artifacts.y_test, dtype=float)
        y_pred = np.asarray(artifacts.preds, dtype=float)
        residuals = y_test - y_pred

        fig_res, ax_res = plt.subplots(figsize=(6, 4))
        ax_res.scatter(y_pred, residuals, alpha=0.6)
        ax_res.axhline(0.0, color="red", linestyle="--")
        ax_res.set_xlabel("Predicted")
        ax_res.set_ylabel("Residual")
        ax_res.set_title("Residuals vs Predicted")
        st.pyplot(fig_res)
    except Exception:
        st.info("Residual plot unavailable (insufficient data or plotting error).")

    # ---------------- Preview of feature set ----------------
    with st.expander("Feature sample (X_test head)", expanded=False):
        try:
            st.dataframe(
                artifacts.X_test.head(20),  # type: ignore[call-arg]
                width="stretch",
            )
        except Exception:
            st.info("No X_test preview available.")

    # In later parts we will add:
    # - Explainability panels (SHAP, permutation, PDP)
    # - AutoML integration
    # - Scenario editor
    # - Tracking (MLflow / W&B / Neptune)
# =========================
# PART 2/6: Model Comparison, Feature Profile & Importance (Pro)
# =========================

def _run_model_on_split(
    builder: Any,
    pre: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv,
    model_name: str,
) -> Tuple[MLArtifacts, Dict[str, float]]:
    """
    Run a single model on a fixed train/test split + CV object and
    return MLArtifacts + compact metrics dict for comparison table.
    """
    model = builder()
    pipe = Pipeline([("pre", pre), ("mdl", model)])

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    y_true = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(preds, dtype=float)

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "reg_sharpe": regression_sharpe(y_true, y_pred),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
    }

    cfg = MLConfig(
        model_name=model_name,
        target_var="",
        feature_cols=[],
        cv_type="",
        cv_folds=getattr(cv, "n_splits", 0),
        holdout_pct=0.0,
    )

    artifacts = MLArtifacts(
        config=cfg,
        pipeline=pipe,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preds=y_pred,
        metrics=metrics,
        cv_scores=cv_scores,
    )
    return artifacts, metrics


def _run_model_grid(
    df_results: pd.DataFrame,
    cfg: MLConfig,
    model_names: List[str],
) -> Tuple[pd.DataFrame, Dict[str, MLArtifacts]]:
    """
    Run a *grid* of models on the same split & CV and return:

        - comparison_df: DataFrame with per-model metrics.
        - artifacts_map: {model_name -> MLArtifacts}
    """
    # 1) Prepare data & preprocessor once
    X_full, y_full = _validate_and_prepare_data(df_results, cfg)
    pre, num_cols, cat_cols = build_preprocessor(X_full)

    shuffle = cfg.cv_type not in {"TimeSeries", "PurgedKFold"}
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=cfg.holdout_pct,
        random_state=SEED,
        shuffle=shuffle,
    )

    cv_factory = CV_REGISTRY.get(cfg.cv_type)
    if cv_factory is None:
        raise ValueError(f"CV type '{cfg.cv_type}' not supported.")
    cv = cv_factory(cfg.cv_folds)

    rows: List[Dict[str, Any]] = []
    artifacts_map: Dict[str, MLArtifacts] = {}

    for name in model_names:
        builder = MODEL_REGISTRY.get(name)
        if builder is None:
            logger.debug("Skipping model %s — unavailable/broken builder", name)
            continue
        try:
            art, metr = _run_model_on_split(
                builder,
                pre,
                X_train,
                X_test,
                y_train,
                y_test,
                cv,
                name,
            )
            artifacts_map[name] = art
            row = {
                "Model": name,
                "CV_R2_mean": metr["cv_r2_mean"],
                "CV_R2_std": metr["cv_r2_std"],
                "Holdout_R2": metr["r2"],
                "RMSE": metr["rmse"],
                "MAE": metr["mae"],
                "MAPE": metr["mape"],
                "Sharpe_like": metr["reg_sharpe"],
            }
            rows.append(row)
        except Exception as e:
            logger.warning("Model %s failed in comparison grid: %s", name, e)
            rows.append(
                {
                    "Model": name,
                    "CV_R2_mean": np.nan,
                    "CV_R2_std": np.nan,
                    "Holdout_R2": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "MAPE": np.nan,
                    "Sharpe_like": np.nan,
                }
            )

    if not rows:
        return pd.DataFrame(), artifacts_map

    comp_df = pd.DataFrame(rows)
    comp_df = comp_df.sort_values("Holdout_R2", ascending=False)
    return comp_df, artifacts_map


def _render_feature_profile(df_results: pd.DataFrame, feature_cols: List[str], target: str) -> None:
    """
    Render a small "feature profile" panel:
        - target distribution
        - correlation matrix of features + target
    """
    with st.expander("🔍 Feature & Target Profile", expanded=False):
        try:
            if target in df_results.columns:
                st.write("**Target distribution (histogram)**")
                st.bar_chart(df_results[target].dropna())
        except Exception:
            st.info("Target histogram unavailable.")

        try:
            cols = [c for c in feature_cols if c in df_results.columns]
            if target in df_results.columns:
                cols = cols + [target]
            if len(cols) >= 2:
                corr = df_results[cols].corr(numeric_only=True)
                st.write("**Correlation matrix (features + target)**")
                st.dataframe(corr, width="stretch")
            else:
                st.info("Not enough numeric columns for correlation matrix.")
        except Exception:
            st.info("Correlation matrix unavailable.")


def _render_feature_importance_panel(
    artifacts: MLArtifacts,
    feature_cols: List[str],
) -> None:
    """
    Feature importance via permutation_importance (from common.automl_tools
    if available, otherwise sklearn.inspection.permutation_importance).

    We keep it light and generic: importance on X_test with respect to y_test.
    """
    with st.expander("📌 Feature Importance (Permutation)", expanded=False):
        try:
            if artifacts.pipeline is None or artifacts.X_test is None or artifacts.y_test is None:
                st.info("No fitted pipeline / test set available.")
                return

            pipe = artifacts.pipeline
            X_test = artifacts.X_test
            y_test = artifacts.y_test

            if permutation_importance is not None:
                # Project-specific wrapper
                pi = permutation_importance(pipe, X_test, y_test, n_repeats=15, random_state=SEED)
                imp_df = pd.DataFrame(
                    {
                        "feature": feature_cols,
                        "importance": np.asarray(pi).astype(float).ravel()[: len(feature_cols)],
                    }
                )
            else:
                # Fallback to sklearn
                try:
                    from sklearn.inspection import permutation_importance as skl_perm  # type: ignore
                except Exception:
                    st.info("permutation_importance unavailable (neither project wrapper nor sklearn).")
                    return

                res = skl_perm(pipe, X_test, y_test, n_repeats=10, random_state=SEED)
                # When using ColumnTransformer+OneHotEncoder, the feature space can explode.
                # For a simple approximation, we aggregate importances per original column
                importances = res.importances_mean
                imp_df = pd.DataFrame(
                    {
                        "feature": [f"f{i}" for i in range(len(importances))],
                        "importance": importances,
                    }
                )

            imp_df = imp_df.sort_values("importance", ascending=False)
            st.dataframe(imp_df.head(30), width="stretch")

            try:
                st.bar_chart(
                    imp_df.set_index("feature").head(20),
                    width="stretch",
                )
            except Exception:
                pass
        except Exception as e:
            st.info(f"Feature importance unavailable: {e}")


# Keep reference to the core (Part 1) implementation
_RENDER_ML_CORE = render_ml_analysis


def render_ml_analysis(df_results: pd.DataFrame, sel_params: List[str]) -> None:  # noqa: C901
    """
    Streamlit UI entry point (Part 2/6 — upgraded).

    הרחבות מעבר לחלק 1:
        - פרופיל פיצ'רים + מטרת מודל (התפלגויות + קורלציות).
        - השוואת מודלים (Model Comparison Grid) על אותו split.
        - חשיבות פיצ'רים (Permutation Importance) על ה-holdout.
        - עדיין שומר על API זהה: render_ml_analysis(df_results, sel_params).

    השלד הבסיסי (מודל יחיד + CV + Holdout + Residuals) נשמר,
    אך עטוף בסביבת UI עשירה יותר.
    """
    st.header("🧠 ML & AutoML Analysis — Pro (v6.0, Part 2/6)")

    if df_results is None or df_results.empty:
        st.warning("df_results is empty — run optimisation first.")
        return

    # ---------------- Sidebar: configuration ----------------
    st.sidebar.header("ML Settings")

    # Features בסיס: sel_params או fallback לנומריים
    if not sel_params:
        blacklist = {"Score", "Sharpe", "Profit", "Drawdown"}
        sel_params = [
            c
            for c in df_results.columns
            if c not in blacklist and pd.api.types.is_numeric_dtype(df_results[c])
        ]

    target_candidates = [c for c in ("Score", "Profit", "Drawdown", "Sharpe") if c in df_results.columns]
    if not target_candidates:
        target_candidates = [c for c in df_results.columns if pd.api.types.is_numeric_dtype(df_results[c])]

    target_var = st.sidebar.selectbox(
        "Target variable",
        options=target_candidates,
        index=0,
    )

    # מודל ראשי
    available_models = [m for m, fn in MODEL_REGISTRY.items() if fn is not None]
    default_model_index = available_models.index("XGBoost") if "XGBoost" in available_models else 0
    model_choice = st.sidebar.selectbox("Primary model", available_models, index=default_model_index)

    # מודלים להשוואה
    compare_models = st.sidebar.multiselect(
        "Models to compare",
        options=available_models,
        default=list({model_choice, "RandomForest", "GradientBoosting", "XGBoost"} & set(available_models)),
    )

    cv_type = st.sidebar.selectbox("CV type", list(CV_REGISTRY.keys()), index=0)
    cv_folds = int(st.sidebar.number_input("Folds", 3, 10, value=5))
    holdout_pct = float(st.sidebar.number_input("Hold-out %", 0.1, 0.4, value=0.2, step=0.05))

    # ---------------- MLConfig ----------------
    cfg = MLConfig(
        model_name=model_choice,
        target_var=target_var,
        feature_cols=sel_params,
        cv_type=cv_type,
        cv_folds=cv_folds,
        holdout_pct=holdout_pct,
    )

    # ---------------- Feature profile ----------------
    _render_feature_profile(df_results, sel_params, target_var)

    # ---------------- Core ML run for primary model ----------------
    st.subheader(f"Primary Model: {model_choice}")
    try:
        artifacts = _run_core_ml_pipeline(df_results, cfg)
    except Exception as e:
        st.error(f"Primary ML run failed: {e}")
        logger.exception("Primary ML run failed")
        return

    # CV scores
    if artifacts.cv_scores is not None:
        st.write(
            f"**CV R² ({cfg.cv_type}, k={cfg.cv_folds}):** "
            f"{artifacts.cv_scores.mean():.3f} ± {artifacts.cv_scores.std():.3f}"
        )

    # Metrics block
    m = artifacts.metrics
    display_metrics_block(
        r2=m.get("r2", 0.0),
        rmse_val=m.get("rmse", 0.0),
        mae_val=m.get("mae", 0.0),
        mape_val=m.get("mape", 0.0),
        sharpe_like=m.get("reg_sharpe", 0.0),
    )

    # Residuals
    st.subheader("Residuals Diagnostics — Primary Model")
    try:
        y_test = np.asarray(artifacts.y_test, dtype=float)
        y_pred = np.asarray(artifacts.preds, dtype=float)
        residuals = y_test - y_pred

        fig_res, ax_res = plt.subplots(figsize=(6, 4))
        ax_res.scatter(y_pred, residuals, alpha=0.6)
        ax_res.axhline(0.0, color="red", linestyle="--")
        ax_res.set_xlabel("Predicted")
        ax_res.set_ylabel("Residual")
        ax_res.set_title("Residuals vs Predicted")
        st.pyplot(fig_res)
    except Exception:
        st.info("Residual plot unavailable (insufficient data or plotting error).")

    with st.expander("X_test sample (primary model)", expanded=False):
        try:
            st.dataframe(artifacts.X_test.head(20), width="stretch")  # type: ignore[arg-type]
        except Exception:
            st.info("No X_test preview available.")

    # ---------------- Model comparison grid ----------------
    if compare_models:
        st.subheader("Model Comparison Grid")
        try:
            comp_df, artifacts_map = _run_model_grid(df_results, cfg, compare_models)
            if comp_df.empty:
                st.info("Could not compute comparison grid (no successful models).")
            else:
                st.dataframe(comp_df, width="stretch")
                try:
                    st.bar_chart(
                        comp_df.set_index("Model")["Holdout_R2"],
                        width="stretch",
                    )
                except Exception:
                    pass
        except Exception as e:
            st.info(f"Model comparison failed: {e}")
    else:
        artifacts_map = {cfg.model_name: artifacts}

    # ---------------- Feature importance for primary model ----------------
    _render_feature_importance_panel(artifacts, cfg.feature_cols)

    # =========================
# PART 3/6: Explainability Lab & Advanced Diagnostics (Hyper Pro)
# =========================

# Optional extra stats
try:
    import scipy.stats as _sps  # type: ignore
except Exception:
    _sps = None  # type: ignore


def _render_residual_diagnostics_advanced(artifacts: MLArtifacts) -> None:
    """
    Advanced residual diagnostics:
        - Histogram of residuals
        - QQ-plot vs Normal (אם scipy זמין)
        - Actual vs Predicted scatter
        - Residuals vs Time (אם יש אינדקס דמוי זמן)
        - Binned error analysis לפי deciles של prediction
    """
    with st.expander("📉 Residual Diagnostics — Advanced", expanded=False):
        if artifacts.y_test is None or artifacts.preds is None:
            st.info("Residual diagnostics unavailable — no test set/predictions.")
            return

        y_true = np.asarray(artifacts.y_test, dtype=float)
        y_pred = np.asarray(artifacts.preds, dtype=float)
        if len(y_true) != len(y_pred) or len(y_true) == 0:
            st.info("Residual diagnostics unavailable — incompatible sizes.")
            return

        residuals = y_true - y_pred

        # Summary stats
        try:
            st.markdown(
                f"- **mean(resid)** = `{float(residuals.mean()):.4f}`  "
                f"- **std(resid)** = `{float(residuals.std(ddof=0)):.4f}`  "
                f"- **skew** = `{float(pd.Series(residuals).skew()):.3f}`  "
                f"- **kurtosis** = `{float(pd.Series(residuals).kurtosis()):.3f}`"
            )
        except Exception:
            pass

        c1, c2 = st.columns(2)

        # Histogram
        with c1:
            try:
                fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
                ax_hist.hist(residuals, bins=30, alpha=0.7)
                ax_hist.set_title("Residuals Histogram")
                ax_hist.set_xlabel("Residual")
                ax_hist.set_ylabel("Frequency")
                st.pyplot(fig_hist)
            except Exception:
                st.info("Could not plot residual histogram.")

        # QQ-plot vs Normal
        with c2:
            if _sps is None:
                st.info("scipy not available — QQ-plot disabled.")
            else:
                try:
                    fig_qq, ax_qq = plt.subplots(figsize=(5, 4))
                    _sps.probplot(residuals, dist="norm", plot=ax_qq)
                    ax_qq.set_title("Residuals QQ-Plot (Normal)")
                    st.pyplot(fig_qq)
                except Exception:
                    st.info("Could not compute QQ-plot.")

        # Actual vs Predicted
        try:
            fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
            ax_sc.scatter(y_true, y_pred, alpha=0.6)
            lo = min(np.min(y_true), np.min(y_pred))
            hi = max(np.max(y_true), np.max(y_pred))
            ax_sc.plot([lo, hi], [lo, hi], color="red", linestyle="--")
            ax_sc.set_xlabel("Actual")
            ax_sc.set_ylabel("Predicted")
            ax_sc.set_title("Actual vs Predicted")
            st.pyplot(fig_sc)
        except Exception:
            st.info("Actual vs Predicted plot unavailable.")

        # Residuals vs Time (אם יש אינדקס)
        try:
            idx = getattr(artifacts.y_test, "index", None)
            if idx is not None and len(idx) == len(residuals):
                fig_t, ax_t = plt.subplots(figsize=(6, 3))
                ax_t.plot(idx, residuals, marker=".", linestyle="-", alpha=0.7)
                ax_t.axhline(0.0, color="red", linestyle="--", linewidth=1)
                ax_t.set_title("Residuals Over Index")
                ax_t.set_ylabel("Residual")
                st.pyplot(fig_t)
        except Exception:
            pass

        # Binned error analysis לפי deciles של predicted
        try:
            s_true = pd.Series(y_true, name="y_true")
            s_pred = pd.Series(y_pred, name="y_pred")
            df_err = pd.DataFrame({"y_true": s_true, "y_pred": s_pred})
            df_err["resid"] = df_err["y_true"] - df_err["y_pred"]
            df_err["bin"] = pd.qcut(df_err["y_pred"], q=10, duplicates="drop")
            grp = df_err.groupby("bin")["resid"].agg(["mean", "std", "count"])
            st.write("**Binned residual stats by predicted deciles**")
            st.dataframe(grp, width="stretch")
        except Exception:
            st.info("Binned residual analysis unavailable.")


def _render_error_vs_feature_panel(
    artifacts: MLArtifacts,
    feature_cols: List[str],
) -> None:
    """
    Error vs Feature Explorer:
        - בוחר פיצ'ר מתוך feature_cols שקיים ב-X_test.
        - מציג scatter של residual vs feature.
        - מחשב מתאם Pearson/Spearman בין השגיאה לבין הפיצ'ר.
    """
    with st.expander("🔎 Error vs Feature Explorer", expanded=False):
        if artifacts.X_test is None or artifacts.y_test is None or artifacts.preds is None:
            st.info("No X_test / predictions for error explorer.")
            return

        X_test = artifacts.X_test
        y_true = np.asarray(artifacts.y_test, dtype=float)
        y_pred = np.asarray(artifacts.preds, dtype=float)
        resid = y_true - y_pred

        candidates = [c for c in feature_cols if c in X_test.columns]
        if not candidates:
            st.info("No matching features from feature_cols in X_test.")
            return

        feat = st.selectbox("Choose feature", candidates, index=0, key="err_feat_sel")
        x = pd.to_numeric(X_test[feat], errors="coerce")
        mask = x.notna()
        x = x[mask]
        r = resid[mask.values]

        if len(r) == 0:
            st.info("No valid data points for chosen feature.")
            return

        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(x, r, alpha=0.6)
            ax.axhline(0.0, color="red", linestyle="--")
            ax.set_xlabel(feat)
            ax.set_ylabel("Residual")
            ax.set_title(f"Residual vs {feat}")
            st.pyplot(fig)
        except Exception:
            st.info("Could not plot residual vs feature.")

        # Correlation stats
        try:
            pear = float(pd.Series(x).corr(pd.Series(r), method="pearson"))
        except Exception:
            pear = float("nan")
        try:
            spear = float(pd.Series(x).corr(pd.Series(r), method="spearman"))
        except Exception:
            spear = float("nan")

        st.markdown(
            f"- **Pearson corr(resid, {feat})** = `{pear:.3f}`\n"
            f"- **Spearman corr(resid, {feat})** = `{spear:.3f}`"
        )


def _get_shap_feature_names_from_pre(pre: Any, original_cols: List[str]) -> Optional[List[str]]:
    """
    Try to infer feature names *after* ColumnTransformer + OneHotEncoder.

    If we can't reliably infer them, return None and we'll fall back to
    anonymous indexing in SHAP plots.
    """
    try:
        # Newer sklearn: ColumnTransformer.get_feature_names_out
        if hasattr(pre, "get_feature_names_out"):
            names = pre.get_feature_names_out()
            return [str(n) for n in names]
    except Exception:
        pass

    # Fallback: רק אם אין OHE ננסה להחזיר original_cols
    try:
        if isinstance(pre, ColumnTransformer):
            for name, trans, cols in pre.transformers_:
                if name == "remainder":
                    continue
                if isinstance(trans, OneHotEncoder):
                    return None
            return original_cols
    except Exception:
        pass

    return None


def _render_shap_panel(
    artifacts: MLArtifacts,
    feature_cols: List[str],
) -> None:
    """
    SHAP explainability panel (אם shap מותקן):

        - עובד מול pipeline: pre + mdl
        - משתמש במדגם מ-X_test (עד 500 נקודות)
        - מנסה להוציא feature names אחרי preprocessor, ואם לא מצליח — עובד עם אינדקסים.
    """
    with st.expander("🧩 SHAP Explainability (Experimental)", expanded=False):
        if shap is None:
            st.info("SHAP not installed — `pip install shap` to enable.")
            return

        if artifacts.pipeline is None or artifacts.X_test is None:
            st.info("No fitted pipeline/X_test available.")
            return

        pipe = artifacts.pipeline
        X_test = artifacts.X_test

        # Try to access named steps
        try:
            pre = pipe.named_steps.get("pre")  # type: ignore[union-attr]
            mdl = pipe.named_steps.get("mdl")  # type: ignore[union-attr]
        except Exception:
            pre = None
            mdl = pipe

        # Sample for performance
        try:
            if len(X_test) > 500:
                X_explain = X_test.sample(n=500, random_state=SEED)
            else:
                X_explain = X_test.copy()
        except Exception:
            X_explain = X_test

        # Transform features if we have a preprocessor
        try:
            if pre is not None:
                X_for_shap = pre.transform(X_explain)
            else:
                X_for_shap = X_explain
        except Exception as e:
            st.info(f"Preprocessing for SHAP failed: {e}")
            return

        # Try to infer feature names after preprocessing
        feat_names = _get_shap_feature_names_from_pre(pre, feature_cols) if pre is not None else feature_cols

        # Build explainer
        try:
            is_tree_like = any(
                s in mdl.__class__.__name__.lower()
                for s in ("randomforest", "gradientboosting", "xgb", "lgbm", "catboost")
            )
            if is_tree_like:
                explainer = shap.TreeExplainer(mdl)  # type: ignore[arg-type]
            else:
                explainer = shap.Explainer(mdl, X_for_shap)  # type: ignore[arg-type]
            shap_values = explainer(X_for_shap)
        except Exception as e:
            st.info(f"SHAP explainer failed: {e}")
            return

        # Summary plot (beeswarm)
        try:
            st.write("**SHAP Summary Plot (Beeswarm)**")
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.summary_plot(
                shap_values.values,  # type: ignore[arg-type]
                X_for_shap,
                show=False,
                feature_names=feat_names,
            )
            st.pyplot(fig)
            plt.clf()
        except Exception:
            st.info("SHAP summary plot unavailable.")

        # Bar plot of mean |SHAP|
        try:
            vals = np.abs(shap_values.values).mean(axis=0)  # type: ignore[arg-type]
            idxs = np.argsort(vals)[::-1]
            top_k = min(20, len(vals))
            vals = vals[idxs][:top_k]
            if feat_names is not None and len(feat_names) == len(shap_values.values[0]):  # type: ignore[index]
                names = np.array(feat_names)[idxs][:top_k]
            else:
                names = np.array([f"f{i}" for i in idxs[:top_k]])

            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            ax_bar.barh(range(top_k), vals[::-1])
            ax_bar.set_yticks(range(top_k))
            ax_bar.set_yticklabels(names[::-1])
            ax_bar.set_title("Mean |SHAP| (Top Features)")
            ax_bar.set_xlabel("Mean |SHAP|")
            st.pyplot(fig_bar)
        except Exception:
            st.info("SHAP importance bar plot failed.")


def _render_partial_dependence_panel(
    artifacts: MLArtifacts,
    feature_cols: List[str],
) -> None:
    """
    Partial Dependence / ICE-like panel (single & pairwise features),
    using sklearn.inspection.PartialDependenceDisplay when available.

    - מציג PDP ל־1–3 פיצ'רים (נבחרים לפי variance).
    - מציע גם PDP דו-ממדי (pairwise) אם יש לפחות 2 פיצ'רים.
    """
    with st.expander("📈 Partial Dependence (PDP / ICE)", expanded=False):
        if artifacts.pipeline is None or artifacts.X_test is None or not feature_cols:
            st.info("No pipeline/X_test/features available for partial dependence.")
            return

        try:
            from sklearn.inspection import PartialDependenceDisplay  # type: ignore
        except Exception:
            st.info("sklearn.inspection.PartialDependenceDisplay not available.")
            return

        pipe = artifacts.pipeline
        X_test = artifacts.X_test

        try:
            X_num = X_test[feature_cols].select_dtypes(include=[np.number])
            if X_num.empty:
                st.info("No numeric features for partial dependence.")
                return
            var = X_num.var().sort_values(ascending=False)
            top_features = var.index.tolist()[:3]
        except Exception:
            top_features = feature_cols[:3]

        # Single-feature PDP
        for feat in top_features:
            if feat not in X_test.columns:
                continue
            try:
                st.write(f"**Partial dependence for `{feat}`**")
                fig, ax = plt.subplots(figsize=(6, 4))
                PartialDependenceDisplay.from_estimator(pipe, X_test, [feat], ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.info(f"Partial dependence for {feat} failed: {e}")

        # Pairwise PDP (אם יש לפחות 2 פיצ'רים)
        if len(top_features) >= 2:
            pair = top_features[:2]
            try:
                st.write(f"**2D PDP for `{pair[0]}` & `{pair[1]}`**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                PartialDependenceDisplay.from_estimator(pipe, X_test, [pair], ax=ax2)
                st.pyplot(fig2)
            except Exception as e:
                st.info(f"2D PDP for {pair} failed: {e}")


def _render_global_metrics_summary(
    artifacts: MLArtifacts,
    cfg: MLConfig,
) -> None:
    """
    Global metrics summary table + download option.
    """
    with st.expander("📜 Global Metrics Summary & Export", expanded=False):
        m = artifacts.metrics
        rows = [
            {"Metric": "R²", "Value": m.get("r2", 0.0)},
            {"Metric": "RMSE", "Value": m.get("rmse", 0.0)},
            {"Metric": "MAE", "Value": m.get("mae", 0.0)},
            {"Metric": "MAPE", "Value": m.get("mape", 0.0)},
            {"Metric": "Sharpe-like(errors)", "Value": m.get("reg_sharpe", 0.0)},
        ]
        if artifacts.cv_scores is not None:
            rows.append(
                {
                    "Metric": f"CV R² ({cfg.cv_type}, k={cfg.cv_folds})",
                    "Value": float(artifacts.cv_scores.mean()),
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        try:
            st.download_button(
                "Download metrics.csv",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"ml_metrics_{cfg.model_name}.csv",
                mime="text/csv",
            )
        except Exception:
            pass


# נשמור רפרנס לגרסת Part 2 למקרה שנרצה להשתמש בה בעתיד
_RENDER_ML_PRO_V2 = render_ml_analysis


def render_ml_analysis(df_results: pd.DataFrame, sel_params: List[str]) -> None:  # noqa: C901
    """
    Streamlit UI entry point (Part 3/6 — Explainability Lab & Advanced Diagnostics).

    מרחיב את חלק 2:
        - פרופיל פיצ'רים + מטרת מודל.
        - מודל ראשי + השוואת מודלים.
        - Feature importance (Permutation).
        - אבחון שאריות מתקדם (Histogram, QQ, Actual vs Pred, Time, Binned).
        - Error vs Feature Explorer.
        - SHAP Explainability (אם מותקן).
        - Partial Dependence (כולל 2D pairwise).
        - Global Metrics Summary & Export.

    עדיין שומר על API זהה:
        render_ml_analysis(df_results, sel_params)
    """
    st.header("🧠 ML & AutoML Analysis — Explainability Lab (v6.0, Part 3/6)")

    if df_results is None or df_results.empty:
        st.warning("df_results is empty — run optimisation first.")
        return

    # ---------------- Sidebar: configuration ----------------
    st.sidebar.header("ML Settings")

    if not sel_params:
        blacklist = {"Score", "Sharpe", "Profit", "Drawdown"}
        sel_params = [
            c
            for c in df_results.columns
            if c not in blacklist and pd.api.types.is_numeric_dtype(df_results[c])
        ]

    target_candidates = [c for c in ("Score", "Profit", "Drawdown", "Sharpe") if c in df_results.columns]
    if not target_candidates:
        target_candidates = [c for c in df_results.columns if pd.api.types.is_numeric_dtype(df_results[c])]

    target_var = st.sidebar.selectbox(
        "Target variable",
        options=target_candidates,
        index=0,
    )

    available_models = [m for m, fn in MODEL_REGISTRY.items() if fn is not None]
    default_model_index = available_models.index("XGBoost") if "XGBoost" in available_models else 0
    model_choice = st.sidebar.selectbox("Primary model", available_models, index=default_model_index)

    compare_models = st.sidebar.multiselect(
        "Models to compare",
        options=available_models,
        default=list({model_choice, "RandomForest", "GradientBoosting", "XGBoost"} & set(available_models)),
    )

    cv_type = st.sidebar.selectbox("CV type", list(CV_REGISTRY.keys()), index=0)
    cv_folds = int(st.sidebar.number_input("Folds", 3, 10, value=5))
    holdout_pct = float(st.sidebar.number_input("Hold-out %", 0.1, 0.4, value=0.2, step=0.05))

    cfg = MLConfig(
        model_name=model_choice,
        target_var=target_var,
        feature_cols=sel_params,
        cv_type=cv_type,
        cv_folds=cv_folds,
        holdout_pct=holdout_pct,
    )

    # ---------------- Feature profile ----------------
    _render_feature_profile(df_results, sel_params, target_var)

    # ---------------- Core ML run — primary model ----------------
    st.subheader(f"Primary Model: {model_choice}")
    try:
        artifacts = _run_core_ml_pipeline(df_results, cfg)
    except Exception as e:
        st.error(f"Primary ML run failed: {e}")
        logger.exception("Primary ML run failed")
        return

    if artifacts.cv_scores is not None:
        st.write(
            f"**CV R² ({cfg.cv_type}, k={cfg.cv_folds}):** "
            f"{artifacts.cv_scores.mean():.3f} ± {artifacts.cv_scores.std():.3f}"
        )

    m = artifacts.metrics
    display_metrics_block(
        r2=m.get("r2", 0.0),
        rmse_val=m.get("rmse", 0.0),
        mae_val=m.get("mae", 0.0),
        mape_val=m.get("mape", 0.0),
        sharpe_like=m.get("reg_sharpe", 0.0),
    )

    # Global metrics summary + export
    _render_global_metrics_summary(artifacts, cfg)

    # Basic residuals (scatter) – כמו בחלק 1/2
    st.subheader("Residuals Diagnostics — Basic")
    try:
        y_test = np.asarray(artifacts.y_test, dtype=float)
        y_pred = np.asarray(artifacts.preds, dtype=float)
        residuals = y_test - y_pred

        fig_res, ax_res = plt.subplots(figsize=(6, 4))
        ax_res.scatter(y_pred, residuals, alpha=0.6)
        ax_res.axhline(0.0, color="red", linestyle="--")
        ax_res.set_xlabel("Predicted")
        ax_res.set_ylabel("Residual")
        ax_res.set_title("Residuals vs Predicted")
        st.pyplot(fig_res)
    except Exception:
        st.info("Basic residual plot unavailable.")

    with st.expander("X_test sample (primary model)", expanded=False):
        try:
            st.dataframe(artifacts.X_test.head(20), width="stretch")  # type: ignore[arg-type]
        except Exception:
            st.info("No X_test preview available.")

    # ---------------- Model comparison grid ----------------
    if compare_models:
        st.subheader("Model Comparison Grid")
        try:
            comp_df, artifacts_map = _run_model_grid(df_results, cfg, compare_models)
            if comp_df.empty:
                st.info("Could not compute comparison grid (no successful models).")
            else:
                st.dataframe(comp_df, width="stretch")
                try:
                    st.bar_chart(
                        comp_df.set_index("Model")["Holdout_R2"],
                        width="stretch",
                    )
                except Exception:
                    pass
        except Exception as e:
            st.info(f"Model comparison failed: {e}")
            artifacts_map = {cfg.model_name: artifacts}
    else:
        artifacts_map = {cfg.model_name: artifacts}

    # ---------------- Feature importance (Permutation) ----------------
    _render_feature_importance_panel(artifacts, cfg.feature_cols)

    # ---------------- Advanced residual diagnostics ----------------
    _render_residual_diagnostics_advanced(artifacts)

    # ---------------- Error vs Feature Explorer ----------------
    _render_error_vs_feature_panel(artifacts, cfg.feature_cols)

    # ---------------- SHAP Explainability (אם זמין) ----------------
    _render_shap_panel(artifacts, cfg.feature_cols)

    # ---------------- Partial Dependence (PDP/ICE) ----------------
    _render_partial_dependence_panel(artifacts, cfg.feature_cols)

   # =========================
# PART 4/6: Scenario Editor, AutoML (Optional) & Tracking Lab
# =========================

def _render_scenario_editor(
    df_results: pd.DataFrame,
    artifacts: MLArtifacts,
    cfg: MLConfig,
) -> None:
    """
    Scenario Editor (What-if) — אופציונלי, אבל משולב ברמה מקצועית:

    יכולות:
    --------
    - בחירת מקור לסנריו:
        • "X_test" (ברירת מחדל)
        • "X_train"
        • "Top-K by target" מתוך df_results (למשל top 100 Score)
    - עריכת טבלה (st.data_editor): הוספה/מחיקה/שינוי.
    - הרצת תחזיות על הסנריוים.
    - הורדה כ-CSV.

    זה רץ רק אם המשתמש בחר enable_scenarios בסיידבר.
    """
    with st.expander("🧪 Scenario Editor (What-if Analysis)", expanded=False):
        if artifacts.pipeline is None:
            st.info("No fitted pipeline available for scenarios.")
            return

        pipe = artifacts.pipeline

        st.markdown("בחר בסיס לסנריוים:")

        base_source = st.selectbox(
            "Scenario base source",
            ["X_test", "X_train", "Top-K by target"],
            index=0,
            key="ml_scen_base_src",
        )

        base_X: Optional[pd.DataFrame] = None

        if base_source == "X_test":
            base_X = artifacts.X_test
        elif base_source == "X_train":
            base_X = artifacts.X_train
        else:  # Top-K by target
            try:
                k_top = int(st.number_input("Top-K rows by target", 20, 1000, value=200, step=20))
                if cfg.target_var in df_results.columns:
                    df_sorted = df_results.sort_values(cfg.target_var, ascending=False)
                else:
                    df_sorted = df_results
                base_X = df_sorted[cfg.feature_cols].dropna().head(k_top)
            except Exception as e:
                st.info(f"Could not build Top-K base: {e}")
                base_X = None

        if base_X is None or base_X.empty:
            st.info("No base data available to seed scenarios.")
            return

        st.caption(
            "ערוך את הטבלה, הוסף/מחק שורות – ואז לחץ **Run Scenarios** כדי לקבל תחזיות. "
            "המודל שמופעל הוא המודל הראשי שנבחר למעלה."
        )

        try:
            base_sample = base_X.head(10).reset_index(drop=True)
        except Exception:
            base_sample = base_X.reset_index(drop=True)

        edited = st.data_editor(
            base_sample,
            num_rows="dynamic",
            width="stretch",
            key="ml_scenario_editor",
        )

        if edited is None or edited.empty:
            st.info("No scenarios defined yet.")
            return

        run_scen = st.button("🚀 Run Scenarios", key="ml_run_scenarios")
        if not run_scen:
            return

        try:
            scen_df = edited.copy()
            scen_preds = pipe.predict(scen_df)
            scen_df[cfg.target_var + "_pred"] = np.asarray(scen_preds, dtype=float)

            st.subheader("Scenario Results")
            st.dataframe(scen_df, width="stretch")

            # השוואה מול התפלגות אמיתית אם קיימת
            try:
                if cfg.target_var in df_results.columns:
                    fig_cmp, ax_cmp = plt.subplots(figsize=(6, 4))
                    ax_cmp.hist(
                        df_results[cfg.target_var].dropna(),
                        bins=40,
                        alpha=0.5,
                        label="historical",
                    )
                    ax_cmp.hist(
                        scen_df[cfg.target_var + "_pred"],
                        bins=40,
                        alpha=0.5,
                        label="scenarios",
                    )
                    ax_cmp.set_title(f"Distribution: {cfg.target_var} — Historical vs Scenarios")
                    ax_cmp.legend()
                    st.pyplot(fig_cmp)
            except Exception:
                pass

            try:
                st.download_button(
                    "Download scenarios.csv",
                    data=scen_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"scenarios_{cfg.model_name}.csv",
                    mime="text/csv",
                    key="ml_scen_dl",
                )
            except Exception:
                pass
        except Exception as e:
            st.error(f"Scenario prediction failed: {e}")


def _render_automl_panel(
    df_results: pd.DataFrame,
    cfg: MLConfig,
) -> None:
    """
    AutoML panel (PyCaret) — אופציונלי, מופעל ע"י checkbox בסיידבר.

    הנחות:
    -------
    - run_pycaret_regression מחובר ל-PyCaret בצד שלך.
    - אנחנו לוקחים subset מדגמי (max_rows) כדי לשמור על זמן ריצה.

    פאנל זה *לא* מחליף את המודל הראשי — הוא נותן Benchmark נוסף.
    """
    with st.expander("🤖 AutoML (PyCaret) — Optional", expanded=False):
        if run_pycaret_regression is None:
            st.info(
                "AutoML (PyCaret) integration not available — "
                "`common.automl_tools.run_pycaret_regression` missing. "
                "חבר אותו כשתרצה."
            )
            return

        try:
            cols = list(dict.fromkeys(cfg.feature_cols + [cfg.target_var]))
            data = df_results[cols].dropna()
        except Exception as e:
            st.info(f"AutoML data prep failed: {e}")
            return

        if data.empty:
            st.info("No valid rows (after dropping NA) for AutoML.")
            return

        st.caption(
            "AutoML ירוץ על תת-קבוצה מהדאטה (לטובת מהירות) "
            "ויחזיר Leaderboard של דגמים. "
            "כך תוכל להשוות את המודל הידני למודל 'אוטומטי'."
        )

        max_rows = int(st.number_input("Max rows for AutoML sample", 200, 10000, value=1500, step=200))
        sample_mode = st.selectbox(
            "Sampling mode for AutoML",
            ["Head (first rows)", "Random sample"],
            index=0,
            key="ml_automl_sample_mode",
        )

        if sample_mode == "Random sample":
            data_small = data.sample(n=min(max_rows, len(data)), random_state=SEED)
        else:
            data_small = data.head(max_rows)

        if st.button("🚀 Run AutoML (PyCaret)", key="ml_run_automl"):
            with st.spinner("Running PyCaret AutoML session…"):
                try:
                    # ננסה שתי חתימות נפוצות
                    try:
                        result = run_pycaret_regression(data_small, target=cfg.target_var)  # type: ignore[call-arg]
                    except TypeError:
                        result = run_pycaret_regression(data_small, cfg.target_var)  # type: ignore[call-arg]

                    if isinstance(result, pd.DataFrame):
                        st.write("**AutoML Leaderboard (Top models)**")
                        st.dataframe(result.head(50), width="stretch")
                        try:
                            # אם יש עמודה ל-R2 או Metric
                            score_col = None
                            for cand in ("R2", "R2_test", "R2_train", "Score", "Metric"):
                                if cand in result.columns:
                                    score_col = cand
                                    break
                            if score_col:
                                fig_lb, ax_lb = plt.subplots(figsize=(6, 4))
                                sub = result.head(15)
                                ax_lb.barh(sub["Model"].astype(str), sub[score_col])
                                ax_lb.set_title(f"Top models by {score_col}")
                                ax_lb.invert_yaxis()
                                st.pyplot(fig_lb)
                        except Exception:
                            pass
                    else:
                        st.write("**AutoML result (raw)**")
                        st.json(result)
                except Exception as e:
                    st.error(f"AutoML run failed: {e}")


def _log_tracking_backends(
    artifacts: MLArtifacts,
    cfg: MLConfig,
) -> None:
    """
    Integration hooks for MLflow / W&B / Neptune.

    המטרה:
        - לתת לך אפשרות לוגינג לפרודקשן / ניסויים.
        - לא להפיל את ה-UI אם משהו בחוץ לא מוגדר.

    כל שגיאה: נרשמת בלוג ברמת DEBUG בלבד.
    """
    metrics = artifacts.metrics
    cfg_dict = cfg.to_dict()

    # MLflow
    if cfg.use_mlflow and mlflow is not None:
        try:
            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "pairs_ml_analysis"))
            with mlflow.start_run(run_name=f"{cfg.model_name}_{cfg.target_var}_{datetime.now():%Y%m%d_%H%M%S}"):
                mlflow.log_params(cfg_dict)
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
                if artifacts.cv_scores is not None:
                    mlflow.log_metric("cv_r2_mean", float(artifacts.cv_scores.mean()))
                    mlflow.log_metric("cv_r2_std", float(artifacts.cv_scores.std()))
        except Exception as e:
            logger.debug("MLflow logging failed (ignored): %s", e)

    # Weights & Biases
    if cfg.use_wandb and wandb is not None:
        try:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "pairs_ml_analysis"),
                config=cfg_dict,
                reinit=True,
            )
            wandb.log({k: float(v) for k, v in metrics.items()})
            if artifacts.cv_scores is not None:
                wandb.log(
                    {
                        "cv_r2_mean": float(artifacts.cv_scores.mean()),
                        "cv_r2_std": float(artifacts.cv_scores.std()),
                    }
                )
            wandb.finish()
        except Exception as e:
            logger.debug("W&B logging failed (ignored): %s", e)

    # Neptune
    if cfg.use_neptune and neptune is not None:
        try:
            run = neptune.init_run(project=os.environ.get("NEPTUNE_PROJECT", "pairs/ml-analysis"))  # type: ignore[arg-type]
            run["config"] = cfg_dict  # type: ignore[index]
            for k, v in metrics.items():
                run[f"metrics/{k}"] = float(v)  # type: ignore[index]
            if artifacts.cv_scores is not None:
                run["metrics/cv_r2_mean"] = float(artifacts.cv_scores.mean())  # type: ignore[index]
                run["metrics/cv_r2_std"] = float(artifacts.cv_scores.std())  # type: ignore[index]
            run.stop()
        except Exception as e:
            logger.debug("Neptune logging failed (ignored): %s", e)


# נשמור רפרנס לגרסת Part 3 (לפני ההרחבה)
_RENDER_ML_EXPLAIN_V3 = render_ml_analysis


def render_ml_analysis(df_results: pd.DataFrame, sel_params: List[str]) -> None:  # noqa: C901
    """
    Streamlit UI entry point (Part 4/6 — Scenario, AutoML & Tracking Lab).

    מה קבוע ומה אופציונלי?
    -----------------------
    ✅ קבוע:
        - מודל ML ראשי (עם CV ו-Holdout).
        - השוואת מודלים (Model Grid).
        - Explainability מלאה:
            • Feature profile
            • Permutation importance
            • Residual diagnostics (basic + advanced)
            • Error vs Feature Explorer
            • SHAP (אם קיים)
            • PDP/ICE (אם קיים)

    🧪 אופציונלי דרך הסיידבר:
        - Scenario Editor (What-if) — enable_scenarios
        - AutoML (PyCaret) — enable_automl (אם מחובר)
        - Tracking: MLflow / W&B / Neptune (טוגלים)

    API נשמר:
        render_ml_analysis(df_results, sel_params)
    """
    st.header("🧠 ML & AutoML Analysis — Scenario & AutoML Lab (v6.0, Part 4/6)")

    if df_results is None or df_results.empty:
        st.warning("df_results is empty — run optimisation first.")
        return

    # ---------------- Sidebar: configuration ----------------
    st.sidebar.header("ML Settings")

    if not sel_params:
        blacklist = {"Score", "Sharpe", "Profit", "Drawdown"}
        sel_params = [
            c
            for c in df_results.columns
            if c not in blacklist and pd.api.types.is_numeric_dtype(df_results[c])
        ]

    target_candidates = [c for c in ("Score", "Profit", "Drawdown", "Sharpe") if c in df_results.columns]
    if not target_candidates:
        target_candidates = [c for c in df_results.columns if pd.api.types.is_numeric_dtype(df_results[c])]

    target_var = st.sidebar.selectbox(
        "Target variable",
        options=target_candidates,
        index=0,
    )

    available_models = [m for m, fn in MODEL_REGISTRY.items() if fn is not None]
    default_model_index = available_models.index("XGBoost") if "XGBoost" in available_models else 0
    model_choice = st.sidebar.selectbox("Primary model", available_models, index=default_model_index)

    compare_models = st.sidebar.multiselect(
        "Models to compare",
        options=available_models,
        default=list({model_choice, "RandomForest", "GradientBoosting", "XGBoost"} & set(available_models)),
    )

    cv_type = st.sidebar.selectbox("CV type", list(CV_REGISTRY.keys()), index=0)
    cv_folds = int(st.sidebar.number_input("Folds", 3, 10, value=5))
    holdout_pct = float(st.sidebar.number_input("Hold-out %", 0.1, 0.4, value=0.2, step=0.05))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced ML Tools")

    enable_scenarios = st.sidebar.checkbox("Enable Scenario Editor (What-if)", value=True)
    enable_automl = st.sidebar.checkbox(
        "Enable AutoML panel (PyCaret)",
        value=False,
        disabled=(run_pycaret_regression is None),
    )
    use_mlflow = st.sidebar.checkbox(
        "Track with MLflow",
        value=False,
        disabled=(mlflow is None),
    )
    use_wandb = st.sidebar.checkbox(
        "Track with Weights & Biases",
        value=False,
        disabled=(wandb is None),
    )
    use_neptune = st.sidebar.checkbox(
        "Track with Neptune",
        value=False,
        disabled=(neptune is None),
    )

    cfg = MLConfig(
        model_name=model_choice,
        target_var=target_var,
        feature_cols=sel_params,
        cv_type=cv_type,
        cv_folds=cv_folds,
        holdout_pct=holdout_pct,
        use_mlflow=use_mlflow,
        use_wandb=use_wandb,
        use_neptune=use_neptune,
    )

    # ---------------- Feature profile ----------------
    _render_feature_profile(df_results, sel_params, target_var)

    # ---------------- Core ML run — primary model ----------------
    st.subheader(f"Primary Model: {model_choice}")
    try:
        artifacts = _run_core_ml_pipeline(df_results, cfg)
    except Exception as e:
        st.error(f"Primary ML run failed: {e}")
        logger.exception("Primary ML run failed")
        return

    if artifacts.cv_scores is not None:
        st.write(
            f"**CV R² ({cfg.cv_type}, k={cfg.cv_folds}):** "
            f"{artifacts.cv_scores.mean():.3f} ± {artifacts.cv_scores.std():.3f}"
        )

    m = artifacts.metrics
    display_metrics_block(
        r2=m.get("r2", 0.0),
        rmse_val=m.get("rmse", 0.0),
        mae_val=m.get("mae", 0.0),
        mape_val=m.get("mape", 0.0),
        sharpe_like=m.get("reg_sharpe", 0.0),
    )

    # Global metrics summary + export
    _render_global_metrics_summary(artifacts, cfg)

    # Basic residuals (scatter) – כמו בחלק 1/3
    st.subheader("Residuals Diagnostics — Basic")
    try:
        y_test = np.asarray(artifacts.y_test, dtype=float)
        y_pred = np.asarray(artifacts.preds, dtype=float)
        residuals = y_test - y_pred

        fig_res, ax_res = plt.subplots(figsize=(6, 4))
        ax_res.scatter(y_pred, residuals, alpha=0.6)
        ax_res.axhline(0.0, color="red", linestyle="--")
        ax_res.set_xlabel("Predicted")
        ax_res.set_ylabel("Residual")
        ax_res.set_title("Residuals vs Predicted")
        st.pyplot(fig_res)
    except Exception:
        st.info("Basic residual plot unavailable.")

    with st.expander("X_test sample (primary model)", expanded=False):
        try:
            st.dataframe(artifacts.X_test.head(20), width="stretch")  # type: ignore[arg-type]
        except Exception:
            st.info("No X_test preview available.")

    # ---------------- Model comparison grid ----------------
    if compare_models:
        st.subheader("Model Comparison Grid")
        try:
            comp_df, artifacts_map = _run_model_grid(df_results, cfg, compare_models)
            if comp_df.empty:
                st.info("Could not compute comparison grid (no successful models).")
            else:
                st.dataframe(comp_df, width="stretch")
                try:
                    st.bar_chart(
                        comp_df.set_index("Model")["Holdout_R2"],
                        width="stretch",
                    )
                except Exception:
                    pass
        except Exception as e:
            st.info(f"Model comparison failed: {e}")
            artifacts_map = {cfg.model_name: artifacts}
    else:
        artifacts_map = {cfg.model_name: artifacts}

    # ---------------- Feature importance (Permutation) ----------------
    _render_feature_importance_panel(artifacts, cfg.feature_cols)

    # ---------------- Advanced residual diagnostics ----------------
    _render_residual_diagnostics_advanced(artifacts)

    # ---------------- Error vs Feature Explorer ----------------
    _render_error_vs_feature_panel(artifacts, cfg.feature_cols)

    # ---------------- SHAP Explainability (אם זמין) ----------------
    _render_shap_panel(artifacts, cfg.feature_cols)

    # ---------------- Partial Dependence (PDP/ICE) ----------------
    _render_partial_dependence_panel(artifacts, cfg.feature_cols)

    # ---------------- Scenario Editor (What-if, Optional but On by default) ----------------
    if enable_scenarios:
        _render_scenario_editor(df_results, artifacts, cfg)

    # ---------------- AutoML Panel (PyCaret, Optional) ----------------
    if enable_automl:
        _render_automl_panel(df_results, cfg)

    # ---------------- Tracking backends (MLflow / W&B / Neptune) ----------------
    _log_tracking_backends(artifacts, cfg)

    # =========================
# PART 5/6: Model Card, Export & Programmatic API (Research-Grade)
# =========================

@dataclass
class ModelCard:
    """
    Structured model card for the ML analysis of Optuna results.

    Fields:
    -------
    - title           : כותרת ברורה (כולל מודל/טארגט).
    - created_at      : timestamp ISO.
    - model_name      : שם המודל (XGBoost וכו').
    - target_var      : המשתנה המנובא (Score/Profit/Sharpe/Drawdown).
    - feature_cols    : רשימת הפיצ'רים.
    - metrics         : Holdout + CV metrics.
    - data_profile    : מידע בסיסי על הדאטה (מספר שורות/עמודות וכו').
    - notes           : טקסט חופשי (מסקנות / אזהרות / TODO).
    - comparison_head : טבלת השוואה בין מודלים (אם קיימת).
    """

    title: str
    created_at: str

    model_name: str
    target_var: str
    feature_cols: List[str]

    metrics: Dict[str, Any] = field(default_factory=dict)
    data_profile: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    comparison_head: Optional[pd.DataFrame] = None

    def to_markdown(self) -> str:
        """Build a Markdown representation of the model card."""
        lines: List[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**Created at:** `{self.created_at}`")
        lines.append("")
        lines.append("## Model & Target")
        lines.append(f"- **Model:** `{self.model_name}`")
        lines.append(f"- **Target variable:** `{self.target_var}`")
        lines.append("")
        lines.append("## Features")
        if self.feature_cols:
            lines.append("```text")
            for f in self.feature_cols:
                lines.append(f"- {f}")
            lines.append("```")
        else:
            lines.append("_No features list available_")
        lines.append("")
        lines.append("## Metrics (Hold-out & CV)")
        if self.metrics:
            for k, v in self.metrics.items():
                try:
                    lines.append(f"- **{k}**: `{float(v):.6f}`")
                except Exception:
                    lines.append(f"- **{k}**: `{v}`")
        else:
            lines.append("_No metrics recorded_")

        lines.append("")
        lines.append("## Data Profile")
        if self.data_profile:
            lines.append("```json")
            import json as _json
            lines.append(_json.dumps(self.data_profile, ensure_ascii=False, indent=2))
            lines.append("```")
        else:
            lines.append("_No data profile available_")

        if self.comparison_head is not None and not self.comparison_head.empty:
            lines.append("")
            lines.append("## Model Comparison (Top rows)")
            lines.append("")
            try:
                # טבלת Markdown פשוטה
                head = self.comparison_head.head(10)
                lines.append(head.to_markdown(index=False))
            except Exception:
                lines.append("_Failed to render comparison table_")

        if self.notes:
            lines.append("")
            lines.append("## Notes / Commentary")
            lines.append(self.notes)

        return "\n".join(lines)


def _build_data_profile(df_results: pd.DataFrame, cfg: MLConfig) -> Dict[str, Any]:
    """
    Compute a small, JSON-safe data profile for the Optuna results used
    in the ML analysis.
    """
    profile: Dict[str, Any] = {}
    try:
        profile["shape"] = {"rows": int(df_results.shape[0]), "cols": int(df_results.shape[1])}
        profile["columns"] = list(map(str, df_results.columns))
        if cfg.target_var in df_results.columns:
            tgt = pd.to_numeric(df_results[cfg.target_var], errors="coerce")
            profile["target_summary"] = {
                "mean": float(tgt.mean()),
                "std": float(tgt.std(ddof=0)),
                "min": float(tgt.min()),
                "max": float(tgt.max()),
                "n": int(tgt.notna().sum()),
            }
    except Exception as e:
        profile["error"] = f"failed_to_profile: {e}"
    return profile


def _build_model_card(
    df_results: pd.DataFrame,
    artifacts: MLArtifacts,
    cfg: MLConfig,
    comparison_df: Optional[pd.DataFrame] = None,
    extra_notes: str = "",
) -> ModelCard:
    """
    Build a ModelCard instance from current artifacts + config + data.

    אפשר להשתמש בה גם מתוך Streamlit (ייצוא ל-Markdown),
    וגם מתוך API פרוגרמטי.
    """
    title = f"ML Model Card — {cfg.model_name} on {cfg.target_var}"
    created_at = datetime.now().isoformat(timespec="seconds")

    data_profile = _build_data_profile(df_results, cfg)

    # organise metrics nicely
    metrics_dict = dict(artifacts.metrics)
    if artifacts.cv_scores is not None:
        metrics_dict["cv_r2_mean"] = float(artifacts.cv_scores.mean())
        metrics_dict["cv_r2_std"] = float(artifacts.cv_scores.std())

    card = ModelCard(
        title=title,
        created_at=created_at,
        model_name=cfg.model_name,
        target_var=cfg.target_var,
        feature_cols=list(cfg.feature_cols),
        metrics=metrics_dict,
        data_profile=data_profile,
        comparison_head=comparison_df,
        notes=extra_notes,
    )
    return card


def _render_model_card_panel(
    df_results: pd.DataFrame,
    artifacts: MLArtifacts,
    cfg: MLConfig,
    comparison_df: Optional[pd.DataFrame],
) -> None:
    """
    Streamlit panel that shows and exports a Markdown model card.

    - מציג preview בתוך טאב.
    - מאפשר הורדה כ-`.md` (Model Card).
    """
    with st.expander("📝 Model Card (Markdown Export)", expanded=False):
        extra_notes = st.text_area(
            "Notes / Commentary (will be included in model card)",
            value="",
            key="ml_modelcard_notes",
            height=150,
        )

        card = _build_model_card(df_results, artifacts, cfg, comparison_df, extra_notes)
        md_text = card.to_markdown()

        st.markdown(md_text)

        try:
            st.download_button(
                "Download model_card.md",
                data=md_text.encode("utf-8"),
                file_name=f"model_card_{cfg.model_name}_{cfg.target_var}.md",
                mime="text/markdown",
                key="ml_modelcard_dl",
            )
        except Exception:
            pass


def export_pipeline_to_joblib(
    artifacts: MLArtifacts,
    folder: str | Path,
    name: Optional[str] = None,
) -> Optional[Path]:
    """
    Export the fitted pipeline (pre + model) to a `.joblib` file.

    Usage (from code):
        path = export_pipeline_to_joblib(artifacts, "models/", "xgb_score_v1")

    Returns:
        Path to the saved file, or None if export not possible.
    """
    if artifacts.pipeline is None:
        logger.warning("export_pipeline_to_joblib: no pipeline to export.")
        return None

    try:
        from joblib import dump  # type: ignore
    except Exception:
        logger.warning("joblib not installed — cannot export pipeline.")
        return None

    folder_p = Path(folder).expanduser().resolve()
    folder_p.mkdir(parents=True, exist_ok=True)

    if not name:
        name = f"{artifacts.config.model_name}_{datetime.now():%Y%m%d_%H%M%S}"

    file_path = folder_p / f"{name}.joblib"
    try:
        dump(artifacts.pipeline, file_path)
        logger.info("Exported pipeline to %s", file_path)
        return file_path
    except Exception as e:
        logger.warning("export_pipeline_to_joblib failed: %s", e)
        return None


def export_dataset_to_parquet(
    df_results: pd.DataFrame,
    folder: str | Path,
    name: str = "optuna_results",
) -> Optional[Path]:
    """
    Export the Optuna results DataFrame to Parquet (compressed).

    Usage:
        path = export_dataset_to_parquet(df_results, "data/", "optuna_score_results")
    """
    folder_p = Path(folder).expanduser().resolve()
    folder_p.mkdir(parents=True, exist_ok=True)
    file_path = folder_p / f"{name}.parquet"
    try:
        df_results.to_parquet(file_path, index=False)
        logger.info("Exported dataset to %s", file_path)
        return file_path
    except Exception as e:
        logger.warning("export_dataset_to_parquet failed: %s", e)
        return None


def _render_export_panel(
    df_results: pd.DataFrame,
    artifacts: MLArtifacts,
    cfg: MLConfig,
) -> None:
    """
    Streamlit UI for exporting pipeline & data.

    - Pipeline → joblib (לשימוש בקוד אחר).
    - Dataset → Parquet.
    """
    with st.expander("📦 Export (Pipeline & Dataset)", expanded=False):
        base_folder = st.text_input(
            "Base export folder (on server)",
            value=str(Path("exports").absolute()),
            key="ml_export_folder",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Export pipeline (.joblib)", key="ml_export_pipe_btn"):
                p = export_pipeline_to_joblib(artifacts, base_folder)
                if p is not None:
                    st.success(f"Pipeline exported to: {p}")
                else:
                    st.info("Pipeline export failed or unavailable.")
        with c2:
            if st.button("Export dataset (.parquet)", key="ml_export_data_btn"):
                p = export_dataset_to_parquet(df_results, base_folder)
                if p is not None:
                    st.success(f"Dataset exported to: {p}")
                else:
                    st.info("Dataset export failed or unavailable.")


def analyze_optuna_results(
    df_results: pd.DataFrame,
    feature_cols: List[str],
    target_var: str = "Score",
    model_name: str = "XGBoost",
    cv_type: str = "KFold",
    cv_folds: int = 5,
    holdout_pct: float = 0.2,
) -> Dict[str, Any]:
    """
    Programmatic API (ללא Streamlit) לשימוש בקוד/סקריפטים:

        result = analyze_optuna_results(
            df_results,
            feature_cols=["z_open", "z_close", "lookback", ...],
            target_var="Score",
            model_name="XGBoost",
        )

    מחזיר:
        dict עם:
            - "metrics": מדדים על holdout.
            - "cv_r2_mean", "cv_r2_std".
            - "config": MLConfig as dict.
    """
    cfg = MLConfig(
        model_name=model_name,
        target_var=target_var,
        feature_cols=feature_cols,
        cv_type=cv_type,
        cv_folds=cv_folds,
        holdout_pct=holdout_pct,
    )
    artifacts = _run_core_ml_pipeline(df_results, cfg)
    out = artifacts.to_perf_dict()
    return out


# נוסיף את Model Card & Export לפונקציית render_ml_analysis (v4) באמצעות wrapper קל
_RENDER_ML_SCENARIO_V4 = render_ml_analysis  # מגרסת Part 4


def render_ml_analysis(df_results: pd.DataFrame, sel_params: List[str]) -> None:  # noqa: C901
    """
    Streamlit UI entry point (Part 5/6 — Model Card, Export & Research API).

    מרחיב את Part 4:
        - כל מה שהיה (מודל ראשי, השוואת מודלים, Explainability, Scenario, AutoML, Tracking).
        - בנוסף:
            • Model Card (Markdown) עם export.
            • Export pipeline (joblib) ו-dataset (Parquet).
            • API פרוגרמטי analyze_optuna_results (ללא UI) — מחוץ ל-Streamlit.

    API נשאר:
        render_ml_analysis(df_results, sel_params)
    """
    # נריץ קודם את כל ה-UI הקיים מפארט 4
    _RENDER_ML_SCENARIO_V4(df_results, sel_params)

    # כדי לקבל artifacts/cfg/comparison_df לצרכים של model card/export,
    # אנחנו צריכים "לחשב מחדש" בצורה קלה, כמו בחלק 4.
    try:
        # שחזור config מינימלי מברירת המחדל של הסיידבר (זהה ללוגיקה של Part 4)
        st.sidebar.header("Model Card & Export", divider="grey")

        # שיחזור מהסיידבר (אותם פקדים אבל בתוך חלק 5):
        # כדי לא לדרוס את הסיידבר המקורי, ננסה לקרוא את ה-state ישירות.
        target_candidates = [c for c in ("Score", "Profit", "Drawdown", "Sharpe") if c in df_results.columns]
        if not target_candidates:
            target_candidates = [c for c in df_results.columns if pd.api.types.is_numeric_dtype(df_results[c])]

        target_var = st.sidebar.selectbox(
            "Target for Model Card",
            options=target_candidates,
            index=0,
            key="ml_mc_target_var",
        )

        if not sel_params:
            blacklist = {"Score", "Sharpe", "Profit", "Drawdown"}
            sel_params = [
                c
                for c in df_results.columns
                if c not in blacklist and pd.api.types.is_numeric_dtype(df_results[c])
            ]

        available_models = [m for m, fn in MODEL_REGISTRY.items() if fn is not None]
        default_model_index = available_models.index("XGBoost") if "XGBoost" in available_models else 0
        model_choice = st.sidebar.selectbox(
            "Model for Model Card",
            available_models,
            index=default_model_index,
            key="ml_mc_model",
        )

        cv_type = st.sidebar.selectbox(
            "CV type (Model Card)",
            list(CV_REGISTRY.keys()),
            index=0,
            key="ml_mc_cv_type",
        )
        cv_folds = int(st.sidebar.number_input("Folds (Model Card)", 3, 10, value=5, key="ml_mc_cv_folds"))
        holdout_pct = float(
            st.sidebar.number_input("Hold-out % (Model Card)", 0.1, 0.4, value=0.2, step=0.05, key="ml_mc_holdout")
        )

        cfg_mc = MLConfig(
            model_name=model_choice,
            target_var=target_var,
            feature_cols=sel_params,
            cv_type=cv_type,
            cv_folds=cv_folds,
            holdout_pct=holdout_pct,
        )
        artifacts_mc = _run_core_ml_pipeline(df_results, cfg_mc)

        # נעשה גם השוואת מודלים קצרה לצורך הכנסת חלק לטבלת model card
        try:
            comp_df_mc, _ = _run_model_grid(df_results, cfg_mc, [model_choice, "RandomForest", "GradientBoosting"])
        except Exception:
            comp_df_mc = None  # type: ignore[assignment]

        # Model Card panel + Export panel
        _render_model_card_panel(df_results, artifacts_mc, cfg_mc, comp_df_mc)
        _render_export_panel(df_results, artifacts_mc, cfg_mc)

    except Exception as e:
        st.info(f"Model Card & Export panel failed (non-critical): {e}")
# =========================
# PART 6/6: Integration Bridge, ML Summary & Scenario API (System-Grade)
# =========================

def build_ml_summary(
    artifacts: MLArtifacts,
    cfg: MLConfig,
    comparison_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Build a compact, JSON-safe ML summary dict that other parts of the
    system (למשל טאב האופטימיזציה / insights) יכולים להשתמש בו.

    כולל:
        - config (model, target, CV וכו').
        - metrics (Holdout + CV).
        - top models from comparison grid (אם קיים).
    """
    summary: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "metrics": dict(artifacts.metrics),
    }
    if artifacts.cv_scores is not None:
        summary["metrics"]["cv_r2_mean"] = float(artifacts.cv_scores.mean())
        summary["metrics"]["cv_r2_std"] = float(artifacts.cv_scores.std())

    if comparison_df is not None and not comparison_df.empty:
        try:
            head = comparison_df.head(5).copy()
            # נעשה הכל float/str כדי שיהיה JSON-safe
            for c in head.columns:
                if pd.api.types.is_numeric_dtype(head[c]):
                    head[c] = head[c].astype(float)
                else:
                    head[c] = head[c].astype(str)
            summary["model_comparison_head"] = head.to_dict(orient="records")
        except Exception as e:
            summary["model_comparison_error"] = f"{e}"

    return summary


def publish_ml_summary_to_session(
    summary: Dict[str, Any],
    key: str = "ml_analysis_summary",
) -> None:
    """
    Save ML summary into st.session_state so other tabs (optimization,
    insights, execution) can consume it.

    Usage:
        summary = build_ml_summary(artifacts, cfg, comp_df)
        publish_ml_summary_to_session(summary)
    """
    try:
        if not isinstance(st.session_state.get(key, None), dict):
            st.session_state[key] = {}
        st.session_state[key] = summary
    except Exception as e:
        logger.debug("publish_ml_summary_to_session failed: %s", e)


def render_ml_bridge_panel(key: str = "ml_analysis_summary") -> None:
    """
    Streamlit-side “bridge” panel שמראה מה נשמר ב-session:

        - config
        - metrics
        - model comparison head

    כך טאב אחר (למשל Dashboard ראשי) יכול פשוט לקרוא:
        from core.ml_analysis import render_ml_bridge_panel
        render_ml_bridge_panel()
    """
    with st.expander("🔗 ML Analysis Bridge (System Summary)", expanded=False):
        val = st.session_state.get(key)
        if not isinstance(val, dict):
            st.info("No ML summary found in session — run ML analysis first.")
            return

        cfg_dict = val.get("config", {})
        metrics = val.get("metrics", {})
        comp_head = val.get("model_comparison_head")

        st.write("**Config (ML)**")
        st.json(cfg_dict)

        st.write("**Metrics (ML)**")
        st.json(metrics)

        if comp_head:
            st.write("**Model comparison (top rows)**")
            try:
                df = pd.DataFrame(comp_head)
                st.dataframe(df, use_container_width=True)
            except Exception:
                st.json(comp_head)


def predict_single_config(
    config_row: Mapping[str, Any],
    artifacts: MLArtifacts,
    cfg: MLConfig,
) -> float:
    """
    Evaluate a *single* configuration (שורה אחת של פרמטרים) לפי המודל המאומן.

    Parameters:
        config_row : dict-like with keys matching cfg.feature_cols.
        artifacts  : MLArtifacts עם pipeline מאומן.
        cfg        : MLConfig, for feature list.

    Returns:
        float — predicted value עבור target (למשל Score).
    """
    if artifacts.pipeline is None:
        raise RuntimeError("predict_single_config: pipeline not fitted.")

    # build one-row DataFrame
    row_dict = {}
    for f in cfg.feature_cols:
        row_dict[f] = config_row.get(f, np.nan)
    df = pd.DataFrame([row_dict])

    y_pred = artifacts.pipeline.predict(df)
    return float(y_pred[0])


def predict_configs_df(
    df_configs: pd.DataFrame,
    artifacts: MLArtifacts,
    cfg: MLConfig,
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Evaluate a whole DataFrame של קונפיגים (למשל טבלת פרמטרים מהאופטימיזציה)
    לפי המודל המאומן.

    Parameters:
        df_configs : DataFrame that contains columns matching cfg.feature_cols.
        artifacts  : MLArtifacts with fitted pipeline.
        cfg        : MLConfig.
        out_col    : Optional name for prediction column; default target_var + '_pred'.

    Returns:
        DataFrame with prediction column appended.
    """
    if artifacts.pipeline is None:
        raise RuntimeError("predict_configs_df: pipeline not fitted.")

    if out_col is None:
        out_col = cfg.target_var + "_pred"

    X = df_configs.copy()
    missing = [c for c in cfg.feature_cols if c not in X.columns]
    if missing:
        raise ValueError(f"predict_configs_df: missing feature columns: {missing}")

    preds = artifacts.pipeline.predict(X[cfg.feature_cols])
    X[out_col] = np.asarray(preds, dtype=float)
    return X


def _render_integration_tools_panel(
    df_results: pd.DataFrame,
    artifacts: MLArtifacts,
    cfg: MLConfig,
    comparison_df: Optional[pd.DataFrame],
) -> None:
    """
    Small Streamlit panel שמרכז את ה-"גשר" למערכת:

        - Save summary to session.
        - Quick single-config test (editable JSON).
    """
    with st.expander("⚙️ Integration Tools (Bridge to System)", expanded=False):
        # 1) Summary → session
        if st.button("Save ML summary to session_state['ml_analysis_summary']", key="ml_save_summary"):
            summary = build_ml_summary(artifacts, cfg, comparison_df)
            publish_ml_summary_to_session(summary)
            st.success("ML summary saved to session.")

        # 2) Single config tester
        st.write("**Quick config tester** — הזן dict-like JSON עם הפיצ'רים:")

        default_payload = {
            f: float(df_results[f].median()) if f in df_results.columns and pd.api.types.is_numeric_dtype(df_results[f]) else 0.0
            for f in cfg.feature_cols
        }
        raw = st.text_area(
            "Configuration JSON",
            value=str(default_payload),
            key="ml_quick_cfg_json",
            height=150,
        )
        if st.button("Test single config", key="ml_test_cfg_btn"):
            import ast as _ast

            try:
                parsed = _ast.literal_eval(raw)
                score = predict_single_config(parsed, artifacts, cfg)
                st.success(f"Predicted {cfg.target_var} = {score:.6f}")
            except Exception as e:
                st.error(f"Failed to parse/predict config: {e}")


# נסגור מעגל: נעטוף שוב את render_ml_analysis (v5) כדי להוסיף Integration Tools + Bridge
_RENDER_ML_V5 = render_ml_analysis


def render_ml_analysis(df_results: pd.DataFrame, sel_params: List[str]) -> None:  # noqa: C901
    """
    FINAL Streamlit UI entry point (Part 6/6 — Full Integration).

    שכבות:
    -------
    1–4:  Core ML, Model comparison, Explainability (Residuals, SHAP, PDP),
          Scenario Editor (אופציונלי), AutoML (אופציונלי), Tracking.
    5:   Model Card + Export pipeline/dataset.
    6:   Integration Bridge:
         - ML summary → session_state['ml_analysis_summary'].
         - Quick config tester (JSON → prediction).
         - Predict APIs (predict_single_config / predict_configs_df) לפרוגרמטיקה.

    ה-API נשאר:
        render_ml_analysis(df_results, sel_params)
    """
    # להריץ את כל ה-UI הקיים (עד Part 5)
    _RENDER_ML_V5(df_results, sel_params)

    # לבנות artifacts/cfg/comparison_df מחדש באופן דומה לפאנל model-card:
    try:
        # לבחירת Target & Model ל"מלך המערכת"
        target_candidates = [c for c in ("Score", "Profit", "Drawdown", "Sharpe") if c in df_results.columns]
        if not target_candidates:
            target_candidates = [c for c in df_results.columns if pd.api.types.is_numeric_dtype(df_results[c])]

        if not sel_params:
            blacklist = {"Score", "Sharpe", "Profit", "Drawdown"}
            sel_params = [
                c
                for c in df_results.columns
                if c not in blacklist and pd.api.types.is_numeric_dtype(df_results[c])
            ]

        target_var = target_candidates[0]
        model_choice = "XGBoost" if "XGBoost" in MODEL_REGISTRY and MODEL_REGISTRY["XGBoost"] is not None else list(MODEL_REGISTRY.keys())[0]

        cfg_bridge = MLConfig(
            model_name=model_choice,
            target_var=target_var,
            feature_cols=sel_params,
            cv_type="KFold",
            cv_folds=5,
            holdout_pct=0.2,
        )
        artifacts_bridge = _run_core_ml_pipeline(df_results, cfg_bridge)
        try:
            comp_df_bridge, _ = _run_model_grid(df_results, cfg_bridge, [model_choice, "RandomForest", "GradientBoosting"])
        except Exception:
            comp_df_bridge = None  # type: ignore[assignment]

        # Panel ששולט ב-Bridge ו-Quick Tester
        _render_integration_tools_panel(df_results, artifacts_bridge, cfg_bridge, comp_df_bridge)

    except Exception as e:
        st.info(f"Integration Tools panel failed (non-critical): {e}")

def render_ml_for_optuna_session(
    opt_df_key: str = "opt_df",
    *,
    default_target: str = "Score",
) -> None:
    """
    גשר ישיר מ-session של ה-אופטימיזציה ל-ML:

    - לוקח df_results מתוך st.session_state[opt_df_key]
    - בוחר target (ברירת מחדל Score, אם אין – Profit/Sharpe/Drawdown)
    - בוחר feature_cols (כל העמודות הנומריות חוץ מהמדדים)
    - קורא ל-render_ml_analysis(df_results, feature_cols)

    שימוש בדשבורד / optimization_tab:
        from core.ml_analysis import render_ml_for_optuna_session
        render_ml_for_optuna_session("opt_df")
    """
    if "st" not in globals():
        # ביטחון: אם מישהו קרא לזה מחוץ ל-Streamlit.
        raise RuntimeError("render_ml_for_optuna_session must be used inside Streamlit.")

    df_results = st.session_state.get(opt_df_key)
    if df_results is None or not isinstance(df_results, pd.DataFrame) or df_results.empty:
        st.warning(f"No DataFrame found in st.session_state['{opt_df_key}'] — run optimisation first.")
        return

    # בחירת target
    candidates = []
    for col in ("Score", default_target, "Sharpe", "Profit", "Drawdown"):
        if col in df_results.columns and col not in candidates:
            candidates.append(col)
    # אם אין Score/Profit/Sharpe/Drawdown – נבחר נומרי ראשון
    if not candidates:
        numeric_cols = [c for c in df_results.columns if pd.api.types.is_numeric_dtype(df_results[c])]
        if not numeric_cols:
            st.error("No numeric columns in opt_df for ML target.")
            return
        candidates = [numeric_cols[0]]

    # נרשה למשתמש לבחור target מתוך ה-candidates
    target_var = st.selectbox(
        "Target variable (from opt_df)",
        options=candidates,
        index=0,
        key="ml_opt_session_target",
    )

    # בחירת פיצ'רים: כל הנומריים חוץ מהמדדים
    blacklist = {target_var, "Score", "Sharpe", "Profit", "Drawdown"}
    feature_cols = [
        c
        for c in df_results.columns
        if (c not in blacklist) and pd.api.types.is_numeric_dtype(df_results[c])
    ]

    if not feature_cols:
        st.error("No numeric feature columns found in opt_df after filtering metrics.")
        return

    st.caption(
        f"Using st.session_state['{opt_df_key}'] as df_results, "
        f"target = `{target_var}`, features = {len(feature_cols)} cols."
    )

    # קורא לפונקציה הראשית (הגרסה הסופית שלנו אחרי Part 6)
    render_ml_analysis(df_results, feature_cols)
