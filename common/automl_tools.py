# -*- coding: utf-8 -*-
"""
common/automl_tools.py – AutoML & Analytics Toolkit
====================================================

Purpose
-------
A *one-stop* toolbox that augments Optuna-based optimisation with a rich
suite of data-science helpers: AutoML, explainability, fairness, visual
analytics, and quick-export utilities.

Design principles
-----------------
* Heavy dependencies (PyCaret, SHAP, AutoTS, Fairlearn, sklearn, Plotly)
  are imported lazily – רק כשבאמת משתמשים בפונקציה.
* כל פונקציה בטוחה ל-import גם אם חבילות אופציונליות חסרות –
  תתקבל שגיאה ברורה עם pip install מתאים.
* הפונקציות מחזירות אובייקטים "ידידותיים לדשבורד" (DataFrame / Figure וכו').

Public API (__all__)
--------------------
run_pycaret_regression   shap_summary            permutation_importance
ensemble_blend           run_autots              fairness_report
optimization_path_plot   leaderboard_plot        scatter_3d_params
sensitivity_analysis     save_leaderboard_csv    ensure_figure
trials_to_features       run_automl_regression   shap_summary_plot
export_leaderboard
"""

from __future__ import annotations

###############################################################################
# Standard libs
###############################################################################
import importlib
import logging
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple, TYPE_CHECKING


import pandas as pd
from common.json_safe import make_json_safe, json_default as _json_default
if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


###############################################################################
# Lazy import helper
###############################################################################

def _require(module: str) -> Any:
    """
    Import a module lazily, raising a friendly ModuleNotFoundError.

    Parameters
    ----------
    module : str
        Fully-qualified module path, e.g. "pycaret.regression",
        "sklearn.inspection", "fairlearn.metrics".

    Returns
    -------
    module object

    Raises
    ------
    ModuleNotFoundError
        With a helpful pip install hint.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        base = module.split(".")[0]
        msg = (
            f"Optional dependency '{module}' is required for this function.\n"
            f"Install via:  pip install {base}"
        )
        raise ModuleNotFoundError(msg) from exc


def _lazy(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator: פונקציה "כבדה" שנשענת על חבילות חיצוניות.

    אין כאן קסם מיוחד – העיקרון הוא:
    * ה-importים הכבדים נעשים **בתוך** הגוף של הפונקציה.
    * הקובץ עצמו נטען מהר, ורק כשקוראים לפונקציה נמשכות החבילות.
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


###############################################################################
# Small JSON helper (שימושי אם תרצה לשמור תוצאות)
###############################################################################

def _to_json_safe(obj: Any) -> Any:
    """Convert nested results (figs, arrays, DataFrames) to JSON-safe form."""
    try:
        return make_json_safe(obj, default=_json_default)  # type: ignore[arg-type]
    except TypeError:
        return make_json_safe(obj)


###############################################################################
# 1. PyCaret AutoML – Regression
###############################################################################

@_lazy
def run_pycaret_regression(
    df: pd.DataFrame,
    *,
    target: str = "target",
    sort_by: str = "MAE",
    top_n: int = 3,
    include_models: Sequence[str] | None = None,
    exclude_models: Sequence[str] | None = None,
) -> Tuple[List[Any], pd.DataFrame]:
    """
    Run tabular AutoML using PyCaret (regression).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing *all* features **and** the target column.
    target : str, default "target"
        Column name to predict.
    sort_by : str, default "MAE"
        Metric to rank models by (e.g. "MAE", "RMSE", "R2").
    top_n : int, default 3
        Number of best models to return.
    include_models : sequence of str or None
        Optional whitelist of model IDs for PyCaret.
    exclude_models : sequence of str or None
        Optional blacklist of model IDs for PyCaret.

    Returns
    -------
    best_models : list[Any]
        The top-n models returned by PyCaret.
    leaderboard : pd.DataFrame
        Full leaderboard table from PyCaret (compare_models).
    """
    pyc = _require("pycaret.regression")

    logger.info(
        "Running PyCaret regression AutoML on %d rows, target='%s'",
        len(df),
        target,
    )

    exp = pyc.setup(  # type: ignore[attr-defined]
        df.copy(),
        target=target,
        silent=True,
        session_id=42,
        html=False,
        verbose=False,
        use_gpu=False,
        train_size=0.8,
        log_experiment=False,
        experiment_name="automl_regression",
    )
    logger.debug("PyCaret setup complete: %s", exp)

    compare_kwargs: dict[str, Any] = {
        "sort": sort_by,
        "n_select": top_n,
    }
    if include_models is not None:
        compare_kwargs["include"] = list(include_models)
    if exclude_models is not None:
        compare_kwargs["exclude"] = list(exclude_models)

    best_models = pyc.compare_models(**compare_kwargs)  # type: ignore[attr-defined]
    leaderboard = pyc.pull()  # type: ignore[attr-defined]

    logger.info("PyCaret regression complete – %d models compared", len(leaderboard))
    return best_models, leaderboard


###############################################################################
# 2. SHAP global summary plot
###############################################################################

@_lazy
def shap_summary(
    model: Any,
    X: pd.DataFrame,
    max_display: int = 20,
    show: bool = False,
):
    """
    Generate a SHAP summary plot (returns the matplotlib figure).

    Parameters
    ----------
    model : Any
        Trained model (sklearn / PyCaret model).
    X : pd.DataFrame
        Feature matrix used for SHAP.
    max_display : int, default 20
        Maximum number of features to show in the plot.
    show : bool, default False
        If True, call plt.show() inside the function.

    Returns
    -------
    fig : matplotlib.figure.Figure
        SHAP summary figure (bar/bee swarm).
    """
    shap = _require("shap")
    import matplotlib.pyplot as plt  # noqa: WPS433

    logger.info("Computing SHAP summary for %d rows, %d features", *X.shape)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    if show:
        plt.show()
    return fig


###############################################################################
# 3. Permutation importance (sklearn)
###############################################################################

@_lazy
def permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 10,
    scoring: str | None = None,
) -> pd.DataFrame:
    """
    Return permutation importance as sorted DataFrame.

    Parameters
    ----------
    model : Any
        Estimator implementing `predict` (sklearn API).
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    n_repeats : int, default 10
        Number of permutation rounds.
    scoring : str or None, default None
        Optional sklearn scoring string.

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance_mean, importance_std.
    """
    insp = _require("sklearn.inspection")
    logger.info("Computing permutation importance for %d features", X.shape[1])

    r = insp.permutation_importance(  # type: ignore[attr-defined]
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
        scoring=scoring,
    )
    df_imp = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return df_imp


###############################################################################
# 4. Ensemble blend (simple or weighted average)
###############################################################################

@_lazy
def ensemble_blend(
    models: Sequence[Any],
    X: pd.DataFrame,
    *,
    weights: Sequence[float] | None = None,
    predict_proba: bool = False,
):
    """
    Return blended prediction across *models* (scikit-learn API).

    Parameters
    ----------
    models : sequence of estimators
        Each must implement `predict` or `predict_proba` (if predict_proba=True).
    X : pd.DataFrame
        Feature matrix to predict on.
    weights : sequence of float or None, default None
        Optional weights for each model (same length as models).
        If None, all models get equal weight.
    predict_proba : bool, default False
        If True, blend `predict_proba` instead of `predict`.

    Returns
    -------
    np.ndarray
        Blended predictions (1D for regression, or 2D for proba).
    """
    import numpy as np

    if not models:
        raise ValueError("ensemble_blend: models sequence is empty")

    n_models = len(models)
    if weights is None:
        w = np.ones(n_models) / n_models
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != n_models:
            raise ValueError("weights length must match number of models")
        w = w / w.sum()

    logger.info("Blending %d models (predict_proba=%s)", n_models, predict_proba)

    if predict_proba:
        preds = [m.predict_proba(X) for m in models]
        arr = np.stack(preds, axis=0)  # (n_models, n_samples, n_classes)
        return np.tensordot(w, arr, axes=(0, 0))
    else:
        preds = [m.predict(X) for m in models]
        arr = np.column_stack(preds)  # (n_samples, n_models)
        return (arr * w).sum(axis=1)


###############################################################################
# 5. AutoTS quick forecast
###############################################################################

@_lazy
def run_autots(
    df: pd.DataFrame,
    *,
    target: str = "value",
    forecast_length: int = 3,
    date_col: str | None = None,
):
    """
    Run AutoTS on a time series dataframe and return forecast DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data. Either:
          * single target column (index = DatetimeIndex), or
          * two columns (date + target), ואז משתמשים ב-`date_col`.
    target : str, default "value"
        Target column name.
    forecast_length : int, default 3
        Number of future periods to forecast.
    date_col : str or None, default None
        If not None, name of the date column in df.

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame as returned by AutoTS.
    """
    autots = _require("autots")

    logger.info(
        "Running AutoTS: target='%s', forecast_length=%d, date_col=%r",
        target,
        forecast_length,
        date_col,
    )

    model = autots.AutoTS(  # type: ignore[attr-defined]
        forecast_length=forecast_length,
        frequency="infer",
        ensemble="simple",
    )

    if date_col is None:
        model = model.fit(df, date_col=None, value_col=target, id_col=None)
    else:
        model = model.fit(df, date_col=date_col, value_col=target, id_col=None)

    forecast = model.predict().forecast
    return forecast


###############################################################################
# 6. Fairness / Bias report (Fairlearn)
###############################################################################

@_lazy
def fairness_report(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
):
    """
    Compute fairness metrics per sensitive group using Fairlearn.

    Currently returns selection_rate per group (can be extended easily).

    Parameters
    ----------
    model : Any
        Estimator with `predict`.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        True labels.
    sensitive : pd.Series
        Sensitive attribute (e.g. gender, region).

    Returns
    -------
    pd.DataFrame
        Index = group values, columns = fairness metrics (currently selection_rate).
    """
    fair_metrics = _require("fairlearn.metrics")

    metrics = {"selection_rate": fair_metrics.selection_rate}  # type: ignore[attr-defined]
    frame = fair_metrics.MetricFrame(  # type: ignore[attr-defined]
        metrics=metrics,
        y_true=y,
        y_pred=model.predict(X),
        sensitive_features=sensitive,
    )
    result = frame.by_group
    return result


###############################################################################
# 7. Optimisation path plot (Plotly)
###############################################################################

@_lazy
def optimization_path_plot(study: "optuna.Study", best_so_far: bool = True):
    """
    Plot optimisation path (trial value vs. trial number).

    Parameters
    ----------
    study : optuna.Study
        Optuna study object.
    best_so_far : bool, default True
        If True, plot cumulative best value over time.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    values = [t.value for t in study.trials if t.value is not None]
    if not values:
        return go.Figure()

    if best_so_far:
        best_vals: list[float] = []
        current = float("inf")
        for v in values:
            current = min(current, v)
            best_vals.append(current)
        y_vals = best_vals
        name = "Best so far"
    else:
        y_vals = values
        name = "Trial value"

    fig = go.Figure(go.Scatter(y=y_vals, mode="lines+markers", name=name))
    fig.update_layout(
        title="Optimisation path",
        xaxis_title="Trial index",
        yaxis_title="Objective value",
    )
    return fig


###############################################################################
# 8. Leaderboard bar chart (Plotly)
###############################################################################

@_lazy
def leaderboard_plot(
    leaderboard_df: pd.DataFrame,
    metric: str = "MAE",
    *,
    top_n: int | None = None,
    ascending: bool | None = None,
):
    """
    Plot a bar chart of leaderboard results.

    Parameters
    ----------
    leaderboard_df : pd.DataFrame
        Leaderboard מקורי מפייקארט (pycaret.compare_models / pull).
        חייב להכיל עמודה 'Model' ועוד מדד (למשל 'MAE').
    metric : str, default "MAE"
        Metric column to plot.
    top_n : int or None, default None
        אם לא None – לוקחים רק את ה-top_n לפי המדד.
    ascending : bool or None, default None
        אם None – מניחים שקטן יותר טוב (כלומר ascending=True).
        אחרת משתמש בערך הנתון.

    Returns
    -------
    plotly.express.Figure
    """
    import plotly.express as px

    df = leaderboard_df.copy()
    if metric not in df.columns:
        raise ValueError(f"leaderboard_plot: metric '{metric}' not found in columns")

    if ascending is None:
        ascending = True  # לרוב MAE/RMSE – קטן יותר טוב

    df = df.sort_values(metric, ascending=ascending)
    if top_n is not None:
        df = df.head(top_n)

    if "Model" not in df.columns:
        # בגרסאות מסוימות של PyCaret השם שונה – ננסה לנחש
        model_col = next((c for c in df.columns if "Model" in c or "model" in c), None)
        if model_col is None:
            raise ValueError("leaderboard_plot: could not find 'Model' column")
        df = df.rename(columns={model_col: "Model"})

    fig = px.bar(
        df,
        x="Model",
        y=metric,
        title=f"Leaderboard – lower {metric} better",
        text_auto=".3f",
    )
    fig.update_layout(xaxis_tickangle=-30)
    return fig


###############################################################################
# 9. 3-D scatter of three params + target (Plotly)
###############################################################################

@_lazy
def scatter_3d_params(
    df: pd.DataFrame,
    p1: str,
    p2: str,
    p3: str,
    *,
    target: str = "value",
):
    """
    3-D scatter plot של שלושה פרמטרים מול מטרת האופטימיזציה.

    Parameters
    ----------
    df : pd.DataFrame
        נניח df של Optuna (trial params + value).
    p1, p2, p3 : str
        שמות העמודות לפרמטרים (למשל 'params_learning_rate' וכו').
    target : str, default "value"
        עמודת יעד לצביעה (objective value).

    Returns
    -------
    plotly.express.Figure
    """
    import plotly.express as px

    return px.scatter_3d(
        df,
        x=p1,
        y=p2,
        z=p3,
        color=target,
        title="Param space – 3D",
    )


###############################################################################
# 10. Sensitivity analysis (Matplotlib)
###############################################################################

@_lazy
def sensitivity_analysis(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    low: float,
    high: float,
    *,
    steps: int = 8,
    show: bool = False,
):
    """
    Simple 1D sensitivity analysis: sweep a feature and see mean prediction.

    Parameters
    ----------
    model : Any
        Estimator with `predict`.
    X : pd.DataFrame
        Base feature matrix (copied internally).
    feature : str
        Feature name to sweep.
    low, high : float
        Range of values to test.
    steps : int, default 8
        Number of points in the sweep.
    show : bool, default False
        If True, call plt.show().

    Returns
    -------
    vals : np.ndarray
        Grid של הערכים שנבדקו.
    preds : list[float]
        Mean prediction לכל ערך.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if feature not in X.columns:
        raise ValueError(f"sensitivity_analysis: feature '{feature}' not in X")

    vals = np.linspace(low, high, steps)
    preds: list[float] = []
    X_mod = X.copy()

    for v in vals:
        X_mod[feature] = v
        preds.append(float(model.predict(X_mod).mean()))

    plt.figure(figsize=(6, 3))
    plt.plot(vals, preds, marker="o")
    plt.xlabel(feature)
    plt.ylabel("Mean prediction")
    plt.title("Sensitivity analysis")
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()

    return vals, preds


###############################################################################
# 11. Convenience: save leaderboard
###############################################################################

def save_leaderboard_csv(leaderboard_df: pd.DataFrame, path: str | Path):
    """
    Save leaderboard CSV to *path* (create parents if needed).

    Parameters
    ----------
    leaderboard_df : pd.DataFrame
        Leaderboard from PyCaret.
    path : str | Path
        Destination path for CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_df.to_csv(path, index=False)
    logger.info("Leaderboard saved → %s", path)


###############################################################################
# 12. Figure helper
###############################################################################

def ensure_figure(fig: Any) -> Any:
    """
    Coerce Plotly / Matplotlib object into something Streamlit can show.

    Supports:
        * plotly.graph_objects.Figure
        * matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If the figure type is not supported.
    """
    import plotly.graph_objects as go
    import matplotlib.figure as mpl

    if isinstance(fig, (go.Figure, mpl.Figure)):
        return fig
    raise TypeError(f"Unsupported figure type: {type(fig)!r}")


###############################################################################
# 13. Trials → features helper
###############################################################################

def trials_to_features(df: pd.DataFrame, drop_nan: bool = True) -> pd.DataFrame:
    """
    Extract `params_*` columns as numeric features + `value` target.

    Assumes an Optuna-style DataFrame with columns:
        - params_*
        - value

    Parameters
    ----------
    df : pd.DataFrame
    drop_nan : bool, default True
        Drop rows with NaN values.

    Returns
    -------
    pd.DataFrame
        Columns: all params_* as features + 'target' column.
    """
    feat_cols = [c for c in df.columns if c.startswith("params_")]
    if "value" not in df.columns:
        raise ValueError("trials_to_features: expected 'value' column in df")

    out = df[feat_cols + ["value"]].copy()
    if drop_nan:
        out = out.dropna()
    out = out.rename(columns={"value": "target"})
    return out


###############################################################################
# Backwards-compatible aliases
###############################################################################

run_automl_regression = run_pycaret_regression
shap_summary_plot = shap_summary
export_leaderboard = save_leaderboard_csv


###############################################################################
# Public exports
###############################################################################

__all__ = [
    "run_pycaret_regression",
    "run_automl_regression",
    "trials_to_features",
    "shap_summary",
    "shap_summary_plot",
    "permutation_importance",
    "ensemble_blend",
    "run_autots",
    "fairness_report",
    "optimization_path_plot",
    "leaderboard_plot",
    "scatter_3d_params",
    "sensitivity_analysis",
    "save_leaderboard_csv",
    "export_leaderboard",
    "ensure_figure",
]
