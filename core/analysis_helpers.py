# -*- coding: utf-8 -*-
"""
core/analysis_helper.py — Advanced Analysis Toolkit (HF-grade)
==============================================================

מודול עזר לניתוח ברמת קרן גידור:

1. חשיבות פיצ'רים גלובלית באמצעות XGBoost + SHAP.
2. ניתוח PCA (קומפוננטות, explained variance, loadings).
3. Clustering (KMeans) כולל בחירת k אוטומטית לפי Silhouette.
4. שכבת הכנה וניקוי פיצ'רים (NaN, פיצ'רים קבועים, סקיילינג).

מאפיינים חשובים:
----------------
- שמירה על התאימות לאחור:
    * compute_shap_importance_df(X, y)
    * compute_pca_transform(df)
    * compute_clusters(df, k=3)

- קונפיגים מבוססי dataclass לניהול פרמטרים:
    * XGBConfig
    * PCAConfig
    * ClusterConfig
    * FeaturePrepConfig

- כל הפונקציות מיועדות ל־core בלבד ללא תלות ב־Streamlit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Literal, Union, Dict, Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# XGBoost / SHAP – imports (now required)
# ------------------------------------------------------------
from xgboost import XGBRegressor
import shap


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ============================================================
# Logger
# ============================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ============================================================
# Dataclasses for configs
# ============================================================

@dataclass
class FeaturePrepConfig:
    """
    קונפיג להכנת פיצ'רים:
    - טיפול ב־NaN
    - הסרת עמודות קבועות
    - Winsorization אופציונלי
    - Scaling אופציונלי
    """
    drop_constant: bool = True
    fillna: bool = True
    winsorize: bool = False
    winsor_q: float = 0.01  # 1% בכל זנב
    scale: bool = False     # אם True: StandardScaler


@dataclass
class XGBConfig:
    """קונפיג בסיסי ל־XGBRegressor (ניתן להרחבה בקלות)."""
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class PCAConfig:
    """קונפיג ל־PCA + preprocessing."""
    n_components: Optional[int] = None  # אם None -> min(3, n_features)
    scale: bool = True                  # להשתמש ב־StandardScaler
    whiten: bool = False                # whiten בפי־סי־איי
    drop_constant_features: bool = True


@dataclass
class ClusterConfig:
    """קונפיג ל־KMeans + אפשרות ל־auto-k."""
    k: int = 3
    auto_k: bool = False                      # אם True – ננסה לבחור k לפי Silhouette
    min_k: int = 2
    max_k: int = 10
    scale: bool = True
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 300


# ============================================================
# Internal helpers
# ============================================================

def _ensure_2d_frame(df: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """המרה ל־DataFrame + וידוא אינדקס מסודר."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame/Series, got {type(df)!r}")
    return df.copy()


def _winsorize_array(x: np.ndarray, q: float) -> np.ndarray:
    """Winsorization פשוטה לפי אחוזון q מכל צד."""
    lower = np.nanquantile(x, q)
    upper = np.nanquantile(x, 1 - q)
    return np.clip(x, lower, upper)


def _clean_features(
    X: pd.DataFrame,
    *,
    drop_constant: bool = True,
    fillna: bool = True,
    winsorize: bool = False,
    winsor_q: float = 0.01,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """
    ניקוי פיצ'רים: הסרת עמודות קבועות + מילוי NaN + Winsorization אופציונלי.
    מחזיר DataFrame נקי + רשימת פיצ'רים שנשמרו.
    """
    X = _ensure_2d_frame(X)

    # Drop constant features
    if drop_constant:
        nunique = X.nunique(dropna=False)
        keep_cols = nunique[nunique > 1].index.tolist()
        dropped = set(X.columns) - set(keep_cols)
        if dropped:
            logger.info("Dropping %d constant features: %s", len(dropped), list(dropped))
        X = X[keep_cols]
    else:
        keep_cols = X.columns.tolist()

    if X.empty:
        return X, keep_cols

    # Fill NaN with column median
    if fillna:
        medians = X.median(numeric_only=True)
        X = X.fillna(medians)

    # Winsorize numeric columns
    if winsorize:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            X_num = X[numeric_cols].to_numpy(dtype=float)
            X_num = _winsorize_array(X_num, q=winsor_q)
            X[numeric_cols] = X_num

    return X, keep_cols


def _scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Apply StandardScaler and return scaled DF + scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    return X_scaled, scaler


def prepare_features(
    df: Union[pd.DataFrame, pd.Series],
    *,
    prep_config: Optional[FeaturePrepConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    שכבת הכנה כללית לפיצ'רים: ניקוי, Winsorization וסקיילינג אופציונלי.

    מחזירה:
        X_prepared : DataFrame נקי ומוכן למודלים
        meta : dict עם מידע עזר (columns_kept, scaler וכו')
    """
    prep_config = prep_config or FeaturePrepConfig()
    X = _ensure_2d_frame(df)

    # נשמור רק עמודות נומריות לפיצ'ר אנג'ינירינג
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric columns available for feature preparation.")

    X_clean, kept_cols = _clean_features(
        X,
        drop_constant=prep_config.drop_constant,
        fillna=prep_config.fillna,
        winsorize=prep_config.winsorize,
        winsor_q=prep_config.winsor_q,
    )

    scaler: Optional[StandardScaler] = None
    if prep_config.scale:
        X_clean, scaler = _scale_features(X_clean)

    meta: Dict[str, Any] = {
        "kept_columns": list(kept_cols),
        "scaler": scaler,
        "config": prep_config,
        "n_samples": X_clean.shape[0],
        "n_features": X_clean.shape[1],
    }
    return X_clean, meta


# ============================================================
# XGBoost + SHAP feature importance
# ============================================================

def train_xgb_regressor(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    config: Optional[XGBConfig] = None,
) -> "XGBRegressor":
    """
    Train XGBRegressor with sane defaults for tabular financial features.
    """
    if XGBRegressor is None:
        raise ModuleNotFoundError("xgboost not installed. Install via `pip install xgboost`")

    config = config or XGBConfig()
    X_clean, kept_cols = _clean_features(X, drop_constant=True, fillna=True)

    if X_clean.empty:
        raise ValueError("No usable features to train XGBRegressor.")

    model = XGBRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    logger.info("Training XGBRegressor on %d samples, %d features", X_clean.shape[0], X_clean.shape[1])
    model.fit(X_clean.values, np.asarray(y))

    # שמירה של שמות הפיצ'רים על האובייקט עצמו – שימושי ל־SHAP
    setattr(model, "feature_names_in_", np.array(kept_cols, dtype=object))
    return model


def compute_shap_matrix(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    *,
    model: Optional["XGBRegressor"] = None,
    config: Optional[XGBConfig] = None,
) -> pd.DataFrame:
    """
    מחשב מטריצת SHAP מלאה (דוגמה × פיצ'ר).

    משתמש ב־XGBoost + SHAP:
        - אם model קיים – נשתמש בו.
        - אחרת נאמן מודל חדש לפי config.
    """
    if shap is None:
        raise ModuleNotFoundError("shap is not installed. Install via `pip install shap`")

    X_clean, kept_cols = _clean_features(X, drop_constant=True, fillna=True)
    if X_clean.empty:
        raise ValueError("No usable features after cleaning (all constant or NaN).")

    if model is None:
        model = train_xgb_regressor(X_clean, y, config=config)
    else:
        logger.info("Using provided XGBRegressor model for SHAP matrix.")

    logger.info("Computing SHAP values for %d samples, %d features", X_clean.shape[0], X_clean.shape[1])
    explainer = shap.Explainer(model, X_clean.values)
    shap_values = explainer(X_clean.values)

    values = shap_values.values
    # For multi-output / multi-class, average across outputs
    if values.ndim == 3:
        values = values.mean(axis=1)

    shap_df = pd.DataFrame(values, index=X_clean.index, columns=kept_cols)
    return shap_df


def compute_shap_importance_df(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    *,
    model: Optional["XGBRegressor"] = None,
    config: Optional[XGBConfig] = None,
    top_n: Optional[int] = None,
    normalize: bool = True,
    as_frame: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Train an XGBoost model (if not provided) and compute global SHAP feature importance.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : Series or ndarray
        Target.
    model : XGBRegressor, optional
        Pre-trained model. If None, a new one is trained with `train_xgb_regressor`.
    config : XGBConfig, optional
        Configuration for training when model is None.
    top_n : int, optional
        If given, return only the top N most important features.
    normalize : bool, default True
        If True, normalize importances to sum to 1.
    as_frame : bool, default False
        אם True – מחזיר DataFrame עם עמודות ["feature", "importance"].

    Returns
    -------
    pd.Series or pd.DataFrame
        Feature importance indexed by feature names, sorted descending.
    """
    # גרסה תואמת לאחור: אם קראו בלי kwargs, זה עדיין יעבוד
    if shap is None:
        raise ModuleNotFoundError("shap is not installed. Install via `pip install shap`")

    shap_df = compute_shap_matrix(X, y, model=model, config=config)
    imp = np.abs(shap_df.values).mean(axis=0)
    ser = pd.Series(imp, index=shap_df.columns)

    # Sort descending
    ser = ser.sort_values(ascending=False)

    if normalize and ser.sum() > 0:
        ser = ser / ser.sum()

    if top_n is not None and top_n > 0:
        ser = ser.head(top_n)

    if not as_frame:
        return ser

    out = pd.DataFrame(
        {"feature": ser.index.to_list(), "importance": ser.values},
        index=ser.index,
    )
    return out


# ============================================================
# PCA utilities
# ============================================================

def compute_pca_transform(
    df: Union[pd.DataFrame, pd.Series],
    *,
    config: Optional[PCAConfig] = None,
    return_model: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, PCA]]:
    """
    Fit PCA on the DataFrame and return transformed components.

    Parameters
    ----------
    df : DataFrame or Series
        Input data. Non-numeric columns are ignored.
    config : PCAConfig, optional
        Controls n_components, scaling, and constant-feature dropping.
    return_model : bool, default False
        If True, also return the fitted PCA object.

    Returns
    -------
    components_df : DataFrame
        Principal components with index aligned to input index.
    pca : PCA (optional)
        Returned only if return_model=True.
    """
    config = config or PCAConfig()
    X = _ensure_2d_frame(df)

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric columns available for PCA.")

    X, kept_cols = _clean_features(
        X,
        drop_constant=config.drop_constant_features,
        fillna=True,
        winsorize=False,
    )

    logger.info("Running PCA on %d samples, %d features", X.shape[0], X.shape[1])

    if config.scale:
        X, _ = _scale_features(X)

    n_features = X.shape[1]
    n_components = config.n_components or min(3, n_features)
    n_components = min(n_components, n_features)

    pca = PCA(n_components=n_components, whiten=config.whiten)
    Xp = pca.fit_transform(X.values)

    cols = [f"PC{i + 1}" for i in range(Xp.shape[1])]
    components_df = pd.DataFrame(Xp, index=X.index, columns=cols)

    if return_model:
        return components_df, pca
    return components_df


def summarize_pca(
    pca: PCA,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    מחזיר טבלה של loadings + explained variance לכל קומפוננטה.

    Columns:
        - component
        - feature
        - loading
        - explained_variance_ratio
    """
    loadings = pca.components_.T  # shape: (n_features, n_components)
    n_components = loadings.shape[1]
    rows = []
    for j in range(n_components):
        for i, fname in enumerate(feature_names):
            rows.append(
                {
                    "component": f"PC{j + 1}",
                    "feature": fname,
                    "loading": loadings[i, j],
                    "explained_variance_ratio": pca.explained_variance_ratio_[j],
                }
            )
    df = pd.DataFrame(rows)
    return df


# ============================================================
# Clustering utilities
# ============================================================

def _auto_choose_k(
    X: pd.DataFrame,
    *,
    min_k: int = 2,
    max_k: int = 10,
) -> int:
    """
    בוחר k אופטימלי לפי Silhouette בין min_k ל־max_k.
    אם נכשלים (מעט דוגמאות וכו') – חוזרים ל־min_k.
    """
    n_samples = X.shape[0]
    if n_samples <= min_k:
        logger.warning(
            "Not enough samples (%d) for silhouette-based auto-k; using k=%d",
            n_samples,
            min_k,
        )
        return min_k

    best_k = min_k
    best_score = -1.0
    for k in range(min_k, min(max_k, n_samples - 1) + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as exc:  # pragma: no cover
            logger.warning("Silhouette computation failed for k=%d: %s", k, exc)
            continue

    logger.info("Auto-selected k=%d (silhouette=%.4f)", best_k, best_score)
    return best_k


def compute_clusters(
    df: Union[pd.DataFrame, pd.Series],
    k: int = 3,
    *,
    config: Optional[ClusterConfig] = None,
    return_model: bool = False,
    return_profile: bool = False,
) -> Union[pd.Series, Tuple[pd.Series, KMeans], Tuple[pd.Series, KMeans, pd.DataFrame]]:
    """
    Perform KMeans clustering and return cluster labels as a pandas Series.

    Parameters
    ----------
    df : DataFrame or Series
        Data to cluster (numeric columns only).
    k : int, default 3
        Number of clusters (ignored if config.auto_k=True).
    config : ClusterConfig, optional
        Clustering configuration (k/auto_k/scale).
    return_model : bool, default False
        If True, also return the fitted KMeans model.
    return_profile : bool, default False
        אם True – נחזיר גם טבלת פרופיל אשכולות (ממוצעים וכו').

    Returns
    -------
    labels : Series
        Cluster labels, aligned with df.index.
    model : KMeans (optional)
    profile : DataFrame (optional)
        אם return_profile=True, טבלת ממוצעים/סטיות תקן לכל אשכול.
    """
    config = config or ClusterConfig(k=k)
    X = _ensure_2d_frame(df)

    X = X.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric columns available for clustering.")

    X_clean, kept_cols = _clean_features(
        X, drop_constant=True, fillna=True, winsorize=False
    )
    if X_clean.empty:
        raise ValueError("No usable features for clustering after cleaning.")

    if config.scale:
        X_clean, _ = _scale_features(X_clean)

    if config.auto_k:
        k_eff = _auto_choose_k(
            X_clean,
            min_k=config.min_k,
            max_k=config.max_k,
        )
    else:
        k_eff = config.k

    logger.info("Running KMeans clustering with k=%d on %d samples, %d features",
                k_eff, X_clean.shape[0], X_clean.shape[1])

    km = KMeans(
        n_clusters=k_eff,
        random_state=config.random_state,
        n_init=config.n_init,
        max_iter=config.max_iter,
    )
    labels_arr = km.fit_predict(X_clean)
    labels = pd.Series(labels_arr, index=X_clean.index, name="cluster")

    if not (return_model or return_profile):
        return labels

    outputs: Tuple[Any, ...] = (labels, km)  # type: ignore[assignment]

    if return_profile:
        profile_rows = []
        df_numeric = X_clean.copy()
        df_numeric["cluster"] = labels
        grouped = df_numeric.groupby("cluster")
        for cluster_id, grp in grouped:
            stats = grp.drop(columns=["cluster"]).agg(["mean", "std"])
            for feature in stats.columns:
                profile_rows.append(
                    {
                        "cluster": cluster_id,
                        "feature": feature,
                        "mean": stats.loc["mean", feature],
                        "std": stats.loc["std", feature],
                        "n": len(grp),
                    }
                )
        profile_df = pd.DataFrame(profile_rows)
        outputs = (labels, km, profile_df)

    return outputs  # type: ignore[return-value]


# ============================================================
# Public exports
# ============================================================

__all__ = [
    # Configs
    "FeaturePrepConfig",
    "XGBConfig",
    "PCAConfig",
    "ClusterConfig",
    # Prep
    "prepare_features",
    # XGB + SHAP
    "train_xgb_regressor",
    "compute_shap_matrix",
    "compute_shap_importance_df",
    # PCA
    "compute_pca_transform",
    "summarize_pca",
    # Clustering
    "compute_clusters",
]
