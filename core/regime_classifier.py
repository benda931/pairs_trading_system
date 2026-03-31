# -*- coding: utf-8 -*-
"""
core/regime_classifier.py — RegimeModel (HMM/KMeans/XGB, guarded)
-----------------------------------------------------------------
מטרה: לספק ממשק אחיד לחיזוי הסתברויות משטרים מתוך מטריצת פיצ'רים מאקרו.
המודול כתוב בגישת "guarded imports" כך שיעבוד גם אם תלויות מסוימות אינן מותקנות.

Public API:
- class RegimeModel:
    def __init__(self, mode: str = "hmm", labels: list[str] | None = None, **kwargs)
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RegimeModel"
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame

Modes (בוחרים בפרמטר mode):
- "hmm"   : GaussianHMM (אם hmmlearn מותקן), אחרת fallback ל-kmeans.
- "kmeans": KMeans לא-מפוקח + המרה הסתברותית עם softmax על מרחקים.
- "xgb"   : מודל מפוקח אם יש y (אם xgboost מותקן), אחרת fallback ל-kmeans.

Notes:
- X צפוי להיות DataFrame עם אינדקס זמן ועמודות פיצ'רים (מספריות).
- אם לא סופק labels — נשתמש בברירת מחדל: ["risk_off","neutral","risk_on"].
- predict_proba מחזירה DataFrame עם index=X.index ועמודות=labels (סכום כל שורה ≈ 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("core.regime_classifier")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)

# --------- Guarded imports ---------
_has_hmm = False
_has_xgb = False
_has_skl = False
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    _has_hmm = True
except Exception:
    _has_hmm = False

try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    _has_skl = True
except Exception:
    _has_skl = False

try:
    import xgboost as xgb  # type: ignore
    _has_xgb = True
except Exception:
    _has_xgb = False


# --------- Utilities ---------
_DEFAULT_LABELS = ["risk_off", "neutral", "risk_on"]


def _to_matrix(X: pd.DataFrame) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    Z = X.select_dtypes(include=[np.number]).astype(float).values
    if Z.size == 0:
        raise ValueError("X has no numeric columns")
    return Z


def _softmax_negdist(dist: np.ndarray, axis: int = 1, eps: float = 1e-9) -> np.ndarray:
    # קטֵן מרחק → הסתברות גבוהה
    inv = 1.0 / (dist + eps)
    inv /= inv.sum(axis=axis, keepdims=True) + eps
    return inv


@dataclass
class RegimeModel:
    mode: str = "hmm"  # "hmm" | "kmeans" | "xgb"
    labels: Optional[List[str]] = None
    n_states: int = 3
    random_state: int = 42
    pca_components: int = 0  # 0=כבוי

    # fitted artifacts
    _scaler: Any = None
    _pca: Any = None
    _hmm: Any = None
    _kmeans: Any = None
    _xgb: Any = None

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = list(_DEFAULT_LABELS[: self.n_states])
        else:
            if len(self.labels) != self.n_states:
                raise ValueError("len(labels) must equal n_states")
        if not _has_skl and self.mode in ("kmeans", "xgb"):
            LOGGER.warning("sklearn not available — falling back to HMM/kmeans fallback")
        if self.mode == "hmm" and not _has_hmm:
            LOGGER.warning("hmmlearn not available — falling back to kmeans")
            self.mode = "kmeans"
        if self.mode == "xgb" and not _has_xgb:
            LOGGER.warning("xgboost not available — falling back to kmeans")
            self.mode = "kmeans"

    # ------------- Fit -------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RegimeModel":
        Xn = X.select_dtypes(include=[np.number]).astype(float)
        if Xn.empty:
            raise ValueError("X has no numeric features")
        # scale
        if _has_skl:
            self._scaler = StandardScaler()
            Z = self._scaler.fit_transform(Xn.values)
            if self.pca_components and self.pca_components > 0:
                self._pca = PCA(n_components=min(self.pca_components, Z.shape[1]))
                Z = self._pca.fit_transform(Z)
        else:
            Z = _to_matrix(Xn)
        # fit mode
        if self.mode == "hmm" and _has_hmm:
            self._hmm = GaussianHMM(n_components=self.n_states, covariance_type="diag", random_state=self.random_state)
            self._hmm.fit(Z)
        elif self.mode == "xgb" and _has_xgb and y is not None:
            # מפוקח: תיוג חייב להיות באורך X
            yv = pd.Series(y).reindex(Xn.index)
            if yv.isna().any():
                yv = yv.fillna(method="ffill").fillna(method="bfill").fillna(0)
            dtrain = xgb.DMatrix(Z, label=yv.values)
            params = {"objective": "multi:softprob", "num_class": self.n_states, "seed": self.random_state, "verbosity": 0}
            self._xgb = xgb.train(params, dtrain, num_boost_round=100)
        else:
            # kmeans fallback
            if not _has_skl:
                raise RuntimeError("sklearn is required for kmeans fallback")
            self._kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state, n_init=10)
            self._kmeans.fit(Z)
        return self

    # ------------- Predict Proba -------------
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = X.select_dtypes(include=[np.number]).astype(float)
        if Xn.empty:
            raise ValueError("X has no numeric features")
        # transform like fit
        if self._scaler is not None:
            Z = self._scaler.transform(Xn.values)
            if self._pca is not None:
                Z = self._pca.transform(Z)
        else:
            Z = _to_matrix(Xn)
        # mode logic
        if self.mode == "hmm" and self._hmm is not None:
            # hmmlearn אינו מוציא softprobs בקלות; נשתמש ב-frame log prob כ-approx
            # בתור קירוב: תגיות ואח"כ החלקה עם חלון קצר ל-onehot
            tags = self._hmm.predict(Z)
            P = np.zeros((len(tags), self.n_states), dtype=float)
            for i, t in enumerate(tags):
                P[i, int(t)] = 1.0
            # החלקה קטנה
            if P.shape[0] > 3:
                k = min(5, P.shape[0])
                P = pd.DataFrame(P).rolling(k, min_periods=1).mean().values
        elif self.mode == "xgb" and self._xgb is not None:
            dtest = xgb.DMatrix(Z)
            P = self._xgb.predict(dtest)
        elif self._kmeans is not None:
            # הסתברות רכה לפי מרחק למרכזים
            centers = self._kmeans.cluster_centers_
            dists = np.linalg.norm(Z[:, None, :] - centers[None, :, :], axis=2)
            P = _softmax_negdist(dists)
        else:
            # Fallback uniform
            P = np.ones((Z.shape[0], self.n_states), dtype=float) / float(self.n_states)
        # build DataFrame
        cols = list(self.labels or _DEFAULT_LABELS[: self.n_states])
        proba_df = pd.DataFrame(P, index=Xn.index, columns=cols)
        # normalize strictly
        s = proba_df.sum(axis=1)
        proba_df = proba_df.div(s.replace(0, 1.0), axis=0)
        return proba_df


# --------- Enhancements: smoothing / labeling / factory / IO ---------

@dataclass
class RegimeConfig:
    mode: str = "hmm"              # "hmm" | "kmeans" | "xgb"
    n_states: int = 3
    labels: Optional[List[str]] = None
    random_state: int = 42
    pca_components: int = 0         # 0=off
    smooth_window: int = 5          # post smoothing of probabilities (rolling mean)
    hysteresis: int = 0             # min consecutive steps before switching label


def _smooth_probs(P: pd.DataFrame, k: int) -> pd.DataFrame:
    if k and k > 1:
        return P.rolling(k, min_periods=1).mean()
    return P


def _labels_from_probs(P: pd.DataFrame, labels: List[str], hysteresis: int = 0) -> pd.Series:
    idxmax = P.values.argmax(axis=1)
    lab = pd.Series([labels[i] for i in idxmax], index=P.index)
    if hysteresis and hysteresis > 1:
        # require consecutive steps before switching
        cur = None
        count = 0
        out = []
        for v in lab.values:
            if cur is None:
                cur, count = v, 1
            elif v == cur:
                count += 1
            else:
                count = 1
                cur = v
            if count < hysteresis:
                out.append(out[-1] if out else v)
            else:
                out.append(v)
        lab = pd.Series(out, index=lab.index)
    return lab


def regime_model_from_config(cfg: RegimeConfig) -> RegimeModel:
    return RegimeModel(
        mode=cfg.mode,
        labels=cfg.labels or list(_DEFAULT_LABELS[: cfg.n_states]),
        n_states=cfg.n_states,
        random_state=cfg.random_state,
        pca_components=cfg.pca_components,
    )


def fit_and_predict(
    X_train: pd.DataFrame,
    X_pred: Optional[pd.DataFrame] = None,
    cfg: Optional[RegimeConfig] = None,
    y_train: Optional[pd.Series] = None,
) -> tuple[RegimeModel, pd.DataFrame, pd.Series]:
    """Utility: fit a model and return (model, smoothed proba, labels).
    X_pred defaults to X_train if not provided.
    """
    cfg = cfg or RegimeConfig()
    model = regime_model_from_config(cfg)
    model.fit(X_train, y=y_train)
    Xp = X_pred if X_pred is not None else X_train
    P = model.predict_proba(Xp)
    P = _smooth_probs(P, cfg.smooth_window)
    labels = _labels_from_probs(P, model.labels or _DEFAULT_LABELS[: cfg.n_states], hysteresis=cfg.hysteresis)
    return model, P, labels


def save_model(model: RegimeModel, path: str) -> None:
    """שומר את המודל לקובץ (pickle, guarded)."""
    try:
        import pickle  # noqa: S403
        with open(path, "wb") as f:
            pickle.dump(model, f)  # noqa: S301
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("save_model failed: %s", e)


def load_model(path: str) -> Optional[RegimeModel]:
    try:
        import pickle  # noqa: S403
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if isinstance(obj, RegimeModel):
            return obj
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("load_model failed: %s", e)
    return None


__all__ = ["RegimeModel", "RegimeConfig", "regime_model_from_config", "fit_and_predict", "save_model", "load_model"]
