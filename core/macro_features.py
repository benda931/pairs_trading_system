# -*- coding: utf-8 -*-
"""
core/macro_features.py ג€” ׳©׳›׳‘׳× ׳₪׳™׳¦'׳¨׳™׳ ׳׳׳§׳¨׳• (MVP)
--------------------------------------------------
׳׳˜׳¨׳× ׳”׳§׳•׳‘׳¥: ׳׳§׳‘׳ ׳׳₪׳” ׳©׳ ׳׳™׳ ׳“׳™׳§׳˜׳•׳¨׳™׳ ׳׳׳§׳¨׳• (id -> DataFrame ׳¢׳ ׳¢׳׳•׳“׳” 'value'
׳•׳׳™׳ ׳“׳§׳¡ ׳–׳׳) ׳•׳׳”׳—׳–׳™׳¨ ׳׳˜׳¨׳™׳¦׳× ׳₪׳™׳¦'׳¨׳™׳ ׳׳—׳™׳“׳”: ־” (׳©׳™׳ ׳•׳™׳™׳), yoy, ׳–׳™""׳¡׳§׳•׳¨
׳׳•׳×׳׳ ׳—׳׳•׳, spreads ׳׳•׳₪׳¦׳™׳•׳ ׳׳™׳™׳, ׳•-PCA ׳׳•׳₪׳¦׳™׳•׳ ׳׳™ ׳¢׳ ׳¡׳ ׳₪׳™׳¦'׳¨׳™׳.

Public API:
- build_features(df_map: dict[str, pd.DataFrame], params: dict) -> pd.DataFrame

׳§׳׳˜ ׳¦׳₪׳•׳™:
- df_map[id] = DataFrame(index=DatetimeIndex, columns=['value'])
- params (׳׳•׳₪׳¦׳™׳•׳ ׳׳™):
    {
      'z_window': 60,
      'z_winsor': 3.0,
      'mom_windows': [20, 60],      # ׳™׳׳™׳ ~ 1M, 3M
      'yoy_window': 252,            # ׳™׳׳™׳ ~ 1Y
      'spreads': [
          {'a': 'hyg', 'b': 'lqd', 'name': 'hyg_minus_lqd'},
      ],
      'pca': {
          'n_components': 3,
          'on_columns': ['*_z'],   # glob-like; ׳׳ ׳¨׳™׳§ ג†’ ׳¢׳ ׳›׳ ׳”׳₪׳™׳¦'׳¨׳™׳ ׳”׳׳¡׳₪׳¨׳™׳™׳
      }
    }

׳”׳—׳–׳¨: DataFrame ׳׳׳•׳—׳“ (index=DatetimeIndex), ׳©׳׳•׳× ׳׳׳•׳–׳ ׳™׳, ׳׳׳ NaN (ffill/bfill ׳¢׳“׳™׳).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("core.macro_features")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


# ==========================
# Helpers
# ==========================

def _finite(s: pd.Series) -> pd.Series:
    return s[np.isfinite(s)] if hasattr(np, "isfinite") else s

def _ensure_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if not isinstance(df.index, pd.DatetimeIndex):
        # ׳ ׳¡׳” ׳׳–׳”׳•׳× ׳¢׳׳•׳“׳× ׳×׳׳¨׳™׳
        date_col = None
        for c in df.columns:
            if str(c).lower() in ("date", "time", "timestamp"):
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df = df.set_index(pd.to_datetime(df[date_col])).drop(columns=[date_col], errors="ignore")
    ser = df.get("value")
    if ser is None:
        # ׳§׳— ׳׳× ׳”׳¢׳׳•׳“׳” ׳”׳׳¡׳₪׳¨׳™׳× ׳”׳¨׳׳©׳•׳ ׳”
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        ser = df[num_cols[0]] if num_cols else pd.Series(dtype=float)
    return pd.to_numeric(ser, errors="coerce").astype(float)


def _zscore(s: pd.Series, window: int, winsor: float) -> pd.Series:
    if s.empty:
        return s
    z = s.rolling(window, min_periods=max(5, window // 5)).apply(
        lambda w: (w.iloc[-1] - w.mean()) / (w.std(ddof=0) + 1e-9), raw=False
    )
    return z.clip(-winsor, winsor)


def _pct_change_annualized(s: pd.Series, window: int) -> pd.Series:
    # ׳©׳™׳ ׳•׳™ ׳׳¦׳˜׳‘׳¨ ׳‘׳—׳׳•׳ ג†’ ׳׳ ׳•׳׳׳™׳–׳¦׳™׳” ׳‘׳§׳™׳¨׳•׳‘ ׳-252 ׳™׳׳™ ׳׳¡׳—׳¨
    ret = s.pct_change(window)
    return (1.0 + ret).pow(252.0 / max(1.0, float(window))) - 1.0


def _align_union(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    if not series_map:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    # ׳׳™׳—׳•׳“ ׳׳™׳ ׳“׳§׳¡׳™׳ + ffill ׳§׳
    idx = None
    for s in series_map.values():
        idx = s.index if idx is None else idx.union(s.index)
    out = {}
    for k, s in series_map.items():
        out[k] = s.reindex(idx).sort_index().ffill().bfill()
    return pd.DataFrame(out)


# ==========================
# Main API
# ==========================

def _ewma(s: pd.Series, span: int) -> pd.Series:
    if s.empty:
        return s
    return s.ewm(span=span, adjust=False, min_periods=max(5, span // 5)).mean()


def _rolling_vol(s: pd.Series, window: int) -> pd.Series:
    if s.empty:
        return s
    return s.pct_change().rolling(window, min_periods=max(5, window // 5)).std(ddof=0) * np.sqrt(252.0)


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    if s.empty:
        return s
    return s.rolling(window, min_periods=max(5, window // 5)).mean()


def _rolling_skew(s: pd.Series, window: int) -> pd.Series:
    if s.empty:
        return s
    return s.rolling(window, min_periods=max(5, window // 5)).skew()


def _rolling_kurt(s: pd.Series, window: int) -> pd.Series:
    if s.empty:
        return s
    return s.rolling(window, min_periods=max(5, window // 5)).kurt()


def _robust_zscore(s: pd.Series, window: int, winsor: float) -> pd.Series:
    if s.empty:
        return s
    def last_rz(w: pd.Series) -> float:
        med = float(w.median())
        mad = float((w - med).abs().median()) + 1e-9
        z = (float(w.iloc[-1]) - med) / (1.4826 * mad)
        return max(min(z, winsor), -winsor)
    return s.rolling(window, min_periods=max(5, window // 5)).apply(last_rz, raw=False)


def _rolling_rank_pct(s: pd.Series, window: int) -> pd.Series:
    if s.empty:
        return s
    def last_rank_pct(w: pd.Series) -> float:
        r = w.rank(pct=True)
        return float(r.iloc[-1])
    return s.rolling(window, min_periods=max(5, window // 5)).apply(last_rank_pct, raw=False)


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    idx = a.index.union(b.index)
    sa = a.reindex(idx).ffill()
    sb = b.reindex(idx).ffill()
    return sa.rolling(window, min_periods=max(5, window // 5)).corr(sb)


def build_features(df_map: Dict[str, pd.DataFrame], params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """׳‘׳•׳ ׳” ׳׳˜׳¨׳™׳¦׳× ׳₪׳™׳¦'׳¨׳™׳ ׳¢׳ **׳׳™׳ ׳“׳™׳§׳˜׳•׳¨׳™׳ ׳׳׳§׳¨׳• ׳‘׳׳‘׳“**.

    ׳©׳™׳ ׳׳‘: ׳›׳ ׳”׳₪׳™׳¦'׳¨׳™׳ ׳›׳׳ ׳׳—׳•׳©׳‘׳™׳ ׳¢׳ ׳¡׳“׳¨׳•׳× ׳׳׳§׳¨׳• (value) ג€” **׳׳** ׳¢׳ ׳¡׳₪׳¨׳“׳™׳ ׳©׳ ׳–׳•׳’׳•׳×.
    ׳ ׳™׳×׳•׳— ׳¡׳˜׳˜׳™׳¡׳˜׳™ ׳©׳ ׳¡׳₪׳¨׳“ ׳–׳•׳’׳•׳× ׳©׳™׳™׳ ׳׳©׳›׳‘׳•׳× ׳׳—׳¨׳•׳× ׳‘׳§׳•׳“ (backtester/signals).
    """
    params = dict(params or {})
    z_window = int(params.get("z_window", 60))
    z_winsor = float(params.get("z_winsor", 3.0))
    mom_windows: List[int] = list(params.get("mom_windows", [20, 60]))
    yoy_window = int(params.get("yoy_window", 252))
    spreads = list(params.get("spreads", []))
    pca_cfg = dict(params.get("pca", {})) if params.get("pca") else None
    vol_windows: List[int] = list(params.get("vol_windows", [20]))
    ewma_spans: List[int] = list(params.get("ewma_spans", []))
    mean_windows: List[int] = list(params.get("rolling_mean_windows", []))
    rank_windows: List[int] = list(params.get("rank_windows", []))
    lags: List[int] = list(params.get("lags", []))
    corr_pairs: List[Dict[str, Any]] = list(params.get("corr_pairs", []))
    seasonal_dummies: bool = bool(params.get("seasonal_dummies", False))
    # ׳׳¦׳‘ ׳—׳™׳©׳•׳‘ ׳׳₪׳™׳¦'׳¨׳™׳ ׳׳¡׳•׳™׳׳™׳ (׳‘׳¨׳™׳¨׳× ׳׳—׳“׳ ׳׳׳§׳¨׳•):
    #   vol_on: 'returns'|'levels' ג€” ׳×׳ ׳•׳“׳×׳™׳•׳× ׳¢׳ ׳×׳©׳•׳׳•׳× (׳‘׳¨׳™׳¨׳× ׳׳—׳“׳) ׳׳• ׳¢׳ ׳¨׳׳•׳×
    #   mean_on/ewma_on: 'levels'|'returns' ג€” ׳׳׳•׳¦׳¢׳™׳ ׳¢׳ ׳¨׳׳•׳× (׳‘׳¨׳™׳¨׳× ׳׳—׳“׳) ׳׳• ׳¢׳ ׳×׳©׳•׳׳•׳×
    vol_on = str(params.get("vol_on", "returns"))
    mean_on = str(params.get("mean_on", "levels"))
    ewma_on = str(params.get("ewma_on", "levels"))
    # ׳”׳¨׳—׳‘׳•׳×
    skew_windows: List[int] = list(params.get("skew_windows", []))
    kurt_windows: List[int] = list(params.get("kurt_windows", []))
    robust_z: bool = bool(params.get("robust_z", False))
    robust_window: int = int(params.get("robust_window", z_window))
    robust_winsor: float = float(params.get("robust_winsor", z_winsor))
    quality_window: int = int(params.get("quality_window", 60))
    # change-point heuristic
    cpd_cfg: Optional[Dict[str, Any]] = params.get("cpd")
    # ׳™׳¦׳™׳׳•׳× ׳›׳׳׳™׳•׳×
    standardize: bool = bool(params.get("standardize", False))
    clip_range = params.get("clip_range")  # tuple(low, high) or None
    prefix: Optional[str] = params.get("prefix")
    dropna_strategy: str = str(params.get("dropna_strategy", "ffill"))


    # 1) normalize to series & align
    base_series: Dict[str, pd.Series] = {}
    for key, df in df_map.items():
        base_series[key] = _ensure_series(df)
    base_df = _align_union(base_series)
    if base_df.empty:
        return base_df

    feats: Dict[str, pd.Series] = {}

    # 2) raw values
    for col in base_df.columns:
        feats[f"{col}_val"] = base_df[col]

    # 3) momentum windows (־”, annualized)
    for w in mom_windows:
        for col in base_df.columns:
            feats[f"{col}_mom{w}"] = _pct_change_annualized(base_df[col], w)

    # 4) yoy change
    for col in base_df.columns:
        feats[f"{col}_yoy"] = base_df[col].pct_change(yoy_window)

    # 5) z-score
    for col in base_df.columns:
        feats[f"{col}_z"] = _zscore(base_df[col], z_window, z_winsor)
        if robust_z:
            feats[f"{col}_rz"] = _robust_zscore(base_df[col], robust_window, robust_winsor)

    # 6) rolling volatility / mean / ranks / EWMA
    for w in vol_windows:
        for col in base_df.columns:
            base_series_for_vol = base_df[col] if vol_on == "levels" else base_df[col].pct_change()
            feats[f"{col}_vol{w}"] = base_series_for_vol.rolling(w, min_periods=max(5, w // 5)).std(ddof=0) * np.sqrt(252.0)
    for w in mean_windows:
        for col in base_df.columns:
            base_series_for_mean = base_df[col] if mean_on == "levels" else base_df[col].pct_change()
            feats[f"{col}_mean{w}"] = _rolling_mean(base_series_for_mean, w)
    for w in rank_windows:
        for col in base_df.columns:
            feats[f"{col}_rankpct{w}"] = _rolling_rank_pct(base_df[col], w)
    # skew/kurt (׳¢׳ ׳׳•׳×׳” ׳¡׳“׳¨׳× ׳‘׳¡׳™׳¡ ׳›׳׳• mean_on/ewma_on? ׳ ׳©׳׳™׳¨ ׳‘׳¨׳׳•׳×)
    for w in skew_windows:
        for col in base_df.columns:
            feats[f"{col}_skew{w}"] = _rolling_skew(base_df[col], w)
    for w in kurt_windows:
        for col in base_df.columns:
            feats[f"{col}_kurt{w}"] = _rolling_kurt(base_df[col], w)
    for span in ewma_spans:
        for col in base_df.columns:
            base_series_for_ewma = base_df[col] if ewma_on == "levels" else base_df[col].pct_change()
            feats[f"{col}_ewma{span}"] = _ewma(base_series_for_ewma, span)

    # 7) spreads (a - b)
    for sp in spreads:
        try:
            a = str(sp.get("a"))
            b = str(sp.get("b"))
            name = str(sp.get("name", f"{a}_minus_{b}"))
            if a in base_df.columns and b in base_df.columns:
                feats[f"{name}_val"] = base_df[a] - base_df[b]
                feats[f"{name}_z"] = _zscore(feats[f"{name}_val"], z_window, z_winsor)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("spread build failed for %s: %s", sp, e)

    # 8) rolling correlations
    for cp in corr_pairs:
        try:
            a = str(cp.get("a")); b = str(cp.get("b"))
            w = int(cp.get("window", 60))
            name = str(cp.get("name", f"corr_{a}_{b}_{w}"))
            if a in base_df.columns and b in base_df.columns:
                feats[name] = _rolling_corr(base_df[a], base_df[b], w)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("corr build failed for %s: %s", cp, e)

    # 9) lags
    for L in lags:
        for col in base_df.columns:
            feats[f"{col}_lag{L}"] = base_df[col].shift(L)

    feat_df = pd.DataFrame(feats).sort_index().ffill().bfill()

    # 9.5) change-point heuristic (׳₪׳©׳•׳˜): ׳”׳₪׳¨׳© ׳׳׳•׳¦׳¢׳™׳ ׳ ׳¢׳™׳ ׳׳•׳ ׳¡׳˜׳™׳™׳× ׳×׳§׳ ׳‘׳—׳׳•׳
    if cpd_cfg:
        try:
            w = int(cpd_cfg.get("window", 60))
            k = float(cpd_cfg.get("k", 2.0))
            for col in base_df.columns:
                mu = _rolling_mean(base_df[col], w)
                sig = base_df[col].rolling(w, min_periods=max(5, w // 5)).std(ddof=0) + 1e-9
                flag = (mu.diff().abs() > k * sig).astype(float)
                feat_df[f"{col}_cpd{w}"] = flag
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("cpd heuristic skipped: %s", e)

    # 10) seasonal dummies (month of year)
    if seasonal_dummies and not feat_df.empty:
        months = feat_df.index.month
        for m in range(1, 13):
            feat_df[f"month_{m:02d}"] = (months == m).astype(float)

    # 11) quality (finite/missing) ׳¢׳ ׳—׳׳•׳
    if quality_window > 0 and not feat_df.empty:
        for col in base_df.columns:
            fin = base_df[col].apply(lambda x: float(np.isfinite(x)))
            miss = base_df[col].apply(lambda x: float(not np.isfinite(x)))
            feat_df[f"{col}_finite{quality_window}"] = fin.rolling(quality_window, min_periods=1).mean()
            feat_df[f"{col}_missing{quality_window}"] = miss.rolling(quality_window, min_periods=1).mean()

    # 12) PCA (optional)
    if pca_cfg:
        try:
            from sklearn.decomposition import PCA  # type: ignore
            import fnmatch
            cols = feat_df.columns
            patt = pca_cfg.get("on_columns") or []
            if patt:
                mask = np.zeros(len(cols), dtype=bool)
                for p in patt:
                    mask = mask | np.array([fnmatch.fnmatch(c, p) for c in cols])
                use_cols = list(cols[mask])
            else:
                use_cols = [c for c in cols if np.issubdtype(feat_df[c].dtype, np.number)]
            n_comp = int(pca_cfg.get("n_components", 3))
            X = feat_df[use_cols].fillna(0.0).values
            pca = PCA(n_components=min(n_comp, X.shape[1]))
            pcs = pca.fit_transform(X)
            for i in range(pcs.shape[1]):
                feat_df[f"pca_{i+1}"] = pcs[:, i]
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("PCA skipped: %s", e)

    # 13) standardize / clip / prefix / dropna-policy
    if standardize:
        try:
            num_cols = [c for c in feat_df.columns if np.issubdtype(feat_df[c].dtype, np.number)]
            for c in num_cols:
                mu = float(feat_df[c].mean())
                sd = float(feat_df[c].std(ddof=0)) + 1e-9
                feat_df[c] = (feat_df[c] - mu) / sd
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("standardize skipped: %s", e)
    if clip_range and isinstance(clip_range, (list, tuple)) and len(clip_range) == 2:
        try:
            feat_df = feat_df.clip(lower=clip_range[0], upper=clip_range[1])
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("clip skipped: %s", e)
    if prefix:
        feat_df = feat_df.add_prefix(str(prefix))

    if dropna_strategy == "ffill":
        feat_df = feat_df.ffill()
    elif dropna_strategy == "bfill":
        feat_df = feat_df.bfill()
    elif dropna_strategy == "zero":
        feat_df = feat_df.fillna(0.0)
    elif dropna_strategy == "drop":
        feat_df = feat_df.dropna()

    return feat_df


def build_features_with_meta(df_map: Dict[str, pd.DataFrame], params: Optional[Dict[str, Any]] = None) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """׳¢׳˜׳™׳₪׳” ׳©׳׳—׳–׳™׳¨׳” ׳’׳ ׳׳˜׳ג€‘׳“׳׳˜׳”: ׳©׳׳•׳×, ׳₪׳¨׳׳˜׳¨׳™׳, ׳˜׳•׳•׳— ׳×׳׳¨׳™׳›׳™׳ ׳•׳׳¡׳₪׳¨ ׳₪׳™׳¦'׳¨׳™׳."""
    df = build_features(df_map, params=params)
    meta: Dict[str, Any] = {
        "n_features": int(df.shape[1]),
        "columns": list(df.columns),
        "params_used": dict(params or {}),
        "index_start": (None if df.empty else str(df.index[0])),
        "index_end": (None if df.empty else str(df.index[-1])),
    }
    return df, meta


__all__ = ["build_features", "build_features_with_meta"]
