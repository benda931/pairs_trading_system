# -*- coding: utf-8 -*-
"""
root/api_server.py — Fair Value HTTP API (FastAPI, v1)
=====================================================

Thin HTTP layer on top of:
- core.fair_value_engine.FairValueEngine
- core.fair_value_optimizer_v2.optimize_fair_value
- core.fair_value_advisor.analyze_engine_rows

Endpoints:
- GET  /health
- POST /engine/run
- POST /optimizer/run
- POST /advisor/run

Async jobs/artifacts endpoints מוחזרים כרגע כ-not_implemented.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, RootModel, field_validator



# ==== Imports from core ====
try:
    from core.fair_value_engine import Config as EngineConfig, FairValueEngine  # type: ignore
except Exception as exc:
    raise RuntimeError(f"Failed to import core.fair_value_engine: {exc}") from exc

try:
    from core.fair_value_optimizer_v2 import (
        OptConfig as CoreOptConfig,
        optimize_fair_value,
    )  # type: ignore
except Exception as exc:
    raise RuntimeError(f"Failed to import core.fair_value_optimizer_v2: {exc}") from exc

try:
    from core.fair_value_advisor import analyze_engine_rows  # type: ignore
except Exception as exc:
    raise RuntimeError(f"Failed to import core.fair_value_advisor: {exc}") from exc


# =========================
# Pydantic models (API)
# =========================

class ErrorModel(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class EngineMetaModel(BaseModel):
    asof: datetime
    window: int
    secondary_windows: List[int] = Field(default_factory=list)
    log_mode: bool
    ensemble_mode: Optional[str] = None


class PairModel(RootModel[Tuple[str, str]]):
    """
    RootModel עבור זוג (y,x), כדי ש-JSON מהצורה ["SPY","QQQ"]
    יפורש כ-(y="SPY", x="QQQ").
    """

    @property
    def y(self) -> str:
        return self.root[0]

    @property
    def x(self) -> str:
        return self.root[1]

    def as_tuple(self) -> Tuple[str, str]:
        return self.root



class EngineConfigOverridesModel(BaseModel):
    # Subset of core.Config, matching openapi.yaml names
    window: Optional[int] = None
    min_overlap: Optional[int] = None
    log_mode: Optional[bool] = None

    secondary_windows: Optional[List[int]] = None
    use_winsor: Optional[bool] = None
    winsor_p: Optional[float] = None
    zscore_clip: Optional[List[float]] = None
    volatility_adjust: Optional[bool] = None

    use_returns_for_corr: Optional[bool] = None
    use_returns_for_dcor: Optional[bool] = None

    ensemble_mode: Optional[str] = None
    ensemble_target: Optional[str] = None

    mean_revert_pvalue: Optional[float] = None

    coint_pvalue: Optional[float] = None
    residual_adf_enabled: Optional[bool] = None

    borrow_bps: Optional[float] = None
    costs_bps: Optional[float] = None
    slippage_bps: Optional[float] = None

    target_vol_ann: Optional[float] = None
    kelly_fraction: Optional[float] = None
    max_leverage: Optional[float] = None

    use_winsor_for_z: Optional[bool] = None
    z_in: Optional[float] = None
    z_out: Optional[float] = None

    beta_mode: Optional[str] = None
    kalman_q: Optional[float] = None
    kalman_r: Optional[float] = None

    volatility_regime_windows: Optional[List[int]] = None
    volatility_regime_labels: Optional[List[str]] = None

    @field_validator("zscore_clip")
    @classmethod
    def _validate_zscore_clip(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None and len(v) != 2:
            raise ValueError("zscore_clip must be length 2 [low, high]")
        return v


class EngineRunRequest(BaseModel):
    timeIndex: List[datetime]
    pricesWide: Dict[str, List[float]]
    pairs: List[PairModel]
    configOverrides: Optional[EngineConfigOverridesModel] = None

    @field_validator("pricesWide")
    @classmethod
    def _validate_prices(
        cls,
        v: Dict[str, List[float]],
        info: Any,
    ) -> Dict[str, List[float]]:
        n = None
        for arr in v.values():
            if n is None:
                n = len(arr)
            elif len(arr) != n:
                raise ValueError("All price series in pricesWide must have same length")

        # timeIndex כבר עבר ולידציה כי הוא מופיע לפני pricesWide במודל
        ti = getattr(info, "data", {}).get("timeIndex") if info is not None else None
        if ti is not None and n is not None and len(ti) != n:
            raise ValueError("timeIndex length must match pricesWide series length")

        return v




class PairRowModel(BaseModel):
    pair: PairModel
    window: int
    action: Optional[str] = None
    mispricing: float
    vol_adj_mispricing: Optional[float] = None
    zscore: Optional[float] = None
    band_p95: Optional[float] = None
    band_upper: Optional[float] = None
    band_lower: Optional[float] = None
    halflife: Optional[float | str] = None
    rolling_corr: Optional[float] = None
    distance_corr: Optional[float] = None
    adf_p: Optional[float] = None
    residual_adf_p: Optional[float] = None
    is_coint: Optional[bool] = None
    y_fair: float
    target_pos_units: float
    cost_spread_units: float
    rp_weight: Optional[float] = None
    sr_net: Optional[float] = None
    psr_net: Optional[float] = None
    dsr_net: Optional[float] = None
    avg_hold_days: Optional[float] = None
    turnover_est: Optional[float] = None
    net_edge_z: Optional[float] = None
    reason: Optional[str] = None


class EngineResultModel(BaseModel):
    meta: EngineMetaModel
    rows: List[PairRowModel]


class OptConfigModel(BaseModel):
    # Mirrors openapi.yaml OptConfig (subset of CoreOptConfig)
    n_trials: int = 100
    timeout_sec: Optional[int] = None
    sampler: str = Field("tpe", pattern="^(tpe|cmaes)$")
    seed: Optional[int] = 42
    pruner: str = Field("median", pattern="^(none|median|sha)$")

    include_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None

    target: str = Field("dsr_net", pattern="^(dsr_net|psr_net|sr_net)$")
    use_ensemble: bool = False
    agg: str = Field("median", pattern="^(mean|median|trimmed_mean)$")
    trim_alpha: float = 0.1

    penalty_turnover: float = 0.0
    min_avg_hold: float = 0.0

    n_folds: int = 3
    test_frac: float = 0.2
    purge_frac: float = 0.02


class OptimizerRunRequest(BaseModel):
    timeIndex: List[datetime]
    pricesWide: Dict[str, List[float]]
    pairs: List[PairModel]
    optConfig: Optional[OptConfigModel] = None
    includeSpecs: Optional[List[str]] = None
    bridgeOverrides: Optional[Dict[str, str]] = None  # currently ignored

    @field_validator("pricesWide")
    @classmethod
    def _validate_prices(
        cls,
        v: Dict[str, List[float]],
        info: Any,
    ) -> Dict[str, List[float]]:
        n = None
        for arr in v.values():
            if n is None:
                n = len(arr)
            elif len(arr) != n:
                raise ValueError("All price series in pricesWide must have same length")

        ti = getattr(info, "data", {}).get("timeIndex") if info is not None else None
        if ti is not None and n is not None and len(ti) != n:
            raise ValueError("timeIndex length must match pricesWide series length")

        return v


class OptimizeResultModel(BaseModel):
    studyId: str
    bestParams: Dict[str, Any]
    bestScore: float
    artifacts: List[str] = Field(default_factory=list)


class AdviceItemModel(BaseModel):
    id: str
    severity: str
    category: str
    message: str
    rationale: str
    suggested_changes: Dict[str, str]


class AdviceSummaryModel(BaseModel):
    n_pairs: int
    n_good_pairs: Optional[int] = None
    frac_good_pairs: Optional[float] = None
    concentration_top5_pct: Optional[float] = None
    concentration_top10_pct: Optional[float] = None
    avg_turnover_est: Optional[float] = None
    avg_hold_days: Optional[float] = None
    avg_edge_z: Optional[float] = None
    avg_cost_spread_units: Optional[float] = None
    avg_adf_p: Optional[float] = None
    avg_residual_adf_p: Optional[float] = None
    coint_rate_pct: Optional[float] = None
    avg_halflife: Optional[float] = None


class AdviceResultModel(BaseModel):
    summary: AdviceSummaryModel
    advice: List[AdviceItemModel]


# =========================
# Helpers
# =========================

def _make_error(status: int, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> JSONResponse:
    payload = ErrorModel(code=code, message=message, details=details or {})
    return JSONResponse(status_code=status, content=payload.model_dump())


def _prices_to_df(time_index: List[datetime], prices_wide: Dict[str, List[float]]) -> pd.DataFrame:
    """Convert (timeIndex, pricesWide) into a wide price DataFrame."""
    if not prices_wide:
        raise ValueError("pricesWide must not be empty")

    arr = {}
    for sym, series in prices_wide.items():
        arr[sym] = np.asarray(series, dtype="float64")

    idx = pd.to_datetime(time_index, errors="raise", utc=True)
    df = pd.DataFrame(arr, index=idx)
    return df.sort_index()


def _apply_engine_overrides(cfg: EngineConfig, overrides: EngineConfigOverridesModel) -> EngineConfig:
    data = overrides.model_dump(exclude_none=True)
    for key, value in data.items():
        if key == "secondary_windows" and isinstance(value, list):
            value = tuple(int(x) for x in value)
        if key == "zscore_clip" and isinstance(value, list):
            value = (float(value[0]), float(value[1]))
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _row_to_pair_model(row: Dict[str, Any]) -> PairModel:
    pair_val = row.get("pair")

    # אם כבר PairModel – נחזיר אותו
    if isinstance(pair_val, PairModel):
        return pair_val

    # list/tuple באורך 2
    if isinstance(pair_val, (list, tuple)) and len(pair_val) == 2:
        return PairModel((str(pair_val[0]), str(pair_val[1])))

    # dict עם "root" או "__root__" (פורמט ישן/חדש)
    if isinstance(pair_val, dict):
        if "root" in pair_val:
            root = pair_val["root"]
        elif "__root__" in pair_val:
            root = pair_val["__root__"]
        else:
            root = None
        if isinstance(root, (list, tuple)) and len(root) == 2:
            return PairModel((str(root[0]), str(root[1])))

    raise ValueError(f"Invalid pair value in engine row: {pair_val!r}")


def _row_to_pair_row_model(row: Dict[str, Any]) -> PairRowModel:
    hl = row.get("halflife")
    if isinstance(hl, (float, int)) and math.isinf(hl):
        hl_out: float | str = "inf"
    else:
        hl_out = hl  # type: ignore[assignment]

    return PairRowModel(
        pair=_row_to_pair_model(row),
        window=int(row.get("window", 0)),
        action=row.get("action"),
        mispricing=float(row.get("mispricing", 0.0)),
        vol_adj_mispricing=row.get("vol_adj_mispricing"),
        zscore=row.get("zscore"),
        band_p95=row.get("band_p95"),
        band_upper=row.get("band_upper"),
        band_lower=row.get("band_lower"),
        halflife=hl_out,
        rolling_corr=row.get("rolling_corr"),
        distance_corr=row.get("distance_corr"),
        adf_p=row.get("adf_p"),
        residual_adf_p=row.get("residual_adf_p"),
        is_coint=row.get("is_coint"),
        y_fair=float(row.get("y_fair", 0.0)),
        target_pos_units=float(row.get("target_pos_units", 0.0)),
        cost_spread_units=float(row.get("cost_spread_units", 0.0)),
        rp_weight=row.get("rp_weight"),
        sr_net=row.get("sr_net"),
        psr_net=row.get("psr_net"),
        dsr_net=row.get("dsr_net"),
        avg_hold_days=row.get("avg_hold_days"),
        turnover_est=row.get("turnover_est"),
        net_edge_z=row.get("net_edge_z"),
        reason=row.get("reason"),
    )


# =========================
# FastAPI app
# =========================

app = FastAPI(
    title="Fair Value Pairs Trading API (local)",
    version="1.0.0",
    description="Local FastAPI wrapper over core.fair_value_engine and core.fair_value_optimizer_v2.",
)


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine": "fair_value",
        "optimizer": "optuna",
        "advisor": "online",
    }


@app.post("/engine/run", response_model=EngineResultModel)
def engine_run(req: EngineRunRequest):
    try:
        df_prices = _prices_to_df(req.timeIndex, req.pricesWide)
    except ValueError as exc:
        return _make_error(400, code="bad_request", message=str(exc))

    cfg = EngineConfig()
    if req.configOverrides is not None:
        cfg = _apply_engine_overrides(cfg, req.configOverrides)

    pairs_tuples = [p.as_tuple() for p in req.pairs]
    cfg.pairs = pairs_tuples  # type: ignore[attr-defined]

    engine = FairValueEngine(config=cfg, provider=None)

    try:
        df_rows = engine.run(prices_wide=df_prices, pairs=pairs_tuples)
    except Exception as exc:
        return _make_error(
            422,
            code="engine_error",
            message="Engine run failed",
            details={"error": str(exc)},
        )

    if not isinstance(df_rows, pd.DataFrame):
        return _make_error(
            500,
            code="internal_error",
            message="Engine returned non-DataFrame result",
            details={"type": str(type(df_rows))},
        )

    df_rows = df_rows.copy().reset_index(drop=True)

    rows: List[PairRowModel] = []
    for _, r in df_rows.iterrows():
        row_dict = r.to_dict()
        try:
            rows.append(_row_to_pair_row_model(row_dict))
        except Exception:
            continue

    if df_prices.empty:
        return _make_error(
            422,
            code="no_data",
            message="No price data after preprocessing",
        )

    asof = df_prices.index[-1].to_pydatetime()

    meta = EngineMetaModel(
        asof=asof,
        window=int(cfg.window),
        secondary_windows=list(getattr(cfg, "secondary_windows", [])),
        log_mode=bool(getattr(cfg, "log_mode", True)),
        ensemble_mode=getattr(cfg, "ensemble_mode", None),
    )

    return EngineResultModel(meta=meta, rows=rows)


@app.post("/optimizer/run", response_model=OptimizeResultModel)
def optimizer_run(req: OptimizerRunRequest):
    try:
        df_prices = _prices_to_df(req.timeIndex, req.pricesWide)
    except ValueError as exc:
        return _make_error(400, code="bad_request", message=str(exc))

    cfg_kwargs: Dict[str, Any] = {}
    if req.optConfig is not None:
        opt_dict = req.optConfig.model_dump(exclude_none=True)
        if "include_tags" in opt_dict and opt_dict["include_tags"] is not None:
            opt_dict["include_tags"] = tuple(opt_dict["include_tags"])
        if "exclude_tags" in opt_dict and opt_dict["exclude_tags"] is not None:
            opt_dict["exclude_tags"] = tuple(opt_dict["exclude_tags"])
        cfg_kwargs.update(opt_dict)

    cfg = CoreOptConfig(**cfg_kwargs)
    pairs_tuples = [p.as_tuple() for p in req.pairs]

    try:
        study, best = optimize_fair_value(
            prices_wide=df_prices,
            pairs=pairs_tuples,
            cfg=cfg,
            include_specs=req.includeSpecs,
        )
    except Exception as exc:
        return _make_error(
            422,
            code="optimizer_error",
            message="Optimization failed",
            details={"error": str(exc)},
        )

    best_score = float(getattr(study, "best_value", float("nan")))
    study_id = getattr(cfg, "study_name", "fv_opt")

    return OptimizeResultModel(
        studyId=str(study_id),
        bestParams=best,
        bestScore=best_score,
        artifacts=[],
    )


@app.post("/advisor/run", response_model=AdviceResultModel)
def advisor_run(req: EngineRunRequest):
    """
    מריץ FairValueEngine על payload כמו /engine/run,
    אבל במקום להחזיר rows גולמיים – מחזיר ניתוח ועצות לשיפור.
    """
    try:
        df_prices = _prices_to_df(req.timeIndex, req.pricesWide)
    except ValueError as exc:
        return _make_error(400, code="bad_request", message=str(exc))

    cfg = EngineConfig()
    if req.configOverrides is not None:
        cfg = _apply_engine_overrides(cfg, req.configOverrides)

    pairs_tuples = [p.as_tuple() for p in req.pairs]
    cfg.pairs = pairs_tuples  # type: ignore[attr-defined]

    engine = FairValueEngine(config=cfg, provider=None)

    try:
        df_rows = engine.run(prices_wide=df_prices, pairs=pairs_tuples)
    except Exception as exc:
        return _make_error(
            422,
            code="engine_error",
            message="Engine run failed in advisor endpoint",
            details={"error": str(exc)},
        )

    if not isinstance(df_rows, pd.DataFrame):
        return _make_error(
            500,
            code="internal_error",
            message="Engine returned non-DataFrame result in advisor endpoint",
            details={"type": str(type(df_rows))},
        )

    analysis = analyze_engine_rows(df_rows)
    summary_raw = analysis.get("summary", {})
    advice_raw = analysis.get("advice", [])

    summary = AdviceSummaryModel(**summary_raw)
    advice_items = [AdviceItemModel(**item) for item in advice_raw]

    return AdviceResultModel(summary=summary, advice=advice_items)


# ========== Stubs for async jobs & artifacts ==========

@app.post("/jobs")
def create_job() -> JSONResponse:  # type: ignore[override]
    return _make_error(
        501,
        code="not_implemented",
        message="Async jobs are not implemented in this local API. Use /optimizer/run instead.",
    )


@app.get("/jobs/{jobId}")
def get_job(jobId: str) -> JSONResponse:  # type: ignore[override]
    return _make_error(
        404,
        code="not_found",
        message="Jobs are not persisted in this local API.",
        details={"jobId": jobId},
    )


@app.post("/jobs/{jobId}/cancel")
def cancel_job(jobId: str) -> JSONResponse:  # type: ignore[override]
    return _make_error(
        404,
        code="not_found",
        message="Jobs cannot be canceled in this local API.",
        details={"jobId": jobId},
    )


@app.get("/artifacts")
def list_artifacts() -> JSONResponse:  # type: ignore[override]
    return _make_error(
        501,
        code="not_implemented",
        message="Artifact listing is not implemented yet.",
    )


@app.get("/artifacts/{artifactId}")
def get_artifact(artifactId: str) -> JSONResponse:  # type: ignore[override]
    return _make_error(
        404,
        code="not_found",
        message="Artifacts are not stored in this local API.",
        details={"artifactId": artifactId},
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("root.api_server:app", host="0.0.0.0", port=8000, reload=True)
