# -*- coding: utf-8 -*-
"""
db/writer.py — Domain Writer Classes
======================================

Typed, upsert-safe persistence writers. Each writer owns exactly one
logical table with explicit conflict keys (no silent duplicates).

All writers delegate to SqlStore._upsert() for INSERT OR REPLACE semantics.

Usage:
    from db import SignalWriter, PositionWriter

    sw = SignalWriter(store)
    sw.write(signals_df, run_id="abc123", env="paper")
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseWriter:
    """
    Abstract base for all domain writers.
    Requires a SqlStore instance (or any object that exposes _upsert()).
    """

    TABLE: str = ""                          # Subclasses must override
    CONFLICT_COLS: List[str] = []            # Primary key for upsert

    def __init__(self, store: Any) -> None:
        self._store = store

    def _upsert(self, df: pd.DataFrame, schema: Optional[str] = None) -> int:
        """Delegate to store._upsert(); returns rows written or -1 on error."""
        if df.empty:
            return 0
        try:
            self._store._upsert(df, self.TABLE, self.CONFLICT_COLS, schema=schema)
            return len(df)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] _upsert failed: %s", self.__class__.__name__, exc)
            return -1

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Signal Writer
# ---------------------------------------------------------------------------

class SignalWriter(BaseWriter):
    """
    Persists universe signals. De-duplicates on (pair_id, ts_utc, run_id).

    Schema: signals_v2
        pair_id    TEXT     — e.g. "XLF_XLK"
        ts_utc     TEXT     — ISO-8601 UTC timestamp
        run_id     TEXT     — pipeline run UUID
        z_score    REAL
        grade      TEXT     — A+/A/B/C/D/F
        composite  REAL     — composite quality score [0,1]
        regime     TEXT
        env        TEXT     — paper/live/research
        written_at TEXT
    """

    TABLE = "signals_v2"
    CONFLICT_COLS = ["pair_id", "ts_utc", "run_id"]

    def write(
        self,
        signals: pd.DataFrame,
        run_id: str,
        env: str = "paper",
        schema: Optional[str] = None,
    ) -> int:
        """
        Persist signals DataFrame. Adds run_id, env, written_at columns if missing.
        Returns number of rows written.
        """
        df = signals.copy()
        df["run_id"] = run_id
        df["env"] = env
        df["written_at"] = self._now_utc()
        if "ts_utc" not in df.columns:
            df["ts_utc"] = self._now_utc()
        return self._upsert(df, schema=schema or env)


# ---------------------------------------------------------------------------
# Position Writer
# ---------------------------------------------------------------------------

class PositionWriter(BaseWriter):
    """
    Persists open positions. De-duplicates on (pair_id, ts_utc, env).

    Schema: positions_v2
        pair_id        TEXT
        ts_utc         TEXT
        env            TEXT
        direction      INTEGER   — +1 / -1
        gross_notional REAL
        z_score        REAL
        hedge_ratio    REAL
        grade          TEXT
        run_id         TEXT
        written_at     TEXT
    """

    TABLE = "positions_v2"
    CONFLICT_COLS = ["pair_id", "ts_utc", "env"]

    def write(
        self,
        positions: pd.DataFrame,
        env: str = "paper",
        run_id: str = "",
        schema: Optional[str] = None,
    ) -> int:
        df = positions.copy()
        df["env"] = env
        df["run_id"] = run_id
        df["written_at"] = self._now_utc()
        if "ts_utc" not in df.columns:
            df["ts_utc"] = self._now_utc()
        return self._upsert(df, schema=schema or env)


# ---------------------------------------------------------------------------
# Risk State Writer
# ---------------------------------------------------------------------------

class RiskStateWriter(BaseWriter):
    """
    Persists risk state snapshots. De-duplicates on (run_id, ts_utc).

    Schema: risk_state_v2
        run_id           TEXT
        ts_utc           TEXT
        equity           REAL
        gross_leverage   REAL
        net_leverage     REAL
        realized_vol     REAL
        daily_pnl_pct    REAL
        vix              REAL
        heat_level       TEXT
        anomaly_flag     INTEGER
        written_at       TEXT
    """

    TABLE = "risk_state_v2"
    CONFLICT_COLS = ["run_id", "ts_utc"]

    def write(
        self,
        risk_state: Any,
        run_id: str,
        schema: Optional[str] = None,
    ) -> int:
        """
        Accept either a RiskState dataclass or a dict.
        Returns rows written.
        """
        from dataclasses import asdict, is_dataclass
        if is_dataclass(risk_state):
            data = asdict(risk_state)
        elif isinstance(risk_state, dict):
            data = dict(risk_state)
        else:
            logger.error("RiskStateWriter: unsupported type %s", type(risk_state))
            return -1

        data["run_id"] = run_id
        data["ts_utc"] = data.get("ts_utc") or self._now_utc()
        data["written_at"] = self._now_utc()
        df = pd.DataFrame([data])
        return self._upsert(df, schema=schema)


# ---------------------------------------------------------------------------
# Run Manifest Writer
# ---------------------------------------------------------------------------

class RunManifestWriter(BaseWriter):
    """
    Persists pipeline run manifests. De-duplicates on (run_id,).

    Schema: pipeline_run_manifests_v2
        run_id          TEXT
        started_at      TEXT
        finished_at     TEXT
        config_hash     TEXT
        status          TEXT     — success/partial/failed
        n_pairs         INTEGER
        n_signals       INTEGER
        stages_json     TEXT     — JSON list of stage results
        written_at      TEXT
    """

    TABLE = "pipeline_run_manifests_v2"
    CONFLICT_COLS = ["run_id"]

    def write(
        self,
        manifest: Dict[str, Any],
        schema: Optional[str] = None,
    ) -> int:
        """
        Accept manifest dict. Adds written_at if missing.
        Returns rows written.
        """
        import json
        data = dict(manifest)
        data["written_at"] = self._now_utc()
        # Serialize any non-scalar fields
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                data[k] = json.dumps(v)
        df = pd.DataFrame([data])
        return self._upsert(df, schema=schema)


# ---------------------------------------------------------------------------
# ML Artifact Writer
# ---------------------------------------------------------------------------

class MLArtifactWriter(BaseWriter):
    """
    Persists ML training run artifacts. De-duplicates on (model_id, run_id).

    Schema: ml_artifacts_v2
        model_id         TEXT
        run_id           TEXT
        task             TEXT     — e.g. "meta_label"
        status           TEXT     — candidate/champion/retired
        ic_mean          REAL
        ic_tstat         REAL
        ic_n_samples     INTEGER
        robustness_score REAL
        trained_until    TEXT
        artifact_path    TEXT
        written_at       TEXT
    """

    TABLE = "ml_artifacts_v2"
    CONFLICT_COLS = ["model_id", "run_id"]

    def write(
        self,
        artifact: Any,
        run_id: str,
        schema: Optional[str] = None,
    ) -> int:
        """
        Accept a TrainingRunArtifact dataclass or dict.
        Returns rows written.
        """
        from dataclasses import asdict, is_dataclass
        if is_dataclass(artifact):
            data = asdict(artifact)
        elif isinstance(artifact, dict):
            data = dict(artifact)
        else:
            logger.error("MLArtifactWriter: unsupported type %s", type(artifact))
            return -1

        data["run_id"] = run_id
        data["written_at"] = self._now_utc()
        # Flatten nested structures
        for k, v in list(data.items()):
            if isinstance(v, (list, dict)):
                import json
                data[k] = json.dumps(v)
        df = pd.DataFrame([data])
        return self._upsert(df, schema=schema)
