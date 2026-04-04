# -*- coding: utf-8 -*-
"""
agents/signal_agents.py — Signal Engine Agent Layer
====================================================

Four narrow-mandate agents that interface between the signal/regime/lifecycle
core layer and the rest of the system:

  SignalAnalystAgent      — classify spread state, propose signals, emit rationale
  RegimeSurveillanceAgent — monitor regime shifts, instability, broken conditions
  TradeLifecycleAgent     — track state transitions, stale states, blocked actions
  ExitOversightAgent      — scan open trades for exit/reduction opportunities

Each agent:
  - Accepts AgentTask, returns AgentResult (never raises)
  - Calls audit.log() for every significant decision
  - Returns structured output dicts (described per class)
  - Does NOT mutate shared state — all decisions are proposals

Task type naming convention:  "<agent_short>.<verb>"
  e.g. "signal_analyst.classify", "regime_surveillance.scan", ...

Usage:
    from agents.signal_agents import SignalAnalystAgent
    agent = SignalAnalystAgent()
    task = agent.create_task("signal_analyst.classify", {
        "pair_id": pair_id,
        "spread": spread_series,
        "prices_x": px,
        "prices_y": py,
        "as_of": datetime.utcnow(),
    })
    result = agent.execute(task)
    if result.status == AgentStatus.COMPLETED:
        decision = result.output["decision"]
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from agents.base import AgentAuditLogger, BaseAgent
from core.contracts import (
    AgentResult,
    AgentStatus,
    BlockReason,
    ExitReason,
    IntentAction,
    PairId,
    RegimeLabel,
    SignalQualityGrade,
    TradeLifecycleState,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# SIGNAL ANALYST AGENT
# ══════════════════════════════════════════════════════════════════

class SignalAnalystAgent(BaseAgent):
    """
    Analyzes current spread state for one pair and proposes a signal
    classification with full rationale artifacts.

    Task type: "signal_analyst.classify"

    Required payload:
      pair_id   : PairId or {"sym_x": ..., "sym_y": ...}
      spread    : pd.Series — spread values (not z-scored)
      prices_x  : pd.Series — price series for leg X
      prices_y  : pd.Series — price series for leg Y

    Optional payload:
      as_of              : datetime (default: last index of spread)
      regime             : str RegimeLabel value (default: auto-detected)
      signal_confidence  : float [0,1] from upstream conviction model
      half_life_days     : float (default: estimated from spread)
      lookback_vol       : int — window for current vol (default 20)
      baseline_vol       : int — window for baseline vol (default 252)

    Output:
      decision           : dict — SignalDecision serialisation
        action           : str (WATCH / ENTER / HOLD / REDUCE / EXIT / ...)
        quality_grade    : str (A+ / A / B / C / D / F)
        conviction       : float [0,1]
        regime           : str RegimeLabel value
        block_reasons    : list[str]
        rationale        : str
      z_score            : float — current spread z-score
      spread_vol         : float — current spread volatility (std)
      half_life_days     : float
      thresholds         : dict — ThresholdSet used
      regime_features    : dict — key regime feature values
      warnings           : list[str]
    """

    NAME = "signal_analyst"
    ALLOWED_TASK_TYPES = {"signal_analyst.classify"}
    REQUIRED_PAYLOAD_KEYS = {"pair_id", "spread"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:  # type: ignore[name-defined]
        from core.regime_engine import RegimeEngine, build_regime_features
        from core.signal_quality import SignalQualityEngine
        from core.threshold_engine import ThresholdEngine

        # ── Unpack payload ────────────────────────────────────────
        raw_pair = task.payload["pair_id"]
        pair_id = _coerce_pair_id(raw_pair)
        if pair_id is None:
            audit.log(f"Skipping: cannot parse pair_id from {raw_pair!r}")
            return {"skipped": True, "reason": f"invalid_pair_id: {raw_pair!r}"}
        spread: pd.Series = task.payload.get("spread")
        if spread is None or (hasattr(spread, '__len__') and len(spread) == 0):
            audit.log("Skipping: no spread data provided")
            return {"skipped": True, "reason": "no_spread_data"}
        prices_x: Optional[pd.Series] = task.payload.get("prices_x")
        prices_y: Optional[pd.Series] = task.payload.get("prices_y")
        as_of: datetime = task.payload.get("as_of") or _last_ts(spread)
        signal_confidence: float = float(task.payload.get("signal_confidence", 0.5))
        half_life_days: float = float(task.payload.get("half_life_days", math.nan))
        lookback_vol: int = int(task.payload.get("lookback_vol", 20))
        baseline_vol_window: int = int(task.payload.get("baseline_vol", 252))

        audit.log(f"pair={pair_id.label} as_of={as_of.date()} len(spread)={len(spread)}")

        warnings: list[str] = []

        # ── Clip spread to as_of ──────────────────────────────────
        spread_clip = spread[spread.index <= as_of] if hasattr(spread.index, "tz") or isinstance(spread.index, pd.DatetimeIndex) else spread
        if len(spread_clip) < 30:
            warnings.append(f"Only {len(spread_clip)} observations — estimates unreliable")
            audit.warn(f"Thin spread history: {len(spread_clip)} obs")

        # ── Compute z-score (canonical) ───────────────────────────
        from common.feature_engineering import compute_zscore_scalar
        spread_mean = float(spread_clip.mean())
        spread_std  = float(spread_clip.std())
        z_score = compute_zscore_scalar(spread_clip, lookback=None)
        audit.log(f"z={z_score:.3f} mean={spread_mean:.4f} std={spread_std:.4f}")

        # ── Volatility ratio ─────────────────────────────────────
        current_vol  = float(spread_clip.iloc[-lookback_vol:].std()) if len(spread_clip) >= lookback_vol else spread_std
        baseline_vol = float(spread_clip.iloc[-baseline_vol_window:].std()) if len(spread_clip) >= baseline_vol_window else spread_std
        audit.log(f"vol current={current_vol:.5f} baseline={baseline_vol:.5f}")

        # ── Half-life estimate ────────────────────────────────────
        if math.isnan(half_life_days) and len(spread_clip) >= 30:
            half_life_days = _estimate_half_life(spread_clip)
            audit.log(f"half_life_days estimated={half_life_days:.1f}")

        # ── Regime ───────────────────────────────────────────────
        raw_regime = task.payload.get("regime")
        if raw_regime is not None:
            try:
                regime = RegimeLabel(raw_regime)
                audit.log(f"regime supplied externally: {regime.value}")
            except ValueError:
                audit.warn(f"Unknown regime string '{raw_regime}' — auto-detecting")
                regime = None
        else:
            regime = None

        regime_features_dict: dict[str, Any] = {}
        if regime is None:
            try:
                px = prices_x if prices_x is not None else pd.Series(dtype=float)
                py = prices_y if prices_y is not None else pd.Series(dtype=float)
                feat = build_regime_features(spread_clip, px, py, as_of=as_of)
                regime_engine = RegimeEngine()
                regime_result = regime_engine.classify(feat)
                regime = regime_result.regime
                regime_features_dict = {k: v for k, v in feat.__dict__.items()
                                        if not isinstance(v, pd.Series)}
                audit.log(f"regime auto-detected: {regime.value} conf={regime_result.confidence:.2f}")
            except Exception as exc:
                regime = RegimeLabel.UNKNOWN
                audit.warn(f"Regime detection failed: {exc} — defaulting to UNKNOWN")

        # ── Thresholds ────────────────────────────────────────────
        t_engine = ThresholdEngine.default()
        thresholds = t_engine.compute(
            regime=regime,
            current_spread_vol=current_vol,
            baseline_spread_vol=baseline_vol,
            signal_confidence=signal_confidence,
            half_life_days=half_life_days,
        )
        audit.log(
            f"thresholds entry={thresholds.entry_z:.2f} exit={thresholds.exit_z:.2f} "
            f"stop={thresholds.stop_z:.2f} mode={thresholds.mode.value}"
        )

        # ── Signal quality ────────────────────────────────────────
        quality_engine = SignalQualityEngine()
        # Build a minimal mr_score proxy from half-life and regime
        mr_score = _proxy_mr_score(half_life_days, regime)
        quality = quality_engine.assess(
            conviction=signal_confidence,
            mr_score=mr_score,
            regime=regime,
        )
        audit.log(
            f"quality grade={quality.grade.value} score={quality.score:.3f} "
            f"skip={quality.skip_recommended}"
        )
        if quality.warnings:
            warnings.extend(quality.warnings)

        # ── Action proposal ───────────────────────────────────────
        block_reasons: list[str] = []
        action = _propose_action(
            z_score=z_score,
            thresholds=thresholds,
            quality=quality,
            regime=regime,
            t_engine=t_engine,
            block_reasons=block_reasons,
            audit=audit,
        )

        # ── Build rationale string ────────────────────────────────
        rationale = _build_rationale(
            action=action,
            z_score=z_score,
            regime=regime,
            quality=quality,
            thresholds=thresholds,
            block_reasons=block_reasons,
            half_life_days=half_life_days,
        )
        audit.log(f"action={action.value} block_reasons={block_reasons}")

        # ── Assemble output ───────────────────────────────────────
        decision = {
            "pair_id": pair_id.label,
            "action": action.value,
            "quality_grade": quality.grade.value,
            "conviction": signal_confidence,
            "regime": regime.value,
            "block_reasons": block_reasons,
            "rationale": rationale,
            "generated_at": as_of.isoformat(),
        }

        return {
            "decision": decision,
            "z_score": round(z_score, 4),
            "spread_vol": round(current_vol, 6),
            "half_life_days": round(half_life_days, 1) if not math.isnan(half_life_days) else None,
            "thresholds": thresholds.to_dict(),
            "regime_features": regime_features_dict,
            "warnings": warnings,
        }


# ══════════════════════════════════════════════════════════════════
# REGIME SURVEILLANCE AGENT
# ══════════════════════════════════════════════════════════════════

class RegimeSurveillanceAgent(BaseAgent):
    """
    Monitors regime shifts, instability, and broken conditions across
    a portfolio of pairs.  Designed for periodic (e.g. daily) runs.

    Task type: "regime_surveillance.scan"

    Required payload:
      spreads   : dict[str, pd.Series] — pair_label → spread series
      prices    : pd.DataFrame — all instrument prices (optional; improves detection)

    Optional payload:
      as_of          : datetime (default: now)
      prior_regimes  : dict[str, str] — pair_label → previous RegimeLabel value

    Output:
      regime_map          : dict[str, str]  pair_label → current regime value
      shift_alerts        : list[dict]      pairs where regime changed
      instability_alerts  : list[dict]      pairs showing instability warnings
      broken_alerts       : list[dict]      pairs classified BROKEN
      crisis_alert        : bool            True if any pair is CRISIS
      summary             : dict            counts by regime
    """

    NAME = "regime_surveillance"
    ALLOWED_TASK_TYPES = {"regime_surveillance.scan"}
    REQUIRED_PAYLOAD_KEYS = {"spreads"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:  # type: ignore[name-defined]
        from core.regime_engine import RegimeEngine, build_regime_features

        spreads: dict[str, pd.Series] = task.payload["spreads"]
        prices: pd.DataFrame = task.payload.get("prices", pd.DataFrame())
        as_of: datetime = task.payload.get("as_of") or datetime.utcnow()
        prior_regimes: dict[str, str] = task.payload.get("prior_regimes") or {}

        audit.log(f"Scanning {len(spreads)} pairs as_of={as_of.date()}")

        regime_engine = RegimeEngine()
        regime_map: dict[str, str] = {}
        shift_alerts: list[dict] = []
        instability_alerts: list[dict] = []
        broken_alerts: list[dict] = []
        crisis_alert = False

        for label, spread in spreads.items():
            try:
                spread_clip = spread[spread.index <= as_of] if isinstance(spread.index, pd.DatetimeIndex) else spread
                if len(spread_clip) < 30:
                    regime_map[label] = RegimeLabel.UNKNOWN.value
                    continue

                # Extract per-leg prices if available
                parts = label.split("/")
                px = prices[parts[0]] if len(parts) == 2 and parts[0] in prices.columns else pd.Series(dtype=float)
                py = prices[parts[1]] if len(parts) == 2 and parts[1] in prices.columns else pd.Series(dtype=float)

                feat = build_regime_features(spread_clip, px, py, as_of=as_of)
                result = regime_engine.classify(feat)
                regime_map[label] = result.regime.value

                # Shift detection
                prior = prior_regimes.get(label)
                if prior and prior != result.regime.value:
                    shift_alerts.append({
                        "pair": label,
                        "from_regime": prior,
                        "to_regime": result.regime.value,
                        "confidence": round(result.confidence, 3),
                        "as_of": as_of.isoformat(),
                    })
                    audit.log(f"REGIME SHIFT {label}: {prior} → {result.regime.value}")

                # Instability warnings
                if result.warnings:
                    instability_alerts.append({
                        "pair": label,
                        "regime": result.regime.value,
                        "warnings": result.warnings,
                    })

                # Broken / crisis
                if result.regime == RegimeLabel.BROKEN:
                    broken_alerts.append({
                        "pair": label,
                        "confidence": round(result.confidence, 3),
                        "as_of": as_of.isoformat(),
                    })
                    audit.warn(f"BROKEN: {label}")

                if result.regime == RegimeLabel.CRISIS:
                    crisis_alert = True
                    audit.warn(f"CRISIS detected: {label}")

            except Exception as exc:
                regime_map[label] = RegimeLabel.UNKNOWN.value
                audit.warn(f"{label}: regime scan failed — {exc}")

        # Summary counts
        from collections import Counter
        summary = dict(Counter(regime_map.values()))

        audit.log(
            f"Done: {len(shift_alerts)} shifts, {len(broken_alerts)} broken, "
            f"crisis={crisis_alert}, summary={summary}"
        )

        return {
            "regime_map": regime_map,
            "shift_alerts": shift_alerts,
            "instability_alerts": instability_alerts,
            "broken_alerts": broken_alerts,
            "crisis_alert": crisis_alert,
            "summary": summary,
        }


# ══════════════════════════════════════════════════════════════════
# TRADE LIFECYCLE AGENT
# ══════════════════════════════════════════════════════════════════

class TradeLifecycleAgent(BaseAgent):
    """
    Inspects trade lifecycle states across open positions.  Identifies
    stale states, blocked transitions, required actions, and timeout risks.

    Task type: "lifecycle.inspect"

    Required payload:
      states : dict[str, str]  pair_label → current TradeLifecycleState value
                               (or a dict of state machine snapshots)

    Optional payload:
      as_of              : datetime
      entry_timestamps   : dict[str, datetime]  when each pair entered current state
      max_setup_days     : int   (default 5)
      max_entry_ready_days : int (default 3)
      max_active_days    : int   (default 30)

    Output:
      stale_alerts       : list[dict]  pairs in a state too long
      blocked_alerts     : list[dict]  pairs that cannot advance (and why)
      action_required    : list[dict]  pairs needing manual intervention
      timeout_risk       : list[dict]  pairs approaching timeout
      summary            : dict        counts per state
    """

    NAME = "trade_lifecycle"
    ALLOWED_TASK_TYPES = {"lifecycle.inspect"}
    REQUIRED_PAYLOAD_KEYS = {"states"}

    # Default timeout thresholds (days) per state
    _DEFAULT_MAX_DAYS: dict[str, int] = {
        TradeLifecycleState.SETUP_FORMING.value: 5,
        TradeLifecycleState.ENTRY_READY.value: 3,
        TradeLifecycleState.PENDING_ENTRY.value: 2,
        TradeLifecycleState.ACTIVE.value: 30,
        TradeLifecycleState.EXIT_READY.value: 2,
        TradeLifecycleState.PENDING_EXIT.value: 2,
        TradeLifecycleState.COOLDOWN.value: 14,
        TradeLifecycleState.SUSPENDED.value: 60,
    }

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:  # type: ignore[name-defined]
        states: dict[str, str] = task.payload["states"]
        as_of: datetime = task.payload.get("as_of") or datetime.utcnow()
        entry_ts: dict[str, datetime] = task.payload.get("entry_timestamps") or {}

        # Allow payload overrides for thresholds
        max_days: dict[str, int] = dict(self._DEFAULT_MAX_DAYS)
        if "max_setup_days" in task.payload:
            max_days[TradeLifecycleState.SETUP_FORMING.value] = int(task.payload["max_setup_days"])
        if "max_entry_ready_days" in task.payload:
            max_days[TradeLifecycleState.ENTRY_READY.value] = int(task.payload["max_entry_ready_days"])
        if "max_active_days" in task.payload:
            max_days[TradeLifecycleState.ACTIVE.value] = int(task.payload["max_active_days"])

        audit.log(f"Inspecting {len(states)} pairs as_of={as_of.date()}")

        stale_alerts: list[dict] = []
        timeout_risk: list[dict] = []
        blocked_alerts: list[dict] = []
        action_required: list[dict] = []

        for label, state_val in states.items():
            # Age in state
            entered = entry_ts.get(label)
            age_days: Optional[float] = None
            if entered is not None:
                age_days = (as_of - entered).total_seconds() / 86_400

            limit_days = max_days.get(state_val)

            if age_days is not None and limit_days is not None:
                if age_days >= limit_days:
                    stale_alerts.append({
                        "pair": label,
                        "state": state_val,
                        "age_days": round(age_days, 1),
                        "limit_days": limit_days,
                        "severity": "HIGH" if age_days >= 2 * limit_days else "MEDIUM",
                    })
                    audit.warn(f"STALE {label} in {state_val} for {age_days:.1f}d (limit {limit_days}d)")
                elif age_days >= 0.8 * limit_days:
                    timeout_risk.append({
                        "pair": label,
                        "state": state_val,
                        "age_days": round(age_days, 1),
                        "limit_days": limit_days,
                        "pct_used": round(age_days / limit_days, 2),
                    })

            # States that imply action required
            if state_val == TradeLifecycleState.SUSPENDED.value:
                action_required.append({
                    "pair": label,
                    "state": state_val,
                    "action": "REVIEW_SUSPENSION",
                    "note": "Pair is suspended; requires manual review before resuming",
                })
            elif state_val == TradeLifecycleState.EXIT_READY.value:
                action_required.append({
                    "pair": label,
                    "state": state_val,
                    "action": "INITIATE_EXIT",
                    "note": "Exit signal confirmed; waiting for execution",
                })

            # Blocked detection: ACTIVE pairs near stop without EXIT_READY
            if state_val == TradeLifecycleState.WATCHLIST.value:
                blocked_alerts.append({
                    "pair": label,
                    "state": state_val,
                    "reason": "Stuck on watchlist — may need re-classification or retirement",
                })

        from collections import Counter
        summary = dict(Counter(states.values()))
        audit.log(
            f"stale={len(stale_alerts)} timeout_risk={len(timeout_risk)} "
            f"action_required={len(action_required)} summary={summary}"
        )

        return {
            "stale_alerts": stale_alerts,
            "blocked_alerts": blocked_alerts,
            "action_required": action_required,
            "timeout_risk": timeout_risk,
            "summary": summary,
        }


# ══════════════════════════════════════════════════════════════════
# EXIT OVERSIGHT AGENT
# ══════════════════════════════════════════════════════════════════

class ExitOversightAgent(BaseAgent):
    """
    Monitors all open (ACTIVE / SCALING_IN / REDUCING) trades for
    exit and reduction opportunities.  Flags risk/regime conflicts.

    Task type: "exit_oversight.scan"

    Required payload:
      open_positions : list[dict]  one dict per open position, each with:
        pair_id      : str  "SYM_X/SYM_Y"
        entry_z      : float  z-score at entry
        current_z    : float  current z-score
        regime       : str   current RegimeLabel value
        entry_time   : str ISO datetime
        state        : str  TradeLifecycleState value
        direction    : str  "LONG_SPREAD" or "SHORT_SPREAD"
        quality_grade: str  SignalQualityGrade value at entry

    Optional payload:
      as_of             : datetime
      max_holding_days  : int  (default 30)
      profit_target_z   : float  (default 0.3)  z at which to signal target hit
      stop_z            : float  (default 4.5)  z at which to hard-stop

    Output:
      exit_signals   : list[dict]  positions recommended for full exit
      reduce_signals : list[dict]  positions recommended for partial reduction
      risk_alerts    : list[dict]  positions with regime/risk conflicts
      clean_holds    : list[dict]  positions with no flags
      summary        : dict
    """

    NAME = "exit_oversight"
    ALLOWED_TASK_TYPES = {"exit_oversight.scan"}
    REQUIRED_PAYLOAD_KEYS = {"open_positions"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:  # type: ignore[name-defined]
        positions: list[dict] = task.payload["open_positions"]
        as_of: datetime = task.payload.get("as_of") or datetime.utcnow()
        max_holding_days: int = int(task.payload.get("max_holding_days", 30))
        profit_target_z: float = float(task.payload.get("profit_target_z", 0.3))
        stop_z_override: float = float(task.payload.get("stop_z", 4.5))

        audit.log(
            f"Scanning {len(positions)} open positions as_of={as_of.date()} "
            f"max_hold={max_holding_days}d target_z={profit_target_z}"
        )

        exit_signals: list[dict] = []
        reduce_signals: list[dict] = []
        risk_alerts: list[dict] = []
        clean_holds: list[dict] = []

        _RISK_REGIMES = {RegimeLabel.CRISIS.value, RegimeLabel.BROKEN.value, RegimeLabel.TRENDING.value}
        _CAUTION_REGIMES = {RegimeLabel.HIGH_VOL.value}

        for pos in positions:
            label    = pos.get("pair_id", "?")
            entry_z  = float(pos.get("entry_z", 2.0))
            curr_z   = float(pos.get("current_z", 0.0))
            regime   = pos.get("regime", RegimeLabel.UNKNOWN.value)
            entry_ts_str = pos.get("entry_time")
            direction = pos.get("direction", "LONG_SPREAD")
            grade_str = pos.get("quality_grade", SignalQualityGrade.B.value)
            state = pos.get("state", TradeLifecycleState.ACTIVE.value)

            # Age in position
            age_days: float = 0.0
            if entry_ts_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_ts_str)
                    age_days = (as_of - entry_dt).total_seconds() / 86_400
                except Exception:
                    pass

            # Z-score sign convention: positive z → spread is above mean
            # LONG_SPREAD: profit when spread rises from negative → 0
            # SHORT_SPREAD: profit when spread falls from positive → 0
            abs_z = abs(curr_z)
            converging = (
                (direction == "SHORT_SPREAD" and curr_z < entry_z) or
                (direction == "LONG_SPREAD" and curr_z > entry_z) or
                abs_z <= abs(entry_z) * 0.5  # halfway to mean
            )

            exit_reasons: list[str] = []
            reduce_reasons: list[str] = []
            risk_reason: Optional[str] = None
            is_exit = False
            is_reduce = False

            # ── Hard exits ─────────────────────────────────────
            if abs_z >= stop_z_override:
                exit_reasons.append(ExitReason.ADVERSE_EXCURSION_STOP.value)
                is_exit = True

            if regime in _RISK_REGIMES:
                exit_reasons.append(ExitReason.REGIME_FLIP.value)
                risk_reason = f"Regime={regime} — mandatory exit"
                is_exit = True

            if age_days >= max_holding_days:
                exit_reasons.append(ExitReason.TIME_STOP.value)
                is_exit = True

            # ── Target / mean reversion complete ──────────────
            if abs_z <= profit_target_z:
                exit_reasons.append(ExitReason.MEAN_REVERSION_COMPLETE.value)
                is_exit = True

            # ── Soft reductions ────────────────────────────────
            if not is_exit:
                if regime in _CAUTION_REGIMES:
                    reduce_reasons.append(ExitReason.REGIME_WEAKENING.value)
                    is_reduce = True

                if grade_str == SignalQualityGrade.D.value:
                    reduce_reasons.append(ExitReason.CONFIDENCE_COLLAPSE.value)
                    is_reduce = True

                if age_days >= max_holding_days * 0.7:
                    reduce_reasons.append(ExitReason.STALE_TRADE.value)
                    is_reduce = True

                if converging and abs_z <= abs(entry_z) * 0.5 and abs_z > profit_target_z:
                    reduce_reasons.append(ExitReason.STAGED_PROFIT_TAKING.value)
                    is_reduce = True

            # ── Route to bucket ───────────────────────────────
            signal_base = {
                "pair": label,
                "current_z": round(curr_z, 4),
                "entry_z": round(entry_z, 4),
                "age_days": round(age_days, 1),
                "regime": regime,
                "state": state,
            }

            if is_exit:
                exit_signals.append({**signal_base, "exit_reasons": exit_reasons})
                if risk_reason:
                    risk_alerts.append({**signal_base, "risk_reason": risk_reason})
                audit.log(f"EXIT {label}: {exit_reasons}")
            elif is_reduce:
                reduce_signals.append({**signal_base, "reduce_reasons": reduce_reasons})
                audit.log(f"REDUCE {label}: {reduce_reasons}")
            else:
                clean_holds.append(signal_base)

        summary = {
            "n_exit": len(exit_signals),
            "n_reduce": len(reduce_signals),
            "n_risk_alerts": len(risk_alerts),
            "n_clean_holds": len(clean_holds),
            "total": len(positions),
        }
        audit.log(f"Done: {summary}")

        return {
            "exit_signals": exit_signals,
            "reduce_signals": reduce_signals,
            "risk_alerts": risk_alerts,
            "clean_holds": clean_holds,
            "summary": summary,
        }


# ══════════════════════════════════════════════════════════════════
# HELPERS (module-private)
# ══════════════════════════════════════════════════════════════════

def _coerce_pair_id(raw: Any) -> Optional[PairId]:
    """Coerce various formats to PairId, or return None if not possible."""
    if isinstance(raw, PairId):
        return raw
    if isinstance(raw, dict):
        sx = raw.get("sym_x") or raw.get("symbol_x")
        sy = raw.get("sym_y") or raw.get("symbol_y")
        if sx and sy:
            return PairId(sx, sy)
        return None
    if isinstance(raw, str) and "/" in raw:
        parts = raw.split("/")
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return PairId(parts[0].strip(), parts[1].strip())
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return PairId(str(raw[0]), str(raw[1]))
    return None


def _last_ts(series: pd.Series) -> datetime:
    if isinstance(series.index, pd.DatetimeIndex) and len(series) > 0:
        ts = series.index[-1]
        return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.utcnow()
    return datetime.utcnow()


def _estimate_half_life(spread: pd.Series) -> float:
    """Estimate Ornstein-Uhlenbeck half-life from AR(1) coefficient."""
    try:
        s = spread.dropna().values.astype(float)
        if len(s) < 10:
            return math.nan
        lagged = s[:-1]
        delta  = s[1:] - lagged
        # OLS: delta = kappa*(mu - s[t-1]) → equivalent: delta ~ -rho * s[t-1]
        rho = float(np.cov(delta, lagged)[0, 1] / np.var(lagged))
        if rho >= 0 or math.isnan(rho):
            return math.nan
        hl = -math.log(2) / rho
        return max(1.0, min(500.0, hl))
    except Exception:
        return math.nan


def _proxy_mr_score(half_life_days: float, regime: RegimeLabel) -> float:
    """
    Heuristic MR quality proxy from half-life and regime.
    Used when the full signals engine isn't available in the agent context.
    """
    base = 0.50
    if not math.isnan(half_life_days):
        # Ideal half-life: 5-30 days
        if 5 <= half_life_days <= 30:
            base = 0.80
        elif 3 <= half_life_days < 5 or 30 < half_life_days <= 60:
            base = 0.60
        elif half_life_days > 60:
            base = 0.35
        else:
            base = 0.30  # < 3 days — noisy

    regime_modifiers = {
        RegimeLabel.MEAN_REVERTING: +0.10,
        RegimeLabel.HIGH_VOL:       -0.15,
        RegimeLabel.TRENDING:       -0.40,
        RegimeLabel.CRISIS:         -0.50,
        RegimeLabel.BROKEN:         -1.00,
    }
    base += regime_modifiers.get(regime, 0.0)
    return max(0.0, min(1.0, base))


def _propose_action(
    *,
    z_score: float,
    thresholds,
    quality,
    regime: RegimeLabel,
    t_engine,
    block_reasons: list[str],
    audit: AgentAuditLogger,
) -> IntentAction:
    """
    Propose an IntentAction based on z-score, thresholds, quality, and regime.
    """
    # Grade F or skip recommended → hard veto
    if quality.skip_recommended or quality.grade == SignalQualityGrade.F:
        block_reasons.append(BlockReason.LOW_QUALITY_SIGNAL.value)
        audit.log("Action=WATCH: quality skip_recommended or grade F")
        return IntentAction.WATCH

    # Entry blocked by regime
    if t_engine.is_entry_blocked_by_regime(regime):
        block_reasons.append(BlockReason.REGIME_INVALID.value)
        audit.log(f"Action=WATCH: entry blocked by regime={regime.value}")
        return IntentAction.WATCH

    abs_z = abs(z_score)

    # Stop zone
    if thresholds.would_stop(z_score):
        block_reasons.append(BlockReason.DIVERGENCE_TOO_LARGE.value)
        audit.log(f"Action=SUSPEND: |z|={abs_z:.2f} ≥ stop_z={thresholds.stop_z:.2f}")
        return IntentAction.SUSPEND

    # Entry zone
    if thresholds.would_enter(z_score):
        if quality.grade in (SignalQualityGrade.D,):
            block_reasons.append(BlockReason.LOW_QUALITY_SIGNAL.value)
            return IntentAction.WATCH
        audit.log(f"Action=ENTER: |z|={abs_z:.2f} ≥ entry_z={thresholds.entry_z:.2f}")
        return IntentAction.ENTER

    # Exit zone (position assumed active)
    if thresholds.would_exit(z_score):
        audit.log(f"Action=EXIT: |z|={abs_z:.2f} ≤ exit_z={thresholds.exit_z:.2f}")
        return IntentAction.EXIT

    # No-trade band
    if thresholds.in_no_trade_band(z_score):
        block_reasons.append(BlockReason.DIVERGENCE_TOO_SMALL.value)
        return IntentAction.WATCH

    # Mid-zone: watch/hold
    audit.log(f"Action=HOLD: |z|={abs_z:.2f} in [{thresholds.exit_z:.2f}, {thresholds.entry_z:.2f}]")
    return IntentAction.HOLD


def _build_rationale(
    *,
    action: IntentAction,
    z_score: float,
    regime: RegimeLabel,
    quality,
    thresholds,
    block_reasons: list[str],
    half_life_days: float,
) -> str:
    parts = [
        f"Action={action.value}",
        f"z={z_score:.3f}",
        f"regime={regime.value}",
        f"grade={quality.grade.value}",
        f"entry_z={thresholds.entry_z:.2f}",
        f"exit_z={thresholds.exit_z:.2f}",
        f"stop_z={thresholds.stop_z:.2f}",
    ]
    if not math.isnan(half_life_days):
        parts.append(f"hl={half_life_days:.1f}d")
    if block_reasons:
        parts.append(f"blocked=[{','.join(block_reasons)}]")
    if quality.reasons:
        parts.append(f"quality_notes=[{','.join(quality.reasons[:3])}]")
    return " | ".join(parts)


__all__ = [
    "SignalAnalystAgent",
    "RegimeSurveillanceAgent",
    "TradeLifecycleAgent",
    "ExitOversightAgent",
]
