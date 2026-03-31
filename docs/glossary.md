# Glossary — Pairs Trading System

Canonical terminology used across the platform. When in doubt, these definitions are authoritative.

---

## Core Domain

| Term | Definition | Canonical Type | Module |
|------|-----------|----------------|--------|
| **Pair ID** | Lexicographically ordered pair of symbols (e.g., AAPL-MSFT). Constructor auto-orders. | `PairId` | `core/contracts.py` |
| **Spread Definition** | Parameters for computing a spread: hedge ratio, method, lookback, intercept | `SpreadDefinition` | `core/contracts.py` |
| **Spread** | Raw difference: `price_Y - beta * price_X - alpha`. NOT z-scored. | float series | `research/spread_constructor.py` |
| **Z-Score** | Spread normalized by rolling mean and std. The primary signal input. | float | `common/signal_generator.py` |
| **Half-Life** | Mean time for spread to revert halfway to its mean (in trading days). | float | `research/pair_validator.py` |
| **Validation Result** | Outcome of statistical pair validation: PASS, FAIL, WARN, SKIP | `ValidationResult` | `core/contracts.py` |
| **Rejection Reason** | Explicit reason a pair failed validation (never fail silently) | `RejectionReason` | `research/discovery_contracts.py` |

## Signal Layer

| Term | Definition | Canonical Type | Module |
|------|-----------|----------------|--------|
| **Regime Label** | Market regime: MEAN_REVERTING, TRENDING, HIGH_VOL, CRISIS, BROKEN, UNKNOWN | `RegimeLabel` | `core/contracts.py` |
| **Signal Direction** | Trade direction: LONG_SPREAD, SHORT_SPREAD, FLAT, EXIT | `SignalDirection` | `core/contracts.py` |
| **Signal Quality Grade** | Grade from A+ (highest) to F (block). CRISIS/BROKEN/TRENDING always get F. | `SignalQualityGrade` | `core/contracts.py` |
| **Entry Intent** | A proposal to enter a trade. NOT an order. Must pass portfolio/risk checks. | `EntryIntent` | `core/intents.py` |
| **Exit Intent** | A proposal to exit a trade with explicit `ExitReason`. | `ExitIntent` | `core/intents.py` |
| **Exit Reason** | Why a trade was exited (19 canonical reasons). | `ExitReason` | `core/contracts.py` |
| **Trade Lifecycle State** | 12-state machine: NOT_ELIGIBLE through RETIRED | `TradeLifecycleState` | `core/contracts.py` |
| **Signal Decision** | Output of `SignalPipeline`: direction, confidence, regime, quality, block reasons | `SignalDecision` | `core/signal_pipeline.py` |

## Portfolio Layer

| Term | Definition | Canonical Type | Module |
|------|-----------|----------------|--------|
| **Ranked Opportunity** | Entry intent scored across 7 dimensions for capital competition | `RankedOpportunity` | `portfolio/contracts.py` |
| **Allocation Decision** | Funded (or rejected) opportunity with rationale | `AllocationDecision` | `portfolio/contracts.py` |
| **Sizing Decision** | Notional amount after vol-target + scalar stack | `SizingDecision` | `portfolio/contracts.py` |
| **Drawdown State** | 6-level heat: NORMAL, CAUTIOUS, THROTTLED, DEFENSIVE, RECOVERY_ONLY, HALTED | `DrawdownState` | `portfolio/risk_ops.py` |
| **Kill-Switch Mode** | OFF, SOFT, REDUCE, HARD — escalation-only | `KillSwitchMode` | `portfolio/risk_ops.py` |

## ML Layer

| Term | Definition | Canonical Type | Module |
|------|-----------|----------------|--------|
| **Model Scorer** | The ONLY correct inference path. Handles fallback, TTL cache, never raises. | `ModelScorer` | `ml/inference/scorer.py` |
| **Fallback** | When no model is available, returns neutral probability (0.5). Check `fallback_triggered`. | `InferenceResult` | `ml/inference/scorer.py` |
| **Point-in-Time** | All features clip data to `as_of` as the first operation. Non-negotiable. | — | `ml/features/` |
| **Leakage Auditor** | Checks feature/label temporal alignment. CRITICAL findings block training. | `LeakageAuditor` | `ml/datasets/leakage.py` |
| **Champion/Challenger** | Model promotion lifecycle: CANDIDATE -> CHALLENGER -> CHAMPION -> RETIRED | `PromotionOutcome` | `ml/contracts.py` |

## Runtime / Operations

| Term | Definition | Canonical Type | Module |
|------|-----------|----------------|--------|
| **is_safe_to_trade()** | Conservative safety gate. Returns (False, [reasons]) if any blocker exists. | method | `runtime/state.py` |
| **Throttle Level** | NORMAL, REDUCED, EXITS_ONLY, HALTED — one-way downward automatically | `ThrottleLevel` | `control_plane/contracts.py` |
| **Service State** | STARTING, RUNNING, DEGRADED, HALTED, UNHEALTHY, DRAINING, STOPPED | `ServiceState` | `runtime/contracts.py` |
| **Exits-Only Mode** | Only exit trades allowed; no new entries. Set via control plane. | — | `control_plane/engine.py` |

## Governance

| Term | Definition | Canonical Type | Module |
|------|-----------|----------------|--------|
| **Audit Chain** | Append-only hash-linked log. Never modify existing entries. | `AuditChain` | `audit/chain.py` |
| **Evidence Bundle** | Collection of evidence items required before promotion/approval | `EvidenceBundle` | `evidence/` |
| **Surveillance Case** | Investigation triggered by surveillance rule breach | `SurveillanceCase` | `surveillance/engine.py` |
| **Exception Waiver** | Time-bounded exception to a control or policy. Has explicit expiry. | `ExceptionWaiver` | `exceptions_mgmt/` |
| **Attestation** | Periodic recertification that a control/process is still effective | `AttestationRequest` | `attestations/` |

## Deprecated Terms

| Old Term | Replacement | Status |
|----------|------------|--------|
| `TradeSide` (backtest) | `SignalDirection` | Deprecated, sunset pending |
| `TradeSide` (trade_logic) | `SignalDirection` | Deprecated, sunset pending |
| `secrets` package | `secrets_mgmt` package | Removed and renamed |
