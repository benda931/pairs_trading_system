# ML Architecture — Pairs Trading System

**Version:** 1.0
**Last updated:** 2026-03-28
**Scope:** Full ML platform — features, labels, datasets, models, evaluation, registry, inference, monitoring, governance, explainability

---

## Table of Contents

1. [Overview and Philosophy](#1-overview-and-philosophy)
2. [ML Task Families](#2-ml-task-families)
3. [Directory Structure](#3-directory-structure)
4. [Feature Platform](#4-feature-platform)
5. [Label Platform](#5-label-platform)
6. [Dataset Construction and Leakage Prevention](#6-dataset-construction-and-leakage-prevention)
7. [Model Layer](#7-model-layer)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Model Registry and Governance](#9-model-registry-and-governance)
10. [Inference Layer](#10-inference-layer)
11. [Monitoring and Drift Detection](#11-monitoring-and-drift-detection)
12. [Explainability and Lineage](#12-explainability-and-lineage)
13. [Integration Points with Core System](#13-integration-points-with-core-system)
14. [MLUsageContract — Hard Rules](#14-mlusagecontract--hard-rules)
15. [Known Limitations and Future Roadmap](#15-known-limitations-and-future-roadmap)

---

## 1. Overview and Philosophy

### North Star

ML is a **performance enhancement overlay**, not the system's foundation.
Every ML-augmented decision has a deterministic rule-based fallback. If the model
is stale, drifted, or missing features, the system falls back gracefully to the
parametric rule — it does not fail, and it does not silently degrade.

### Core Doctrine

**1. Point-in-time correctness is non-negotiable.**
Every feature computation clips its input data to `as_of` before any calculation.
The `as_of` parameter is never optional in backtesting contexts. Future information
must never contaminate a training observation.

**2. Labels are explicitly forward-looking — and documented as such.**
Every `LabelDefinition` carries a `leakage_risks` field. Every `LabelBuilder` method
contains an explicit `# LEAKAGE NOTE:` comment explaining what future data it uses
and why that is correct by design (label construction intentionally uses future data;
feature construction never does).

**3. Purged K-Fold is the only acceptable cross-validation for time-series.**
Standard K-Fold introduces data leakage through temporal proximity. All CV uses
`TemporalSplitter` with `purge_days` and `embargo_days` gaps to prevent label leakage
across folds.

**4. ML never overrides hard risk rules.**
`MLUsageContract.may_override_risk_limit` is always `False`. An ML model may *recommend*
a different size or threshold, but `KillSwitchManager`, `DrawdownManager`, and
`TradeLifecycleStateMachine` hard vetoes are never bypassed by ML output.

**5. Champion/challenger governance; never silent promotion.**
A model only enters CHAMPION status after passing `GovernanceEngine.check_promotion_criteria()`.
The previous champion is recorded in `champion_challenger_history` before demotion.
Rollback means simply re-promoting the previous champion from registry.

**6. Fallback policy is explicit and tiered.**
`ModelScorer` tries CHAMPION → CHALLENGER → CANDIDATE in order. If all fail or
all models are stale/drifted, it returns a neutral `InferenceResult` with
`fallback_triggered=True`. Callers must inspect this flag.

---

## 2. ML Task Families

The platform supports 9 ML task families, all defined in `ml/contracts.py` as
`MLTaskFamily` enum values:

| Family | Purpose | Primary Label | Entry Point |
|--------|---------|---------------|-------------|
| `CANDIDATE_RANKING` | Score pairs for portfolio inclusion | `pair_reversion_success_10d` | `OpportunityRanker` ML hook |
| `META_LABELING` | Filter/size trade entry proposals | `meta_take_profit_binary` | `SignalQualityEngine` meta-label hook |
| `REGIME_CLASSIFICATION` | Classify current market regime | `regime_label_fwd` | `RegimeEngine` ML hook |
| `BREAK_DETECTION` | Detect structural breaks | `structural_break_fwd_20d` | `SignalQualityEngine` break score |
| `HOLDING_TIME` | Estimate optimal holding period | `optimal_holding_days` | Trade sizing / risk ops |
| `THRESHOLD_RECOMMENDATION` | Suggest entry/exit z-score thresholds | `threshold_return_efficiency` | `ThresholdEngine` |
| `SIZING_ASSISTANCE` | Adjust position size | `pair_max_favorable_size` | `SizingEngine` |
| `PORTFOLIO_RANKING` | Portfolio-level opportunity scoring | `portfolio_utility_score` | `PortfolioAllocator` |
| `ANOMALY_DETECTION` | Detect anomalous spread behaviour | `spread_anomaly_binary` | Trade monitoring |

---

## 3. Directory Structure

```
ml/
├── __init__.py
├── contracts.py              # Single source of truth for all ML domain objects
│
├── features/                 # Feature platform
│   ├── definitions.py        # 61 FeatureDefinition objects + FEATURE_REGISTRY + FEATURE_GROUPS
│   └── builder.py            # PointInTimeFeatureBuilder — all data clipped to as_of
│
├── labels/                   # Label platform
│   ├── definitions.py        # 26 LabelDefinition objects + LABEL_REGISTRY
│   └── builder.py            # LabelBuilder — explicit LEAKAGE NOTEs, shift(-horizon) labels
│
├── datasets/                 # Dataset construction
│   ├── splits.py             # TemporalSplitter — walk-forward / rolling-origin with purge+embargo
│   ├── leakage.py            # LeakageAuditor — 6 structural checks before training
│   └── builder.py            # MLDatasetBuilder — orchestrates features→labels→splits→audit
│
├── models/                   # Model implementations
│   ├── base.py               # MLModel — leakage guard, structured predictions, model card
│   ├── classifiers.py        # LR / RF / GBM / XGBoost + build_model() factory
│   ├── meta_labeler.py       # MetaLabelModel — implements MetaLabelProtocol
│   ├── regime_classifier.py  # RegimeClassificationModel — implements RegimeClassifierHookProtocol
│   ├── break_detector.py     # BreakDetectionModel — structural break probability
│   └── calibration.py        # CalibratedModelWrapper — isotonic / Platt calibration
│
├── evaluation/               # Evaluation framework
│   ├── metrics.py            # IC, IC-t, AUC-ROC, PR-AUC, Brier, ECE, meta_label_utility
│   └── reports.py            # ModelEvaluator — full evaluation + champion comparison
│
├── registry/                 # Model registry
│   └── registry.py           # MLModelRegistry — thread-safe, champion/challenger, JSON persistence
│
├── inference/                # Inference layer
│   └── scorer.py             # ModelScorer — TTL caching, tiered fallback, never raises
│
├── monitoring/               # Runtime monitoring
│   ├── drift.py              # FeatureDriftMonitor (PSI), ModelHealthMonitor
│   └── health.py             # psi_score(), kolmogorov_smirnov_drift()
│
├── governance/               # Governance policies
│   └── policies.py           # GovernanceEngine — promotion criteria, usage contracts
│
├── explainability/           # Explainability and lineage
│   ├── importance.py         # compute_feature_importance(), generate_importance_report()
│   └── lineage.py            # FeatureLineageTracker — training lineage records
│
└── research/                 # (reserved for offline research tooling)
```

---

## 4. Feature Platform

### Feature Definitions (`ml/features/definitions.py`)

61 features are registered across 6 entity scopes:

| Entity Scope | Count | Examples |
|---|---|---|
| `INSTRUMENT` | ~10 | `inst_ret_1d`, `inst_vol_20d`, `inst_atr_14d`, `inst_beta_spy_60d` |
| `PAIR` | ~20 | `pair_zscore`, `pair_zscore_lag1`, `pair_half_life`, `pair_rolling_corr_60d`, `kalman_residual_z` |
| `REGIME` | ~6 | `regime_label_encoded`, `regime_break_confidence`, `vix_level`, `yield_curve_slope` |
| `SIGNAL` | ~10 | `signal_conviction`, `signal_days_open`, `unrealized_pnl_z`, `entry_zscore_distance` |
| `PORTFOLIO` | ~8 | `portfolio_utilization`, `active_pairs_count`, `drawdown_pct`, `heat_level_encoded` |
| `EXECUTION` | ~7 | `bid_ask_spread_bps`, `market_impact_est_bps`, `adv_ratio`, `slippage_est_bps` |

All features are grouped into 12 named `FeatureGroup` sets accessible via `FEATURE_GROUPS`:

```python
from ml.features.definitions import FEATURE_GROUPS

meta_label_features = FEATURE_GROUPS["meta_label_features"].feature_names
# → ["pair_zscore", "pair_zscore_lag1", "signal_conviction", "regime_label_encoded", ...]
```

### Feature Builder (`ml/features/builder.py`)

`PointInTimeFeatureBuilder` enforces point-in-time correctness:

```python
builder = PointInTimeFeatureBuilder()
features = builder.build_feature_vector(
    pair_id=pair_id,
    as_of=datetime(2024, 6, 15),          # Hard clip boundary
    prices_x=prices_x,
    prices_y=prices_y,
    spread=spread,
    signal_context=ctx,
)
```

**Critical invariant:** The first operation inside every `compute_*` method is:
```python
data = data[data.index <= as_of]
```
This guard is applied before any calculation, even if the caller passes a full history.

---

## 5. Label Platform

### Label Definitions (`ml/labels/definitions.py`)

26 labels across 8 families, each with explicit `leakage_risks` and `valid_use_cases`:

| Family | Count | Key Labels |
|---|---|---|
| `REVERSION_SUCCESS` | 4 | `pair_reversion_success_10d`, `pair_reversion_success_20d`, `reversion_return_5d` |
| `META_LABEL` | 3 | `meta_take_profit_binary`, `meta_stop_loss_binary`, `meta_size_multiplier` |
| `RELATIONSHIP_PERSISTENCE` | 3 | `cointegration_holds_30d`, `correlation_stable_30d`, `spread_stationary_30d` |
| `BREAK_INSTABILITY` | 3 | `structural_break_fwd_20d`, `hr_instability_fwd_10d`, `regime_change_fwd_5d` |
| `HOLDING_TIME` | 2 | `optimal_holding_days`, `max_favorable_holding_days` |
| `THRESHOLD_BAND` | 3 | `threshold_return_efficiency`, `optimal_entry_z`, `optimal_exit_z` |
| `PORTFOLIO_UTILITY` | 3 | `portfolio_utility_score`, `pair_max_favorable_size`, `capital_efficiency` |
| `EXECUTION_VIABILITY` | 3 | `execution_cost_pct`, `spread_anomaly_binary`, `adverse_fill_probability` |

### Label Builder (`ml/labels/builder.py`)

`LabelBuilder` uses `pd.Series.shift(-horizon)` for all forward-looking labels:

```python
builder = LabelBuilder()

# Binary reversion success: did spread return to within exit_z within horizon?
reversion = builder.build_reversion_label(spread=spread, horizon=10, exit_z=0.5)

# Meta-label: did the signal result in take-profit within horizon?
meta = builder.build_meta_take_label(spread=spread, entry_z=2.0, tp_z=0.5, horizon=15)
```

**Leakage is intentional in labels.** `shift(-N)` uses future data — that is what
creates the label. Leakage prevention applies to *features*, not labels.

---

## 6. Dataset Construction and Leakage Prevention

### Temporal Splitting (`ml/datasets/splits.py`)

`TemporalSplitter` produces leak-free train/test splits for time-series:

```python
splitter = TemporalSplitter(purge_days=5, embargo_days=10)

# Walk-forward (expanding window)
plans = splitter.walk_forward_splits(data, n_splits=5, test_days=252, min_train_days=504)

# Rolling origin (fixed window)
plans = splitter.rolling_origin_splits(data, train_days=504, test_days=252)
```

Each `DatasetSplitPlan` records the exact index boundaries for full auditability.

**Purge days:** removes observations from the end of the training window whose labels
may contain information from the test period (label horizon overlap).

**Embargo days:** removes the first N days of the test window to prevent near-boundary
regime/spread momentum leakage.

### Leakage Auditor (`ml/datasets/leakage.py`)

`LeakageAuditor` runs 6 structural checks before any model is trained:

| Check | What it detects |
|---|---|
| Index monotonicity | Non-monotonic timestamps (shuffled data) |
| Future feature detection | Feature timestamps after the label timestamp |
| Train/test overlap | Overlapping index between train and test sets |
| Normalization leak heuristic | Columns with mean ≈ 0 and std ≈ 1 (suspect if not normalized on train only) |
| Label-horizon contamination | Test labels that use data within purge_days of train end |
| Null fraction | Feature columns with > 20% NaN (unreliable inputs) |

Any `CRITICAL` severity finding blocks training. `WARNING` findings are logged but
do not block. All findings are returned in `LeakageAuditReport` with full details.

### Dataset Builder (`ml/datasets/builder.py`)

```python
builder = MLDatasetBuilder(feature_builder=fb, label_builder=lb, splitter=splitter)

# Single train/test split
snapshot, X_train, X_test, y_train, y_test = builder.build(
    pair_id=pair_id,
    prices_x=prices_x,
    prices_y=prices_y,
    spread=spread,
    label_name="meta_take_profit_binary",
    feature_group="meta_label_features",
    train_end=datetime(2023, 12, 31),
)

# Walk-forward dataset (list of fold tuples)
folds = builder.build_walk_forward_datasets(...)
```

---

## 7. Model Layer

### Base Model (`ml/models/base.py`)

`MLModel` wraps any sklearn-compatible estimator with institutional guardrails:

```python
model = MLModel(
    estimator=LogisticRegression(),
    model_id="meta_label_lr_v1",
    task_family=MLTaskFamily.META_LABELING,
    feature_names=META_LABEL_FEATURES,
    train_end=datetime(2023, 12, 31),   # Leakage guard boundary
)

model.fit(X_train, y_train)  # Raises ValueError if X contains timestamps after train_end

predictions: list[PredictionRecord] = model.predict_structured(X_test, entity_ids=[...])
card: ModelCard = model.generate_model_card(metrics=eval_result.metrics)
```

**Leakage guard in `fit()`:** If the training `DataFrame` has a `DatetimeIndex`, any
row with `index > train_end` raises `ValueError`. This is not a soft warning — it
is a hard failure.

### Canonical Feature Lists

`ml/models/base.py` exports canonical feature name lists that wire models to the
correct feature groups:

```python
from ml.models.base import META_LABEL_FEATURES, REGIME_FEATURES, BREAK_DETECTION_FEATURES
```

These lists match the `FEATURE_GROUPS` definitions. Use them as `feature_names` when
constructing `MLModel` to ensure registry alignment.

### Specialized Models

| Model Class | File | Protocol Implemented |
|---|---|---|
| `MetaLabelModel` | `ml/models/meta_labeler.py` | `MetaLabelProtocol` (core/signal_quality.py) |
| `RegimeClassificationModel` | `ml/models/regime_classifier.py` | `RegimeClassifierHookProtocol` (core/regime_engine.py) |
| `BreakDetectionModel` | `ml/models/break_detector.py` | — (standalone, consumed by SignalQualityEngine) |
| `CalibratedModelWrapper` | `ml/models/calibration.py` | Wraps any `MLModel` |

**`MetaLabelModel`** — three configurable thresholds:
- `take_threshold` (default 0.55): route to take-profit
- `downsize_threshold` (default 0.45): reduce size
- `skip_threshold` (default 0.35): skip trade entirely

**`RegimeClassificationModel`** — safety floor enforced:
If the underlying estimator predicts MEAN_REVERTING but `RegimeFeatureSet.break_confidence > 0.80`,
the output is overridden to BROKEN. ML cannot override the break confidence floor.
Additionally, if the model's confidence is below `min_confidence` (default 0.55), it falls
back to `fallback_regime` (default `UNKNOWN`).

**`CalibratedModelWrapper`** — calibration is always on held-out data:
```python
wrapper = CalibratedModelWrapper(base_model=model, method="isotonic")
wrapper.calibrate(X_calib, y_calib)   # Never on training data
```

### Model Factory

```python
from ml.models.classifiers import build_model

model = build_model(
    model_type="gradient_boosting",    # "logistic", "random_forest", "gradient_boosting", "xgboost"
    model_id="regime_clf_gbm_v2",
    task_family=MLTaskFamily.REGIME_CLASSIFICATION,
    feature_names=REGIME_FEATURES,
    train_end=datetime(2023, 12, 31),
    n_estimators=200,
    max_depth=4,
)
```

---

## 8. Evaluation Framework

### Metrics (`ml/evaluation/metrics.py`)

Domain-appropriate metrics for statistical arbitrage ML tasks:

| Metric | Function | Use Case |
|---|---|---|
| Information Coefficient | `information_coefficient(y_true, y_pred)` | Ranking quality (Spearman ρ) |
| IC t-stat | `ic_t_stat(ic_series)` | Statistical significance of IC |
| AUC-ROC | `auc_roc(y_true, y_prob)` | Binary classifier discrimination |
| PR-AUC | `pr_auc(y_true, y_prob)` | Imbalanced binary classification |
| Brier Score | `brier_score(y_true, y_prob)` | Probability calibration quality |
| Expected Calibration Error | `calibration_error(y_true, y_prob)` | Reliability of confidence scores |
| Meta-Label Utility | `meta_label_utility(y_true, y_pred, returns)` | Trade P&L filtered by meta-labels |
| Walk-Forward IC | `walk_forward_ic(y_true_list, y_pred_list)` | Robustness across folds |
| Regime-Sliced Metrics | `regime_sliced_metrics(y_true, y_pred, regimes)` | Per-regime performance |
| Robustness Score | `robustness_score(fold_metrics)` | IC stability across folds |

### Model Evaluator (`ml/evaluation/reports.py`)

```python
evaluator = ModelEvaluator()
result = evaluator.evaluate(model, X_test, y_test)
# Returns: EvaluationResult with metrics dict + summary text

# Compare to a rule-based baseline
comparison = evaluator.compare_to_baseline(model, baseline_predictions, X_test, y_test)
# champion_better=True / False with delta metrics

# Walk-forward evaluation
wf_result = evaluator.walk_forward_evaluation(model, folds)
# fold_results + aggregate_metrics + robustness_score
```

---

## 9. Model Registry and Governance

### Model Registry (`ml/registry/registry.py`)

Thread-safe registry with champion/challenger tracking:

```python
registry = get_ml_registry()  # Singleton

registry.register(model)
registry.promote(model.model_id, PromotionOutcome.CHAMPION)
# → Previous CHAMPION automatically demoted to RETIRED (recorded in history)

champion = registry.get_champion(MLTaskFamily.META_LABELING)
challenger = registry.get_challenger(MLTaskFamily.META_LABELING)

comparison = registry.compare_champion_challenger(MLTaskFamily.META_LABELING)
# Returns dict with metric deltas

registry.save("studies/ml_registry.json")
registry.load("studies/ml_registry.json")
```

**Statuses:** `CANDIDATE → CHALLENGER → CHAMPION → DEPRECATED / RETIRED`

Promotion flow:
1. Train and evaluate a new `CANDIDATE`
2. Pass `GovernanceEngine.check_promotion_criteria()` → elevate to `CHALLENGER`
3. Shadow-run for ≥ 2 weeks in parallel with CHAMPION
4. Pass IC, AUC-ROC, Brier thresholds on out-of-time data → elevate to `CHAMPION`
5. Previous CHAMPION demoted to `RETIRED` (never deleted)

### Governance Engine (`ml/governance/policies.py`)

```python
engine = GovernanceEngine()

# Check if candidate can be promoted (thresholds vary by task family)
result = engine.check_promotion_criteria(artifact)
# result.approved = True/False, result.failing_criteria = [...]

# Enforce usage contract
engine.check_usage_contract(contract)
# Raises if may_override_risk_limit=True

# Evaluate retirement
engine.evaluate_retirement_criteria(model)
# Returns RetirementRecommendation
```

**Promotion thresholds by task family:**

| Task Family | Min IC | Min AUC-ROC | Max Brier |
|---|---|---|---|
| `META_LABELING` | 0.05 | 0.55 | 0.25 |
| `REGIME_CLASSIFICATION` | 0.10 | 0.60 | 0.22 |
| `BREAK_DETECTION` | 0.08 | 0.58 | 0.24 |
| `CANDIDATE_RANKING` | 0.06 | 0.55 | 0.25 |
| All others | 0.05 | 0.55 | 0.25 |

---

## 10. Inference Layer

### Model Scorer (`ml/inference/scorer.py`)

`ModelScorer` is the **only** correct way to get ML predictions at runtime:

```python
scorer = ModelScorer(registry=get_ml_registry(), ttl_seconds=300)

result = scorer.score(
    task_family=MLTaskFamily.META_LABELING,
    features={"pair_zscore": 2.1, "signal_conviction": 0.72, ...},
)
# result.prediction: float
# result.confidence: float
# result.fallback_triggered: bool  ← ALWAYS CHECK THIS
# result.model_id: str
```

**Never raises.** On any error (missing model, stale TTL, exception inside predict),
`score()` returns a neutral `InferenceResult` with `fallback_triggered=True` and
`prediction=0.5` (neutral probability). The caller is responsible for checking
`fallback_triggered` before acting on the prediction.

**Fallback order:** CHAMPION → CHALLENGER → CANDIDATE → neutral result

**TTL caching:** Models are cached in-process for `ttl_seconds`. After TTL expiry,
the next `score()` call reloads from registry. Set `ttl_seconds=0` to disable caching
(not recommended in production).

---

## 11. Monitoring and Drift Detection

### Feature Drift (`ml/monitoring/drift.py`)

```python
monitor = FeatureDriftMonitor(psi_warning=0.10, psi_critical=0.20)
monitor.fit_reference(X_reference)  # Fit on out-of-time holdout (not training data)

report = monitor.generate_drift_report(X_current, model_id="meta_label_lr_v1")
# report.overall_severity: DriftSeverity (NONE / LOW / MEDIUM / HIGH / CRITICAL)
# report.feature_psi: Dict[str, float]
# report.drifted_features: List[str]
```

**PSI thresholds:**
- `< 0.10` — NONE (population stable)
- `0.10 – 0.20` — MEDIUM (monitor closely)
- `> 0.20` — HIGH/CRITICAL (consider retraining)

### Model Health (`ml/monitoring/drift.py`, `ml/monitoring/health.py`)

```python
health_monitor = ModelHealthMonitor(max_staleness_days=30, min_ic_threshold=0.03)

health = health_monitor.check_health(model_metadata, recent_metrics)
# health.state: ModelHealthState (HEALTHY / DEGRADED / STALE / DRIFTED / RETIRED)

summary = health_monitor.generate_health_summary(model_metadata, recent_metrics, reference_data, current_data)
```

**Health states:**
- `HEALTHY` — IC above threshold, data fresh, no significant drift
- `DEGRADED` — IC has declined but model is still marginally positive
- `STALE` — last training > `max_staleness_days` ago
- `DRIFTED` — PSI > `psi_critical` on feature distribution
- `RETIRED` — model should no longer be used

---

## 12. Explainability and Lineage

### Feature Importance (`ml/explainability/importance.py`)

```python
artifact = generate_importance_report(
    model=model,
    X_test=X_test,
    y_test=y_test,
    method="gain",              # "gain" / "coef" / "permutation"
)
# artifact.feature_importances: Dict[str, float]
# artifact.top_features: List[str]  (sorted by importance)
# artifact.method: str
```

Three importance methods:
- **`gain`** — tree-based feature importance (Gini impurity reduction)
- **`coef`** — absolute coefficient magnitude (linear models)
- **`permutation`** — model-agnostic; measures accuracy drop when feature is shuffled

### Feature Lineage (`ml/explainability/lineage.py`)

```python
tracker = FeatureLineageTracker()

tracker.record_training_lineage(
    model_id="meta_label_lr_v1",
    feature_names=META_LABEL_FEATURES,
    train_start=datetime(2020, 1, 1),
    train_end=datetime(2023, 12, 31),
    data_sources=["pt_prices", "spread_series"],
)

lineage = tracker.get_model_lineage("meta_label_lr_v1")
report = tracker.generate_lineage_report("meta_label_lr_v1")

# Verify feature availability before scoring
compatible = tracker.check_feature_compatibility(
    model_id="meta_label_lr_v1",
    available_features=set(current_feature_vector.keys()),
)
```

---

## 13. Integration Points with Core System

Three protocol interfaces are implemented by the ML models and consumed by the core system:

### 1. `MetaLabelProtocol` — `core/signal_quality.py`

```python
# core/signal_quality.py wires it at construction time
from ml.models.meta_labeler import MetaLabelModel

engine = SignalQualityEngine(meta_label_hook=MetaLabelModel.load("registry/meta_label_v1"))
quality = engine.assess(spread=spread, regime=regime, conviction=0.72)
# quality.meta_label_action: MetaLabelAction (TAKE / DOWNSIZE / SKIP)
```

### 2. `RegimeClassifierHookProtocol` — `core/regime_engine.py`

```python
# core/regime_engine.py wires it at construction time
from ml.models.regime_classifier import RegimeClassificationModel

engine = RegimeEngine(ml_hook=RegimeClassificationModel.load("registry/regime_v2"))
label, confidence = engine.classify(features=regime_features)
# Safety floor: BROKEN/CRISIS cannot be overridden by ML
```

### 3. `RankingMLHookProtocol` — `portfolio/ranking.py`

```python
# portfolio/ranking.py wires it at construction time
scorer = ModelScorer(registry=get_ml_registry())
ranker = OpportunityRanker(ml_hook=scorer)
ranked = ranker.rank(intents, portfolio_state, regime)
# ML score contributes one dimension to the 7-dimension composite
```

---

## 14. MLUsageContract — Hard Rules

`MLUsageContract` (defined in `ml/contracts.py`) codifies what ML is and is not
permitted to do. `GovernanceEngine.check_usage_contract()` enforces these at
registration time.

```python
@dataclass(frozen=True)
class MLUsageContract:
    task_family: MLTaskFamily
    may_override_risk_limit: bool = False     # ALWAYS False — hard rule
    may_block_entry: bool = True              # Meta-label may SKIP
    may_adjust_size: bool = True              # Sizing assistance
    may_recommend_exit: bool = True           # Break/holding-time models
    max_size_adjustment_pct: float = 0.50     # Max ±50% size change
    requires_fallback: bool = True            # Fallback must be defined
    min_confidence_to_act: float = 0.55       # Below this → use fallback
```

**Invariant enforced at registration:** `may_override_risk_limit` is always `False`.
Any attempt to register a model with this flag set to `True` raises `ValueError`
in `GovernanceEngine.check_usage_contract()`.

---

## 15. Known Limitations and Future Roadmap

### Current Limitations

**1. No online learning.**
All models are batch-trained. Market regime changes may cause performance degradation
before the next scheduled retraining. The `ModelHealthMonitor` will detect this as
`DEGRADED` / `DRIFTED`, but the human must trigger retraining.

**2. Synthetic data in tests.**
The 115 ML platform tests use synthetic numpy arrays with fixed seeds. While they
validate architecture, temporal constraints, and API contracts, they do not validate
model accuracy on real market data.

**3. PSI requires ≥ 200 reference observations.**
`FeatureDriftMonitor.fit_reference()` will still compute PSI on smaller samples, but
the percentile bins will be noisy. Minimum 252 days of reference data is recommended.

**4. No online feature store.**
`PointInTimeFeatureBuilder` computes features on-demand. For high-frequency monitoring
(intraday), a pre-computed feature cache with TTL would be needed.

**5. Explainability coverage is partial.**
`FeatureLineageTracker` tracks training provenance. SHAP-based explanations are planned
but not yet implemented (the `enable_shap` config flag exists but is currently unused
by the ML platform; SHAP was used in the legacy `core/ml_analysis.py` module).

**6. `RegimeClassificationModel` uses a closed label set.**
The classifier maps to `RegimeLabel` enum values. If a new regime type is added
(e.g., `STAGFLATION`), the model must be retrained from scratch — there is no
zero-shot regime extension.

### Future Roadmap

| Priority | Item | Rationale |
|---|---|---|
| P0 | Scheduled retraining pipeline | Detect STALE/DEGRADED → auto-retrain on fresh data |
| P0 | SHAP integration via `ExplainabilityArtifact` | Use existing `enable_shap` config flag |
| P1 | Online feature store with TTL cache | Required for live trading use cases |
| P1 | Isotonic calibration evaluation harness | Reliability diagrams in dashboard |
| P2 | Uncertainty quantification (conformal prediction) | Non-parametric prediction intervals |
| P2 | Multi-output model support (threshold + size simultaneously) | Reduce inference latency |
| P3 | AutoML integration for hyperparameter search | Extend existing Optuna infrastructure |
| P3 | Causal ML exploration (DoubleML) | Deconfound macro factor effects on spread returns |

---

## How to Add a New ML Task (Quick Reference)

1. Add `MLTaskFamily.YOUR_TASK` to `ml/contracts.py`
2. Define feature group in `ml/features/definitions.py` → add to `FEATURE_GROUPS`
3. Define label in `ml/labels/definitions.py` → add to `LABEL_REGISTRY`
4. Add label builder method in `ml/labels/builder.py` with explicit `# LEAKAGE NOTE:`
5. Add promotion thresholds in `ml/governance/policies.py` `PROMOTION_CRITERIA` dict
6. Create model class (subclass `MLModel` or use `build_model()` factory)
7. Wire protocol hook in the consuming core/portfolio module
8. Add tests in `tests/test_ml_platform.py` — at minimum: leakage check, temporal split, inference fallback
