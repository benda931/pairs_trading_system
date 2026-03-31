# Discovery Methodology

This document describes the pair discovery pipeline introduced in Phase 4 of the
platform build. It explains the philosophy, each stage of the pipeline, known
limitations, and extension points.

---

## Philosophy

### Correlation is a discovery primitive, not a tradability proof

Correlation between two return series tells you that common factors drive both
instruments. That is useful — it narrows the search space dramatically. But high
correlation does not guarantee:

- A mean-reverting spread exists
- The spread is stationary (cointegration)
- The relationship is stable over time
- The current regime supports mean-reversion trading

**Rule:** Use correlation (and distance) for *candidate generation*. Use
cointegration, stability analysis, and regime checks for *validation*.

### Residual mean reversion is the primary alpha abstraction

The tradable edge is not correlation — it is the predictable mean reversion of
the residual spread `log(X) - β·log(Y) - α`. A pair that is cointegrated but
whose spread is currently trending, vol-expanded, or mean-shifted is **not
tradable today**, regardless of its historical statistics.

### False discovery control

With N=100 instruments there are ~4,950 candidate pairs. Even at a 5% false-
positive rate across independent tests, ~247 pairs would pass by chance. In
practice pairs are correlated, making naive Bonferroni too conservative, but the
direction is clear: **be skeptical of high pass rates**. The pipeline tracks
`candidate_yield_rate` (final / screened) and produces a `rejection_breakdown`
in every `ResearchRunArtifact`.

---

## Universe Construction

**Entry point:** `research/universe.py`

### Eligibility filtering

`EligibilityFilter` applies hard threshold checks per instrument:

| Check | Default threshold |
|-------|------------------|
| Minimum history | 252 trading days |
| Minimum average dollar volume | $1M/day |
| Minimum price | $1.00 |
| Maximum missing data fraction | 5% |
| Maximum stale-price run | 5 consecutive days |
| Price anomaly (single-day return) | ±50% |

Every excluded instrument receives an `EligibilityDecision` with populated
`rejection_reasons` — never a silent drop. The `UniverseSnapshot` carries a full
`excluded_symbols` dict keyed by rejection reason for audit.

### Metadata inference

`InstrumentRegistry` holds a static sector/industry map for ~80 large-cap equities
and ~30 sector ETFs. For instruments not in the registry, `get_or_infer()` infers
sector from price co-movement patterns (first principal component loadings). Prefer
explicit registration for any instrument you intend to trade.

### Built-in universes

`BuiltinUniverses` provides ready-made symbol lists:

- `sp_sector_etfs()` — 11 SPDR sector ETFs
- `us_large_cap_equities()` — ~120 S&P 500 components
- `us_factor_etfs()` — value, growth, momentum, quality, low-vol ETFs
- `commodity_etfs()` — gold, oil, agriculture, metals
- `country_etfs()` — major country ETFs
- `custom(symbols)` — any user-supplied list

---

## Candidate Generation

**Entry point:** `research/candidate_generator.py`

The `CandidateGenerator` runs one or more discovery *families* and merges their
outputs. When the same pair is found by multiple families, the best (highest)
`discovery_score` is kept.

### Correlation family

Computes the full pairwise return correlation matrix using `pd.DataFrame.corr()`
(vectorised, O(N²) memory, appropriate for N ≤ ~500). Candidate filters:

- `|corr| ≥ min_correlation` (default 0.50 at discovery, 0.60 at validation)
- `|corr| ≤ max_correlation` (default 0.995, avoids near-identical ETFs)
- Minimum overlapping days ≥ `min_overlap_days` (default 252)
- Vol ratio `max(σx/σy, σy/σx) ≤ 5.0` (avoids micro/mega pairings)

Same-sector pairs receive a +0.05 `discovery_score` bonus (economic rationale
makes the relationship more stable across regimes).

### Distance family

Normalises each price series to 1.0 at the start of the window, then computes
the sum of squared differences (SSD). Lower SSD = more similar paths. Converts
to a `distance_score ∈ [0, 1]` (1 = identical paths). This family complements
correlation by capturing tracking-error-style similarity rather than directional
co-movement.

### Cluster family

Runs hierarchical clustering (Ward linkage, Euclidean distance on normalised
log-price returns) using `scipy.cluster.hierarchy`. Every within-cluster pair
is proposed as a candidate. This finds non-obvious pairs that cluster together
via factor co-movement without necessarily having high pairwise correlation.

### Cointegration-aware family

Applies a quick Engle-Granger cointegration pre-screen (α=0.20, intentionally
loose to avoid false rejections at this stage) to a shortlist of correlation/
distance survivors. This is stage-3 filtering, not stage-2, to avoid running
expensive statsmodels on all ~4,950 pairs.

### Stage pipeline summary

```
Stage 1: Universe eligibility (EligibilityFilter)
Stage 2: Cheap filter — corr bounds, overlap, vol ratio per family
Stage 3: EG cointegration pre-screen (optional, α=0.20)
Stage 4: Full PairValidator — ADF, half-life, Hurst, correlation, data quality
```

---

## Stability Analysis

**Entry point:** `research/stability_analysis.py`

Runs after candidate generation and before final scoring. Stability analysis
uses only data up to `train_end`.

### Rolling correlation

63-day (≈1 quarter) rolling Pearson correlation of log returns. Reports:
`corr_mean`, `corr_std`, `corr_min`, `corr_max`, `corr_trend` (slope of
rolling series). High `corr_std` (>0.30) triggers `UNSTABLE_CORRELATION`.

### Rolling hedge ratio

63-day rolling OLS beta (log_x ~ log_y). The coefficient of variation
`β_cv = σ(β) / |μ(β)|` measures how much the hedge ratio drifts. `β_cv > 0.40`
triggers `UNSTABLE_HEDGE_RATIO`. `SpreadSpecBuilder` uses `β_cv` to choose
between static OLS (< 0.20), rolling OLS (0.20–0.40), and Kalman filter (> 0.40).

### Structural break detection

`StructuralBreakDetector` uses two signals:

1. **CUSUM range statistic**: cumulative sum of standardised spread residuals,
   normalised by √n. Range > 2.5σ suggests a level shift.
2. **Rolling mean shift**: rolling-mean changes standardised by their own σ.
   A z-score > 3.0 on any change flags a mean shift.

`has_break = bool(cusum_break OR mean_shift)`. Break confidence is a weighted
combination of both signals. Triggers `STRUCTURAL_BREAK` instability reason when
`break_confidence > 0.5`.

The detector also computes `rolling_adf_pass_rate`: the fraction of 63-day
rolling windows where the spread passes ADF at α=0.10. A rate < 0.60 triggers
`FAILED_ADF_STATIONARITY`.

### Regime suitability

`RegimeSuitabilityChecker` examines the most recent 63 days of the spread:

| Signal | Threshold | Label |
|--------|-----------|-------|
| Autocorr lag-1 > 0.5 AND positive | Trending | `TRENDING` |
| σ_recent / σ_historical > 2.0 | Vol expansion | `HIGH_VOL` |
| Trending AND High Vol | Combined | `CRISIS` |
| Recent z-mean > 1.5 | Mean shift | `MEAN_SHIFTED` |
| None of the above | OK | `MEAN_REVERTING` |

`regime_suitable = False` triggers `REGIME_UNSUITABLE` and a -0.15 penalty on
the stability score. Note: a pair can be historically cointegrated but currently
in a `MEAN_SHIFTED` regime — do not trade it until the regime resolves.

### Stability score

```
score_raw = weighted_avg(
    corr_stability   * 0.25,   # 1 - corr_std / 0.5
    beta_stability   * 0.30,   # 1 - β_cv / 0.6
    adf_pass_rate    * 0.25,
    coint_pass_rate  * 0.20,   # optional, if run_rolling_coint=True
)
score_raw -= break_confidence * 0.3   # structural break penalty
score_raw -= 0.15 if not regime_suitable
stability_score = clip(score_raw, 0, 1)
```

---

## Discovery Scoring

**Entry point:** `research/discovery_pipeline.py` → `DiscoveryScorer`

`DiscoveryScore` has six dimension scores (each ∈ [0,1]) and a composite:

| Dimension | Weight | Key inputs |
|-----------|--------|-----------|
| correlation | 0.20 | |corr|; penalises near-1.0 |
| cointegration | 0.30 | ADF p-value, EG p-value, half-life sweet spot (5–60d) |
| stability | 0.25 | StabilityReport.stability_score |
| regime | 0.10 | regime_suitable flag |
| economic | 0.10 | same_sector bonus, economic_context presence |
| liquidity | 0.05 | liquidity tier of both instruments |

```
composite = Σ(dimension_score × weight) / Σ(weights)
```

Grades: A+ (≥0.85), A (≥0.70), B (≥0.55), C (≥0.40), D (≥0.25), F (<0.25).

---

## Validation Decision

```
REJECTED       — ValidationResult.FAIL from PairValidator
PORTFOLIO_READY — composite ≥ 0.70 AND is_stable AND regime_suitable
WATCHLIST      — composite ≥ 0.45
RESEARCH_ONLY  — everything else that isn't rejected
```

---

## Spread Proposal

`SpreadSpecBuilder` in `discovery_pipeline.py` selects the spread model based
on the rolling hedge ratio's coefficient of variation:

| β_cv | Model | Rationale |
|------|-------|-----------|
| < 0.20 | `OLS` (static) | Stable relationship; no need for adaptive beta |
| 0.20–0.40 | `ROLLING_OLS` | Moderate drift; rolling window adapts |
| > 0.40 | `KALMAN` | High drift; Kalman tracks regime changes in real time |

Entry/exit z-score thresholds are calibrated to the Hurst exponent:
- Hurst < 0.30 (strongly mean-reverting): tighter thresholds (entry 1.5, exit 0.3)
- Hurst 0.30–0.45: standard thresholds (entry 2.0, exit 0.5)
- Hurst > 0.45 (less mean-reverting): wider thresholds (entry 2.5, exit 0.7)

---

## ML and Agent Integration

### ML hook

`DiscoveryPipeline` produces a `ResearchRunArtifact` containing all
`PairValidationReport` objects and `StabilityReport` objects. Pass these to
`DatasetBuilder.build()` to construct leakage-safe feature matrices for
ranking-model training. The spread z-score at time T, forward half-life at T,
and stability score at T are the primary features; forward realized return is
the label.

### Agent hook

`UniverseDiscoveryAgent` (in `agents/base.py`) wraps `CandidateGenerator`.
`PairValidationAgent` wraps `PairValidator`. Compose them via `AgentRegistry`
to build a fully audited, reproducible research run. Each agent call produces
an `AgentAuditLogger` trail with timestamps and decision provenance.

---

## Known Limitations

1. **Cluster family is sensitive to window length.** Hierarchical clustering
   uses the full price history — changing the window changes cluster membership.
   Use a fixed `train_end` for reproducibility.

2. **Cointegration pre-screen at α=0.20 has high false-positive rate.** It
   is intentional (better to over-generate candidates and filter at stage 4),
   but be aware that stage-4 rejection rates from the cointegration family will
   be high (~60–70%).

3. **Regime suitability uses lookback_days=63 (one quarter).** For pairs with
   long half-lives (60–120 days), a 63-day look-back may not capture one full
   mean-reversion cycle. Consider extending `lookback_days` for slow-moving pairs.

4. **StabilityAnalyzer requires ≥ 3× rolling_window observations.** With
   rolling_window=63, you need ≥ 189 trading days (~9 months) of clean data.
   Pairs with shorter histories return `is_stable=True` by default (no penalty
   for lack of data, see `analyze()` early return logic).

5. **Discovery scoring weights are hand-calibrated, not learned.** The 0.25/0.30/
   0.25/... weights are reasonable defaults, not backtested optima. Use the ML
   ranking model (via `DatasetBuilder` + `BaseModel`) to learn data-driven weights
   before routing to live portfolio.
