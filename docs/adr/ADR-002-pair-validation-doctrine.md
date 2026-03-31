# ADR-002: Explicit Pair Validation Doctrine

**Status:** Accepted
**Date:** 2026-03-27

## Context

Pair selection was previously implicit — pairs passed or failed based on a loose
set of checks spread across different functions, with no record of *why* a pair
was rejected. This made it impossible to debug false rejections or audit the
selection process.

## Decision

All pair validation runs through `research/pair_validator.py::PairValidator`.

Every test is classified as either **hard** (failure = REJECT) or **soft** (failure = WARN).
Every test result is captured in a `ValidationTest` object with:
- `name`: the test identifier
- `result`: PASS / FAIL / WARN / SKIP
- `value`: the computed metric
- `threshold`: what the threshold was
- `message`: human-readable explanation

The `PairValidationReport` always carries a populated `rejection_reasons` list
when `result == FAIL`. A pair that passes all hard tests but fails some soft tests
gets `result == WARN` and is still eligible for trading at reduced size.

### Hard tests (any failure → REJECT):
1. Minimum correlation (default 0.60)
2. ADF stationarity of spread (p-value ≤ 0.10)
3. Engle-Granger cointegration (p-value ≤ 0.10)
4. Half-life bounds (2 ≤ days ≤ 120)

### Soft tests (failure → WARN only):
5. Hurst exponent (should be < 0.45 for mean reversion)
6. Rolling hedge ratio stability (coefficient of variation < 0.30)

## Consequences

**Positive:**
- Every rejection is explainable
- Easy to tune thresholds by pair type
- Audit trail for compliance / research review

**Negative:**
- More verbose output than a simple boolean pass/fail
- Requires statsmodels dependency for ADF/cointegration

## Validation Thresholds Are Configurable

Pass a `ValidationThresholds` object to `PairValidator()` to override defaults.
This allows different strictness levels for research vs production.
