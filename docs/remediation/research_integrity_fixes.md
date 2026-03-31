# Research Integrity Remediation — Fix Log

**Date:** 2026-03-31
**Program:** P0/P1 Research Integrity Hardening
**Engineer:** Principal Quant Platform Hardening

---

## Fix Summary Table

| Finding ID | Severity | File(s) | Line(s) Changed | Change Made | Before | After | Residual Risk |
|------------|----------|---------|-----------------|-------------|--------|-------|---------------|
| R-001 | P0 | `root/optimization_tab.py` | ~7126-7165 (docstring, min_seg_days, row dict) | Added `_WF_DISCLAIMER` constant, calendar-validation docstring warning, raised min segment floor, added `wf_mode` column to output | `min_seg_days = max(5, ...)` with no semantic label on output | `min_seg_days = max(63, ...)`, `wf_mode = "calendar_validation"` in every row, explicit WARNING in docstring | Function still runs pre-optimized params across segments, not true IS+OOS walk-forward. True WF is in `research/walk_forward.py`. |
| R-002 | P1 | `research/pair_validator.py`, `core/contracts.py` | pair_validator.py ~74, contracts.py ~855 | Added ADR-007 comment on `min_obs` field; confirmed `ValidationThresholds.MIN_OBS = 252`; added inline reference comment in contracts.py | `min_obs: int = ValidationThresholds.MIN_OBS` with no rationale comment | Same value (252 was already correct) with ADR-007 rationale documented at definition site | Downstream callers that construct `ValidationConfig(min_obs=60)` explicitly can still override. No runtime enforcement of the floor beyond the default. |
| R-003 | P1 | `core/optimization_backtester.py` | ~197-205 (BacktestConfig), ~1701-1719 (method docstring), ~1762-1765 (PnL loop) | Added `execution_lag_bars: int = 1` field to `BacktestConfig`; expanded `_bt_simulate_equity` docstring to explain lagged-weight PnL semantics; added inline comment at the `w[i-1]*ret[i]` line | No documentation of execution timing; `execution_lag_bars` not exposed | `execution_lag_bars=1` explicit in config; docstring explains EOD signal → next-bar execution equivalence; PnL comment references the lag | Signals still computed on close[t], which assumes perfect fill at close. Intraday execution quality (open vs VWAP vs close) is not modeled. Commission/slippage bps partially compensate. |
| R-004 | P2 | `research/universe.py`, `research/discovery_contracts.py` | universe.py ~381-404, ~407-431; discovery_contracts.py ~239-243 | Added survivorship-bias-mitigation docstring to `EligibilityFilter`; added per-check inline comments; added R-004 comments to `UniverseDefinition` eligibility threshold fields | No mention of survivorship bias in any filter; `min_price = 1.0` with only terse comment | Each filter explicitly labeled as survivorship-bias mitigation with rationale; residual risk noted in class docstring | CRITICAL RESIDUAL: filters apply to current universe snapshot only. Stocks that were eligible historically but have since delisted remain in training data if they are present in price history. True point-in-time survivorship-bias-free data requires CRSP or equivalent — not available in this system. |

---

## Narrative Summary

### What Was Fixed

**R-001 (P0) — Walk-Forward Mislabeling:** The function `_run_walkforward_for_params` in `root/optimization_tab.py` was operating as a post-optimization calendar stability validator, not a true walk-forward optimizer, with no documentation of this distinction. The minimum segment floor of 5 days allowed statistically meaningless sub-segments. Fixes: (1) raised the minimum segment floor from 5 to 63 calendar days (approximately one calendar quarter), (2) added an explicit `_WF_DISCLAIMER = "calendar_validation"` constant and a `wf_mode` column in every output row so downstream consumers and UI displays can identify the validation mode, (3) added a detailed WARNING in the function docstring explaining the pre-optimization-then-validate pattern and pointing to `research/walk_forward.py` (WalkForwardHarness) for true walk-forward.

**R-002 (P1) — Minimum Observations Documentation:** The `ValidationConfig.min_obs` field already correctly defaulted to `ValidationThresholds.MIN_OBS = 252` in `core/contracts.py`. The gap was the absence of documented rationale. Added the ADR-007 citation explaining that 252 trading days (1 full trading year) is the minimum for reliable AR(1) half-life estimation, ADF tests, and Johansen cointegration. Callers who previously relied on the lack of documentation to pass `min_obs=60` are now on notice that this violates the documented policy.

**R-003 (P1) — Execution Timing Semantics:** The `_bt_simulate_equity` method implicitly implements one-bar execution lag via the `w[i-1] * ret[i]` pattern (position set at close[t] earns returns from t to t+1), but this was entirely undocumented. Added: (1) `execution_lag_bars: int = 1` to `BacktestConfig` with explanatory comment, (2) a documentation block in the `_bt_simulate_equity` docstring explaining the EOD signal → next-bar execution equivalence and confirming no look-ahead bias in the PnL loop, (3) an inline comment at the `w[i-1]*ret[i]` line referencing R-003 and the implicit lag.

**R-004 (P2) — Survivorship Bias Documentation:** The `EligibilityFilter` already had `min_history_days`, `min_dollar_volume`, and `min_price` checks, but none were labeled as survivorship-bias mitigations. Added: (1) a class-level docstring explaining all four filters as survivorship-bias mitigations with specific mechanisms, (2) per-check inline comments explaining how each filter addresses a specific survivorship-bias vector, (3) comments in `discovery_contracts.py` `UniverseDefinition` eligibility threshold fields. The default `min_history_days=504` (2 years) already exceeds the 252-day minimum; this was documented and the R-004 floor comment was added.

### Residual Risk

The most significant residual risk is in **R-001**: the optimization tab still does not offer true walk-forward optimization (IS re-estimation + OOS holdout per fold). The calendar validation mode is now correctly labeled, but users who interpret `wf_mode = "calendar_validation"` results as out-of-sample performance estimates are still at risk of overfitting bias. The correct mitigation is to route optimization workflows through `research/walk_forward.py` (WalkForwardHarness), which implements proper purge + embargo + per-fold IS estimation.

The second significant residual risk is in **R-004**: survivorship bias in historical price feeds is not eliminated by these eligibility filters. Any stock present in the price database because it survived long enough to be included will pass through the filters if it currently meets the thresholds. True resolution requires a point-in-time database (CRSP-style) with delisted security price histories. This is a known infrastructure gap.

---

## Files Changed

- `root/optimization_tab.py` — R-001 (docstring, min_seg_days, wf_mode column)
- `research/pair_validator.py` — R-002 (ADR-007 comment on min_obs)
- `core/contracts.py` — R-002 (R-002/ADR-007 tag on MIN_OBS constant)
- `core/optimization_backtester.py` — R-003 (execution_lag_bars field, docstring, inline comment)
- `research/universe.py` — R-004 (EligibilityFilter docstring, per-check comments)
- `research/discovery_contracts.py` — R-004 (UniverseDefinition field comments)
