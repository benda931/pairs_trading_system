# ADR-007: Backtest Realism Limitations and Documented Assumptions

**Status:** Accepted
**Date:** 2026-03-31
**Deciders:** Principal Quant Review
**Findings Addressed:** P0-EXEC, P0-WF, P2-COSTS

## Documented Backtest Assumptions

These limitations are **known and accepted** for the current research phase. They must be disclosed in all performance reporting.

### Execution Timing
- Signals are generated using close prices at bar t
- Position changes take effect at bar t (via lagged-weight PnL, position earns returns from t to t+1)
- **Equivalent to:** End-of-day signal → next-bar execution
- **Known bias:** Aggressive for liquid pairs; acceptable for daily data
- **Required disclosure:** All backtest Sharpe ratios are based on EOD-signal/EOD-execution assumption

### Walk-Forward Validation
- `root/optimization_tab.py:_run_walkforward_for_params` is **calendar stability validation**, not true walk-forward optimization
- Parameters are optimized on the FULL sample period before this function is called
- Walk-forward segments test whether those parameters are temporally stable
- **This does NOT constitute out-of-sample validation**
- **True walk-forward:** Use `research/walk_forward.py:WalkForwardHarness` which implements purged K-fold with embargo
- **Required disclosure:** Label all results from optimization_tab walk-forward as "in-sample temporal stability" not "out-of-sample performance"

### Cost Model
- Flat commission + slippage (default: 1bps commission + 2bps slippage = 3bps total)
- No volume-based market impact modeling
- No bid-ask spread modeling
- No partial fill simulation
- **Applies to:** Daily data, liquid large-cap pairs
- **Known bias:** Understates costs for less-liquid pairs or large positions
- **Haircut guidance:** Apply 30% Sharpe haircut for live deployment planning

### Survivorship Bias
- `research/universe.py:EligibilityFilter` applies min_price (default $5) and min_dollar_volume filters
- These reduce (but do not eliminate) survivorship bias
- No explicit delisting/bankruptcy screening
- **Required action:** Use universe snapshots from research date, not current date

## Consequence
Any report citing "walk-forward validated" or "institutional-grade backtesting" must reference this ADR and note the specific limitations.
