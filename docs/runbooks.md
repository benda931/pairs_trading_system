# Operational Runbooks — Pairs Trading System

> **Integration Status: SCAFFOLD**
> These runbooks describe operational procedures for a future runtime environment.
> As of 2026-03-31, no live or paper trading system exists. The control-plane and
> alert infrastructure is implemented and tested but not wired to operational paths.
> These runbooks are provided for architectural completeness and future readiness.
> See: `INTEGRATION_STATUS.md`

---

## RB-001: Stale Market Data

**When:** Data feed returns data older than the staleness threshold (default: 5 trading days)

**Detection:**
- `SurveillanceEngine.detect("SURV-DI-001", ...)` fires (when wired)
- `get_price_metadata(df)` shows `days_since_last` > threshold
- Dashboard data freshness indicators show stale

**Response:**
1. Check data provider status (FMP API, Yahoo Finance)
2. Verify API key validity: `common/fmp_client.py` uses `FMP_API_KEY`
3. Check network connectivity
4. If FMP down, system should fall back to Yahoo Finance (priority routing in `common/data_providers.py`)
5. Do NOT run optimization or backtest on stale data — results will be misleading

**Control-plane action (when wired):**
```python
control_plane.set_throttle(ThrottleLevel.REDUCED, reason="stale_data", actor="ops")
```

**Recovery:** Resume normal operations when fresh data is confirmed.

---

## RB-002: Broker Disconnect

**When:** IBKR connection drops or `ib_insync` is unavailable.

**Detection:**
- Dashboard shows `broker=False` in services status
- Log message: `ib_insync is not available — IBKRProvider will be disabled`

**Response:**
1. This is normal in research/backtest mode — IBKR is optional
2. If paper/live trading is intended, verify TWS/Gateway is running
3. Check `root/ibkr_connection.py` for connection parameters
4. Verify firewall allows connection to IBKR port

**Impact:** Research and backtesting continue normally. Only live data feed and order routing are affected.

---

## RB-003: Emergency Halt

**When:** Catastrophic condition requires immediate cessation of all activity.

**Control-plane actions:**
```python
# Activate kill switch (when wired)
control_plane.engage_kill_switch(reason="emergency", actor="ops")

# Set exits-only mode
control_plane.set_exits_only(reason="emergency_halt", actor="ops")

# Verify state
state_mgr = get_runtime_state_manager()
safe, reasons = state_mgr.is_safe_to_trade()
assert safe is False
```

**Current limitation:** `is_safe_to_trade()` is defined but not called from execution paths (P1-SAFE). In research mode, stopping the Streamlit process is sufficient.

**Recovery:** Requires explicit acknowledgment:
```python
control_plane.reset_kill_switch(actor="ops", approval_id="...")
```

---

## RB-004: Reconciliation Mismatch

**When:** Internal position state differs from broker/external state.

**Detection (when wired):**
```python
report = reconciliation_engine.reconcile()
if not report.is_clean():
    alert_engine.fire("RECONCILIATION_BREAK", ...)
```

**Response:**
1. Identify which positions differ (count, size, direction)
2. Check if fills were missed or double-counted
3. Check SqlStore for data integrity: `test_sql_store.py` (note: `_ensure_writable` issue exists)
4. Manually reconcile against broker statement

**Current limitation:** ReconciliationEngine exists and passes 54 tests but is not connected to any live data source.

---

## RB-005: Model Stale or Disabled

**When:** ML model exceeds freshness TTL or health check fails.

**Detection (when wired):**
```python
result = model_scorer.score(request)
if result.fallback_triggered:
    # Model unavailable — neutral probability (0.5) returned
    ...
```

**Current reality:** As of 2026-03-31, zero models are trained. `ModelScorer` always returns
neutral probability (0.5) with `fallback_triggered=True`. This is by design — ML is an overlay,
not the foundation. All decisions fall back to parametric rules.

**Response if models existed:**
1. Check `ModelHealthMonitor` for drift/staleness
2. Check feature availability via `PointInTimeFeatureBuilder`
3. If model is stale, it auto-demotes; inference falls back to next tier
4. Do NOT manually override model health — let the governance pipeline handle promotion

---

## RB-006: Walk-Forward Validation Concerns

**When:** Backtest results seem too good — possible overfitting.

**Detection:**
- Sharpe ratio > 3.0 on short history
- Walk-forward segments show inconsistent performance across windows
- Calendar-segment "walk-forward" in optimization tab shows < 63 days per segment

**Response:**
1. Remember: optimization tab uses **calendar-segment stability checks**, NOT true walk-forward
2. For rigorous evaluation, use `research/walk_forward.py:WalkForwardHarness` directly
3. Check ADR-007 for documented backtest limitations
4. Same-close execution bias makes all results structurally optimistic
5. A minimum of 63 trading days per segment is enforced (R-001)

---

## RB-007: Dashboard Not Loading

**When:** `streamlit run root/dashboard.py` starts but pages don't render.

**Common causes:**
1. **Numpy import failure:** Check if a `secrets/` directory exists in the project root.
   If so, rename to `secrets_mgmt/` — it shadows Python's stdlib `secrets` module,
   breaking numpy's `bit_generator.pyx` initialization. (Fixed: MIG-006)
2. **Missing dependencies:** Run `pip install -r requirements.txt`
3. **Port conflict:** Default port 8501 may be in use. Use `streamlit run root/dashboard.py --server.port 8502`

**Verification:**
```bash
.venv/Scripts/python -c "import numpy; import numpy.random; import pandas; print('OK')"
```

---

## RB-008: Test Failures After Code Change

**When:** Test suite shows new failures after modifications.

**Response:**
1. Run full suite: `python -m pytest tests/ -v`
2. Compare against known baseline: 765 passed, 17 failed (pre-existing)
3. If new failures appear, they are regressions — fix before committing
4. Check which layer was affected using the test-to-layer mapping in `docs/testing.md`
5. The 17 pre-existing failures are tracked in `docs/remediation/remediation_ledger.md`

**Pre-existing failures (do NOT count as regressions):**
- 7x P0-KS kill-switch bridge
- 2x P1-GOV governance gate
- 3x P1-SURV2 surveillance hook
- 3x bug fixes (signal generator, risk breaches)
- 2x SqlStore._ensure_writable

---

## RB-009: Controlled Restart

**When:** System needs a clean restart after configuration change or incident.

**Steps:**
1. Stop streamlit process (Ctrl+C or kill process)
2. Clear Python cache if needed: `find . -type d -name __pycache__ -exec rm -r {} +`
3. Verify environment: `.venv/Scripts/python -c "import numpy; print('OK')"`
4. Restart: `streamlit run root/dashboard.py`
5. Verify: Check dashboard shows all 15 tabs, services status panel shows expected state

---

## Cross-Reference

| Scenario | Related Docs |
|----------|-------------|
| All runbooks | `docs/production_architecture.md` (runtime model) |
| Kill-switch | `docs/portfolio_architecture.md` (DrawdownManager, KillSwitchManager) |
| Data issues | `docs/discovery_methodology.md` (EligibilityFilter thresholds) |
| ML issues | `docs/ml_architecture.md` (inference fallback) |
| Backtest concerns | `docs/adr/ADR-007-backtest-realism-limitations.md` |
| Test failures | `docs/testing.md`, `docs/remediation/remediation_ledger.md` |
| Dashboard fix | `docs/migration/migration_ledger.md` (MIG-006: secrets/ rename) |
