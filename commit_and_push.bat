@echo off
echo ============================================================
echo  Pairs Trading System — Commit All Changes and Push
echo ============================================================
echo.

:: Remove stale lock
if exist ".git\index.lock" (
    del /F /Q ".git\index.lock"
    echo Removed stale git lock.
)

:: Stage Phase 1-2: Data + ML Platform
git add common/data_service.py common/fmp_client.py common/market_data_router.py ^
        common/config.json common/data_loader.py common/data_providers.py ^
        scripts/ingest_prices_fmp.py scripts/backfill_backtest_metrics_from_trials.py ^
        core/ ml/ portfolio/ research/ params.py ranges.py meta_optimizer.py ^
        conftest.py pyproject.toml

:: Stage Phase 3: Agent Architecture
git add agents/ orchestration/ approvals/ governance/ incidents/ agent_artifacts/

:: Stage Phase 4: Production Operations Layer
git add runtime/ control_plane/ monitoring/ alerts/ reconciliation/ deployment/ secrets/

:: Stage Tests
git add tests/

:: Stage Docs + CLAUDE.md
git add docs/ CLAUDE.md

echo Staged all files. Committing...
echo.

git commit -m "Add production operations layer, agent architecture, ML platform, FMP data layer

=== Production Operations Layer (Phase 4) ===
runtime/: RuntimeState, RuntimeStateManager, EnvironmentSpec (6 envs),
  ActivationRequest/Decision, RuntimeOverride with expiry, LiveTradingReadinessReport
control_plane/: ControlPlaneEngine (18 action types), KillSwitchState, ThrottleState,
  OperatorActionRecord, preflight checks, permission-gated operator actions
monitoring/: SystemHealthMonitor (10 checks), ServiceStatusSummary, HeartbeatRecord,
  BrokerConnectionStatus, MarketDataFeedStatus, EndOfDayReport
alerts/: AlertEngine with 20 default rules, dedup+flap+escalation, AlertAcknowledgement,
  AlertFamily/Severity/Status enums, auto-incident creation
reconciliation/: ReconciliationEngine (position/order/fill/leg imbalance/hedge ratio),
  ReconcileDiffRecord (10 diff types), EOD report generation
deployment/: DeploymentEngine (11-stage lifecycle, deployed!=activated invariant),
  RolloutPlan (canary/staged/blue-green), RollbackDecision, freeze windows
secrets/: SecretReference (never stores values), SecretLoader (env_var/config_file)
113 tests in tests/test_production_ops.py, all passing

=== Agent Architecture (Phase 3) ===
37 agents across 5 classes (research x8, ML x7, monitoring x7, governance x5,
  signal x4, portfolio x6); orchestration/approvals/governance/incidents packages;
WorkflowEngine; GovernancePolicyEngine; ApprovalEngine; IncidentManager;
95 tests in tests/test_agent_architecture.py; docs/agent_architecture.md (933 lines)

=== ML Platform (Phase 2) ===
ml/: features (61 defs), labels (26 defs), datasets, models, evaluation,
  registry, inference, monitoring, governance, explainability;
115 tests in tests/test_ml_platform.py

=== Data Infrastructure (Phase 1) ===
FMP as canonical provider (priority 20); SQL-first DataService;
daily price ingestion script; data_freshness observability tables

=== Documentation ===
docs/production_architecture.md, docs/agent_architecture.md, docs/ml_architecture.md
docs/signal_architecture.md, docs/portfolio_architecture.md, docs/architecture.md
CLAUDE.md: full how-to guides for all platform layers

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Commit failed.
    echo Please close VS Code Source Control or commit from VS Code Source Control panel.
    echo Then run: git push origin main
    pause
    exit /b 1
)

echo.
echo SUCCESS: Commit created!
echo.
echo Pushing to GitHub...
git push origin main

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Push failed. Run manually: git push origin main
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  ALL DONE! Code committed and pushed to GitHub.
echo  https://github.com/benda931/pairs_trading_system
echo ============================================================
pause
