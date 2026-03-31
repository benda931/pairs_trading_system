# -*- coding: utf-8 -*-
"""
monitoring/contracts.py — Monitoring Domain Contracts
======================================================

All typed domain objects for the monitoring subsystem.

Design principles:
  - Frozen dataclasses for immutable value objects.
  - Enums inherit from str for JSON round-trip compatibility.
  - stdlib only — no external dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS
# ══════════════════════════════════════════════════════════════════


class CheckSeverity(str, enum.Enum):
    """Result severity of a single health check."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class FeedStatus(str, enum.Enum):
    """Operational status of a market-data feed."""

    LIVE = "live"
    STALE = "stale"
    MISSING = "missing"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


# ══════════════════════════════════════════════════════════════════
# 2. HEALTH-CHECK VALUE OBJECTS
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HealthStatus:
    """Result of a single component health check.

    Fields
    ------
    component : str
        Logical name of the component being checked.
    checked_at : str
        ISO-8601 timestamp when the check ran.
    severity : CheckSeverity
        Outcome severity.
    message : str
        Human-readable description of the check result.
    details : dict
        Component-specific key/value diagnostic data.
    latency_ms : Optional[float]
        Round-trip latency of the check, if measurable.
    remediation : str
        Suggested remediation action if severity != OK.
    """

    component: str
    checked_at: str
    severity: CheckSeverity
    message: str
    details: dict
    latency_ms: Optional[float]
    remediation: str = ""


@dataclass(frozen=True)
class DependencyHealthStatus:
    """Health status of an external or internal dependency.

    Fields
    ------
    dependency_name : str
        Logical name of the dependency.
    dep_type : str
        Category: "broker" | "data_feed" | "sql_store" |
        "ml_model" | "agent" | "external_api".
    healthy : bool
        True if the dependency is reachable and responding.
    severity : CheckSeverity
        Derived severity based on health state.
    last_ok_at : Optional[str]
        ISO-8601 timestamp of most recent successful check.
    error : Optional[str]
        Error message if unhealthy.
    latency_ms : Optional[float]
        Measured response latency.
    """

    dependency_name: str
    dep_type: str
    healthy: bool
    severity: CheckSeverity
    last_ok_at: Optional[str]
    error: Optional[str]
    latency_ms: Optional[float]


@dataclass(frozen=True)
class BrokerConnectionStatus:
    """Snapshot of broker connectivity and session state.

    Fields
    ------
    broker_name : str
        Name of the broker (e.g. "IBKR").
    env : str
        Deployment environment: "paper" | "live".
    connected : bool
        TCP / WebSocket connection established.
    authenticated : bool
        Session-level authentication passed.
    account_valid : bool
        Target account is accessible and not restricted.
    session_type : str
        "paper" | "live".
    last_heartbeat : Optional[str]
        ISO-8601 timestamp of most recent broker heartbeat.
    pending_orders : int
        Number of orders awaiting broker acknowledgment.
    open_positions : int
        Number of open positions reported by the broker.
    error : Optional[str]
        Most recent connection error, if any.
    checked_at : str
        ISO-8601 timestamp when this snapshot was captured.
    """

    broker_name: str
    env: str
    connected: bool
    authenticated: bool
    account_valid: bool
    session_type: str
    last_heartbeat: Optional[str]
    pending_orders: int
    open_positions: int
    error: Optional[str]
    checked_at: str


@dataclass(frozen=True)
class MarketDataFeedStatus:
    """Snapshot of market-data feed health.

    Fields
    ------
    feed_name : str
        Name of the feed (e.g. "FMP", "IBKR", "YahooFinance").
    env : str
        Deployment environment.
    status : FeedStatus
        Operational status.
    symbols_tracked : int
        Total symbols being consumed from this feed.
    symbols_stale : int
        Symbols whose last tick is older than the staleness threshold.
    last_update : Optional[str]
        ISO-8601 timestamp of the most recent tick received.
    oldest_stale_symbol : Optional[str]
        Symbol identifier with the most outdated data.
    max_staleness_seconds : float
        Staleness of the oldest symbol in seconds.
    session_active : bool
        Whether the feed session is open.
    checked_at : str
        ISO-8601 timestamp when this snapshot was captured.
    """

    feed_name: str
    env: str
    status: FeedStatus
    symbols_tracked: int
    symbols_stale: int
    last_update: Optional[str]
    oldest_stale_symbol: Optional[str]
    max_staleness_seconds: float
    session_active: bool
    checked_at: str


@dataclass(frozen=True)
class OrderRouterStatus:
    """Snapshot of order-router health.

    Fields
    ------
    router_name : str
        Name of the order router.
    env : str
        Deployment environment.
    healthy : bool
        Router is operational and accepting new orders.
    pending_acks : int
        Orders awaiting broker acknowledgment.
    recent_rejects : int
        Number of order rejects in the last 5 minutes.
    reject_rate : float
        Reject fraction [0, 1] over the recent window.
    rate_limit_ok : bool
        False if the router is approaching or at a rate limit.
    checked_at : str
        ISO-8601 timestamp when this snapshot was captured.
    """

    router_name: str
    env: str
    healthy: bool
    pending_acks: int
    recent_rejects: int
    reject_rate: float
    rate_limit_ok: bool
    checked_at: str


# ══════════════════════════════════════════════════════════════════
# 3. AGGREGATE SUMMARY
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ServiceStatusSummary:
    """Aggregate health summary for the entire platform.

    Fields
    ------
    summary_id : str
        UUID for this summary snapshot.
    captured_at : str
        ISO-8601 timestamp when the summary was produced.
    env : str
        Deployment environment.
    overall_severity : CheckSeverity
        Worst severity across all components.
    components : tuple[HealthStatus, ...]
        Per-component health check results.
    dependencies : tuple[DependencyHealthStatus, ...]
        Per-dependency health statuses.
    broker : Optional[BrokerConnectionStatus]
        Broker connection snapshot, or None if not configured.
    market_data : tuple[MarketDataFeedStatus, ...]
        Market data feed snapshots.
    order_router : Optional[OrderRouterStatus]
        Order-router snapshot, or None if not configured.
    unhealthy_components : tuple[str, ...]
        Names of components with severity != OK.
    recommendations : tuple[str, ...]
        Operator recommendations derived from check results.
    safe_to_trade : bool
        True only when no CRITICAL or EMERGENCY checks are firing
        and the kill switch is inactive.
    """

    summary_id: str
    captured_at: str
    env: str
    overall_severity: CheckSeverity
    components: Tuple[HealthStatus, ...]
    dependencies: Tuple[DependencyHealthStatus, ...]
    broker: Optional[BrokerConnectionStatus]
    market_data: Tuple[MarketDataFeedStatus, ...]
    order_router: Optional[OrderRouterStatus]
    unhealthy_components: Tuple[str, ...]
    recommendations: Tuple[str, ...]
    safe_to_trade: bool


# ══════════════════════════════════════════════════════════════════
# 4. HEARTBEAT AND REPORTING RECORDS
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HeartbeatRecord:
    """A single heartbeat emitted by a system component.

    Fields
    ------
    component : str
        Logical name of the emitting component.
    env : str
        Deployment environment.
    heartbeat_at : str
        ISO-8601 timestamp when the heartbeat was emitted.
    sequence : int
        Monotonically increasing counter for gap detection.
    state : str
        ServiceState.value of the emitting component.
    metadata : dict
        Optional component-specific key/value annotations.
    """

    component: str
    env: str
    heartbeat_at: str
    sequence: int
    state: str
    metadata: dict


@dataclass(frozen=True)
class EndOfDayReport:
    """End-of-day operational summary.

    Captures trading activity, PnL, reconciliation outcome,
    and operational metadata for the trading session.

    Fields
    ------
    report_id : str
        UUID for this report.
    date : str
        Trading date in YYYY-MM-DD format.
    env : str
        Deployment environment.
    generated_at : str
        ISO-8601 timestamp when the report was produced.
    total_trades : int
        Number of completed round-trip trades.
    open_positions : int
        Number of open positions at end of day.
    gross_pnl : float
        Gross realised + unrealised PnL (USD).
    net_pnl : float
        Net PnL after commissions and fees.
    gross_exposure : float
        Sum of absolute position notional values.
    net_exposure : float
        Net directional notional exposure.
    max_drawdown_pct : float
        Maximum intra-day drawdown as a fraction [0, 1].
    signals_generated : int
        Total signals produced by the signal pipeline.
    orders_submitted : int
        Total orders sent to the broker.
    orders_filled : int
        Total orders confirmed filled.
    orders_rejected : int
        Total orders rejected by the broker or router.
    reconciliation_clean : bool
        True if end-of-day reconciliation found no critical diffs.
    reconciliation_issues : tuple[str, ...]
        Short descriptions of any unresolved reconciliation issues.
    incidents_today : int
        Number of incidents opened during this session.
    alerts_today : int
        Total alert events fired during this session.
    model_versions_in_use : dict
        Mapping of model_name -> version string for audit purposes.
    config_version : str
        Active configuration version at end of day.
    notes : str
        Free-text operator notes.
    """

    report_id: str
    date: str
    env: str
    generated_at: str
    total_trades: int
    open_positions: int
    gross_pnl: float
    net_pnl: float
    gross_exposure: float
    net_exposure: float
    max_drawdown_pct: float
    signals_generated: int
    orders_submitted: int
    orders_filled: int
    orders_rejected: int
    reconciliation_clean: bool
    reconciliation_issues: Tuple[str, ...]
    incidents_today: int
    alerts_today: int
    model_versions_in_use: Dict[str, str]
    config_version: str
    notes: str = ""
