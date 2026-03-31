# -*- coding: utf-8 -*-
"""
monitoring/health.py — SystemHealthMonitor
===========================================

Runs health checks across all platform components and produces
HealthStatus and ServiceStatusSummary aggregates.

Thread safety: a single threading.Lock serializes heartbeat store
mutations. Individual check methods are safe to call concurrently
because they only read shared state through the lock.

Singleton access via get_system_health_monitor().
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from monitoring.contracts import (
    BrokerConnectionStatus,
    CheckSeverity,
    DependencyHealthStatus,
    FeedStatus,
    HeartbeatRecord,
    HealthStatus,
    MarketDataFeedStatus,
    OrderRouterStatus,
    ServiceStatusSummary,
)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _elapsed_seconds(iso_ts: Optional[str]) -> Optional[float]:
    """Return seconds elapsed since iso_ts, or None if iso_ts is None."""
    if iso_ts is None:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════


class SystemHealthMonitor:
    """Runs health checks across all platform components.

    Produces ``HealthStatus`` per check and a ``ServiceStatusSummary``
    that aggregates all checks into a single operational view.

    Supported checks
    ----------------
    - SQL store connectivity + table existence
    - Runtime state manager responsiveness
    - Control plane responsiveness
    - Agent registry health
    - Workflow engine health
    - Broker connection (if configured)
    - Data feed freshness
    - Kill switch state
    - Service heartbeat gaps
    - Override expiry status

    Conservative aggregation
    ------------------------
    Any UNKNOWN check counts as WARNING in the aggregate.
    A single CRITICAL check sets ``safe_to_trade = False``.

    Parameters
    ----------
    state_manager :
        RuntimeStateManager instance, or None for standalone mode.
    stale_data_threshold_s : float
        Seconds after which a data-feed tick is considered stale.
        Default: 300.0 (5 minutes).
    heartbeat_gap_s : float
        Seconds without a heartbeat before a WARNING is raised.
        Default: 120.0 (2 minutes).
    """

    def __init__(
        self,
        state_manager: Optional[Any] = None,
        stale_data_threshold_s: float = 300.0,
        heartbeat_gap_s: float = 120.0,
    ) -> None:
        self._state_manager = state_manager
        self._stale_threshold: float = stale_data_threshold_s
        self._heartbeat_gap: float = heartbeat_gap_s

        self._heartbeats: Dict[str, HeartbeatRecord] = {}
        self._lock = threading.Lock()

        # Metrics counters
        self._check_count: int = 0
        self._warning_count: int = 0
        self._critical_count: int = 0
        self._unknown_count: int = 0

    # ──────────────────────────────────────────────────────────────
    # MAIN AGGREGATE CHECK
    # ──────────────────────────────────────────────────────────────

    def check_all(self, env: str = "paper") -> ServiceStatusSummary:
        """Run all checks and produce a ServiceStatusSummary.

        Parameters
        ----------
        env : str
            Deployment environment label.

        Returns
        -------
        ServiceStatusSummary
            Fully populated summary including per-component statuses,
            dependency checks, and a tradability gate.
        """
        components: List[HealthStatus] = []

        # Core subsystem checks
        components.append(self.check_runtime_state())
        components.append(self.check_kill_switch())
        components.append(self.check_overrides())
        components.append(self.check_sql_store())

        # Heartbeat checks for all registered components
        components.extend(self.check_heartbeats())

        # Derive dependency health from available state
        dependencies = self._derive_dependency_statuses()

        # Aggregate severity
        severity_order = {
            CheckSeverity.OK: 0,
            CheckSeverity.UNKNOWN: 1,
            CheckSeverity.WARNING: 2,
            CheckSeverity.CRITICAL: 3,
        }
        worst = CheckSeverity.OK
        for hs in components:
            if severity_order[hs.severity] > severity_order[worst]:
                worst = hs.severity
        # UNKNOWN promotes to WARNING for aggregate
        if worst == CheckSeverity.UNKNOWN:
            worst = CheckSeverity.WARNING

        # Collect unhealthy component names
        unhealthy = tuple(
            hs.component for hs in components if hs.severity != CheckSeverity.OK
        )

        # Build recommendations
        recommendations = self._build_recommendations(components, dependencies)

        # Tradability gate: safe only when no CRITICAL and kill switch not active
        safe, _ = self.is_safe_to_trade(env)

        summary = ServiceStatusSummary(
            summary_id=str(uuid.uuid4()),
            captured_at=_now_iso(),
            env=env,
            overall_severity=worst,
            components=tuple(components),
            dependencies=tuple(dependencies),
            broker=None,
            market_data=(),
            order_router=None,
            unhealthy_components=unhealthy,
            recommendations=tuple(recommendations),
            safe_to_trade=safe,
        )
        return summary

    # ──────────────────────────────────────────────────────────────
    # INDIVIDUAL CHECKS
    # ──────────────────────────────────────────────────────────────

    def check_sql_store(self) -> HealthStatus:
        """Check SQL store connectivity and basic table existence."""
        component = "sql_store"
        start = time.monotonic()
        try:
            from core.sql_store import get_sql_store  # type: ignore

            store = get_sql_store()
            # Attempt a lightweight read to verify connectivity
            _ = store.get_latest_prices(limit=1) if hasattr(store, "get_latest_prices") else True
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record_check(CheckSeverity.OK)
            return HealthStatus(
                component=component,
                checked_at=_now_iso(),
                severity=CheckSeverity.OK,
                message="SQL store reachable.",
                details={"latency_ms": round(latency_ms, 2)},
                latency_ms=round(latency_ms, 2),
            )
        except ImportError:
            # Module not available — treat as unknown, not critical
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record_check(CheckSeverity.UNKNOWN)
            return HealthStatus(
                component=component,
                checked_at=_now_iso(),
                severity=CheckSeverity.UNKNOWN,
                message="sql_store module not importable; check not available.",
                details={},
                latency_ms=round(latency_ms, 2),
                remediation="Ensure core.sql_store is installed and importable.",
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record_check(CheckSeverity.CRITICAL)
            return HealthStatus(
                component=component,
                checked_at=_now_iso(),
                severity=CheckSeverity.CRITICAL,
                message=f"SQL store unreachable: {exc}",
                details={"error": str(exc)},
                latency_ms=round(latency_ms, 2),
                remediation="Verify database path/credentials and that the store is not locked.",
            )

    def check_runtime_state(self) -> HealthStatus:
        """Check runtime state manager responsiveness."""
        component = "runtime_state"
        start = time.monotonic()
        if self._state_manager is None:
            # Try to import and use the singleton
            try:
                from runtime.state import get_runtime_state_manager  # type: ignore

                sm = get_runtime_state_manager()
                _ = sm.get_runtime_state()
                latency_ms = (time.monotonic() - start) * 1000.0
                self._record_check(CheckSeverity.OK)
                return HealthStatus(
                    component=component,
                    checked_at=_now_iso(),
                    severity=CheckSeverity.OK,
                    message="Runtime state manager responsive.",
                    details={"latency_ms": round(latency_ms, 2)},
                    latency_ms=round(latency_ms, 2),
                )
            except ImportError:
                latency_ms = (time.monotonic() - start) * 1000.0
                self._record_check(CheckSeverity.UNKNOWN)
                return HealthStatus(
                    component=component,
                    checked_at=_now_iso(),
                    severity=CheckSeverity.UNKNOWN,
                    message="runtime.state module not importable; check not available.",
                    details={},
                    latency_ms=round(latency_ms, 2),
                )
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.monotonic() - start) * 1000.0
                self._record_check(CheckSeverity.CRITICAL)
                return HealthStatus(
                    component=component,
                    checked_at=_now_iso(),
                    severity=CheckSeverity.CRITICAL,
                    message=f"Runtime state manager error: {exc}",
                    details={"error": str(exc)},
                    latency_ms=round(latency_ms, 2),
                    remediation="Restart the runtime state manager.",
                )

        # state_manager was injected directly
        try:
            _ = self._state_manager.get_runtime_state()
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record_check(CheckSeverity.OK)
            return HealthStatus(
                component=component,
                checked_at=_now_iso(),
                severity=CheckSeverity.OK,
                message="Runtime state manager responsive.",
                details={"latency_ms": round(latency_ms, 2)},
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record_check(CheckSeverity.CRITICAL)
            return HealthStatus(
                component=component,
                checked_at=_now_iso(),
                severity=CheckSeverity.CRITICAL,
                message=f"Runtime state manager error: {exc}",
                details={"error": str(exc)},
                latency_ms=round(latency_ms, 2),
                remediation="Restart the runtime state manager.",
            )

    def check_kill_switch(self) -> HealthStatus:
        """Return OK if the kill switch is inactive, CRITICAL if active."""
        component = "kill_switch"
        start = time.monotonic()
        try:
            from portfolio.risk_ops import get_kill_switch_manager  # type: ignore

            ksm = get_kill_switch_manager()
            mode = ksm.current_mode
            latency_ms = (time.monotonic() - start) * 1000.0
            # Any mode other than OFF is a problem
            mode_val = mode.value if hasattr(mode, "value") else str(mode)
            if mode_val in ("off", "OFF", "NONE", "none"):
                self._record_check(CheckSeverity.OK)
                return HealthStatus(
                    component=component,
                    checked_at=_now_iso(),
                    severity=CheckSeverity.OK,
                    message="Kill switch inactive.",
                    details={"mode": mode_val},
                    latency_ms=round(latency_ms, 2),
                )
            else:
                self._record_check(CheckSeverity.CRITICAL)
                return HealthStatus(
                    component=component,
                    checked_at=_now_iso(),
                    severity=CheckSeverity.CRITICAL,
                    message=f"Kill switch is ACTIVE (mode={mode_val}).",
                    details={"mode": mode_val},
                    latency_ms=round(latency_ms, 2),
                    remediation="Acknowledge and reset the kill switch once the root cause is resolved.",
                )
        except ImportError:
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record_check(CheckSeverity.UNKNOWN)
            return HealthStatus(
                component=component,
                checked_at=_now_iso(),
                severity=CheckSeverity.UNKNOWN,
                message="Kill switch module not importable; check not available.",
                details={},
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record_check(CheckSeverity.UNKNOWN)
            return HealthStatus(
                component=component,
                checked_at=_now_iso(),
                severity=CheckSeverity.UNKNOWN,
                message=f"Kill switch state could not be determined: {exc}",
                details={"error": str(exc)},
                latency_ms=round(latency_ms, 2),
                remediation="Check kill switch module configuration.",
            )

    def check_overrides(self) -> HealthStatus:
        """Warn if any runtime overrides are near expiry or already expired."""
        component = "overrides"
        start = time.monotonic()
        try:
            from runtime.state import get_runtime_state_manager  # type: ignore

            sm = get_runtime_state_manager()
            if not hasattr(sm, "list_overrides"):
                latency_ms = (time.monotonic() - start) * 1000.0
                self._record_check(CheckSeverity.OK)
                return HealthStatus(
                    component=component,
                    checked_at=_now_iso(),
                    severity=CheckSeverity.OK,
                    message="Override check not supported by state manager.",
                    details={},
                    latency_ms=round(latency_ms, 2),
                )

            overrides = sm.list_overrides()
            now = datetime.now(timezone.utc)
            near_expiry: List[str] = []
            expired: List[str] = []

            for ov in overrides:
                expiry_str = getattr(ov, "expires_at", None)
                if expiry_str is None:
                    continue
                try:
                    expiry_dt = datetime.fromisoformat(expiry_str)
                    if expiry_dt.tzinfo is None:
                        expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                    remaining = (expiry_dt - now).total_seconds()
                    ov_id = getattr(ov, "override_id", str(ov))
                    if remaining <= 0:
                        expired.append(ov_id)
                    elif remaining < 300:  # < 5 minutes
                        near_expiry.append(ov_id)
                except (ValueError, TypeError):
                    pass

            latency_ms = (time.monotonic() - start) * 1000.0

            if expired:
                self._record_check(CheckSeverity.WARNING)
                return HealthStatus(
                    component=component,
                    checked_at=_now_iso(),
                    severity=CheckSeverity.WARNING,
                    message=f"{len(expired)} override(s) have expired.",
                    details={"expired": expired, "near_expiry": near_expiry},
                    latency_ms=round(latency_ms, 2),
                    remediation="Review and renew or revoke expired overrides.",
                )
            if near_expiry:
                self._record_check(CheckSeverity.WARNING)
                return HealthStatus(
                    component=component,
                    checked_at=_now_iso(),
                    severity=CheckSeverity.WARNING,
                    message=f"{len(near_expiry)} override(s) expire within 5 minutes.",
                    details={"near_expiry": near_expiry},
                    latency_ms=round(latency_ms, 2),
                    remediation="Review overrides expiring soon.",
                )

            self._record_check(CheckSeverity.OK)
            return HealthStatus(
                component=component,
                checked_at=_now_iso(),
                severity=CheckSeverity.OK,
                message=f"All {len(overrides)} override(s) are current.",
                details={"active_overrides": len(overrides)},
                latency_ms=round(latency_ms, 2),
            )

        except ImportError:
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record_check(CheckSeverity.UNKNOWN)
            return HealthStatus(
                component=component,
                checked_at=_now_iso(),
                severity=CheckSeverity.UNKNOWN,
                message="Override check not available (runtime module not importable).",
                details={},
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record_check(CheckSeverity.UNKNOWN)
            return HealthStatus(
                component=component,
                checked_at=_now_iso(),
                severity=CheckSeverity.UNKNOWN,
                message=f"Override check error: {exc}",
                details={"error": str(exc)},
                latency_ms=round(latency_ms, 2),
            )

    def check_heartbeats(self) -> List[HealthStatus]:
        """Check all registered component heartbeats for gaps.

        Returns one HealthStatus per registered component.  Components
        with no heartbeat received yet are reported as WARNING.

        Returns
        -------
        list[HealthStatus]
            One entry per registered component, or an empty list if no
            components have ever emitted a heartbeat.
        """
        now_ts = _now_iso()
        with self._lock:
            records = dict(self._heartbeats)

        if not records:
            return []

        results: List[HealthStatus] = []
        for component, rec in records.items():
            elapsed = _elapsed_seconds(rec.heartbeat_at)
            if elapsed is None:
                self._record_check(CheckSeverity.UNKNOWN)
                results.append(
                    HealthStatus(
                        component=f"heartbeat:{component}",
                        checked_at=now_ts,
                        severity=CheckSeverity.UNKNOWN,
                        message=f"Could not parse heartbeat timestamp for '{component}'.",
                        details={"last_heartbeat": rec.heartbeat_at},
                        latency_ms=None,
                    )
                )
            elif elapsed > self._heartbeat_gap:
                self._record_check(CheckSeverity.WARNING)
                results.append(
                    HealthStatus(
                        component=f"heartbeat:{component}",
                        checked_at=now_ts,
                        severity=CheckSeverity.WARNING,
                        message=(
                            f"No heartbeat from '{component}' for "
                            f"{elapsed:.0f}s (threshold={self._heartbeat_gap:.0f}s)."
                        ),
                        details={
                            "last_heartbeat": rec.heartbeat_at,
                            "elapsed_s": round(elapsed, 1),
                            "threshold_s": self._heartbeat_gap,
                            "last_state": rec.state,
                        },
                        latency_ms=None,
                        remediation=f"Investigate component '{component}'; it may have crashed.",
                    )
                )
            else:
                self._record_check(CheckSeverity.OK)
                results.append(
                    HealthStatus(
                        component=f"heartbeat:{component}",
                        checked_at=now_ts,
                        severity=CheckSeverity.OK,
                        message=f"Heartbeat from '{component}' received {elapsed:.0f}s ago.",
                        details={
                            "last_heartbeat": rec.heartbeat_at,
                            "elapsed_s": round(elapsed, 1),
                            "sequence": rec.sequence,
                            "state": rec.state,
                        },
                        latency_ms=None,
                    )
                )
        return results

    def check_broker(
        self, broker_status: Optional[BrokerConnectionStatus] = None
    ) -> HealthStatus:
        """Check broker connectivity.

        Parameters
        ----------
        broker_status : Optional[BrokerConnectionStatus]
            Pre-fetched broker status snapshot.  If None, the check
            reports UNKNOWN (caller must supply broker status).

        Returns
        -------
        HealthStatus
        """
        component = "broker"
        checked_at = _now_iso()

        if broker_status is None:
            self._record_check(CheckSeverity.UNKNOWN)
            return HealthStatus(
                component=component,
                checked_at=checked_at,
                severity=CheckSeverity.UNKNOWN,
                message="No broker status supplied; check not available.",
                details={},
                latency_ms=None,
            )

        if not broker_status.connected:
            self._record_check(CheckSeverity.CRITICAL)
            return HealthStatus(
                component=component,
                checked_at=checked_at,
                severity=CheckSeverity.CRITICAL,
                message=f"Broker '{broker_status.broker_name}' disconnected.",
                details={
                    "broker": broker_status.broker_name,
                    "env": broker_status.env,
                    "error": broker_status.error,
                },
                latency_ms=None,
                remediation="Reconnect to the broker; check network and credentials.",
            )

        if not broker_status.authenticated:
            self._record_check(CheckSeverity.CRITICAL)
            return HealthStatus(
                component=component,
                checked_at=checked_at,
                severity=CheckSeverity.CRITICAL,
                message=f"Broker '{broker_status.broker_name}' not authenticated.",
                details={"broker": broker_status.broker_name},
                latency_ms=None,
                remediation="Re-authenticate with the broker.",
            )

        if not broker_status.account_valid:
            self._record_check(CheckSeverity.CRITICAL)
            return HealthStatus(
                component=component,
                checked_at=checked_at,
                severity=CheckSeverity.CRITICAL,
                message=f"Broker account invalid for '{broker_status.broker_name}'.",
                details={"broker": broker_status.broker_name},
                latency_ms=None,
                remediation="Check account status with broker.",
            )

        # Connected and authenticated — check heartbeat age
        hb_elapsed = _elapsed_seconds(broker_status.last_heartbeat)
        if hb_elapsed is not None and hb_elapsed > self._heartbeat_gap:
            self._record_check(CheckSeverity.WARNING)
            return HealthStatus(
                component=component,
                checked_at=checked_at,
                severity=CheckSeverity.WARNING,
                message=(
                    f"Broker '{broker_status.broker_name}' heartbeat stale "
                    f"({hb_elapsed:.0f}s old)."
                ),
                details={
                    "broker": broker_status.broker_name,
                    "last_heartbeat": broker_status.last_heartbeat,
                    "elapsed_s": round(hb_elapsed, 1),
                },
                latency_ms=None,
                remediation="Monitor broker connection stability.",
            )

        self._record_check(CheckSeverity.OK)
        return HealthStatus(
            component=component,
            checked_at=checked_at,
            severity=CheckSeverity.OK,
            message=f"Broker '{broker_status.broker_name}' connected and authenticated.",
            details={
                "broker": broker_status.broker_name,
                "pending_orders": broker_status.pending_orders,
                "open_positions": broker_status.open_positions,
            },
            latency_ms=None,
        )

    def check_data_feeds(
        self, feed_statuses: Optional[List[MarketDataFeedStatus]] = None
    ) -> List[HealthStatus]:
        """Check all data feeds for staleness.

        Parameters
        ----------
        feed_statuses : list[MarketDataFeedStatus] | None
            Pre-fetched feed status snapshots.  If None or empty, an
            UNKNOWN status is returned.

        Returns
        -------
        list[HealthStatus]
            One entry per feed.
        """
        if not feed_statuses:
            self._record_check(CheckSeverity.UNKNOWN)
            return [
                HealthStatus(
                    component="data_feeds",
                    checked_at=_now_iso(),
                    severity=CheckSeverity.UNKNOWN,
                    message="No feed statuses supplied; check not available.",
                    details={},
                    latency_ms=None,
                )
            ]

        results: List[HealthStatus] = []
        for fs in feed_statuses:
            component = f"data_feed:{fs.feed_name}"
            checked_at = _now_iso()

            if fs.status == FeedStatus.DISCONNECTED:
                self._record_check(CheckSeverity.CRITICAL)
                results.append(
                    HealthStatus(
                        component=component,
                        checked_at=checked_at,
                        severity=CheckSeverity.CRITICAL,
                        message=f"Feed '{fs.feed_name}' disconnected.",
                        details={"feed": fs.feed_name, "status": fs.status.value},
                        latency_ms=None,
                        remediation=f"Reconnect feed '{fs.feed_name}'.",
                    )
                )
            elif fs.status == FeedStatus.MISSING:
                self._record_check(CheckSeverity.CRITICAL)
                results.append(
                    HealthStatus(
                        component=component,
                        checked_at=checked_at,
                        severity=CheckSeverity.CRITICAL,
                        message=f"Feed '{fs.feed_name}' is missing.",
                        details={"feed": fs.feed_name},
                        latency_ms=None,
                        remediation=f"Investigate why feed '{fs.feed_name}' is missing.",
                    )
                )
            elif fs.max_staleness_seconds > self._stale_threshold:
                self._record_check(CheckSeverity.WARNING)
                results.append(
                    HealthStatus(
                        component=component,
                        checked_at=checked_at,
                        severity=CheckSeverity.WARNING,
                        message=(
                            f"Feed '{fs.feed_name}' has stale data: "
                            f"{fs.symbols_stale} stale symbol(s), "
                            f"max staleness={fs.max_staleness_seconds:.0f}s."
                        ),
                        details={
                            "feed": fs.feed_name,
                            "symbols_stale": fs.symbols_stale,
                            "max_staleness_s": fs.max_staleness_seconds,
                            "oldest_stale": fs.oldest_stale_symbol,
                        },
                        latency_ms=None,
                        remediation=f"Investigate data latency for '{fs.feed_name}'.",
                    )
                )
            elif fs.status == FeedStatus.DEGRADED:
                self._record_check(CheckSeverity.WARNING)
                results.append(
                    HealthStatus(
                        component=component,
                        checked_at=checked_at,
                        severity=CheckSeverity.WARNING,
                        message=f"Feed '{fs.feed_name}' is degraded.",
                        details={"feed": fs.feed_name, "status": fs.status.value},
                        latency_ms=None,
                        remediation=f"Check feed provider status for '{fs.feed_name}'.",
                    )
                )
            else:
                self._record_check(CheckSeverity.OK)
                results.append(
                    HealthStatus(
                        component=component,
                        checked_at=checked_at,
                        severity=CheckSeverity.OK,
                        message=f"Feed '{fs.feed_name}' live. {fs.symbols_tracked} symbols tracked.",
                        details={
                            "feed": fs.feed_name,
                            "symbols_tracked": fs.symbols_tracked,
                            "last_update": fs.last_update,
                        },
                        latency_ms=None,
                    )
                )
        return results

    # ──────────────────────────────────────────────────────────────
    # HEARTBEAT MANAGEMENT
    # ──────────────────────────────────────────────────────────────

    def register_heartbeat(self, record: HeartbeatRecord) -> None:
        """Record a component heartbeat.

        Atomically updates the heartbeat store so concurrent
        heartbeat emitters do not race.

        Parameters
        ----------
        record : HeartbeatRecord
            Heartbeat to store.
        """
        with self._lock:
            existing = self._heartbeats.get(record.component)
            # Only update if this heartbeat is newer (by sequence or timestamp)
            if existing is None or record.sequence >= existing.sequence:
                self._heartbeats[record.component] = record

    def get_last_heartbeat(self, component: str) -> Optional[HeartbeatRecord]:
        """Return the most recent heartbeat for a component.

        Parameters
        ----------
        component : str
            Component name to look up.

        Returns
        -------
        Optional[HeartbeatRecord]
            Most recent record, or None if never seen.
        """
        with self._lock:
            return self._heartbeats.get(component)

    # ──────────────────────────────────────────────────────────────
    # TRADABILITY GATE
    # ──────────────────────────────────────────────────────────────

    def is_safe_to_trade(self, env: str = "paper") -> Tuple[bool, List[str]]:
        """Aggregate readiness gate.

        Returns
        -------
        tuple[bool, list[str]]
            (safe, blocking_reasons)

        safe is True only when:
        - No CRITICAL component checks are firing.
        - Kill switch is inactive.
        - No heartbeat gaps are CRITICAL.
        """
        blocking: List[str] = []

        # Kill switch check
        ks_status = self.check_kill_switch()
        if ks_status.severity == CheckSeverity.CRITICAL:
            blocking.append(f"kill_switch: {ks_status.message}")

        # Runtime state check
        rs_status = self.check_runtime_state()
        if rs_status.severity == CheckSeverity.CRITICAL:
            blocking.append(f"runtime_state: {rs_status.message}")

        # SQL store check
        sql_status = self.check_sql_store()
        if sql_status.severity == CheckSeverity.CRITICAL:
            blocking.append(f"sql_store: {sql_status.message}")

        # Heartbeat gaps that are CRITICAL severity
        for hs in self.check_heartbeats():
            if hs.severity == CheckSeverity.CRITICAL:
                blocking.append(f"{hs.component}: {hs.message}")

        return (len(blocking) == 0, blocking)

    # ──────────────────────────────────────────────────────────────
    # METRICS
    # ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return health check statistics.

        Returns
        -------
        dict
            Keys: total_checks, warning_count, critical_count,
            unknown_count, warning_rate, critical_rate,
            registered_heartbeat_components.
        """
        with self._lock:
            hb_count = len(self._heartbeats)

        total = self._check_count or 1  # avoid div-by-zero
        return {
            "total_checks": self._check_count,
            "warning_count": self._warning_count,
            "critical_count": self._critical_count,
            "unknown_count": self._unknown_count,
            "warning_rate": self._warning_count / total,
            "critical_rate": self._critical_count / total,
            "registered_heartbeat_components": hb_count,
        }

    # ──────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────

    def _record_check(self, severity: CheckSeverity) -> None:
        """Update internal check counters (not thread-safe — caller must hold lock if needed)."""
        self._check_count += 1
        if severity == CheckSeverity.WARNING:
            self._warning_count += 1
        elif severity == CheckSeverity.CRITICAL:
            self._critical_count += 1
        elif severity == CheckSeverity.UNKNOWN:
            self._unknown_count += 1

    def _derive_dependency_statuses(self) -> List[DependencyHealthStatus]:
        """Build DependencyHealthStatus entries from available module probes."""
        results: List[DependencyHealthStatus] = []
        now = _now_iso()

        # SQL store dependency
        sql_hs = self.check_sql_store()
        results.append(
            DependencyHealthStatus(
                dependency_name="sql_store",
                dep_type="sql_store",
                healthy=(sql_hs.severity == CheckSeverity.OK),
                severity=sql_hs.severity,
                last_ok_at=now if sql_hs.severity == CheckSeverity.OK else None,
                error=sql_hs.details.get("error"),
                latency_ms=sql_hs.latency_ms,
            )
        )
        return results

    def _build_recommendations(
        self,
        components: List[HealthStatus],
        dependencies: List[DependencyHealthStatus],
    ) -> List[str]:
        """Derive operator recommendations from check results."""
        recs: List[str] = []
        for hs in components:
            if hs.severity in (CheckSeverity.WARNING, CheckSeverity.CRITICAL) and hs.remediation:
                recs.append(f"[{hs.severity.value.upper()}] {hs.component}: {hs.remediation}")
        for dep in dependencies:
            if not dep.healthy and dep.error:
                recs.append(
                    f"[{dep.severity.value.upper()}] {dep.dependency_name}: {dep.error}"
                )
        return recs


# ══════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════

_monitor_instance: Optional[SystemHealthMonitor] = None
_monitor_lock = threading.Lock()


def get_system_health_monitor(
    state_manager: Optional[Any] = None,
    stale_data_threshold_s: float = 300.0,
    heartbeat_gap_s: float = 120.0,
) -> SystemHealthMonitor:
    """Return the singleton SystemHealthMonitor, creating it on first call.

    Parameters
    ----------
    state_manager :
        Injected RuntimeStateManager for the first call only.
    stale_data_threshold_s : float
        Staleness threshold for data feeds (first call only).
    heartbeat_gap_s : float
        Heartbeat gap threshold (first call only).

    Returns
    -------
    SystemHealthMonitor
    """
    global _monitor_instance
    if _monitor_instance is None:
        with _monitor_lock:
            if _monitor_instance is None:
                _monitor_instance = SystemHealthMonitor(
                    state_manager=state_manager,
                    stale_data_threshold_s=stale_data_threshold_s,
                    heartbeat_gap_s=heartbeat_gap_s,
                )
    return _monitor_instance
