from __future__ import annotations

from core.allocation_guard import AllocationBatchGuard, AllocationGuardConfig


def test_allocation_guard_first_check_and_start_returns_true(tmp_path):
    guard = AllocationBatchGuard(
        AllocationGuardConfig(state_path=tmp_path / "allocation_guard.json")
    )

    assert guard.check_and_start("daily:20260501") is True


def test_allocation_guard_second_same_batch_returns_false(tmp_path):
    guard = AllocationBatchGuard(
        AllocationGuardConfig(state_path=tmp_path / "allocation_guard.json")
    )

    assert guard.check_and_start("daily:20260501") is True
    assert guard.check_and_start("daily:20260501") is False


def test_allocation_guard_completed_batch_remains_blocked(tmp_path):
    guard = AllocationBatchGuard(
        AllocationGuardConfig(state_path=tmp_path / "allocation_guard.json")
    )

    assert guard.check_and_start("daily:20260501") is True
    guard.mark_completed("daily:20260501")

    assert guard.has_run("daily:20260501") is True
    assert guard.check_and_start("daily:20260501") is False


def test_allocation_guard_failed_batch_can_retry(tmp_path):
    guard = AllocationBatchGuard(
        AllocationGuardConfig(state_path=tmp_path / "allocation_guard.json")
    )

    assert guard.check_and_start("daily:20260501") is True
    guard.mark_failed("daily:20260501", meta={"error": "transient"})

    assert guard.has_run("daily:20260501") is True
    assert guard.check_and_start("daily:20260501") is True
