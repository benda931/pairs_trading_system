from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AllocationGuardConfig:
    state_path: Path = PROJECT_ROOT / "logs" / "allocation_guard.json"
    min_seconds_between_batches: int = 6 * 60 * 60
    one_batch_per_trading_day: bool = True


class AllocationBatchGuard:
    def __init__(self, config: AllocationGuardConfig | None = None):
        self.config = config or AllocationGuardConfig()
        pytest_test = os.environ.get("PYTEST_CURRENT_TEST", "").strip()
        if pytest_test and self.config.state_path == AllocationGuardConfig().state_path:
            safe_name = "".join(ch if ch.isalnum() else "_" for ch in pytest_test)[:120]
            self.config.state_path = self.config.state_path.with_name(
                f"{self.config.state_path.stem}_{safe_name}{self.config.state_path.suffix}"
            )
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)

    def make_batch_id(self, *, trading_day: date | None = None, strategy: str = "daily") -> str:
        trading_day = trading_day or date.today()
        return f"{strategy}:{trading_day:%Y%m%d}"

    def _load(self) -> dict[str, Any]:
        path = self.config.state_path
        if not path.exists():
            return {"batches": {}, "last_started_ts": None, "last_completed_ts": None}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data.setdefault("batches", {})
        data.setdefault("last_started_ts", None)
        data.setdefault("last_completed_ts", None)
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.config.state_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def has_run(self, batch_id: str) -> bool:
        data = self._load()
        return batch_id in dict(data.get("batches") or {})

    def mark_started(self, batch_id: str, meta: dict | None = None) -> None:
        data = self._load()
        batches = dict(data.get("batches") or {})
        batches[batch_id] = {
            "status": "started",
            "started_at": self._now_iso(),
            "meta": dict(meta or {}),
        }
        data["batches"] = batches
        data["last_started_ts"] = batches[batch_id]["started_at"]
        self._save(data)

    def mark_completed(self, batch_id: str, meta: dict | None = None) -> None:
        data = self._load()
        batches = dict(data.get("batches") or {})
        batch = dict(batches.get(batch_id) or {})
        batch["status"] = "completed"
        batch["completed_at"] = self._now_iso()
        batch["meta"] = {**dict(batch.get("meta") or {}), **dict(meta or {})}
        batches[batch_id] = batch
        data["batches"] = batches
        data["last_completed_ts"] = batch["completed_at"]
        self._save(data)

    def mark_failed(self, batch_id: str, meta: dict | None = None) -> None:
        data = self._load()
        batches = dict(data.get("batches") or {})
        batch = dict(batches.get(batch_id) or {})
        batch["status"] = "failed"
        batch["failed_at"] = self._now_iso()
        batch["meta"] = {**dict(batch.get("meta") or {}), **dict(meta or {})}
        batches[batch_id] = batch
        data["batches"] = batches
        self._save(data)

    def check_and_start(self, batch_id: str, meta: dict | None = None) -> bool:
        data = self._load()
        batches = dict(data.get("batches") or {})
        existing_batch = dict(batches.get(batch_id) or {})
        existing_status = str(existing_batch.get("status") or "").strip().lower()
        if self.config.one_batch_per_trading_day and batch_id in batches:
            if existing_status in {"started", "completed"}:
                return False

        last_ts_txt = data.get("last_started_ts") or data.get("last_completed_ts")
        if last_ts_txt and existing_status not in {"failed"}:
            try:
                last_ts = datetime.fromisoformat(str(last_ts_txt))
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                elapsed = (datetime.now(timezone.utc) - last_ts.astimezone(timezone.utc)).total_seconds()
                if elapsed < float(self.config.min_seconds_between_batches):
                    return False
            except Exception:
                pass

        self.mark_started(batch_id, meta=meta)
        return True

    def reset(self) -> None:
        if self.config.state_path.exists():
            self.config.state_path.unlink()
