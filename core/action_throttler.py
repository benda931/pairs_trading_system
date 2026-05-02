from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ActionThrottler:
    DEFAULT_COOLDOWNS = {
        "KILL_SWITCH": 14400,
        "DELEVERAGE": 1800,
        "FORCE_EXIT": 300,
        "BLOCK_ENTRY": 600,
        "RETRAIN_MODEL": 86400,
        "OPTIMIZE_PARAMS": 86400,
        "UPDATE_CONFIG": 3600,
    }

    def __init__(self, state_path: Path | None = None, *, max_actions_per_cycle: int = 10):
        self.state_path = state_path or (PROJECT_ROOT / "logs" / "action_throttler.json")
        pytest_test = os.environ.get("PYTEST_CURRENT_TEST", "").strip()
        if pytest_test and state_path is None:
            safe_name = "".join(ch if ch.isalnum() else "_" for ch in pytest_test)[:120]
            self.state_path = self.state_path.with_name(
                f"{self.state_path.stem}_{safe_name}{self.state_path.suffix}"
            )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_actions_per_cycle = int(max_actions_per_cycle)
        self._actions_this_cycle = 0

    def _load(self) -> dict:
        if not self.state_path.exists():
            return {"actions": {}}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data.setdefault("actions", {})
        return data

    def _save(self, data: dict) -> None:
        self.state_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _state_key(self, action_type: str, key: str | None = None) -> str:
        action_txt = str(action_type or "").strip().upper()
        key_txt = str(key or "global").strip()
        return f"{action_txt}:{key_txt}"

    def allow(self, action_type: str, key: str | None = None) -> bool:
        if self._actions_this_cycle >= self.max_actions_per_cycle:
            return False

        state = self._load()
        record = (state.get("actions") or {}).get(self._state_key(action_type, key)) or {}
        last_ts_txt = record.get("last_ts")
        if not last_ts_txt:
            return True

        try:
            last_ts = datetime.fromisoformat(str(last_ts_txt))
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last_ts.astimezone(timezone.utc)).total_seconds()
        except Exception:
            return True

        cooldown = int(self.DEFAULT_COOLDOWNS.get(str(action_type or "").strip().upper(), 300))
        return elapsed >= cooldown

    def mark(self, action_type: str, key: str | None = None) -> None:
        state = self._load()
        actions = dict(state.get("actions") or {})
        actions[self._state_key(action_type, key)] = {
            "last_ts": datetime.now(timezone.utc).isoformat(),
        }
        state["actions"] = actions
        self._actions_this_cycle += 1
        self._save(state)
