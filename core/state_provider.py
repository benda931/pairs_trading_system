from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StateProvider(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def has(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...


@dataclass
class InMemoryStateProvider:
    _store: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def has(self, key: str) -> bool:
        return key in self._store

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


class StreamlitStateProvider:
    def _session_state(self) -> Any:
        import streamlit as st  # noqa: PLC0415

        return st.session_state

    def get(self, key: str, default: Any = None) -> Any:
        return self._session_state().get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._session_state()[key] = value

    def has(self, key: str) -> bool:
        return key in self._session_state()

    def delete(self, key: str) -> None:
        self._session_state().pop(key, None)


def get_default_state_provider() -> StateProvider:
    try:
        provider = StreamlitStateProvider()
        provider.get("__state_provider_probe__", None)
        return provider
    except Exception:
        return InMemoryStateProvider()
