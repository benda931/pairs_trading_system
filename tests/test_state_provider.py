from __future__ import annotations

import importlib

from core.state_provider import InMemoryStateProvider


def test_inmemory_state_provider_get_set_has_delete() -> None:
    provider = InMemoryStateProvider()

    assert provider.get("missing") is None
    assert provider.get("missing", 123) == 123
    assert provider.has("answer") is False

    provider.set("answer", 42)

    assert provider.has("answer") is True
    assert provider.get("answer") == 42

    provider.delete("answer")

    assert provider.has("answer") is False
    assert provider.get("answer") is None


def test_importing_app_context_does_not_require_streamlit_runtime() -> None:
    mod = importlib.import_module("core.app_context")
    assert hasattr(mod, "AppContext")
    assert hasattr(mod, "get_app_context")


def test_app_context_get_global_works_with_inmemory_provider(monkeypatch) -> None:
    app_context = importlib.import_module("core.app_context")
    provider = InMemoryStateProvider()

    monkeypatch.setattr(app_context.AppContext, "_GLOBAL_CTX", None)
    monkeypatch.setattr(app_context.AppContext, "init_services", lambda self: None)
    app_context.set_state_provider(provider)

    ctx = app_context.AppContext.get_global()

    assert isinstance(ctx, app_context.AppContext)
    assert provider.get("app_ctx") is ctx
    assert ctx.section == "dashboard"
