# -*- coding: utf-8 -*-
"""
core/runtime.py — Phase 1 Runtime Resolution (HF-grade)
======================================================

Single place to resolve:
- env/profile precedence (explicit > PTS_* env vars > legacy env vars > config.json > defaults)
- deterministic seed derivation (env, profile, run_id)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


_ALLOWED_ENVS = {"dev", "research", "paper", "live"}


def normalize_env_value(env: str) -> str:
    env_norm = (env or "").strip().lower()
    if not env_norm:
        return "dev"
    if env_norm in _ALLOWED_ENVS:
        return env_norm
    # common aliases
    if env_norm in {"backtest", "research-dev", "sandbox"}:
        return "research"
    if env_norm in {"paper-trading", "papertrade"}:
        return "paper"
    if env_norm in {"production", "prod"}:
        return "live"
    return "dev"


def _get_env_var(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def resolve_env_profile(
    cfg: Mapping[str, Any] | None,
    *,
    explicit_env: Optional[str] = None,
    explicit_profile: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Precedence:
      env:
        1) explicit_env
        2) PTS_ENV, then legacy: APP_ENV, DASH_ENV
        3) cfg["environment"]["default_env"]
        4) "dev"

      profile:
        1) explicit_profile
        2) PTS_PROFILE, then legacy: APP_PROFILE, DASH_PROFILE
        3) cfg["environment"]["default_profile"]
        4) "default"
    """
    cfg = cfg or {}
    env_block = cfg.get("environment") or {}

    env_raw = (
        explicit_env
        or _get_env_var("PTS_ENV", "APP_ENV", "DASH_ENV")
        or (str(env_block.get("default_env") or "").strip() or None)
        or "dev"
    )
    profile_raw = (
        explicit_profile
        or _get_env_var("PTS_PROFILE", "APP_PROFILE", "DASH_PROFILE")
        or (str(env_block.get("default_profile") or "").strip() or None)
        or "default"
    )

    env = normalize_env_value(env_raw)
    profile = (profile_raw or "default").strip().lower() or "default"
    return env, profile


def stable_seed(env: str, profile: str, run_id: str, *, mod: int = 2**31 - 1) -> int:
    """
    Deterministic seed derived from (env, profile, run_id).
    """
    s = f"{env}|{profile}|{run_id}".encode("utf-8", errors="ignore")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:12], 16) % mod


@dataclass(frozen=True)
class RuntimeResolved:
    env: str
    profile: str
    run_id: str
    seed: int
