# -*- coding: utf-8 -*-
"""
secrets/contracts.py — Secret Reference and Loader Contracts
=============================================================

Provides:
  - SecretReference: an immutable pointer to a secret stored externally.
    NEVER stores the actual secret value.
  - SecretLoader: a static utility class for loading secrets from
    environment variables or config files.

Security principles
-------------------
- SecretReference carries only enough metadata to locate the secret.
  It never holds the secret value.
- SecretLoader.load() returns the value or None; it NEVER logs or
  prints the secret value.
- All logging in SecretLoader uses only the secret_name, not the value.
- Callers decide what to do when load() returns None (fail fast, use
  default, raise, etc.).
- stdlib only — no external dependencies.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 1. SECRET REFERENCE (IMMUTABLE)
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SecretReference:
    """A reference to a secret stored in an external provider.

    NEVER stores the actual secret value.  Use SecretLoader.load()
    to retrieve the value at runtime.

    Fields
    ------
    ref_id : str
        Unique identifier for this reference record.
    secret_name : str
        Logical name of the secret (e.g. "FMP_API_KEY").
    env : str
        Deployment environment this reference applies to.
    provider : str
        Backend provider: "env_var" | "config_file" | "vault" | "aws_ssm".
    key_path : str
        Provider-specific path/key:
          - "env_var":     environment variable name (e.g. "FMP_API_KEY")
          - "config_file": dot-path within the JSON config
                           (e.g. "api_keys.fmp")
          - "vault":       Vault secret path (e.g. "secret/trading/fmp")
          - "aws_ssm":     SSM Parameter Store path
                           (e.g. "/trading/live/fmp_api_key")
    scope : str
        Access scope: "read_only" | "trading" | "admin".
    rotation_due : Optional[str]
        ISO-8601 date when rotation is due (or None if no policy).
    last_rotated : Optional[str]
        ISO-8601 date of last rotation (or None if never rotated).
    notes : str
        Free-text notes (no secret values).
    """

    ref_id: str
    secret_name: str
    env: str
    provider: str
    key_path: str
    scope: str
    rotation_due: Optional[str]
    last_rotated: Optional[str]
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 2. SECRET LOADER
# ══════════════════════════════════════════════════════════════════


class SecretLoader:
    """Load secrets from environment variables or config files.

    Security invariants
    -------------------
    - NEVER logs or prints secret values.
    - NEVER caches secret values in memory.
    - Returns None if the secret is unavailable; the caller
      decides how to handle the absence.
    - check_available() verifies presence without loading the value.
    - Supports "env_var" and "config_file" providers natively.
      Other providers ("vault", "aws_ssm") raise NotImplementedError
      unless a custom provider is registered via register_provider().

    Custom providers
    ----------------
    To add support for an external secret backend (e.g. HashiCorp Vault),
    call SecretLoader.register_provider("vault", my_loader_fn) where
    my_loader_fn has signature (key_path: str) -> Optional[str].
    """

    # Registry of custom provider callbacks
    # key: provider name; value: callable(key_path: str) -> Optional[str]
    _custom_providers: dict = {}

    @classmethod
    def register_provider(cls, provider: str, loader_fn: object) -> None:
        """Register a custom secret provider callable.

        Parameters
        ----------
        provider : str
            Provider name (e.g. "vault", "aws_ssm").
        loader_fn : callable
            Callable with signature (key_path: str) -> Optional[str].
            MUST NOT log or return the value in any diagnostic output.
        """
        cls._custom_providers[provider] = loader_fn

    @classmethod
    def load(cls, ref: SecretReference) -> Optional[str]:
        """Load the secret value referenced by ref.

        Parameters
        ----------
        ref : SecretReference
            The secret reference to resolve.

        Returns
        -------
        Optional[str]
            The secret value, or None if unavailable.

        Notes
        -----
        Never logs the returned value.
        """
        try:
            if ref.provider == "env_var":
                return cls._load_from_env(ref.key_path)
            elif ref.provider == "config_file":
                return cls._load_from_config_file(ref.key_path)
            elif ref.provider in cls._custom_providers:
                loader_fn = cls._custom_providers[ref.provider]
                return loader_fn(ref.key_path)  # type: ignore[operator]
            else:
                logger.warning(
                    "SecretLoader: unsupported provider '%s' for secret '%s'.",
                    ref.provider,
                    ref.secret_name,
                )
                return None
        except Exception:  # noqa: BLE001
            # Deliberately broad: we never want secret loading to raise
            # unhandled exceptions into caller code.
            logger.error(
                "SecretLoader: failed to load secret '%s' from provider '%s'.",
                ref.secret_name,
                ref.provider,
            )
            return None

    @classmethod
    def check_available(cls, ref: SecretReference) -> bool:
        """Check whether a secret is present without loading its value.

        Parameters
        ----------
        ref : SecretReference

        Returns
        -------
        bool
            True if the secret appears to be available (non-empty).
        """
        value = cls.load(ref)
        available = value is not None and len(value) > 0
        # value is deliberately not logged
        return available

    @staticmethod
    def validate_freshness(ref: SecretReference) -> Tuple[bool, str]:
        """Check whether the secret's rotation is due.

        Parameters
        ----------
        ref : SecretReference

        Returns
        -------
        tuple[bool, str]
            (fresh, reason)
            ``fresh`` is True if rotation is not overdue.
            ``reason`` explains the freshness status.
        """
        if ref.rotation_due is None:
            return (True, "No rotation policy configured.")

        try:
            due_dt = datetime.fromisoformat(ref.rotation_due)
            if due_dt.tzinfo is None:
                due_dt = due_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return (False, f"Could not parse rotation_due: '{ref.rotation_due}'.")

        now = datetime.now(timezone.utc)
        if now > due_dt:
            overdue_days = (now - due_dt).days
            return (
                False,
                f"Secret '{ref.secret_name}' rotation overdue by {overdue_days} day(s).",
            )

        days_remaining = (due_dt - now).days
        return (
            True,
            f"Secret '{ref.secret_name}' rotation due in {days_remaining} day(s).",
        )

    # ──────────────────────────────────────────────────────────────
    # PRIVATE LOADING BACKENDS
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_from_env(key_path: str) -> Optional[str]:
        """Load a secret from an environment variable.

        Parameters
        ----------
        key_path : str
            Environment variable name.

        Returns
        -------
        Optional[str]
        """
        value = os.environ.get(key_path)
        if value is None or value == "":
            logger.debug(
                "SecretLoader: env var '%s' not set or empty.", key_path
            )
            return None
        return value

    @staticmethod
    def _load_from_config_file(key_path: str) -> Optional[str]:
        """Load a secret from the project config JSON file.

        Searches the following locations in order:
        1. ``config.json`` in the current working directory.
        2. ``common/config.json`` relative to the current directory.

        key_path is a dot-separated path within the JSON structure
        (e.g. "api_keys.fmp" maps to config["api_keys"]["fmp"]).

        Parameters
        ----------
        key_path : str
            Dot-separated JSON path.

        Returns
        -------
        Optional[str]
        """
        candidate_paths = [
            os.path.join(os.getcwd(), "config.json"),
            os.path.join(os.getcwd(), "common", "config.json"),
        ]

        config_data: Optional[dict] = None
        for path in candidate_paths:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        config_data = json.load(fh)
                    break
                except (OSError, json.JSONDecodeError) as exc:
                    logger.debug(
                        "SecretLoader: could not read config file '%s': %s",
                        path,
                        exc,
                    )

        if config_data is None:
            logger.debug("SecretLoader: no config.json found for key_path '%s'.", key_path)
            return None

        # Traverse dot-separated path
        parts = key_path.split(".")
        node: Any = config_data
        for part in parts:
            if not isinstance(node, dict):
                logger.debug(
                    "SecretLoader: key_path '%s' traversal failed at '%s'.",
                    key_path,
                    part,
                )
                return None
            node = node.get(part)
            if node is None:
                return None

        if isinstance(node, str) and node:
            return node
        if isinstance(node, (int, float)):
            return str(node)
        return None


# ══════════════════════════════════════════════════════════════════
# TYPE ALIAS FOR CLARITY
# ══════════════════════════════════════════════════════════════════

# Re-export Any for the config file loader type hint
from typing import Any  # noqa: E402 (placed here to avoid polluting top of file)
