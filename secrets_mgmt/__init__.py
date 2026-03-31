# -*- coding: utf-8 -*-
"""
secrets — Secret reference and loading package.

Provides SecretReference (an immutable pointer to an externally stored
secret) and SecretLoader (a static utility for retrieving secret values).

SECURITY: This package NEVER stores or logs actual secret values.
Callers must handle None returns from SecretLoader.load() appropriately.

Public re-exports for convenient top-level imports.
"""

from secrets_mgmt.contracts import SecretLoader, SecretReference

__all__ = [
    "SecretLoader",
    "SecretReference",
]
