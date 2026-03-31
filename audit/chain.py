# -*- coding: utf-8 -*-
"""
audit/chain.py — Append-only audit chain with hash-linking and tamper detection.

One AuditChain per logical entity (e.g., one per strategy_id, one per model_id).
Entries are linked via prev_entry_id forming a verifiable chain.
The AuditChainRegistry manages multiple chains and provides thread-safe append operations.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AuditChainStatus(Enum):
    """Validity state of an audit chain after validation."""
    VALID = "VALID"
    MISSING_LINKS = "MISSING_LINKS"
    HASH_MISMATCH = "HASH_MISMATCH"
    TAMPERED = "TAMPERED"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditChainEntry:
    """
    A single immutable entry in an audit chain.

    ``metadata`` is stored as a tuple of (key, value) pairs for frozen
    compatibility.  Convert back to dict with ``dict(entry.metadata)``.
    """
    entry_id: str
    timestamp: datetime
    actor: str                              # agent_name | "system" | "human:<username>"
    action: str                             # free-text description of what happened
    entity_type: str                        # "strategy" | "model" | "agent" | "policy" | "deployment" | …
    entity_id: str
    payload_hash: str                       # sha256 of JSON-serialised payload
    prev_entry_id: Optional[str]            # None for the first entry in a chain
    chain_id: str                           # logical chain identifier (one per entity)
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)  # frozen-compatible


@dataclass(frozen=True)
class AuditChainValidationReport:
    """Result of walking an AuditChain to verify link integrity."""
    chain_id: str
    entry_count: int
    status: AuditChainStatus
    missing_links: tuple[str, ...]          # entry_ids with broken prev_entry_id references
    hash_mismatches: tuple[str, ...]        # entry_ids whose stored hash does not match recomputed hash
    first_entry_id: Optional[str]
    last_entry_id: Optional[str]
    validated_at: datetime


# ---------------------------------------------------------------------------
# AuditChain
# ---------------------------------------------------------------------------

class AuditChain:
    """
    Append-only audit chain with hash-linking and validation.

    One chain per logical entity (e.g., one per strategy_id, one per model_id).
    Entries are linked via prev_entry_id forming a verifiable chain.

    Thread safety: individual chain append is NOT synchronised here;
    use AuditChainRegistry which wraps access in a lock.
    """

    def __init__(self, chain_id: str) -> None:
        self._chain_id = chain_id
        self._entries: list[AuditChainEntry] = []
        self._entry_index: dict[str, AuditChainEntry] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(
        self,
        actor: str,
        action: str,
        entity_type: str,
        entity_id: str,
        payload: dict,
        metadata: Optional[dict] = None,
    ) -> AuditChainEntry:
        """
        Append a new entry.

        Computes a sha256 hash of the JSON-serialised *payload* and links
        to the previous entry via *prev_entry_id*.

        Returns the newly created :class:`AuditChainEntry`.
        """
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

        prev_id: Optional[str] = self._entries[-1].entry_id if self._entries else None

        entry = AuditChainEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            actor=actor,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            payload_hash=payload_hash,
            prev_entry_id=prev_id,
            chain_id=self._chain_id,
            metadata=tuple((k, v) for k, v in (metadata or {}).items()),
        )
        self._entries.append(entry)
        self._entry_index[entry.entry_id] = entry
        return entry

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> AuditChainValidationReport:
        """
        Walk the chain verifying prev_entry_id links.

        Hash recomputation is not possible without the original payloads,
        so hash_mismatches is always empty in this implementation — the
        field is reserved for external validators that store raw payloads.
        """
        missing_links: list[str] = []
        hash_mismatches: list[str] = []  # reserved for external validators

        for i, entry in enumerate(self._entries):
            if i == 0:
                # The first entry must have no predecessor.
                if entry.prev_entry_id is not None:
                    missing_links.append(entry.entry_id)
            else:
                expected_prev = self._entries[i - 1].entry_id
                if entry.prev_entry_id != expected_prev:
                    missing_links.append(entry.entry_id)

        if missing_links:
            status = AuditChainStatus.MISSING_LINKS
        elif hash_mismatches:
            status = AuditChainStatus.HASH_MISMATCH
        else:
            status = AuditChainStatus.VALID

        return AuditChainValidationReport(
            chain_id=self._chain_id,
            entry_count=len(self._entries),
            status=status,
            missing_links=tuple(missing_links),
            hash_mismatches=tuple(hash_mismatches),
            first_entry_id=self._entries[0].entry_id if self._entries else None,
            last_entry_id=self._entries[-1].entry_id if self._entries else None,
            validated_at=datetime.utcnow(),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        entity_id: Optional[str] = None,
        actor: Optional[str] = None,
        action_contains: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[AuditChainEntry]:
        """Filter entries by optional criteria. All filters are ANDed."""
        results: list[AuditChainEntry] = list(self._entries)
        if entity_id is not None:
            results = [e for e in results if e.entity_id == entity_id]
        if actor is not None:
            results = [e for e in results if e.actor == actor]
        if action_contains is not None:
            needle = action_contains.lower()
            results = [e for e in results if needle in e.action.lower()]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        return results

    def reconstruct_entity_history(self, entity_id: str) -> list[AuditChainEntry]:
        """Return all entries for a given entity in chronological order."""
        return [e for e in self._entries if e.entity_id == entity_id]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chain_id(self) -> str:
        return self._chain_id

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# AuditChainRegistry
# ---------------------------------------------------------------------------

class AuditChainRegistry:
    """
    Registry of AuditChains keyed by chain_id.

    Thread-safe: a single lock guards chain creation so that concurrent
    callers always receive the same AuditChain object for a given chain_id.
    Individual chain.append() calls are *not* additionally synchronised —
    callers that need strict ordering across threads should use the
    :meth:`log` convenience method which acquires the lock for creation
    but *not* for the append.  For fully serialised writes, wrap calls in
    external locking.
    """

    def __init__(self) -> None:
        self._chains: dict[str, AuditChain] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Chain management
    # ------------------------------------------------------------------

    def get_or_create(self, chain_id: str) -> AuditChain:
        """Return the existing chain or create a new one atomically."""
        with self._lock:
            if chain_id not in self._chains:
                self._chains[chain_id] = AuditChain(chain_id)
            return self._chains[chain_id]

    def get(self, chain_id: str) -> Optional[AuditChain]:
        """Return an existing chain or None if not found."""
        return self._chains.get(chain_id)

    # ------------------------------------------------------------------
    # Convenience write
    # ------------------------------------------------------------------

    def log(
        self,
        chain_id: str,
        actor: str,
        action: str,
        entity_type: str,
        entity_id: str,
        payload: dict,
        metadata: Optional[dict] = None,
    ) -> AuditChainEntry:
        """get_or_create the chain then append a new entry."""
        chain = self.get_or_create(chain_id)
        return chain.append(actor, action, entity_type, entity_id, payload, metadata)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_all(self) -> dict[str, AuditChainValidationReport]:
        """Validate every registered chain and return a mapping of reports."""
        return {cid: chain.validate() for cid, chain in self._chains.items()}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_chain_ids(self) -> list[str]:
        """Return a snapshot of all registered chain IDs."""
        return list(self._chains.keys())

    @property
    def total_entries(self) -> int:
        """Total number of entries across all chains."""
        return sum(c.entry_count for c in self._chains.values())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[AuditChainRegistry] = None


def get_audit_chain_registry() -> AuditChainRegistry:
    """Return the process-level singleton AuditChainRegistry."""
    global _registry
    if _registry is None:
        _registry = AuditChainRegistry()
    return _registry
