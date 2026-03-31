# Documentation Index — Pairs Trading System

> **Platform maturity:** Disciplined research platform with comprehensive infrastructure scaffolding.
> See `INTEGRATION_STATUS.md` for what is operational vs scaffolded.

---

## Quick Start by Role

| If you are a... | Start here | Then read |
|-----------------|-----------|-----------|
| **New contributor** | `../README.md` -> `../CONTRIBUTING.md` | `../CLAUDE.md`, architecture docs |
| **Quant researcher** | `discovery_methodology.md` | `signal_architecture.md`, ADR-007 |
| **ML engineer** | `ml_architecture.md` | `../CLAUDE.md` ML section, `domain-model.md` |
| **Platform engineer** | `INTEGRATION_STATUS.md` | `architecture.md`, `production_architecture.md` |
| **Reviewer / auditor** | `../docs/remediation/remediation_ledger.md` | `INTEGRATION_STATUS.md`, `governance_architecture.md` |
| **Operator** | `runbooks.md` | `production_architecture.md` |
| **Future Claude Code** | `../CLAUDE.md` | `migration/migration_ledger.md`, `promptbook.md` |

---

## Architecture & Design

| Document | Lines | Description | Integration Status |
|----------|-------|-------------|-------------------|
| `architecture.md` | 171 | Platform overview, 6-layer stack, pair lifecycle | Accurate |
| `signal_architecture.md` | 586 | Signal engine, regime, lifecycle, quality | Partially Integrated |
| `portfolio_architecture.md` | 367 | Allocation, sizing, risk operations | Scaffold |
| `ml_architecture.md` | 714 | ML features, labels, models, inference | Scaffold |
| `agent_architecture.md` | 939 | 33 agents, orchestration, governance | Scaffold |
| `governance_architecture.md` | 786 | Policies, controls, surveillance, audit | Scaffold |
| `production_architecture.md` | 651 | Runtime, control plane, monitoring, alerts | Scaffold |
| `discovery_methodology.md` | 302 | Research methodology and validation doctrine | Operational |

## Reference

| Document | Description |
|----------|-------------|
| `domain-model.md` | Canonical objects handbook — all major types and their contracts |
| `glossary.md` | 50+ canonical terms with definitions |
| `INTEGRATION_STATUS.md` | System A (operational) vs System B (scaffold) register |
| `testing.md` | Test taxonomy, quality gates, coverage map |
| `runbooks.md` | Operational runbooks for incident/recovery scenarios |
| `promptbook.md` | Claude Code role prompts for disciplined future sessions |

## Architecture Decision Records

| ADR | Title |
|-----|-------|
| ADR-001 | Domain Contracts — `core/contracts.py` as single source of truth |
| ADR-002 | Pair Validation Doctrine — explicit rejection reasons |
| ADR-003 | Spread Construction — OLS/Rolling OLS/Kalman philosophy |
| ADR-004 | Agent Design — bounded mandates, typed I/O, audit logging |
| ADR-005 | Canonical Kill Switch — escalation-only, no auto-recovery |
| ADR-006 | Canonical Signal Pipeline — regime -> threshold -> quality -> intent |
| ADR-007 | Backtest Realism Limitations — same-close timing, flat costs, calendar WF |

## Migration & Remediation

| Document | Description |
|----------|-------------|
| `migration/migration_ledger.md` | 7 tracked migration items with status and sunset conditions |
| `remediation/remediation_ledger.md` | 21+ findings (P0-P4) with severity and status |
| `remediation/research_integrity_fixes.md` | 4 research integrity fixes with before/after |

## Root-Level Documents

| Document | Description |
|----------|-------------|
| `../README.md` | Platform overview with truthful maturity labeling |
| `../CONTRIBUTING.md` | Contributor safety guide — import rules, deprecated paths, test requirements |
| `../CLAUDE.md` | Platform handbook for Claude Code — 24 sections, 1150+ lines |
