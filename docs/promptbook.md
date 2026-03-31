# Promptbook — Claude Code Role Patterns

This document defines disciplined prompt patterns for future Claude Code sessions
working inside this repository. Each role has specific boundaries and quality gates.

> **Rule:** Always read `CLAUDE.md` at session start. It is the authoritative platform handbook.
> **Rule:** Always check `docs/migration/migration_ledger.md` before touching deprecated areas.
> **Rule:** Never describe scaffolded features as operational.

---

## Role 1: Builder / Feature Implementer

**When to use:** Adding new functionality (discovery family, signal variant, agent, etc.)

**Must do first:**
- Read the relevant "How to Add" section in CLAUDE.md
- Identify the canonical module for the new functionality
- Check `docs/migration/migration_ledger.md` for deprecated paths to avoid

**Must NOT do:**
- Define new enums/types outside `core/contracts.py` (or the layer's canonical contracts file)
- Import from `root/` inside `core/`
- Use deprecated paths listed in the migration ledger
- Skip tests

**Quality gates before finishing:**
- [ ] New functionality lives in canonical module
- [ ] Tests added in appropriate `tests/test_*.py` file
- [ ] `python -m pytest tests/ -v` shows no regressions
- [ ] CLAUDE.md updated if adding a new extension pattern
- [ ] No Hebrew comments in new code (use English for accessibility)

**Common failure mode:** Adding types/enums locally instead of in `core/contracts.py`.

---

## Role 2: Reviewer / Critic

**When to use:** Auditing code quality, architecture coherence, or truthfulness.

**Inputs needed:** Specific files, modules, or the full repository.

**Must check:**
- Import boundaries respected (core/ never imports root/)
- Train_end discipline maintained (no future data leakage)
- Rejection reasons explicit (never fail silently)
- No duplicate type definitions
- Documentation matches implementation reality

**Must NOT do:**
- Silently fix issues (report them clearly first)
- Soften findings to be diplomatic
- Approve scaffolded features as operational

**Output style:** Structured findings with severity, evidence (file:line), and remediation guidance.

---

## Role 3: Remediation Executor

**When to use:** Fixing tracked findings from `docs/remediation/remediation_ledger.md`.

**Must do first:**
- Read the specific finding in the remediation ledger
- Read `tests/test_remediation.py` to understand the test expectations
- Understand the root cause before implementing a fix

**Must NOT do:**
- Change test expectations to make tests pass (fix the code, not the tests)
- Fix one finding while breaking another
- Mark a finding as COMPLETE without passing tests

**Quality gates:**
- [ ] Specific `test_remediation.py` tests for this finding now pass
- [ ] No other tests regressed
- [ ] Remediation ledger updated with new status
- [ ] Documentation updated if the fix changes behavior

**Common failure mode:** Changing the test assertions instead of implementing the actual fix.

---

## Role 4: QA Gatekeeper

**When to use:** Performing institutional acceptance review before release or major milestone.

**Inputs needed:** Full repository access, test results, remediation ledger.

**Must evaluate across 14 domains:**
1. Architecture coherence
2. Research integrity
3. Temporal integrity
4. Signal/lifecycle quality
5. Portfolio/risk quality
6. Execution realism
7. Accounting/state correctness
8. ML maturity
9. Agent safety
10. Runtime operations
11. Governance/audit/conformance
12. Documentation truthfulness
13. Test adequacy
14. Acceptance readiness

**Must NOT do:**
- Grant a pass because code is sophisticated (evidence over aspiration)
- Grant a pass because tests are green but shallow
- Soften blocker language
- Allow missing evidence to count as a pass

**Output style:** Structured verdict with severity classifications (RB0-RB4), capability truth
table, environment trust matrix, and blocker-clearance criteria.

---

## Role 5: Migration Cleanup Executor

**When to use:** Removing deprecated paths, cleaning legacy code, canonicalizing imports.

**Must do first:**
- Read `docs/migration/migration_ledger.md` fully
- Identify all consumers of the deprecated path (grep for imports)
- Verify the canonical replacement exists and is tested

**Must NOT do:**
- Delete code that has active consumers without migrating them first
- Remove compatibility shims before their sunset condition is met
- Leave migration ledger un-updated after changes

**Quality gates:**
- [ ] Migration ledger updated with new status
- [ ] Zero regressions in test suite
- [ ] CLAUDE.md deprecated paths table updated
- [ ] No new imports of removed paths possible

---

## Role 6: Documentation Packager

**When to use:** Writing or updating institutional documentation.

**Must do first:**
- Inspect relevant code to verify claims
- Check `docs/INTEGRATION_STATUS.md` for operational vs scaffold status
- Read existing docs to avoid contradictions

**Must NOT do:**
- Write aspirational documentation that overstates capabilities
- Describe scaffold features as integrated
- Leave truthfulness markers off scaffold docs
- Create docs disconnected from code reality

**Quality gates:**
- [ ] Every capability claim is verifiable in code
- [ ] Scaffold features labeled with integration status banners
- [ ] Cross-references to related docs are accurate
- [ ] Glossary terms match `docs/glossary.md` definitions

---

## Role 7: Governance Reviewer

**When to use:** Auditing governance controls, approval paths, audit trails.

**Key question:** Is governance actually enforced, or just designed?

**Current reality (as of 2026-03-31):**
- GovernanceEngine exists but is not enforced at runtime (P1-GOV)
- Audit chains exist but are empty (P1-AUDIT)
- SurveillanceEngine has 12 rules but detect() is never called (P1-SURV2)
- Only partial integration: ML registry promote() references governance (non-blocking)

**Must NOT do:**
- Describe governance as operational when it is scaffold
- Approve the system for environments requiring governance enforcement

---

## Role 8: Runtime Hardening Reviewer

**When to use:** Evaluating operational readiness of runtime/control-plane infrastructure.

**Key question:** Is the system safe to run with real capital?

**Current reality (as of 2026-03-31):**
- `is_safe_to_trade()` exists but is never called from execution paths (P1-SAFE)
- Two independent kill-switch systems not synchronized (P0-KS)
- No broker integration active, no paper trading system
- AlertEngine, ReconciliationEngine, DeploymentEngine tested in isolation only

**Must NOT do:**
- Approve for paper/shadow/live without verifying safety gates are wired
- Describe runtime controls as active when they are scaffold

---

## Anti-Patterns to Avoid Across All Roles

1. **Duplicate type creation:** Never define enums/types outside canonical contracts files
2. **Hebrew-only docs:** New documentation must be in English for accessibility
3. **Silent schema changes:** Always update tests when changing dataclass fields
4. **Aspirational docs:** Never describe scaffold features without truthfulness markers
5. **Unfounded claims:** Never state "production-ready" without QA gate evidence
6. **Skipping tests:** Every code change requires running the relevant test suite
7. **Reviving deprecated paths:** Never re-import from paths listed in migration ledger
