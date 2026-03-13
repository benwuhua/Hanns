# Fast-Lane Authority Workflow Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a faster workflow for narrow performance hypotheses by adding a local `screen` phase before any authority-tracked reopen round, while preserving the repository's remote-x86-only verdict policy.

**Architecture:** The implementation is documentation-first. First update the operator workflow docs so future sessions distinguish `screen`, `authority`, and `durable closure`. Then align durable-state conventions so `feature-list.json` only tracks authority-grade work. Only after the docs settle should validator rules be tightened, and even then only with narrowly scoped checks.

**Tech Stack:** Markdown workflow docs, JSON durable inventory, Python validator, existing long-task remote command wrappers

---

## File Map

- Modify: `long-task-guide.md`
  - Add the new `screen -> authority -> durable closure` flow and define when each phase starts and ends.
- Modify: `AGENTS.md`
  - Align repository instructions with the new policy that local performance ideas start in `screen`, not immediately in `feature-list.json`.
- Modify: `task-progress.md`
  - Add a durable note describing the new workflow and record adoption in the session log.
- Modify: `feature-list.json`
  - Update project-level policy text only if needed so tracked features are clearly authority-grade work.
- Modify: `docs/PARITY_AUDIT.md`
  - Record the workflow change and explain why reopen rounds now start later.
- Modify: `RELEASE_NOTES.md`
  - Add a concise changelog entry for the workflow shift.
- Modify: `scripts/validate_features.py` (optional second phase)
  - Add only minimal checks needed after the documentation-based workflow is proven stable.
- Test: `python3 scripts/validate_features.py feature-list.json`
  - Main acceptance check for durable-state consistency.

## Chunk 1: Workflow Docs

### Task 1: Update the worker guide

**Files:**
- Modify: `long-task-guide.md`

- [ ] **Step 1: Write the failing doc expectation**

Define the expected additions before editing:

- a `screen` phase exists before tracked authority work
- `screen` allows local tests/benchmarks but not verdict claims
- tracked rounds begin only after `screen_result=promote`
- durable closure remains mandatory after authority

- [ ] **Step 2: Inspect the current guide sections that will change**

Run: `rg -n "Session Workflow|Feature Selection Policy|Rules" long-task-guide.md`

Expected: locate the current worker flow and rule sections that still assume every idea becomes a tracked feature immediately.

- [ ] **Step 3: Edit the guide**

Add or update:

- a short overview of the three phases
- worker instructions for recording `screen` output in `task-progress.md`
- promotion/rejection rules
- guidance that authority remains the only source of performance truth

- [ ] **Step 4: Re-read the edited guide for contradictions**

Run: `sed -n '1,220p' long-task-guide.md`

Expected: the new phase flow is explicit and does not contradict the remote-authority rule.

- [ ] **Step 5: Commit**

```bash
git add long-task-guide.md
git commit -m "docs(workflow): add fast-lane screen phase"
```

## Chunk 2: Repo-Level Operator Instructions

### Task 2: Align AGENTS and durable docs

**Files:**
- Modify: `AGENTS.md`
- Modify: `task-progress.md`
- Modify: `docs/PARITY_AUDIT.md`
- Modify: `RELEASE_NOTES.md`

- [ ] **Step 1: Define the required operator-facing changes**

Expected updates:

- AGENTS explains that narrow performance ideas begin in `screen`
- `task-progress.md` records workflow adoption and current operating policy
- `docs/PARITY_AUDIT.md` records the reason for the workflow shift
- `RELEASE_NOTES.md` records the repository-level change

- [ ] **Step 2: Inspect the current wording**

Run: `rg -n "remote|workflow|reopen|authority|Current focus|Next feature" AGENTS.md task-progress.md docs/PARITY_AUDIT.md RELEASE_NOTES.md`

Expected: find the current text that assumes reopen work is directly tracked.

- [ ] **Step 3: Edit the docs**

Make the minimum edits that:

- preserve existing authority policy
- explain `screen` vs tracked rounds
- avoid rewriting historical session records except where a new session entry is needed

- [ ] **Step 4: Verify the docs read coherently together**

Run: `sed -n '1,220p' AGENTS.md`

Run: `sed -n '1,120p' task-progress.md`

Run: `sed -n '1,40p' docs/PARITY_AUDIT.md`

Run: `sed -n '1,40p' RELEASE_NOTES.md`

Expected: all four documents describe the same workflow and there is no conflict over when a tracked feature starts.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md task-progress.md docs/PARITY_AUDIT.md RELEASE_NOTES.md
git commit -m "docs(progress): adopt fast-lane authority workflow"
```

## Chunk 3: Durable Inventory Policy

### Task 3: Clarify `feature-list.json` scope

**Files:**
- Modify: `feature-list.json`

- [ ] **Step 1: Write the intended policy**

The inventory should clearly imply:

- tracked entries are authority-grade work
- pre-screen exploration does not immediately become a tracked feature

- [ ] **Step 2: Inspect the top-level project metadata**

Run: `sed -n '1,40p' feature-list.json`

Expected: identify where the selection rule and project-level policy text should be adjusted, if at all.

- [ ] **Step 3: Edit only project-level metadata**

Do not rewrite historical features. Only adjust the top-level policy text if needed to reflect the new fast-lane rule.

- [ ] **Step 4: Validate the inventory**

Run: `python3 scripts/validate_features.py feature-list.json`

Expected: `VALID` with no workflow/doc inconsistency.

- [ ] **Step 5: Commit**

```bash
git add feature-list.json
git commit -m "docs(workflow): clarify authority-grade feature tracking"
```

## Chunk 4: Optional Validator Tightening

### Task 4: Add validator support only if the doc-only rollout reveals ambiguity

**Files:**
- Modify: `scripts/validate_features.py`
- Test: `tests/test_gate_profile_hygiene.py` or a new targeted validator unittest if needed

- [ ] **Step 1: Decide whether validator changes are actually required**

Review the updated docs and `feature-list.json` first.

Expected: if the workflow is already unambiguous, skip this chunk.

- [ ] **Step 2: If needed, write the failing validator test**

Add a narrow test around one specific policy, for example:

- tracked work cannot be described as pre-screen local exploration
- hard-stop reopen attempts must mention screen promotion in durable workflow text

- [ ] **Step 3: Run the validator test to confirm failure**

Run the smallest targeted test command for the added validator coverage.

Expected: FAIL before implementation.

- [ ] **Step 4: Implement the minimum validator change**

Keep the rule narrow. Do not turn the validator into a policy engine for every workflow nuance.

- [ ] **Step 5: Re-run the targeted test and the full validator**

Run: targeted validator test command

Run: `python3 scripts/validate_features.py feature-list.json`

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_features.py <validator-test-path>
git commit -m "test(workflow): tighten fast-lane validator policy"
```

## Chunk 5: Final Verification and Handoff

### Task 5: Verify the workflow change as a repository policy update

**Files:**
- Modify: none expected beyond prior chunks

- [ ] **Step 1: Run the validator on final durable state**

Run: `python3 scripts/validate_features.py feature-list.json`

Expected: `VALID - ... workflow/doc checks passed`

- [ ] **Step 2: Re-read the new spec and plan side by side with the updated docs**

Run: `sed -n '1,220p' docs/superpowers/specs/2026-03-13-fast-lane-authority-workflow-design.md`

Run: `sed -n '1,260p' docs/superpowers/plans/2026-03-13-fast-lane-authority-workflow.md`

Expected: the implementation matches the approved design with no obvious drift.

- [ ] **Step 3: Summarize the operational change in `task-progress.md`**

Ensure the latest session entry says:

- what changed
- what future sessions should now do first
- whether validator tightening was deferred or completed

- [ ] **Step 4: Commit any final doc-only adjustments**

```bash
git add task-progress.md docs/PARITY_AUDIT.md RELEASE_NOTES.md
git commit -m "docs(progress): finalize fast-lane workflow rollout"
```

