# Workflow And Doc Consistency Gate Design

## Goal

Upgrade the existing `scripts/validate_features.py` utility from a JSON schema check into a lightweight workflow/doc consistency gate for the long-task process.

The gate should catch the specific classes of drift already observed in this repository:

- `feature-list.json` says one thing while `task-progress.md` says another
- latest session log still contains stale placeholders like `pending rerun after durable-state update`
- verification steps use commands that can silently execute `0 tests`
- temporary or orphaned artifacts remain in the repo and undermine trust in the durable state

## Non-Goals

This change will not:

- parse arbitrary Markdown structure across the whole repository
- verify that every logged command was truly executed on the authority host
- replace `cargo test`, remote harness scripts, or gate profiles
- solve remote parallelism or worktree isolation
- lint all docs for style or all shell commands for correctness

## Context

The repository already has the right durable files for a harness-oriented workflow:

- `feature-list.json` as source of truth for feature state
- `task-progress.md` as session ledger
- `RELEASE_NOTES.md` as durable changelog
- `long-task-guide.md` / `AGENTS.md` as operator instructions

What is missing is a mechanical cross-check between them. Today, `scripts/validate_features.py` validates only local JSON shape and dependency references. That is too weak for a workflow where future sessions rely on repo state rather than hidden memory.

## Approaches Considered

### 1. Extend `scripts/validate_features.py`

Pros:

- reuses the existing entrypoint already referenced by `feature-list.json` and `long-task-guide.md`
- smallest workflow change
- easiest to adopt in remote and local verification

Cons:

- one script now owns both JSON validation and workflow/doc consistency logic

### 2. Add a second script such as `scripts/validate_workflow_state.py`

Pros:

- cleaner separation between data-shape checks and cross-file checks

Cons:

- introduces a second command that humans and future sessions can forget to run
- more likely to drift from the existing long-task instructions

### 3. Build a larger hygiene suite

Pros:

- could eventually cover stale docs, orphan files, gate drift, and remote artifacts

Cons:

- too broad for the current need
- higher chance of false positives and rollout friction

### Recommendation

Use approach 1. Extend `scripts/validate_features.py` with narrow, repository-specific workflow checks that target known failure modes.

## Design

### Inputs

The validator will read:

- `feature-list.json`
- `task-progress.md`
- `RELEASE_NOTES.md`

It should still accept `feature-list.json` as its CLI argument for compatibility, and it should resolve the other durable files relative to the repository root inferred from that path.

### Validation Layers

#### Layer 1: Existing feature inventory checks

Preserve the current checks for:

- required fields
- status/priority enums
- non-empty `verification_steps`
- dependency references

#### Layer 2: Current-state consistency

Validate that:

- `task-progress.md` current focus exists in `feature-list.json`
- `task-progress.md` next feature exists in `feature-list.json`
- if current focus or next feature names a feature, its status is compatible with the ledger
- the progress summary in `task-progress.md` matches the counted `passing` features in `feature-list.json`

This layer should be intentionally conservative. It should reject obvious contradictions, not infer project intent.

#### Layer 3: Latest-session hygiene

Parse only the latest session block in `task-progress.md` and fail when it still contains known stale placeholders such as:

- `pending rerun after durable-state update`
- `pending`
- equivalent unresolved post-verify markers in the latest session verification list

This check should focus on the most recent session only, because older sessions intentionally preserve historical truth.

#### Layer 4: Verification-step bad-pattern detection

Reject known command patterns that can silently under-verify in this repository, including:

- `cargo test --tests -q <name>` or similar forms that intend to target an integration test binary by suffix filter
- commands previously documented in the session log as `0 tests` shells

This is not a general shell parser. It is a repository-specific denylist backed by already observed failures.

#### Layer 5: Repo hygiene sentinels

Fail if the repository contains obvious temporary or orphaned implementation artifacts, starting with:

- `*.new`
- `*.old`
- `*.tmp`

This list should stay small and explicit at first to avoid accidental noise.

### Output

The script should keep the current human-readable style:

- on failure: grouped validation errors with enough detail to fix them quickly
- on success: existing feature counts, plus a short summary that workflow/doc checks also passed

## Error Handling

- Missing durable companion files should fail validation with explicit paths
- Markdown parse failures should fail closed with a message that the expected section could not be found
- ambiguous matches should fail rather than guess

## Testing Strategy

Add focused unit tests for the validator covering:

- happy path with consistent durable state
- mismatch between `task-progress.md` progress summary and `feature-list.json`
- stale placeholder remaining in the latest session
- bad verification command pattern
- orphan temp file detection

Use fixture strings and temp directories so tests do not mutate the real repo.

## Rollout

1. Extend the validator and add tests.
2. Run it against the current repo and fix any real issues it surfaces.
3. Keep the CLI unchanged so existing long-task verification steps continue to work.

## Future Follow-Ups

If this lands cleanly, the next logical harness upgrades are:

- remote harness isolation and per-run worktrees
- a doc-garden pass for stale benchmark narratives
- richer gate-profile validation against actual test files
