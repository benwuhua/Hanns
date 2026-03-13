# knowhere-rs-non-gpu-production - Long Task Worker Guide

Follow this guide at the start of every Codex session.

## Scope

This workflow covers the full non-GPU production-replacement program for `knowhere-rs`:

- baseline and methodology closure
- HNSW family
- IVF-PQ family
- DiskANN family
- cross-cutting production gates
- final acceptance

## Authority Rule

The existing remote x86 machine is the only authoritative execution surface.

- Performance claims count only when backed by remote x86 artifacts.
- Stop-go verdicts count only when backed by remote x86 artifacts.
- Local cargo results are allowed for quick smoke or file-level iteration, but they do not justify marking a feature `passing`.

## Phase Model

Narrow performance work now uses a three-phase fast lane:

1. `screen`
2. `authority`
3. `durable closure`

The purpose of `screen` is to reject weak ideas before they consume a full tracked reopen round.

- `screen` may use local tests, local benchmarks, local profiles, or small local audits
- `screen` must not make performance or stop-go claims
- `screen` work is recorded in `task-progress.md`, not immediately added to `feature-list.json`

A hypothesis is promoted into tracked work only after `screen_result=promote`.

## Default Autonomy Rule

Future Codex sessions should continue autonomously by default.

- If a documented choice already has a recommended option, take the recommended option without asking the user again.
- If multiple paths are possible, prefer the smallest change that preserves the project goal and keeps the remote-x86 evidence chain trustworthy.
- Only stop to ask the user when blocked by missing external information, a requirement conflict, or a materially destructive or scope-changing action.
- Do not ask for confirmation just to continue the next queued feature.

## Session Workflow

### 1. Orient
1. Confirm the working directory.
2. Read `long-task-guide.md`.
3. Read `AGENTS.md`.
4. Read `task-progress.md`.
5. Read `feature-list.json`.
6. Review recent git history.
7. Decide whether the next step is:
   - a local `screen` for a new narrow hypothesis, or
   - the highest-priority `failing` tracked feature whose dependencies are all `passing`.

### 2. Bootstrap
1. If you are entering `screen`, bootstrap only what is needed for the local test or local benchmark you are about to run.
2. If you are entering tracked `authority` work, run `bash init.sh`.
3. Confirm the remote x86 configuration is printed and the workspace sync completes before any authority claim or tracked verification.
4. If the selected tracked feature depends on a previously passing remote behavior, re-run that remote smoke before making changes.

### 3. TDD Red
1. Add or tighten failing tests first.
2. For benchmark or artifact features, add the failing regression/parser/schema check before changing implementation or docs.

### 4. TDD Green
1. Implement the minimum change needed to make the new tests pass.
2. Keep changes scoped to the selected feature.

### 5. Refactor
1. Clean up without changing the feature boundary.
2. If a refactor expands scope, stop and update durable docs before continuing.

### 6. Verify and Mark
1. If you are in `screen`, produce a single local decision:
   - `screen_result=promote`
   - `screen_result=reject`
   - `screen_result=needs_more_local`
2. Record the `screen` hypothesis, expected mechanism, threshold, commands run, and result in `task-progress.md`.
3. If you are in tracked work, run every command in the feature's `verification_steps`.
4. Use remote x86 outputs as the source of truth for any performance or verdict claim.
5. Mark a tracked feature as `passing` only after all required verification succeeds.

### 7. Persist
1. If the work ended in `screen_result=reject`, commit only if the local exploration itself is worth preserving; otherwise discard it cleanly.
2. If the work ended in `screen_result=promote`, open tracked authority-grade work only after the screen result is recorded.
3. For tracked work, commit only the files relevant to the completed feature.
4. For tracked work, update `task-progress.md`.
5. For tracked work, update `RELEASE_NOTES.md`.
6. For tracked work, run `python3 scripts/validate_features.py feature-list.json`.

### 8. Handoff
1. Stop after one fully verified tracked feature or one completed `screen` decision unless a second step is trivial and already unblocked.
2. Record:
   - what changed
   - whether this was `screen`, `authority`, or `durable closure`
   - which verification commands passed
   - which feature should be taken next
   - any newly discovered blockers or stop-go evidence
3. The next session starts again from Step 1.

## Feature Selection Policy

- Prefer `high` priority over `medium`, `medium` over `low`.
- Never skip unmet dependencies.
- Do not add a new narrow performance idea directly to `feature-list.json` unless it has already passed `screen`.
- For a family or sub-line that previously ended in `hard_stop`, require a fresh `screen` before reopening tracked work.
- If a family already has enough remote evidence for a no-go or constrained verdict, follow-up features in that family should focus on boundary-locking and documentation, not reopening unsupported parity claims.

## Rules

- Do not skip tests.
- Do not treat local screen output as authoritative evidence.
- Do not mark features as `passing` without remote verification.
- Do not delete or weaken `verification_steps` without a requirement change recorded in durable files.
- Do not rely on hidden session memory.
- Do not treat local benchmarks as acceptance evidence.
- Do not reopen a narrow performance round as tracked work before recording a screen result.
- Leave the repository in a working state before ending the session.
