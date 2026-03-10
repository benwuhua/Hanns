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
7. Pick the highest-priority `failing` feature whose dependencies are all `passing`.

### 2. Bootstrap
1. Run `bash init.sh`.
2. Confirm the remote x86 configuration is printed and the workspace sync completes.
3. If the selected feature depends on a previously passing remote behavior, re-run that remote smoke before making changes.

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
1. Run every command in the feature's `verification_steps`.
2. Use remote x86 outputs as the source of truth.
3. Mark the feature as `passing` only after all required verification succeeds.

### 7. Persist
1. Commit only the files relevant to the completed feature.
2. Update `task-progress.md`.
3. Update `RELEASE_NOTES.md`.
4. Run `python3 scripts/validate_features.py feature-list.json`.

### 8. Handoff
1. Stop after one fully verified feature unless a second feature is trivial and already unblocked.
2. Record:
   - what changed
   - which verification commands passed
   - which feature should be taken next
   - any newly discovered blockers or stop-go evidence
3. The next session starts again from Step 1.

## Feature Selection Policy

- Prefer `high` priority over `medium`, `medium` over `low`.
- Never skip unmet dependencies.
- If a family already has enough remote evidence for a no-go or constrained verdict, follow-up features in that family should focus on boundary-locking and documentation, not reopening unsupported parity claims.

## Rules

- Do not skip tests.
- Do not mark features as `passing` without remote verification.
- Do not delete or weaken `verification_steps` without a requirement change recorded in durable files.
- Do not rely on hidden session memory.
- Do not treat local benchmarks as acceptance evidence.
- Leave the repository in a working state before ending the session.
