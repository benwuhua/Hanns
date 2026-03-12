# Prod FFI Observability Persistence Gate Design

## Goal

Close `prod-ffi-observability-persistence-gate` by proving that the existing cross-cutting production contracts still pass together on the remote x86 authority lane.

## Problem

The repository already contains the focused contract work that closed the underlying gaps:

- HNSW and IVF-PQ FFI persistence behavior was aligned with metadata
- `knowhere_get_index_meta` already exposes the current observability schema
- `tests/bench_json_export.rs` was already converted from a default-lane shell into a real contract regression

What remained open was not missing behavior. It was missing closure evidence for the combined production-facing surface:

- `cargo test --lib ffi -- --nocapture`
- `cargo test --lib serialize -- --nocapture`
- `cargo test --test bench_json_export -q`

If those three lanes are green on the authority machine, the feature is complete. If they fail, only the exposed contract drift should be fixed.

## Options

### Option 1: Honest gate replay

Treat this as a verification-first closure:

1. rerun the three recorded local surfaces
2. refresh the remote authority workspace
3. rerun the same three commands on x86
4. update durable state only after the authority replay is green

Pros:

- keeps scope tied to the actual feature definition
- avoids inventing new FFI/schema work that the feature does not require
- produces replayable authority evidence

Cons:

- may reveal no new code changes are needed, so the session becomes mostly governance/durable-state work

### Option 2: Expand the FFI capability surface

Add new metadata fields, observability details, or persistence behaviors as part of this feature.

Pros:

- could improve the contract surface beyond the current minimum

Cons:

- changes scope
- risks reopening already-closed contract areas
- is not required for the recorded verification steps

### Option 3: Local-only closure

Use the local test results as sufficient evidence and mark the feature passing without a new authority replay.

Pros:

- fastest

Cons:

- violates the remote-x86 authority rule for production acceptance
- leaves the feature without fresh replayable evidence

## Decision

Take Option 1.

This feature should close as an authority-backed contract replay, not as a new implementation sprint.

## Scope

This feature will:

1. rerun the three recorded local contract surfaces as a prefilter
2. refresh the authority workspace with `bash init.sh`
3. rerun the three recorded remote commands on isolated target/log directories
4. update durable workflow state once the authority runs pass

## Non-Goals

This feature will not:

- widen the FFI schema
- add new observability capabilities
- reopen family-level benchmark or parity work
- weaken any verification step

## Design

### 1. Local prefilter

Run:

- `cargo test --lib ffi -- --nocapture`
- `cargo test --lib serialize -- --nocapture`
- `cargo test --test bench_json_export -q`

If all three already pass, treat the feature as a gate-closure candidate rather than a coding task.

### 2. Serial authority replay with isolated directories

Use:

- `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-prod-ffi-observability-persistence-gate`
- `KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-prod-ffi-observability-persistence-gate`

and rerun the three commands on the remote x86 authority machine.

The remote wrapper uses a per-log-dir lock. A naive parallel replay can return `status=conflict`, so the authoritative evidence for this feature should come from serialized runs under the isolated directories.

### 3. Durable closure

Once the remote evidence is green:

- mark `prod-ffi-observability-persistence-gate` as `passing`
- record the log paths in `task-progress.md` and release/governance docs
- advance the queue to `prod-readme-remote-workflow-docs`
- rerun `python3 scripts/validate_features.py feature-list.json`

## Testing Strategy

1. local prefilter on the three recorded lanes
2. `bash init.sh`
3. serialized remote replays on isolated directories
4. durable-state sync
5. `python3 scripts/validate_features.py feature-list.json`

## Risks

### Risk: remote lock conflict obscures the real status

Mitigation:

Treat `status=conflict` as scheduler noise, not verification evidence. Re-run serially under isolated directories.

### Risk: unnecessary scope expansion

Mitigation:

If the recorded lanes are already green, do not invent new FFI or observability work just to make the feature look implementation-heavy.
