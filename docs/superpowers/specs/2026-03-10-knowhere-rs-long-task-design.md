# knowhere-rs Long-Task Design

**Date:** 2026-03-10

## Goal

Create a durable multi-session workflow for `knowhere-rs` that can survive across Codex sessions while pursuing the strict completion target:

- non-GPU CPU-path replaceability against native Knowhere
- production engineering closure
- at least one credible remote-x86 performance leadership result over native

## Requirement Sources

- `TASK_QUEUE.md`
- `DEV_ROADMAP.md`
- `GAP_ANALYSIS.md`
- `docs/PARITY_AUDIT.md`
- `docs/FFI_CAPABILITY_MATRIX.md`

## Scope

This workflow covers the full non-GPU program, not a single implementation task. The durable artifacts must therefore organize work across multiple independent but related lanes:

- baseline and methodology closure
- HNSW
- IVF-PQ
- DiskANN
- cross-cutting production gates
- final acceptance

## Key Design Decisions

### 1. Organize by index family, not by file

The user selected an index-family-first workflow. Each family gets its own correctness, performance, and production-facing features. Shared infrastructure remains in separate baseline and cross-cutting lanes.

### 2. Use the existing remote x86 machine as the sole authority

All performance claims, stop-go verdicts, and acceptance gates must be backed by remote x86 evidence. Local results may assist editing or debugging, but they cannot justify `passing` state in the durable workflow.

### 3. Make `feature-list.json` the source of truth

`feature-list.json` records durable feature state. `task-progress.md` remains a session log only. To keep cross-session reasoning hard-edged:

- features start as `failing`
- features only become `passing` after all `verification_steps` succeed
- dependencies must already be `passing` before a later feature can be selected

### 4. Keep features single-session sized

Each feature should be completable and verifiable in one Codex session. Large goals such as “fix HNSW performance” are broken into audit, regression, repair, benchmark, and verdict steps.

### 5. End each family with a verdict

HNSW, IVF-PQ, and DiskANN each terminate in a stop-go classification:

- production-candidate
- functional-but-not-leading
- constrained
- no-go

That prevents endless open-ended tuning after enough evidence already exists.

## Durable Files

- `feature-list.json`: task truth source
- `task-progress.md`: session log and current handoff
- `long-task-guide.md`: worker instructions for all future sessions
- `RELEASE_NOTES.md`: ongoing change log
- `scripts/validate_features.py`: feature inventory validator
- `init.sh`: remote-x86 bootstrap entrypoint

## Feature Taxonomy

The first-pass inventory is split into:

- `baseline`
- `hnsw`
- `ivfpq`
- `diskann`
- `cross-cutting-prod`
- `final-acceptance`

## Completion Definition

The project is only complete when all of the following are true:

1. Non-GPU core paths have honest final classifications.
2. Production gates are closed on the remote x86 authority surface.
3. At least one core CPU path demonstrates credible remote-x86 leadership over native at the same recall gate.
4. All claims are replayable from durable artifacts and logs.

## Out of Scope for This Workflow Initialization

- immediate algorithm changes
- reclassification of already-open family verdicts without new evidence
- GPU parity
- replacing existing roadmap/governance docs as the product truth source

## Initial Next Step

The first worker feature is `baseline-remote-bootstrap`, because every later session depends on a reproducible remote-authority bootstrap path.
