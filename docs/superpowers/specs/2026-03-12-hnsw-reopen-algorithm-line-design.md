# HNSW Reopen Algorithm Line Design

## Context

`knowhere-rs` is currently in a durable terminal state: all tracked long-task features are closed, the project-level verdict is archived as `not accepted`, and the HNSW family is archived as `functional-but-not-leading` on the remote x86 authority surface. That archival state is honest, but it also biases future work toward preserving verdict artifacts instead of pushing the only remaining high-value technical path.

The new requirement is narrower than a full project reset:

- reopen only the HNSW line
- allow direct changes to core graph-build logic in `src/faiss/hnsw.rs`
- keep recall from regressing on the authority lane
- preserve the current API/FFI/persistence contracts
- defer any project-level acceptance rewrite until HNSW produces materially better remote evidence

## Problem

The repository has two conflicting truths:

1. The workflow is healthy and reproducible.
2. The current workflow is optimized for verdict preservation, not HNSW performance improvement.

If work continues under the current terminal-state model, new sessions will keep treating the HNSW verdict as something to defend rather than something to challenge with new authority evidence. That is the wrong operating mode for algorithm work.

## Goals

- Reopen HNSW as an active multi-session algorithm-improvement line.
- Preserve the 2026-03-12 final artifacts as historical baseline evidence.
- Shift HNSW feature acceptance from "artifact added and regression-locked" to "core code changed and remote evidence improved or sharpened the bottleneck model."
- Create a new durable baseline for HNSW improvement work that can be compared against each iteration.
- Keep IVF-PQ, DiskANN, and project-level final acceptance in archived state unless a later HNSW result justifies reopening them.

## Non-Goals

- Do not reopen the whole non-GPU production-replacement program.
- Do not weaken or delete historical final-verdict artifacts.
- Do not redefine remote x86 authority rules.
- Do not chase IVF-PQ or DiskANN optimization in this reopen cycle.
- Do not claim project-level acceptance from local-only benchmarks.

## Design

### 1. Durable-State Reopen Model

The current final artifacts remain in place as historical truth:

- `benchmark_results/hnsw_p3_002_final_verdict.json`
- `benchmark_results/final_core_path_classification.json`
- `benchmark_results/final_performance_leadership_proof.json`
- `benchmark_results/final_production_acceptance.json`

These artifacts are reinterpreted as "last settled baseline" rather than "active branch stop signal."

Durable workflow state is reopened only for HNSW:

- `task-progress.md` moves from `Current focus: none` / `Next feature: none` to a new active HNSW reopen feature.
- `feature-list.json` gains a new HNSW reopen feature chain with explicit dependencies and verification steps.
- `TASK_QUEUE.md`, `GAP_ANALYSIS.md`, and `docs/PARITY_AUDIT.md` document that the project remains historically `not accepted`, but the HNSW family has been reopened for algorithm work.

This keeps the governance model honest: the old verdict still exists, but it is now challengeable by new authority artifacts.

### 2. New HNSW Reopen Artifacts

The reopen line introduces artifacts with a different role from the final verdict chain:

- `benchmark_results/hnsw_reopen_baseline.json`
  - freezes the current trusted baseline for the reopen effort
  - records the Rust/native authority metrics that future iterations must compare against
- one or more profiling/audit artifacts for the build/search hot path
  - used to explain where time is being spent
  - not treated as final-family verdicts
- refreshed HNSW benchmark artifacts only after a real code change materially affects the authority lane

These artifacts are progress trackers, not archival stop/go endpoints.

### 3. First Reopen Feature Chain

The first reopen cycle is intentionally narrow:

1. `hnsw-reopen-baseline-freeze`
   - creates the new reopen baseline artifact from existing trusted evidence
   - rewrites durable state so HNSW is active again

2. `hnsw-build-path-profiler`
   - instruments or audits `src/faiss/hnsw.rs` to separate build/search bottlenecks
   - focuses on layer descent, candidate expansion, diversification, reverse-link shrink, repair, and scratch reuse

3. `hnsw-build-quality-rework`
   - performs the first direct algorithm change in `src/faiss/hnsw.rs`
   - likely targets insertion/build behavior rather than public API surface
   - aims for measurable authority improvement without recall regression

4. `hnsw-authority-rerun-and-verdict-refresh`
   - re-runs the same-schema authority lane
   - decides whether the HNSW family verdict changes or remains blocked
   - only if this produces materially different authority evidence can higher-level final artifacts be reconsidered

### 4. Allowed HNSW Algorithm Surface

The reopen line explicitly allows changes to:

- top-down layer descent during insertion
- candidate-set expansion during build
- layer-0 vs upper-layer neighbor selection policy
- reverse-link shrink behavior
- graph-repair behavior and when it runs
- scratch/visited reuse where it affects hot-path cost without altering semantics

The reopen line must preserve:

- current external API behavior
- FFI contract semantics
- persistence semantics already archived as supported/constrained
- search correctness at existing recall gates

### 5. Verification Policy

Local verification is used only for TDD and safety:

- new unit/property tests in `src/faiss/hnsw.rs`
- targeted HNSW regression tests
- compile/build/lint surfaces that protect API and contract stability

Remote x86 remains the only acceptance surface for improvement claims.

A reopen HNSW feature is considered complete only if:

- HNSW core code changed, and
- local regressions are green, and
- remote authority evidence shows either:
  - recall is preserved and performance improved, or
  - a profiling hypothesis was decisively tested, narrowing the next iteration

Features that only add artifacts or docs without new HNSW code or new authority evidence do not count as successful reopen iterations.

### 6. Stop/Go Rules

Continue the reopen line when:

- recall is stable or better and authority QPS improves
- or a profiling iteration materially sharpens the bottleneck model

Stop or reassess when:

- two consecutive core-logic iterations produce no measurable authority improvement
- preserving contracts becomes impossible without larger architectural changes
- or the authority lane shows the current Rust HNSW architecture is fundamentally capped far below native

At that point the repo can either:

- keep HNSW archived as `functional-but-not-leading`, or
- open a much larger architecture rewrite discussion

## Testing Strategy

- Unit tests in `src/faiss/hnsw.rs` for diversification, shrink, repair, and layer-specific behavior
- Existing HNSW compare/default-lane tests continue to protect the historical verdict chain
- New reopen-specific tests will lock the new baseline and profiler artifacts
- Authority commands must stay aligned with the existing same-schema HDF5 benchmark surface so results remain comparable to the archived baseline

## Risks

- Reopening only HNSW could create state drift if durable docs are not explicit about "historical final verdict" versus "active reopen line"
- Profiling work can turn into endless instrumentation if it is not forced to feed a concrete algorithm change
- It is possible that HNSW improves but still does not justify reopening project-level final acceptance

## Success Condition

This design is successful if the repository moves from "terminal archived verdict" to "active HNSW improvement line" without erasing history, and if future HNSW sessions are judged by core-code changes plus new authority evidence rather than by artifact maintenance alone.
