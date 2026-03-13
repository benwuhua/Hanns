# HNSW Round 9 Search Fast Path Audit Design

## Summary

Round 9 reopens HNSW on a narrow, code-quality hypothesis:

- production `layer0 + L2 + no-filter` search still pays profiling-oriented overhead in the hot loop
- `l2_batch_4_ptrs()` still re-runs CPU feature detection on every call

Round 9 does not attempt another broad HNSW "breakthrough" claim. It isolates a small search-path fast-path cut so that a later authority rerun can answer a single question: does removing profiling leakage and batch-4 dispatch overhead produce a measurable same-schema gain on the real lane?

## Context

The round-8 graph-quality line is complete and archived as a hard stop. The bulk-build path now matches the intended native-style semantics on the synthetic audit surface, but the authority same-schema rerun regressed Rust HNSW from `828.725` qps to `750.732` qps while native also drifted down. That result blocks any further "graph-quality is the root cause" narrative.

The remaining credible near-term hypothesis is implementation quality inside the real search hot path rather than another build-path theory.

## Goals

- Prove or disprove that production search is still polluted by profiling-oriented control flow in the `layer0 + L2 + no-filter` hot path.
- Remove repeated batch-4 kernel feature detection from the steady-state path.
- Preserve all existing profiled/audit surfaces used by rounds 4-8.
- Produce a new authority-backed artifact chain that keeps round-9 claims narrowly attributable.

## Non-Goals

- No new graph-construction changes.
- No accepted-candidate link-list prefetch in this round.
- No visited-table compression in this round.
- No co-located/slab vector-plus-graph memory layout rework in this round.
- No leadership/verdict refresh claim unless a fresh authority rerun justifies opening a separate tracked feature.

## Proposed Approaches

### Option 1: Narrow fast-path split plus batch-4 cache

Create a dedicated production fast path for `layer0 + L2 + no-filter` search and keep the existing profiled function for audit surfaces. Add a cached batch-4 kernel selector in `src/simd.rs`.

Pros:

- smallest change that directly targets the suspected production overhead
- cleanest attribution on the authority lane
- preserves existing profiling artifacts and tests

Cons:

- if the gain is small, the HNSW reopen line likely stops again

### Option 2: Fast path plus link-list prefetch

Take Option 1 and also add prefetch of accepted candidates' layer-0 link lists.

Pros:

- potentially larger gain

Cons:

- mixes two separate hypotheses
- makes any authority gain harder to attribute

### Option 3: Full search micro-optimization bundle

Take Option 2 and also add visited compression, AVX2/FMA dispatch work, and additional search-side refactors.

Pros:

- maximizes the amount of code changed in one pass

Cons:

- poor attribution
- higher regression risk
- conflicts with the long-task policy of keeping post-hard-stop reopen lines narrow

## Recommendation

Choose Option 1.

Round 8 already showed that structurally plausible changes can fail on the authority lane. The next round should therefore minimize scope and maximize explainability. If this small fast-path split does not move the real benchmark, the reopen line has much stronger evidence that remaining HNSW gaps are not caused by lightweight search-side overhead.

## Design

### 1. Production search fast path

Add a dedicated `layer0 + L2 + no-filter` fast path in `src/faiss/hnsw.rs`.

Required properties:

- no `Instant::now()` calls
- no `Option<...profile...>` argument
- no profiling stats recording branches
- no extra audit-only bookkeeping
- directly uses the existing ordered-pool and flat-graph search behavior

The existing profiled/optional-profile function remains in place for round-4 through round-8 profiling and audit generators. Production search entry points should dispatch to the new fast path only when all of these conditions are true:

- metric is `L2`
- no filter is active
- search is operating on layer 0
- the ordered-pool fast path is otherwise eligible

This keeps behavioral scope narrow and avoids rewriting unrelated search modes.

### 2. Cached batch-4 kernel dispatch

Add a cached selector for `l2_batch_4_ptrs()` in `src/simd.rs`, mirroring the existing single-vector pointer-kernel cache.

Required properties:

- select once per process via `OnceLock`
- preserve current architecture gating semantics
- keep the public API stable so HNSW search can switch to the cached path with minimal call-site churn

If AVX2 requires FMA for the existing batch-4 kernel, that contract remains unchanged in round 9. This round is about removing repeated detection overhead, not broadening SIMD capability support.

### 3. Durable artifact chain

Round 9 should create a new durable workflow slice analogous to round 8:

- round-9 activation/baseline artifact
- round-9 default-lane contract test
- round-9 profile/audit artifact proving the production path split exists
- round-9 authority same-schema summary artifact

The audit artifact should prove facts about code shape and execution mode, not just benchmark numbers. At minimum it should record:

- production layer-0 fast-path mode string
- profiled path mode string
- whether the production path avoids profiling counters/timing calls
- batch-4 dispatch mode string

## Data Flow

1. Round-9 activation freezes round-8 hard-stop evidence as the new starting point.
2. A profile/audit test generates a durable artifact that proves the new fast path exists and is selected under the intended conditions.
3. Production search entry points use the fast path on the narrow eligible lane.
4. Authority rerun refreshes:
   - Rust same-schema row
   - native capture
   - round-9 audit/profile artifact
   - round-9 default contract
5. A round-9 authority summary decides whether the hypothesis moved the real lane or should be archived as another stop.

## Error Handling and Safety

- The fast path must preserve current search results/contracts for the same query and parameters.
- If eligibility checks fail, existing generic/profilled paths remain the fallback.
- SIMD dispatch caching must keep the same CPU feature requirements as today.
- No persistence or serialization format changes are allowed in round 9.

## Testing Strategy

### Red phase

- Add a new default-lane round-9 contract test that fails until a round-9 baseline artifact exists.
- Add focused library tests that fail until:
  - production search selects the new fast path on `L2 + no-filter + layer0`
  - the profiled path remains available and distinct
  - batch-4 dispatch is cached instead of re-detected per call

### Green phase

- Implement the minimum fast-path split and batch-4 cache needed to turn those tests green.

### Verification

Local prefilters:

- targeted HNSW library tests
- round-9 default contract
- round-9 audit/profile generator
- formatting checks

Authority acceptance:

- `bash init.sh`
- authority round-9 audit/profile replay
- authority round-9 default contract replay
- authority same-schema HDF5 rerun
- fresh native HNSW capture
- validator pass on updated durable state

## Expected Outcomes

There are only two acceptable round-9 conclusions:

1. Measurable authority-lane gain with a clean attribution to fast-path cleanup, which may justify a later small follow-up feature.
2. No meaningful gain, in which case the HNSW reopen line gains strong evidence that remaining gaps are not explained by lightweight search-loop overhead.

Round 9 is successful if it produces a trustworthy answer, not only if qps goes up.
