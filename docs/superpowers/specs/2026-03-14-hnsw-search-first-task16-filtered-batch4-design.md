# HNSW Search-First Task 16 Filtered-Batch4 Design

## Summary

Task 13 showed that the shared generic/bitset kernel can still produce a real local signal when a change removes a meaningful hot-path cost. Task 14 then showed the limit of small heap-threshold cleanups, and Task 15 closed as a local reject after the sorted result-container swap produced a stable regression.

That gives the next step a clear shape: if the pure-Rust `search_first` line continues, it should switch mechanisms again and target a larger cost that still sits inside the shared generic/bitset kernel.

The strongest remaining candidate on the current deterministic bitset lane is the distance loop itself. Today `search_layer_idx_shared(...)` still computes filtered/shared layer-0 L2 neighbor distances one-by-one with `l2_distance_to_idx_ptr(...)`, even though the project already has a working batch-4 L2 kernel in the ordered-pool fast path.

Task 16 therefore targets a narrow but distinct cut: keep the shared generic/bitset traversal shape, visited semantics, and result handling intact, but batch valid layer-0 L2 neighbor distance computations in groups of four on the filtered/shared path.

## Context

The current pure-Rust `search_first` sequence is now:

1. Task 9: unify generic and bitset layer-search onto one shared traversal core
2. Task 10: move generic frontier/results heaps into `SearchScratch`
3. Task 11: add a deterministic bitset diagnosis lane
4. Task 12: specialize the shared L2 distance path
5. Task 13: switch the shared L2 layer-0 neighbor source to `layer0_flat_graph`
6. Task 14: cache the current worst accepted result in `SearchScratch`
7. Task 15: try a shared result-container swap, measure a stable regression, and reject the line

The important boundary after Task 15 is that `search_first` is not closed, but the next slice cannot be “another result-structure experiment.” It needs a different mechanism and must preserve the current deterministic bitset lane as the only local judge.

## Goals

- Reduce filtered/shared layer-0 L2 distance overhead in `search_layer_idx_shared(...)`.
- Reuse the existing batch-4 distance kernel rather than inventing a new SIMD surface.
- Keep shared generic/bitset traversal, filtering, frontier ordering, and result semantics unchanged.
- Expose the new shared-kernel distance mode through the public generic-kernel report.
- Re-measure against the current Task-14 canonical local bitset baseline and decide whether the slice is `promote`, `needs_more_local`, or `reject`.

## Non-Goals

- No changes to the layer-0 ordered-pool fast path.
- No changes to upper-layer paths.
- No changes to result containers or threshold policy.
- No new diagnosis harness or new dataset lane.
- No authority workflow reopen or family-verdict claims.
- No persistence, FFI, or serialization changes.

## Problem Statement

On the current shared generic/bitset path, the per-neighbor loop still looks like this:

1. read one neighbor id / idx
2. check visited state
3. check bitset filter
4. compute one scalar L2 distance
5. decide whether to push the accepted neighbor into frontier/results

That means the shared filtered path pays scalar distance-call overhead even when several adjacent neighbors survive `visited` and bitset checks. The codebase already has a batch-4 L2 kernel and production fast-path precedent for grouping distances, but the shared generic/bitset path has not been allowed to reuse that idea yet.

This is a different hypothesis from Task 15:

- Task 15 changed result maintenance and got slower
- Task 16 keeps result maintenance unchanged and changes only distance batching

## Proposed Approaches

### Option 1: Batch valid shared-path neighbors in groups of four

Keep the shared traversal loop, but accumulate up to four valid, unfiltered neighbor indices before calling the existing batch-4 L2 kernel.

Properties:

- visited and bitset checks still happen per neighbor
- only distance computation is grouped
- tail groups smaller than four continue to use the scalar helper

Pros:

- different mechanism from the rejected Task-15 line
- likely larger upside than more heap/frontier micro-cleanup
- stays attributable to the distance loop

Cons:

- more complex local control flow than pure scalar iteration
- must ensure stats and ordering stay consistent

### Option 2: Add prefetch only on the shared filtered path

Keep scalar distances and only add more explicit vector prefetch on the shared path.

Pros:

- smaller code delta

Cons:

- weaker than the size of the remaining local gap
- too close to earlier small locality-style slices

### Option 3: Batch both distance and accepted-neighbor mutation together

Bundle batch-4 distance with a larger rewrite of frontier/result mutation.

Pros:

- larger upside if it works

Cons:

- broader than one screen slice
- weaker attribution if the lane moves or regresses

## Recommendation

Choose Option 1.

Task 16 should remain a narrow shared-kernel distance slice: collect valid neighbors into groups of four, reuse the existing batch-4 L2 helper, keep tail fallback scalar, and leave frontier/result policy alone.

## Design

### 1. Shared filtered batch-4 mode

Add a dedicated shared-kernel distance mode for the filtered/generic layer-0 L2 path:

- visited and bitset checks still gate which neighbors are considered
- valid neighbors are accumulated into a small local fixed-size array
- full groups of four call the existing batch-4 L2 helper
- any remaining neighbors at the end of the loop fall back to the scalar helper

This keeps the traversal semantics stable while changing only the shape of distance computation.

### 2. Keep acceptance logic unchanged

Task 16 must not change:

- when a neighbor becomes eligible for distance computation
- the result-threshold check
- frontier push behavior
- final nearest-first output

The batch-4 path should feed distances into the existing accepted-neighbor logic rather than replacing that logic.

### 3. Public contract surface

Extend `HnswGenericSearchKernelReport` with a new field such as:

- `shared_bitset_distance_mode`

Expected value for Task 16:

- `idx_ptr_batch4_when_grouped`

This keeps the mechanism visible to default-lane contract tests in `tests/bench_hnsw_generic_search_kernel.rs`.

### 4. Stats constraints

The deterministic bitset diagnosis lane should continue to hold:

- recall constant
- visited-node counts constant
- frontier push/pop counts constant

Distance-call accounting may change shape depending on how batch calls are counted, but the new shape must be explainable and consistent across reruns.

## Data Flow

1. The shared kernel pops a frontier candidate as it does today.
2. It walks candidate neighbors and applies visited / bitset checks one by one.
3. Surviving neighbor indices are queued into a local batch buffer.
4. Each full batch of four calls the existing batch-4 L2 helper.
5. Each returned distance flows through the current accepted-neighbor logic unchanged.
6. Tail neighbors smaller than four use the current scalar helper.
7. The deterministic bitset diagnosis lane compares qps against the Task-14 canonical baseline while checking recall and traversal counters.

## Verification Plan

Task 16 should require all of the following:

- a red/green contract update in `tests/bench_hnsw_generic_search_kernel.rs` for `shared_bitset_distance_mode`
- focused library regressions covering:
  - shared generic layer-search result parity
  - shared bitset layer-search result parity
  - repeated shared-kernel scratch reuse across calls
  - shared filtered batch-4 path activation on a deterministic fixture
- fresh deterministic bitset diagnosis rerun against the canonical Task-14 baseline
- one same-code `/tmp` rerun to compare measured delta against lane drift
- fresh `cargo test hnsw --lib -- --nocapture`
- fresh `cargo fmt --all -- --check`
- fresh `python3 scripts/validate_features.py feature-list.json`

## Success Criteria

Task 16 counts as a good screen slice only if:

- contract and focused regressions stay green
- recall remains unchanged on the deterministic bitset lane
- visited/frontier counters remain unchanged
- the measured qps delta is materially larger than same-code rerun drift

If the batch-4 shared-path cut does not beat the lane’s noise floor, it should be recorded as `screen_result=needs_more_local` or `reject` based on the measured sign and magnitude.
