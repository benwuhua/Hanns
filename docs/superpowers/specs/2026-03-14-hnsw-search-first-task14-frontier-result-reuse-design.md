# HNSW Search-First Task 14 Frontier-Result Reuse Design

## Summary

Task 13 proved that the dedicated local bitset diagnosis lane can now see real shared-kernel gains: moving the shared `level=0 + L2` neighbor traversal onto `layer0_flat_graph` improved qps by roughly `13% .. 15%` while recall, visited-node counts, and distance-call counts stayed fixed.

That changes the bar for the next slice. The next `search_first` task should stay on the same local screen surface and target another cost that is:

- inside the shared generic/bitset kernel
- local to `search_layer_idx_shared(...)`
- visible on the deterministic bitset lane
- small enough to attribute cleanly if it moves qps again

The narrowest credible next target is the `frontier/result` maintenance work that still happens on nearly every accepted neighbor:

- `generic_results.peek()` is consulted repeatedly to derive the current worst-result threshold
- `generic_results.pop()` / `push()` still maintain the bounded result heap
- `generic_frontier.push()` remains coupled to accepted results

Task 14 therefore focuses on shrinking the cost of maintaining the shared result threshold without changing search semantics.

## Context

The current pure-Rust `search_first` sequence is now:

1. Task 9: unify generic and bitset layer-search onto one shared traversal core
2. Task 10: move generic frontier/results heaps into `SearchScratch`
3. Task 11: add a deterministic bitset diagnosis lane
4. Task 12: specialize the shared L2 distance path
5. Task 13: switch the shared L2 layer-0 neighbor source to `layer0_flat_graph`

Tasks 10 and 12 were directionally correct but too close to local timing noise to promote. Task 13 was different: the qps gain was much larger than same-code rerun drift, so the shared kernel now has a trustworthy local optimization line.

The next step should preserve that attribution discipline. This is not the moment to broaden into upper-layer rewrites, result-format changes, or another measurement-harness task.

## Goals

- Reduce per-neighbor result-threshold overhead inside `search_layer_idx_shared(...)`.
- Keep generic and bitset layer-search semantics unchanged.
- Preserve recall, visited-node counts, and distance-call counts on the deterministic bitset lane.
- Expose the result-threshold mode through the existing public generic-kernel report so the new behavior is contract-visible.
- Re-measure against the same Task-13 local bitset baseline and decide whether the slice is `promote`, `needs_more_local`, or `reject`.

## Non-Goals

- No changes to the ordered-pool layer-0 fast path.
- No changes to authority workflow state or family verdict artifacts.
- No new diagnosis harness or new dataset lane.
- No graph-layout rewrite beyond what Task 13 already introduced.
- No upper-layer search redesign.
- No serialization or FFI changes.

## Problem Statement

After Task 13, the shared generic/bitset kernel still performs result-threshold work in the hot loop by consulting the result heap directly:

- once to decide whether the current candidate can terminate traversal
- again for each accepted neighbor to determine whether the neighbor is better than the current worst result
- then possibly again when updating the bounded result heap

The current implementation is correct, but it keeps the hot loop coupled to repeated heap-top reads and branchy worst-threshold handling. This is a smaller cost than the old `id -> idx` neighbor path, but it is still paid throughout the shared generic/bitset traversal and should be visible on the deterministic bitset lane.

## Proposed Approaches

### Option 1: Cache the current worst-result threshold in `SearchScratch`

Add a lightweight cached threshold to `SearchScratch` for the shared generic/bitset kernel:

- initialize it from the entry result state
- update it only when the result heap actually changes
- let the hot loop read the cached value instead of repeatedly unpacking `generic_results.peek()`

Pros:

- smallest code delta
- strongest attribution to result-threshold maintenance
- keeps the existing heap representation and result ordering

Cons:

- still leaves heap push/pop work in place
- requires careful invalidation to avoid stale threshold bugs

### Option 2: Split result-threshold maintenance into a dedicated helper around the same heap

Keep the same heap state, but centralize:

- worst-threshold reads
- conditional replacement
- frontier/result push-pop accounting

Pros:

- better code clarity
- easier future changes

Cons:

- risks becoming mostly refactor with weak measurable payoff
- may not reduce the hot-loop work enough on its own

### Option 3: Replace the result heap in the shared kernel with a different bounded container

Use a small sorted buffer or another structure for bounded results while keeping `generic_frontier` as a heap.

Pros:

- potentially larger upside

Cons:

- broader semantic risk
- much harder to attribute cleanly in one screen slice
- likely too large for the current `search_first` fast lane

## Recommendation

Choose Option 1.

Task 14 should remain a narrow screen slice: keep `generic_results` as the source of truth, but cache the currently worst accepted distance in `SearchScratch` and only refresh it when the result heap changes. This keeps the optimization local, measurable, and easy to roll back if the local bitset lane does not move.

## Design

### 1. Scratch-owned result-threshold cache

Extend `SearchScratch` with a field representing the current worst accepted result distance for the shared generic/bitset kernel.

The cache should have simple semantics:

- empty or `f32::INFINITY` before any accepted result exists
- equal to the actual worst result distance whenever `generic_results` is non-empty
- updated immediately after any result-heap mutation

The cache is not a second result representation. It is only a cheap read path for threshold checks.

### 2. Shared helper for accepted-neighbor updates

Move the repeated “should this neighbor enter results?” and “if yes, how do we mutate results/frontier?” logic behind one local helper or one compact code path so the shared kernel does not repeat:

- threshold comparison
- conditional result eviction
- result insertion
- frontier insertion
- threshold-cache refresh

This should stay inside `src/faiss/hnsw.rs` and should not change public search APIs.

### 3. Public contract surface

Extend `HnswGenericSearchKernelReport` with a new field such as:

- `shared_result_threshold_mode`

Expected value for Task 14:

- `scratch_cached_worst`

This keeps the optimization visible to default-lane contract tests in `tests/bench_hnsw_generic_search_kernel.rs`.

### 4. Semantics constraints

Task 14 must preserve:

- bitset filtering behavior
- traversal stop condition behavior
- result ordering
- visited-node behavior
- distance-call behavior

If recall or traversal counters move, the slice should be treated as a semantic change rather than a pure cost reduction.

## Data Flow

1. The shared kernel prepares scratch state for one layer-search call.
2. Entry result insertion initializes the cached worst threshold.
3. Each candidate pop reads the cached threshold for early termination checks.
4. Each neighbor acceptance check compares against the cached threshold.
5. Only accepted neighbors mutate `generic_results` and `generic_frontier`.
6. Any result-heap mutation refreshes the cached threshold immediately.
7. The deterministic bitset diagnosis lane compares qps against the Task-13 canonical baseline while checking that recall and traversal counters remain fixed.

## Verification Plan

Task 14 should require all of the following:

- a red/green contract update in `tests/bench_hnsw_generic_search_kernel.rs` for `shared_result_threshold_mode`
- focused library regressions for repeated shared-kernel calls so cached-threshold state cannot leak across calls
- fresh deterministic bitset diagnosis rerun against the canonical Task-13 baseline
- one same-code `/tmp` rerun to compare measured delta against lane drift
- fresh `cargo test hnsw --lib -- --nocapture`
- fresh `cargo fmt --all -- --check`
- fresh `python3 scripts/validate_features.py feature-list.json`

## Success Criteria

Task 14 counts as a good screen slice only if:

- contract and regression tests stay green
- recall remains unchanged on the deterministic bitset lane
- visited-node and distance-call counts remain unchanged
- measured qps gain is meaningfully larger than same-code rerun drift

If the slice only improves code structure but does not beat the lane’s own noise floor, it should be recorded as `screen_result=needs_more_local` rather than promoted.
