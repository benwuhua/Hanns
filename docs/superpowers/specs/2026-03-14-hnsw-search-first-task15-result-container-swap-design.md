# HNSW Search-First Task 15 Result-Container Swap Design

## Summary

Task 13 showed that the shared generic/bitset kernel can still produce meaningful local gains when a change removes a real hot-path cost. Task 14 then showed the opposite side of the boundary: caching the current worst-result threshold cleaned up the shared kernel, but its qps movement stayed inside the deterministic bitset lane’s own drift band.

That narrows the next credible target. The next `search_first` slice should stay on the same local screen surface and target a larger accepted-neighbor mutation cost than threshold caching alone.

The current shared generic/bitset kernel still uses:

- `generic_frontier: BinaryHeap<(SearchMinDist, usize)>`
- `generic_results: BinaryHeap<(SearchMaxDist, usize)>`

The frontier heap is structurally necessary for the current traversal shape. The bounded result heap is a different story: it exists to maintain the best `ef` accepted results and expose the current worst accepted distance, but it also forces heap `push/pop/peek` maintenance on nearly every accepted neighbor.

Task 15 therefore targets a narrow but higher-leverage cut: keep the shared traversal and frontier heap intact, but replace the shared-kernel result heap with a scratch-owned sorted result container that is cheaper to maintain on the deterministic bitset lane.

## Context

The current pure-Rust `search_first` sequence is now:

1. Task 9: unify generic and bitset layer-search onto one shared traversal core
2. Task 10: move generic frontier/results heaps into `SearchScratch`
3. Task 11: add a deterministic bitset diagnosis lane
4. Task 12: specialize the shared L2 distance path
5. Task 13: switch the shared L2 layer-0 neighbor source to `layer0_flat_graph`
6. Task 14: cache the current worst accepted result in `SearchScratch`

Task 13 produced clearly promotable local evidence. Task 14 did not; its qps movement was roughly `-0.11% .. +0.84%` while same-code rerun drift was roughly `-0.50% .. +0.78%`.

That means the next slice should not be another sub-1% heap micro-tuning pass. It should remove a larger unit of shared-kernel result-maintenance work while preserving the same measurement lane, semantics, and attribution discipline.

## Goals

- Reduce accepted-neighbor result-maintenance overhead inside `search_layer_idx_shared(...)`.
- Keep `generic_frontier` as the traversal frontier structure.
- Replace only the shared-kernel bounded result container.
- Preserve recall, visited-node counts, distance-call counts, and bitset semantics on the deterministic bitset lane.
- Expose the new shared-kernel result-container mode through the public generic-kernel report.
- Re-measure against the Task-14 canonical local bitset baseline and decide whether the slice is `promote`, `needs_more_local`, or `reject`.

## Non-Goals

- No changes to the layer-0 ordered-pool fast path.
- No changes to the upper-layer L2 heap path.
- No new diagnosis harness or new dataset lane.
- No graph-layout rewrite.
- No FFI, persistence, or serialization changes.
- No authority workflow reopen or family-verdict claims.

## Problem Statement

Today the shared generic/bitset kernel uses the result heap for three related jobs:

1. keep at most `ef` accepted results
2. expose the current worst accepted distance
3. emit nearest-first final results after traversal

That is correct, but `BinaryHeap<(SearchMaxDist, usize)>` is not obviously the cheapest structure for this job on the current shared-kernel workload:

- accepted-neighbor updates still cause heap mutation on the result side
- Task 14 only optimized around the heap, not away from it
- the shared kernel eventually drains and sorts the result heap anyway

The shared traversal does not need a general-purpose heap for results; it needs a bounded best-`ef` container with cheap worst-threshold access and stable nearest-first extraction.

## Proposed Approaches

### Option 1: Replace shared-kernel results with a scratch-owned sorted `Vec`

Keep `generic_frontier` as a heap, but replace `generic_results` with a sorted scratch vector for the shared generic/bitset kernel.

Properties:

- nearest-first order is maintained directly
- the current worst accepted result is the last element
- accepted-neighbor insertion becomes bounded insert/truncate rather than heap push/pop

Pros:

- strongest chance of producing a signal larger than Task 14
- still local to the shared-kernel result side
- keeps traversal shape intact

Cons:

- larger semantic surface than cached-threshold reuse
- needs careful regression coverage for result ordering and truncation behavior

### Option 2: Keep the heap and further compress mutation helpers

Keep `generic_results` as a heap and only squeeze more operations out of the accepted-neighbor path.

Pros:

- lower implementation risk

Cons:

- likely another sub-1% slice
- too similar to Task 14 in mechanism

### Option 3: Replace both frontier and results together in the shared kernel

Move the shared kernel to a more radical accepted-neighbor mutation design.

Pros:

- larger upside

Cons:

- too broad for one screen slice
- weaker attribution if results move
- higher risk of semantic regressions

## Recommendation

Choose Option 1.

Task 15 should only replace the bounded result container in the shared generic/bitset kernel. That keeps the slice large enough to matter but still narrow enough to attribute. The traversal frontier remains a heap, the public search APIs remain the same, and the deterministic bitset lane remains the only measurement surface.

## Design

### 1. Scratch-owned sorted result container

Extend `SearchScratch` with a result buffer dedicated to the shared generic/bitset kernel:

- sorted nearest-first by distance
- bounded to `ef`
- reused across calls like the existing scratch-owned structures

The current worst accepted distance becomes:

- `f32::INFINITY` when the buffer is empty
- the last distance in the sorted buffer otherwise

### 2. Accepted-neighbor insertion path

Replace the shared-kernel result-heap mutation path with a bounded insertion helper:

- if the result buffer has fewer than `ef` entries, insert the accepted neighbor at the correct sorted position
- if it already has `ef` entries, only insert when the new neighbor is better than the current worst
- after insertion, truncate the worst entry if the buffer grew past `ef`

This keeps the shared kernel’s threshold and final output path aligned with one structure instead of maintaining a heap that later gets drained and reversed.

### 3. Frontier stays unchanged

`generic_frontier` remains a `BinaryHeap<(SearchMinDist, usize)>`.

This is important for scope control:

- the traversal order stays the same
- only the result-maintenance side changes
- if the deterministic bitset lane moves, the attribution remains clean

### 4. Public contract surface

Extend `HnswGenericSearchKernelReport` with a new field such as:

- `shared_result_container_mode`

Expected value for Task 15:

- `sorted_scratch_vec`

This keeps the swap visible to default-lane contract tests in `tests/bench_hnsw_generic_search_kernel.rs`.

### 5. Semantics constraints

Task 15 must preserve:

- bitset filtering behavior
- traversal stop condition behavior
- nearest-first final result ordering
- visited-node behavior
- distance-call behavior

If recall or traversal counters move, the slice should be treated as a semantic change instead of a pure result-maintenance cost reduction.

## Data Flow

1. The shared kernel prepares scratch state for one layer-search call.
2. Entry insertion seeds the sorted result buffer when the entry is not filtered.
3. Each candidate pop reads the current worst accepted distance from the end of the buffer.
4. Each accepted neighbor is inserted into the sorted buffer only if it belongs in the bounded top-`ef`.
5. `generic_frontier` continues to receive accepted neighbors exactly as before.
6. Final results are read directly from the sorted result buffer without heap draining/reversal.
7. The deterministic bitset diagnosis lane compares qps against the Task-14 canonical baseline while checking recall and traversal counters.

## Verification Plan

Task 15 should require all of the following:

- a red/green contract update in `tests/bench_hnsw_generic_search_kernel.rs` for `shared_result_container_mode`
- focused library regressions covering:
  - shared generic layer-search ordering
  - shared bitset layer-search ordering
  - repeated shared-kernel scratch reuse across calls
  - bounded insertion/truncation semantics
- fresh deterministic bitset diagnosis rerun against the canonical Task-14 baseline
- one same-code `/tmp` rerun to compare measured delta against lane drift
- fresh `cargo test hnsw --lib -- --nocapture`
- fresh `cargo fmt --all -- --check`
- fresh `python3 scripts/validate_features.py feature-list.json`

## Success Criteria

Task 15 counts as a good screen slice only if:

- contract and focused regressions stay green
- recall remains unchanged on the deterministic bitset lane
- visited-node, distance-call, and frontier counters remain unchanged
- measured qps gain is materially larger than same-code rerun drift

If the result-container swap does not beat the lane’s own noise floor, it should be recorded as `screen_result=needs_more_local` rather than promoted.
