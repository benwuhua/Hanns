# HNSW Fairness Gate Batch Dispatch Screen Design

## Context

The fair-lane authority rerun closed the effective-`ef` mismatch, but the current HNSW fairness gate still fails because Rust records:

- `query_dispatch_model = "serial_per_query_index_search"`
- `query_batch_size = 1`

This makes the Rust same-schema lane structurally different from native `Search()`, so the current qps gap cannot be treated as fair leadership evidence.

The next step should keep scope narrow. We do not want to reopen HNSW algorithm internals or start datatype work in the same slice. The cheapest next screen is to change only the Rust HDF5 baseline harness so it can execute queries in batch-first parallel mode and report that mode in its artifact metadata.

## Options Considered

### Option 1: Add batch-first dispatch to the HDF5 baseline binary only

Implement an explicit query-dispatch mode in `src/bin/generate_hdf5_hnsw_baseline.rs`, keeping HNSW search itself unchanged. The binary gains a non-serial query batching mode backed by `rayon`, plus metadata that records the chosen dispatch model and batch size.

Pros:

- Smallest change that directly addresses the fairness blocker
- No API or FFI surface expansion
- Easy to A/B compare against current serial mode on the same local machine
- Produces an authority-ready command line once the local screen promotes

Cons:

- This is harness-level parallelism, not a product-level batch API
- Dispatch parity improves, but datatype parity still remains open

### Option 2: Add a real batch-search API to `HnswIndex`

Introduce a new Rust-side batch search method and route the baseline binary through it.

Pros:

- Cleaner long-term API shape
- Could be reused outside the benchmark binary

Cons:

- Larger scope and more risk
- Pulls benchmark fairness work into library/API design
- Harder to isolate whether regressions come from API changes or dispatch changes

### Option 3: Tackle datatype and dispatch together

Attempt to change both query dispatch and datatype in one slice.

Pros:

- Moves two fairness blockers at once

Cons:

- Blurs attribution
- Harder to debug and harder to review
- Unnecessarily expensive before we know the dispatch change is sound

## Recommendation

Choose Option 1.

This keeps the fair-lane work aligned with the program rule: remove one blocker at a time, keep the evidence chain trustworthy, and use local `screen` work to validate the direction before another authority rerun.

## Proposed Design

### 1. Explicit query-dispatch configuration in the HDF5 baseline binary

Add a small dispatch configuration to `src/bin/generate_hdf5_hnsw_baseline.rs`:

- `serial`
- `rayon_query_batch_parallel_search`

The binary should continue to default to `serial` so existing authority artifacts are not silently redefined. A new CLI flag pair should opt into the batch-first lane:

- `--query-dispatch-mode parallel`
- `--query-batch-size <n>`

The dispatch metadata written into the Rust baseline artifact should come from this configuration rather than from hard-coded strings.

### 2. Batch-first parallel search at the harness level

When parallel mode is selected, query execution should:

- partition `query_vectors` into ordered chunks of `query_batch_size * dim`
- process chunks with `rayon::ParallelSlice::par_chunks`
- execute each query inside a chunk using the existing `index.search(...)`
- preserve overall query-result order when collecting results

This keeps the search semantics unchanged while moving the benchmark lane away from per-query serial dispatch.

### 3. TDD contract

The first failing test should prove two things:

- the non-serial dispatch mode reports non-serial artifact metadata
- parallel batch execution returns the same ordered result IDs as the current serial path on a small deterministic fixture

This should live alongside the existing unit tests in `src/bin/generate_hdf5_hnsw_baseline.rs` so the harness behavior is locked locally without touching the current authority artifacts.

### 4. Screen decision rule

This slice remains a local `screen`, not authority work.

The local screen should compare:

- current serial mode
- new parallel batch mode

on the same HDF5 dataset/config. The screen promotes if:

- artifact metadata flips to non-serial batch mode as designed
- recall remains unchanged at the configured gate
- local qps does not show an obvious structural regression versus serial mode

If the mode works but the local qps signal is too noisy or flat, record `screen_result=needs_more_local`. If it clearly regresses, record `screen_result=reject`.

## Files In Scope

- `src/bin/generate_hdf5_hnsw_baseline.rs`
- `task-progress.md`
- `docs/superpowers/plans/2026-03-14-hnsw-fairness-gate-batch-dispatch-screen.md`

Potential local-only screen outputs:

- `/tmp/hnsw_fairness_dispatch_serial.json`
- `/tmp/hnsw_fairness_dispatch_parallel.json`

## Non-Goals

- No datatype changes
- No remote authority artifact refresh in this slice
- No `HnswIndex` public API expansion
- No FFI or Python/JNI changes
