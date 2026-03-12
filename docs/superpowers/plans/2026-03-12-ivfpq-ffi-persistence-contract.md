# IVF-PQ FFI Persistence Contract Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the IVF-PQ production-facing contract gap by making FFI file save/load match advertised metadata and by turning the JSON export verification into a real default-lane regression.

**Architecture:** Keep the change narrowly scoped. First, add focused failing regressions that prove IVF-PQ metadata overstates file persistence support and that `bench_json_export` is still a no-op on the default lane. Then wire the existing `src/faiss/ivfpq.rs` save/load implementation through `src/ffi.rs`, and replace the file-level `long-tests` shell in `tests/bench_json_export.rs` with a fast default-lane contract test while preserving the heavy benchmark path behind `long-tests` / `#[ignore]`.

**Tech Stack:** Rust 2021, existing FFI contract tests in `src/ffi.rs`, IVF trait tests in `tests/test_ivf_index_trait.rs`, benchmark JSON export test harness in `tests/bench_json_export.rs`, remote authority wrappers in `scripts/remote/test.sh`

---

## Chunk 1: FFI Persistence Red Test

### Task 1: Prove IVF-PQ metadata and FFI persistence behavior disagree today

**Files:**
- Modify: `src/ffi.rs`

- [x] **Step 1: Write the failing test**

Add a focused regression in `src/ffi.rs` that creates an IVF-PQ index through the C ABI, trains/adds vectors, saves to a temp file, loads into a fresh IVF-PQ handle, and asserts the load succeeds plus the restored count is correct.

- [x] **Step 2: Run test to verify it fails**

Run: `cargo test --lib test_ffi_persistence_ivfpq_file_roundtrip -- --nocapture`
Expected: FAIL because `IndexWrapper::save/load` does not currently route IVF-PQ.

## Chunk 2: FFI Persistence Green

### Task 2: Route the existing IVF-PQ file persistence through the FFI wrapper

**Files:**
- Modify: `src/ffi.rs`
- Verify: `src/faiss/ivfpq.rs`

- [x] **Step 1: Write the minimal implementation**

Extend `IndexWrapper::save` and `IndexWrapper::load` to delegate to `IvfPqIndex::save` / `IvfPqIndex::load`, keeping metadata semantics unchanged because they already advertise the intended contract.

- [x] **Step 2: Run focused tests to verify green**

Run: `cargo test --lib test_ffi_persistence_ivfpq_file_roundtrip -- --nocapture`
Expected: PASS

Run: `cargo test --lib test_ffi_abi_metadata_contract -- --nocapture`
Expected: PASS

## Chunk 3: Default-Lane JSON Export Contract

### Task 3: Replace the `0 tests` shell with a fast contract regression

**Files:**
- Modify: `tests/bench_json_export.rs`

- [x] **Step 1: Write the failing default-lane regression**

Add a fast, non-ignored test that exercises the JSON export path with a tiny deterministic dataset and asserts the emitted JSON has stable structure plus the expected index names. Preserve the existing heavy benchmark path behind `feature = "long-tests"` and `#[ignore]`.

- [x] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_json_export -- --nocapture`
Expected: FAIL until the fast default-lane regression exists without the file-level `long-tests` gate.

- [x] **Step 3: Write the minimal implementation**

Refactor only enough test helper code so both the fast regression and the long benchmark path can share JSON assembly logic without changing benchmark semantics.

- [x] **Step 4: Run test to verify it passes**

Run: `cargo test --test bench_json_export -- --nocapture`
Expected: PASS

## Chunk 4: Authority Verification And Durable State

### Task 4: Re-run the feature verification and persist status

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`

- [x] **Step 1: Run local verification**

Run: `cargo test --lib ffi -- --nocapture`
Expected: PASS

Run: `cargo test --test test_ivf_index_trait -- --nocapture`
Expected: PASS

Run: `cargo test --test bench_json_export -- --nocapture`
Expected: PASS

- [x] **Step 2: Run authority verification**

Run: `bash scripts/remote/test.sh --command "cargo test --lib ffi -- --nocapture"`
Expected: `test=ok`

Run: `bash scripts/remote/test.sh --command "cargo test --test test_ivf_index_trait -- --nocapture"`
Expected: `test=ok`

Run: `bash scripts/remote/test.sh --command "cargo test --test bench_json_export -- --nocapture"`
Expected: `test=ok`

- [x] **Step 3: Update durable state**

Mark `ivfpq-ffi-persistence-contract` as passing, record the evidence in `task-progress.md` and `RELEASE_NOTES.md`, then run `python3 scripts/validate_features.py feature-list.json`.
