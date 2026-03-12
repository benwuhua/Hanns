# IVF-PQ Stop-Go Verdict Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the IVF-PQ family with a durable `no-go` verdict artifact and replace the current false-green benchmark verification shells with real default-lane regressions.

**Architecture:** Follow the established HNSW family-verdict pattern. First, add focused failing tests that prove the three recorded verification entrypoints are still default-lane shells or otherwise fail to lock the IVF-PQ family conclusion. Then add a final verdict artifact plus minimal default-lane artifact/verdict regressions, while preserving the heavy benchmark generators behind `feature = "long-tests"` and `#[ignore]`. Finish by syncing durable project state and re-running the recorded remote authority checks.

**Tech Stack:** Rust 2021 integration tests in `tests/`, JSON benchmark artifacts in `benchmark_results/`, durable workflow files (`feature-list.json`, `task-progress.md`, `RELEASE_NOTES.md`, `TASK_QUEUE.md`, `GAP_ANALYSIS.md`, `docs/PARITY_AUDIT.md`), remote authority wrapper `scripts/remote/test.sh`

---

## Chunk 1: Lock The Verdict Contract With Failing Tests

### Task 1: Add default-lane regressions for the IVF-PQ family verdict and artifact facts

**Files:**
- Modify: `tests/bench_ivf_pq_perf.rs`
- Modify: `tests/bench_recall_gated_baseline.rs`
- Modify: `tests/bench_cross_dataset_sampling.rs`

- [x] **Step 1: Write the failing family-verdict regression**

Add a default-lane test in `tests/bench_ivf_pq_perf.rs` that expects a final IVF-PQ verdict artifact declaring `classification = no-go` and `leadership_claim_allowed = false`.

- [x] **Step 2: Write the failing baseline-artifact regression**

Add a default-lane test in `tests/bench_recall_gated_baseline.rs` that loads the existing baseline artifact and asserts the IVF-PQ row exists, stays below the recall gate, and is not `trusted`.

- [x] **Step 3: Write the failing cross-dataset regression**

Add a default-lane test in `tests/bench_cross_dataset_sampling.rs` that asserts every IVF-PQ row in the cross-dataset artifact remains below the recall gate or otherwise non-trusted.

- [x] **Step 4: Run the focused tests to verify RED**

Run: `cargo test --test bench_ivf_pq_perf -- --nocapture`
Expected: FAIL because the final verdict artifact does not exist yet.

Run: `cargo test --test bench_recall_gated_baseline -- --nocapture`
Expected: PASS or FAIL only after the new assertions land; either way the file must now execute real default-lane tests.

Run: `cargo test --test bench_cross_dataset_sampling -- --nocapture`
Expected: PASS or FAIL only after the new assertions land; either way the file must now execute real default-lane tests.

## Chunk 2: Add The Verdict Artifact And Keep Long Tests Isolated

### Task 2: Implement the minimal artifact/verdict closure

**Files:**
- Create: `benchmark_results/ivfpq_p3_003_final_verdict.json`
- Modify: `tests/bench_ivf_pq_perf.rs`
- Modify: `tests/bench_recall_gated_baseline.rs`
- Modify: `tests/bench_cross_dataset_sampling.rs`

- [x] **Step 1: Add the final IVF-PQ verdict artifact**

Create `benchmark_results/ivfpq_p3_003_final_verdict.json` by summarizing the current trusted authority evidence and the already-closed contract surface.

- [x] **Step 2: Preserve the heavy benchmark generators behind `long-tests`**

Refactor the three test files so the new default-lane regressions coexist with the existing heavy artifact-generation paths without changing long-test semantics.

- [x] **Step 3: Run the focused tests to verify GREEN**

Run: `cargo test --test bench_ivf_pq_perf -- --nocapture`
Expected: PASS

Run: `cargo test --test bench_recall_gated_baseline -- --nocapture`
Expected: PASS

Run: `cargo test --test bench_cross_dataset_sampling -- --nocapture`
Expected: PASS

## Chunk 3: Re-run Recorded Verification And Sync Durable State

### Task 3: Close the feature honestly in the long-task workflow

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `docs/PARITY_AUDIT.md`

- [x] **Step 1: Run local feature verification**

Run: `cargo test --test bench_ivf_pq_perf -q`
Expected: PASS

Run: `cargo test --test bench_recall_gated_baseline -q`
Expected: PASS

Run: `cargo test --test bench_cross_dataset_sampling -q`
Expected: PASS

- [x] **Step 2: Run remote authority verification**

Run: `bash init.sh`
Expected: PASS

Run: `bash scripts/remote/test.sh --command "cargo test --test bench_ivf_pq_perf -q"`
Expected: `test=ok`

Run: `bash scripts/remote/test.sh --command "cargo test --test bench_recall_gated_baseline -q"`
Expected: `test=ok`

Run: `bash scripts/remote/test.sh --command "cargo test --test bench_cross_dataset_sampling -q"`
Expected: `test=ok`

- [x] **Step 3: Update durable state**

Mark `ivfpq-stop-go-verdict` as `passing`, record the `no-go` classification and authority evidence in the durable docs, set the next feature handoff, and re-run `python3 scripts/validate_features.py feature-list.json`.
