# Final Performance Leadership Proof Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive an explicit remote-only final-proof artifact that records the project has not yet proven any core CPU path leads native on the same authority benchmark lane.

**Architecture:** Keep the change narrow and replayable. Add one final-acceptance artifact in `benchmark_results/`, then extend the existing default-lane HNSW compare regression so it locks that artifact to the already-closed HNSW stop-go evidence and the cross-family rollup artifact. Do not regenerate heavy benchmark JSON locally or reopen family verdicts.

**Tech Stack:** Rust 2021 integration tests in `tests/`, JSON verdict artifacts in `benchmark_results/`, durable workflow state in repo-root markdown/JSON, remote authority wrappers in `scripts/remote/`

---

## Chunk 1: Add The Failing Final-Proof Regression

### Task 1: Make the HNSW compare lane fail for the missing final-proof artifact

**Files:**
- Modify: `tests/bench_hnsw_cpp_compare.rs`

- [ ] **Step 1: Add a default-lane final-proof regression**

Extend `tests/bench_hnsw_cpp_compare.rs` with a new test that loads `benchmark_results/final_performance_leadership_proof.json` and asserts:

- `criterion_met` is `false`
- the artifact references `benchmark_results/baseline_p3_001_stop_go_verdict.json`
- the artifact references `benchmark_results/final_core_path_classification.json`
- HNSW remains the trusted blocker for leadership proof
- IVF-PQ and DiskANN remain non-leadership families

- [ ] **Step 2: Run the compare lane to verify RED**

Run: `cargo test --test bench_hnsw_cpp_compare -- --nocapture`
Expected: FAIL because `benchmark_results/final_performance_leadership_proof.json` does not exist yet.

## Chunk 2: Add The Artifact And Reach GREEN

### Task 2: Implement the minimum final-proof closure

**Files:**
- Create: `benchmark_results/final_performance_leadership_proof.json`
- Modify: `tests/bench_hnsw_cpp_compare.rs`

- [ ] **Step 1: Create the final-proof artifact**

Add `benchmark_results/final_performance_leadership_proof.json` with:

- `criterion_met = false`
- references to the current HNSW stop-go artifact, HNSW family verdict artifact, and final core path rollup artifact
- one row each for HNSW, IVF-PQ, and DiskANN describing why none of them satisfy the performance-leadership criterion

- [ ] **Step 2: Keep the regression narrow**

Assert only the fields needed to prove the current completion criterion is unmet and tied to the existing authority evidence chain.

- [ ] **Step 3: Run the compare lane to verify GREEN**

Run: `cargo test --test bench_hnsw_cpp_compare -- --nocapture`
Expected: PASS

## Chunk 3: Run Authority Verification And Persist Durable State

### Task 3: Close the feature honestly in the long-task workflow

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Run the recorded local verification command**

Run: `cargo test --test bench_hnsw_cpp_compare -q`
Expected: PASS

- [ ] **Step 2: Refresh remote authority sync**

Run: `bash init.sh`
Expected: PASS

- [ ] **Step 3: Run the recorded remote verification commands**

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-final-performance-leadership KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-final-performance-leadership bash scripts/remote/native_hnsw_qps_capture.sh --gtest-filter Benchmark_float_qps.TEST_HNSW`
Expected: native benchmark capture completes and writes an authority log under `/data/work/knowhere-rs-logs-final-performance-leadership/`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-final-performance-leadership KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-final-performance-leadership bash scripts/remote/test.sh --command "cargo build --features hdf5 --bin generate_hdf5_hnsw_baseline --verbose"`
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-final-performance-leadership KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-final-performance-leadership bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_cpp_compare -q"`
Expected: `test=ok`

- [ ] **Step 4: Update durable workflow state and validate**

Mark `final-performance-leadership-proof` as `passing`, record the final-proof artifact and fresh authority logs in durable docs, select the next ready feature, and run:

`python3 scripts/validate_features.py feature-list.json`

Expected: `VALID`
