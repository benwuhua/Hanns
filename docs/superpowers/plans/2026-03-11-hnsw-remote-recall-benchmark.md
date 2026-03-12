# HNSW Remote Recall Benchmark Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the authority-backed HNSW recall-gated benchmark artifact after the layer-0 neighbor fix and record the updated Rust-vs-native evidence chain.

**Architecture:** Reuse the existing remote benchmark entrypoints instead of adding new surfaces. First regenerate the Rust HDF5 baseline artifact on the authority machine, then re-run the Rust comparison lane and native HNSW capture, and finally update the durable benchmark JSON/docs only if the authority outputs complete successfully.

**Tech Stack:** Rust 2021, existing benchmark binaries/tests, remote authority wrappers in `scripts/remote/test.sh` and `scripts/remote/native_hnsw_qps_capture.sh`, JSON artifacts under `benchmark_results/`

---

## Chunk 1: Authority Artifact Refresh

### Task 1: Rebuild the Rust HDF5 baseline artifact on authority

**Files:**
- Verify: `src/bin/generate_hdf5_hnsw_baseline.rs`
- Verify: `benchmark_results/rs_hnsw_sift128.local.json`

- [ ] **Step 1: Run authority Rust HDF5 baseline generation**

Run: `bash scripts/remote/test.sh --command "cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input /data/work/knowhere-native-src/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.full_k100.json --base-limit 1000000 --query-limit 1000 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95"`
Expected: `test=ok`

- [ ] **Step 2: Inspect the generated artifact path and payload**

Run: inspect `benchmark_results/rs_hnsw_sift128.local.json` and any log emitted by the authority run
Expected: updated Rust HDF5 row with current qps/recall metadata

## Chunk 2: Comparison Evidence

### Task 2: Re-run the Rust/native comparison lanes on authority

**Files:**
- Verify: `tests/bench_hnsw_cpp_compare.rs`
- Verify: `scripts/remote/native_hnsw_qps_capture.sh`
- Verify: `benchmark_results/native_hnsw_sift128.remote.json`

- [ ] **Step 1: Run the Rust HNSW comparison lane**

Run: `bash scripts/remote/native_hnsw_qps_capture.sh --gtest-filter Benchmark_float_qps.TEST_HNSW`
Expected: `exit_code=0`

- [ ] **Step 2: Validate the refreshed benchmark cross-references**

Run: `python3 -m unittest tests/test_baseline_methodology_lock.py`
Expected: `OK`

- [ ] **Step 3: Inspect updated authority evidence**

Run: inspect the resulting authority logs and benchmark JSON inputs consumed by the baseline docs
Expected: clear current Rust/native evidence for the post-fix HNSW lane

## Chunk 3: Durable Refresh

### Task 3: Update artifact docs and feature state if authority evidence is complete

**Files:**
- Modify: `benchmark_results/rs_hnsw_sift128.local.json`
- Modify: `benchmark_results/hnsw_p1_001_after.remote.json`
- Modify: `benchmark_results/hnsw_p1_001_compare.json`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `memory/CURRENT_WORK_ORDER.json`

- [ ] **Step 1: Refresh benchmark artifacts from authority evidence**

Run: update the benchmark JSON files above only with values backed by the new authority logs
Expected: before/after comparison reflects the post-layer0-fix HNSW lane

- [ ] **Step 2: Refresh durable narrative**

Run: update `TASK_QUEUE.md`, `GAP_ANALYSIS.md`, `task-progress.md`, `RELEASE_NOTES.md`, and `memory/CURRENT_WORK_ORDER.json`
Expected: the new authority verdict and next feature are recorded

- [ ] **Step 3: Validate and verify**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`
