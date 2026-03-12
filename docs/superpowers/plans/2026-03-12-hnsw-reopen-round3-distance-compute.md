# HNSW Reopen Round 3 Distance-Compute Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reopen HNSW for a third focused algorithm line and judge a distance-compute optimization cut by fresh remote same-schema recall-gated QPS evidence.

**Architecture:** Keep the historical HNSW verdict chain intact, add round-3-specific reopen artifacts, and constrain implementation changes to the hot L2 search path in `src/faiss/hnsw.rs` plus the durable workflow files that describe and verify the new line. The first chunk only activates round 3; later chunks deepen profiling, rework the L2 inner loop, and rerun the authority lane.

**Tech Stack:** Rust 2021, cargo tests, remote x86 authority wrappers, JSON benchmark artifacts, durable markdown/json workflow state.

---

## Chunk 1: Activate Round 3

### Task 1: Add a failing round-3 activation regression

**Files:**
- Create: `tests/bench_hnsw_reopen_round3.rs`

- [ ] **Step 1: Write the failing test**

Write a default-lane regression that loads `benchmark_results/hnsw_reopen_round3_baseline.json` and asserts:

- `task_id == "HNSW-REOPEN-ROUND3-BASELINE"`
- `family == "HNSW"`
- `authority_scope == "remote_x86_only"`
- `historical_verdict_source == "benchmark_results/hnsw_p3_002_final_verdict.json"`
- `round2_authority_summary_source == "benchmark_results/hnsw_reopen_round2_authority_summary.json"`
- `round3_target == "distance_compute_inner_loop"`
- the summary mentions the unchanged `functional-but-not-leading` historical verdict

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round3 -- --nocapture`  
Expected: FAIL because `benchmark_results/hnsw_reopen_round3_baseline.json` does not exist yet.

### Task 2: Add the round-3 baseline artifact and durable state

**Files:**
- Create: `benchmark_results/hnsw_reopen_round3_baseline.json`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Create the round-3 baseline artifact**

Write `benchmark_results/hnsw_reopen_round3_baseline.json` with:

- the round-2 authority same-schema metrics copied from `benchmark_results/hnsw_reopen_round2_authority_summary.json`
- references to the historical HNSW verdict and round-2 artifacts
- `round3_target` set to `distance_compute_inner_loop`
- a summary explaining that round 2 improved synthetic signals but ended in a hard stop because authority QPS regressed

- [ ] **Step 2: Extend durable workflow state**

Add four new failing features to `feature-list.json`:

- `hnsw-reopen-round3-activation`
- `hnsw-distance-compute-profiler`
- `hnsw-distance-l2-fast-path-rework`
- `hnsw-round3-authority-same-schema-rerun`

Mark only `hnsw-reopen-round3-activation` as the active feature for this chunk.

- [ ] **Step 3: Update progress and governance docs**

Update `task-progress.md`, `TASK_QUEUE.md`, `GAP_ANALYSIS.md`, `DEV_ROADMAP.md`, `RELEASE_NOTES.md`, and `docs/PARITY_AUDIT.md` so they describe:

- round 2 as closed with a hard stop
- round 3 as active
- `distance_compute_inner_loop` as the next explicit HNSW hypothesis

- [ ] **Step 4: Run the regression to verify it passes**

Run: `cargo test --test bench_hnsw_reopen_round3 -- --nocapture`  
Expected: PASS

- [ ] **Step 5: Refresh remote sync and authority replay**

Run: `bash init.sh`  
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round3 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round3 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round3 -q"`  
Expected: `test=ok`

- [ ] **Step 6: Validate durable state and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add tests/bench_hnsw_reopen_round3.rs benchmark_results/hnsw_reopen_round3_baseline.json feature-list.json task-progress.md TASK_QUEUE.md GAP_ANALYSIS.md DEV_ROADMAP.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): activate reopen round 3"
```

## Chunk 2: Profile Distance Compute More Precisely

### Task 3: Add a failing round-3 distance-compute contract

**Files:**
- Modify: `tests/bench_hnsw_reopen_round3.rs`
- Create: `tests/bench_hnsw_reopen_round3_profile.rs`

- [ ] **Step 1: Tighten the round-3 contract**

Extend `tests/bench_hnsw_reopen_round3.rs` so it also requires `benchmark_results/hnsw_reopen_distance_compute_profile_round3.json` with buckets and counts for:

- `upper_layer_query_distance_ms`
- `layer0_query_distance_ms`
- `node_node_distance_ms`
- `upper_layer_query_distance_calls`
- `layer0_query_distance_calls`
- `node_node_distance_calls`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round3 -- --nocapture`  
Expected: FAIL because the new profile artifact does not exist yet.

### Task 4: Implement the round-3 profiler

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Create: `benchmark_results/hnsw_reopen_distance_compute_profile_round3.json`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Add profiling support in `src/faiss/hnsw.rs`**

Instrument the HNSW search path so the profiler can separate:

- upper-layer greedy `query -> node` distance work
- layer-0 candidate-expansion `query -> node` distance work
- `node -> node` distance work
- residual `visited/frontier` bookkeeping still paired with those loops

- [ ] **Step 2: Add the long-test generator**

Create `tests/bench_hnsw_reopen_round3_profile.rs` to run the profiler and emit `benchmark_results/hnsw_reopen_distance_compute_profile_round3.json`.

- [ ] **Step 3: Generate the artifact locally**

Run: `cargo test --features long-tests --test bench_hnsw_reopen_round3_profile -- --ignored --nocapture`  
Expected: PASS and writes `benchmark_results/hnsw_reopen_distance_compute_profile_round3.json`

- [ ] **Step 4: Re-run the contract test**

Run: `cargo test --test bench_hnsw_reopen_round3 -- --nocapture`  
Expected: PASS

- [ ] **Step 5: Authority replay**

Run: `bash init.sh`  
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round3 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round3 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round3_profile -- --ignored --nocapture"`  
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round3 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round3 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round3 -q"`  
Expected: `test=ok`

- [ ] **Step 6: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_reopen_round3.rs tests/bench_hnsw_reopen_round3_profile.rs benchmark_results/hnsw_reopen_distance_compute_profile_round3.json feature-list.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): profile round 3 distance compute"
```

## Chunk 3: Rework the L2 Distance Hot Path

### Task 5: Add focused L2 fast-path regressions

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add failing focused regressions**

Add targeted tests proving:

- the new L2 fast path returns the same ordering and distances as the current generic path on deterministic fixtures
- upper-layer greedy descent is unchanged for `L2 + no filter`
- layer-0 candidate expansion remains semantically aligned with the generic path
- filter-bearing searches still go through the generic logic and are not silently rerouted

- [ ] **Step 2: Run focused library tests to verify failure**

Run: `cargo test hnsw --lib -- --nocapture`  
Expected: FAIL on the new focused L2 regressions.

### Task 6: Implement the L2 fast-path rework

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Implement the minimal hot-path changes**

Rework the L2 search path with:

- one-time outer dispatch for `L2 + no filter`
- specialized internal helpers for L2 `query -> node` distance computation
- tighter upper-layer and layer-0 hot loops that avoid repeated generic-path branching

Do not change build semantics, storage format, public API, or FFI.

- [ ] **Step 2: Re-run focused library tests**

Run: `cargo test hnsw --lib -- --nocapture`  
Expected: PASS

- [ ] **Step 3: Re-run round-3 and historical safety contracts**

Run: `cargo test --test bench_hnsw_cpp_compare -q`  
Expected: PASS

Run: `cargo test --test bench_hnsw_reopen_round2 -q`  
Expected: PASS

Run: `cargo test --test bench_hnsw_reopen_round3 -q`  
Expected: PASS

- [ ] **Step 4: Refresh the round-3 synthetic profile locally**

Run: `cargo test --features long-tests --test bench_hnsw_reopen_round3_profile -- --ignored --nocapture`  
Expected: PASS and refreshed `benchmark_results/hnsw_reopen_distance_compute_profile_round3.json`

- [ ] **Step 5: Authority synthetic replay**

Run: `bash init.sh`  
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round3 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round3 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_cpp_compare -q"`  
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round3 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round3 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round3_profile -- --ignored --nocapture"`  
Expected: `test=ok`

- [ ] **Step 6: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_reopen_round3.rs tests/bench_hnsw_reopen_round3_profile.rs benchmark_results/hnsw_reopen_distance_compute_profile_round3.json feature-list.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): rework round 3 L2 distance path"
```

## Chunk 4: Refresh the Real Authority Lane

### Task 7: Re-run same-schema authority and summarize round 3

**Files:**
- Create: `benchmark_results/hnsw_reopen_round3_authority_summary.json`
- Modify: `tests/bench_hnsw_reopen_round3.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Tighten the round-3 contract again**

Extend `tests/bench_hnsw_reopen_round3.rs` so it requires `benchmark_results/hnsw_reopen_round3_authority_summary.json` and asserts:

- the recorded round-3 target is `distance_compute_inner_loop`
- the summary includes recall/QPS deltas versus `benchmark_results/hnsw_reopen_round3_baseline.json`
- the artifact explicitly states whether a later verdict-refresh feature is justified
- the artifact explicitly states whether the next action is `continue`, `soft_stop`, or `hard_stop`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round3 -- --nocapture`  
Expected: FAIL because the authority summary artifact does not exist yet.

- [ ] **Step 3: Refresh remote sync**

Run: `bash init.sh`  
Expected: PASS

- [ ] **Step 4: Re-run the authoritative same-schema lane**

Run the recorded remote same-schema HNSW benchmark command needed to regenerate `benchmark_results/rs_hnsw_sift128.full_k100.json`. Save the resulting log path in the round-3 summary artifact.

- [ ] **Step 5: Refresh native and synthetic evidence**

Run: `bash scripts/remote/native_hnsw_qps_capture.sh --log-dir /data/work/knowhere-rs-logs-hnsw-reopen-round3 --gtest-filter Benchmark_float_qps.TEST_HNSW`  
Expected: `exit_code=0`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round3 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round3 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round3_profile -- --ignored --nocapture"`  
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round3 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round3 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round3 -q"`  
Expected: `test=ok`

- [ ] **Step 6: Write the round-3 authority summary**

Create `benchmark_results/hnsw_reopen_round3_authority_summary.json` with:

- fresh same-schema Rust and native recall/QPS results
- deltas versus `benchmark_results/hnsw_reopen_round3_baseline.json`
- references to the round-3 profile artifact and authority logs
- `verdict_refresh_allowed` as `true` or `false`
- `next_action` as `continue`, `soft_stop`, or `hard_stop`

Keep the historical HNSW family verdict unchanged in this chunk even if `verdict_refresh_allowed == true`; only record whether a later verdict-refresh feature is justified.

- [ ] **Step 7: Re-run the contract test**

Run: `cargo test --test bench_hnsw_reopen_round3 -- --nocapture`  
Expected: PASS

- [ ] **Step 8: Update durable workflow state**

If `verdict_refresh_allowed == true`:

- keep the historical family verdict unchanged
- record that the next honest tracked feature is a verdict-refresh follow-up

If `verdict_refresh_allowed == false`:

- keep the historical family verdict intact
- record whether round 3 ends in `soft_stop` or `hard_stop`

- [ ] **Step 9: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add tests/bench_hnsw_reopen_round3.rs benchmark_results/hnsw_reopen_round3_authority_summary.json benchmark_results/hnsw_reopen_distance_compute_profile_round3.json benchmark_results/rs_hnsw_sift128.full_k100.json feature-list.json task-progress.md TASK_QUEUE.md GAP_ANALYSIS.md DEV_ROADMAP.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "test(hnsw): refresh round 3 authority summary"
```
