# HNSW Round 4 Layer-0 Searcher Parity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reopen HNSW for a fourth focused algorithm line and judge whether a native-like layer-0 search core can materially improve the remote same-schema recall-gated Rust HNSW lane.

**Architecture:** Keep the historical HNSW verdict chain intact, add round-4-specific reopen artifacts, and constrain implementation changes to the `L2 + no-filter` search path in `src/faiss/hnsw.rs` plus a small SIMD helper if needed. The round is split into activation, audit, core rework, and authority rerun so each stage either proves the hypothesis or narrows the next cut.

**Tech Stack:** Rust 2021, cargo tests, remote x86 authority wrappers, JSON benchmark artifacts, durable markdown/json workflow state.

---

## Chunk 1: Activate Round 4

### Task 1: Add a failing round-4 activation regression

**Files:**
- Create: `tests/bench_hnsw_reopen_round4.rs`

- [ ] **Step 1: Write the failing test**

Write a default-lane regression that loads `benchmark_results/hnsw_reopen_round4_baseline.json` and asserts:

- `task_id == "HNSW-REOPEN-ROUND4-BASELINE"`
- `family == "HNSW"`
- `authority_scope == "remote_x86_only"`
- `historical_verdict_source == "benchmark_results/hnsw_p3_002_final_verdict.json"`
- `round3_authority_summary_source == "benchmark_results/hnsw_reopen_round3_authority_summary.json"`
- `round4_target == "layer0_searcher_parity"`
- the summary mentions the unchanged `functional-but-not-leading` historical verdict

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round4 -- --nocapture`  
Expected: FAIL because `benchmark_results/hnsw_reopen_round4_baseline.json` does not exist yet.

### Task 2: Add the round-4 baseline artifact and durable state

**Files:**
- Create: `benchmark_results/hnsw_reopen_round4_baseline.json`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Create the round-4 baseline artifact**

Write `benchmark_results/hnsw_reopen_round4_baseline.json` with:

- the round-3 authority same-schema metrics copied from `benchmark_results/hnsw_reopen_round3_authority_summary.json`
- references to the historical HNSW verdict and round-3 artifacts
- `round4_target` set to `layer0_searcher_parity`
- a summary explaining that round 3 improved the authority row but ended in a soft stop because the native gap remained too large

- [ ] **Step 2: Extend durable workflow state**

Add four new failing features to `feature-list.json`:

- `hnsw-reopen-round4-activation`
- `hnsw-layer0-searcher-audit`
- `hnsw-layer0-searcher-core-rework`
- `hnsw-round4-authority-same-schema-rerun`

Mark only `hnsw-reopen-round4-activation` as the active feature for this chunk.

- [ ] **Step 3: Update progress and governance docs**

Update `task-progress.md`, `TASK_QUEUE.md`, `GAP_ANALYSIS.md`, `DEV_ROADMAP.md`, `RELEASE_NOTES.md`, and `docs/PARITY_AUDIT.md` so they describe:

- round 3 as closed with a soft stop
- round 4 as active
- `layer0_searcher_parity` as the next explicit HNSW hypothesis

- [ ] **Step 4: Run the regression to verify it passes**

Run: `cargo test --test bench_hnsw_reopen_round4 -- --nocapture`  
Expected: PASS

- [ ] **Step 5: Refresh remote sync and authority replay**

Run: `bash init.sh`  
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round4 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round4 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round4 -q"`  
Expected: `test=ok`

- [ ] **Step 6: Validate durable state and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add tests/bench_hnsw_reopen_round4.rs benchmark_results/hnsw_reopen_round4_baseline.json feature-list.json task-progress.md TASK_QUEUE.md GAP_ANALYSIS.md DEV_ROADMAP.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): activate reopen round 4"
```

## Chunk 2: Audit Layer-0 Searcher Parity

### Task 3: Add a failing round-4 audit contract

**Files:**
- Modify: `tests/bench_hnsw_reopen_round4.rs`
- Create: `tests/bench_hnsw_reopen_round4_profile.rs`

- [ ] **Step 1: Tighten the round-4 contract**

Extend `tests/bench_hnsw_reopen_round4.rs` so it also requires `benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json` with:

- `native_reference_files`
- `rust_reference_files`
- `search_core_shape`
- `batch_distance_mode`
- `distance_compute_breakdown`
- `distance_compute_call_counts`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round4 -- --nocapture`  
Expected: FAIL because the audit/profile artifact does not exist yet.

### Task 4: Implement the round-4 layer-0 audit

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Create: `benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json`
- Create: `tests/bench_hnsw_reopen_round4_profile.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Extend profiling support in `src/faiss/hnsw.rs`**

Instrument the HNSW search path so the round-4 audit can report:

- layer-0 `query -> node` distance timing and call counts
- whether batch-4 distance evaluation is enabled
- enough metadata to distinguish heap-based versus ordered-pool search core behavior

- [ ] **Step 2: Add the long-test generator**

Create `tests/bench_hnsw_reopen_round4_profile.rs` to generate `benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json`.

- [ ] **Step 3: Generate the artifact locally**

Run: `cargo test --features long-tests --test bench_hnsw_reopen_round4_profile -- --ignored --nocapture`  
Expected: PASS and writes `benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json`

- [ ] **Step 4: Re-run the contract test**

Run: `cargo test --test bench_hnsw_reopen_round4 -- --nocapture`  
Expected: PASS

- [ ] **Step 5: Authority replay**

Run: `bash init.sh`  
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round4 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round4 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round4_profile -- --ignored --nocapture"`  
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round4 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round4 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round4 -q"`  
Expected: `test=ok`

- [ ] **Step 6: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_reopen_round4.rs tests/bench_hnsw_reopen_round4_profile.rs benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json feature-list.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): audit round 4 layer0 searcher"
```

## Chunk 3: Rework the Layer-0 Search Core

### Task 5: Add focused round-4 regressions

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add failing focused regressions**

Add targeted tests proving:

- the new `L2 + no-filter` layer-0 search core returns the same ordering and distances as the current path on deterministic fixtures
- filter-bearing searches still use the generic logic and are not silently rerouted
- the ordered candidate structure preserves correct stop conditions and result capacity behavior
- the new batch-4 helper matches scalar `query -> node` distance results on deterministic inputs

- [ ] **Step 2: Run focused library tests to verify failure**

Run: `cargo test hnsw --lib -- --nocapture`  
Expected: FAIL on the new round-4 focused regressions.

### Task 6: Implement the round-4 layer-0 search-core rework

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Modify: `src/simd.rs` (only if a new batch-4 helper is needed)
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Implement the minimal search-core changes**

Rework the `L2 + no-filter` layer-0 search path with:

- a scratch-owned fixed-capacity ordered candidate/result structure
- query-scoped reuse of the layer-0 frontier/result buffers
- a batch-4 `query -> node` L2 helper for neighbor expansion

Do not change build semantics, storage format, public API, or FFI.

- [ ] **Step 2: Re-run focused library tests**

Run: `cargo test hnsw --lib -- --nocapture`  
Expected: PASS

- [ ] **Step 3: Re-run round-4 and historical safety contracts**

Run: `cargo test --test bench_hnsw_cpp_compare -q`  
Expected: PASS

Run: `cargo test --test bench_hnsw_reopen_round3 -q`  
Expected: PASS

Run: `cargo test --test bench_hnsw_reopen_round4 -q`  
Expected: PASS

- [ ] **Step 4: Refresh the round-4 synthetic profile locally**

Run: `cargo test --features long-tests --test bench_hnsw_reopen_round4_profile -- --ignored --nocapture`  
Expected: PASS and refreshed `benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json`

- [ ] **Step 5: Authority synthetic replay**

Run: `bash init.sh`  
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round4 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round4 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_cpp_compare -q"`  
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round4 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round4 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round4_profile -- --ignored --nocapture"`  
Expected: `test=ok`

- [ ] **Step 6: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add src/faiss/hnsw.rs src/simd.rs tests/bench_hnsw_reopen_round4.rs tests/bench_hnsw_reopen_round4_profile.rs benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json feature-list.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): rework round 4 layer0 search core"
```

## Chunk 4: Refresh the Real Authority Lane

### Task 7: Re-run same-schema authority and summarize round 4

**Files:**
- Create: `benchmark_results/hnsw_reopen_round4_authority_summary.json`
- Modify: `tests/bench_hnsw_reopen_round4.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Tighten the round-4 contract again**

Extend `tests/bench_hnsw_reopen_round4.rs` so it requires `benchmark_results/hnsw_reopen_round4_authority_summary.json` and asserts:

- the recorded round-4 target is `layer0_searcher_parity`
- the summary includes recall/QPS deltas versus `benchmark_results/hnsw_reopen_round4_baseline.json`
- the artifact explicitly states whether a later verdict-refresh feature is justified
- the artifact explicitly states whether the next action is `continue`, `soft_stop`, or `hard_stop`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round4 -- --nocapture`  
Expected: FAIL because the authority summary artifact does not exist yet.

- [ ] **Step 3: Refresh remote sync**

Run: `bash init.sh`  
Expected: PASS

- [ ] **Step 4: Re-run the authoritative same-schema lane**

Run the recorded remote same-schema HNSW benchmark command needed to regenerate `benchmark_results/rs_hnsw_sift128.full_k100.json`. Save the resulting log path in the round-4 summary artifact.

- [ ] **Step 5: Refresh native and synthetic evidence**

Run: `bash scripts/remote/native_hnsw_qps_capture.sh --log-dir /data/work/knowhere-rs-logs-hnsw-reopen-round4 --gtest-filter Benchmark_float_qps.TEST_HNSW`  
Expected: `exit_code=0`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round4 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round4 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round4_profile -- --ignored --nocapture"`  
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round4 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round4 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round4 -q"`  
Expected: `test=ok`

- [ ] **Step 6: Write the round-4 authority summary**

Create `benchmark_results/hnsw_reopen_round4_authority_summary.json` with:

- fresh same-schema Rust and native recall/QPS results
- deltas versus `benchmark_results/hnsw_reopen_round4_baseline.json`
- references to the round-4 audit/profile artifact and authority logs
- `verdict_refresh_allowed` as `true` or `false`
- `next_action` as `continue`, `soft_stop`, or `hard_stop`

Keep the historical HNSW family verdict unchanged in this chunk even if `verdict_refresh_allowed == true`; only record whether a later verdict-refresh feature is justified.

- [ ] **Step 7: Re-run the contract to verify it passes**

Run: `cargo test --test bench_hnsw_reopen_round4 -- --nocapture`  
Expected: PASS

- [ ] **Step 8: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add benchmark_results/hnsw_reopen_round4_authority_summary.json tests/bench_hnsw_reopen_round4.rs feature-list.json task-progress.md TASK_QUEUE.md GAP_ANALYSIS.md DEV_ROADMAP.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "test(hnsw): refresh round 4 authority summary"
```

Plan complete and saved to `docs/superpowers/plans/2026-03-12-hnsw-round4-layer0-searcher-parity.md`. Ready to execute.
