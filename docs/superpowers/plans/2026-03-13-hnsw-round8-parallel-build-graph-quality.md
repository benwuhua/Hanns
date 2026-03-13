# HNSW Round 8 Parallel-Build Graph Quality Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reopen HNSW around parallel-build graph-quality parity and determine whether bulk-build insertion alignment can materially improve the remote same-schema recall-gated authority lane.

**Architecture:** Keep the historical verdict chain intact, add round-8 baseline/audit/authority artifacts, and constrain implementation changes to the bulk-build insertion path in `src/faiss/hnsw.rs`. Round 8 deliberately isolates graph-quality fixes from remaining search-side micro-optimizations so the authority rerun stays attributable.

**Tech Stack:** Rust 2021, cargo tests, remote x86 authority wrappers, JSON benchmark artifacts, durable markdown/json workflow state.

---

## Chunk 1: Activate Round 8

### Task 1: Add a failing round-8 activation regression

**Files:**
- Create: `tests/bench_hnsw_reopen_round8.rs`

- [ ] **Step 1: Write the failing test**

Write a default-lane regression that loads `benchmark_results/hnsw_reopen_round8_baseline.json` and asserts:

- `task_id == "HNSW-REOPEN-ROUND8-BASELINE"`
- `family == "HNSW"`
- `authority_scope == "remote_x86_only"`
- `round7_audit_source == "benchmark_results/hnsw_reopen_layer0_flat_graph_audit_round7.json"`
- `round5_stability_source == "benchmark_results/hnsw_reopen_round5_stability_gate.json"`
- `round8_target == "parallel_build_graph_quality_parity"`
- the summary explicitly says round-6/round-7 evidence did not yet change the family verdict

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round8 -- --nocapture`  
Expected: FAIL because `benchmark_results/hnsw_reopen_round8_baseline.json` does not exist yet.

### Task 2: Add the round-8 baseline artifact and durable state

**Files:**
- Create: `benchmark_results/hnsw_reopen_round8_baseline.json`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Create the round-8 baseline artifact**

Write `benchmark_results/hnsw_reopen_round8_baseline.json` with:

- references to `benchmark_results/hnsw_reopen_round5_stability_gate.json`, `benchmark_results/hnsw_reopen_layer0_prefetch_audit_round6.json`, and `benchmark_results/hnsw_reopen_layer0_flat_graph_audit_round7.json`
- the latest attributable same-schema baseline row and the latest non-attributable stability rerun row
- `round8_target` set to `parallel_build_graph_quality_parity`
- a summary explaining that search-side structure improved, but the next explicit hypothesis moves to graph quality

- [ ] **Step 2: Extend durable workflow state**

Add four new failing features to `feature-list.json`:

- `hnsw-reopen-round8-activation`
- `hnsw-parallel-build-graph-audit-round8`
- `hnsw-parallel-build-graph-rework-round8`
- `hnsw-round8-authority-same-schema-rerun`

Set `task-progress.md` current focus and next feature to `hnsw-reopen-round8-activation`.

- [ ] **Step 3: Update progress and governance docs**

Update `task-progress.md`, `TASK_QUEUE.md`, `GAP_ANALYSIS.md`, `DEV_ROADMAP.md`, `RELEASE_NOTES.md`, and `docs/PARITY_AUDIT.md` so they describe:

- round 5 stability as the latest authority variability gate
- round 6 and round 7 as search-side audit lines, not verdict-refresh evidence
- `parallel_build_graph_quality_parity` as the next explicit HNSW hypothesis

- [ ] **Step 4: Run the regression to verify it passes**

Run: `cargo test --test bench_hnsw_reopen_round8 -- --nocapture`  
Expected: PASS

- [ ] **Step 5: Refresh remote sync and authority replay**

Run: `bash init.sh`  
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round8 -q"`  
Expected: `test=ok`

- [ ] **Step 6: Validate durable state and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add tests/bench_hnsw_reopen_round8.rs benchmark_results/hnsw_reopen_round8_baseline.json feature-list.json task-progress.md TASK_QUEUE.md GAP_ANALYSIS.md DEV_ROADMAP.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): activate reopen round 8"
```

## Chunk 2: Audit Parallel-Build Graph Quality

### Task 3: Tighten the round-8 contract around build parity evidence

**Files:**
- Modify: `tests/bench_hnsw_reopen_round8.rs`
- Create: `tests/bench_hnsw_reopen_round8_profile.rs`

- [ ] **Step 1: Extend the round-8 contract**

Update `tests/bench_hnsw_reopen_round8.rs` so it also requires `benchmark_results/hnsw_reopen_parallel_build_audit_round8.json` with:

- `native_reference_files`
- `rust_reference_files`
- `parallel_insert_entry_descent_mode`
- `upper_layer_overflow_shrink_mode`
- `build_profile_fields`
- `build_graph_quality_notes`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round8 -- --nocapture`  
Expected: FAIL because the audit/profile artifact does not exist yet.

### Task 4: Implement the round-8 build audit

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Create: `benchmark_results/hnsw_reopen_parallel_build_audit_round8.json`
- Create: `tests/bench_hnsw_reopen_round8_profile.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Extend build profiling support in `src/faiss/hnsw.rs`**

Instrument the bulk-build path so the round-8 audit can report:

- whether upper-layer greedy descent ran before the node-level candidate loop
- how many descent levels or descent updates occurred
- whether upper-layer overflow used heuristic shrink or truncate-to-best
- any counters needed to prove the new path is exercised during profile generation

- [ ] **Step 2: Add the long-test generator**

Create `tests/bench_hnsw_reopen_round8_profile.rs` to generate `benchmark_results/hnsw_reopen_parallel_build_audit_round8.json`.

- [ ] **Step 3: Generate the artifact locally**

Run: `cargo test --features long-tests --test bench_hnsw_reopen_round8_profile -- --ignored --nocapture`  
Expected: PASS and writes `benchmark_results/hnsw_reopen_parallel_build_audit_round8.json`

- [ ] **Step 4: Re-run the contract test**

Run: `cargo test --test bench_hnsw_reopen_round8 -- --nocapture`  
Expected: PASS

- [ ] **Step 5: Authority replay**

Run: `bash init.sh`  
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round8_profile -- --ignored --nocapture"`  
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round8 -q"`  
Expected: `test=ok`

- [ ] **Step 6: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_reopen_round8.rs tests/bench_hnsw_reopen_round8_profile.rs benchmark_results/hnsw_reopen_parallel_build_audit_round8.json feature-list.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): audit round 8 build graph quality"
```

## Chunk 3: Rework the Parallel Build Path

### Task 5: Add focused round-8 graph-quality regressions

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add failing focused regressions**

Add targeted tests proving:

- bulk-build insertion performs upper-layer greedy descent before searching the node's own top layer
- upper-layer overflow shrink in the bulk-build path uses heuristic diversification instead of naive truncation
- deterministic small-fixture graph quality from bulk build stays aligned with repeated serial insertion on upper layers
- round-6/round-7 search-path surfaces remain intact after the build-path change

- [ ] **Step 2: Run focused library tests to verify failure**

Run: `cargo test hnsw --lib -- --nocapture`  
Expected: FAIL on the new round-8 focused regressions.

### Task 6: Implement the round-8 build-graph-quality rework

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Implement the minimal build-path changes**

Rework the bulk-build insertion path with:

- top-down greedy descent from `self.max_level` to `node_level + 1` before the per-layer candidate loop
- heuristic shrink for upper-layer overflow during bulk-build connection updates
- minimal helper extraction or deferred shrink bookkeeping needed to avoid borrow conflicts

Do not change search API behavior, serialization format, FFI, or the round-6/round-7 search-loop structure.

- [ ] **Step 2: Re-run focused library tests**

Run: `cargo test hnsw --lib -- --nocapture`  
Expected: PASS

- [ ] **Step 3: Re-run round-8 and historical safety contracts**

Run: `cargo test --test bench_hnsw_reopen_round8 -- --nocapture`  
Expected: PASS

Run: `cargo test --test bench_hnsw_reopen_round7_flat_graph -- --nocapture`  
Expected: PASS

Run: `cargo test --test bench_hnsw_reopen_round6_prefetch -- --nocapture`  
Expected: PASS

Run: `cargo test --test bench_hnsw_reopen_round5 -- --nocapture`  
Expected: PASS

- [ ] **Step 4: Refresh the round-8 build audit locally**

Run: `cargo test --features long-tests --test bench_hnsw_reopen_round8_profile -- --ignored --nocapture`  
Expected: PASS and refreshed `benchmark_results/hnsw_reopen_parallel_build_audit_round8.json`

- [ ] **Step 5: Authority safety replay**

Run: `bash init.sh`  
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round7_flat_graph -q"`  
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round8_profile -- --ignored --nocapture"`  
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round8 -q"`  
Expected: `test=ok`

- [ ] **Step 6: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_reopen_round8.rs tests/bench_hnsw_reopen_round8_profile.rs benchmark_results/hnsw_reopen_parallel_build_audit_round8.json feature-list.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): rework round 8 bulk-build graph quality"
```

## Chunk 4: Refresh the Real Authority Lane

### Task 7: Re-run same-schema authority and summarize round 8

**Files:**
- Create: `benchmark_results/hnsw_reopen_round8_authority_summary.json`
- Modify: `tests/bench_hnsw_reopen_round8.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Tighten the round-8 contract again**

Extend `tests/bench_hnsw_reopen_round8.rs` so it requires `benchmark_results/hnsw_reopen_round8_authority_summary.json` and asserts:

- the recorded round-8 target is `parallel_build_graph_quality_parity`
- the summary includes recall/QPS deltas versus `benchmark_results/hnsw_reopen_round8_baseline.json`
- the artifact explicitly states whether a later verdict-refresh feature is justified
- the artifact explicitly states whether the next action is `continue`, `soft_stop`, or `hard_stop`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round8 -- --nocapture`  
Expected: FAIL because the authority summary artifact does not exist yet.

- [ ] **Step 3: Refresh remote sync**

Run: `bash init.sh`  
Expected: PASS

- [ ] **Step 4: Re-run the authoritative same-schema lane**

Run the recorded remote same-schema HNSW benchmark command needed to regenerate `benchmark_results/rs_hnsw_sift128.full_k100.json`. Save the resulting log path in the round-8 summary artifact.

- [ ] **Step 5: Refresh native and audit evidence**

Run: `bash scripts/remote/native_hnsw_qps_capture.sh --log-dir /data/work/knowhere-rs-logs-hnsw-reopen-round8 --gtest-filter Benchmark_float_qps.TEST_HNSW`  
Expected: `exit_code=0`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round8_profile -- --ignored --nocapture"`  
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round8 -q"`  
Expected: `test=ok`

- [ ] **Step 6: Write the round-8 authority summary**

Create `benchmark_results/hnsw_reopen_round8_authority_summary.json` with:

- fresh same-schema Rust and native recall/QPS results
- deltas versus `benchmark_results/hnsw_reopen_round8_baseline.json`
- references to the round-8 build audit and authority logs
- `verdict_refresh_allowed` as `true` or `false`
- `next_action` as `continue`, `soft_stop`, or `hard_stop`

Keep the historical HNSW family verdict unchanged in this chunk even if `verdict_refresh_allowed == true`; only record whether a later verdict-refresh feature is justified.

- [ ] **Step 7: Re-run the contract and durable-state validation**

Run: `cargo test --test bench_hnsw_reopen_round8 -- --nocapture`  
Expected: PASS

Run: `python3 scripts/validate_features.py feature-list.json`  
Expected: `VALID`

- [ ] **Step 8: Commit**

```bash
git add benchmark_results/hnsw_reopen_round8_authority_summary.json tests/bench_hnsw_reopen_round8.rs feature-list.json task-progress.md TASK_QUEUE.md GAP_ANALYSIS.md DEV_ROADMAP.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): refresh round 8 authority evidence"
```

Plan complete and saved to `docs/superpowers/plans/2026-03-13-hnsw-round8-parallel-build-graph-quality.md`. Ready to execute?
