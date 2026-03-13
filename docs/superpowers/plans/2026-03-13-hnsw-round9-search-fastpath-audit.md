# HNSW Round 9 Search Fast Path Audit Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrowly-scoped HNSW round-9 reopen line that separates the production `layer0 + L2 + no-filter` hot path from profiling instrumentation, caches batch-4 SIMD dispatch, and produces an authority-backed answer about whether that cleanup materially improves the real same-schema benchmark.

**Architecture:** Round 9 keeps all existing round-4 through round-8 audit surfaces intact, but introduces a dedicated production fast path in `src/faiss/hnsw.rs` and a cached batch-4 kernel selector in `src/simd.rs`. Durable workflow artifacts mirror round 8: activation baseline, audit/profile artifact, and final authority summary, so the new claim stays small and attributable.

**Tech Stack:** Rust 2021, cargo tests, existing HNSW audit/test harnesses in `tests/`, JSON durability in `benchmark_results/`, remote authority wrappers in `scripts/remote/`.

---

## File Map

- Modify: `feature-list.json`
  - Add the new round-9 tracked features and their verification steps.
- Modify: `task-progress.md`
  - Reopen durable workflow state around round 9 and record execution history.
- Modify: `RELEASE_NOTES.md`
  - Record round-9 activation, audit, and authority conclusions.
- Modify: `docs/PARITY_AUDIT.md`
  - Append round-9 planning and execution evidence.
- Create: `benchmark_results/hnsw_reopen_round9_baseline.json`
  - Freeze round-8 hard-stop evidence as the round-9 starting point.
- Create: `benchmark_results/hnsw_reopen_search_fastpath_audit_round9.json`
  - Durable artifact proving production fast-path mode and batch-4 dispatch mode.
- Create: `benchmark_results/hnsw_reopen_round9_authority_summary.json`
  - Final same-schema round-9 authority outcome.
- Modify: `.gitignore`
  - Unignore the two new round-9 benchmark artifacts if needed.
- Create: `tests/bench_hnsw_reopen_round9.rs`
  - Default-lane contract covering round-9 baseline, audit, and authority summary artifacts.
- Create: `tests/bench_hnsw_reopen_round9_profile.rs`
  - Long-test generator for the round-9 audit/profile artifact.
- Modify: `src/faiss/hnsw.rs`
  - Add the production layer-0 fast path, audit/profile metadata, and focused regressions.
- Modify: `src/simd.rs`
  - Add cached batch-4 kernel dispatch and focused SIMD regressions.

## Chunk 1: Reopen Durable Round 9 Workflow

### Task 1: Add failing round-9 default contract

**Files:**
- Create: `tests/bench_hnsw_reopen_round9.rs`
- Modify: `feature-list.json`
- Test: `tests/bench_hnsw_reopen_round9.rs`

- [ ] **Step 1: Write the failing default-lane contract**

Create `tests/bench_hnsw_reopen_round9.rs` with three JSON loaders and three tests mirroring the round-8 style:

```rust
use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND9_BASELINE_PATH: &str =
    "benchmark_results/hnsw_reopen_round9_baseline.json";
const HNSW_REOPEN_SEARCH_FASTPATH_AUDIT_ROUND9_PATH: &str =
    "benchmark_results/hnsw_reopen_search_fastpath_audit_round9.json";
const HNSW_REOPEN_ROUND9_AUTHORITY_SUMMARY_PATH: &str =
    "benchmark_results/hnsw_reopen_round9_authority_summary.json";
```

Add contract assertions for:

- baseline target string = `search_fastpath_cleanup`
- audit mode strings for production fast path and batch-4 dispatch
- authority summary with `verdict_refresh_allowed` boolean and `next_action`

- [ ] **Step 2: Run the new contract to verify it fails**

Run:

```bash
cargo test --test bench_hnsw_reopen_round9 -- --nocapture
```

Expected: FAIL because the new round-9 baseline artifact does not exist yet.

- [ ] **Step 3: Add round-9 feature inventory entries**

Append three new features in `feature-list.json`:

- `hnsw-reopen-round9-activation`
- `hnsw-search-fastpath-audit-round9`
- `hnsw-round9-authority-same-schema-rerun`

Use round-8 naming/style as the template. Dependencies should be linear:

- activation depends on `hnsw-round8-authority-same-schema-rerun`
- audit depends on activation
- authority rerun depends on audit

- [ ] **Step 4: Run validator and expect workflow to remain failing**

Run:

```bash
python3 scripts/validate_features.py feature-list.json
```

Expected: VALID structure, but new round-9 features still marked `failing`.

- [ ] **Step 5: Commit the red baseline setup**

```bash
git add tests/bench_hnsw_reopen_round9.rs feature-list.json
git commit -m "test(hnsw): add round9 default contract"
```

### Task 2: Create round-9 baseline artifact and reopen durable state

**Files:**
- Create: `benchmark_results/hnsw_reopen_round9_baseline.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`
- Modify: `.gitignore`
- Test: `tests/bench_hnsw_reopen_round9.rs`

- [ ] **Step 1: Write the round-9 baseline artifact**

Create `benchmark_results/hnsw_reopen_round9_baseline.json` with fields modeled on round 8:

- `task_id`
- `family`
- `authority_scope`
- `historical_verdict_source`
- `round8_authority_summary_source`
- `round9_target = "search_fastpath_cleanup"`
- `historical_classification`
- `round8_hard_stop_context`
- `summary`
- `action`

Freeze the round-8 authority evidence (`750.732` qps, `0.9945` recall, hard stop) as the starting point.

- [ ] **Step 2: Unignore the new round-9 artifacts**

Add these lines to `.gitignore` if not already present:

```gitignore
!benchmark_results/hnsw_reopen_round9_baseline.json
!benchmark_results/hnsw_reopen_search_fastpath_audit_round9.json
!benchmark_results/hnsw_reopen_round9_authority_summary.json
```

- [ ] **Step 3: Reopen durable docs**

Update:

- `task-progress.md`
- `RELEASE_NOTES.md`
- `docs/PARITY_AUDIT.md`

Set:

- `Current focus: hnsw-reopen-round9-activation`
- `Next feature: hnsw-reopen-round9-activation`
- progress should reflect new total feature count with the new round-9 entries marked failing

- [ ] **Step 4: Run the round-9 contract again**

Run:

```bash
cargo test --test bench_hnsw_reopen_round9 -- --nocapture
```

Expected: baseline test passes, audit/authority summary tests still fail.

- [ ] **Step 5: Commit the activation baseline**

```bash
git add .gitignore benchmark_results/hnsw_reopen_round9_baseline.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "docs(progress): activate round9 hnsw reopen"
```

## Chunk 2: Implement Fast Path and Batch-4 Dispatch Audit

### Task 3: Add focused failing library regressions

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Modify: `src/simd.rs`
- Test: `src/faiss/hnsw.rs`
- Test: `src/simd.rs`

- [ ] **Step 1: Add HNSW fast-path regressions**

In `src/faiss/hnsw.rs` test module, add focused tests that fail until:

- production `L2 + no-filter + layer0` dispatches to a dedicated fast-path mode string
- profiled search still reports a different mode string
- the fast path avoids profile bookkeeping flags/counters

Suggested assertions:

```rust
assert_eq!(index.layer0_l2_search_mode_for(false), "fast_unprofiled");
assert_eq!(index.layer0_l2_search_mode_for(true), "profiled_optional");
```

Use whatever helper names fit the existing file, but make the distinction executable.

- [ ] **Step 2: Add SIMD dispatch regressions**

In `src/simd.rs`, add tests that fail until batch-4 dispatch is cached:

- repeated calls return the same selected function pointer
- cached dispatch stays numerically equivalent to the current implementation

Example shape:

```rust
let k1 = l2_batch_4_ptrs_kernel();
let k2 = l2_batch_4_ptrs_kernel();
assert_eq!(k1 as usize, k2 as usize);
```

- [ ] **Step 3: Run targeted tests to verify they fail**

Run:

```bash
cargo test hnsw::tests::test_parallel_build_profile_reports_fastpath_mode --lib -- --nocapture
cargo test simd::tests::test_l2_batch_4_ptrs_kernel_is_cached --lib -- --nocapture
```

Expected: FAIL because the fast path and cached selector do not exist yet.

- [ ] **Step 4: Commit the red tests**

```bash
git add src/faiss/hnsw.rs src/simd.rs
git commit -m "test(hnsw): add round9 fast-path regressions"
```

### Task 4: Implement cached batch-4 dispatch

**Files:**
- Modify: `src/simd.rs`
- Test: `src/simd.rs`

- [ ] **Step 1: Add kernel alias and selector**

In `src/simd.rs`, introduce:

```rust
pub type L2Batch4PtrsKernel =
    unsafe fn(*const f32, *const f32, *const f32, *const f32, *const f32, usize) -> [f32; 4];

static L2_BATCH_4_PTRS_KERNEL: OnceLock<L2Batch4PtrsKernel> = OnceLock::new();
```

Add:

- `select_l2_batch_4_ptrs_kernel()`
- `l2_batch_4_ptrs_kernel()`

Preserve current AVX512 / AVX2+FMA / NEON / scalar gating behavior.

- [ ] **Step 2: Route `l2_batch_4_ptrs()` through the cached selector**

Refactor:

```rust
pub unsafe fn l2_batch_4_ptrs(...) -> [f32; 4] {
    let kernel = l2_batch_4_ptrs_kernel();
    kernel(query, db0, db1, db2, db3, dim)
}
```

- [ ] **Step 3: Run targeted SIMD tests**

Run:

```bash
cargo test simd::tests::test_l2_batch_4_ptrs_kernel_is_cached --lib -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Commit the SIMD dispatch change**

```bash
git add src/simd.rs
git commit -m "feat(simd): cache batch4 l2 dispatch"
```

### Task 5: Implement the production layer-0 fast path

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Test: `src/faiss/hnsw.rs`

- [ ] **Step 1: Extract a dedicated unprofiled fast path**

Add a new helper in `src/faiss/hnsw.rs` for the narrow production lane. It should:

- operate on `layer0 + L2 + no-filter`
- reuse ordered pool and flat-graph traversal
- avoid any `Instant::now()` calls
- avoid `if let Some(stats)` branches

The profiled optional-profile function remains intact for audit generators.

- [ ] **Step 2: Switch production search dispatch to the new helper**

Update the existing production entry point (the current `search_single_l2_unfiltered()` path) so that when the same narrow eligibility conditions hold, it uses the new helper instead of the optional-profile variant.

- [ ] **Step 3: Keep explicit fallback behavior**

Retain the existing generic/profilled path for:

- non-L2 metrics
- filtered searches
- non-layer0 portions
- any path that needs profiling

- [ ] **Step 4: Run targeted HNSW regressions**

Run:

```bash
cargo test hnsw::tests::test_parallel_build_profile_reports_fastpath_mode --lib -- --nocapture
cargo test hnsw --lib -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit the fast-path implementation**

```bash
git add src/faiss/hnsw.rs
git commit -m "feat(hnsw): add round9 layer0 fast path"
```

## Chunk 3: Generate Round-9 Audit and Close the Authority Rerun

### Task 6: Add round-9 audit/profile generator

**Files:**
- Create: `tests/bench_hnsw_reopen_round9_profile.rs`
- Create: `benchmark_results/hnsw_reopen_search_fastpath_audit_round9.json`
- Modify: `tests/bench_hnsw_reopen_round9.rs`

- [ ] **Step 1: Write the long-test generator**

Create `tests/bench_hnsw_reopen_round9_profile.rs` modeled on round 8. The generated artifact must include:

- `task_id`
- `family`
- `benchmark_lane`
- `authority_scope`
- `round9_baseline_source`
- `production_layer0_fastpath_mode`
- `profiled_layer0_mode`
- `production_avoids_profile_timing`
- `batch4_dispatch_mode`
- timing/call-count summaries if already available

- [ ] **Step 2: Tighten the default round-9 contract**

Update `tests/bench_hnsw_reopen_round9.rs` so it requires the new audit artifact and checks the mode strings/boolean fields.

- [ ] **Step 3: Run the generator locally**

Run:

```bash
cargo test --features long-tests --test bench_hnsw_reopen_round9_profile -- --ignored --nocapture
```

Expected: PASS and writes `benchmark_results/hnsw_reopen_search_fastpath_audit_round9.json`.

- [ ] **Step 4: Run the round-9 default contract**

Run:

```bash
cargo test --test bench_hnsw_reopen_round9 -- --nocapture
```

Expected: baseline + audit pass, authority summary still fails.

- [ ] **Step 5: Commit the audit artifact slice**

```bash
git add tests/bench_hnsw_reopen_round9.rs tests/bench_hnsw_reopen_round9_profile.rs benchmark_results/hnsw_reopen_search_fastpath_audit_round9.json
git commit -m "feat(hnsw): audit round9 search fast path"
```

### Task 7: Run authority round-9 verification and archive the result

**Files:**
- Create: `benchmark_results/hnsw_reopen_round9_authority_summary.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`
- Modify: `feature-list.json`

- [ ] **Step 1: Run local prefilters before authority**

Run:

```bash
cargo test hnsw --lib -- --nocapture
cargo test --test bench_hnsw_reopen_round9 -- --nocapture
cargo test --features long-tests --test bench_hnsw_reopen_round9_profile -- --ignored --nocapture
cargo fmt --all -- --check
```

Expected: all PASS.

- [ ] **Step 2: Bootstrap authority**

Run:

```bash
bash init.sh
```

Expected: remote authority ready.

- [ ] **Step 3: Run the authority same-schema Rust lane**

Run:

```bash
KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round9 \
KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round9 \
bash scripts/remote/test.sh --command "cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input /data/work/knowhere-native-src/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.full_k100.json --base-limit 1000000 --query-limit 1000 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95"
```

Expected: `test=ok` with a fresh round-9 Rust log path.

- [ ] **Step 4: Run fresh native capture**

Run:

```bash
bash scripts/remote/native_hnsw_qps_capture.sh --log-dir /data/work/knowhere-rs-logs-hnsw-reopen-round9 --gtest-filter Benchmark_float_qps.TEST_HNSW
```

Expected: `exit_code=0` and a fresh native log path.

- [ ] **Step 5: Replay round-9 audit/profile and contract on authority**

Run:

```bash
KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round9 \
KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round9 \
bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round9_profile -- --ignored --nocapture"

KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round9 \
KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round9 \
bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round9 -q"
```

Expected: both `test=ok`.

- [ ] **Step 6: Sync back the new Rust same-schema artifact**

Copy the updated `benchmark_results/rs_hnsw_sift128.full_k100.json` from the authority repo back into the local worktree using `scp` or the existing rsync wrapper.

- [ ] **Step 7: Write the round-9 authority summary artifact**

Create `benchmark_results/hnsw_reopen_round9_authority_summary.json` with:

- current Rust/native qps and recall
- deltas vs round-8 hard-stop baseline
- ratio vs historical verdict band
- `verdict_refresh_allowed`
- `next_action`
- summary/action text

Only two acceptable outcomes:

- measurable attributable gain => `next_action=continue`
- no meaningful gain => `next_action=soft_stop` or `hard_stop`

- [ ] **Step 8: Mark the round-9 feature chain complete in durable state**

Update:

- `feature-list.json`
- `task-progress.md`
- `RELEASE_NOTES.md`
- `docs/PARITY_AUDIT.md`

If round 9 is terminal, set:

- `Current focus: none`
- `Next feature: none`

Otherwise, document the exact next tracked feature.

- [ ] **Step 9: Run final validator**

Run:

```bash
python3 scripts/validate_features.py feature-list.json
```

Expected: VALID with the updated passing/failing counts.

- [ ] **Step 10: Commit the authority closure**

```bash
git add benchmark_results/hnsw_reopen_round9_authority_summary.json benchmark_results/rs_hnsw_sift128.full_k100.json feature-list.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "docs(progress): record round9 authority rerun"
```
