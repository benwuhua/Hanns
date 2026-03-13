# HNSW Round 10 Layer-0 Slab Audit Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans or superpowers:subagent-driven-development when implementing this plan. Steps use checkbox syntax for tracking.

**Goal:** Reopen HNSW around a narrow layer-0 memory-locality hypothesis by adding a derived co-located slab for production `layer0 + L2 + no-filter` search, proving the layout switch via a durable audit artifact, and measuring the real same-schema authority impact.

**Architecture:** Round 10 keeps round-9 search logic as the algorithmic baseline, but swaps the production layer-0 data source from split `flat_graph + vectors` storage to a derived slab that physically co-locates per-node layer-0 neighbor ids and vector payload. Canonical graph/vector ownership remains unchanged so persistence and rebuild semantics stay stable.

**Tech Stack:** Rust 2021, cargo tests, existing HNSW audit harnesses in `tests/`, JSON durability in `benchmark_results/`, authority wrappers in `scripts/remote/`.

---

## File Map

- Modify: `feature-list.json`
  - Add round-10 tracked features and verification steps.
- Modify: `task-progress.md`
  - Reopen durable workflow state around round 10 and record execution history.
- Modify: `RELEASE_NOTES.md`
  - Record round-10 activation, audit, and authority conclusions.
- Modify: `docs/PARITY_AUDIT.md`
  - Append round-10 planning and execution evidence.
- Modify: `.gitignore`
  - Unignore new round-10 benchmark artifacts if needed.
- Create: `benchmark_results/hnsw_reopen_round10_baseline.json`
  - Freeze round-9 authority evidence as the round-10 starting point.
- Create: `benchmark_results/hnsw_reopen_layer0_slab_audit_round10.json`
  - Durable artifact proving slab-backed layer-0 mode is active.
- Create: `benchmark_results/hnsw_reopen_round10_authority_summary.json`
  - Final same-schema round-10 authority outcome.
- Create: `tests/bench_hnsw_reopen_round10.rs`
  - Default-lane contract for round-10 baseline, audit, and summary artifacts.
- Create: `tests/bench_hnsw_reopen_round10_profile.rs`
  - Long-test generator for the round-10 slab audit artifact.
- Modify: `src/faiss/hnsw.rs`
  - Add layer-0 slab structure, rebuild path, fast-path eligibility, and focused regressions.

## Chunk 1: Reopen Durable Round 10 Workflow

### Task 1: Add round-10 default contract and feature inventory

**Files:**
- Create: `tests/bench_hnsw_reopen_round10.rs`
- Modify: `feature-list.json`

- [ ] **Step 1: Write the failing default-lane contract**

Create `tests/bench_hnsw_reopen_round10.rs` with JSON loaders and tests for:

- `benchmark_results/hnsw_reopen_round10_baseline.json`
- `benchmark_results/hnsw_reopen_layer0_slab_audit_round10.json`
- `benchmark_results/hnsw_reopen_round10_authority_summary.json`

Required assertions:

- baseline target string = `layer0_slab_locality`
- audit layout mode strings record slab activation
- authority summary records `verdict_refresh_allowed`, `next_action`, and current same-schema qps values

- [ ] **Step 2: Run the new contract and confirm red**

```bash
cargo test --test bench_hnsw_reopen_round10 -- --nocapture
```

Expected: FAIL because no round-10 artifacts exist yet.

- [ ] **Step 3: Add round-10 features to `feature-list.json`**

Add:

- `hnsw-reopen-round10-activation`
- `hnsw-layer0-slab-audit-round10`
- `hnsw-round10-authority-same-schema-rerun`

Dependencies should be linear:

- activation depends on `hnsw-round9-authority-same-schema-rerun`
- audit depends on activation
- authority rerun depends on audit

- [ ] **Step 4: Run validator**

```bash
python3 scripts/validate_features.py feature-list.json
```

Expected: structure valid, new features still failing.

### Task 2: Create round-10 baseline artifact and reopen durable state

**Files:**
- Create: `benchmark_results/hnsw_reopen_round10_baseline.json`
- Modify: `.gitignore`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Write the round-10 baseline artifact**

Freeze round-9 authority evidence into the new starting point with fields analogous to round 9:

- `task_id`
- `family`
- `authority_scope`
- `historical_verdict_source`
- `round9_authority_summary_source`
- `round10_target = "layer0_slab_locality"`
- `summary`
- `action`

- [ ] **Step 2: Unignore new artifacts**

Ensure `.gitignore` allows:

```gitignore
!benchmark_results/hnsw_reopen_round10_baseline.json
!benchmark_results/hnsw_reopen_layer0_slab_audit_round10.json
!benchmark_results/hnsw_reopen_round10_authority_summary.json
```

- [ ] **Step 3: Reopen durable docs**

Update:

- `task-progress.md`
- `RELEASE_NOTES.md`
- `docs/PARITY_AUDIT.md`

Set:

- `Current focus: hnsw-reopen-round10-activation`
- `Next feature: hnsw-reopen-round10-activation`

- [ ] **Step 4: Re-run the contract**

```bash
cargo test --test bench_hnsw_reopen_round10 -- --nocapture
```

Expected: baseline passes, audit and summary still fail.

## Chunk 2: Implement Layer-0 Slab and Audit Surface

### Task 3: Add focused failing HNSW regressions

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add slab rebuild regressions**

Add focused tests that fail until:

- slab rebuild copies layer-0 neighbor ids and vectors into co-located per-node layout
- slab stays disabled on unsupported states and falls back safely
- production layer-0 fast path reports slab mode when eligible

Suggested assertions:

- per-node slab vector pointer matches canonical vector content
- slab neighbor slice matches canonical layer-0 neighbors
- audit mode string distinguishes `layer0_slab` from `flat_u32_adjacency`

- [ ] **Step 2: Run targeted tests and confirm red**

```bash
cargo test hnsw::tests::test_layer0_slab_rebuild_tracks_canonical_state --lib -- --nocapture
cargo test hnsw::tests::test_layer0_fast_path_reports_slab_layout_when_enabled --lib -- --nocapture
```

### Task 4: Implement derived layer-0 slab

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add `Layer0Slab` structure**

Implement fixed-stride per-node storage for:

- degree/count
- fixed-capacity `u32` layer-0 neighbors
- contiguous vector payload

- [ ] **Step 2: Add slab rebuild/clear hooks**

Implement `refresh_layer0_slab()` and ensure it is called after:

- add/build completion
- load/deserialize paths
- repair paths that mutate layer-0 connectivity

- [ ] **Step 3: Add slab-backed access helpers**

Helpers should expose:

- neighbor ids for a node
- direct vector pointer for a node
- eligibility and layout metadata for audits

### Task 5: Switch production fast path to slab when eligible

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Narrow production fast-path swap**

Update the round-9 production `search_layer_idx_l2_ordered_pool_fast()` so that when slab mode is active it reads:

- neighbors from slab
- vector pointers from slab

Keep the algorithm, frontier behavior, and fallback path unchanged.

- [ ] **Step 2: Preserve profiled path**

The profiled path can either:

- continue reading split storage but report slab eligibility, or
- explicitly report slab-backed layout if the implementation can do so without muddying timing semantics

Either choice is acceptable as long as the audit artifact makes the behavior explicit.

- [ ] **Step 3: Re-run HNSW library tests**

```bash
cargo test hnsw --lib -- --nocapture
```

## Chunk 3: Round-10 Audit Artifact and Authority Verdict

### Task 6: Add round-10 slab audit generator

**Files:**
- Create: `tests/bench_hnsw_reopen_round10_profile.rs`
- Create: `benchmark_results/hnsw_reopen_layer0_slab_audit_round10.json`

- [ ] **Step 1: Build the long-test generator**

Mirror round-9 style, but the artifact must record:

- `production_layer0_layout_mode`
- `profiled_layer0_layout_mode`
- `layer0_slab_enabled`
- `layer0_slab_stride_bytes`
- `layer0_slab_vector_offset_bytes`
- `layer0_slab_max_neighbors`
- `layer0_slab_rebuild_source`

- [ ] **Step 2: Generate the local artifact**

```bash
cargo test --features long-tests --test bench_hnsw_reopen_round10_profile -- --ignored --nocapture
```

- [ ] **Step 3: Re-run the round-10 contract**

```bash
cargo test --test bench_hnsw_reopen_round10 -- --nocapture
```

Expected: baseline and audit pass, authority summary still missing.

### Task 7: Authority same-schema rerun and final summary

**Files:**
- Create: `benchmark_results/hnsw_reopen_round10_authority_summary.json`
- Modify: `benchmark_results/rs_hnsw_sift128.full_k100.json`

- [ ] **Step 1: Run authority same-schema benchmark**

```bash
bash init.sh
KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round10 \
KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round10 \
bash scripts/remote/test.sh --command "cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input /data/work/knowhere-native-src/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.full_k100.json --base-limit 1000000 --query-limit 1000 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95"
```

- [ ] **Step 2: Refresh native capture**

```bash
bash scripts/remote/native_hnsw_qps_capture.sh --log-dir /data/work/knowhere-rs-logs-hnsw-reopen-round10 --gtest-filter Benchmark_float_qps.TEST_HNSW
```

- [ ] **Step 3: Replay authority audit and contract**

```bash
KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round10 \
KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round10 \
bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round10_profile -- --ignored --nocapture"

KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round10 \
KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round10 \
bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round10 -q"
```

- [ ] **Step 4: Write final summary artifact**

Archive:

- current Rust qps/recall
- current native qps/recall
- delta vs round 9
- current ratio
- `verdict_refresh_allowed`
- `next_action`

## Chunk 4: Persist and Close

### Task 8: Update durable state and validate closure

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Mark passing features only after authority verification**

Set round-10 features to `passing` only after all recorded remote commands succeed.

- [ ] **Step 2: Run final validator**

```bash
python3 scripts/validate_features.py feature-list.json
```

- [ ] **Step 3: Commit in logical slices**

Suggested commit boundaries:

1. round-10 activation/docs red state
2. slab implementation + local audit artifact
3. authority summary + durable closure
