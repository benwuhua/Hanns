# HNSW Unified Core Rewrite Roadmap Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose whether pure-Rust HNSW is primarily graph-limited, search-kernel-limited, or both, then execute the correct rewrite branch instead of continuing isolated reopen experiments.

**Architecture:** The plan starts with a shared diagnostic spine: lock one representative baseline, export comparable Rust/native graph statistics, and measure matched-recall `ef` and visited-node cost. Only after that decision gate should implementation fork into either a build-first rewrite or a unified generic-search rewrite. Compatibility work stays visible, but it is intentionally sequenced after the dominant performance cause is known.

**Tech Stack:** Rust 2021, existing HNSW implementation in `src/faiss/hnsw.rs`, repo benchmark/test harnesses in `tests/`, HDF5 benchmark binary in `src/bin/generate_hdf5_hnsw_baseline.rs`, native authority wrappers in `scripts/remote/`, JSON artifacts in `benchmark_results/`.

---

## File Map

- Modify: `src/faiss/hnsw.rs`
  - Add reusable graph-diagnostic exports, visited-count reporting, and later the chosen rewrite branch.
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`
  - Expose optional diagnostic outputs so the main HDF5 lane can emit matched-recall and visited-node evidence.
- Create: `src/bin/hnsw_graph_diagnose.rs`
  - Local and remote Rust-side graph diagnostic/export tool for level histogram, degree histogram, connectivity proxies, and search-cost stats.
- Create: `tests/bench_hnsw_graph_diagnosis.rs`
  - Regression around the baseline lock and graph-diagnosis JSON schema.
- Create: `tests/bench_hnsw_search_cost_diagnosis.rs`
  - Regression around matched-recall `ef` / visited / per-query search-cost outputs.
- Modify: `tests/debug_hnsw_graph.rs`
  - Reuse or tighten existing graph-debug utilities into stable assertions where possible.
- Modify: `tests/diagnose_hnsw_graph.rs`
  - Convert ad hoc graph diagnosis into a durable, repeatable test or fixture helper if it materially overlaps.
- Create: `benchmark_results/hnsw_core_rewrite_baseline_lock.json`
  - One locked baseline context for the entire rewrite program.
- Create: `benchmark_results/hnsw_graph_diagnosis_rust.json`
  - Rust graph-structure export artifact.
- Create: `benchmark_results/hnsw_graph_diagnosis_native.json`
  - Native graph-structure export artifact or parsed summary artifact.
- Create: `benchmark_results/hnsw_search_cost_diagnosis.json`
  - Matched-recall `ef` / visited / cost comparison artifact.
- Create: `benchmark_results/hnsw_core_rewrite_decision_gate.json`
  - Decision memo declaring `build_first`, `search_first`, or `dual_rewrite`.
- Create: `scripts/remote/native_hnsw_graph_diag.sh`
  - Remote helper to build or run a native-side HNSW graph diagnostic exporter.
- Create: `scripts/remote/native_hnsw_graph_diag_parser.py`
  - Parser or normalizer for native graph-diagnostic output into repo JSON schema.
- Modify: `RELEASE_NOTES.md`
  - Record diagnosis results and whichever rewrite branch is chosen.
- Modify: `docs/PARITY_AUDIT.md`
  - Record branch-selection evidence and final HNSW reasoning.

## Chunk 1: Shared Baseline and Diagnostic Harness

### Task 1: Lock the rewrite baseline

**Files:**
- Create: `benchmark_results/hnsw_core_rewrite_baseline_lock.json`
- Create: `tests/bench_hnsw_graph_diagnosis.rs`

- [ ] **Step 1: Write the failing baseline-lock contract**

Create `tests/bench_hnsw_graph_diagnosis.rs` with a JSON loader for `benchmark_results/hnsw_core_rewrite_baseline_lock.json`.

Required assertions:

- dataset is fixed to the same SIFT HDF5 lane used by the current authority chain
- `M`, `ef_construction`, `recall_at`, `top_k`, query count, and thread count are explicitly recorded
- artifact declares both Rust and native evidence sources

- [ ] **Step 2: Run the contract to confirm red**

```bash
cargo test --test bench_hnsw_graph_diagnosis -- --nocapture
```

Expected: FAIL because the baseline-lock artifact does not exist yet.

- [ ] **Step 3: Write the baseline-lock artifact**

Create `benchmark_results/hnsw_core_rewrite_baseline_lock.json` with:

- benchmark lane identity
- dataset path/name
- fixed query count and base size
- fixed HNSW params
- current Rust authority source
- current native authority source
- a note that later diagnosis must not drift from this context

- [ ] **Step 4: Re-run the baseline contract**

```bash
cargo test --test bench_hnsw_graph_diagnosis -- --nocapture
```

Expected: baseline-lock assertions pass; later graph/search diagnosis assertions still fail until their artifacts land.

- [ ] **Step 5: Commit**

```bash
git add tests/bench_hnsw_graph_diagnosis.rs benchmark_results/hnsw_core_rewrite_baseline_lock.json
git commit -m "test(hnsw): lock core rewrite baseline"
```

### Task 2: Add a reusable Rust graph-diagnostic exporter

**Files:**
- Create: `src/bin/hnsw_graph_diagnose.rs`
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write the failing library-level export tests**

Add focused HNSW tests for a future export/report method, for example:

- level histogram sums to node count
- per-layer degree histogram keys are stable
- reported layer-0 average degree matches existing graph stats
- export can include search-cost counters without running the full authority lane

- [ ] **Step 2: Run the targeted tests to confirm red**

```bash
cargo test hnsw::tests::test_graph_diagnosis_report_counts_nodes --lib -- --nocapture
cargo test hnsw::tests::test_graph_diagnosis_report_has_degree_histograms --lib -- --nocapture
```

Expected: FAIL because no diagnosis report API exists yet.

- [ ] **Step 3: Implement minimal diagnosis-report structs**

In `src/faiss/hnsw.rs`, add internal or public report structs covering:

- level histogram
- per-layer degree histogram
- layer count and max level
- average / P50 / P95 degree by layer
- optional connectivity proxies available from current graph state

- [ ] **Step 4: Implement the CLI exporter**

In `src/bin/hnsw_graph_diagnose.rs`, add a CLI that:

- loads or builds an HNSW index using fixed params
- emits a stable JSON report to a given output path
- optionally runs a small query batch to attach visited-node and distance-call counters

- [ ] **Step 5: Re-run the targeted tests**

```bash
cargo test hnsw::tests::test_graph_diagnosis_report_counts_nodes --lib -- --nocapture
cargo test hnsw::tests::test_graph_diagnosis_report_has_degree_histograms --lib -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/faiss/hnsw.rs src/bin/hnsw_graph_diagnose.rs
git commit -m "feat(hnsw): add graph diagnosis exporter"
```

## Chunk 2: Rust vs Native Graph Structure Diagnosis

### Task 3: Generate the Rust graph-structure artifact

**Files:**
- Create: `benchmark_results/hnsw_graph_diagnosis_rust.json`
- Modify: `tests/bench_hnsw_graph_diagnosis.rs`

- [ ] **Step 1: Extend the contract for the Rust artifact**

Add assertions for:

- level histogram
- per-layer degree histogram
- max level
- node count
- a stable summary string or shape marker

- [ ] **Step 2: Run the test to confirm red**

```bash
cargo test --test bench_hnsw_graph_diagnosis -- --nocapture
```

Expected: FAIL because `benchmark_results/hnsw_graph_diagnosis_rust.json` is missing.

- [ ] **Step 3: Generate the Rust artifact locally**

Run:

```bash
cargo run --release --bin hnsw_graph_diagnose -- --output benchmark_results/hnsw_graph_diagnosis_rust.json
```

Record the exact benchmark parameters from the baseline-lock artifact.

- [ ] **Step 4: Re-run the contract**

```bash
cargo test --test bench_hnsw_graph_diagnosis -- --nocapture
```

Expected: Rust-artifact assertions pass; native-artifact assertions still fail.

- [ ] **Step 5: Commit**

```bash
git add benchmark_results/hnsw_graph_diagnosis_rust.json tests/bench_hnsw_graph_diagnosis.rs
git commit -m "test(hnsw): add rust graph diagnosis artifact"
```

### Task 4: Add native graph export and normalization

**Files:**
- Create: `scripts/remote/native_hnsw_graph_diag.sh`
- Create: `scripts/remote/native_hnsw_graph_diag_parser.py`
- Create: `benchmark_results/hnsw_graph_diagnosis_native.json`
- Modify: `tests/bench_hnsw_graph_diagnosis.rs`

- [ ] **Step 1: Define the normalized native schema in the test**

Add assertions that the native artifact contains the same top-level fields as the Rust artifact where comparable:

- node count
- level histogram
- per-layer degree summary
- recall-lane param context

- [ ] **Step 2: Run the contract to confirm red**

```bash
cargo test --test bench_hnsw_graph_diagnosis -- --nocapture
```

Expected: FAIL because native graph-diagnosis artifact is missing.

- [ ] **Step 3: Implement the remote helper**

`scripts/remote/native_hnsw_graph_diag.sh` should:

- enter the native authority repo
- build or run a small native-side graph diagnostic helper
- write raw output to the remote log directory

If native code changes are required, keep them in a self-contained helper rather than patching the benchmark harness broadly.

- [ ] **Step 4: Implement the parser**

`scripts/remote/native_hnsw_graph_diag_parser.py` should normalize the helper output into `benchmark_results/hnsw_graph_diagnosis_native.json`.

- [ ] **Step 5: Run the helper and parser**

Run the remote helper and local parser against the same baseline parameters used for Rust.

Expected: native normalized artifact lands locally.

- [ ] **Step 6: Re-run the contract**

```bash
cargo test --test bench_hnsw_graph_diagnosis -- --nocapture
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/remote/native_hnsw_graph_diag.sh scripts/remote/native_hnsw_graph_diag_parser.py benchmark_results/hnsw_graph_diagnosis_native.json tests/bench_hnsw_graph_diagnosis.rs
git commit -m "feat(hnsw): add native graph diagnosis export"
```

## Chunk 3: Matched-Recall Search-Cost Diagnosis

### Task 5: Add matched-recall `ef` / visited comparison harness

**Files:**
- Create: `tests/bench_hnsw_search_cost_diagnosis.rs`
- Modify: `src/faiss/hnsw.rs`
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`
- Create: `benchmark_results/hnsw_search_cost_diagnosis.json`

- [ ] **Step 1: Write the failing search-cost contract**

Create `tests/bench_hnsw_search_cost_diagnosis.rs` with assertions for:

- Rust matched-recall `ef`
- Rust visited-node totals or averages
- native matched-recall `ef`
- a direct `graph_limited` versus `kernel_limited` evidence section

- [ ] **Step 2: Run the contract to confirm red**

```bash
cargo test --test bench_hnsw_search_cost_diagnosis -- --nocapture
```

Expected: FAIL because the diagnosis artifact does not exist yet.

- [ ] **Step 3: Add reusable per-query counters**

In `src/faiss/hnsw.rs`, add reporting surfaces for:

- visited-node count
- frontier pushes/pops
- distance-call count
- batch-4 usage

Keep these counters separable from the old reopen-profile artifacts; they should support the generic diagnosis path directly.

- [ ] **Step 4: Extend the HDF5 benchmark binary**

In `src/bin/generate_hdf5_hnsw_baseline.rs`, add an optional diagnosis mode that sweeps `ef` values and writes:

- recall
- qps
- visited-node statistics

for the fixed baseline workload.

- [ ] **Step 5: Generate the diagnosis artifact**

Run the diagnosis mode locally first, then normalize the result into `benchmark_results/hnsw_search_cost_diagnosis.json`.

- [ ] **Step 6: Re-run the contract**

```bash
cargo test --test bench_hnsw_search_cost_diagnosis -- --nocapture
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/bench_hnsw_search_cost_diagnosis.rs src/faiss/hnsw.rs src/bin/generate_hdf5_hnsw_baseline.rs benchmark_results/hnsw_search_cost_diagnosis.json
git commit -m "feat(hnsw): add matched-recall search cost diagnosis"
```

### Task 6: Write the decision gate artifact

**Files:**
- Create: `benchmark_results/hnsw_core_rewrite_decision_gate.json`
- Modify: `tests/bench_hnsw_search_cost_diagnosis.rs`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Extend the contract for the decision gate**

Require:

- `selected_branch` is one of `build_first`, `search_first`, or `dual_rewrite`
- artifact cites the Rust and native diagnosis inputs
- artifact includes a short rationale grounded in:
  - matched-recall `ef`
  - visited-node comparison
  - graph-stat comparison

- [ ] **Step 2: Run the test to confirm red**

```bash
cargo test --test bench_hnsw_search_cost_diagnosis -- --nocapture
```

Expected: FAIL because the decision-gate artifact is missing.

- [ ] **Step 3: Write the decision gate**

Create `benchmark_results/hnsw_core_rewrite_decision_gate.json` after the diagnosis results are in hand.

- [ ] **Step 4: Update the reasoning docs**

Record:

- what the diagnosis proved
- which branch is now the priority
- which branch is explicitly deferred

- [ ] **Step 5: Re-run both diagnosis contracts**

```bash
cargo test --test bench_hnsw_graph_diagnosis -- --nocapture
cargo test --test bench_hnsw_search_cost_diagnosis -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmark_results/hnsw_core_rewrite_decision_gate.json tests/bench_hnsw_search_cost_diagnosis.rs RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "docs(hnsw): record core rewrite decision gate"
```

## Chunk 4: Branch A, Build-First Rewrite

> Execute this chunk only if `benchmark_results/hnsw_core_rewrite_decision_gate.json` selects `build_first` or `dual_rewrite`.

### Task 7: Build topology regressions

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Create: `tests/bench_hnsw_build_topology.rs`

- [ ] **Step 1: Write failing build-topology tests**

Cover:

- serial vs parallel topology drift
- level-distribution stability
- neighbor diversity or reciprocal-edge expectations
- recall-sensitive build invariants on a fixed fixture

- [ ] **Step 2: Run the new tests to confirm red**

```bash
cargo test --test bench_hnsw_build_topology -- --nocapture
```

- [ ] **Step 3: Refactor build internals without changing query paths**

Target:

- insertion ordering
- level-order handling
- neighbor selection and back-link policy
- parallel-build alignment with serial build semantics

- [ ] **Step 4: Re-run build topology tests and HNSW lib tests**

```bash
cargo test --test bench_hnsw_build_topology -- --nocapture
cargo test hnsw --lib -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_build_topology.rs
git commit -m "feat(hnsw): align build topology with diagnosis results"
```

### Task 8: Re-measure `recall / ef / visited`

**Files:**
- Modify: `benchmark_results/hnsw_search_cost_diagnosis.json`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Re-run the diagnosis harness after build changes**

Run the same commands used in Chunk 3.

- [ ] **Step 2: Confirm the graph-limited metrics moved**

Expected:

- lower matched-recall `ef`
- lower visited-node counts at comparable recall

- [ ] **Step 3: Update the diagnosis artifact and docs**

- [ ] **Step 4: Commit**

```bash
git add benchmark_results/hnsw_search_cost_diagnosis.json RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "perf(hnsw): improve build-limited diagnosis metrics"
```

## Chunk 5: Branch B, Search-First Rewrite

> Execute this chunk only if `benchmark_results/hnsw_core_rewrite_decision_gate.json` selects `search_first` or `dual_rewrite`.

### Task 9: Unify generic search traversal

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Create: `tests/bench_hnsw_generic_search_kernel.rs`

- [ ] **Step 1: Write failing generic-kernel tests**

Cover one shared traversal core for:

- unfiltered KNN
- bitset-filtered KNN
- future range-search-compatible traversal hooks

Assertions should lock:

- shared frontier/result bookkeeping
- shared visited-table reuse
- correctness parity against current known-good paths

- [ ] **Step 2: Run the new tests to confirm red**

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

- [ ] **Step 3: Refactor toward one `idx`-based search kernel**

The new core should:

- keep scratch reuse
- separate policy hooks from traversal mechanics
- reduce special-casing between generic path and fast path where practical

- [ ] **Step 4: Re-run kernel tests and HNSW lib tests**

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
cargo test hnsw --lib -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_generic_search_kernel.rs
git commit -m "feat(hnsw): unify generic search kernel"
```

### Task 10: Re-measure per-visited-node cost

**Files:**
- Modify: `benchmark_results/hnsw_search_cost_diagnosis.json`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Re-run the diagnosis harness after search changes**

- [ ] **Step 2: Confirm the kernel-limited metrics moved**

Expected:

- visited-node counts stay similar
- total time and cost per visited node decrease

- [ ] **Step 3: Update the artifact and docs**

- [ ] **Step 4: Commit**

```bash
git add benchmark_results/hnsw_search_cost_diagnosis.json RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "perf(hnsw): lower generic search cost per visited node"
```

## Chunk 6: Late Structural and Compatibility Work

### Task 11: Broader memory-layout work

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Create: `tests/bench_hnsw_layout_regression.rs`

- [ ] **Step 1: Only start after the branch decision work stabilizes**

Do not start this chunk while graph versus kernel diagnosis is still unresolved.

- [ ] **Step 2: Write failing layout regressions**

Cover:

- broader contiguous storage expectations
- stable save/load rebuild behavior
- no correctness drift between canonical and accelerated layouts

- [ ] **Step 3: Implement the minimal layout extension**

Avoid mixing layout work with unrelated policy changes.

- [ ] **Step 4: Re-run targeted layout tests and HNSW lib tests**

```bash
cargo test --test bench_hnsw_layout_regression -- --nocapture
cargo test hnsw --lib -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_layout_regression.rs
git commit -m "feat(hnsw): extend contiguous layout beyond layer0"
```

### Task 12: Compatibility and production-contract review

**Files:**
- Modify: `src/ffi.rs`
- Modify: `src/jni/mod.rs`
- Modify: `src/faiss/hnsw.rs`
- Create: `tests/bench_hnsw_contract_review.rs`

- [ ] **Step 1: Write failing contract-review tests**

Cover:

- save/load semantics
- FFI/JNI search behavior under filtered and generic queries
- explicit decision on native-compatible serialization scope

- [ ] **Step 2: Run the tests to confirm red**

```bash
cargo test --test bench_hnsw_contract_review -- --nocapture
```

- [ ] **Step 3: Implement only the decided compatibility scope**

Do not silently expand into full native serialization parity unless the decision gate explicitly requires it.

- [ ] **Step 4: Re-run the contract-review tests**

```bash
cargo test --test bench_hnsw_contract_review -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add src/ffi.rs src/jni/mod.rs src/faiss/hnsw.rs tests/bench_hnsw_contract_review.rs
git commit -m "feat(hnsw): tighten compatibility surfaces after core rewrite"
```

## Chunk 7: Final Verification and Execution Gate

### Task 13: End-of-phase verification

**Files:**
- Modify: none expected beyond prior chunks

- [ ] **Step 1: Re-run the shared diagnosis contracts**

```bash
cargo test --test bench_hnsw_graph_diagnosis -- --nocapture
cargo test --test bench_hnsw_search_cost_diagnosis -- --nocapture
```

- [ ] **Step 2: Re-run HNSW library coverage**

```bash
cargo test hnsw --lib -- --nocapture
```

- [ ] **Step 3: Re-run representative release benchmarks locally**

Use the same commands established in the baseline-lock artifact.

- [ ] **Step 4: Re-run authority verification only after one branch is complete**

At minimum:

```bash
bash init.sh
KNOWHERE_RS_REMOTE_TARGET_DIR=<isolated-target> KNOWHERE_RS_REMOTE_LOG_DIR=<isolated-log-dir> bash scripts/remote/test.sh --command "cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input /data/work/knowhere-native-src/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.full_k100.json --base-limit 1000000 --query-limit 1000 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95"
bash scripts/remote/native_hnsw_qps_capture.sh --log-dir <isolated-log-dir> --gtest-filter Benchmark_float_qps.TEST_HNSW
```

- [ ] **Step 5: Record the post-branch verdict**

Update:

- `RELEASE_NOTES.md`
- `docs/PARITY_AUDIT.md`

with the exact branch executed and whether the rewrite changed the HNSW family verdict.

