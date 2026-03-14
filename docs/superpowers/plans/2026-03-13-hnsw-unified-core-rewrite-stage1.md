# HNSW Unified Core Rewrite Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock the pure-Rust HNSW rewrite baseline, measure Rust search-cost behavior on the trusted HNSW lane, and produce a durable decision gate that says whether the next major rewrite should be `search_first`, `build_first`, or `dual_rewrite`.

**Architecture:** This stage is a constrained execution subset of the broader roadmap in `docs/superpowers/specs/2026-03-13-hnsw-unified-core-rewrite-roadmap-design.md` and `docs/superpowers/plans/2026-03-13-hnsw-unified-core-rewrite-roadmap.md`. It does not rewrite the HNSW kernel yet. Instead it adds one locked baseline artifact, one reusable Rust-side diagnosis surface in `src/faiss/hnsw.rs`, one CLI exporter, one HDF5 diagnosis mode, and one decision-gate artifact that future sessions can trust.

**Tech Stack:** Rust 2021, `src/faiss/hnsw.rs`, repo integration tests in `tests/`, HDF5 benchmark binary in `src/bin/generate_hdf5_hnsw_baseline.rs`, authority wrappers in `scripts/remote/`, JSON artifacts in `benchmark_results/`.

---

## File Map

- Create: `docs/superpowers/plans/2026-03-13-hnsw-unified-core-rewrite-stage1.md`
  - This focused execution plan.
- Modify: `src/faiss/hnsw.rs`
  - Add reusable diagnosis/report structs and per-query counters for generic search-cost measurement.
- Create: `src/bin/hnsw_graph_diagnose.rs`
  - Emit stable JSON for graph shape and optional search-cost counters.
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`
  - Add a diagnosis mode that sweeps `ef` values and emits recall/qps/visited counters under the locked baseline.
- Create: `tests/bench_hnsw_core_rewrite_stage1.rs`
  - Durable contract over baseline lock, Rust diagnosis artifact, search-cost artifact, and decision gate artifact.
- Modify: `tests/debug_hnsw_graph.rs`
  - Reuse any existing graph utility helpers instead of duplicating graph-stat logic.
- Create: `benchmark_results/hnsw_core_rewrite_baseline_lock.json`
  - Frozen HNSW rewrite context for the authority compare lane.
- Create: `benchmark_results/hnsw_graph_diagnosis_rust.json`
  - Rust-side graph structure export artifact.
- Create: `benchmark_results/hnsw_search_cost_diagnosis.json`
  - Rust search-cost diagnosis artifact with matched-recall `ef` sweep output.
- Create: `benchmark_results/hnsw_core_rewrite_decision_gate.json`
  - Durable decision artifact with `selected_branch`.
- Modify: `RELEASE_NOTES.md`
  - Record the baseline lock and diagnosis decision.
- Modify: `docs/PARITY_AUDIT.md`
  - Record why isolated reopen rounds have paused and what Stage 1 concluded.

## Chunk 1: Lock the Baseline and Add Rust Diagnosis Surfaces

### Task 1: Add the Stage 1 contract and freeze the baseline context

**Files:**
- Create: `tests/bench_hnsw_core_rewrite_stage1.rs`
- Create: `benchmark_results/hnsw_core_rewrite_baseline_lock.json`

- [ ] **Step 1: Write the failing baseline-lock test**

Create `tests/bench_hnsw_core_rewrite_stage1.rs` with a JSON loader and a first test like:

```rust
#[test]
fn baseline_lock_declares_authority_context() {
    let baseline = load_json("benchmark_results/hnsw_core_rewrite_baseline_lock.json");
    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(baseline["dataset"], "sift-128-euclidean.hdf5");
}
```

- [ ] **Step 2: Run the contract to confirm red**

Run: `cargo test --test bench_hnsw_core_rewrite_stage1 -- --nocapture`

Expected: FAIL because `benchmark_results/hnsw_core_rewrite_baseline_lock.json` does not exist yet.

- [ ] **Step 3: Write the baseline-lock artifact**

Create `benchmark_results/hnsw_core_rewrite_baseline_lock.json` with fixed fields:

```json
{
  "family": "HNSW",
  "authority_scope": "remote_x86_only",
  "dataset": "sift-128-euclidean.hdf5",
  "top_k": 100,
  "recall_at": 10,
  "m": 16,
  "ef_construction": 100,
  "thread_model": "authority_same_schema_default",
  "rust_source": "benchmark_results/rs_hnsw_sift128.full_k100.json",
  "native_source": "benchmark_results/native_hnsw_sift128.remote.json"
}
```

- [ ] **Step 4: Re-run the contract**

Run: `cargo test --test bench_hnsw_core_rewrite_stage1 -- --nocapture`

Expected: baseline-lock test passes, later artifact tests still fail.

- [ ] **Step 5: Commit**

```bash
git add tests/bench_hnsw_core_rewrite_stage1.rs benchmark_results/hnsw_core_rewrite_baseline_lock.json
git commit -m "test(hnsw): lock stage1 rewrite baseline"
```

### Task 2: Add reusable graph-diagnosis structs in `src/faiss/hnsw.rs`

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add failing library tests for graph diagnosis**

Add focused tests near the existing HNSW tests:

```rust
#[test]
fn test_graph_diagnosis_report_counts_nodes() {
    let index = deterministic_parallel_build_entry_descent_fixture();
    let report = index.graph_diagnosis_report();
    assert_eq!(report.node_count, index.count());
}
```

```rust
#[test]
fn test_graph_diagnosis_report_has_layer0_degree_summary() {
    let index = deterministic_parallel_build_entry_descent_fixture();
    let report = index.graph_diagnosis_report();
    assert!(report.layer_degree_histograms.contains_key("0"));
}
```

- [ ] **Step 2: Run the targeted tests to confirm red**

Run: `cargo test hnsw::tests::test_graph_diagnosis_report_counts_nodes --lib -- --nocapture`

Expected: FAIL because `graph_diagnosis_report()` does not exist yet.

- [ ] **Step 3: Add minimal diagnosis structs**

In `src/faiss/hnsw.rs`, add report structs with exactly the fields Stage 1 needs:

```rust
#[derive(Clone, Debug, Serialize)]
pub struct HnswGraphDiagnosisReport {
    pub node_count: usize,
    pub max_level: usize,
    pub level_histogram: BTreeMap<String, usize>,
    pub layer_degree_histograms: BTreeMap<String, BTreeMap<String, usize>>,
}
```

- [ ] **Step 4: Implement `graph_diagnosis_report()`**

Use existing graph state only. Do not add new graph mutation logic in this task.

- [ ] **Step 5: Re-run the targeted tests**

Run:
- `cargo test hnsw::tests::test_graph_diagnosis_report_counts_nodes --lib -- --nocapture`
- `cargo test hnsw::tests::test_graph_diagnosis_report_has_layer0_degree_summary --lib -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "feat(hnsw): add graph diagnosis report"
```

### Task 3: Add a CLI exporter for Rust graph diagnosis

**Files:**
- Create: `src/bin/hnsw_graph_diagnose.rs`
- Modify: `tests/bench_hnsw_core_rewrite_stage1.rs`
- Create: `benchmark_results/hnsw_graph_diagnosis_rust.json`

- [ ] **Step 1: Extend the contract for the Rust diagnosis artifact**

Add a new test like:

```rust
#[test]
fn rust_graph_diagnosis_has_stable_shape() {
    let artifact = load_json("benchmark_results/hnsw_graph_diagnosis_rust.json");
    assert!(artifact["node_count"].as_u64().unwrap() > 0);
    assert!(artifact["level_histogram"].is_object());
}
```

- [ ] **Step 2: Run the contract to confirm red**

Run: `cargo test --test bench_hnsw_core_rewrite_stage1 -- --nocapture`

Expected: FAIL because `benchmark_results/hnsw_graph_diagnosis_rust.json` does not exist yet.

- [ ] **Step 3: Implement the CLI**

Create `src/bin/hnsw_graph_diagnose.rs` with a minimal main:

```rust
fn main() -> anyhow::Result<()> {
    // load fixed baseline params
    // build or load HNSW
    // write graph_diagnosis_report() JSON to --output
    Ok(())
}
```

- [ ] **Step 4: Generate the artifact locally**

Run: `cargo run --release --bin hnsw_graph_diagnose -- --output benchmark_results/hnsw_graph_diagnosis_rust.json`

Expected: JSON file is created and matches the contract schema.

- [ ] **Step 5: Re-run the contract**

Run: `cargo test --test bench_hnsw_core_rewrite_stage1 -- --nocapture`

Expected: baseline-lock and Rust graph-diagnosis tests pass; search-cost and decision-gate tests still fail.

- [ ] **Step 6: Commit**

```bash
git add src/bin/hnsw_graph_diagnose.rs tests/bench_hnsw_core_rewrite_stage1.rs benchmark_results/hnsw_graph_diagnosis_rust.json
git commit -m "feat(hnsw): add stage1 rust graph diagnosis artifact"
```

## Chunk 2: Measure Search Cost and Write the Decision Gate

### Task 4: Add reusable generic search-cost counters

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add failing library tests for search-cost reporting**

Add tests that assert a new diagnosis surface exposes:

```rust
#[test]
fn test_search_cost_diagnosis_reports_visited_nodes() {
    let index = deterministic_parallel_build_entry_descent_fixture();
    let report = index.search_cost_diagnosis(&[0.1, 0.2, 0.3, 0.4], 16, 4);
    assert!(report.visited_nodes > 0);
}
```

- [ ] **Step 2: Run the targeted tests to confirm red**

Run: `cargo test hnsw::tests::test_search_cost_diagnosis_reports_visited_nodes --lib -- --nocapture`

Expected: FAIL because no generic diagnosis surface exists yet.

- [ ] **Step 3: Add a minimal generic-search diagnosis struct**

Use fields needed by the decision gate only:

```rust
#[derive(Clone, Debug, Serialize)]
pub struct HnswSearchCostDiagnosis {
    pub visited_nodes: usize,
    pub frontier_pushes: usize,
    pub frontier_pops: usize,
    pub distance_calls: usize,
}
```

- [ ] **Step 4: Implement the diagnosis method on the generic search path**

Do not wire it to the optimized reopen audit structs. Use one clean surface that future branch work can keep.

- [ ] **Step 5: Re-run the targeted tests**

Run: `cargo test hnsw::tests::test_search_cost_diagnosis_reports_visited_nodes --lib -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "feat(hnsw): add generic search cost diagnosis"
```

### Task 5: Extend the HDF5 benchmark binary with diagnosis mode

**Files:**
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`
- Modify: `tests/bench_hnsw_core_rewrite_stage1.rs`
- Create: `benchmark_results/hnsw_search_cost_diagnosis.json`

- [ ] **Step 1: Extend the contract for the search-cost artifact**

Add a new test like:

```rust
#[test]
fn search_cost_diagnosis_declares_ef_sweep() {
    let artifact = load_json("benchmark_results/hnsw_search_cost_diagnosis.json");
    assert!(artifact["ef_sweep"].is_array());
    assert!(artifact["selected_recall_gate"].is_number());
}
```

- [ ] **Step 2: Run the contract to confirm red**

Run: `cargo test --test bench_hnsw_core_rewrite_stage1 -- --nocapture`

Expected: FAIL because `benchmark_results/hnsw_search_cost_diagnosis.json` does not exist yet.

- [ ] **Step 3: Add diagnosis flags to the HDF5 binary**

Expose CLI flags like:

```rust
#[derive(clap::Parser)]
struct Args {
    #[arg(long)]
    diagnosis_output: Option<PathBuf>,
    #[arg(long, value_delimiter = ',')]
    ef_sweep: Vec<usize>,
}
```

- [ ] **Step 4: Generate the diagnosis artifact**

Run:

```bash
cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- \
  --input /data/work/knowhere-native-src/sift-128-euclidean.hdf5 \
  --output benchmark_results/rs_hnsw_sift128.full_k100.json \
  --diagnosis-output benchmark_results/hnsw_search_cost_diagnosis.json \
  --ef-sweep 64,96,128,160
```

Expected: the JSON includes recall, qps, visited nodes, and distance-call counters for each `ef`.

- [ ] **Step 5: Re-run the contract**

Run: `cargo test --test bench_hnsw_core_rewrite_stage1 -- --nocapture`

Expected: search-cost artifact test passes; decision-gate test still fails.

- [ ] **Step 6: Commit**

```bash
git add src/bin/generate_hdf5_hnsw_baseline.rs tests/bench_hnsw_core_rewrite_stage1.rs benchmark_results/hnsw_search_cost_diagnosis.json
git commit -m "feat(hnsw): add stage1 search cost diagnosis artifact"
```

### Task 6: Write the decision gate and close Stage 1 docs

**Files:**
- Create: `benchmark_results/hnsw_core_rewrite_decision_gate.json`
- Modify: `tests/bench_hnsw_core_rewrite_stage1.rs`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Extend the contract for the decision gate**

Add a final test like:

```rust
#[test]
fn decision_gate_selects_one_branch() {
    let gate = load_json("benchmark_results/hnsw_core_rewrite_decision_gate.json");
    let branch = gate["selected_branch"].as_str().unwrap();
    assert!(matches!(branch, "build_first" | "search_first" | "dual_rewrite"));
}
```

- [ ] **Step 2: Run the contract to confirm red**

Run: `cargo test --test bench_hnsw_core_rewrite_stage1 -- --nocapture`

Expected: FAIL because `benchmark_results/hnsw_core_rewrite_decision_gate.json` does not exist yet.

- [ ] **Step 3: Write the decision gate artifact**

Create JSON with fields:

```json
{
  "selected_branch": "search_first",
  "inputs": {
    "baseline_lock": "benchmark_results/hnsw_core_rewrite_baseline_lock.json",
    "graph_diagnosis_rust": "benchmark_results/hnsw_graph_diagnosis_rust.json",
    "search_cost_diagnosis": "benchmark_results/hnsw_search_cost_diagnosis.json"
  },
  "rationale": "matched-recall traversal effort is close enough, but generic search still pays too much per visited node"
}
```

Use the actual measured conclusion. Do not hard-code `search_first` unless the artifact evidence supports it.

- [ ] **Step 4: Update durable docs**

Record in:
- `RELEASE_NOTES.md`
- `docs/PARITY_AUDIT.md`

Required notes:
- Stage 1 froze one HNSW rewrite baseline
- Stage 1 added reusable diagnosis surfaces
- the next stage must follow the selected branch rather than reopening another isolated round

- [ ] **Step 5: Re-run the full Stage 1 contract**

Run:
- `cargo test --test bench_hnsw_core_rewrite_stage1 -- --nocapture`
- `cargo test hnsw --lib -- --nocapture`

Expected: PASS locally.

- [ ] **Step 6: Run authority verification**

Run:
- `bash init.sh`
- `bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_core_rewrite_stage1 -q"`
- `bash scripts/remote/test.sh --command "cargo test hnsw --lib -- --nocapture"`
- `bash scripts/remote/test.sh --command "cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input /data/work/knowhere-native-src/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.full_k100.json --diagnosis-output benchmark_results/hnsw_search_cost_diagnosis.json --ef-sweep 64,96,128,160"`

Expected: remote replay succeeds and the Stage 1 artifacts remain valid on the authority surface.

- [ ] **Step 7: Commit**

```bash
git add benchmark_results/hnsw_core_rewrite_decision_gate.json tests/bench_hnsw_core_rewrite_stage1.rs RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "docs(hnsw): close unified core rewrite stage1"
```

## Notes for the Next Plan

- If `selected_branch == "search_first"`, the next plan should rewrite the generic search kernel around one shared `idx`-based traversal core.
- If `selected_branch == "build_first"`, the next plan should focus on graph-quality and parallel-build topology alignment.
- If `selected_branch == "dual_rewrite"`, create two separate plans instead of one blended rewrite plan.
