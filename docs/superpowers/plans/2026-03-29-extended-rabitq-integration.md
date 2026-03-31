# Extended-RaBitQ knowhere-rs Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Extended-RaBitQ to `knowhere-rs` as a new IVF-based compressed index family with scalar correctness, AVX512-accelerated fast scan, save/load, API exposure, and remote-x86 benchmark closure, without regressing the existing simplified `RaBitQEncoder` path.

**Architecture:** Implement Extended-RaBitQ as a parallel stack under `src/quantization/exrabitq/` plus a new `src/faiss/ivf_exrabitq.rs` consumer. Mirror the upstream pipeline instead of mutating `src/quantization/rabitq.rs` in place: residual transform around IVF centroids, padded-dimension rotation, 1-bit short code for fast scan, `(bits - 1)`-bit long code for rerank, block-transposed per-cluster layout, then scalar and AVX512 search paths selected at runtime. Treat this as a fresh local `screen` line first; only reopen durable long-task state if the local screen ends with `screen_result=promote`.

**Tech Stack:** Rust 2021, existing `KMeans` coarse quantizer, current index patterns in `src/faiss/ivf_rabitq.rs` and `src/faiss/ivf_turboquant.rs`, SIMD intrinsics with runtime feature detection, `serde` or manual binary IO for persistence, local `cargo test` / `cargo fmt` / `cargo clippy`, then `bash init.sh` and remote-x86 verification scripts for authority.

---

## File Map

- Create: `src/quantization/exrabitq/mod.rs`
  - Public exports for the Extended-RaBitQ family.
- Create: `src/quantization/exrabitq/config.rs`
  - `ExRaBitQConfig`, supported-bit validation, padded-dimension helpers, `FAST_SIZE` constants, scan-mode options.
- Create: `src/quantization/exrabitq/rotator.rs`
  - Deterministic dense orthogonal rotation over padded dimension and its transpose for inverse application.
- Create: `src/quantization/exrabitq/quantizer.rs`
  - Residual preprocessing, `fast_quantize`, short-code emission, long-code emission, `ExFactor`, decode/reference helpers.
- Create: `src/quantization/exrabitq/layout.rs`
  - Cluster-local block layout: block-transposed short codes, long codes, factors, ids, and builders/readers.
- Create: `src/quantization/exrabitq/fastscan.rs`
  - Query quantization, LUT/precompute state, scalar scan kernel, AVX512 fast path, runtime dispatch.
- Create: `src/quantization/exrabitq/searcher.rs`
  - Lower-bound scan, candidate selection, rerank math using long codes and factors.
- Modify: `src/quantization/mod.rs`
  - Export the new Extended-RaBitQ family.
- Create: `src/faiss/ivf_exrabitq.rs`
  - New IVF index backed by Extended-RaBitQ cluster layouts; train/add/search/save/load and `IndexTrait` impl.
- Modify: `src/faiss/mod.rs`
  - Export `IvfExRaBitqIndex` and config; gate any FFI module if added.
- Modify: `src/lib.rs`
  - Re-export the new public index/config types.
- Modify: `src/api/index.rs`
  - Add `IndexType::IvfExRaBitq`, parser aliases, and user-facing params.
- Modify: `src/api/legal_matrix.rs`
  - Add legal combinations for `IvfExRaBitq`; first-class metric support is `L2` only.
- Create: `src/faiss/exrabitq_ffi.rs`
  - C-facing build/search/save/load wrappers, modeled after `src/faiss/rabitq_ffi.rs`.
- Create: `tests/test_exrabitq_quantizer.rs`
  - Public codec and rotator tests.
- Create: `tests/test_exrabitq_fastscan.rs`
  - Layout, scalar fast scan, and SIMD parity tests.
- Create: `tests/test_ivf_exrabitq.rs`
  - End-to-end train/add/search/save/load tests plus small recall checks against brute-force `L2`.
- Create: `examples/exrabitq_compare.rs`
  - Dedicated local screen benchmark harness for accuracy/throughput against brute-force and selected baselines.
- Modify: `task-progress.md`
  - Record the new local screen line and final `screen_result`.
- Modify: `RELEASE_NOTES.md`
  - Add a line only if the screen is promoted or intentionally shipped.

## Chunk 1: Bootstrap the Screen Line and Public API Contract

### Task 1: Open a fresh local screen line before touching implementation code

**Files:**
- Modify: `task-progress.md`

- [ ] **Step 1: Add a new session entry for the screen**

Add a new section shaped like:

```markdown
### Session XXX - 2026-03-29
- Focus: `extended-rabitq-screen`
- Mode:
  - `screen`
- Hypothesis:
  - A parallel Extended-RaBitQ stack can land cleanly without breaking the existing `RaBitQEncoder` or `IvfRaBitqIndex`.
- Expected mechanism:
  - new `src/quantization/exrabitq/` family
  - new `src/faiss/ivf_exrabitq.rs`
  - scalar correctness first, AVX512 fast scan second
```

- [ ] **Step 2: Save the stub without `screen_result=`**

Do not mark success or failure yet. That only happens after local and remote verification.

- [ ] **Step 3: Commit the bootstrap note**

```bash
git add task-progress.md
git commit -m "docs(screen): open Extended-RaBitQ local screen line"
```

### Task 2: Lock the new public type and param surface with failing tests

**Files:**
- Modify: `src/api/index.rs`
- Modify: `src/api/legal_matrix.rs`

- [ ] **Step 1: Add failing `IndexType` parser tests**

Extend `src/api/index.rs` tests with:

```rust
#[test]
fn test_index_type_parses_ivf_exrabitq_aliases() {
    assert_eq!("ivf_exrabitq".parse::<IndexType>().unwrap(), IndexType::IvfExRaBitq);
    assert_eq!("ivf-exrabitq".parse::<IndexType>().unwrap(), IndexType::IvfExRaBitq);
    assert_eq!("extended-rabitq".parse::<IndexType>().unwrap(), IndexType::IvfExRaBitq);
}

#[test]
fn test_index_params_accept_exrabitq_fields() {
    let params = IndexParams {
        exrabitq_bits_per_dim: Some(4),
        exrabitq_use_high_accuracy_scan: Some(true),
        exrabitq_rerank_k: Some(128),
        exrabitq_rotation_seed: Some(17),
        ..Default::default()
    };
    assert_eq!(params.exrabitq_bits_per_dim, Some(4));
    assert_eq!(params.exrabitq_rerank_k, Some(128));
}
```

- [ ] **Step 2: Add a failing legal-matrix test**

Add to `src/api/legal_matrix.rs`:

```rust
#[test]
fn test_validate_index_config_ivf_exrabitq_l2_only() {
    assert!(validate_index_config(IndexType::IvfExRaBitq, DataType::Float, MetricType::L2).is_ok());
    assert!(validate_index_config(IndexType::IvfExRaBitq, DataType::Float, MetricType::Ip).is_err());
    assert!(validate_index_config(IndexType::IvfExRaBitq, DataType::Binary, MetricType::Hamming).is_err());
}
```

- [ ] **Step 3: Run the new tests to verify red**

```bash
cargo test --lib test_index_type_parses_ivf_exrabitq_aliases -- --nocapture
cargo test --lib test_index_params_accept_exrabitq_fields -- --nocapture
cargo test --lib test_validate_index_config_ivf_exrabitq_l2_only -- --nocapture
```

Expected: FAIL because the new enum variant, aliases, and params do not exist yet.

- [ ] **Step 4: Implement the minimal API surface**

In `src/api/index.rs`:

- add `IndexType::IvfExRaBitq`
- add parser aliases
- extend `IndexParams` with:

```rust
pub exrabitq_bits_per_dim: Option<usize>,
pub exrabitq_use_high_accuracy_scan: Option<bool>,
pub exrabitq_rerank_k: Option<usize>,
pub exrabitq_rotation_seed: Option<u64>,
```

In `src/api/legal_matrix.rs`:

- add `IvfExRaBitq` for `Float`, `Float16`, `BFloat16`
- allow `MetricType::L2` only for the first implementation

- [ ] **Step 5: Re-run the tests to verify green**

Run the three commands above again. Expected: PASS.

- [ ] **Step 6: Commit the API contract**

```bash
git add src/api/index.rs src/api/legal_matrix.rs
git commit -m "feat(api): add Extended-RaBitQ index contract"
```

## Chunk 2: Build the Core Codec and Rotation Layer

### Task 3: Add failing public tests for config, rotation, and quantization

**Files:**
- Create: `tests/test_exrabitq_quantizer.rs`
- Modify: `src/quantization/mod.rs`

- [ ] **Step 1: Add config and determinism tests**

Create tests like:

```rust
use knowhere_rs::quantization::exrabitq::{ExRaBitQConfig, ExRaBitQQuantizer};

#[test]
fn test_exrabitq_config_pads_dim_to_multiple_of_64() {
    let cfg = ExRaBitQConfig::new(768, 4).unwrap();
    assert_eq!(cfg.padded_dim(), 768);
    let cfg = ExRaBitQConfig::new(770, 4).unwrap();
    assert_eq!(cfg.padded_dim(), 832);
}

#[test]
fn test_exrabitq_config_rejects_unsupported_bits() {
    assert!(ExRaBitQConfig::new(768, 6).is_err());
}

#[test]
fn test_exrabitq_rotation_is_seed_deterministic() {
    let cfg = ExRaBitQConfig::new(96, 4).unwrap().with_rotation_seed(7);
    let q1 = ExRaBitQQuantizer::new(cfg.clone()).unwrap();
    let q2 = ExRaBitQQuantizer::new(cfg).unwrap();
    assert_eq!(q1.rotation_matrix(), q2.rotation_matrix());
}
```

- [ ] **Step 2: Add a failing `fast_quantize` dominance test**

Add a reference-only test:

```rust
#[test]
fn test_fast_quantize_is_not_worse_than_reference_greedy() {
    let cfg = ExRaBitQConfig::new(64, 4).unwrap().with_rotation_seed(11);
    let q = ExRaBitQQuantizer::new(cfg).unwrap();
    let v = random_unit_vector(64, 123);
    let fast = q.fast_quantize_for_test(&v);
    let greedy = q.greedy_quantize_for_test(&v, 3);
    assert!(fast.objective + 1e-6 >= greedy.objective);
}
```

- [ ] **Step 3: Run the test file to verify red**

```bash
cargo test --test test_exrabitq_quantizer -- --nocapture
```

Expected: FAIL because the module and public types do not exist yet.

- [ ] **Step 4: Implement the config and rotator files**

Create:

- `src/quantization/exrabitq/config.rs`
- `src/quantization/exrabitq/rotator.rs`
- `src/quantization/exrabitq/mod.rs`

Required behaviors:

- supported total bits: `3, 4, 5, 7, 8, 9`
- `short_bits` fixed at `1`
- `ex_bits = bits - 1`
- padded dimension rounded up to the next multiple of `64`
- deterministic dense orthogonal rotation with stored transpose

- [ ] **Step 5: Implement the quantizer file**

Create `src/quantization/exrabitq/quantizer.rs` with:

- residual preprocessing around a centroid
- unit-vector normalization on the transformed residual
- `fast_quantize`
- short-code builder
- long-code builder
- `ExFactor` carrying the correction term used in rerank
- reference decode helpers for tests only

- [ ] **Step 6: Export the module**

Modify `src/quantization/mod.rs` to export:

```rust
pub mod exrabitq;
pub use exrabitq::{ExFactor, ExRaBitQConfig, ExRaBitQQuantizer, ExRaBitQRotator};
```

- [ ] **Step 7: Re-run the codec tests**

```bash
cargo test --test test_exrabitq_quantizer -- --nocapture
```

Expected: PASS.

- [ ] **Step 8: Commit the codec layer**

```bash
git add src/quantization/mod.rs src/quantization/exrabitq tests/test_exrabitq_quantizer.rs
git commit -m "feat(exrabitq): add config rotation and quantizer core"
```

## Chunk 3: Add Block Layout, Scalar Fast Scan, and Rerank

### Task 4: Lock the cluster layout and scalar searcher with failing tests

**Files:**
- Create: `tests/test_exrabitq_fastscan.rs`
- Create: `src/quantization/exrabitq/layout.rs`
- Create: `src/quantization/exrabitq/fastscan.rs`
- Create: `src/quantization/exrabitq/searcher.rs`

- [ ] **Step 1: Add a failing block-layout roundtrip test**

```rust
#[test]
fn test_short_code_layout_roundtrips_blockwise() {
    let fixture = small_cluster_fixture();
    let layout = build_layout(&fixture.codes, &fixture.long_codes, &fixture.factors, &fixture.ids);
    assert_eq!(layout.len(), fixture.ids.len());
    for i in 0..fixture.ids.len() {
        assert_eq!(layout.id_at(i), fixture.ids[i]);
    }
}
```

- [ ] **Step 2: Add a failing scalar fast-scan parity test**

```rust
#[test]
fn test_scalar_fastscan_matches_reference_lower_bounds() {
    let fixture = small_cluster_fixture();
    let state = fixture.quantizer.precompute_query_state(&fixture.query, fixture.centroid());
    let fast = scalar_scan_cluster(&state, &fixture.layout, 32);
    let slow = reference_scan_cluster(&state, &fixture.raw_entries, 32);
    assert_eq!(fast.ids(), slow.ids());
}
```

- [ ] **Step 3: Add a failing rerank parity test**

```rust
#[test]
fn test_rerank_matches_reference_distance_formula() {
    let fixture = small_cluster_fixture();
    let state = fixture.quantizer.precompute_query_state(&fixture.query, fixture.centroid());
    let candidates = reference_scan_cluster(&state, &fixture.raw_entries, 16);
    let reranked = rerank_candidates(&state, &fixture.layout, &candidates);
    let brute = brute_force_cluster(&fixture.query, &fixture.raw_vectors);
    assert_eq!(reranked[0].0, brute[0].0);
}
```

- [ ] **Step 4: Run the fast-scan test file to verify red**

```bash
cargo test --test test_exrabitq_fastscan -- --nocapture
```

Expected: FAIL because layout and searcher code do not exist yet.

- [ ] **Step 5: Implement `layout.rs`**

Required layout:

- `FAST_SIZE = 32`
- block-transposed 1-bit short-code storage per cluster
- packed long codes stored contiguously per vector
- `Vec<ExFactor>` aligned with ids
- helper accessors that never allocate in the hot path

- [ ] **Step 6: Implement scalar query preprocessing and scan**

In `fastscan.rs`, add:

- query preprocessing into the scalar state
- LUT or equivalent lower-bound state for the short code
- scalar cluster scan over block layout
- top-k candidate maintenance without heap churn

- [ ] **Step 7: Implement `searcher.rs`**

Add:

- candidate selection over the scalar scan output
- rerank using long code + factor math
- reference helpers used only by tests

- [ ] **Step 8: Re-run the fast-scan tests**

```bash
cargo test --test test_exrabitq_fastscan -- --nocapture
```

Expected: PASS.

- [ ] **Step 9: Commit the scalar search layer**

```bash
git add src/quantization/exrabitq tests/test_exrabitq_fastscan.rs
git commit -m "feat(exrabitq): add scalar block layout fast scan and rerank"
```

## Chunk 4: Integrate the Codec into a New IVF Index

### Task 5: Add failing end-to-end IVF tests

**Files:**
- Create: `tests/test_ivf_exrabitq.rs`
- Create: `src/faiss/ivf_exrabitq.rs`
- Modify: `src/faiss/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add a failing train/add/search test**

```rust
#[test]
fn test_ivf_exrabitq_train_add_search() {
    let (data, ids, queries) = synthetic_l2_fixture(1000, 32, 10, 7);
    let cfg = IvfExRaBitqConfig::new(32, 32, 4).with_nprobe(4).with_rerank_k(64);
    let mut index = IvfExRaBitqIndex::new(cfg);
    index.train(&data).unwrap();
    index.add(&data, Some(&ids)).unwrap();

    let req = SearchRequest { top_k: 10, nprobe: 4, filter: None, params: None, radius: None };
    let res = index.search(&queries[..32], &req).unwrap();
    assert_eq!(res.ids.len(), 10);
}
```

- [ ] **Step 2: Add a failing save/load test**

```rust
#[test]
fn test_ivf_exrabitq_save_load_roundtrip() {
    let (mut index, query) = trained_small_index();
    let path = temp_index_path("ivf_exrabitq_roundtrip.bin");
    index.save(&path).unwrap();
    let loaded = IvfExRaBitqIndex::load(&path).unwrap();
    let req = SearchRequest { top_k: 5, nprobe: 2, filter: None, params: None, radius: None };
    assert_eq!(index.search(&query, &req).unwrap().ids, loaded.search(&query, &req).unwrap().ids);
}
```

- [ ] **Step 3: Add a failing small recall test**

```rust
#[test]
fn test_ivf_exrabitq_recall_at_10_is_reasonable() {
    let (data, ids, queries) = synthetic_l2_fixture(500, 32, 20, 9);
    let gt = brute_force_l2(&data, &ids, &queries, 10);
    let index = build_small_exrabitq_index(&data, &ids);
    let recall = measure_recall_at_10(&index, &queries, &gt);
    assert!(recall >= 0.5);
}
```

- [ ] **Step 4: Run the IVF test file to verify red**

```bash
cargo test --test test_ivf_exrabitq -- --nocapture
```

Expected: FAIL because the new index does not exist yet.

- [ ] **Step 5: Implement `src/faiss/ivf_exrabitq.rs`**

Required pieces:

- `IvfExRaBitqConfig`
- `IvfExRaBitqIndex`
- KMeans coarse training
- assignment of vectors to clusters
- per-cluster layout construction using the codec from chunk 3
- search over `nprobe` clusters using scalar searcher
- save/load
- `IndexTrait` impl

- [ ] **Step 6: Export the new index**

Modify:

- `src/faiss/mod.rs`
- `src/lib.rs`

Export:

```rust
pub use ivf_exrabitq::{IvfExRaBitqConfig, IvfExRaBitqIndex};
```

- [ ] **Step 7: Re-run the end-to-end tests**

```bash
cargo test --test test_ivf_exrabitq -- --nocapture
```

Expected: PASS.

- [ ] **Step 8: Commit the new IVF index**

```bash
git add src/faiss/ivf_exrabitq.rs src/faiss/mod.rs src/lib.rs tests/test_ivf_exrabitq.rs
git commit -m "feat(faiss): add IVF Extended-RaBitQ index"
```

## Chunk 5: Add AVX512 Fast Scan with Runtime Dispatch

### Task 6: Add failing SIMD parity tests before touching the kernel

**Files:**
- Modify: `tests/test_exrabitq_fastscan.rs`
- Modify: `src/quantization/exrabitq/fastscan.rs`

- [ ] **Step 1: Add AVX512-parity tests for supported hot widths**

Extend `tests/test_exrabitq_fastscan.rs` with:

```rust
#[test]
fn test_fastscan_avx512_matches_scalar_for_4bit() {
    if !std::is_x86_feature_detected!("avx512bw") {
        return;
    }
    let fixture = medium_cluster_fixture(4);
    let scalar = fixture.scalar_scan();
    let simd = fixture.simd_scan().unwrap();
    assert_eq!(scalar.ids(), simd.ids());
}

#[test]
fn test_fastscan_avx512_matches_scalar_for_8bit() {
    if !std::is_x86_feature_detected!("avx512bw") {
        return;
    }
    let fixture = medium_cluster_fixture(8);
    let scalar = fixture.scalar_scan();
    let simd = fixture.simd_scan().unwrap();
    assert_eq!(scalar.ids(), simd.ids());
}
```

- [ ] **Step 2: Run the parity tests to verify red**

```bash
cargo test --test test_exrabitq_fastscan test_fastscan_avx512_matches_scalar_for_4bit -- --nocapture
cargo test --test test_exrabitq_fastscan test_fastscan_avx512_matches_scalar_for_8bit -- --nocapture
```

Expected: FAIL because no SIMD kernel exists yet.

- [ ] **Step 3: Implement runtime dispatch in `fastscan.rs`**

Add:

- scalar default path
- x86_64 runtime detection using `is_x86_feature_detected!`
- AVX512 entry points guarded with `#[target_feature(enable = \"avx512bw,avx512f\")]`
- no allocations in the scan hot path

- [ ] **Step 4: Implement the AVX512 kernel**

Follow the existing repository pattern used by HVQ fast-scan code:

- reuse fixed-size block traversal
- keep query precompute outside the scan loop
- specialize the hot 4-bit and 8-bit cases first
- leave 3/5/7/9-bit support functionally correct via scalar fallback until a benchmark justifies more SIMD variants

- [ ] **Step 5: Re-run the fast-scan test file**

```bash
cargo test --test test_exrabitq_fastscan -- --nocapture
```

Expected: PASS. On non-AVX512 hosts, the parity tests should self-skip.

- [ ] **Step 6: Commit the SIMD fast-scan layer**

```bash
git add src/quantization/exrabitq/fastscan.rs tests/test_exrabitq_fastscan.rs
git commit -m "perf(exrabitq): add AVX512 fast scan dispatch"
```

## Chunk 6: FFI, Example Benchmark, and Authority Closure

### Task 7: Add public bindings and an isolated benchmark harness

**Files:**
- Create: `src/faiss/exrabitq_ffi.rs`
- Modify: `src/faiss/mod.rs`
- Create: `examples/exrabitq_compare.rs`

- [ ] **Step 1: Add failing FFI smoke tests**

Model them after `src/faiss/rabitq_ffi.rs`:

```rust
#[test]
fn test_ffi_build_search_exrabitq() {
    // build through C-style wrapper, search, and free
}
```

- [ ] **Step 2: Run the FFI smoke test to verify red**

```bash
cargo test --lib test_ffi_build_search_exrabitq -- --nocapture
```

Expected: FAIL because the wrapper does not exist yet.

- [ ] **Step 3: Implement `src/faiss/exrabitq_ffi.rs`**

Mirror the `rabitq_ffi` contract:

- build
- load
- save
- search
- batch search
- count/size/dim accessors
- free

- [ ] **Step 4: Export the FFI module**

In `src/faiss/mod.rs`, add:

```rust
#[cfg(feature = "ffi")]
pub mod exrabitq_ffi;
```

- [ ] **Step 5: Add the dedicated benchmark example**

Create `examples/exrabitq_compare.rs` that can:

- load a local dataset fixture
- build an `IvfExRaBitqIndex`
- report build time, recall@10, and scan/search QPS
- optionally compare against `IvfRaBitqIndex`, `IvfPqIndex`, and brute force

- [ ] **Step 6: Run the new local benchmark harness**

```bash
cargo run --example exrabitq_compare --release
```

Expected: completes locally on a small fixture and prints one `EXRABITQ` result row.

- [ ] **Step 7: Commit the public integration layer**

```bash
git add src/faiss/exrabitq_ffi.rs src/faiss/mod.rs examples/exrabitq_compare.rs
git commit -m "feat(ffi): add Extended-RaBitQ bindings and benchmark harness"
```

### Task 8: Run the repository gates and authority verification before closing the screen

**Files:**
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`

- [ ] **Step 1: Run the local quality gates**

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib --verbose
cargo test --tests --verbose
```

Expected: PASS locally.

- [ ] **Step 2: Push the current workspace to the authority machine**

```bash
bash init.sh
```

Expected: the script prints the resolved remote workspace and sync target.

- [ ] **Step 3: Run authoritative remote build and tests**

```bash
bash scripts/remote/build.sh --no-all-targets
bash scripts/remote/test.sh --command "cargo test --test test_exrabitq_quantizer -- --nocapture"
bash scripts/remote/test.sh --command "cargo test --test test_exrabitq_fastscan -- --nocapture"
bash scripts/remote/test.sh --command "cargo test --test test_ivf_exrabitq -- --nocapture"
```

Expected: PASS on the remote x86 machine.

- [ ] **Step 4: Run the remote benchmark harness**

```bash
bash scripts/remote/test.sh --command "cargo run --example exrabitq_compare --release"
```

Expected: prints the `EXRABITQ` result row plus any baseline rows the harness emits.

- [ ] **Step 5: Record the screen outcome**

Update `task-progress.md` with either:

```markdown
- screen_result=promote
```

or:

```markdown
- screen_result=hold
```

Use `promote` only if remote build, tests, and benchmark are all complete.

- [ ] **Step 6: Update `RELEASE_NOTES.md` only if promoted**

Add a single line summarizing the new index family and its validation status.

- [ ] **Step 7: Commit the closure**

```bash
git add task-progress.md RELEASE_NOTES.md
git commit -m "docs(exrabitq): record screen closure"
```

## Notes and Guardrails

- Keep `src/quantization/rabitq.rs` and `src/faiss/ivf_rabitq.rs` intact during the first pass. They remain the compatibility path while `exrabitq` is validated.
- Do not promise `IP` or `Cosine` support in the first closure. The upstream paper and repository target Euclidean search; `MetricType::L2` is the correct first-class target.
- Prefer scalar correctness before SIMD. If AVX512 work slips, the screen can still close as `hold` with a correct scalar implementation and a clear benchmark gap.
- Reuse existing repository patterns where they fit, especially the fast-scan state and runtime dispatch style already present in HVQ code, but do not contort the math to share structs that do not match Extended-RaBitQ semantics.
- Keep hot paths allocation-free: query preprocessing once per query, cluster layout once at build time, and no `Vec` allocation inside per-code scan loops.

## References

- Upstream repository: `VectorDB-NTU/Extended-RaBitQ`
- Paper: `Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search`
