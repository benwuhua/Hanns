# TurboQuant knowhere-rs Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TurboQuant to `knowhere-rs` as a new quantization family plus a new `IvfTurboQuantIndex`, validating it as a local `screen` line before any long-task durable reopen or authority claim.

**Architecture:** Keep the change aligned with the repository’s current concrete-per-index style. Implement a new `src/quantization/turboquant/` family first, then add `src/faiss/ivf_turboquant.rs` as a peer to `ivfpq.rs`, `ivf_sq8.rs`, and `ivf_rabitq.rs`, and only then wire the new index through `src/api/index.rs` and `src/api/legal_matrix.rs`. Treat this work as a fresh local `screen` because the current long-task program is already closed; do not touch `feature-list.json` unless the local screen later ends with `screen_result=promote`.

**Tech Stack:** Rust 2021, existing `KMeans` coarse quantizer, current quantization modules under `src/quantization/`, existing IVF/refine patterns in `src/faiss/ivfpq.rs` and `src/faiss/ivf_rabitq.rs`, `nalgebra` for dense reference rotation, `rand` for seeded randomness, local `cargo test` / `cargo fmt` / `cargo clippy`

---

## File Map

- Create: `src/quantization/turboquant/mod.rs`
  - Public exports for TurboQuant types.
- Create: `src/quantization/turboquant/config.rs`
  - `TurboQuantConfig`, `TurboQuantMode`, `TurboRotationBackend`, validation, code-size helpers.
- Create: `src/quantization/turboquant/rotation.rs`
  - Seeded rotation backends: dense reference and structured production backend.
- Create: `src/quantization/turboquant/codebook.rs`
  - Scalar centroid tables by bit-width and nearest-centroid search.
- Create: `src/quantization/turboquant/packed.rs`
  - Bit packing/unpacking for `b`-bit coordinate indices and 1-bit sidecars.
- Create: `src/quantization/turboquant/mse.rs`
  - `TurboQuantMse` encode/decode path.
- Create: `src/quantization/turboquant/prod.rs`
  - `TurboQuantProd` residual-correction path for inner-product estimation.
- Modify: `src/quantization/mod.rs`
  - Export TurboQuant family.
- Create: `src/faiss/ivf_turboquant.rs`
  - New IVF index backed by TurboQuant residual codes.
- Modify: `src/faiss/mod.rs`
  - Export `IvfTurboQuantIndex`.
- Modify: `src/api/index.rs`
  - Add `IndexType::IvfTurboQuant`, string aliases, and TurboQuant params.
- Modify: `src/api/legal_matrix.rs`
  - Add legal combinations for `IvfTurboQuant`.
- Create: `tests/test_turboquant.rs`
  - Public TurboQuant codec tests.
- Create: `tests/test_ivf_turboquant.rs`
  - End-to-end train/add/search/save/load tests for the new index.
- Create: `tests/bench_turboquant_accuracy.rs`
  - Dedicated local screen harness for TurboQuant vs flat and, optionally, existing compressed baselines.
- Modify: `tests/test_ivf_index_trait.rs`
  - Add `IvfTurboQuantIndex` trait-path coverage if the new index implements the unified trait.
- Modify: `task-progress.md`
  - Record the local screen result and next recommendation.
- Modify: `RELEASE_NOTES.md`
  - Record the new TurboQuant screen line if the implementation survives local verification.

## Chunk 1: Screen Bootstrap and Public API Contract

### Task 1: Mark the work as a new local screen line before changing code

**Files:**
- Modify: `task-progress.md`

- [ ] **Step 1: Add a session stub for the TurboQuant line**

Add a new session entry in `task-progress.md` before implementation starts. The entry should state:

```markdown
### Session XXX - 2026-03-26
- Focus: `turboquant-ivf-screen`
- Mode:
  - `screen`
- Hypothesis:
  - A TurboQuant-backed IVF residual codec can fit the current `knowhere-rs` architecture cleanly without a repository-wide quantizer trait refactor.
- Expected mechanism:
  - new `src/quantization/turboquant/` family
  - new `src/faiss/ivf_turboquant.rs`
  - local correctness + smoke-search verification first
```

- [ ] **Step 2: Save the stub without a screen result yet**

Do not add `screen_result=` yet. That result is only written after local verification at the end of the plan.

- [ ] **Step 3: Commit the bootstrap-only durable note**

```bash
git add task-progress.md
git commit -m "docs(screen): open TurboQuant IVF local screen line"
```

### Task 2: Lock the new index type and config contract with failing tests

**Files:**
- Modify: `src/api/index.rs`
- Modify: `src/api/legal_matrix.rs`

- [ ] **Step 1: Write failing API/config tests in `src/api/index.rs`**

Extend the existing `mod tests` in `src/api/index.rs` with coverage like:

```rust
#[test]
fn test_index_type_parses_ivf_turboquant_aliases() {
    assert_eq!("ivf_turboquant".parse::<IndexType>().unwrap(), IndexType::IvfTurboQuant);
    assert_eq!("ivf-turboquant".parse::<IndexType>().unwrap(), IndexType::IvfTurboQuant);
    assert_eq!("turboquant".parse::<IndexType>().unwrap(), IndexType::IvfTurboQuant);
}

#[test]
fn test_index_params_accept_turboquant_fields() {
    let params = IndexParams {
        turbo_bits_per_dim: Some(4),
        turbo_mode: Some("mse".to_string()),
        turbo_use_residual: Some(true),
        turbo_rotation_seed: Some(42),
        turbo_rotation_backend: Some("structured".to_string()),
        turbo_reorder_k: Some(64),
        ..Default::default()
    };

    assert_eq!(params.turbo_bits_per_dim, Some(4));
    assert_eq!(params.turbo_mode.as_deref(), Some("mse"));
}
```

- [ ] **Step 2: Write a failing legal-matrix test in `src/api/legal_matrix.rs`**

Add a test like:

```rust
#[test]
fn test_validate_index_config_ivf_turboquant() {
    assert!(validate_index_config(IndexType::IvfTurboQuant, DataType::Float, MetricType::Ip).is_ok());
    assert!(validate_index_config(IndexType::IvfTurboQuant, DataType::Float, MetricType::Cosine).is_ok());
    assert!(validate_index_config(IndexType::IvfTurboQuant, DataType::Binary, MetricType::Hamming).is_err());
}
```

- [ ] **Step 3: Run the API tests to verify red**

Run:

```bash
cargo test --lib test_index_type_parses_ivf_turboquant_aliases -- --nocapture
cargo test --lib test_index_params_accept_turboquant_fields -- --nocapture
cargo test --lib test_validate_index_config_ivf_turboquant -- --nocapture
```

Expected: FAIL because the new enum variant, aliases, params, and legal combinations do not exist yet.

- [ ] **Step 4: Implement the minimal API/config changes**

In `src/api/index.rs`:

- add `IndexType::IvfTurboQuant`
- add `FromStr` aliases
- extend `IndexParams` with:

```rust
pub turbo_bits_per_dim: Option<usize>,
pub turbo_mode: Option<String>,
pub turbo_use_residual: Option<bool>,
pub turbo_rotation_seed: Option<u64>,
pub turbo_rotation_backend: Option<String>,
pub turbo_reorder_k: Option<usize>,
```

In `src/api/legal_matrix.rs`:

- add `IvfTurboQuant` for `Float`, `Float16`, `BFloat16`
- allow `L2`, `Ip`, `Cosine`

- [ ] **Step 5: Re-run the API tests to verify green**

Run:

```bash
cargo test --lib test_index_type_parses_ivf_turboquant_aliases -- --nocapture
cargo test --lib test_index_params_accept_turboquant_fields -- --nocapture
cargo test --lib test_validate_index_config_ivf_turboquant -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Commit the API contract**

```bash
git add src/api/index.rs src/api/legal_matrix.rs task-progress.md
git commit -m "feat(api): add TurboQuant index config contract"
```

## Chunk 2: TurboQuant Core MSE Codec

### Task 3: Add public TurboQuant tests that prove the MSE codec does not exist yet

**Files:**
- Create: `tests/test_turboquant.rs`
- Modify: `src/quantization/mod.rs`

- [ ] **Step 1: Write failing codec tests**

Create `tests/test_turboquant.rs` with focused public-contract tests:

```rust
use knowhere_rs::quantization::{TurboQuantConfig, TurboQuantMode, TurboRotationBackend, TurboQuantMse};

#[test]
fn test_turboquant_mse_encode_decode_shape() {
    let cfg = TurboQuantConfig::new(8, 4, TurboQuantMode::Mse)
        .with_rotation_backend(TurboRotationBackend::DenseOrthogonal)
        .with_rotation_seed(7);
    let codec = TurboQuantMse::new(cfg).unwrap();
    let vector = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];

    let code = codec.encode(&vector).unwrap();
    let decoded = codec.decode(&code).unwrap();

    assert_eq!(decoded.len(), vector.len());
    assert_eq!(code.len(), codec.code_size_bytes());
}

#[test]
fn test_turboquant_mse_seed_is_deterministic() {
    let cfg = TurboQuantConfig::new(8, 4, TurboQuantMode::Mse)
        .with_rotation_backend(TurboRotationBackend::DenseOrthogonal)
        .with_rotation_seed(42);
    let a = TurboQuantMse::new(cfg.clone()).unwrap();
    let b = TurboQuantMse::new(cfg).unwrap();
    let vector = vec![0.25; 8];

    assert_eq!(a.encode(&vector).unwrap(), b.encode(&vector).unwrap());
}

#[test]
fn test_turboquant_mse_higher_bitwidth_not_worse_on_fixture() {
    let data = vec![
        0.1, -0.2, 0.3, -0.4,
        -0.1, 0.2, -0.3, 0.4,
    ];
    let low = TurboQuantMse::new(TurboQuantConfig::new(4, 2, TurboQuantMode::Mse)).unwrap();
    let high = TurboQuantMse::new(TurboQuantConfig::new(4, 4, TurboQuantMode::Mse)).unwrap();

    let low_err = low.reconstruction_mse(&data).unwrap();
    let high_err = high.reconstruction_mse(&data).unwrap();
    assert!(high_err <= low_err + 1e-6);
}
```

- [ ] **Step 2: Run the new tests to verify red**

Run:

```bash
cargo test --test test_turboquant -- --nocapture
```

Expected: FAIL because TurboQuant types are not exported and the codec does not exist yet.

### Task 4: Implement the minimal TurboQuant MSE path

**Files:**
- Create: `src/quantization/turboquant/mod.rs`
- Create: `src/quantization/turboquant/config.rs`
- Create: `src/quantization/turboquant/rotation.rs`
- Create: `src/quantization/turboquant/codebook.rs`
- Create: `src/quantization/turboquant/packed.rs`
- Create: `src/quantization/turboquant/mse.rs`
- Modify: `src/quantization/mod.rs`

- [ ] **Step 1: Add config and export surface**

Implement public types along these lines:

```rust
pub enum TurboQuantMode {
    Mse,
    Prod,
}

pub enum TurboRotationBackend {
    DenseOrthogonal,
    StructuredOrthogonal,
}

pub struct TurboQuantConfig {
    pub dim: usize,
    pub bits_per_dim: u8,
    pub mode: TurboQuantMode,
    pub rotation_backend: TurboRotationBackend,
    pub rotation_seed: u64,
    pub normalize_for_cosine: bool,
}
```

Include `new(...)`, builder helpers, `validate()`, and a `code_size_bits()` / `code_size_bytes()` helper.

- [ ] **Step 2: Add packed-code helpers**

Implement bit-packing primitives for non-byte-aligned coordinate codes, for example:

```rust
pub fn pack_indices(indices: &[u16], bits_per_dim: u8) -> Vec<u8> { /* ... */ }
pub fn unpack_indices(bytes: &[u8], dim: usize, bits_per_dim: u8) -> Vec<u16> { /* ... */ }
```

Also add unit tests local to `packed.rs` for `1/2/4/8` bits.

- [ ] **Step 3: Add rotation backends**

Implement:

- a dense reference path using a seeded orthogonal matrix
- a structured path with deterministic seeded transforms

Keep the interface narrow:

```rust
pub trait Rotation {
    fn apply(&self, x: &[f32]) -> Vec<f32>;
    fn apply_transpose(&self, y: &[f32]) -> Vec<f32>;
}
```

The dense backend may use `nalgebra`; the structured backend should be a production-oriented deterministic transform, even if the first version is simple.

- [ ] **Step 4: Add deterministic centroid tables**

Implement centroid lookup in `codebook.rs`:

```rust
pub struct ScalarCodebook {
    pub bits_per_dim: u8,
    pub centroids: Vec<f32>,
}

impl ScalarCodebook {
    pub fn nearest_index(&self, value: f32) -> u16 { /* ... */ }
    pub fn centroid(&self, idx: u16) -> f32 { /* ... */ }
}
```

For `v1`, use fixed deterministic tables for the supported bit-widths rather than dataset-trained centroids.

- [ ] **Step 5: Implement `TurboQuantMse`**

Implement the public path:

```rust
pub struct TurboQuantMse { /* cfg + rotation + codebook */ }

impl TurboQuantMse {
    pub fn new(cfg: TurboQuantConfig) -> Result<Self> { /* ... */ }
    pub fn encode(&self, x: &[f32]) -> Result<Vec<u8>> { /* rotate -> quantize -> pack */ }
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>> { /* unpack -> centroids -> inverse rotate */ }
    pub fn code_size_bytes(&self) -> usize { /* ... */ }
    pub fn reconstruction_mse(&self, data: &[f32]) -> Result<f32> { /* helper for tests */ }
}
```

- [ ] **Step 6: Re-run the TurboQuant MSE tests**

Run:

```bash
cargo test --test test_turboquant -- --nocapture
```

Expected: PASS for the MSE tests.

- [ ] **Step 7: Commit the MSE codec**

```bash
git add src/quantization/mod.rs src/quantization/turboquant tests/test_turboquant.rs
git commit -m "feat(quantization): add TurboQuant MSE codec"
```

## Chunk 3: TurboQuant Prod Codec

### Task 5: Lock the inner-product correction contract with failing tests

**Files:**
- Modify: `tests/test_turboquant.rs`
- Create: `src/quantization/turboquant/prod.rs`

- [ ] **Step 1: Add failing `prod` tests**

Extend `tests/test_turboquant.rs` with:

```rust
use knowhere_rs::quantization::TurboQuantProd;

#[test]
fn test_turboquant_prod_rejects_l2_mode() {
    let cfg = TurboQuantConfig::new(8, 4, TurboQuantMode::Prod);
    let codec = TurboQuantProd::new(cfg, MetricType::L2);
    assert!(codec.is_err());
}

#[test]
fn test_turboquant_prod_inner_product_estimate_is_nearly_unbiased() {
    let cfg = TurboQuantConfig::new(16, 4, TurboQuantMode::Prod)
        .with_rotation_seed(123);
    let codec = TurboQuantProd::new(cfg, MetricType::Ip).unwrap();
    let x = vec![0.25; 16];
    let y = vec![0.5; 16];
    let code = codec.encode(&x).unwrap();

    let est = codec.estimate_inner_product(&y, &code).unwrap();
    let gt: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    assert!((est - gt).abs() < 0.25);
}
```

Use a tolerance wide enough for a deterministic first implementation, but still strict enough to prevent a degenerate no-op estimator.

- [ ] **Step 2: Run the prod-focused tests to verify red**

Run:

```bash
cargo test --test test_turboquant test_turboquant_prod -- --nocapture
```

Expected: FAIL because the `prod` codec does not exist.

### Task 6: Implement the minimal `prod` path for inner-product estimation

**Files:**
- Create: `src/quantization/turboquant/prod.rs`
- Modify: `src/quantization/turboquant/mod.rs`
- Modify: `src/quantization/turboquant/packed.rs`
- Modify: `tests/test_turboquant.rs`

- [ ] **Step 1: Add the packed residual sidecar**

Extend the packed helpers with a 1-bit sign stream:

```rust
pub fn pack_bits(bits: &[bool]) -> Vec<u8> { /* ... */ }
pub fn unpack_bits(bytes: &[u8], len: usize) -> Vec<bool> { /* ... */ }
```

- [ ] **Step 2: Implement `TurboQuantProd`**

Implement a narrow `v1` codec:

```rust
pub struct TurboQuantProdCode {
    pub mse_code: Vec<u8>,
    pub residual_signs: Vec<u8>,
}

pub struct TurboQuantProd { /* mse codec + prod metadata */ }

impl TurboQuantProd {
    pub fn new(cfg: TurboQuantConfig, metric: MetricType) -> Result<Self> { /* reject L2 */ }
    pub fn encode(&self, x: &[f32]) -> Result<TurboQuantProdCode> { /* MSE + residual sign sidecar */ }
    pub fn estimate_inner_product(&self, query: &[f32], code: &TurboQuantProdCode) -> Result<f32> { /* base estimate + correction */ }
}
```

This first implementation does not need to reproduce every constant from the paper perfectly; it must preserve the intended shape:

- MSE quantization first
- residual correction second
- inner-product-focused path only

- [ ] **Step 3: Re-run the `prod` tests**

Run:

```bash
cargo test --test test_turboquant test_turboquant_prod -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Commit the prod codec**

```bash
git add src/quantization/turboquant tests/test_turboquant.rs
git commit -m "feat(quantization): add TurboQuant prod codec"
```

## Chunk 4: IVF-TurboQuant Index

### Task 7: Add failing end-to-end index tests

**Files:**
- Create: `tests/test_ivf_turboquant.rs`
- Modify: `src/faiss/mod.rs`

- [ ] **Step 1: Write failing train/add/search tests**

Create `tests/test_ivf_turboquant.rs` with:

```rust
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::IvfTurboQuantIndex;

#[test]
fn test_ivf_turboquant_train_add_search_ip() {
    let dim = 8usize;
    let config = IndexConfig {
        index_type: IndexType::IvfTurboQuant,
        metric_type: MetricType::Ip,
        dim,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            nlist: Some(4),
            nprobe: Some(2),
            turbo_bits_per_dim: Some(4),
            turbo_mode: Some("mse".to_string()),
            turbo_rotation_backend: Some("structured".to_string()),
            turbo_rotation_seed: Some(42),
            turbo_reorder_k: Some(0),
            ..Default::default()
        },
    };

    let mut index = IvfTurboQuantIndex::new(&config).unwrap();
    let train = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    index.train(&train).unwrap();
    index.add(&train, None).unwrap();
    let result = index.search(&train[0..dim], &SearchRequest { top_k: 2, nprobe: 2, ..Default::default() }).unwrap();
    assert!(!result.ids.is_empty());
}

#[test]
fn test_ivf_turboquant_save_load_roundtrip() {
    // train -> add -> save -> load -> search returns non-empty result
}
```

- [ ] **Step 2: Run the new index tests to verify red**

Run:

```bash
cargo test --test test_ivf_turboquant -- --nocapture
```

Expected: FAIL because the index does not exist yet.

### Task 8: Implement the new IVF consumer

**Files:**
- Create: `src/faiss/ivf_turboquant.rs`
- Modify: `src/faiss/mod.rs`
- Modify: `tests/test_ivf_turboquant.rs`
- Modify: `tests/test_ivf_index_trait.rs`

- [ ] **Step 1: Scaffold the new index**

Follow the repository’s existing style:

```rust
pub struct IvfTurboQuantIndex {
    config: IndexConfig,
    dim: usize,
    nlist: usize,
    nprobe: usize,
    centroids: Vec<f32>,
    mode: TurboQuantMode,
    mse: Option<TurboQuantMse>,
    prod: Option<TurboQuantProd>,
    invlist_ids: Vec<Vec<i64>>,
    invlist_codes: Vec<Vec<u8>>,
    invlist_prod_sidecars: Vec<Vec<Vec<u8>>>,
    vectors: Vec<f32>,
    ids: Vec<i64>,
    trained: bool,
}
```

Keep the first version concrete and simple. Mirror `IvfPqIndex` and `IvfRaBitqIndex` storage patterns instead of introducing a new generic inverted-list abstraction.

- [ ] **Step 2: Implement `new()` and config parsing**

Parse:

- `nlist`
- `nprobe`
- `turbo_bits_per_dim`
- `turbo_mode`
- `turbo_rotation_backend`
- `turbo_rotation_seed`
- `turbo_reorder_k`

Reject malformed mode/backend strings early.

- [ ] **Step 3: Implement `train()`**

`train()` should:

- validate input shape
- train IVF centroids with `KMeans`
- initialize TurboQuant codec(s)
- clear inverted lists
- mark the index trained

Do not add a PQ-style fine training pass.

- [ ] **Step 4: Implement `add()`**

`add()` should:

- assign each vector to the nearest centroid
- compute residual
- encode residual with TurboQuant
- append to the matching inverted list
- optionally append exact vectors/ids for later rerank if reorder is enabled

- [ ] **Step 5: Implement `search()`**

For the first version:

- score coarse centroids
- probe `nprobe` lists
- compute query residual per probed centroid
- for `mse` mode, decode residuals and score approximately
- for `prod` mode, use the inner-product estimator
- return top-k merged candidates

Keep the first version correct and readable; do not prematurely add SIMD or custom heaps beyond current patterns already used in neighboring indices.

- [ ] **Step 6: Add a save/load path**

Mirror `IvfSq8Index` / `IvfRaBitqIndex` persistence shape:

- write config fields
- write centroids
- write TurboQuant config
- write per-list ids and packed codes
- write optional prod sidecars

- [ ] **Step 7: Re-run end-to-end index tests**

Run:

```bash
cargo test --test test_ivf_turboquant -- --nocapture
```

Expected: PASS.

- [ ] **Step 8: Extend unified index-trait coverage if applicable**

If the new index is exposed through the unified trait, add a focused regression to `tests/test_ivf_index_trait.rs` and run:

```bash
cargo test --test test_ivf_index_trait -- --nocapture
```

Expected: PASS.

- [ ] **Step 9: Commit the new IVF consumer**

```bash
git add src/faiss/mod.rs src/faiss/ivf_turboquant.rs tests/test_ivf_turboquant.rs tests/test_ivf_index_trait.rs
git commit -m "feat(ivf): add TurboQuant-backed IVF index"
```

## Chunk 5: Local Screen Verification and Durable Closure

### Task 9: Run the local screen verification suite

**Files:**
- Create: `tests/bench_turboquant_accuracy.rs`
- Modify: `RELEASE_NOTES.md`
- Modify: `task-progress.md`

- [ ] **Step 1: Add a dedicated local comparison harness**

Create `tests/bench_turboquant_accuracy.rs` with a narrow local-only comparison:

- `IvfTurboQuantIndex` at 4 bits
- compare against a flat ground truth on a small deterministic fixture
- optionally include one existing compressed baseline such as `IvfPqIndex` or `IvfRaBitqIndex`, but keep runtime short enough for routine local screen use

- [ ] **Step 2: Run focused correctness checks**

Run:

```bash
cargo test --test test_turboquant -- --nocapture
cargo test --test test_ivf_turboquant -- --nocapture
cargo test --test test_ivf_index_trait -- --nocapture
```

Expected: PASS.

- [ ] **Step 3: Run repository hygiene gates**

Run:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

Expected: PASS.

- [ ] **Step 4: Run one local screen comparison command**

Run either:

```bash
cargo test --test bench_turboquant_accuracy -- --nocapture
```

Expected: PASS and enough local evidence to classify the line as one of:

- `screen_result=promote`
- `screen_result=needs_more_local`
- `screen_result=reject`

### Task 10: Record the screen outcome without reopening durable tracked work

**Files:**
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`

- [ ] **Step 1: Record the exact commands and outcome in `task-progress.md`**

Append:

```markdown
- Verification:
  - local:
    - `cargo test --test test_turboquant -- --nocapture` -> `ok`
    - `cargo test --test test_ivf_turboquant -- --nocapture` -> `ok`
    - `cargo test --test test_ivf_index_trait -- --nocapture` -> `ok`
    - `cargo fmt --all -- --check` -> `ok`
    - `cargo clippy --all-targets --all-features -- -D warnings` -> `ok`
    - `cargo test --test bench_turboquant_accuracy -- --nocapture` -> `ok`
- Result:
  - `screen_result=...`
```

- [ ] **Step 2: Record the summary in `RELEASE_NOTES.md`**

Add a short entry stating:

- TurboQuant quantization family was added
- a new `IvfTurboQuantIndex` local screen line was implemented
- the line remains local-screen evidence only until explicitly promoted

- [ ] **Step 3: Do not touch `feature-list.json` unless the result is `promote` and the next session explicitly chooses to open tracked work**

This is the long-task guardrail for this repository. The current plan ends at local screen closure.

- [ ] **Step 4: Commit the local screen closure**

```bash
git add task-progress.md RELEASE_NOTES.md tests/bench_turboquant_accuracy.rs
git commit -m "docs(screen): record TurboQuant IVF local screen result"
```
