# TurboQuant knowhere-rs Integration Design

**Date:** 2026-03-26

## Goal

Define a concrete integration design for adding TurboQuant to `knowhere-rs` as a new quantization family and a new IVF-based consumer, without implementing code in this phase.

The design must:

- reflect the current `knowhere-rs` module layout and API patterns
- avoid premature repository-wide quantizer abstraction refactors
- preserve a clean path for later implementation of both `TurboQuantMse` and `TurboQuantProd`

## Requirement Sources

- User request in this session:
  - translate the paper into engineering structure
  - write an integration plan for `knowhere-rs`
  - current repository already contains `pq`, `rabitq`, `hvq`, `sq`, and `pca`
- Paper:
  - [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
  - PDF: https://arxiv.org/pdf/2504.19874

## Current Repository Context

`knowhere-rs` currently exposes multiple independent quantization implementations under `src/quantization/`, including:

- `pq.rs`
- `rabitq.rs`
- `sq.rs`
- `hvq.rs`
- `pca.rs`
- `opq.rs`
- `prq.rs`
- `rq.rs`

The upper-layer Faiss-style consumers do not currently share a single generic quantizer trait. Instead, each index owns or directly depends on a specific quantizer path:

- `IvfPqIndex` owns `ProductQuantizer`
- `IvfSq8Index` owns `ScalarQuantizer`
- `IvfRaBitqIndex` owns `RaBitQEncoder`
- DiskANN and AISAQ own custom compression/search paths

This means the repository already prefers concrete per-index integration over an abstract quantizer plugin architecture.

## Scope

This design covers:

- where TurboQuant should live in `src/quantization/`
- which first consumer should be added
- how the query/train/add/search data flow should look
- what API and config changes should be introduced
- what tests and rollout stages should exist

This design does not cover:

- implementation code
- benchmark claims
- remote-x86 authority validation
- immediate DiskANN integration
- a repository-wide unification of all quantizer interfaces

## Paper Summary Mapped to Engineering Constraints

The paper defines two related quantizers:

- `TurboQuantMse`
  - random rotation
  - per-coordinate nearest-centroid scalar quantization
  - dequantization by centroid lookup followed by inverse rotation
- `TurboQuantProd`
  - uses the MSE quantizer as the first stage
  - adds a 1-bit residual correction path for unbiased inner-product estimation

Algorithm 1 in the paper defines the MSE path as:

- rotate `x` by a random orthogonal transform `Π`
- quantize each rotated coordinate against a fixed `2^b` centroid set
- reconstruct by inverse rotation with `Π^T`

The paper positions TurboQuant as:

- data-oblivious
- online-friendly
- not dependent on training a PQ codebook
- strong for high-dimensional inner-product workloads

These properties imply that the most natural first `knowhere-rs` integration is not a replacement for existing PQ internals, but a new compression family with its own storage and query semantics.

## Design Decision Summary

### 1. Add TurboQuant as a new quantization family, not as a PQ variant

TurboQuant should not be merged into `ProductQuantizer` or `ScalarQuantizer`.

Reasons:

- PQ is codebook-trained and subvector-based; TurboQuant is coordinate-wise after random rotation
- SQ is simple scalar quantization, but TurboQuant also depends on a random rotation layer and, in `prod` mode, a residual correction sidecar
- forcing TurboQuant into an existing quantizer type would blur semantics and make tests harder to reason about

### 2. First consumer should be a new `IvfTurboQuantIndex`

The first concrete consumer should be a new IVF-style index rather than DiskANN or AISAQ.

Reasons:

- existing repository structure already has one-index-one-quantizer patterns
- IVF integration is lower-risk than modifying DiskANN search hot paths
- TurboQuant is naturally suitable for compressed residual storage in an IVF layout
- this creates a real end-to-end train/add/search path without forcing broad refactors

### 3. Do not start with a generic `Quantizer` trait refactor

A shared trait layer for PQ, SQ, RaBitQ, HVQ, and TurboQuant may eventually be useful, but it should not be the first step.

Reasons:

- current codebase does not rely on such an abstraction
- the trait would need to cover incompatible concepts:
  - reconstruction-oriented codecs
  - asymmetric distance computers
  - residual encoders
  - inner-product estimators
- it would expand scope beyond what the user asked for

### 4. Support `Ip` and `Cosine` first; keep `L2` secondary

TurboQuant is strongest, both theoretically and practically, on inner-product-preserving use cases.

Recommendation:

- `v1` primary support: `MetricType::Ip`, `MetricType::Cosine`
- `v1` optional support: `MetricType::L2` only through `TurboQuantMse`
- do not require `TurboQuantProd` to support `L2`

## Proposed Module Layout

### Quantization Layer

Add a new directory:

- `src/quantization/turboquant/`

Recommended files:

- `src/quantization/turboquant/mod.rs`
  - exports public types
- `src/quantization/turboquant/config.rs`
  - `TurboQuantConfig`
  - `TurboQuantMode`
  - `TurboRotationBackend`
- `src/quantization/turboquant/rotation.rs`
  - rotation backends
  - deterministic seeded setup
- `src/quantization/turboquant/codebook.rs`
  - fixed scalar centroid tables
  - nearest-centroid lookup
- `src/quantization/turboquant/packed.rs`
  - bit packing and unpacking
- `src/quantization/turboquant/mse.rs`
  - `TurboQuantMse`
  - encode/decode path
- `src/quantization/turboquant/prod.rs`
  - `TurboQuantProd`
  - residual 1-bit sidecar
  - inner-product estimator helpers

Modify:

- `src/quantization/mod.rs`
  - export TurboQuant types

### Index Layer

Add:

- `src/faiss/ivf_turboquant.rs`

Modify:

- `src/faiss/mod.rs`
  - export `IvfTurboQuantIndex`

### API Layer

Modify:

- `src/api/index.rs`
  - add `IndexType::IvfTurboQuant`
  - add `IndexParams` fields for TurboQuant
- `src/api/legal_matrix.rs`
  - allow `IvfTurboQuant` for float-family datatypes and `L2/Ip/Cosine`

### FFI Layer

Optional later step:

- `src/ffi.rs`

The first implementation should not require FFI support. Rust-native integration should come first.

## Quantization Architecture

### Core Types

Recommended top-level public types:

- `TurboQuantConfig`
- `TurboQuantMode`
  - `Mse`
  - `Prod`
- `TurboRotationBackend`
  - `DenseOrthogonal`
  - `StructuredOrthogonal`
- `TurboQuantMse`
- `TurboQuantProd`

### `TurboQuantConfig`

Recommended fields:

- `dim: usize`
- `bits_per_dim: u8`
- `mode: TurboQuantMode`
- `rotation_backend: TurboRotationBackend`
- `rotation_seed: u64`
- `normalize_for_cosine: bool`

Optional later fields:

- `store_vector_norm: bool`
- `residual_sign_bitpack: bool`

### Rotation Backend

The paper assumes a random rotation matrix `Π`. For `knowhere-rs`, two backends should be supported:

- `DenseOrthogonal`
  - reference implementation
  - easiest to reason about
  - expensive at high dimensions
- `StructuredOrthogonal`
  - production-oriented default
  - should preserve the paper’s “random rotation” intent while reducing runtime and memory cost

This is an engineering inference from the paper, not a claim that the paper mandates structured rotations.

### Codebook Strategy

TurboQuant’s scalar centroids should be treated as a deterministic global table derived from bit-width, not as a dataset-trained codebook.

Engineering rule:

- centroid tables are part of quantizer configuration/state
- they are not learned from the repository’s vector dataset the way PQ codebooks are

## `IvfTurboQuantIndex` Data Flow

### Train

Training should follow the same broad index shape as `IvfPqIndex` and `IvfRaBitqIndex`:

1. Train IVF coarse centroids with `KMeans`
2. For each training vector, assign nearest coarse centroid
3. Build residuals `x - centroid`
4. Initialize TurboQuant state
5. If cosine mode is enabled, normalize as required by the query/storage path

Important distinction:

- IVF centroids are data-trained
- TurboQuant itself is data-oblivious and should not require a PQ-style fine codebook training pass

### Add

For each vector:

1. Find nearest coarse centroid
2. Compute residual against that centroid
3. Encode residual with TurboQuant
4. Store into the corresponding inverted list
5. Optionally append to a refine index if `reorder_k` / refine mode is enabled

Recommended entry layout:

- `id`
- packed TurboQuant code
- optional `prod` sidecar payload
- optional lightweight metadata needed for dequantization/query estimation

### Search

For each query:

1. Find nearest coarse centroids
2. Compute query residual per probed centroid
3. Apply TurboQuant query-side preparation
4. Score compressed vectors inside each inverted list
5. Merge candidates
6. Optional exact or higher-precision rerank through existing refine machinery

Recommended search behavior:

- `Mse` mode:
  - decode or partially decode residual approximation
  - compute approximate score
- `Prod` mode:
  - use MSE path plus residual correction for inner-product estimation

### Persistence

Persistence should serialize:

- index config
- coarse centroids
- TurboQuant config
- rotation state or deterministic seed
- packed code payloads
- optional `prod` sidecar payloads
- refine index payload if present

It should not serialize any learned PQ-style fine codebook because TurboQuant does not depend on one.

## Relationship to Existing Compression Algorithms

### PQ

TurboQuant is adjacent to PQ in use case, but not in mechanics.

- PQ:
  - subvector partitioning
  - learned codebooks
  - ADC-oriented
- TurboQuant:
  - random rotation
  - per-coordinate scalar quantization
  - no data-trained fine codebook

Recommendation:

- keep `ProductQuantizer` unchanged
- add TurboQuant as a peer module
- compare against PQ in tests and benchmarks later, but do not merge implementations

### RaBitQ

TurboQuantProd is closest in purpose to RaBitQ because both target inner-product workloads with lightweight compressed representations.

Recommendation:

- reuse engineering patterns from `IvfRaBitqIndex`
  - query preprocessing
  - compressed scoring
  - optional refine
- do not reuse RaBitQ encoding logic directly

### SQ

TurboQuantMse can be thought of as “rotation plus scalar quantization,” but that is not enough reason to place it inside `sq.rs`.

Recommendation:

- keep `ScalarQuantizer` unchanged
- optionally share low-level helper ideas later
  - decode-into
  - precompute query path
  - pack/unpack helpers

### HVQ

HVQ is the closest existing structural relative because it already combines rotation and scalar quantization.

Recommendation:

- do not repurpose `HvqQuantizer` as TurboQuant
- do not mutate the experimental HVQ path into the new production design
- at most, later extract common math helpers if duplication becomes material

### PCA

PCA is a transform, not a quantizer.

Recommendation:

- no TurboQuant dependency on PCA in `v1`
- allow future composition such as `PCA -> IVF -> TurboQuant`, but keep it out of the first integration

## API Changes

### `IndexType`

Add:

- `IvfTurboQuant`

String aliases should match existing naming patterns:

- `ivf_turboquant`
- `ivf-turboquant`
- `turboquant`

### `IndexParams`

Recommended initial fields:

- `turbo_bits_per_dim: Option<usize>`
- `turbo_mode: Option<String>`
  - allowed values: `mse`, `prod`
- `turbo_use_residual: Option<bool>`
  - default `true`
- `turbo_rotation_seed: Option<u64>`
- `turbo_rotation_backend: Option<String>`
  - allowed values: `dense`, `structured`
- `turbo_reorder_k: Option<usize>`

Recommended defaults:

- `turbo_bits_per_dim = 4`
- `turbo_mode = mse`
- `turbo_use_residual = true`
- `turbo_rotation_seed = 42`
- `turbo_rotation_backend = structured`
- `turbo_reorder_k = 0`

### Legal Matrix

Allow `IvfTurboQuant` for:

- data types:
  - `Float`
  - `Float16`
  - `BFloat16`
- metrics:
  - `L2`
  - `Ip`
  - `Cosine`

## Testing Strategy

### Unit Tests

Add unit tests for:

- code size computation from `dim` and `bits_per_dim`
- centroid table selection per bit-width
- encode/decode roundtrip shape and determinism
- higher bit-width should not regress average MSE on fixed random input
- identical seed should reproduce identical rotation state and codes
- pack/unpack roundtrip for non-byte-aligned bit-widths
- `prod` mode inner-product estimate should be approximately unbiased in Monte Carlo testing

### Integration Tests

Add integration tests for:

- `IvfTurboQuantIndex::train`
- `IvfTurboQuantIndex::add`
- `IvfTurboQuantIndex::search`
- save/load roundtrip
- cosine and inner-product query paths
- refine/rerank path compatibility
- legal matrix acceptance and config validation

### Benchmark and Validation Plan

Benchmarking is not part of this design phase, but later evaluation should compare:

- quantization time vs PQ and RaBitQ
- recall vs PQ and RaBitQ
- query latency with and without rerank
- memory footprint per vector

## Error Handling and Validation

The implementation plan should include explicit validation for:

- `dim == 0`
- unsupported `bits_per_dim`
- unsupported `turbo_mode`
- unsupported `turbo_rotation_backend`
- `prod` mode on unsupported metric combinations
- malformed serialized payloads
- query dimension mismatch
- add/search before train

## Recommended Rollout

### Phase 1

Implement only the design skeleton:

- TurboQuant module family
- `IvfTurboQuantIndex`
- API/config/legal-matrix plumbing
- Rust-only path

### Phase 2

Add:

- persistence
- refine integration
- stronger query-side optimization

### Phase 3

Evaluate optional follow-ups:

- FFI exposure
- DiskANN compression-path experiments
- shared helper extraction with HVQ or SQ if justified by duplication

## Rejected Alternatives

### Alternative 1: Add TurboQuant inside `pq.rs`

Rejected because PQ and TurboQuant have materially different training and storage semantics.

### Alternative 2: Add TurboQuant inside `sq.rs`

Rejected because TurboQuant is not only scalar quantization; the random rotation and `prod` residual correction are first-class parts of the design.

### Alternative 3: Start with DiskANN integration

Rejected because it would immediately affect a performance-sensitive search path and broaden validation scope too early.

### Alternative 4: Start with a generic quantizer trait

Rejected because the repository does not currently organize quantizers that way and the abstraction would likely overfit multiple incompatible families.

## Recommendation

The first implementation should add:

- a new `src/quantization/turboquant/` family
- a new `IvfTurboQuantIndex`
- new API/config/legal-matrix wiring

The implementation should not initially:

- refactor all quantizers behind a common trait
- modify DiskANN/AISAQ compression paths
- expose FFI before the Rust-native path is stable

This is the smallest design that is:

- faithful to the paper’s structure
- aligned with the existing `knowhere-rs` architecture
- testable end to end
- open to later expansion into `TurboQuantProd`, persistence, and more advanced integrations
