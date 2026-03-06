# Knowhere-RS Gap Analysis (Non-GPU)

Last updated: 2026-03-06  
Scope: Non-GPU production parity against C++ knowhere

## 1. Baseline and Method

- Rust repo: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs`
- C++ repo: `/Users/ryan/Code/vectorDB/knowhere`
- Source of truth for parity status: `docs/PARITY_AUDIT.md`
- Priority order: BUG > PARITY > OPT > BENCH

Evaluation dimensions:

1. Functional parity (core index lifecycle and query behavior)
2. Interface parity (Build/Train/Add/Search/RangeSearch/AnnIterator/GetVectorByIds/Serialize/Deserialize)
3. Validation quality (ground truth, recall constraints, regression tests)
4. Production readiness (stability, configurability, observability)

## 2. Current High-Level Gaps

## P0 (Critical)

- Interface contract is not fully closed across all non-GPU indexes.
  - Impact: behavior drift vs C++ in edge paths.
  - Exit criteria: all targeted indexes implement or explicitly reject the full contract consistently.

- FFI behavior is inconsistent for several index types and advanced operations.
  - Impact: integration breakage for embedding-side consumers.
  - Exit criteria: FFI matrix and runtime behavior match documented capability table.

## P1 (Important)

- DiskANN/AISAQ lifecycle and parameter semantics still differ from C++ behavior.
- HNSW/IVF advanced paths (range search, iterator, vector retrieval, serialization nuances) are not uniformly aligned.
- MinHash-LSH parity incomplete: `mh_*` naming刚完成别名映射，但 Index trait wrapper 与 FFI 查询长度语义仍未对齐 C++。
- Legal matrix (index × datatype × metric) is not centralized as a strict runtime gate.

## P2 (Optimization)

- Performance validation is not fully normalized under recall constraints for all benchmark scripts.
- Documentation and task governance need strict synchronization (queue/roadmap/parity audit).

## 3. Validation Gaps

Required validation policy:

- All performance claims must include ground truth origin.
- If R@10 < 80%, QPS must be marked as "unreliable".
- If R@10 < 50%, benchmark result must be marked as "recheck required".
- Regression gate: `cargo test` must pass for merge-ready changes.

## 4. Module Focus (Non-GPU)

Primary modules for parity closure:

- Core contract: `src/index.rs`, `src/api/*`
- Index implementations: `src/faiss/hnsw.rs`, `src/faiss/ivf*.rs`, `src/faiss/diskann*.rs`, `src/faiss/sparse*.rs`, `src/index/minhash_lsh.rs`
- FFI surface: `src/ffi.rs`, `src/ffi/*`

Detailed per-file mapping and missing items are tracked in `docs/PARITY_AUDIT.md`.

## 5. Completion Definition

This gap analysis is considered closed when:

1. `docs/PARITY_AUDIT.md` shows no unresolved P0/P1 parity gaps in non-GPU scope.
2. `TASK_QUEUE.md` has no open P0/P1 tasks tied to parity closure.
3. Validation policy is enforced in benchmark outputs and review reports.
