# Knowhere-RS Development Roadmap (Non-GPU)

Last updated: 2026-03-05

## Goal

Deliver production-grade non-GPU parity with C++ knowhere while maintaining measurable performance advantage under valid recall constraints.

## Phase Plan

## Phase 1: Contract Closure (P0)

Objective:

- Close lifecycle/interface contract gaps across core indexes and FFI.

Milestones:

1. Unified interface behavior for Build/Train/Add/Search/RangeSearch/AnnIterator/GetVectorByIds/Serialize/Deserialize.
2. FFI capability matrix aligned with actual runtime behavior.
3. Deterministic error handling for unsupported paths.

Exit criteria:

- No open P0 contract mismatch in `docs/PARITY_AUDIT.md`.

## Phase 2: Behavioral Parity (P1)

Objective:

- Align non-GPU behavior with C++ semantics for HNSW/IVF/DiskANN/AISAQ/Sparse/MinHash.

Milestones:

1. HNSW advanced path parity (range/iterator/vector retrieval/serialization behavior).
2. IVF and quantization path consistency (training/add/search edge cases).
3. DiskANN/AISAQ lifecycle and configuration parity.
4. Centralized legality matrix for index/datatype/metric combinations.

Exit criteria:

- No open P1 parity tasks in `TASK_QUEUE.md`.

## Phase 3: Validation and Performance Hardening (P2)

Objective:

- Standardize benchmarks and guardrails for trusted performance reporting.

Milestones:

1. Ground-truth-backed benchmark templates.
2. Recall-gated performance interpretation (R@10 policy).
3. Expanded regression and stability coverage.

Exit criteria:

- Bench reports include recall credibility tags and reproducibility metadata.

## Governance Rules

- Task execution source: `TASK_QUEUE.md`
- Parity source: `docs/PARITY_AUDIT.md`
- Gap source: `GAP_ANALYSIS.md`

Any claimed completion must update all three files coherently.
