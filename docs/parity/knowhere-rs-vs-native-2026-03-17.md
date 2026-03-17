# Knowhere-RS vs Native Knowhere (2026-03-17)

## Scope and compare lanes

- Authority surface: remote x86 only.
- Leadership lane (project-level): near-equal recall authority lane.
- Strict methodology lane (fairness guardrail): same-schema `ef=138` HDF5 lane.

This document intentionally keeps the two lanes separate:

- `strict-ef same-schema` is used for methodology/fairness continuity.
- `near-equal recall` is used for project-level leadership verdicts.

## Snapshot metrics (2026-03-17)

Near-equal recall leadership lane:

- Rust HNSW: recall@10 `0.9518`, QPS median `28479.544` (post runs: `28574.171`, `28384.918`)
- Native HNSW BF16: recall@10 `0.9500`, QPS `15918.091`
- Rust/Native QPS ratio: `1.789x` (about `+78.91%`)

Strict-ef same-schema lane (`ef=138`) for fairness continuity:

- Rust HNSW: recall@10 `0.9880`, QPS `15099.205`
- Native HNSW BF16: recall@10 `0.9500`, QPS `15918.091`
- Native/Rust QPS ratio: `1.054x`

## Code-level deltas (current understanding)

### HNSW (Rust strengths now realized)

- Visited-state tracking: epoch-based `Vec<u32>` with O(1) reset vs per-search clearing patterns in native-style visited lists.
- Hot-path scratch reuse (`opt56`): thread-local search scratch reuse removed repeated allocation/initialization in unfiltered L2 lane.
- SIMD specialization (`opt53`): x86 `dim=128` AVX-512 specialization and batch-4 path improved distance kernel throughput.
- Layer0 locality: slab layout keeps vector and neighbor data close, comparable to native contiguous per-node level0 packing.

Important caveat:

- `opt56` uplift (`+113%` on `1M/1000, ef=60, unfiltered L2`) is lane-specific evidence and must not be generalized to every HNSW scenario.

### IVF-PQ (implemented but quality/perf gap remains)

- Gaps remain in coarse quantizer quality/perf and fast scan path quality relative to FAISS-class implementations.
- Current project verdict still treats IVF-PQ as `no-go` in the final core-path chain.

### DiskANN (boundary is in-memory simplification)

- Current Rust DiskANN is an in-memory simplified boundary and not an SSD pipeline equivalent.
- Project verdict remains `constrained`; native-comparable DiskANN leadership claims are out of scope with current implementation boundary.

### SCANN-like path

- Rust path has anisotropic quantization design value.
- But coarse-stage all-list scan behavior is still the key complexity/perf blocker versus IVF+nprobe style pruning.

## Governance notes

- Keep references date-pinned to `2026-03-17` unless newer authority artifacts supersede them.
- If leadership numbers are updated, update both:
  - this document, and
  - `benchmark_results/final_performance_leadership_proof.json`.

## Sources

- `benchmark_results/hnsw_p3_002_final_verdict.json`
- `benchmark_results/final_performance_leadership_proof.json`
- `benchmark_results/baseline_p3_001_same_schema_hnsw_hdf5.json`
- `benchmark_results/hnsw_fairness_gate.json`
- `benchmark_results/final_core_path_classification.json`
