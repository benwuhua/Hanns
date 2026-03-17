# DiskANN Capability Closure Plan (Native/Official Alignment)

Date: 2026-03-17
Status: active

## Goal

Switch from parameter-tuning-first to capability-closure-first.
Before further optimization rounds, close the major functional gaps against:

- Native Knowhere DiskANN (Microsoft DiskANN C++ integration)
- Official DiskANN Rust implementation

## Closure Order

1. Search-path capability closure (functional parity first)
- Multi-entry/medoid seeding
- Rerank stage (candidate recall recovery path)
- Filtering behavior parity for constrained search

2. Build-path capability closure
- Parallel build pipeline
- Graph slack in build + final prune to target degree
- Randomized initial connectivity strategy with deterministic controls

3. Storage/I/O capability closure
- Disk-backed graph/vector layout
- Aligned read path and async I/O abstraction
- Cache budget and memory budget controls

4. Validation closure
- Capability-focused regression suite (behavior + persistence + invariants)
- Authority benchmark only after capability lane passes

## Execution Rules

- Do not claim performance improvement as primary outcome before capability item is implemented and validated.
- Each capability item requires:
  - focused tests (red/green)
  - authority verification command(s)
  - durable update in `task-progress.md`
- Reject parameter-only rounds that do not advance capability closure.

## Immediate Next Item

Implement DiskANN rerank stage (post-candidate exact reorder path), then validate with authority recall/QPS A/B under fixed lane.
