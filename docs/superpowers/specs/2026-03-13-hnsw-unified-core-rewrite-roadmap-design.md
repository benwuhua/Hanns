# HNSW Unified Core Rewrite Roadmap Design

## Summary

This spec replaces the recent HNSW reopen pattern with a single roadmap for pure-Rust HNSW recovery.

The roadmap is based on four settled facts from the current repository state:

- round 8 showed that the previously identified bulk-build graph-quality fixes did not produce an authority-side win
- round 9 showed that search-path implementation quality can materially move same-schema Rust QPS
- round 10 showed that a layer-0-only locality cut is not enough on its own
- round 11 showed that a locally promising filtered brute-force fallback can catastrophically regress the authority workload

The next phase therefore should not be another isolated reopen. It should be a coordinated HNSW program with one ordering rule:

1. diagnose whether the current gap is graph-limited, search-kernel-limited, or both
2. choose the dominant branch
3. only then rewrite the corresponding core path

## Context

The current accepted project verdict remains negative:

- HNSW is archived as `functional-but-not-leading`
- final project acceptance remains `not_accepted`

Recent HNSW evidence matters because it narrows what is still plausible:

- round 9 raised Rust same-schema HNSW to `1845.608` qps and proved the production query path still had meaningful implementation debt
- round 10 did not produce a Rust-side authority gain from layer-0 slab locality
- round 11 ended as a hard stop because a native-aligned filtered brute-force policy caused a catastrophic same-schema authority regression

That evidence means the remaining HNSW gap can no longer be approached as a sequence of loosely related micro-optimizations. The work must shift to a structured diagnosis of the core engine.

## Goals

- Establish whether Rust HNSW is primarily limited by graph quality, generic search-kernel cost, or a combination of both.
- Replace ad hoc reopen slices with a staged pure-Rust HNSW roadmap that covers query, build, filter/range behavior, and compatibility surfaces in one place.
- Keep future performance work aligned with realistic ANN profiling practice: representative workload first, hotspot and scaling diagnosis second, low-level tuning last.
- Preserve the distinction between:
  - production-contract readiness
  - performance competitiveness
  - native-compatibility semantics

## Non-Goals

- This spec does not propose bridging native FAISS or abandoning pure Rust.
- This spec does not claim that graph quality is already proven to be the dominant root cause.
- This spec does not reopen a new authority round by itself.
- This spec does not require native binary-format compatibility as part of the first performance phase.

## Problem Statement

The current HNSW implementation has converged on several local improvements, but the remaining gap is still too large and too structurally mixed to attack with another narrow hypothesis.

There are three still-plausible explanations for the remaining deficit:

1. the graph itself is lower quality than native, so Rust needs higher `ef` to reach similar recall
2. the graph is good enough, but the generic search kernel still makes each visited node too expensive
3. both are true, and previous rounds only improved a narrow subset of the total search workload

At the moment, the repository has enough evidence to reject some specific theories, but not enough to rank these three causes confidently.

## Design Principles

### 1. Diagnose before rewriting

No further HNSW core rewrite work should begin from intuition alone. The next phase should first answer:

- does Rust need materially larger `ef` than native at similar recall?
- does Rust visit materially more nodes than native at similar recall?
- when using comparable traversal effort, is the per-node search cost still significantly higher?

### 2. Separate graph cost from kernel cost

The roadmap should treat these as different problems:

- graph topology quality
- search-kernel efficiency

They interact, but they should not be optimized blindly in the same step.

### 3. Keep production contract and performance work separate

FFI, JNI, save/load, and other external-facing contracts should remain visible in the roadmap, but they are not the leading blocker for HNSW performance. They should not drive kernel-level priorities.

### 4. Follow realistic Rust performance workflow

Future performance work should follow this order:

1. representative workload
2. trustworthy measurement
3. hotspot or scaling diagnosis
4. data-layout and control-flow changes
5. only then low-level SIMD or instruction inspection

## Proposed Roadmap

### Phase 0: Locked Baseline

Create and freeze a single working baseline for the whole HNSW rewrite program:

- dataset
- query set
- `M`
- `efConstruction`
- search recall target
- thread count
- reporting schema

This phase exists so later comparisons do not drift between build experiments and search experiments.

### Phase 1: Graph-Quality Diagnosis

The first serious question is whether the Rust graph is structurally worse than native.

Minimum outputs:

- level histogram
- per-layer degree histogram
- same-recall `ef` comparison
- visited-nodes comparison at matched recall

Optional second-line outputs if the first set already points to graph weakness:

- bidirectional-edge ratio
- component/connectivity health
- neighborhood diversity or local clustering proxy
- entry-descent landing quality

Success criterion for this phase:

- determine whether graph weakness is a first-order cause or only a secondary factor

### Phase 2: Search-Cost Diagnosis

If graph diagnosis alone does not explain the gap, measure search-kernel cost directly.

Key outputs:

- per-query visited count
- frontier operations
- result-set operations
- distance-call count
- batch-distance utilization
- layer-0 versus generic-path share

The goal is to answer:

- is Rust expensive because it traverses more?
- or because it pays more per traversal step?

### Phase 3: Decision Gate

At this point the roadmap forks.

#### Branch A: Build-First Rewrite

Choose this branch if Rust consistently needs higher `ef` and higher visited-node counts at matched recall.

Priority themes:

- build-path topology quality
- serial versus parallel build alignment
- level-order behavior
- candidate selection and connection policy
- broader graph-health invariants

Primary success metric:

- pull the Rust `recall / ef / visited` curve toward native before claiming query-speed gains

#### Branch B: Search-First Rewrite

Choose this branch if matched-recall traversal effort is already close, but Rust still spends materially more time per query.

Priority themes:

- a unified `idx`-based, scratch-reused generic search kernel
- one shared traversal core for:
  - unfiltered KNN
  - bitset-filtered KNN
  - range or threshold-style search extensions
- unified frontier/result/visited infrastructure
- reduced path splitting between layer-0 fast path and generic path

Primary success metric:

- lower per-visited-node cost on representative workloads

#### Branch C: Dual Rewrite

Choose this only if both graph and kernel diagnosis remain clearly first-order.

In that case, the implementation should still be split into separate workstreams with separate measurements:

- graph-quality stream
- generic-search-kernel stream

The work should not be blended into a single opaque rewrite.

### Phase 4: Structural Layout Work

Only after Phases 1 through 3 should the roadmap escalate to broader memory-layout work.

This phase may include:

- broader contiguous graph storage
- unifying visited/frontier/result layouts around cache locality
- reducing dependence on per-node fragmented allocations

This phase is intentionally later because earlier rounds already showed that layer-0-only locality cuts are not enough to justify broad claims by themselves.

### Phase 5: Compatibility and Replaceability Surfaces

This phase is not the primary performance blocker, but it must remain visible:

- save/load semantics
- FFI/JNI contract stability
- whether native-compatible serialization is required or merely desirable
- filtered and range-query behavior under production-like constraints

This phase should be planned independently from the core performance rewrite, even if implemented in parallel later.

## Recommended First Deliverables

The roadmap should start with four concrete deliverables:

1. `baseline lock`
2. `rust/native graph structure export`
3. `same-recall ef/visited comparison`
4. `graph-limited vs search-limited decision memo`

These four outputs are enough to decide whether the first major rewrite should target build quality or generic search.

## What This Roadmap Replaces

This roadmap replaces the idea that HNSW should continue through isolated reopen rounds such as:

- one more build tweak
- one more layer-0 locality tweak
- one more filtered fallback tweak

Those experiments were still useful because they eliminated weak theories. But they are no longer the right unit of progress.

The next unit of progress should be a bounded core-engine program with an explicit decision gate.

## Risks

### Risk 1: Over-diagnosis without action

Too much instrumentation can become another delay mechanism.

Mitigation:

- keep the first diagnosis package small
- stop once graph-limited versus search-limited is clear enough to choose a branch

### Risk 2: Mixed-signal rewrites

If graph and search are changed together too early, later benchmark results become difficult to interpret.

Mitigation:

- preserve the decision gate
- keep build and search streams separable even if both are eventually needed

### Risk 3: Compatibility work dilutes performance priorities

FFI and persistence work can consume large effort without moving the HNSW gap.

Mitigation:

- treat contract work as a separate track
- do not let it dictate the first performance branch

## Recommendation

Adopt this roadmap and make the first execution phase diagnostic rather than optimization-driven.

The immediate next move should not be "rewrite everything." It should be:

- export and compare graph structure
- compare matched-recall traversal effort
- decide whether the pure-Rust rewrite starts with build quality or generic search

That is the smallest plan that can still prevent another month of optimizing the wrong layer.
