# Milvus Standalone HNSW via knowhere-rs Shim Integration Design

**Date:** 2026-03-27

## Goal

Integrate `knowhere-rs` into Milvus for a first-stage, remote-x86-only smoke path without changing `knowhere-rs` source code and without changing Milvus core execution logic.

The first-stage success criterion is narrow:

- Milvus standalone runs on the remote 16U x86 VM
- a float-vector collection can build an `HNSW` index
- the collection can be loaded
- a basic search returns valid top-k results

## User Constraints

- execution environment is the remote 16U x86 VM
- working paths should live under `/data/work`
- Milvus uses standalone deployment
- do not modify `knowhere-rs` source code
- if capability gaps or incompatibilities are found, file issues first instead of changing kernel/core logic
- Milvus changes should stay in build, thirdparty, or adapter layers rather than query/segcore/indexbuilder core behavior

## Current Repository Context

`knowhere-rs` already exposes a Rust C ABI in [src/ffi.rs](/Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/ffi.rs) and builds `staticlib`/`cdylib` artifacts from [Cargo.toml](/Users/ryan/.openclaw/workspace-builder/knowhere-rs/Cargo.toml).

That ABI already covers the essential operations needed for a minimal vector index smoke:

- create index
- add/build
- search
- serialize / deserialize
- save / load

However, Milvus does not consume a small C ABI surface. It compiles against Knowhere C++ public headers and object model directly:

- `knowhere::Version`
- `knowhere::DataSet`
- `knowhere::IndexNode`
- `knowhere::IndexFactory`
- `knowhere::BinarySet`
- resource and feature helpers used by Milvus vector index wrappers

Milvus currently pulls Knowhere in through [internal/core/thirdparty/knowhere/CMakeLists.txt](/Users/ryan/code/milvus/internal/core/thirdparty/knowhere/CMakeLists.txt), and `internal/core` builds directly against Knowhere headers.

## Problem Statement

Replacing official Knowhere with `knowhere-rs` is not a simple shared-library swap.

Milvus core code expects a C++ interface layer, while `knowhere-rs` currently exports a Rust-owned C ABI. Therefore the integration problem is:

1. preserve `knowhere-rs` unchanged
2. avoid Milvus core logic edits
3. provide enough Knowhere-compatible C++ surface for Milvus HNSW memory-index flow to compile and execute

## Scope

This design covers only first-stage smoke integration for:

- Milvus standalone
- float vectors
- HNSW
- create_index
- load
- search
- remote x86 build and runtime verification

## Out of Scope

The following are intentionally excluded from stage 1:

- IVF family
- DiskANN
- sparse indexes
- binary vectors
- GPU indexes
- range search
- iterator APIs
- full Knowhere API parity
- performance benchmarking
- modifying `knowhere-rs` internals
- modifying Milvus query/segcore execution semantics

Any request that crosses those boundaries should first become an explicit issue.

## Recommended Architecture

### Summary

Introduce a new C++ compatibility layer, `knowhere-rs-shim`, that presents a minimal Knowhere-like public surface to Milvus while internally calling the existing `knowhere-rs` C ABI.

This yields a 3-layer structure:

1. `knowhere-rs`
   - unchanged Rust implementation
   - built as `libknowhere_rs.so` or equivalent
2. `knowhere-rs-shim`
   - new C++ compatibility layer
   - exposes minimal Knowhere C++ headers and objects
   - translates Milvus/Knowhere-style calls into `knowhere-rs` ABI calls
3. `Milvus`
   - patched only in thirdparty/build integration points
   - continues to use a Knowhere-shaped API from its point of view

### Why this is the recommended path

- it preserves the hard requirement to not modify `knowhere-rs`
- it keeps Milvus changes away from core execution code
- it avoids overcommitting to full Knowhere compatibility
- it lets the first milestone succeed with a tightly scoped HNSW-only surface

## Repository and Remote Layout

The remote worktree should be kept under `/data/work/milvus-rs-integ`:

- `/data/work/milvus-rs-integ/milvus`
- `/data/work/milvus-rs-integ/knowhere-rs`
- `/data/work/milvus-rs-integ/knowhere-rs-shim`
- `/data/work/milvus-rs-integ/run`
- `/data/work/milvus-rs-integ/artifacts`

Recommended local source roots:

- local Milvus source: `~/code/milvus` or `~/code/vectorDB/milvus`
- local `knowhere-rs`: current repository

The shim should live outside the `knowhere-rs` repository so the integration does not violate the no-code-change rule for `knowhere-rs`.

## Milvus Integration Point

Milvus currently fetches and builds Knowhere from source in [internal/core/thirdparty/knowhere/CMakeLists.txt](/Users/ryan/code/milvus/internal/core/thirdparty/knowhere/CMakeLists.txt).

Stage 1 should replace that fetch path with a build-layer option:

- default mode: official Knowhere
- integration mode: local `knowhere-rs-shim`

Recommended mechanism:

- add a CMake option such as `MILVUS_USE_KNOWHERE_RS_SHIM=ON`
- when enabled:
  - skip `FetchContent` for official Knowhere
  - point `KNOWHERE_INCLUDE_DIR` at shim headers
  - link Milvus vector-index targets against the shim library and `libknowhere_rs`

This keeps the change in thirdparty/build code rather than in Milvus core vector execution logic.

## Minimal Knowhere-Compatible Surface

The shim should implement only the subset needed by Milvus HNSW memory-index flow.

### Required public types and functions

- `knowhere::Version`
- `knowhere::DataSet`
- `knowhere::GenDataSet`
- `knowhere::BitsetView`
- `knowhere::BinarySet`
- `knowhere::Status`
- `knowhere::expected<T>`
- `knowhere::IndexNode`
- `knowhere::IndexFactory`
- `knowhere::Index<knowhere::IndexNode>` wrapper
- `knowhere::IndexStaticFaced<T>::HasRawData(...)`
- `knowhere::IndexStaticFaced<T>::EstimateLoadResource(...)`
- `knowhere::UseDiskLoad(...)`
- `knowhere::feature::MMAP` plus `FeatureCheck(...)` minimum behavior
- string constants from `comp/index_param.h` required for HNSW config keys and metric keys

### Required index behavior

For the HNSW stage-1 shim node:

- `Build`
- `Train`
- `Add`
- `Search`
- `Serialize`
- `Deserialize`
- `DeserializeFromFile`
- `HasRawData`
- `Dim`
- `Count`
- `Size`
- `Type`
- `CreateConfig`
- `GetIndexMeta` with minimal valid metadata

### Explicit non-support for stage 1

The following should return a clear unsupported status and get an issue entry:

- `RangeSearch`
- `GetVectorByIds`
- iterator APIs
- disk-load paths
- mmap
- non-HNSW index types

## Data Flow

### Build path

1. Milvus creates a vector index through its existing `IndexFactory`.
2. Milvus `VectorMemIndex<float>` requests an index object from shim `knowhere::IndexFactory`.
3. Shim returns a `HnswRustNode`.
4. `HnswRustNode::Build` translates the Milvus/Knowhere config into `knowhere-rs` `CIndexConfig`.
5. `HnswRustNode` creates a `knowhere-rs` index handle and feeds the raw tensor through existing build/add ABI calls.

### Search path

1. Milvus creates a `knowhere::DataSet` query object.
2. `VectorMemIndex::Query` calls shim `Search`.
3. Shim extracts query tensor, top-k, metric, and supported HNSW parameters.
4. Shim calls existing `knowhere-rs` search ABI.
5. Shim converts the result into a Knowhere-style `DataSet` carrying ids and distances.

### Load / serialize path

1. Milvus serializes or loads through its existing vector-index wrapper flow.
2. Shim translates `BinarySet` operations into the closest existing `knowhere-rs` serialize / deserialize / save / load ABI path.
3. Stage 1 may use a minimal single-blob `BinarySet` contract as long as Milvus load/save flow remains coherent.

## Config Translation

Only a narrow HNSW config subset should be translated in stage 1:

- `index_type = HNSW`
- `metric_type = L2 | IP | COSINE`
- `dim`
- `M`
- `efConstruction`
- `ef`

Any additional HNSW-related fields should be ignored only if safe and documented. If an ignored field changes semantics materially, create an issue instead of silently dropping it.

## Issue-First Policy

Before implementation, create explicit issue records for:

1. shim scope and non-goals
2. ABI mismatch between Knowhere C++ surface and `knowhere-rs` C ABI
3. unsupported stage-1 methods and features
4. BinarySet compatibility assumptions
5. resource estimation and mmap behavior placeholders

Issue format may live in Milvus integration docs or a dedicated issue tracker, but the problems must be explicit before implementation proceeds.

## Standalone Deployment Plan

Stage 1 should use source-built Milvus standalone on the remote x86 host rather than a containerized injection path.

Reasons:

- the integration point is in `internal/core/thirdparty/knowhere`
- build-layer swapping is easier to reason about in source form
- [scripts/start_standalone.sh](/Users/ryan/code/vectorDB/milvus/scripts/start_standalone.sh) already shows the expected runtime library loading pattern

Runtime requirements:

- shim shared library path must be on `LD_LIBRARY_PATH`
- `libknowhere_rs.so` path must also be on `LD_LIBRARY_PATH`
- startup logs and smoke logs should be archived under `/data/work/milvus-rs-integ/artifacts`

## Verification Strategy

Stage 1 verification is smoke-only, not performance-oriented.

### Required smoke lane

1. build `knowhere-rs`
2. build `knowhere-rs-shim`
3. build Milvus with shim mode enabled
4. start Milvus standalone
5. run a fixed Python smoke script against `localhost:19530`
6. assert:
   - collection creation succeeds
   - HNSW index creation succeeds
   - collection load succeeds
   - one search returns exactly `topk` results
   - ids/distances are valid and parseable

### Required artifacts

- build logs
- standalone startup logs
- smoke script output
- exact build flags / environment
- issue list for any unsupported paths discovered during implementation

## Risks

### 1. Milvus expects more Knowhere surface than the minimal audit suggests

This is likely. The mitigation is to keep the first target narrow and log every newly discovered dependency as an issue rather than widening scope casually.

### 2. `BinarySet` compatibility may be more coupled than expected

If Milvus assumes a richer multi-blob structure for HNSW persistence, the shim must either emulate that structure or restrict stage 1 to one verified load path and record the rest as unsupported.

### 3. Resource estimation and mmap hooks may block compilation or runtime even if HNSW works

These should be implemented as minimal conservative stubs first, with explicit issue tracking for full parity.

### 4. Remote environment drift

The remote x86 host must be treated as the authoritative integration environment. Build flags, dependency versions, and library search paths should be scripted rather than hand-applied.

## Rejected Alternatives

### 1. Modify `knowhere-rs` to expose a full Knowhere-compatible C++ surface

Rejected because the user explicitly does not want `knowhere-rs` code changed.

### 2. Patch Milvus core logic to call `knowhere-rs` directly

Rejected because the allowed change boundary is build/adapter layer only, not Milvus core execution logic.

### 3. Replace the integration with an external sidecar service

Rejected for stage 1 because it does not constitute real in-process Milvus index integration.

## Recommended Next Step

Write an implementation plan for:

1. issue inventory
2. shim header and class skeleton
3. Milvus thirdparty build switch
4. remote standalone build/run scripts
5. smoke verification script

That plan should still preserve the rule that unsupported paths become issues before they become code.
