# HNSW Fairness Gate Datatype Screen Design

## Context

The current HNSW fairness gate now has effective-`ef` parity and query-dispatch parity on the remote same-schema lane, but it still fails because Rust records `Float32` while the trusted native lane is `BF16`.

The codebase already exposes datatype enums that include `Float16` and `BFloat16`, and the legal matrix allows those combinations for HNSW-family indexes. That does not mean the current HNSW lane actually executes as BF16:

- `HnswIndex` stores vectors in `Vec<f32>`
- the current HDF5 baseline harness still builds `IndexConfig` with `DataType::Float`
- the current baseline artifact reports only `vector_datatype = "Float32"`

So the next local screen should answer a narrow question first:

Can the current HNSW fair lane honestly enter a BF16 compare surface through harness-only changes, or does that require core HNSW work?

## Options Considered

### Option 1: Add a datatype capability-audit screen to the HDF5 baseline harness only

Teach `src/bin/generate_hdf5_hnsw_baseline.rs` to accept a requested datatype for the local fair lane, and emit both the requested datatype and the currently effective datatype for the HNSW path.

Pros:

- Smallest honest next step
- Keeps the screen local and attributable
- Makes it explicit whether a BF16 request changes the real HNSW lane at all
- Avoids fabricating a fake “BF16 parity” claim from metadata alone

Cons:

- Does not solve datatype parity by itself
- May end in a negative screen result rather than a promotable one

### Option 2: Add a harness-only BF16 proxy by quantizing inputs and converting back to `f32`

Pre-quantize train/query vectors to BF16 in the harness, then immediately decode them back to `f32` before calling the current HNSW code.

Pros:

- Cheap to implement
- Might show whether reduced mantissa precision changes recall

Cons:

- Not honest datatype parity
- Storage and hot-path distance math would still be `f32`
- Risks producing misleading “BF16-like” evidence that cannot support the fairness gate

### Option 3: Start real HNSW BF16 support in the core

Teach HNSW storage and distance dispatch to use a real BF16 vector representation.

Pros:

- Directly targets the remaining fairness blocker
- Could produce a genuinely fair authority lane

Cons:

- Much larger than a screen
- Blurs architecture work with hypothesis testing
- Too expensive before we confirm the blocker is real at the code-path level

## Recommendation

Choose Option 1.

The next slice should be an honest capability-audit screen, not a fake BF16 benchmark and not a full core rewrite. If the screen shows that a requested BF16 lane still resolves to effective `Float32`, that is valuable reject evidence for any harness-only datatype path and gives the next session a clear architectural target.

## Proposed Design

### 1. Requested datatype becomes an explicit baseline-harness input

Add a local-only CLI control to `src/bin/generate_hdf5_hnsw_baseline.rs`:

- `--vector-datatype <float32|bfloat16>`

Default remains `float32`.

This option should drive the requested `IndexConfig.data_type` used to build the HNSW lane, so the screen exercises the real current API surface rather than inventing a separate benchmark-only setting.

### 2. Artifact metadata distinguishes requested datatype from effective datatype

Keep `vector_datatype` as the effective datatype actually used by the current HNSW lane. Add a new field:

- `requested_vector_datatype`

For the current HNSW implementation:

- requested `float32` => effective `Float32`
- requested `bfloat16` => effective still `Float32`

This is the key audit contract. It tells future sessions whether the requested datatype meaningfully changed the lane or not.

### 3. TDD contract locks the current capability boundary

The first failing test should prove that:

- a BF16 request is accepted by the harness-facing configuration
- the current HNSW baseline lane still reports effective `Float32`
- the mismatch is surfaced mechanically in metadata rather than hidden

This should live in `src/bin/generate_hdf5_hnsw_baseline.rs` alongside the current harness tests.

### 4. Screen result uses local-only HDF5 evidence

Run the local fair lane twice:

- requested `float32`
- requested `bfloat16`

using the already fair local lane shape:

- `--hnsw-adaptive-k 0`
- `--query-dispatch-mode parallel`
- `--query-batch-size 32`

The screen promotes only if the requested BF16 lane also changes the effective datatype honestly. If the artifact still reports effective `Float32`, record:

- `screen_result=reject`

because a harness-only datatype path does not cross the fairness boundary.

## Files In Scope

- `src/bin/generate_hdf5_hnsw_baseline.rs`
- `task-progress.md`
- `docs/superpowers/plans/2026-03-14-hnsw-fairness-gate-datatype-screen.md`

Potential local-only outputs:

- `/tmp/hnsw_fairness_datatype_float32.json`
- `/tmp/hnsw_fairness_datatype_bfloat16.json`

## Non-Goals

- No remote authority rerun in this slice
- No claim that BF16 parity is achieved
- No HNSW core storage rewrite
- No FFI / Python / JNI datatype expansion in this screen
