# Milvus HNSW Performance Optimization Round 1 Design

## Goal

Close the Insert / Optimize / Load gap between knowhere-rs (RS) and native knowhere in the Milvus-integrated VectorDBBench benchmark on Cohere-1M 768D COSINE HNSW.

**Baseline (2026-04-05, hannsdb-x86):**

| Metric | RS | Native | Ratio |
|--------|----|--------|-------|
| Insert | 361s | 179s | 2.02x slower |
| Optimize | 559s | 340s | 1.65x slower |
| Load | 921s | 519s | 1.77x slower |
| QPS | 284 | 848 | 0.33x (3x gap) |
| Recall | 0.964 | 0.956 | RS wins |

**Round 1 targets:** Load and Optimize beat native. Insert ≥ parity. QPS: profile and validate hypothesis.

---

## Architecture

Two independent fixes in `src/faiss/hnsw.rs`, each touching a different function group:

1. **Serialize/Deserialize bulk I/O** — changes `write_to` / `read_from`; eliminates 134M individual `write_all` / `read_exact` calls
2. **Eliminate `to_vec()` in serial add** — changes `add()` and `insert_node_with_scratch()`; eliminates 1M heap allocations per 1M-node build

QPS is not changed in Round 1. Instead, we add lightweight instrumentation to measure `nq` per FFI call and per-call latency, to inform Round 2 design.

---

## Fix 1: Serialize/Deserialize Bulk I/O

### Root cause

**Native knowhere** (hnswlib) stores vectors and level-0 neighbors interleaved in one flat buffer (`data_level0_memory_`). Serialize/deserialize is a single `memcpy`:

```cpp
// saveIndex: one bulk write for ALL level-0 data
output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

// loadIndex: one bulk read
input.read(data_level0_memory_, cur_element_count * size_data_per_element_);
```

**RS** writes vectors one `f32` at a time:

```rust
// hnsw.rs: 134M individual calls for 1M×128D
for v in &self.vectors {
    file.write_all(&v.to_le_bytes())?;
}
```

134M `write_all` calls vs 1 `memcpy`. Same pattern on read.

### Fix: batch I/O (256KB chunks, no layout change)

Replace per-f32 loops with chunked byte-buffer writes. No new dependencies, safe Rust, no data layout change.

**Serialize (vectors):**
```rust
const BATCH: usize = 65536; // 256KB buffer (cache-friendly)
let mut byte_buf = vec![0u8; BATCH * 4];
for chunk in self.vectors.chunks(BATCH) {
    let buf = &mut byte_buf[..chunk.len() * 4];
    for (i, &v) in chunk.iter().enumerate() {
        buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    file.write_all(buf)?;
}
```

Write calls: 134M → ~2048 (for 1M×128D).

**Deserialize (vectors):**
```rust
self.vectors = vec![0.0f32; total_f32];
const BATCH: usize = 65536;
let mut byte_buf = vec![0u8; BATCH * 4];
let mut offset = 0;
while offset < total_f32 {
    let count = BATCH.min(total_f32 - offset);
    let buf = &mut byte_buf[..count * 4];
    file.read_exact(buf)?;
    for i in 0..count {
        self.vectors[offset + i] =
            f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
    }
    offset += count;
}
```

**IDs (i64, same pattern):** batch serialize and deserialize, 8 bytes each.

**Graph neighbors:** keep per-neighbor reads/writes (variable-length structure, already fine since M=16 → ~32M calls vs 134M for vectors; acceptable for Round 1).

### Expected impact

- Serialize: 134M → ~2048 `write_all` calls → Optimize time reduced (serialize is part of index build flush)
- Deserialize: 134M → ~2048 `read_exact` calls → Load time reduced significantly
- RS Load target: ~400-500s (from 921s), below native 519s

---

## Fix 2: Eliminate `to_vec()` in Serial Add

### Root cause

In `add()` Phase 2 (graph construction), each node insertion copies the vector to break a borrow conflict:

```rust
// hnsw.rs:1983 (and ~2064 for parallel path fallback)
let vec: Vec<f32> = self.vectors[vec_start..vec_start + self.dim].to_vec(); // heap alloc!
self.insert_node_with_scratch(idx, &vec, node_level, &mut scratch);
```

For 1M vectors: 1M heap allocations, each 128×4=512 bytes. Total: ~512MB of unnecessary allocations.

The borrow conflict: `insert_node_with_scratch` takes `&mut self` (writes graph) while `&vec` borrows `self.vectors` (reads vector). Rust rejects this.

### Fix: pre-allocated reusable scratch buffer (1 alloc instead of 1M)

The `to_vec()` exists to break a borrow conflict: `insert_node_with_scratch` takes `&mut self` (to write graph edges), but also needs to pass `new_vec: &[f32]` (borrowed from `self.vectors`) into inner functions like `search_layer_idx_with_scratch` which also take `&mut self`. Rust rejects this dual-borrow.

The offset-passing approach (`usize` instead of `&[f32]`) does not solve this: the same conflict reappears inside the function when re-borrowing `self.vectors` while also calling `&mut self` methods.

Correct fix: pre-allocate **one** scratch vector outside the per-node loop and reuse it. This reduces 1M heap allocations to 1:

```rust
// In add(), before the graph construction loop:
let mut vec_scratch = vec![0.0f32; self.dim]; // ONE allocation, reused every iteration

for (i, &node_level) in node_levels.iter().enumerate().take(n) {
    let idx = first_new_idx + i;
    if idx > 0 {
        let vec_start = idx * self.dim;
        // copy_from_slice is a memcpy — no allocator overhead
        vec_scratch.copy_from_slice(&self.vectors[vec_start..vec_start + self.dim]);
        self.insert_node_with_scratch(idx, &vec_scratch, node_level, &mut scratch);
    }
}
```

`insert_node_with_scratch` signature is unchanged. `vec_scratch` is a separate owned `Vec`, not a borrow of `self`, so there is no conflict.

The `copy_from_slice` call still copies 512 bytes per node (dim=128) or 3072 bytes (dim=768 for Cohere), but this is a plain `memcpy` with no allocator call — cost is ~1-10ns vs ~200-500ns for a heap allocation.

### Expected impact

- Eliminates 1M heap allocations during HNSW build
- Reduces GC pressure (Rust allocator still has overhead per alloc)
- Expected improvement: modest but measurable on Optimize and Insert duration
- Rough estimate: 5-15% improvement on build time (allocation overhead is not dominant over graph traversal, but reduces fragmentation)

---

## Fix 3: QPS Profiling Instrumentation (non-blocking)

Add lightweight `eprintln!` logging to `ffi.rs` `knowhere_search` to emit nq and latency per call. This runs in parallel with Fix 1/2 development and informs Round 2 QPS design.

```rust
// In knowhere_search(), after computing result:
if std::env::var("KNOWHERE_RS_TRACE_SEARCH").is_ok() {
    eprintln!("TRACE search nq={} elapsed_us={}", count, elapsed.as_micros());
}
```

Controlled by env var `KNOWHERE_RS_TRACE_SEARCH=1`. Zero cost when disabled.

Data needed:
- Distribution of `count` (nq) per call: if always 1 → parallel query loop won't help
- Per-call latency at concurrency=1 vs concurrency=10: shows where overhead is

---

## Test Plan

### Round 1 local validation (before x86)
- `cargo build --release` — zero errors
- Run existing HNSW tests: `cargo test hnsw -- --nocapture` — no regressions
- Verify recall on synthetic 10K benchmark: `cargo run --example benchmark --release` — recall unchanged

### Round 1 x86 authority benchmark
- Codex builds RS on hannsdb-x86, restarts Milvus shim, runs VectorDBBench Cohere-1M HNSW
- Compare all four metrics against `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260405.json`
- Success criterion: Load < 519s (native), Optimize < 340s (native), recall ≥ 0.960

### QPS profiling run (separate, lightweight)
- Build with `KNOWHERE_RS_TRACE_SEARCH=1` enabled
- Run VectorDBBench at concurrency=1 only (short run, ~10min)
- Collect nq distribution and per-call latency from stderr

---

## Out of Scope (Round 2)

- Interleaved layout (vectors + level-0 neighbors in one buffer, matching hnswlib format)
- Parallel query loop (`rayon::par_iter` for nq > 1)
- QPS fix (depends on profiling results)
- Insert root cause investigation (2x gap may be Milvus-side, not RS)
