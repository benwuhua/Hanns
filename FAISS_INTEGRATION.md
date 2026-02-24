# Faiss Integration Guide

## Current Status

The current implementation uses **pure Rust** for all index types:
- `MemIndex` - Flat brute-force index
- `HnswIndex` - HNSW graph-based index

## Performance Comparison

| Index | Add Speed | Search QPS |
|-------|-----------|------------|
| Flat (Rust) | 2.6M vec/s | 473 |
| HNSW (Rust) | 2K vec/s | 733K |

## Faiss Integration Options

### Option 1: CXX Bridge (Recommended)

Use `cxx` crate for safe C++ interop:

```rust
// Cargo.toml
[dependencies]
cxx = "1.0"

[build-dependencies]
cxx-build = "1.0"
```

```rust
// build.rs
fn main() {
    cxx_build::bridge("src/faiss/ffi.rs")
        .include("/opt/homebrew/include")
        .compile("faiss-bridge");
}
```

### Option 2: C API Wrapper

Write a thin C wrapper around Faiss:

```c
// faiss_capi.c
#include "faiss/Index.h"

faiss_Index* faiss_new_flat(int d, int metric) {
    return new faiss::IndexFlat(d, metric);
}

void faiss_add(faiss_Index* index, float* vectors, int64_t n) {
    index->add(n, vectors);
}

void faiss_search(faiss_Index* index, float* query, int k, float* distances, int64_t* labels) {
    index->search(1, query, k, distances, labels);
}
```

Then use `bindgen` to generate Rust bindings.

### Option 3: faiss-sys Crate

Use pre-built bindings:

```toml
[dependencies]
faiss-sys = "0.1"
```

## Roadmap

1. ✅ Pure Rust implementation (current)
2. 🔄 Add Faiss C API wrapper
3. ⬜ Benchmark comparison
4. ⬜ Hybrid mode (Rust + Faiss)

## Files

- `src/faiss/mem_index.rs` - Flat index (Rust)
- `src/faiss/hnsw.rs` - HNSW index (Rust)
- `src/faiss/ffi.rs` - Faiss FFI placeholder
