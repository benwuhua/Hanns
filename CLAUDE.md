# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KnowHere-RS is a Rust implementation of a vector search engine, designed as a Rust-native replacement for Milvus KnowHere (C++). It provides similarity search with various index types and distance metrics.

## Build Commands

```bash
# Development build
cargo build

# Release build (optimized, LTO enabled)
cargo build --release

# Build using script (builds and runs tests)
./build.sh release

# Run all tests (106 tests)
cargo test

# Run specific test
cargo test test_l2_distance

# Run tests with output
cargo test -- --nocapture
```

## CLI Usage

```bash
# Build CLI
cargo build --release

# Create index
./target/release/knowhere-cli create my_index 128 --index-type flat --metric l2

# Add vectors (binary f32 file)
./target/release/knowhere-cli add my_index vectors.bin

# Search
./target/release/knowhere-cli search my_index "0.1,0.2,0.3,..." -k 10

# Save/load index
./target/release/knowhere-cli save my_index /path/to/index
./target/release/knowhere-cli load my_index /path/to/index 128
```

## Examples

```bash
# Basic usage
cargo run --example basic

# HNSW index
cargo run --example hnsw

# Benchmarking
cargo run --example benchmark
```

## Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| `src/api/` | Public API interfaces (SearchRequest, IndexConfig, error types) |
| `src/faiss/` | Index implementations (MemIndex, HnswIndex, IvfPqIndex, DiskAnnIndex) |
| `src/metrics.rs` | Distance calculations (L2, IP, Cosine, Hamming) |
| `src/simd.rs` | SIMD-optimized distance functions (NEON/SSE/AVX detection) |
| `src/dataset.rs` | Dataset abstraction with soft-delete support via BitsetView |
| `src/bitset.rs` | Bitset for soft deletion |
| `src/storage/` | Disk and memory storage backends |
| `src/codec/` | Serialization for index and vector data |
| `src/executor/` | Thread pool and concurrent execution |
| `src/quantization/` | K-means and PQ quantization |

### Index Types

All indexes implement the `Index` trait (`src/index.rs`):

- **MemIndex** (`faiss/mem_index.rs`) - Flat brute-force index
- **HnswIndex** (`faiss/hnsw.rs`) - HNSW graph-based approximate search
- **IvfIndex** (`faiss/ivf.rs`) - IVF clustering index
- **IvfPqIndex** (`faiss/ivfpq.rs`) - IVF with Product Quantization
- **DiskAnnIndex** (`faiss/diskann.rs`) - Disk-based ANN index

### Key Traits

```rust
// Index trait - unified interface for all index types
pub trait Index: Send + Sync {
    fn index_type(&self) -> &str;
    fn dim(&self) -> usize;
    fn count(&self) -> usize;
    fn is_trained(&self) -> bool;
    fn train(&mut self, dataset: &Dataset) -> Result<(), IndexError>;
    fn add(&mut self, dataset: &Dataset) -> Result<usize, IndexError>;
    fn search(&self, query: &Dataset, top_k: usize) -> Result<SearchResult, IndexError>;
    fn save(&self, path: &str) -> Result<(), IndexError>;
    fn load(&mut self, path: &str) -> Result<(), IndexError>;
}

// Distance trait - for pluggable distance metrics
pub trait Distance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32>;
}
```

## Library Exports

The main entry point is `src/lib.rs`. Key exports:

```rust
// From api module
pub use api::{SearchRequest, SearchResult, KnowhereError, Result, IndexConfig, IndexType, MetricType};

// Index types
pub use faiss::{FaissIndex, MemIndex, HnswIndex, IvfPqIndex, DiskAnnIndex};

// Core utilities
pub use bitset::BitsetView;
pub use dataset::{Dataset, DataType};
pub use metrics::{Distance, get_distance_calculator, L2Distance, InnerProductDistance, CosineDistance, HammingDistance};
```

## Configuration

Index parameters are configured via `IndexConfig` and `IndexParams`:

```rust
let config = IndexConfig {
    index_type: IndexType::Hnsw,
    metric_type: MetricType::L2,
    dim: 128,
    params: IndexParams::hnsw(ef_construction: 200, ef_search: 64, ml: 0.36),
};
```

## Crate Types

The library builds as multiple crate types (see `Cargo.toml`):

```toml
crate-type = ["staticlib", "cdylib", "rlib"]
```

This enables use as a Rust library, static C library, or shared library for FFI integration.

## Dependencies

Key dependencies:
- `rayon` - Parallel iterators
- `parking_lot` - High-performance synchronization primitives
- `memmap2` - Memory-mapped file I/O
- `tracing` / `tracing-subscriber` - Logging
- `serde` / `serde_json` - Serialization
- `clap` - CLI argument parsing
- `thiserror` - Error derive macros

## Release Profile

The release profile enables aggressive optimizations:
```toml
[profile.release]
lto = true           # Link-time optimization
codegen-units = 1    # Single codegen unit for better optimization
panic = "abort"      # Smaller binary, no unwinding
```
