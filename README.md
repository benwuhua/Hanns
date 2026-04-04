# Hanns

**Hanns** is a high-performance approximate nearest neighbor (ANN) search library written in pure Rust, targeting CPU vector search workloads.

Not a port or wrapper of any existing library — every line of core code is built from scratch, continuously benchmarked against C++ FAISS/KnowWhere on real-world industrial datasets.

---

## Highlights

- **Pure Rust**: no C/C++ dependencies, no unsafe FFI wrappers. Full type safety and memory safety guarantees.
- **Broad algorithm coverage**: HNSW, IVF-Flat, IVF-SQ8, IVF-USQ, IVF-PQ, DiskANN/AISAQ (flash), ScaNN, Sparse WAND, Binary.
- **Deep SIMD optimization**: hot distance-compute paths cover AVX2, AVX-512, and AVX512VNNI, fully utilizing modern x86 instruction sets.
- **x86 authority benchmarks**: all performance numbers come from x86 server hardware (not Apple Silicon), using real industrial datasets (SIFT-1M, Cohere Wikipedia-1M).
- **Leadership, not parity**: primary indexes measurably outperform KnowWhere C++ native on authority hardware.

---

## Benchmarks (SIFT-1M, x86, March 2026)

> Compiled with `target-cpu=native`. Batch parallel query mode.
> Baseline: KnowWhere C++ native, 8 threads.

### HNSW (M=16, ef\_construction=200)

| ef  | Recall@10 | Hanns batch QPS | Native 8T QPS | Speedup        |
|-----|-----------|-----------------|---------------|----------------|
| 60  | **0.972** | **17,814**      | 15,918        | **+11.9% ✅**  |
| 138 | **0.995** | **17,910**      | 15,918        | **+12.5% ✅**  |

> Root cause fix: unconditional `Instant::now()` calls in `search_layer_idx_shared` consumed 22.57% of query time. Changed to profiling-gated sampling. Result: 15,043 → 17,319 QPS (+15.1%).

### IVF-Flat (nlist=1024)

| nprobe | Recall@10 | Hanns batch QPS | Native 8T QPS | Speedup       |
|--------|-----------|-----------------|---------------|---------------|
| 32     | **0.978** | **3,429**       | 721           | **+5.23x ✅** |
| 32     | 0.978     | 2,339 (serial)  | 341 (1T)      | **+6.9x ✅**  |

### IVF-SQ8 (nlist=1024, fused AVX2 decode\_dot)

| nprobe | Recall@10 | Hanns batch QPS | Native 8T QPS | Speedup       |
|--------|-----------|-----------------|---------------|---------------|
| 32     | **0.958** | **11,717**      | 8,278         | **+1.42x ✅** |

> AVX2 FMA fused u8 decode + dot product, zero-alloc one-pass path.

### IVF-USQ — Unit Sphere Quantizer (768D, Cohere Wikipedia-1M)

| Precision      | nprobe | Recall@10 | QPS       |
|----------------|--------|-----------|-----------|
| 4-bit (8x compression) | 10 | 0.833 | **3,706** |
| 8-bit (4x compression) | 10 | 0.905 | **2,980** |
| 8-bit (4x compression) | 32 | **0.968** | 1,011 |

> USQ uses AVX512VNNI `_mm512_dpbusd_epi32` for integer dot products, replacing the previous HVQ/ExRaBitQ implementations (+3–222% QPS).

### AISAQ / DiskANN Flash (SIFT-1M, mmap mode)

| Mode      | Recall@10 | QPS        |
|-----------|-----------|------------|
| NoPQ L=64 | **0.979** | **5,806**  |
| PQ32      | 0.768     | **10,227** |

---

## Index Status

| Index                    | Status      | Recall@10              | Notes                                              |
|--------------------------|-------------|------------------------|----------------------------------------------------|
| **HNSW**                 | ✅ Leading  | 0.972 (SIFT-1M)        | Graph traversal; cosine TLS scratch, zero-alloc    |
| **HNSW-SQ**              | ✅ Ready    | 0.992                  | Integer precomputed ADC path                       |
| **IVF-Flat**             | ✅ Leading  | 0.978 (SIFT-1M)        | 5x+ faster than native 8T                         |
| **IVF-SQ8**              | ✅ Leading  | 0.958 (SIFT-1M)        | 1.42x faster than native 8T; AVX2 fused decode    |
| **IVF-USQ**              | ✅ Ready    | 0.905–0.968 (Cohere)   | AVX512VNNI; unified 1/4/8-bit quantizer            |
| **IVF-PQ**               | ✅ Ready    | capped by m bytes      | m=32: 0.720 (synthetic data)                       |
| **AISAQ (DiskANN Flash)**| ✅ Ready    | 0.979 NoPQ (SIFT-1M)  | On-demand pread disk mode; Vamana graph build      |
| **ScaNN**                | ✅ Ready    | 0.969                  | Exceeds 0.95 gate at reorder\_k=1600               |
| **Sparse / WAND**        | ✅ Ready    | 1.0                    | Sparse vector retrieval                            |
| **Binary**               | ✅ Complete | —                      | Hamming distance                                   |

---

## Quantization Subsystem

```
src/quantization/
  usq/           — UsqQuantizer: QR orthogonal rotation + unified 1/4/8-bit quantization
    config.rs    — UsqConfig { dim, nbits, seed }
    rotator.rs   — QR decomposition rotation matrix
    quantizer.rs — training + SIMD scoring
    layout.rs    — SoA storage + fastscan transpose
    fastscan.rs  — AVX512 fast scan + topk
    searcher.rs  — two-stage coarse filter + rerank
  pq/            — Product Quantizer: parallel k-means
  sq/            — Scalar Quantizer: SQ8/SQ4
```

---

## Build

```bash
# Development build
cargo build

# Release (LTO + codegen-units=1 + target-cpu=native)
cargo build --release

# Run all tests
cargo test

# Run benchmark (local quick sanity check)
cargo run --example benchmark --release
```

`.cargo/config.toml` automatically enables `target-cpu=native` on x86\_64 and aarch64, enabling AVX2/AVX-512 instruction sets.

---

## Repository Layout

```
src/
  faiss/           core index implementations (HNSW, IVF-*, AISAQ, ScaNN, Sparse, Binary)
  quantization/    quantization subsystem (USQ, PQ, SQ)
  ffi/             FFI layer
tests/             integration and regression tests
benches/           Criterion microbenchmarks
examples/          full benchmark examples
docs/              design docs, performance audits, comparison reports
benchmark_results/ authority verdict artifacts (JSON)
scripts/remote/    x86 remote build/test scripts
```

---

## Datasets

- **SIFT-1M**: standard 128-dim L2 ANN benchmark, 1M vectors
- **Cohere Wikipedia-1M**: 768-dim inner product, 1M real Wikipedia passage embeddings
- **SimpleWiki-OpenAI-260K**: 3072-dim, OpenAI text-embedding-3-large

---

## Authority Hardware

All final performance numbers are produced on an x86 server (not Apple Silicon). Local Mac builds are used for fast iteration and pre-screening only.
