# Hanns

**High-performance approximate nearest neighbor (ANN) search in pure Rust.**

Hanns is the ANN engine powering three production systems:

| Ecosystem | Role |
|-----------|------|
| **[Milvus](https://milvus.io)** | Drop-in replacement for KnowWhere C++ in the world's most popular cloud-native vector database |
| **HannsDB** | Embedded ANN engine for single-machine agent workloads — low-latency, zero-dependency |
| **[Lance](https://lancedb.github.io/lance/)** | ANN backend for the open multimodal vector lake format |

Built from scratch. No C++ dependencies. Benchmarked head-to-head against FAISS, KnowWhere, and Lance on real x86 server hardware.

---

## Performance at a Glance

### vs KnowWhere C++ inside Milvus (Cohere Wikipedia-1M, 768-dim IP, x86)

| Metric | KnowWhere C++ | Hanns | vs Native |
|--------|---------------|-------|-----------|
| Graph build (Optimize) | 854.2s | **336.9s** | **2.53× faster** |
| Index load | 1158.9s | **673.7s** | **1.72× faster** |
| QPS (c=20, ef=128, k=100) | ~500 | **1,051** | **~2.1× faster** |
| QPS (c=80, ef=128, k=100) | ~800 | **1,042** | **~1.3× faster** |
| Recall@100 | 0.960 | 0.957 | parity |

### vs Lance (Cohere-1M, 1024-dim L2, x86)

| ef | Hanns QPS | Lance QPS | Hanns/Lance |
|----|-----------|-----------|-------------|
| 50 | **2,331** | 1,473 | **1.58×** |
| 200 | **794** | 483 | **1.64×** |
| 800 | **245** | 140 | **1.75×** |

Hanns search leads by **1.58–1.75×** across the full ef range; advantage grows with higher recall requirements.

### vs Microsoft DiskANN Rust (SIFT-1M, R=48 L=64, x86)

| System | Recall@10 | QPS |
|--------|-----------|-----|
| DiskANN Rust (Microsoft) | 0.986 | 4,832 |
| **Hanns AISAQ NoPQ** | **0.994** | **5,806** |

**+20% higher QPS, +0.8% better recall** at identical parameters on identical hardware.

---

## Advanced Quantization: USQ

Standard Product Quantization (PQ) — used by FAISS and most ANN libraries — minimizes L2 reconstruction error. On modern embedding search (high-dim, Inner Product metric), this is the wrong objective: PQ recall collapses near zero.

**USQ (Unit Sphere Quantizer)** applies a QR orthogonal rotation before quantizing, making compression metric-agnostic. Derived from RabitQ principles and extended to multi-bit (1/4/8-bit) with:
- QR decomposition rotation matrix (trained once, applied to all vectors)
- Unified 1/4/8-bit scalar quantization in the rotated space
- AVX512VNNI integer dot product scoring (`vpdpbusd`)
- Two-stage pipeline: 1-bit FastScan filter → B-bit rerank

![USQ Quantization](assets/benchmarks/usq_quantization.png)

> Dataset: Cohere Wikipedia-1M (768-dim, Inner Product), nprobe=32, x86 authority.

| Method | Compression | QPS | Recall@10 | Usable? |
|--------|-------------|-----|-----------|---------|
| IVF-PQ m=32 | 8× | 723 | **0.066** | ✗ |
| IVF-PQ m=48 | 5.3× | 502 | **0.127** | ✗ |
| IVF-Flat | 1× | 339 | 0.798 | ✓ |
| IVF-SQ8 | 4× | 605 | 0.805 | ✓ |
| **USQ 4-bit** | **8×** | **1,308** | **0.879** | ✓ |
| **USQ 8-bit** | **4×** | **1,011** | **0.968** | ✓ |

At the same 8× compression ratio where PQ gives recall=0.066, USQ gives **0.879** — a **13× improvement**. USQ 8-bit simultaneously delivers **3× faster QPS** than IVF-Flat with **+17% better recall** at ¼ the memory.

On 3072-dim embeddings (SimpleWiki-OpenAI-260K), USQ 8× still achieves recall **0.925** at 1,607 QPS.

---

## Ecosystem Integration

### Milvus

Hanns ships as a drop-in replacement for KnowWhere C++ inside Milvus standalone and distributed deployments. The FFI layer (`src/ffi/`) exposes the full KnowWhere index interface — same binary, same data format, same query semantics.

Measured end-to-end via [VectorDBBench](https://github.com/zilliztech/VectorDBBench) on a dedicated x86 server:

```
Milvus standalone → KnowWhere FFI → Hanns Rust index
                                   ↕ drop-in, no data migration
                    KnowWhere FFI → Native C++ (FAISS backend)
```

VectorDBBench results confirm the 2× QPS advantage across HNSW, IVF-Flat, IVF-SQ8, and IVF-PQ index types under realistic concurrent load patterns.

**QPS optimization lineage (HNSW Milvus c=80):**

| Round | Change | QPS |
|-------|--------|-----|
| R4 | FFI lazy bitset allocation | 349 |
| R7 | Private rayon ThreadPool (HNSW_NQ_POOL) | 540 |
| **R8** | Eliminate BinaryHeap clone + pre-alloc output buffer | **1,042** |

### HannsDB

HannsDB embeds Hanns as a single-machine agent database. VectorDBBench authority results (x86, 1536-dim, k=100):

| Metric | Result |
|--------|--------|
| Load | 148.0s |
| Optimize | 87.9s |
| p99 latency | **1.8ms** |
| Recall | **0.9756** |

Previously, cosine search p99 reached 110ms due to per-query allocations. After fixing TLS scratch buffer reuse: **p99 = 3.5ms (31× improvement)**.

### Lance

Hanns integrates as the HNSW search backend in the Lance vector lake format. Geometry-mean speedup across the ef=50–800 range: **1.64×** with equivalent recall.

Build is currently ~26% slower than Lance (sequential Hanns build vs. Lance rayon multi-thread). This is a known gap being addressed with parallel build improvements.

---

## Performance Charts

![QPS Comparison](assets/benchmarks/qps_comparison.png)

![Recall vs QPS](assets/benchmarks/recall_qps.png)

![DiskANN Comparison](assets/benchmarks/diskann_comparison.png)

> All benchmarks run on dedicated x86 server, `target-cpu=native`. Apple Silicon builds are for fast iteration only — not used as authority evidence.

---

## Index Coverage

| Index | Status | Recall@10 | Notes |
|-------|--------|-----------|-------|
| **HNSW** | ✅ Leading | 0.972 (SIFT-1M) | +11.9% vs FAISS 8T; 2.53× faster build in Milvus |
| **HNSW-SQ** | ✅ Ready | 0.992 | Integer precomputed ADC path |
| **IVF-Flat** | ✅ Leading | 0.978 (SIFT-1M) | 5.2× faster than FAISS 8T (batch parallel) |
| **IVF-SQ8** | ✅ Leading | 0.958 (SIFT-1M) | 1.42× faster than FAISS 8T; AVX2 fused decode |
| **IVF-USQ** | ✅ Ready | 0.905–0.968 (Cohere-1M) | AVX512VNNI; unified 1/4/8-bit |
| **IVF-PQ** | ✅ Ready | varies | m=32: 0.720 on synthetic data |
| **AISAQ (DiskANN Flash)** | ✅ Ready | 0.994 NoPQ (SIFT-1M) | On-demand pread + io_uring; 6.5× faster build than native |
| **ScaNN** | ✅ Ready | 0.969 | Exceeds 0.95 gate at reorder_k=1600 |
| **Sparse / WAND** | ✅ Ready | 1.0 | Sparse vector retrieval |
| **Binary** | ✅ Ready | — | Hamming distance |

---

## Quantization Subsystem

```
src/quantization/
  usq/           UsqQuantizer — QR rotation + unified 1/4/8-bit quantization
    rotator.rs   QR decomposition rotation matrix
    quantizer.rs training + SIMD scoring (AVX512VNNI)
    fastscan.rs  AVX512 fast scan (1-bit stage) + topk
    searcher.rs  two-stage coarse filter + rerank
  pq/            Product Quantizer — parallel k-means
  sq/            Scalar Quantizer — SQ8/SQ4
  pca/           PCA transform — nalgebra SVD (pure Rust, no BLAS)
```

---

## Build

```bash
cargo build --release          # LTO + codegen-units=1 + target-cpu=native
cargo test
cargo run --example benchmark --release
```

`.cargo/config.toml` enables `target-cpu=native` on x86_64 and aarch64 automatically.

---

## Repository Layout

```
src/
  faiss/           core index implementations
  quantization/    quantization subsystem (USQ, PQ, SQ, PCA)
  ffi/             FFI layer (KnowWhere-compatible ABI)
tests/             integration and regression tests
examples/          full benchmark harness
assets/benchmarks/ comparison charts
docs/              design docs, performance audits
benchmark_results/ authority verdict artifacts (JSON)
scripts/           chart generation, remote build/test
wiki/              operational runbooks and authority numbers
```

---

## Datasets

| Dataset | Dim | Metric | Size |
|---------|-----|--------|------|
| SIFT-1M | 128 | L2 | 1M vectors |
| Cohere Wikipedia-1M | 768 | IP | 1M vectors |
| Cohere-1M | 1024 | L2 | 1M vectors |
| SimpleWiki-OpenAI-260K | 3072 | IP | 260K vectors |

---

## Authority Hardware

Performance numbers are produced on a dedicated x86 server with `target-cpu=native` ([specs](wiki/machines/hannsdb-x86.md)). Apple Silicon builds are for fast iteration and pre-screening only.
