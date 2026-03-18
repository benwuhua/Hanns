# KnowHere-RS Index Implementation Status Matrix

**Date**: 2026-03-18
**Scope**: All index types in `src/faiss/`, compared against native knowhere C++ capabilities
**Authority hardware**: x86 (remote), 8 cores, `/data/work/knowhere-rs-src`
**Native reference**: `/Users/ryan/Code/knowhere/src/index/`

---

## Quick Reference: Recall & QPS

| Index | Recall@10 | QPS (Mac) | QPS (x86) | Verdict | Source |
|-------|-----------|-----------|-----------|---------|--------|
| HNSW | 0.988 (ef=138) | ~15K | 15,099 | ⚠️ verify (see below) | baseline_p3_001 |
| HNSW strict-ef=138 | 0.9856 | — | 17,066 | ✅ Rust 1.073x faster than native 15,918 | hnsw_ef_sweep v2 |
| HNSW near-equal-recall | 0.9500 (ef=60) | — | 33,061 | ✅ **2.077x** vs native 15,918 at ef=138 | hnsw_ef_sweep v2 |
| IVF-Flat | 1.0 (nprobe=256/random) | 2,130 @nprobe=256 | — | ✅ correct, needs SIFT-128 test | ivf_flat_nprobe_sweep.rs |
| IVF-PQ | 0.47 | ~674 | — | ❌ no-go | recall_gated_baseline |
| IVF-SQ8 | 0.174 (plateau, full scan) | 115 @nprobe=256 | — | ❌ quantizer trained on raw vectors not residuals | ivf_sq8_sweep.rs |
| IVF-OPQ | 0.167 (plateau, full scan) | 65 @nprobe=256 | — | ❌ same PQ accuracy issue | ivf_opq_sweep.rs |
| IVF-RaBitQ | 0.260 no_refine / 0.552 with_refine | ~121 | — | ❌ no-go: both below 0.95 even full-scan | ivf_rabitq_sweep.rs |
| DiskANN | 0.009 was train+add bug; real: 0.91@dim=64 (L=32) | — | — | ⚠️ add() no graph edges, diagnosed | diskann_dim_diag.rs |
| AISAQ/PQFlash NoPQ | ~0.997 (10K) | ~30K (Mac) | ~9,722 (1M) | ⚠️ constrained | benchmark.rs |
| ScaNN | 0.699 max (100K, reorder_k=160) | 154 | — | ❌ no-go: below 0.95 gate at 100K | scann_sweep.rs |
| Sparse/WAND | 1.0 | ~138 | — | ✅ ok | recall_gated_baseline |

---

## Per-Index Analysis

### HNSW (`src/faiss/hnsw.rs`, 9255 lines)

**Completeness**: ✅ Full (train/add/search/save/load/bitset filter)
**Recall**: 0.988@10 at ef=138, SIFT-128
**QPS**: 15,099 (x86) vs native 15,918 (both BF16, ef=138, M=16)

**VERIFIED leadership (2026-03-18, BF16, SIFT-1M, 8-thread x86):**
- **Near-equal-recall lane**: Rust ef=60 (recall=0.9500, QPS=33,061) vs Native ef=138 (recall=0.9518, QPS=15,918) → **2.077x Rust faster**
- **Strict-ef lane** (both ef=138): Rust 17,066 QPS vs Native 15,918 QPS → Rust 1.073x faster at same ef
- V1 sweep was invalid (TOP_K=100 causing 10-recall-at-100 instead of recall@10, Float not BF16)
- V2 corrected sweep: BF16, TOP_K=10, proper recall@10, SIFT-1M ground truth

**opt56**: commit `33e9853` — thread-local SearchScratch reuse (avoids per-query heap alloc). Legitimate optimization.

---

### DiskANN (`src/faiss/diskann.rs`, 4825 lines)

**Completeness**: ✅ Full Vamana graph (train/add/search/save/load)
**Recall**: 0.009@10 at 4K vectors dim=64 → **catastrophically broken at dim=64**
**At 10K dim=128** (from benchmark.rs): ~98% recall — **dim=128 only tested**

**Root cause of dim=64 failure**: Likely the `dim` is wired into graph neighbors or PQ sub-space count in a way that breaks for dim ≠ 128. Not investigated.

**Verdict**: "constrained" — not native-comparable, not tested across dims.
**Action**: DISKANN-RECALL-001 — diagnose dim=64 recall collapse.

---

### AISAQ / PQFlashIndex (`src/faiss/diskann_aisaq.rs`, ~3300 lines)

**Completeness**: ✅ Fairly complete (train/add/search/save/load, PQ codec, materialize_storage, bitset filter, rearrange switch, filter_threshold)
**Recall**: ~99.7% (NoPQ 10K), ~98.2% (PQ32 10K) — synthetic random data
**QPS**: 9,722 (x86, 1M NoPQ), 7,673 (x86, 1M PQ32)

**Capability gaps vs native** (see `docs/superpowers/specs/2026-03-18-diskann-aisaq-gap-analysis.md`):
- `rearrange_candidates` uses ADC not exact float rerank → recall ceiling (AISAQ-CAP-001)
- No external ID mapping (AISAQ-CAP-002)
- No on-disk persistence (AISAQ-CAP-003)
- No RangeSearch (AISAQ-CAP-004)

---

### IVF-Flat (`src/faiss/ivf_flat.rs`, 615 lines)

**Completeness**: ✅ Full (train/add/search, no save/load)
**Recall**: 0.535@10 at nprobe=8 (too low) → unreliable baseline
**Expected**: 0.95+ at nprobe=32-64 for SIFT-128

**Root cause**: The baseline tested at very conservative nprobe=8. IVF-Flat is correct by construction (no quantization, exact within cluster). Just needs parameter sweep.

**Action**: IVF-FLAT-001 — run nprobe sweep, confirm 0.95+ recall gate, get authority QPS.

---

### IVF-PQ (`src/faiss/ivfpq.rs`, 1091 lines)

**Completeness**: ✅ Full code, ❌ broken recall
**Recall**: 0.47@10 — well below 0.8 gate
**Verdict**: no-go until fixed

**Hypothesis for failure**: Residual PQ computation bug or codebook assignment mismatch. Not investigated.
**Action**: IVF-PQ-FIX-001 — root cause analysis.

---

### IVF-SQ8 (`src/faiss/ivf_sq8.rs`, 796 lines)

**Completeness**: ✅ Full (train/add/search, partial save, no load)
**Recall**: **unknown — never authority benchmarked**
**Action**: IVF-SQ8-001 — get recall+QPS baseline.

---

### IVF-OPQ (`src/faiss/ivf_opq.rs`, 620 lines)

**Completeness**: ✅ Full (OPQ rotation training, train/add/search, partial save)
**Recall**: **unknown — never authority benchmarked**
**Action**: IVF-OPQ-001 — get recall+QPS baseline.

---

### IVF-RaBitQ (`src/faiss/ivf_rabitq.rs`, 909 lines)

**Completeness**: ✅ Full (external rabitq_ffi bindings)
**Recall**: 0.260@10 no_refine, 0.552@10 with_refine(dataview, k=40) — both below 0.95 gate
**Verdict**: ❌ no-go — 1-bit quantization too lossy even at nprobe=256 (full scan). Refine with k=40 insufficient.

**Sweep results (100K random, nlist=256, Mac)**:
- nprobe has NO effect on recall (0.260 constant for no_refine, 0.552 for with_refine across nprobe=4..256)
- The constant recall is explained by random data + RaBitQ 1-bit quantization: even full-scan, wrong neighbors selected by approx distances
- To improve: need refine_k >> TOP_K (e.g. 100x) or better quantization scheme

**Script**: `examples/ivf_rabitq_sweep.rs`
**Action**: IVF-RABITQ-FIX-001 (P2) — investigate larger refine_k or quantization improvements before revisiting.

---

### ScaNN (`src/faiss/scann.rs`, 1337 lines)

**Completeness**: ✅ Full (anisotropic VQ, train/add/search/save/load)
**Recall**: 1.0@10 (5K, meaningless) → **0.699@10 max** (100K, reorder_k=160) — below 0.95 gate
**QPS**: 154 at max recall (100K, Mac)
**Verdict**: ❌ no-go at 100K scale — anisotropic VQ recall ceiling under current params

**Sweep results (100K random, num_partitions=16, num_centroids=256, Mac)**:

| reorder_k | recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 0.226 | 218 |
| 20 | 0.331 | 211 |
| 40 | 0.443 | 199 |
| 80 | 0.570 | 181 |
| 160 | 0.699 | 154 |

Recall increases with reorder_k but well short of 0.95. Would need reorder_k >> 160 (with further QPS degradation) or larger num_centroids/num_partitions.

**Script**: `examples/scann_sweep.rs`
**Action**: SCANN-FIX-001 (P2) — investigate larger reorder_k, num_centroids, or algorithm fix for recall ceiling.

---

### HNSW Variants (PQ/PRQ/SQ)

**hnsw_pq.rs** (945 lines), **hnsw_prq.rs** (662 lines), **hnsw_quantized.rs** (407 lines)
**Status**: Train/add/search work. **No save/load** (persistence not implemented) → research-only.
**Action**: Either implement persistence or document as research-only.

---

### Binary Indexes (bin_flat, bin_ivf_flat, binary_hnsw)

**Status**: Complete implementations for Hamming distance / binary vectors.
**No authority benchmarks** — untested at scale.

---

## Comparison vs Native knowhere

| Feature | Native knowhere | knowhere-rs status |
|---------|----------------|-------------------|
| HNSW (FP32/BF16) | ✅ FAISS-based, SIMD | ✅ Custom, near-parity strict-ef |
| IVF-Flat | ✅ FAISS | ⚠️ recall untested properly |
| IVF-PQ | ✅ FAISS, 0.95+ recall | ❌ 0.47 recall |
| IVF-SQ8 | ✅ FAISS | unknown |
| DiskANN | ✅ + streaming SSD | ⚠️ dim=64 broken |
| ScaNN/RHNSW | ✅ | ⚠️ slow at small scale |
| Sparse (BM25, WAND) | ✅ | ✅ parity |
| Binary (Hamming) | ✅ | ✅ complete |
| RaBitQ | ✅ | ⚠️ 0.39 recall needs refine |
| External ID mapping | ✅ DataSet level | ❌ none |
| Persistence (all) | ✅ FileManager | ⚠️ partial per-index |
| Range search | ✅ all indexes | ⚠️ missing in most |

---

## Proposed Task Queue (Revised 2026-03-18)

### P0 — Validate claims / diagnose failures
| ID | Task |
|----|------|
| HNSW-VERIFY-001 | Re-run near-equal-recall benchmark on x86 — confirm/deny 28K QPS & 1.789x ratio |
| IVF-FLAT-001 | nprobe sweep → confirm recall gate, get authority QPS |
| DISKANN-RECALL-001 | Diagnose dim=64 recall collapse in diskann.rs |

### P1 — Fill unknown baselines
| ID | Task |
|----|------|
| IVF-SQ8-001 | Authority recall+QPS baseline on x86 |
| IVF-OPQ-001 | Authority recall+QPS baseline |
| SCANN-001 | Authority recall+QPS at 100K scale |
| IVF-RABITQ-001 | Test with refine path enabled |

### P2 — Fix known failures
| ID | Task |
|----|------|
| IVF-PQ-FIX-001 | Root cause 0.47 recall, fix residual PQ bug |
| HNSW-IMP-001 | If strict-ef stays behind native, identify real optimization path |
| AISAQ-CAP-001 | Exact float rerank (after P0/P1 settled) |

### P3 — Production completeness
| ID | Task |
|----|------|
| HNSW-PERSIST-001 | Implement save/load for HNSW-PQ/PRQ/SQ variants |
| AISAQ-CAP-002 | External ID mapping |
| AISAQ-CAP-003 | On-disk persistence |
| DISKANN-DIM-001 | Multi-dim support fix |
