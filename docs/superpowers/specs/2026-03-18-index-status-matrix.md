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
| HNSW vs native | same recall | Rust 1.054x **slower** | Rust 1.054x slower | strict-ef | baseline_p3_001 |
| HNSW near-equal recall | 0.9518 | ~28K | 28,479 | ⚠️ unverified | hnsw_p3_002 (Codex only) |
| IVF-Flat | 0.545 (nprobe=8) | ~4K | — | needs tuning | cross_dataset_sampling |
| IVF-PQ | 0.47 | ~674 | — | ❌ no-go | recall_gated_baseline |
| IVF-SQ8 | unknown | — | — | untested | — |
| IVF-OPQ | unknown | — | — | untested | — |
| IVF-RaBitQ | 0.393 | ~144 | — | recheck (lossy) | recall_gated_baseline |
| DiskANN | 0.009 (4K dim=64) | ~607 | — | ❌ broken params | cross_dataset_sampling |
| AISAQ/PQFlash NoPQ | ~0.997 (10K) | ~30K (Mac) | ~9,722 (1M) | ⚠️ constrained | benchmark.rs |
| ScaNN | 1.0 | 37 | — | slow | recall_gated_baseline |
| Sparse/WAND | 1.0 | ~138 | — | ✅ ok | recall_gated_baseline |

---

## Per-Index Analysis

### HNSW (`src/faiss/hnsw.rs`, 9255 lines)

**Completeness**: ✅ Full (train/add/search/save/load/bitset filter)
**Recall**: 0.988@10 at ef=138, SIFT-128
**QPS**: 15,099 (x86) vs native 15,918 (both BF16, ef=138, M=16)

**The leadership claim problem:**
- **Strict-ef lane** (both at ef=138): Native 1.054x **faster** → no leadership
- **Near-equal-recall lane** (Rust at lower ef hits same recall): 28,479 vs 15,918 = 1.789x claimed
- The near-equal-recall lane was established by Codex independently (opt56 = TLS scratch reuse)
- Rust achieves 0.95 recall at lower ef than native → higher QPS at same accuracy point
- This IS methodologically valid IF ef values are correct, but was never independently verified

**What opt56 is**: commit `33e9853` — thread-local SearchScratch reuse (avoids per-query heap alloc). Legitimate optimization.

**Concern**: Round 12 experiment regressed to 1,497 QPS (5.68x worse than native). The code was rolled back to opt56 state, but the fluctuation raises questions about stability.

**Action**: HNSW-VERIFY-001 — re-run near-equal-recall benchmark on x86 to confirm 28K QPS numbers.

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
**Recall**: 0.393@10 — lossy, needs refine path
**Action**: IVF-RABITQ-001 — test with refine enabled.

---

### ScaNN (`src/faiss/scann.rs`, 1337 lines)

**Completeness**: ✅ Full (anisotropic VQ, train/add/search/save/load)
**Recall**: 1.0@10 (small dataset)
**QPS**: ~37 (tiny dataset, 5K vectors, 100 queries) — likely much higher at real scale

**Action**: SCANN-001 — authority benchmark at 100K/1M scale.

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
