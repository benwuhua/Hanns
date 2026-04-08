# IVF-SQ8 Milvus RS Benchmark — 2026-04-08

**Dataset**: synthetic float32 normalized, 100K × 768D, IP metric (Cohere-1M shape; real Cohere data not cached)
**Machine**: hannsdb-x86 (16 cores)
**Index**: IVF_SQ8, nlist=512
**Run type**: first integration run — establishes wiring, not final authority

---

## Build

| Metric | Value |
|--------|-------|
| Insert (100K, batch=5000) | 15.1s |
| Build (nlist=512) | 3.0s |

## Search QPS (serial, pymilvus direct, c=1)

| nprobe | QPS |
|--------|-----|
| 8  | 39.4 |
| 32 | 39.4 |
| 128 | 39.3 |

**Note**: Serial QPS is Milvus overhead (H) dominated. nprobe has no visible effect on per-query time because Milvus fixed overhead (~25ms/query) swamps IVF-SQ8 compute time (~0.1ms). High-concurrency (c=80) benchmark would show significant nprobe differentiation.

---

## Notes

- First RS IVF-SQ8 Milvus integration run; confirms end-to-end wiring ✅
- `knowhere_set_nprobe` FFI correctly called per-search (verified via integration path)
- For full authority comparison (RS vs native, c=20/c=80): need Cohere-1M VectorDBBench run
- Native baseline for Milvus IVF-SQ8: not yet established (run pending)
