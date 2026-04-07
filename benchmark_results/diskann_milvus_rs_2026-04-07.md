# DiskANN Milvus RS Benchmark — 2026-04-07

Date: 2026-04-07
Host: `hannsdb-x86`
Collection: `diskann_rs_bench`

## ⚠️ Key Finding: DiskANN FFI Not Yet Implemented

The numbers below **do NOT represent real DiskANN graph search**. Root cause in `src/ffi.rs`:

```rust
CIndexType::DiskAnn => {
    eprintln!("DiskANN not yet fully implemented via FFI");
    None  // returns null index → Milvus falls back to brute-force segment search
}
```

Evidence:
- Build=16.6s for 1M vectors: impossibly fast for Vamana graph construction
- Recall=1.0: exact search signature (brute-force)
- Disk I/O during search: zero (iostat confirmed)
- Latency p50=414.7ms matches brute-force 1M×768-dim IP compute budget

**Implication:** RS vs Native DiskANN comparison is not meaningful until this FFI stub is
wired to `diskann_aisaq.rs` (PQFlashIndex) or `diskann.rs`.

Reference: standalone benchmark outside Milvus shows RS DiskANN capability already exists:
- NoPQ SIFT-1M (in-memory): **6,062 QPS**, recall=0.9941
- PQ32 SIFT-1M (disk/mmap): **1,063 QPS**, recall=0.9114

---

## Setup

- Dataset: synthetic normalized `float32`, `1,000,000 x 768`
- Source note: no Cohere `.hdf5` was found on the host, and the VDBBench venv did not have `h5py`, so the setup used synthetic normalized vectors
- Schema: `id INT64 primary`, `vector FLOAT_VECTOR dim=768`
- Index params:
  - `index_type=DISKANN`
  - `metric_type=IP`
  - `max_degree=56`
  - `search_list_size=100`
  - `pq_code_budget_gb=0.0`
  - `build_dram_budget_gb=32.0`
  - `num_threads=16`

## Results

| Metric | Value |
| --- | ---: |
| Insert time | 156.882 s |
| Build time | 16.591 s |
| Serial QPS | 2.4082 |
| Concurrency 1 QPS | 2.4081 |
| Concurrency 20 QPS | 12.4890 |
| Concurrency 80 QPS | 12.0818 |
| Recall@10 proxy (`search_list=100` vs `500`) | 1.0000 |

## Notes (all numbers are brute-force, not DiskANN)

- Proxy recall=1.0 confirms brute-force: `search_list=100` and `search_list=500` return identical results
- Latency 414.7ms matches brute-force 1M×768 IP (memory bandwidth limited)
- c=20→c=80 QPS barely changes (12.5→12.1): saturated at memory bandwidth, not CPU

## Next Steps

1. Wire `CIndexType::DiskAnn` in `src/ffi.rs` to the RS DiskANN implementation
2. Re-run this benchmark to get real DiskANN numbers
3. Compare with native KnowWhere DiskANN (needs native Milvus binary on x86)
