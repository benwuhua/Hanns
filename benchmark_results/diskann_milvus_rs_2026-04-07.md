# DiskANN Milvus RS Benchmark

Date: 2026-04-07
Host: `hannsdb-x86`
Milvus target: RS Milvus on `127.0.0.1:19530`
Collection: `diskann_rs_bench`

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

## Notes

- Proxy recall used 50 random normalized queries with `k=10`.
- Proxy ground truth was Milvus DiskANN search with `search_list=500`.
- Search benchmark params were `metric_type=IP`, `search_list=100`, `limit=10`.
- The concurrency curve flattened between `c=20` and `c=80`, which suggests the current bottleneck is not solved by simply increasing client concurrency.
