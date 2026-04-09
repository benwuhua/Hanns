# DiskANN Native Milvus Benchmark — 2026-04-07

Date: 2026-04-07
Host: `hannsdb-x86`
Collection: `diskann_native_bench`
Native build: knowhere C++ `v2.6.10` via clean `FetchContent` build

## Summary

This run used a clean native knowhere build from:

- `internal/core/cmake_build_native_fresh/lib/libknowhere.so`
- `internal/core/cmake_build_native_fresh/lib/libmilvus-common.so`

Milvus standalone was restarted after replacing the runtime `libknowhere.so` and
`libmilvus-common.so` symlinks under `cmake_build/lib/`. Runtime verification via
`/proc/<pid>/maps` showed Milvus loading the native libraries from
`internal/core/cmake_build_native_fresh/lib/`, and no `libhanns.so` was present.

## Setup

- Dataset: synthetic normalized `float32`, `1,000,000 × 768`
- Schema: `id INT64 primary`, `vector FLOAT_VECTOR dim=768`
- Index params:
  - `index_type=DISKANN`
  - `metric_type=IP`
  - `max_degree=56`
  - `search_list_size=100`
  - `pq_code_budget_gb=0.0`
  - `build_dram_budget_gb=32.0`
  - `num_threads=16`

## Native Results

| Metric | Value |
| --- | ---: |
| Insert time | 151.411 s |
| Build time | 16.090 s |
| Serial QPS | 11.4402 |
| Concurrency 1 QPS | 11.1000 |
| Concurrency 20 QPS | 12.2167 |
| Concurrency 80 QPS | 12.9333 |
| Proxy recall@10 (sl=100 vs sl=500) | 1.0000 |

## RS vs Native

| Metric | RS Milvus | Native Milvus | Delta |
| --- | ---: | ---: | ---: |
| Build time | 16.1 s | 16.090 s | -0.010 s |
| Serial QPS | 2.4 | 11.4402 | +9.0402 |
| Concurrency 1 QPS | 2.5 | 11.1000 | +8.6000 |
| Concurrency 20 QPS | 12.7 | 12.2167 | -0.4833 |
| Concurrency 80 QPS | 13.1 | 12.9333 | -0.1667 |
| Proxy recall@10 | 1.0000 | 1.0000 | 0.0000 |

## Notes

- Native build time is effectively identical to RS on this workload.
- Native search throughput is much better in serial and `c=1`.
- At `c=20` and `c=80`, native and RS are in the same saturation band around `12-13` QPS.
- This indicates the RS shim was the dominant bottleneck for single-query execution here,
  but not for the high-concurrency steady-state path of this particular DiskANN Milvus setup.
