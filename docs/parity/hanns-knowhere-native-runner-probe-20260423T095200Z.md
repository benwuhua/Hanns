# Zilliz Knowhere native benchmark runner probe (2026-04-23)

Authority: `HannsDB-x86`
Official Knowhere: `https://github.com/zilliztech/knowhere` @ `4489f689de32baa70dae28ee1d6784ef87c2c3c8`
Build target: `benchmark_float_qps` exit `0`

## benchmark_float_qps tests

- `Benchmark_float_qps.TEST_IDMAP`
- `Benchmark_float_qps.TEST_IVF_FLAT`
- `Benchmark_float_qps.TEST_IVF_SQ8`
- `Benchmark_float_qps.TEST_IVF_PQ`
- `Benchmark_float_qps.TEST_HNSW`
- `Benchmark_float_qps.TEST_SCANN`
- `Benchmark_float_qps.TEST_DISKANN`
- `Benchmark_float_qps.TEST_AISAQ_P`
- `Benchmark_float_qps.TEST_AISAQ_S`

## Requested family qps-runner coverage

| Family | benchmark_float_qps coverage |
|---|---|
| HNSW | benchmark_float_qps.TEST_HNSW |
| HNSW-SQ | missing from benchmark_float_qps |
| HNSW-PQ | missing from benchmark_float_qps |
| DISKANN | benchmark_float_qps.TEST_DISKANN |
| DISKANN-USQ/RabitQ | benchmark_float_qps.TEST_AISAQ_P / TEST_AISAQ_S |
| IVF-PQ | benchmark_float_qps.TEST_IVF_PQ |
| IVF-SQ | benchmark_float_qps.TEST_IVF_SQ8 |
| IVF-USQ/RabitQ | missing from benchmark_float_qps |

## Limitations

- benchmark_float_qps does not list HNSW_SQ, HNSW_PQ, or IVF_RABITQ tests on this Zilliz main commit.
- Missing qps-runner coverage does not mean source capability is absent; source probe found HNSW_SQ/HNSW_PQ/IVF_RABITQ constants/registrations.
- Full first-pass performance matrix needs either additional official benchmark targets or a custom Zilliz runner for the missing qps families.
